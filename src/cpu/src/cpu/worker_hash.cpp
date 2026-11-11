#include "cpu/worker_hash.hpp"
#include "cpu/thread_utils.hpp"
#include "cpu/hash_validation.hpp"
#include "config/config.hpp"
#include "stats/stats_collector.hpp"
#include "block.hpp"
#include "hash/nexus_hash_utils.hpp"
#include "mining/mining_constants.hpp"
#include <asio.hpp>
#include <optional>
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace nexusminer
{
namespace cpu
{

Worker_hash::Worker_hash(std::shared_ptr<asio::io_context> io_context, Worker_config& config)
: m_io_context{std::move(io_context)}
, m_logger{spdlog::get("logger")}
, m_config{config}
, m_stop{true}
, m_log_leader{"CPU Worker " + m_config.m_id + ": " }
, m_hash_count{0}
, m_best_leading_zeros{0}
, m_met_difficulty_count {0}
, m_pool_nbits{0}
{
	m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "Initialized (Internal ID: {})"), m_config.m_internal_id);

	// Log CPU-specific configuration for multi-core support
	if (std::holds_alternative<config::Worker_config_cpu>(m_config.m_worker_mode)) {
		auto const& cpu_cfg = std::get<config::Worker_config_cpu>(m_config.m_worker_mode);
		if (cpu_cfg.m_threads > 1) {
			m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "Multi-core configuration: {} thread(s)"), cpu_cfg.m_threads);
			m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "Note: Multi-threading support is available"));
		}
		if (cpu_cfg.m_affinity_mask > 0) {
			m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "CPU affinity mask: 0x{:016x}"), cpu_cfg.m_affinity_mask);
		}
		if (cpu_cfg.m_priority_level != 2) {
			m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "Thread priority: {}"), cpu_cfg.m_priority_level);
		}
	}

	// Start persistent thread immediately
	m_shutdown = false;
	m_run_thread = std::thread(&Worker_hash::run, this);
	m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "Persistent worker thread started"));
}

Worker_hash::~Worker_hash()
{
	m_stop = true;  // Interrupt mining loops before competing for m_mtx during shutdown

	// Signal shutdown and wake up the worker thread
	{
		std::scoped_lock<std::mutex> lck(m_mtx);
		m_shutdown = true;
		m_running = false;
	}
	m_cv.notify_all();

	// Wait for main thread to finish
	if (m_run_thread.joinable())
		m_run_thread.join();

	// Join all worker threads
	for (auto& thread : m_worker_threads) {
		if (thread.joinable())
			thread.join();
	}
}

void Worker_hash::set_block(LLP::CBlock block, std::uint32_t nbits, Worker::Block_found_handler result)
{
	// Update work data atomically and signal worker thread
	{
		std::scoped_lock<std::mutex> lck(m_mtx);

		// Update callback
		m_found_nonce_callback = result;
		m_block = Block_data{ block };

		//set the starting nonce for each worker to something different that won't overlap with the others
		m_starting_nonce = static_cast<uint64_t>(m_config.m_internal_id) << 48;
		m_block.nNonce = m_starting_nonce;

		// Validate and set nBits with consistency checks
		if(nbits != 0)	// take nBits provided from pool
		{
			// Validate nbits consistency
			if (m_pool_nbits != 0 && m_pool_nbits != nbits)
			{
				m_logger->warn(spdlog::fmt_lib::runtime(m_log_leader + "m_pool_nbits changed from 0x{:08x} to 0x{:08x}"), m_pool_nbits, nbits);
			}
			m_pool_nbits = nbits;
			m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "Set m_pool_nbits to 0x{:08x} (from pool)"), m_pool_nbits);
		}
		else
		{
			// Use block's nBits when pool doesn't provide one
			if (m_pool_nbits != 0)
			{
				m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "Resetting m_pool_nbits (was 0x{:08x}, using block nBits 0x{:08x})"),
					m_pool_nbits, m_block.nBits);
			}
			m_pool_nbits = 0;
		}

		std::vector<unsigned char> headerB = m_block.GetHeaderBytes();

		// Validate header payload before processing
		if (headerB.empty())
		{
			m_logger->error(spdlog::fmt_lib::runtime(m_log_leader + "GetHeaderBytes() returned empty payload!"));
			throw std::runtime_error("Empty header payload");
		}

		m_logger->debug(spdlog::fmt_lib::runtime(m_log_leader + "Header payload size: {} bytes (expected: 216 for hash, 208 for prime)"),
			headerB.size());

		//calculate midstate
		m_skein.setMessage(headerB);

		// Log midstate calculation for debugging
		log_midstate_calculation();

		// Reset statistics for new block
		reset_statistics();

		// Signal new work is available.
		// m_stop = true interrupts the current while(!m_stop) loop iteration immediately,
		// matching the pattern used in the prime workers (PR #343).
		m_stop = true;
		m_new_work = true;
		m_running = true;
		m_template_consumed = false;  // Option D: fresh template, allow mining again
	}

	// Wake up the worker thread
	m_cv.notify_one();

	m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "New work set (Starting nonce: 0x{:016x}, nBits: 0x{:08x})"),
		m_starting_nonce, m_pool_nbits != 0 ? m_pool_nbits : m_block.nBits);
}

void Worker_hash::set_block(std::shared_ptr<WorkPackage> work_package, Worker::Block_found_handler result)
{
	// Update work data atomically and signal worker thread
	{
		std::scoped_lock<std::mutex> lck(m_mtx);
		m_found_nonce_callback = result;

		// Use precomputed data from WorkPackage
		const auto& block = work_package->get_block();
		m_block = Block_data{ block };

		//set the starting nonce for each worker to something different that won't overlap with the others
		m_starting_nonce = static_cast<uint64_t>(m_config.m_internal_id) << 48;
		m_block.nNonce = m_starting_nonce;

		// Validate and set nBits with consistency checks
		std::uint32_t nbits = work_package->get_nbits();
		if(nbits != 0)	// take nBits provided from pool
		{
			// Validate nbits consistency
			if (m_pool_nbits != 0 && m_pool_nbits != nbits)
			{
				m_logger->warn(spdlog::fmt_lib::runtime(m_log_leader + "m_pool_nbits changed from 0x{:08x} to 0x{:08x}"), m_pool_nbits, nbits);
			}
			m_pool_nbits = nbits;
			m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "Set m_pool_nbits to 0x{:08x} (from pool)"), m_pool_nbits);
		}
		else
		{
			// Use block's nBits when pool doesn't provide one
			if (m_pool_nbits != 0)
			{
				m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "Resetting m_pool_nbits (was 0x{:08x}, using block nBits 0x{:08x})"),
					m_pool_nbits, m_block.nBits);
			}
			m_pool_nbits = 0;
		}

		// Use precomputed header bytes from WorkPackage (optimization!)
		const std::vector<unsigned char>& headerB = work_package->get_header_bytes();

		// Validate header payload before processing
		if (headerB.empty())
		{
			m_logger->error(spdlog::fmt_lib::runtime(m_log_leader + "GetHeaderBytes() returned empty payload!"));
			throw std::runtime_error("Empty header payload");
		}

		m_logger->debug(spdlog::fmt_lib::runtime(m_log_leader + "Header payload size: {} bytes (expected: 216 for hash, 208 for prime)"),
			headerB.size());

		//calculate midstate using precomputed header bytes
		m_skein.setMessage(headerB);

		// Log midstate calculation for debugging
		log_midstate_calculation();

		// Reset statistics for new block
		reset_statistics();

		// Signal new work is available.
		// m_stop = true interrupts the current while(!m_stop) loop iteration immediately,
		// matching the pattern used in the prime workers (PR #343).
		m_stop = true;
		m_new_work = true;
		m_running = true;
		m_template_consumed = false;  // Option D: fresh template, allow mining again
	}

	// Wake up the worker thread
	m_cv.notify_one();

	m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "New work set (Starting nonce: 0x{:016x}, nBits: 0x{:08x})"),
		m_starting_nonce, m_pool_nbits != 0 ? m_pool_nbits : m_block.nBits);
}

void Worker_hash::run()
{
	// Get CPU configuration
	if (!std::holds_alternative<config::Worker_config_cpu>(m_config.m_worker_mode)) {
		m_logger->error(spdlog::fmt_lib::runtime(m_log_leader + "Invalid worker mode for CPU worker"));
		return;
	}

	auto const& cpu_cfg = std::get<config::Worker_config_cpu>(m_config.m_worker_mode);
	uint32_t num_threads = (cpu_cfg.m_threads > 0) ? cpu_cfg.m_threads : 1;

	m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "Persistent worker thread ready, waiting for work..."));

	// Persistent thread loop - runs until shutdown
	while (true) {
		// Wait for new work or shutdown signal
		{
			std::unique_lock<std::mutex> lock(m_mtx);
			m_cv.wait(lock, [this] { return m_new_work || m_shutdown; });

			// Check for shutdown
			if (m_shutdown) {
				m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "Worker thread shutting down"));
				break;
			}

			// Clear new work flag
			m_new_work = false;
			m_stop = false;
		}

		// Start mining with the new work
		m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "Starting {} mining thread(s)"), num_threads);

		// Clear any existing worker threads
		m_worker_threads.clear();

		// Apply hyperthreading and efficiency cores filtering
		uint64_t effective_affinity = cpu_cfg.m_affinity_mask;

		if (effective_affinity == 0) {
			// No specific affinity set, potentially filter based on HT/E-core settings
			if (!cpu_cfg.m_enable_hyperthreading || !cpu_cfg.m_enable_efficiency_cores) {
				// Get total logical processors
				uint32_t total_cores = std::thread::hardware_concurrency();
				uint32_t physical_cores = cpu::get_physical_core_count();
				bool smt_enabled = cpu::is_smt_enabled();

				m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "Core detection: {} logical cores, {} physical cores, SMT {}"),
				              total_cores, physical_cores, smt_enabled ? "enabled" : "disabled");

				// Build affinity mask based on settings
				if (!cpu_cfg.m_enable_hyperthreading && smt_enabled) {
					// Use only physical cores (first half typically)
					for (uint32_t i = 0; i < physical_cores; i++) {
						effective_affinity |= (1ULL << i);
					}
					m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "Hyperthreading disabled, using physical cores only: 0x{:016x}"),
					              effective_affinity);
				}

				if (!cpu_cfg.m_enable_efficiency_cores) {
					// Try to get P-cores only
					auto p_cores = cpu::get_performance_cores();
					if (!p_cores.empty()) {
						effective_affinity = 0;
						for (auto core : p_cores) {
							effective_affinity |= (1ULL << core);
						}
						m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "E-cores disabled, using P-cores only: 0x{:016x}"),
						              effective_affinity);
					} else {
						m_logger->warn(spdlog::fmt_lib::runtime(m_log_leader + "Could not detect P-cores, using all cores"));
					}
				}
			}
		}

		// Spawn mining threads
		if (num_threads > 1) {
			m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "Multi-threading enabled with {} threads"), num_threads);
			m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "Nonce space will be partitioned across threads"));

			for (uint32_t i = 0; i < num_threads; i++) {
				m_worker_threads.emplace_back([this, i, num_threads, &cpu_cfg, effective_affinity]() {
					// Set thread-specific affinity if needed
					uint64_t thread_affinity = 0;

					if (effective_affinity != 0) {
						// Distribute threads across available cores
						std::vector<uint32_t> available_cores;
						for (uint32_t c = 0; c < 64; c++) {
							if (effective_affinity & (1ULL << c)) {
								available_cores.push_back(c);
							}
						}

						if (!available_cores.empty()) {
							// Assign core to this thread (round-robin)
							uint32_t core_idx = i % available_cores.size();
							thread_affinity = 1ULL << available_cores[core_idx];

							if (cpu::set_thread_affinity(thread_affinity)) {
								m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "Thread {} pinned to core {}"),
								              i, available_cores[core_idx]);
							}
						}
					}

					// Set thread priority
					if (cpu::set_thread_priority(cpu_cfg.m_priority_level)) {
						m_logger->debug(spdlog::fmt_lib::runtime(m_log_leader + "Thread {} priority set to level {}"),
						               i, cpu_cfg.m_priority_level);
					} else {
						m_logger->warn(spdlog::fmt_lib::runtime(m_log_leader + "Thread {} failed to set priority"), i);
					}

					// Run mining loop for this thread
					mine_loop(i, num_threads);
				});
			}

			// Wait for all threads to complete
			for (auto& thread : m_worker_threads) {
				if (thread.joinable()) {
					thread.join();
				}
			}

		} else {
			// Single thread mode
			// Apply thread settings
			if (cpu::set_thread_priority(cpu_cfg.m_priority_level)) {
				m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "Thread priority set to level {}"), cpu_cfg.m_priority_level);
			} else {
				m_logger->warn(spdlog::fmt_lib::runtime(m_log_leader + "Failed to set thread priority to level {}"), cpu_cfg.m_priority_level);
			}

			if (effective_affinity != 0) {
				if (cpu::set_thread_affinity(effective_affinity)) {
					m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "Thread affinity set to 0x{:016x}"), effective_affinity);
				} else {
					m_logger->warn(spdlog::fmt_lib::runtime(m_log_leader + "Failed to set thread affinity to 0x{:016x}"), effective_affinity);
				}
			}

			// Run single-threaded mining loop
			mine_loop(0, 1);
		}

		m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "All mining threads stopped, waiting for new work..."));
	}
}

void Worker_hash::mine_loop(uint32_t thread_id, uint32_t total_threads)
{
	m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "Mining thread {} of {} started"), thread_id, total_threads);
	uint64_t last_log_hash_count = 0;
	constexpr uint64_t log_interval = mining::HASH_LOG_INTERVAL;
	constexpr int max_retries = 3;
	uint64_t payload_validation_failures = 0;
	uint64_t hash_mismatches = 0;
	uint64_t thread_hash_count = 0;
	
	while (!m_stop)
	{
		// Copy skein state and advance shared nonce under brief lock.
		// Advancing nonce here ensures each thread/iteration gets a unique nonce
		// without holding the lock during the expensive hash computation.
		// NexusSkein has a deleted copy assignment (const member), so we use
		// std::optional to copy-construct inside the lock, then reference it outside.
		std::optional<NexusSkein> local_skein_opt;
		{
			std::unique_lock<std::mutex> lck(m_mtx);
			if (m_new_work || m_template_consumed)
				break;
			local_skein_opt.emplace(m_skein);  // copy-construct: nonce + midstate
			m_skein.setNonce(m_skein.getNonce() + total_threads);  // advance shared nonce
		}
		NexusSkein& local_skein = *local_skein_opt;

		uint64_t nonce;
		bool hash_calculated = false;
		int retry_count = 0;
		
		// Retry mechanism for hash calculation
		while (!hash_calculated && retry_count < max_retries && !m_stop)
		{
			try
			{
				// Calculate hash with NO lock held — operates on thread-local copy
				local_skein.calculateHash();
				
				// Validate Skein output before passing to Keccak
				NexusSkein::stateType skeinHash = local_skein.getHash();
				if (!validate_skein_output(skeinHash))
				{
					++payload_validation_failures;
					m_logger->warn(spdlog::fmt_lib::runtime(m_log_leader + "Thread {} Skein payload validation failed for nonce 0x{:016x}"), 
					              thread_id, local_skein.getNonce());
					throw std::runtime_error("Invalid Skein output payload");
				}
				
				// Log Skein output for debugging (periodically)
				if (thread_hash_count % (log_interval * 10) == 0)
				{
					log_skein_state(skeinHash, local_skein.getNonce());
				}
				
				// Run keccak on the result from skein
				NexusKeccak keccak(skeinHash);
				keccak.calculateHash();
				uint64_t keccakHash = keccak.getResult();
				
				// Validate Keccak output
				if (!validate_keccak_output(keccakHash))
				{
					++payload_validation_failures;
					m_logger->warn(spdlog::fmt_lib::runtime(m_log_leader + "Thread {} Keccak payload validation failed for nonce 0x{:016x}"), 
					              thread_id, local_skein.getNonce());
					throw std::runtime_error("Invalid Keccak output payload");
				}
				
				// Cross-validate periodically (every 100000 hashes) to minimize performance impact
				// Also validate when we find a candidate nonce
				bool should_cross_validate = (thread_hash_count % 100000 == 0) || ((keccakHash & leading_zero_mask()) == 0);
				
				if (should_cross_validate && !cross_validate_hashes(skeinHash, keccakHash))
				{
					++hash_mismatches;
					m_logger->error(spdlog::fmt_lib::runtime(m_log_leader + "Thread {} Hash cross-validation failed for nonce 0x{:016x} - skipping nonce"), 
					               thread_id, local_skein.getNonce());
					// Log detailed mismatch info for debugging
					log_hash_mismatch(skeinHash, keccakHash, local_skein.getNonce());
					
					// Skip this nonce due to validation failure and move to next
					throw std::runtime_error("Hash cross-validation failed");
				}
				
				nonce = local_skein.getNonce();
				
				// Check the result for leading zeros
				if ((keccakHash & leading_zero_mask()) == 0)
				{
					m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "Thread {} found a nonce candidate {}"), thread_id, nonce);
					// Verify the difficulty — requires shared state; take lock briefly
					std::unique_lock<std::mutex> lck(m_mtx);
					m_skein.setNonce(nonce);  // restore candidate nonce for difficulty_check()
					if (difficulty_check())
					{
						++m_met_difficulty_count;
						// Update the block with the nonce and call the callback function
						m_block.nNonce = nonce;
						if (m_found_nonce_callback)
						{
							m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "💎 Block found! Posting to main io_context..."));
							::asio::post(*m_io_context, [self = shared_from_this()]()
							{
								self->m_found_nonce_callback(self->m_config.m_internal_id, 
									std::make_unique<Block_data>(self->m_block));
							});
							// Option D: stop grinding this template — the next
							// loop iteration's check at the top of mine_loop
							// will observe m_template_consumed and break out,
							// dropping the worker into m_cv.wait until the
							// next set_block.  We still hold m_mtx here so
							// the flag is published under the same lock that
							// the loop's predicate check takes.
							m_template_consumed = true;
						}
						else
						{
							m_logger->debug(spdlog::fmt_lib::runtime(m_log_leader + "Miner callback function not set."));
						}
					}
				}
				
				++thread_hash_count;
				{
					std::unique_lock<std::mutex> lck(m_mtx);
					++m_hash_count;
				}
				hash_calculated = true;
				
				// Log progress periodically with enhanced diagnostics
				if (thread_hash_count - last_log_hash_count >= log_interval)
				{
					m_logger->debug(spdlog::fmt_lib::runtime(m_log_leader + "Thread {} hashing progress: {} hashes computed, current nonce: 0x{:016x}"), 
					               thread_id, thread_hash_count, nonce);
					if (payload_validation_failures > 0 || hash_mismatches > 0)
					{
						m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "Thread {} diagnostics: {} payload validation failures, {} hash mismatches"), 
						              thread_id, payload_validation_failures, hash_mismatches);
					}
					last_log_hash_count = thread_hash_count;
				}
			}
			catch (const std::exception& e)
			{
				++retry_count;
				if (retry_count < max_retries)
				{
					m_logger->warn(spdlog::fmt_lib::runtime(m_log_leader + "Thread {} hash calculation failed (attempt {}/{}): {}. Retrying..."), 
					              thread_id, retry_count, max_retries, e.what());
					std::this_thread::sleep_for(std::chrono::milliseconds(10));
				}
				else
				{
					m_logger->error(spdlog::fmt_lib::runtime(m_log_leader + "Thread {} hash calculation failed after {} retries: {}. Skipping nonce."), 
					               thread_id, max_retries, e.what());
					// Nonce was already advanced at the top of the outer loop; just continue.
				}
			}
		}
	}
	m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "Thread {} stopped. Hashes: {}, Payload failures: {}, Hash mismatches: {}"), 
	              thread_id, thread_hash_count, payload_validation_failures, hash_mismatches);
}

void Worker_hash::update_statistics(stats::Collector& stats_collector)
{
	std::scoped_lock<std::mutex> lck(m_mtx);

	// Issue 3A: typed downcast — the worker knows it's a HASH-mode worker
	// at compile time, and Worker_manager pairs HASH workers with a
	// Hash_collector, so the cast is invariant by construction.
	auto& typed = stats::as_typed<stats::Hash>(stats_collector);
	auto hash_stats = typed.get_worker_stats(m_config.m_internal_id);
	hash_stats.m_hash_count = m_hash_count;
	hash_stats.m_best_leading_zeros = m_best_leading_zeros;
	hash_stats.m_met_difficulty_count = m_met_difficulty_count;

	typed.update_worker_stats(m_config.m_internal_id, hash_stats);

}


bool Worker_hash::difficulty_check()
{
	//perform additional difficulty filtering prior to submitting the nonce 
	
	// Validate m_pool_nbits consistency
	uint32_t nbits_to_use = m_pool_nbits != 0 ? m_pool_nbits : m_block.nBits;
	
	if (m_pool_nbits != 0 && m_block.nBits != 0 && m_pool_nbits != m_block.nBits)
	{
		m_logger->debug(spdlog::fmt_lib::runtime(m_log_leader + "Using pool nBits 0x{:08x} (block nBits: 0x{:08x})"), 
			m_pool_nbits, m_block.nBits);
	}

	//leading zeros in bits required of the hash for it to pass the current difficulty.
	int leadingZerosRequired;
	uint64_t difficultyTest64;
	decodeBits(nbits_to_use, leadingZerosRequired, difficultyTest64);
	
	// Recalculate and validate hash outputs
	m_skein.calculateHash();
	NexusSkein::stateType skeinHash = m_skein.getHash();
	
	// Validate Skein output in difficulty check
	if (!validate_skein_output(skeinHash))
	{
		m_logger->error(spdlog::fmt_lib::runtime(m_log_leader + "Skein validation failed in difficulty_check"));
		return false;
	}
	
	//run keccak on the result from skein
	NexusKeccak keccak(skeinHash);
	keccak.calculateHash();
	uint64_t keccakHash = keccak.getResult();
	
	// Validate Keccak output in difficulty check
	if (!validate_keccak_output(keccakHash))
	{
		m_logger->error(spdlog::fmt_lib::runtime(m_log_leader + "Keccak validation failed in difficulty_check"));
		return false;
	}
	
	int hashActualLeadingZeros = 63 - findMSB(keccakHash);
	m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "Difficulty check: Leading Zeros Found/Required {}/{}, nBits: 0x{:08x}"), 
		hashActualLeadingZeros, leadingZerosRequired, nbits_to_use);
	
	if (hashActualLeadingZeros > m_best_leading_zeros)
	{
		m_best_leading_zeros = hashActualLeadingZeros;
		m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "New best leading zeros: {}"), m_best_leading_zeros);
	}
	
	//check the hash result is less than the difficulty.  We truncate to just use the upper 64 bits for easier calculation.
	if (keccakHash <= difficultyTest64)
	{
		m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "Nonce passes difficulty check (hash: 0x{:016x} <= difficulty: 0x{:016x})"), 
			keccakHash, difficultyTest64);
		
		// Log detailed payload information for successful nonce
		log_skein_state(skeinHash, m_skein.getNonce());
		
		return true;
	}
	else
	{
		m_logger->debug(spdlog::fmt_lib::runtime(m_log_leader + "Nonce fails difficulty check (hash: 0x{:016x} > difficulty: 0x{:016x})"), 
			keccakHash, difficultyTest64);
		return false;
	}
}

uint64_t Worker_hash::leading_zero_mask()
{
	return ((1ull << leading_zeros_required) - 1) << (64 - leading_zeros_required);
}

void Worker_hash::reset_statistics()
{
	m_hash_count = 0;
	m_best_leading_zeros = 0;
	m_met_difficulty_count = 0;
}

bool Worker_hash::validate_skein_output(const NexusSkein::stateType& skeinHash) const
{
	// Validate that Skein output is not all zeros (invalid state)
	bool all_zeros = true;
	bool all_same = true;
	uint64_t first_val = skeinHash[0];
	
	for (size_t i = 0; i < skeinHash.size(); ++i)
	{
		if (skeinHash[i] != 0)
		{
			all_zeros = false;
		}
		if (i > 0 && skeinHash[i] != first_val)
		{
			all_same = false;
		}
	}
	
	if (all_zeros)
	{
		m_logger->error(spdlog::fmt_lib::runtime(m_log_leader + "Skein output is all zeros (invalid state)"));
		return false;
	}
	
	// Additional validation: check for obviously invalid patterns
	// (all same value, which would indicate a calculation error)
	if (all_same && first_val != 0)
	{
		m_logger->warn(spdlog::fmt_lib::runtime(m_log_leader + "Skein output has suspicious pattern (all values = 0x{:016x})"), first_val);
		// Don't reject, but log for debugging
	}
	
	return true;
}

bool Worker_hash::validate_keccak_output(uint64_t keccakHash) const
{
	// Keccak output validation
	// All 64-bit values are theoretically valid for Keccak hash results
	// We primarily rely on cross-validation for comprehensive checking
	return true;
}

bool Worker_hash::cross_validate_hashes(const NexusSkein::stateType& skeinHash, uint64_t keccakHash) const
{
	// Cross-validation: Verify that the Keccak hash was derived from the Skein hash
	// Note: This performs an additional Keccak calculation which has a performance cost
	// This is only done periodically based on hash_count to minimize overhead
	
	// Skip cross-validation for most hashes to maintain performance
	// Only validate every 100000th hash or when we find a candidate
	// The caller should control when to invoke this expensive check
	
	// Recalculate Keccak to verify
	NexusKeccak keccak_verify(skeinHash);
	keccak_verify.calculateHash();
	uint64_t keccak_verify_result = keccak_verify.getResult();
	
	if (keccak_verify_result != keccakHash)
	{
		m_logger->error(spdlog::fmt_lib::runtime(m_log_leader + "Cross-validation failed: Keccak hash mismatch (expected: 0x{:016x}, got: 0x{:016x})"),
			keccak_verify_result, keccakHash);
		return false;
	}
	
	return true;
}

void Worker_hash::log_skein_state(const NexusSkein::stateType& skeinHash, uint64_t nonce) const
{
	static constexpr size_t SKEIN_LOG_WORDS = 4;  // Number of words to log from Skein output
	
	m_logger->debug(spdlog::fmt_lib::runtime(m_log_leader + "Skein output for nonce 0x{:016x}:"), nonce);
	std::stringstream ss;
	ss << std::hex << std::setfill('0');
	for (size_t i = 0; i < std::min(SKEIN_LOG_WORDS, skeinHash.size()); ++i)
	{
		ss << "0x" << std::setw(16) << skeinHash[i] << " ";
	}
	m_logger->debug(spdlog::fmt_lib::runtime(m_log_leader + "  First {} words: {}"), SKEIN_LOG_WORDS, ss.str());
}

void Worker_hash::log_hash_mismatch(const NexusSkein::stateType& skeinHash, uint64_t keccakHash, uint64_t nonce) const
{
	static constexpr size_t SKEIN_LOG_WORDS = 4;  // Number of words to log from Skein output
	
	m_logger->error(spdlog::fmt_lib::runtime(m_log_leader + "Hash mismatch detected for nonce 0x{:016x}"), nonce);
	
	// Log Skein output
	std::stringstream ss_skein;
	ss_skein << std::hex << std::setfill('0');
	for (size_t i = 0; i < std::min(SKEIN_LOG_WORDS, skeinHash.size()); ++i)
	{
		ss_skein << "0x" << std::setw(16) << skeinHash[i] << " ";
	}
	m_logger->error(spdlog::fmt_lib::runtime(m_log_leader + "  Skein output (first {} words): {}"), SKEIN_LOG_WORDS, ss_skein.str());
	
	// Log Keccak result
	m_logger->error(spdlog::fmt_lib::runtime(m_log_leader + "  Keccak result: 0x{:016x}"), keccakHash);
	
	// Log current m_pool_nbits for context
	m_logger->error(spdlog::fmt_lib::runtime(m_log_leader + "  Current m_pool_nbits: 0x{:08x}"), m_pool_nbits);
}

void Worker_hash::log_midstate_calculation()
{
	static constexpr size_t SKEIN_LOG_WORDS = 4;  // Number of words to log from midstate
	
	// Log midstate information for debugging
	auto key2 = m_skein.getKey2();
	auto msg2 = m_skein.getMessage2();
	
	m_logger->debug(spdlog::fmt_lib::runtime(m_log_leader + "Midstate calculated:"));
	
	// Log first few words of key2
	std::stringstream ss_key;
	ss_key << std::hex << std::setfill('0');
	for (size_t i = 0; i < std::min(SKEIN_LOG_WORDS, key2.size()); ++i)
	{
		ss_key << "0x" << std::setw(16) << key2[i] << " ";
	}
	m_logger->debug(spdlog::fmt_lib::runtime(m_log_leader + "  Key2 (first {} words): {}"), SKEIN_LOG_WORDS, ss_key.str());
	
	// Log first few words of message2
	std::stringstream ss_msg;
	ss_msg << std::hex << std::setfill('0');
	for (size_t i = 0; i < std::min(SKEIN_LOG_WORDS, msg2.size()); ++i)
	{
		ss_msg << "0x" << std::setw(16) << msg2[i] << " ";
	}
	m_logger->debug(spdlog::fmt_lib::runtime(m_log_leader + "  Message2 (first {} words): {}"), SKEIN_LOG_WORDS, ss_msg.str());
}

}
}
