#include "cpu/worker_prime.hpp"
#include "cpu/thread_utils.hpp"
#include "cpu/prime_validation.hpp"
#include "cpu/prime/prime_mining_engine.hpp"
#include "config/config.hpp"
#include "stats/stats_collector.hpp"
#include "prime/prime.hpp"
#include "prime/chain_sieve.hpp"
#include "prime/segment_allocator.hpp"
#include "prime/sieving_prime_table.hpp"
#include "block.hpp"
#include <asio.hpp>
#include <primesieve.hpp>
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <sstream> 
#include <boost/random.hpp>

namespace nexusminer
{
namespace cpu
{
namespace
{
// Re-export the canonical constants from prime_validation.hpp so the historical
// in-file aliases keep working for diagnostic log messages, but never drift
// from the single source of truth.  The previous local `kMaxSerializedPrimeOffsets
// = 10` literal was the root cause of length-8+ chains being silently dropped.
using nexusminer::prime::kPrimeOffsetFractionBytes;
using nexusminer::prime::kMinSerializedPrimeOffsets;
using nexusminer::prime::kMaxSerializedPrimeOffsets;
using nexusminer::prime::is_well_formed_prime_offsets;

constexpr std::size_t kBoostUint1kLimbBytes = sizeof(boost::multiprecision::limb_type);
}

void Worker_prime::prewarm_shared_state()
{
	// Touching the singleton once here forces the one-time
	// primesieve::generate_primes() pass to run on the caller's thread.
	// Subsequent calls (including the per-worker constructors) become O(1).
	(void)Sieving_prime_table::instance();
}

Worker_prime::Worker_prime(std::shared_ptr<asio::io_context> io_context, config::Worker_config& config)
	: m_io_context{ std::move(io_context) }
	, m_logger{ spdlog::get("logger") }
	, m_config{ config }
	, m_prime_helper{std::make_unique<Prime>()}
	, m_stop{ true }
	, m_initialized{ false }
	, m_log_leader{ "CPU Worker " + m_config.m_id + ": " }
	, m_primes{ 0 }
	, m_chains{ 0 }
	, m_difficulty{ 0 }
	, m_pool_nbits{ 0 }
{
	try {
		m_logger->debug("Worker_prime constructor: Initializing worker {}", m_config.m_id);

		// Stone 7: capture the prime backend selector once.  Two paths from
		// here on: "workers" runs today's per-worker mining loop, "engine"
		// turns this Worker_prime into a thin adapter (no mining thread,
		// no per-worker Sieve — Worker_manager constructs one
		// PrimeMiningEngine and registers all CPU prime workers with it).
		if (std::holds_alternative<config::Worker_config_cpu>(m_config.m_worker_mode)) {
			m_engine_mode = std::get<config::Worker_config_cpu>(m_config.m_worker_mode).m_engine_mode;
		}
		const bool engine_mode = (m_engine_mode == "engine");

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

		if (engine_mode)
		{
			// Stone 7 — adapter mode.  The PrimeMiningEngine (constructed by
			// Worker_manager after every worker exists) owns the Sieve and
			// the mining threads.  This worker exists only to carry stats
			// and identity (m_internal_id, m_config) and to provide the
			// representative internal_id_for_solution.  No Sieve, no
			// segment allocator, no fermat_performance_test, no run-thread
			// spawn.  The vestigial m_segment_allocator member is left null
			// (Stone 8 cleanup will remove the field entirely).
			m_initialized = true;
			m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "engine_mode = \"engine\" — adapter only (no per-worker mining thread)"));
			return;
		}

		// ── Legacy "workers" path (unchanged from pre-Stone-7) ──────────
		m_segmented_sieve = std::make_unique<Sieve>();

		// Initialize segmented sieve with error handling
		m_segmented_sieve->generate_sieving_primes();

		// Stone 3: build the segment allocator implied by [cpu] engine_mode.
		// Default ("workers") preserves today's per-worker `low += segment_size`
		// cursor; "engine" is reserved/forward-compatible (logged + falls back).
		m_segment_allocator = make_segment_allocator_for_engine_mode(
			m_engine_mode,
			static_cast<std::uint64_t>(m_segmented_sieve->get_segment_size()));

		// Run performance test
		fermat_performance_test();

		// Initialize data structures
		// Sized to 12 buckets to match cpu::Sieve::reset_stats (Option 5 /
		// post-#675).  Lengths 10-11 are now observable; bucket 0 / 1 stay
		// unused for legacy callers but the lockstep size keeps engine
		// fan-in trivial.
		m_chain_histogram = std::vector<std::uint32_t>(12, 0);
		m_segmented_sieve->reset_stats();

		// Mark as initialized
		m_initialized = true;
		m_logger->info("Worker_prime {}: Initialization complete", m_config.m_id);

		// Start persistent thread
		m_shutdown = false;
		m_run_thread = std::thread(&Worker_prime::run, this);
		m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "Persistent worker thread started"));

	} catch (const std::exception& e) {
		m_logger->error("Worker_prime constructor: Failed to initialize worker {}: {}",
		                m_config.m_id, e.what());
		m_initialized = false;
		throw;
	} catch (...) {
		m_logger->error("Worker_prime constructor: Unknown exception during initialization of worker {}",
		                m_config.m_id);
		m_initialized = false;
		throw;
	}
}

Worker_prime::~Worker_prime() noexcept
{
	try {
		m_logger->debug("Worker_prime destructor: Cleaning up worker {}", m_config.m_id);

		m_stop = true;  // Interrupt mining loops before competing for m_mtx during shutdown

		// Signal shutdown and wake up the worker thread
		{
			std::scoped_lock<std::mutex> lck(m_mtx);
			m_shutdown = true;
			m_running = false;
		}
		m_cv.notify_all();

		// Wait for main thread to complete with timeout protection.  Under
		// engine mode the run-thread was never spawned, so joinable() is
		// false and this branch is skipped — no work for the destructor to
		// do beyond clearing the engine binding.
		if (m_run_thread.joinable())
		{
			m_logger->debug("Worker_prime destructor: Waiting for worker {} thread to finish", m_config.m_id);
			m_run_thread.join();
		}

		// Join all worker threads
		for (auto& thread : m_worker_threads) {
			if (thread.joinable())
				thread.join();
		}

		m_logger->debug("Worker_prime destructor: Worker {} cleanup complete", m_config.m_id);

	} catch (const std::exception& e) {
		// Log but don't propagate exceptions from destructor
		if (m_logger) {
			m_logger->error("Worker_prime destructor: Exception during cleanup of worker {}: {}",
			                m_config.m_id, e.what());
		}
	} catch (...) {
		if (m_logger) {
			m_logger->error("Worker_prime destructor: Unknown exception during cleanup of worker {}",
			                m_config.m_id);
		}
	}
}

void Worker_prime::set_block(LLP::CBlock block, std::uint32_t nbits, Worker::Block_found_handler result)
{
	// Stone 7: under engine mode the worker is an adapter; the
	// PrimeMiningEngine subscribes to WorkerTemplateFeed directly.  Worker_manager
	// already skips the legacy per-worker fanout when uses_template_feed()==true,
	// but defend in depth in case a code path constructs us directly.
	if (m_engine_mode == "engine") {
		(void)block; (void)nbits; (void)result;
		return;
	}

	// Validate worker is properly initialized
	if (!m_initialized) {
		m_logger->error("Worker_prime::set_block: Worker {} not properly initialized, cannot set block",
		                m_config.m_id);
		return;
	}

	try {
		m_logger->debug("Worker_prime::set_block: Setting new block for worker {}", m_config.m_id);
		Block_data block_data{ block };
		const auto base_hash = block_data.GetPrimeBaseHash();
		set_block_impl(std::move(block_data), nbits, base_hash, std::move(result));

	} catch (const std::exception& e) {
		m_logger->error("Worker_prime::set_block: Exception for worker {}: {}", m_config.m_id, e.what());
	} catch (...) {
		m_logger->error("Worker_prime::set_block: Unknown exception for worker {}", m_config.m_id);
	}
}

void Worker_prime::set_block(std::shared_ptr<WorkPackage> work_package, Worker::Block_found_handler result)
{
	// Stone 7: see notes on the legacy set_block overload — engine mode
	// makes this a defensive no-op.
	if (m_engine_mode == "engine") {
		(void)work_package; (void)result;
		return;
	}

	// Validate worker is properly initialized
	if (!m_initialized) {
		m_logger->error("Worker_prime::set_block: Worker {} not properly initialized, cannot set block",
		                m_config.m_id);
		return;
	}

	try {
		m_logger->debug("Worker_prime::set_block: Setting new block for worker {} (optimized)", m_config.m_id);
		const auto& block = work_package->get_block();
		Block_data block_data{ block };
		const auto& precomputed_hash = work_package->get_prime_base_hash();
		if (precomputed_hash.has_value()) {
			m_logger->debug("Worker_prime::set_block: Using precomputed base hash from WorkPackage");
			set_block_impl(std::move(block_data), work_package->get_nbits(), precomputed_hash.value(), std::move(result));
		} else {
			m_logger->debug("Worker_prime::set_block: Computing base hash locally (no precomputed hash)");
			const auto base_hash = block_data.GetPrimeBaseHash();
			set_block_impl(std::move(block_data), work_package->get_nbits(), base_hash, std::move(result));
		}

	} catch (const std::exception& e) {
		m_logger->error("Worker_prime::set_block: Exception for worker {}: {}", m_config.m_id, e.what());
	} catch (...) {
		m_logger->error("Worker_prime::set_block: Unknown exception for worker {}", m_config.m_id);
	}
}

void Worker_prime::set_block_impl(Block_data block_data,
                                  std::uint32_t nbits,
                                  const uint1k& base_hash,
                                  Worker::Block_found_handler result)
{
	bool notify_worker = false;

	{
		std::scoped_lock<std::mutex> lck(m_mtx);
		m_found_nonce_callback = std::move(result);

		if (nbits != 0)	// take nBits provided from pool
		{
			m_pool_nbits = nbits;
		}

		const auto new_difficulty = m_pool_nbits != 0 ? m_pool_nbits : block_data.nBits;
		// same_active_search: the worker is already mining this exact proof-hash space,
		// so restarting would only throw away current sieve/segment progress.
		const bool same_active_search = !m_new_work && m_has_active_cycle && base_hash == m_active_base_hash;
		// same_queued_work: an identical template is already pending for the worker
		// thread, so avoid stacking another redundant restart request on top of it.
		const bool same_queued_work = m_new_work && base_hash == m_base_hash;

		// Preserve current search progress when the incoming template maps to the
		// exact same prime proof-hash space. Restarting would only throw away work.
		if (same_active_search || same_queued_work) {
			m_block = std::move(block_data);
			m_difficulty = new_difficulty;
			m_base_hash = base_hash;
			m_logger->debug("Worker_prime::set_block: Preserving current search for worker {} (duplicate prime proof-hash work)",
			                m_config.m_id);
			return;
		}

		m_block = std::move(block_data);
		m_difficulty = new_difficulty;
		m_base_hash = base_hash;

		// set the starting nonce for each worker to something different that won't overlap with the others
		m_starting_nonce = static_cast<uint64_t>(m_config.m_internal_id) << 48;
		m_nonce = m_starting_nonce;

		// NOTE: Sieve initialization (set_sieve_start, clear_chains,
		// calculate_starting_multiples) is intentionally NOT done here.
		// It runs on the worker thread in run() to eliminate the race condition
		// where set_block() could mutate the sieve while run() is using it.

		// Signal new work is available
		m_stop = true;
		m_new_work = true;
		m_running = true;
		m_template_consumed = false;  // Option D: fresh template, allow mining again
		notify_worker = true;
	}

	if (notify_worker) {
		m_cv.notify_one();
		m_logger->debug("Worker_prime::set_block: New work signaled for worker {}", m_config.m_id);
	}
}

void Worker_prime::run()
{
	// Get CPU configuration
	if (!std::holds_alternative<config::Worker_config_cpu>(m_config.m_worker_mode)) {
		m_logger->error(spdlog::fmt_lib::runtime(m_log_leader + "Invalid worker mode for CPU worker"));
		return;
	}

	auto const& cpu_cfg = std::get<config::Worker_config_cpu>(m_config.m_worker_mode);
	uint32_t num_threads = (cpu_cfg.m_threads > 1) ? cpu_cfg.m_threads : 1;

	// Prime mining currently supports single-threaded mode only
	// Multi-threading requires sieve partitioning which is more complex
	if (num_threads > 1) {
		m_logger->warn(spdlog::fmt_lib::runtime(m_log_leader + "Multi-threading requested ({} threads) but prime mining currently supports only single thread"), num_threads);
		m_logger->warn(spdlog::fmt_lib::runtime(m_log_leader + "Falling back to single-threaded mode"));
		num_threads = 1;
	}

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
			// Reset stop flag so the inner mining loop can run
			m_stop = false;
		}

		// Apply thread settings for mining
		if (cpu::set_thread_priority(cpu_cfg.m_priority_level)) {
			m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "Thread priority set to level {}"), cpu_cfg.m_priority_level);
		} else {
			m_logger->warn(spdlog::fmt_lib::runtime(m_log_leader + "Failed to set thread priority to level {}"), cpu_cfg.m_priority_level);
		}

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

		if (effective_affinity != 0) {
			if (cpu::set_thread_affinity(effective_affinity)) {
				m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "Thread affinity set to 0x{:016x}"), effective_affinity);
			} else {
				m_logger->warn(spdlog::fmt_lib::runtime(m_log_leader + "Failed to set thread affinity to 0x{:016x}"), effective_affinity);
			}
		}

		// Make local copies of block data to avoid race conditions
		// These copies are made once per work unit and remain stable during mining
		Block_data local_block;
		uint1k local_base_hash;
		uint64_t local_nonce;
		{
			std::scoped_lock<std::mutex> lck(m_mtx);
			local_block = m_block;
			local_base_hash = m_base_hash;
			local_nonce = m_nonce;
			m_active_base_hash = local_base_hash;
			m_has_active_cycle = true;
		}
		// Initialize the sieve on the worker thread only — set_block() no longer mutates
		// m_segmented_sieve, so all sieve operations are exclusively on this thread,
		// eliminating the race condition that caused the segfault at template transitions.
		uint1k local_sieve_start;
		{
			uint1k startprime = local_base_hash + local_nonce;
			// Stone 2: prepare() bundles set_sieve_start + clear_chains +
			// calculate_starting_multiples and returns the rounded start so we
			// can recover the exact nonce that the sieve will use.
			local_sieve_start = m_segmented_sieve->prepare(startprime);
			// Update local_nonce to reflect the actual sieve start position chosen by the sieve.
			// We do NOT write back to m_nonce here: set_block() always resets m_nonce to
			// m_starting_nonce before run() reads it, so the adjusted value is only needed
			// locally within this mining cycle.
			local_nonce = static_cast<uint64_t>(local_sieve_start - local_base_hash);
		}
		publish_statistics_snapshot();
		uint32_t segment_size = m_segmented_sieve->get_segment_size();
		// Stone 3: route the per-segment cursor through the allocator.  The
		// per-worker default keeps the existing seed (internal_id<<48 baseline,
		// then sequential).  reset() takes the worker-local start so the
		// returned segment offsets are *relative to local_sieve_start* — i.e.
		// the same domain that find_chains(low,...) and the candidate offset
		// arithmetic expect.
		m_segment_allocator->reset(0);
		uint64_t find_chains_ms = 0;
		uint64_t sieving_ms = 0;
		uint64_t test_chains_ms = 0;
	uint64_t elapsed_ms = 0;
	uint64_t high = 0;
	uint64_t low = 0;
	uint64_t range_searched_this_cycle = 0;

	auto start = std::chrono::steady_clock::now();
	auto interval_start = std::chrono::steady_clock::now();

	// Initialize CPU tracking (protected by mutex to prevent races with update_statistics)
	{
		std::scoped_lock<std::mutex> lck(m_mtx);
		m_cpu_tracking_start = std::chrono::steady_clock::now();
		m_cpu_active_time = std::chrono::milliseconds{0};
		m_cpu_total_time = std::chrono::milliseconds{0};
	}

	while (!m_stop)
	{
		// Check for new work at the top of the loop.
		// Option D: also exit when m_template_consumed — the worker has
		// already dispatched a found-block for this template and should
		// idle (via the outer m_cv.wait) until set_block delivers fresh
		// work, instead of re-sieving the spent base hash.
		{
			std::unique_lock<std::mutex> lck(m_mtx);
			if (m_new_work || m_template_consumed)
			{
				break;
			}
		}

		auto iteration_start = std::chrono::steady_clock::now();

		m_segmented_sieve->reset_sieve();
		if (m_stop) break;  // Check if new work arrived during reset_sieve()
		m_segmented_sieve->clear_chains();
		if (m_stop) break;  // Check if new work arrived during clear_chains()

		// Stone 3: ask the allocator for the next segment offset (relative to
		// local_sieve_start).  The default per-worker allocator simply hands
		// out monotonically increasing multiples of segment_size, exactly
		// matching the previous `low += segment_size` cursor.
		low = m_segment_allocator->next_segment_start();

		// current segment = [low, high]
		high = low + segment_size - 1;
		uint64_t sieve_size = (high - low) / 30 + 1;
		m_range_searched += segment_size;
		range_searched_this_cycle += segment_size;

		auto sieve_start = std::chrono::steady_clock::now();
		m_segmented_sieve->sieve_segment();
		auto sieve_stop = std::chrono::steady_clock::now();
		auto sieve_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(sieve_stop - sieve_start);
		sieving_ms += sieve_elapsed.count();
		auto find_chains_start = std::chrono::steady_clock::now();
		m_segmented_sieve->find_chains(low, false);
		auto find_chains_stop = std::chrono::steady_clock::now();
		auto find_chains_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(find_chains_stop - find_chains_start);
		find_chains_ms += find_chains_elapsed.count();
		auto test_chains_start = std::chrono::steady_clock::now();
		// Stone 2: pass the worker-owned sieve_start explicitly so the sieve no
		// longer has to remember it across calls.
		m_segmented_sieve->test_chains(local_sieve_start);
		auto test_chains_stop = std::chrono::steady_clock::now();
		auto test_chains_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(test_chains_stop - test_chains_start);
		test_chains_ms += test_chains_elapsed.count();
		const double required_difficulty = getNetworkDifficulty();
		//check difficulty of any chains that passed through the filter
		for (auto x : m_segmented_sieve->m_long_chain_starts)
		{
			local_block.nNonce = local_nonce + x;
			uint1k chain_start = local_base_hash + local_block.nNonce;

			// Enhanced validation using new prime_validation module
			uint1024_t hashPrime = boost_uint1024_t_to_uint1024_t(chain_start);
			std::vector<uint8_t> offsets;
			double actual_difficulty = 0.0;

			// Use the new comprehensive validation
			bool is_valid = nexusminer::prime::ValidatePrimeCandidate(
				hashPrime,
				required_difficulty,
				offsets,
				actual_difficulty
			);

			if (is_valid)
			{
				assert(is_well_formed_prime_offsets(offsets) &&
					"Expected vOffsets size in [kMin..kMax] (gap bytes + 4-byte LE fraction)");
				if (!is_well_formed_prime_offsets(offsets))
				{
					m_logger->error(spdlog::fmt_lib::runtime(m_log_leader + "Rejecting prime candidate with malformed serialized offsets ({} bytes, expected size in [{}..{}] = (chain_length - 1) gap bytes + {}-byte LE fraction)"),
						offsets.size(),
						kMinSerializedPrimeOffsets,
						kMaxSerializedPrimeOffsets,
						kPrimeOffsetFractionBytes);
					continue;
				}

				m_segmented_sieve->m_best_chain = std::max(actual_difficulty, m_segmented_sieve->m_best_chain);

				// Format offsets for logging
				std::ostringstream offsets_str;
				offsets_str << "[";
				for (size_t i = 0; i < offsets.size(); ++i) {
					if (i > 0) offsets_str << ", ";
					offsets_str << static_cast<int>(offsets[i]);
				}
				offsets_str << "]";

				m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "✓ FOUND VALID PRIME BLOCK! Difficulty: {:.6f} (required: {:.6f}), Chain length: {}, Offsets: {}"),
					actual_difficulty, required_difficulty, offsets.size(), offsets_str.str());

				//we found a valid chain.  submit it.
				{
					if (m_found_nonce_callback)
					{
						m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "💎 Block found! Posting to main io_context..."));
						// Capture block and offsets by value to avoid dangling references
						auto block_copy = local_block;
						auto captured_offsets = std::move(offsets);
						::asio::post(*m_io_context, [self = shared_from_this(), block_copy, captured_offsets = std::move(captured_offsets)]()
						{
							auto bd = std::make_unique<Block_data>(block_copy);
							bd->vOffsets = captured_offsets; // Prime chain offsets for submission
							self->m_found_nonce_callback(self->m_config.m_internal_id,
								std::move(bd));
						});
						// Option D: stop grinding this template — flag is
						// observed at the top of the outer mining loop, which
						// then drops the worker into m_cv.wait until the
						// next set_block.  Brief lock keeps the flag write
						// ordered with the loop's predicate read under m_mtx.
						{
							std::scoped_lock<std::mutex> lck(m_mtx);
							m_template_consumed = true;
						}
					}
					else
					{
						m_logger->debug(spdlog::fmt_lib::runtime(m_log_leader + "Miner callback function not set."));
					}
				}
			}
			else
			{
				m_logger->debug(spdlog::fmt_lib::runtime(m_log_leader + "Candidate validation failed (difficulty {:.6f} < {:.6f} or invalid prime), continuing..."),
					actual_difficulty, required_difficulty);
			}
		}
		publish_statistics_snapshot();
		// Stone 3: cursor advance is owned by the segment allocator; the
		// previous `low += segment_size` write is no longer needed here.

		// Track CPU active time for this iteration (protected by mutex to prevent races with update_statistics)
		auto iteration_end = std::chrono::steady_clock::now();
		auto iteration_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(iteration_end - iteration_start);
		{
			std::scoped_lock<std::mutex> lck(m_mtx);
			m_cpu_active_time += iteration_elapsed;

			// Update total time
			m_cpu_total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
				iteration_end - m_cpu_tracking_start);
		}
		
		//debug
		auto end = std::chrono::steady_clock::now();
		auto interval_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - interval_start);
		bool print_debug = false;
		if (print_debug && interval_elapsed.count() > 10000)
		{
			std::cout << "--debug--" << std::endl;
			auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
			elapsed_ms = elapsed.count();
			double chains_per_mm = 1.0e6 * m_segmented_sieve->m_chain_count / m_range_searched;
			double chains_per_sec = 1.0e3 * m_segmented_sieve->m_chain_count / elapsed_ms;
			double fermat_positive_rate = 1.0 * m_segmented_sieve->m_fermat_prime_count / m_segmented_sieve->m_fermat_test_count;
			double fermat_tests_per_chain = 1.0 * m_segmented_sieve->m_fermat_test_count / m_segmented_sieve->m_chain_count;
			std::cout << std::fixed << std::setprecision(2) << m_range_searched / 1.0e9 << " billion integers searched." <<
				" Found " << m_segmented_sieve->m_chain_count << " chain candidates. (" << chains_per_mm << " chains per million integers)" << std::endl;
			std::cout << "Fermat Tests: " << m_segmented_sieve->m_fermat_test_count << " Fermat Primes: " << m_segmented_sieve->m_fermat_prime_count <<
				" Fermat Positive Rate: " << std::fixed << std::setprecision(3) <<
				100.0 * fermat_positive_rate << "% Fermat tests per million integers sieved: " <<
				1.0e6 * m_segmented_sieve->m_fermat_test_count / m_range_searched << std::endl;

			std::cout << "Search rate: " << std::fixed << std::setprecision(1) << range_searched_this_cycle / (elapsed.count() * 1.0e3) << " million integers per second." << std::endl;
			double predicted_8chain_positivity_rate = std::pow(fermat_positive_rate, 8);
			//std::cout << "Predicted chains tested to find one Fermat 8-chains: " << 1 / predicted_8chain_positivity_rate << std::endl;
			//double predicted_days_between_8chains = 1.0 / (predicted_8chain_positivity_rate * chains_per_sec * 3600 * 24);
			//std::cout << "Predicted days between 8 chains per core: " << std::fixed << std::setprecision(2) << predicted_days_between_8chains << std::endl;
			std::cout << "Elapsed time: " << std::fixed << std::setprecision(2) << elapsed_ms / 1000.0 << "s. Sieving: " <<
				100.0 * sieving_ms / elapsed_ms << "% Chain filtering: " << 100.0 * find_chains_ms / elapsed_ms
				<< "% Fermat testing: " << 100.0 * test_chains_ms / elapsed_ms << "% Other: " <<
				100.0 * (elapsed_ms - (sieving_ms + find_chains_ms + test_chains_ms)) / elapsed_ms << "%" << std::endl;
			interval_start = std::chrono::steady_clock::now();
			std::cout << std::endl;
		}
	}

		{
			std::scoped_lock<std::mutex> lck(m_mtx);
			// Clear the active-cycle marker once the loop yields so a later duplicate
			// template is compared against the next live search, not stale prior state.
			m_has_active_cycle = false;
		}

		m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "Mining stopped, waiting for new work..."));
	}  // End of persistent thread loop
}

double Worker_prime::getDifficulty(const uint1k& p)
{
	std::vector<uint8_t> offsets_to_test;
	double difficulty = 0.0;
	// Pass a zero threshold when we only need the canonical LLL-TAO difficulty
	// calculation and serialized offsets, not submission gating.
	nexusminer::prime::ValidatePrimeCandidate(
		boost_uint1024_t_to_uint1024_t(p),
		0.0,
		offsets_to_test,
		difficulty);
	return difficulty;
}

double Worker_prime::getNetworkDifficulty()
{
	return m_difficulty / 10000000.0;
}

bool Worker_prime::difficulty_check(const uint1k& p)
{
	return getDifficulty(p) >= getNetworkDifficulty();
}



LLC::CBigNum Worker_prime::boost_uint1024_t_to_CBignum(const uint1k& p)
{
	std::stringstream ss;
	ss << std::hex << p;
	std::string p_hex_str = ss.str();
	LLC::CBigNum p_CBignum;
	p_CBignum.SetHex(p_hex_str);
	return p_CBignum;
}

// Helper function to convert boost::multiprecision::uint1024_t to LLC::uint1024_t
uint1024_t Worker_prime::boost_uint1024_t_to_uint1024_t(const uint1k& p)
{
	uint1024_t result{};
	const auto limb_bytes = p.backend().size() * kBoostUint1kLimbBytes;
	if (limb_bytes > sizeof(result))
	{
		throw std::runtime_error("Boost uint1024 limb storage exceeds LLC uint1024_t size");
	}

	// Both boost::uint1024_t and LLC::uint1024_t store little-endian limbs in the
	// active numeric payload. prime_validation_test verifies this limb copy matches
	// the legacy hex round-trip used previously.
	std::memcpy(&result, p.backend().limbs(), limb_bytes);
	return result;
}

void Worker_prime::update_statistics(stats::Collector& stats_collector)
{
	// Stone 7: under engine mode the worker's stats come from the engine's
	// snapshot_stats(), partitioned across registered workers via floor +
	// remainder.  This keeps per-worker GISPS approximately uniform and
	// the sum equal to the engine total — operators see N workers each at
	// ~1/N of the cooperative pool's throughput.
	if (m_engine_mode == "engine")
	{
		stats::Prime prime_stats;
		auto engine = m_prime_engine_weak.lock();
		if (engine)
		{
			const auto snap = engine->snapshot_stats();
			const std::uint32_t denom = std::max<std::uint32_t>(1, m_engine_share_count);
			const std::uint32_t idx   = m_engine_share_index;

			auto split = [&](std::uint64_t total) -> std::uint64_t {
				const std::uint64_t base = total / denom;
				const std::uint64_t rem  = total % denom;
				return base + (idx < rem ? 1u : 0u);
			};

			// Bug-sweep fix: m_primes and m_chains are SEMANTICALLY DIFFERENT
			// counters in the legacy path — m_primes counts Fermat-passing
			// numbers, m_chains counts long chains.  The engine snapshot
			// exposes neither separately; it only knows how many candidates
			// it dispatched (a chain is dispatched iff it passed Fermat
			// validation and met required difficulty).  Map candidates_
			// dispatched to m_chains only; leave m_primes at 0 to avoid
			// inflating the per-worker prime count by 2x, which would
			// confuse the stats display.  Operators wanting Fermat detail
			// in engine mode read the engine's own diagnostic counters.
			//
			// `share` is computed once and reused (was: split(...) twice).
			const std::uint64_t share_candidates = split(snap.candidates_dispatched);
			prime_stats.m_primes        = 0;
			prime_stats.m_chains        = stats::saturating_prime_stat(share_candidates);
			prime_stats.m_range_searched = split(snap.segments_processed * snap.segment_size);
			prime_stats.m_difficulty    = snap.nbits;
			// Stone 6.8 — chain histogram is intentionally NOT partitioned:
			// the engine aggregates buckets across every pool thread's Sieve
			// and every registered Worker_prime sees the same diagnostic
			// view.  Mirroring legacy "workers" mode behaviour where each
			// worker reports its own histogram, this gives operators a
			// consistent chain-row in the stats printer regardless of
			// engine_mode.
			prime_stats.m_chain_histogram = snap.chain_histogram;
			// Stone 6.8 — best chain difficulty is also a channel-wide truth
			// (cooperative pool sieving), so every worker publishes the same
			// value.  Operators see Best lift in lockstep when any pool
			// thread finds a winner.
			prime_stats.m_most_difficult_chain = snap.best_difficulty;
			// Engine does not partition Fermat-test counters either
			// (single-found-block-wins keeps per-thread credit meaningless).
			prime_stats.m_cpu_load = 0.0;
		}

		auto& typed = stats::as_typed<stats::Prime>(stats_collector);
		typed.update_worker_stats(m_config.m_internal_id, prime_stats);
		return;
	}

	auto snapshot = m_published_stats.load();
	// Issue 3A: typed downcast (mode is invariant by construction).
	auto& typed = stats::as_typed<stats::Prime>(stats_collector);
	typed.update_worker_stats(m_config.m_internal_id, *snapshot);

	// Reset the CPU-load accumulators here (the read side) so the next
	// publish_statistics_snapshot() begins a fresh interval. Resetting on
	// every publish would clobber the cumulative ratio (publishes happen
	// every mining iteration; reads happen every print_statistics_interval
	// seconds), pinning m_cpu_load near 1.0 in the latest snapshot.
	{
		std::scoped_lock<std::mutex> lck(m_mtx);
		m_cpu_active_time = {};
		m_cpu_total_time  = {};
		m_cpu_tracking_start = std::chrono::steady_clock::now();
	}

	// [Sieve Diag] log — emitted on stats thread, reads exclusively from the
	// immutable published snapshot so no live sieve state is touched here.
	const auto& diag = snapshot->m_sieve_diag;
	if (diag.m_sieve_calls > 0) {
		m_logger->debug("[Sieve Diag] calls={} hits={} hits/call={:.1f} start_mult={:.2f}ms primes={}",
			diag.m_sieve_calls, diag.m_inner_hits,
			static_cast<double>(diag.m_inner_hits) / diag.m_sieve_calls,
			diag.m_starting_multiples_us / 1000.0,
			diag.m_prime_count);
	}
}

void Worker_prime::bind_to_engine(std::weak_ptr<PrimeMiningEngine> engine,
                                  std::uint32_t share_index,
                                  std::uint32_t share_count)
{
	m_prime_engine_weak = std::move(engine);
	m_engine_share_index = share_index;
	m_engine_share_count = share_count;
	m_engine_bound.store(true, std::memory_order_release);
}

bool Worker_prime::uses_template_feed() const
{
	// Engine mode: the engine subscribes to the feed; this worker must NOT
	// receive the legacy per-worker set_block fanout (it has no run-thread
	// to pick the work up).
	return m_engine_mode == "engine";
}

void Worker_prime::attach_template_feed(std::shared_ptr<WorkerTemplateFeed> /*feed*/)
{
	// Stone 7: explicit no-op.  Under engine mode the PrimeMiningEngine
	// holds the feed pointer (its own consumer thread is the single
	// subscriber per channel); the worker exists only to carry stats and
	// identity, not to consume templates.  Under workers mode this
	// override still applies, but the legacy mining loop reads work from
	// the per-worker set_block() shim — not from the feed — so the no-op
	// is semantically correct in both modes.
}

bool Worker_prime::is_running() const
{
	if (m_engine_mode == "engine")
	{
		// Engine-mode "running" = bound to a live engine AND that engine
		// has at least one pool sieve thread alive.  Stone 6 added the
		// pool_threads_crashed counter for fault visibility; this hook
		// surfaces it through the per-worker stats display.  Without
		// this, if every pool thread crashed the channel would be
		// silently idle while the printer kept showing "N workers
		// running" (the bound flag is sticky).
		if (!m_engine_bound.load(std::memory_order_acquire))
		{
			return false;
		}
		auto engine = m_prime_engine_weak.lock();
		if (!engine)
		{
			// Engine destroyed (e.g. during teardown — engine reset
			// BEFORE workers per Worker_manager::stop_all_workers, or
			// any post-shutdown stats poll).  No live work.
			return false;
		}
		return engine->pool_threads_running() > 0;
	}
	return m_running.load();
}

void Worker_prime::publish_statistics_snapshot()
{
	// Stone 7 defensive guard: under engine_mode == "engine" the per-worker
	// Sieve is never constructed (the PrimeMiningEngine owns one Sieve per
	// pool thread).  publish_statistics_snapshot() is only ever called from
	// the legacy run() loop, which is itself never spawned in engine mode,
	// so reaching here with a null m_segmented_sieve indicates a future
	// regression — fail soft (no stats) rather than dereferencing null.
	if (!m_segmented_sieve)
	{
		return;
	}

	stats::Prime prime_stats;
	prime_stats.m_primes = stats::saturating_prime_stat(m_segmented_sieve->m_fermat_prime_count);
	prime_stats.m_chains = stats::saturating_prime_stat(m_segmented_sieve->m_chain_count);
	prime_stats.m_chain_histogram = stats::copy_prime_histogram(m_segmented_sieve->m_chain_histogram);
	prime_stats.m_range_searched = m_range_searched.load(std::memory_order_relaxed);
	prime_stats.m_most_difficult_chain = m_segmented_sieve->m_best_chain;

	// Snapshot the sieve diagnostic counters here (worker thread) so the stats
	// thread reads a coherent set from the immutable snapshot rather than
	// racing the live atomics.
	prime_stats.m_sieve_diag.m_sieve_calls =
		m_segmented_sieve->m_diag_sieve_calls.load(std::memory_order_relaxed);
	prime_stats.m_sieve_diag.m_inner_hits =
		m_segmented_sieve->m_diag_inner_hits.load(std::memory_order_relaxed);
	prime_stats.m_sieve_diag.m_starting_multiples_us =
		m_segmented_sieve->m_diag_starting_multiples_us.load(std::memory_order_relaxed);
	prime_stats.m_sieve_diag.m_prime_count =
		m_segmented_sieve->m_diag_prime_count.load(std::memory_order_relaxed);

	{
		std::scoped_lock<std::mutex> lck(m_mtx);
		prime_stats.m_difficulty = m_difficulty;

		// CPU-load is computed (but NOT reset) here. publish_statistics_snapshot()
		// runs at the end of every mining iteration, while update_statistics()
		// only runs on the stats-printer interval (~15s). If we reset on publish,
		// every snapshot would only capture the single iteration that just
		// completed — and since that iteration was actively running mining
		// work, m_cpu_load would always pin near 1.0.
		//
		// Both numerator (m_cpu_active_time) and denominator (m_cpu_total_time)
		// grow together between the resets owned by update_statistics(), so the
		// cumulative ratio published here is the correct "load over the current
		// stats interval".
		if (m_cpu_total_time.count() > 0) {
			prime_stats.m_cpu_load = static_cast<double>(m_cpu_active_time.count()) /
			                          static_cast<double>(m_cpu_total_time.count());
			prime_stats.m_cpu_load = std::max(0.0, std::min(1.0, prime_stats.m_cpu_load));
		} else {
			prime_stats.m_cpu_load = 0.0;
		}
	}

	m_published_stats.store(std::move(prime_stats));
}

void Worker_prime::fermat_performance_test()
//test the throughput of fermat primality test
{
	using namespace boost::multiprecision;
	using namespace boost::random;

	typedef independent_bits_engine<mt19937, 1024, boost::multiprecision::uint1024_t> generator1024_type;
	generator1024_type gen1024;
	gen1024.seed(time(0));
	// Generate some random 1024-bit unsigned values:
	std::vector<boost::multiprecision::uint1024_t> big_uints;
	int sample_size = 2000;
	for (unsigned i = 0; i < sample_size; ++i)
	{
		boost::multiprecision::uint1024_t pp = gen1024();
		//make it odd
		pp += 1?(pp % 2) == 0:0;
		big_uints.push_back(pp);
	}

	int p_count = 0;
	auto start = std::chrono::steady_clock::now();
	for (unsigned i = 0; i < sample_size; ++i)
	{
		p_count += 1 ? m_segmented_sieve->primality_test(big_uints[i]) : 0;
	}
	auto end = std::chrono::steady_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	double expected_primes = sample_size * 2 / (1024 * 0.693147);
	std::stringstream ss;
	ss << "Found " << p_count << " primes out of " << sample_size << " tested. Expected about " << expected_primes << ". ";
	ss << std::fixed << std::setprecision(2) << 1000.0* sample_size /elapsed.count()<< " primality tests/second. (" << 1.0*elapsed.count()/ sample_size << "ms)";
	m_logger->info(ss.str());
	
}

}
}
