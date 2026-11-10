#include "gpu/worker_prime.hpp"
#include "cpu/prime_validation.hpp"
#include "config/config.hpp"
#include "stats/stats_collector.hpp"
#include "mining/prime_thresholds.hpp"
#include "prime/prime.hpp"
#include "prime/sieve.hpp"
#include "block.hpp"
#include <asio.hpp>
#include <cassert>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <boost/random.hpp>

namespace nexusminer
{
namespace gpu
{
namespace
{
// Re-export the canonical constants from prime_validation.hpp.  See
// cpu/worker_prime.cpp for the rationale (single source of truth, no drift).
using nexusminer::prime::kPrimeOffsetFractionBytes;
using nexusminer::prime::kMinSerializedPrimeOffsets;
using nexusminer::prime::kMaxSerializedPrimeOffsets;
using nexusminer::prime::is_well_formed_prime_offsets;

static_assert(Cuda_sieve::chain_histogram_max + 1 == stats::kPrimeHistogramBuckets,
    "GPU chain histogram must cover every stats bucket");

constexpr std::size_t kBoostUint1kLimbBytes = sizeof(boost::multiprecision::limb_type);
}

Worker_prime::Worker_prime(std::shared_ptr<asio::io_context> io_context, config::Worker_config& config)
	: m_io_context{ std::move(io_context) }
	, m_logger{ spdlog::get("logger") }
	, m_config{ config }
	, m_prime_helper{std::make_unique<Prime>()}
	, m_segmented_sieve{std::make_unique<Sieve>()}
	, m_stop{ true }
	, m_log_leader{ "GPU Worker " + m_config.m_id + ": " }
	, m_primes{ 0 }
	, m_chains{ 0 }
	, m_difficulty{ 0 }
	, m_pool_nbits{ 0 }
	, m_gpu_initialized{false}
{

	m_segmented_sieve->generate_sieving_primes();
	m_segmented_sieve->generate_small_prime_tables();
	m_segmented_sieve->generate_trial_divisors();

	// Start persistent thread
	m_shutdown = false;
	m_run_thread = std::thread(&Worker_prime::run, this);
	m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "Persistent worker thread started"));
}

Worker_prime::~Worker_prime() noexcept
{
	m_stop = true;  // Interrupt mining loops before competing for m_mtx during shutdown

	// Signal shutdown and wake up the worker thread
	{
		std::lock_guard<std::mutex> lock(m_mtx);
		m_shutdown = true;
		m_running = false;
	}
	m_cv.notify_all();

	// Wait for thread to finish
	if (m_run_thread.joinable())
		m_run_thread.join();

	//free gpu memory
	if (m_gpu_initialized)
	{
		m_segmented_sieve->gpu_sieve_free();
		m_segmented_sieve->gpu_fermat_free();
	}
}

void Worker_prime::set_block(LLP::CBlock block, std::uint32_t nbits, Worker::Block_found_handler result)
{
	// Update work data atomically and signal worker thread
	{
		std::scoped_lock<std::mutex> lck(m_mtx);
		m_found_nonce_callback = result;
		m_block = Block_data{ block };
		if (nbits != 0)	// take nBits provided from pool
		{
			m_pool_nbits = nbits;
		}

		m_difficulty.store(effective_nbits_locked(),
		                   std::memory_order_relaxed);
		m_base_hash = m_block.GetPrimeBaseHash();
		//Now we have the hash of the block header.  We use this to feed the miner.

		//set the starting nonce for each worker to something different that won't overlap with the others
		m_starting_nonce = static_cast<uint64_t>(m_config.m_internal_id) << 48;
		m_nonce = m_starting_nonce;

		// Signal new work is available
		m_stop = true;
		m_new_work = true;
		m_running = true;
		m_template_consumed = false;  // Option D: fresh template, allow mining again
	}

	// Wake up the worker thread
	m_cv.notify_one();
}

void Worker_prime::set_block(std::shared_ptr<WorkPackage> work_package, Worker::Block_found_handler result)
{
	// Update work data atomically and signal worker thread
	{
		std::scoped_lock<std::mutex> lck(m_mtx);
		m_found_nonce_callback = result;

		// Use precomputed data from WorkPackage
		const auto& block = work_package->get_block();
		m_block = Block_data{ block };

		std::uint32_t nbits = work_package->get_nbits();
		if (nbits != 0)	// take nBits provided from pool
		{
			m_pool_nbits = nbits;
		}

		m_difficulty.store(effective_nbits_locked(),
		                   std::memory_order_relaxed);

		// Optimization: Use precomputed base hash from WorkPackage if available
		const auto& precomputed_hash = work_package->get_prime_base_hash();
		if (precomputed_hash.has_value()) {
			// Use shared precomputed hash (computed once in Worker_manager)
			m_base_hash = precomputed_hash.value();
			m_logger->debug("GPU Worker_prime: Using precomputed base hash from WorkPackage");
		} else {
			// Fallback: Compute hash locally (backward compatibility)
			m_logger->debug("GPU Worker_prime: Computing base hash locally (no precomputed hash)");
			m_base_hash = m_block.GetPrimeBaseHash();
		}
		//Now we have the hash of the block header.  We use this to feed the miner.

		//set the starting nonce for each worker to something different that won't overlap with the others
		m_starting_nonce = static_cast<uint64_t>(m_config.m_internal_id) << 48;
		m_nonce = m_starting_nonce;

		// Signal new work is available
		m_stop = true;
		m_new_work = true;
		m_running = true;
		m_template_consumed = false;  // Option D: fresh template, allow mining again
	}

	// Wake up the worker thread
	m_cv.notify_one();
}

void Worker_prime::run()
{
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

			// Initialize GPU if needed (first time)
			if (!m_gpu_initialized)
			{
				auto& worker_config_gpu = std::get<config::Worker_config_gpu>(m_config.m_worker_mode);
				m_segmented_sieve->gpu_sieve_load(worker_config_gpu.m_device);
				m_segmented_sieve->gpu_fermat_test_init(worker_config_gpu.m_device);
				m_gpu_initialized = true;
				m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "GPU memory initialized"));
			}
		}

		// Start mining with the new work
		m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "Starting GPU mining"));

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

			// Stone — per-session target Cunningham chain length T.  Mirrors
			// the CPU PrimeMiningEngine derivation (prime_mining_engine.cpp:
			// target_length = ceil(nbits / 1e7), clamped to >= 2).  Pushed
			// into the GPU sieve BEFORE clear_chains/calculate_starting_multiples
			// so the next find_chains() launch sees the new T via the
			// Cuda_sieve_properties kernel argument.  Without this plumbing
			// the GPU sieve was permanently gated at the prior compile-time
			// Cuda_sieve::m_min_chain_length=9 — silently dropping length-7/8
			// winners at pool difficulty < 9.
			const std::uint32_t nbits_for_target = effective_nbits_locked();
			int target_length = nexusminer::mining::derive_target_length(nbits_for_target);
			const int previous_target_length = m_segmented_sieve->get_target_length();
			if (previous_target_length != target_length)
			{
				m_logger->info(spdlog::fmt_lib::runtime(m_log_leader +
					"target_length " + std::to_string(previous_target_length) +
					" -> " + std::to_string(target_length) +
					" (nbits=" + std::to_string(nbits_for_target) +
					", popcount_floor=" +
					std::to_string(nexusminer::mining::popcount_window_floor(target_length)) +
					", close_chain_min=" +
					std::to_string(nexusminer::mining::close_chain_min(target_length)) +
					")"));
			}
			m_segmented_sieve->set_target_length(target_length);

			uint1k startprime = local_base_hash + local_nonce;
			m_segmented_sieve->set_sieve_start(startprime);
			local_nonce = static_cast<uint64_t>(m_segmented_sieve->get_sieve_start() - local_base_hash);
			m_segmented_sieve->clear_chains();
			m_segmented_sieve->calculate_starting_multiples();
		}
		//copy starting multiples to the sieve
		m_segmented_sieve->gpu_sieve_init();
		m_segmented_sieve->gpu_fermat_test_set_base_int(m_segmented_sieve->get_sieve_start());
		publish_statistics_snapshot();
		uint64_t sieve_batch_range = m_segmented_sieve->m_sieve_range;
		uint64_t find_chains_ms = 0;
		uint64_t sieving_ms = 0;
		uint64_t test_chains_ms = 0;
		uint64_t clean_chains_ms = 0;
		uint64_t elapsed_ms = 0;
		uint64_t low = 0;
		uint64_t range_searched_this_cycle = 0;
		uint64_t fermat_tests_this_cycle_start;
		uint64_t fermat_passes_this_cycle_start;
		uint64_t trial_division_tests, trial_division_composites;
		m_segmented_sieve->gpu_get_fermat_stats(fermat_tests_this_cycle_start, fermat_passes_this_cycle_start,
			trial_division_tests, trial_division_composites);

		//Setting debug to true can impact performance.  we will set it to true if the log level is set to debug or more verbose.
		//setting debug to true is required to measure individual kernel run time
		bool debug = m_logger->level() <= spdlog::level::level_enum::debug;
		auto start = std::chrono::steady_clock::now();
		auto interval_start = std::chrono::steady_clock::now();

		while (!m_stop)
	{
		// Check for new work at the top of the loop.
		// Option D: also exit when this template already produced a submit.
		// Otherwise GPU prime keeps grinding the same base hash after a find,
		// unlike the CPU prime path that is known to mine on testnet.
		{
			std::unique_lock<std::mutex> lck(m_mtx);
			if (m_new_work || m_template_consumed)
			{
				break;
			}
		}

		m_range_searched += sieve_batch_range;
		range_searched_this_cycle += sieve_batch_range;
		// Guard against set_block() arriving between the m_new_work check and the first GPU kernel.
		// Without this, set_sieve_start() + clear_chains() in set_block() could mutate the sieve
		// concurrently while the GPU kernels below are starting to read it.
		if (m_stop) break;

		auto sieve_start = std::chrono::steady_clock::now();
		m_segmented_sieve->gpu_sieve_small_primes(low);
		m_segmented_sieve->gpu_sieve_medium_small_primes(low);
		m_segmented_sieve->sieve_batch(low);
		m_segmented_sieve->gpu_sieve_large_primes(low);
		if (debug) m_segmented_sieve->gpu_sieve_synchronize();
		auto sieve_stop = std::chrono::steady_clock::now();
		auto sieve_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(sieve_stop - sieve_start);
		sieving_ms += sieve_elapsed.count();
		if (m_stop) break;
		auto find_chains_start = std::chrono::steady_clock::now();
		m_segmented_sieve->find_chains();
		if (debug) m_segmented_sieve->gpu_sieve_synchronize();
		auto find_chains_stop = std::chrono::steady_clock::now();
		auto find_chains_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(find_chains_stop - find_chains_start);
		find_chains_ms += find_chains_elapsed.count();
		if (m_stop) break;
		//m_segmented_sieve->do_chain_trial_division_check();
		auto test_chains_start = std::chrono::steady_clock::now();
		m_segmented_sieve->gpu_run_fermat_chain_test();
		if (debug) m_segmented_sieve->gpu_fermat_synchronize();
		auto test_chains_stop = std::chrono::steady_clock::now();
		auto test_chains_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(test_chains_stop - test_chains_start);
		test_chains_ms += test_chains_elapsed.count();
		if (m_stop) break;
		auto clean_chains_start = std::chrono::steady_clock::now();
		m_segmented_sieve->gpu_clean_chains();
		if (debug) m_segmented_sieve->gpu_sieve_synchronize();
		auto clean_chains_stop = std::chrono::steady_clock::now();
		auto clean_chains_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(clean_chains_stop - clean_chains_start);
		clean_chains_ms += clean_chains_elapsed.count();
		if (m_stop) break;
		//check for winners
		m_segmented_sieve->get_long_chains();
		const double required_difficulty = getNetworkDifficulty();
		//check difficulty of any chains that passed through the filter
		for (auto x : m_segmented_sieve->m_long_chain_starts)
		{
			local_block.nNonce = local_nonce + x;
			uint1k chain_start = local_base_hash + local_block.nNonce;
			uint1024_t hashPrime = boost_uint1024_t_to_uint1024_t(chain_start);
			std::vector<uint8_t> offsets;
			double actual_difficulty = 0.0;

			bool is_valid = nexusminer::prime::ValidatePrimeCandidate(
				hashPrime,
				required_difficulty,
				offsets,
				actual_difficulty);

			m_segmented_sieve->m_best_chain = std::max(actual_difficulty, m_segmented_sieve->m_best_chain);
			m_logger->debug("Actual difficulty {} required {}", actual_difficulty, required_difficulty);
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

				m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "✓ FOUND VALID PRIME BLOCK! Difficulty: {:.6f} (required: {:.6f}), offsets: {}"),
					actual_difficulty, required_difficulty, offsets.size());

				//we found a valid chain.  submit it.
				if (m_found_nonce_callback)
				{
					m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "💎 Block found! Posting to main io_context..."));
					auto captured_offsets = std::move(offsets);
					auto block_copy = local_block;
					::asio::post(*m_io_context, [self = shared_from_this(), block_copy, captured_offsets = std::move(captured_offsets)]()
					{
						auto bd = std::make_unique<Block_data>(block_copy);
						bd->vOffsets = captured_offsets;
						self->m_found_nonce_callback(self->m_config.m_internal_id,
							std::move(bd));
					});
					// Option D: stop grinding this template.  Do not set the
					// flag when the callback is null — that matches CPU prime
					// and avoids idling a worker that never submitted.
					{
						std::scoped_lock<std::mutex> lck(m_mtx);
						if (!m_new_work)
							m_template_consumed = true;
					}
				}
				else
				{
					m_logger->debug(spdlog::fmt_lib::runtime(m_log_leader + "Miner callback function not set."));
				}
			}
		}
		m_segmented_sieve->gpu_get_stats();
		publish_statistics_snapshot();
		m_segmented_sieve->m_long_chain_starts = {};
		low += sieve_batch_range;
		if (m_stop) break;

		//debug
		auto end = std::chrono::steady_clock::now();
		auto interval_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - interval_start); 
		
		if (debug && interval_elapsed.count() > 10000)
		{
			uint64_t fermat_test_count, fermat_prime_count, fermat_tests_this_cycle, trial_divisions, trial_division_composites;
			m_segmented_sieve->gpu_get_fermat_stats(fermat_test_count, fermat_prime_count, trial_divisions, trial_division_composites);
			fermat_tests_this_cycle = fermat_test_count - fermat_tests_this_cycle_start;
			auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
			elapsed_ms = elapsed.count();
			double chains_per_mm = 1.0e6 * m_segmented_sieve->m_chain_count / m_range_searched;
			double chains_per_sec = 1.0e3 * m_segmented_sieve->m_chain_count / elapsed_ms;
			double fermat_positive_rate = 1.0 * fermat_prime_count / fermat_test_count;
			double fermat_tests_per_chain = 1.0 * fermat_test_count / m_segmented_sieve->m_chain_count;
			std::stringstream ss;
			ss << std::fixed << std::setprecision(2) << m_range_searched /1.0e12 << " trillion integers searched." <<
				" Found " << chains_per_mm << " chain candidates per million integers." << std::endl;
			/*std::cout << "Avg chain length: " << std::fixed << std::setprecision(2) << 1.0 * m_segmented_sieve->m_chain_candidate_total_length / m_segmented_sieve->m_chain_count
				<< " Max chain: " << m_segmented_sieve->m_chain_candidate_max_length << std::endl;*/
			ss << "Fermat test rate: " << 1.0* fermat_tests_this_cycle /(double)test_chains_ms << "k tests/s. Fermat Positive Rate: " << std::fixed << std::setprecision(3) <<
				100.0 * fermat_positive_rate << "% Fermat tests per million integers sieved: " <<
				1.0e6 * fermat_test_count / m_range_searched << std::endl;

			ss << "Search rate: " << std::fixed << std::setprecision(2) << range_searched_this_cycle / (elapsed.count() * 1.0e6) << " billion integers per second." << std::endl;
			ss << "Elapsed time: " << std::fixed << std::setprecision(2) << elapsed_ms / 1000.0 << "s. Sieving: " <<
				100.0 * sieving_ms / elapsed_ms << "% Chain filtering: " << 100.0 * find_chains_ms / elapsed_ms
				<< "% Fermat testing: " << 100.0 * test_chains_ms / elapsed_ms << "% Clean chains: " << 100.0 * clean_chains_ms / elapsed_ms <<
				"% Other: " << 100.0 * (elapsed_ms - (sieving_ms + find_chains_ms + test_chains_ms + clean_chains_ms)) / elapsed_ms << "%";
			interval_start = std::chrono::steady_clock::now();
			m_logger->debug(ss.str());
		}
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
	return m_difficulty.load(std::memory_order_relaxed) / 10000000.0;
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
	// Issue 3A: typed downcast (mode is invariant by construction).
	auto& typed = stats::as_typed<stats::Prime>(stats_collector);
	typed.update_worker_stats(m_config.m_internal_id, *m_published_stats.load());
}

void Worker_prime::publish_statistics_snapshot()
{
	stats::Prime prime_stats;
	prime_stats.m_primes = stats::saturating_prime_stat(m_segmented_sieve->m_fermat_prime_count);
	prime_stats.m_chains = stats::saturating_prime_stat(m_segmented_sieve->m_chain_count);
	prime_stats.m_chain_histogram = stats::copy_prime_histogram(m_segmented_sieve->m_chain_histogram);
	prime_stats.m_range_searched = m_range_searched;
	prime_stats.m_most_difficult_chain = m_segmented_sieve->m_best_chain;
	{
		// Snapshot m_difficulty under the lock for ordering with the
		// matching m_block snapshot in run() — even though the load
		// itself is now atomic, we keep the lock so the two values
		// (difficulty + most-recent block) stay session-consistent.
		std::scoped_lock<std::mutex> lck(m_mtx);
		prime_stats.m_difficulty = m_difficulty.load(std::memory_order_relaxed);
	}
	m_published_stats.store(std::move(prime_stats));
}



}
}
