#ifndef NEXUSMINER_CPU_WORKER_PRIME_HPP
#define NEXUSMINER_CPU_WORKER_PRIME_HPP
//a software prime channel miner for demo and test purposes

#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include "worker.hpp"
#include "block.hpp"
#include "hash/nexus_skein.hpp"
#include "hash/nexus_keccak.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <spdlog/spdlog.h>
#include "LLC/types/bignum.h"
#include "mining/mining_constants.hpp"
#include "stats/types.hpp"
#include "stats/prime_stats_snapshot.hpp"

namespace asio { class io_context; }

namespace nexusminer {
namespace config { class Worker_config; }
namespace stats { class Collector; }

namespace cpu
{
    using uint1k = boost::multiprecision::uint1024_t;
    class Prime;
    class Sieve;
    class Segment_allocator;
    class PrimeMiningEngine;
class Worker_prime : public Worker, public std::enable_shared_from_this<Worker_prime>
{
public:

    Worker_prime(std::shared_ptr<asio::io_context> io_context, config::Worker_config& config);
    ~Worker_prime() noexcept override;

    // Force-initialise the process-shared CPU sieving prime table on the calling
    // thread.  Worker_prime constructors otherwise trigger this lazily on the
    // first worker that is spawned, which on a high-core-count machine means N
    // workers race into the C++ magic-static once-init in parallel.  Calling
    // this once from Worker_manager (main thread) before spawning prime workers
    // keeps the one-time generate_primes() pass on a deterministic thread and
    // produces a clean startup log, with no behavioural change.
    static void prewarm_shared_state();

    void set_block(::LLP::CBlock block, std::uint32_t nbits, Worker::Block_found_handler result) override;

    // Optimized version: accepts shared WorkPackage to eliminate repeated block data construction
    void set_block(std::shared_ptr<WorkPackage> work_package, Worker::Block_found_handler result) override;

    // ── Stone 7 — engine-mode adapter API ─────────────────────────────────
    //
    // Under engine_mode == "engine" the worker is a thin adapter that owns
    // no mining thread and no sieve.  Worker_manager calls bind_to_engine()
    // after the engine is constructed and every worker has been registered;
    // the worker stores a weak_ptr for stats fan-in (it deliberately does
    // NOT take a shared_ptr so the engine can be torn down before workers).
    //
    // share_index ∈ [0, share_count): which slot of the floor+remainder
    //   partition this worker owns.  share_count is the total number of
    //   registered Worker_prime instances; both are stable for the engine
    //   lifetime under Stone 7.
    void bind_to_engine(std::weak_ptr<class PrimeMiningEngine> engine,
                        std::uint32_t share_index,
                        std::uint32_t share_count);

    // Stone 7 — opt out of the legacy per-worker set_block() shim.  In
    // engine mode the engine subscribes to WorkerTemplateFeed itself; the
    // worker would otherwise receive a redundant template and have no
    // mining thread to consume it.
    bool uses_template_feed() const override;

    // Stone 7 — explicit no-op override.  The feed pointer is held by the
    // engine, not the worker.  Documented inline so future readers don't
    // mistake the empty body for a missing implementation.
    void attach_template_feed(std::shared_ptr<WorkerTemplateFeed> feed) override;

    bool is_running() const override;
    void update_statistics(stats::Collector& stats_collector) override;

private:

    void run();
    void set_block_impl(Block_data block_data,
                        std::uint32_t nbits,
                        const uint1k& base_hash,
                        Worker::Block_found_handler result);
    double getDifficulty(const uint1k& p);
    double getNetworkDifficulty();
    bool difficulty_check(const uint1k& p);
    //std::uint64_t leading_zero_mask();
    bool isPrime(uint1k p);
    void fermat_performance_test();
    void publish_statistics_snapshot();

    //Poor man's difficulty.  Report any nonces with at least this many leading zeros. Let the software perform additional filtering. 
    //static constexpr int leading_zeros_required = 20;    //set lower to find more nonce candidates

    std::shared_ptr<asio::io_context> m_io_context;
    std::shared_ptr<spdlog::logger> m_logger;
    config::Worker_config& m_config;
    std::unique_ptr<Prime> m_prime_helper;
    std::atomic<bool> m_stop;
    std::atomic<bool> m_running{false};  // true once set_block() has been called with valid work
    std::atomic<bool> m_initialized;
    std::thread m_run_thread;
    std::vector<std::thread> m_worker_threads;  // For multi-threading support
    Worker::Block_found_handler m_found_nonce_callback;
    std::unique_ptr<Sieve> m_segmented_sieve;
    std::unique_ptr<Segment_allocator> m_segment_allocator;

    Block_data m_block;
    std::mutex m_mtx;
    std::condition_variable m_cv;  // For persistent thread wake-up
    bool m_new_work = false;       // Flag to indicate new work is available
    bool m_shutdown = false;       // Flag to indicate worker should shut down
    // Option D: set true the moment the worker dispatches the found-block
    // callback for the current template.  Cleared in set_block_impl() when
    // a fresh template arrives.  Checked at the top of the inner mining
    // loop alongside m_new_work so the worker stops grinding the spent
    // template and falls back to m_cv.wait until the next set_block.
    bool m_template_consumed = false;
    std::uint64_t m_starting_nonce = 0;
    std::string m_log_leader;

    std::uint32_t m_primes{ 0 };
    std::uint32_t m_chains{ 0 };
    std::uint32_t m_difficulty{ 0 };

    std::uint32_t m_pool_nbits;

    std::uint64_t m_nonce = 0;
    uint1k m_base_hash;
    uint1k m_active_base_hash{};
    bool m_has_active_cycle = false;


    void generate_seive(uint1k);
    void analyze_chains();
    void mine_region(uint1k);
    static LLC::CBigNum boost_uint1024_t_to_CBignum(const uint1k&);
    static uint1024_t boost_uint1024_t_to_uint1024_t(const uint1k&);


    std::vector<bool>m_sieve;
    static constexpr int m_primorialEndPrime = mining::CPU_PRIMORIAL_END_PRIME;
    static constexpr int m_minChainLength = mining::MIN_CHAIN_LENGTH;  //min chain length
    static constexpr int m_maxGap = mining::MAX_PRIME_GAP_HALF;  //the largest allowable prime gap.Divide by two because the sieve excludes all even numbers.

    //we have finite memory so we have to limit the sieve to some reasonable size
    static constexpr int m_maxSieveLength = mining::CPU_SIEVE_LENGTH_MAX;
    static constexpr int m_sieveLength = m_maxSieveLength;
    static constexpr int m_sieveRange = m_sieveLength * 2;

    //Vectors containing information about chains we find
    //int chainListLength = std::min(sieveLength, 10000000);
    std::vector<uint64_t> m_chainStartPosArray;
    std::vector<int> m_chainLengthArray;
    std::vector<std::vector<int>> m_chainOffsets;
    int m_chainCount = 0;  //number of chains found
    int m_candidateCount = 0;  //number of possible primes in the sieve

    //stats
    std::vector<std::uint32_t> m_chain_histogram;
    std::atomic<uint64_t> m_range_searched{0};
    
    // CPU load tracking
    std::chrono::steady_clock::time_point m_cpu_tracking_start;
    std::chrono::milliseconds m_cpu_active_time{0};
    std::chrono::milliseconds m_cpu_total_time{0};
    nexusminer::stats::Atomic_snapshot<nexusminer::stats::Prime> m_published_stats;

    // ── Stone 7 — engine-mode adapter state ───────────────────────────────
    // m_engine_mode is "workers" (legacy mining loop) or "engine" (thin
    // adapter — engine owns the mining threads, this worker is just a
    // stats/identity shell).  Captured once at construction from
    // Worker_config_cpu::m_engine_mode and never re-read.
    std::string m_engine_mode{"workers"};

    // Engine reference (weak so engine teardown ordering — engine BEFORE
    // workers, see Worker_manager::stop_all_workers — does not crash here).
    // Set by bind_to_engine(); read by update_statistics() and is_running()
    // under engine mode.
    std::weak_ptr<class PrimeMiningEngine> m_prime_engine_weak;
    std::uint32_t m_engine_share_index{0};
    std::uint32_t m_engine_share_count{0};
    std::atomic<bool> m_engine_bound{false};

};
}

}


#endif
