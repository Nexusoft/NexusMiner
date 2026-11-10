#ifndef NEXUSMINER_GPU_WORKER_HASH_HPP
#define NEXUSMINER_GPU_WORKER_HASH_HPP

#include <memory>
#include <atomic>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "worker.hpp"
#include "LLC/types/uint1024.h"
#include <spdlog/spdlog.h>

namespace asio { class io_context; }

namespace nexusminer {
namespace config{ class Worker_config; }
namespace stats { class Collector; }

namespace gpu
{
// Forward-declared test accessor — only a non-null body is provided
// when WITH_GPU_HOST_STUBS=ON (in src/gpu/test/worker_hash_integration_test.cpp).
// The friend declaration is unconditional so the header compiles in all modes.
struct Worker_hash_test_access;

class Worker_hash : public Worker, public std::enable_shared_from_this<Worker_hash>
{
public:

    using Worker_config = config::Worker_config;

    Worker_hash(std::shared_ptr<asio::io_context> io_context, Worker_config& config);
    ~Worker_hash();

    // Sets a new block (nexus data type) for the miner worker. The miner worker must reset the current work.
    // When  the worker finds a new block, the BlockFoundHandler has to be called with the found BlockData
    // ::LLP, not LLP.  worker_manager includes miner_opcodes.hpp first, which
    // opens nexusminer::LLP and would hide the block type.
    void set_block(::LLP::CBlock block, std::uint32_t nbits, Worker::Block_found_handler result) override;

    // Optimized version: accepts shared WorkPackage to eliminate repeated block data construction
    void set_block(std::shared_ptr<WorkPackage> work_package, Worker::Block_found_handler result) override;

    bool is_running() const override { return m_running.load(); }
    void update_statistics(stats::Collector& stats_collector) override;

private:
    friend struct Worker_hash_test_access;

    void run();
    std::uint32_t device_id() const;
    bool bind_device_context(const char* phase);

    // Inner-loop body extracted for testability (WITH_GPU_HOST_STUBS integration
    // tests call this directly without going through the run() background thread).
    // Returns true when the caller should break the mining loop (winner credited
    // or keccak-mismatch fault threshold reached).
    bool step_once(Block_data& local_block,
                   const uint1024_t& local_target,
                   std::uint32_t local_throughput,
                   std::uint32_t local_threads_per_block,
                   std::uint32_t local_device_id);

    std::shared_ptr<asio::io_context> m_io_context;
    std::shared_ptr<spdlog::logger> m_logger;
    Worker_config& m_config;
    Worker::Block_found_handler m_found_nonce_callback;
    std::atomic<bool> m_stop;
    std::atomic<bool> m_running{false};  // true once set_block() has been called with valid work
    std::thread m_run_thread;
    std::mutex m_mtx;
    std::condition_variable m_cv;  // For persistent thread wake-up
    bool m_new_work = false;       // Flag to indicate new work is available
    bool m_shutdown = false;       // Flag to indicate worker should shut down

    std::string m_log_leader;
    Block_data m_block;
    std::uint32_t m_pool_nbits;
    uint1024_t m_target;
    // PR #681 follow-up §4(4): worker thread writes (Worker_hash::run, after
    // every kernel call), stats thread reads + resets (update_statistics).
    // Plain uint64_t was a TSan-flaggable data race; relaxed atomic is enough
    // because the stats path only needs eventual visibility — no ordering
    // requirement against any other memory operation.
    std::atomic<std::uint64_t> m_hashes{0};
    std::uint32_t m_intensity;
    std::uint32_t m_throughput;
    std::uint32_t m_threads_per_block;
    std::atomic<int> m_best_leading_zeros;
    std::atomic<int> m_met_difficulty_count;

    // Stone — bug #7 fix: per-worker nonce sharding.
    // Worker_prime carefully shards via (m_internal_id << 48); CPU worker_hash
    // does the same (src/cpu/src/cpu/worker_hash.cpp:88).  GPU worker_hash
    // previously just used m_block.nNonce as-is, so multiple GPU hash workers
    // on the same template would collide in nonce space.  We now compute and
    // store the same shard per session.
    std::uint64_t m_starting_nonce = 0;

    // Stone — bug #8 fix: keccak (CPU revalidation) mismatch tracking.
    // The CUDA kernel previously printed a `std::cout` line on mismatch and
    // kept mining; sk1024_cpu_hash now reports mismatches via an out-counter
    // so the worker can route them through spdlog and treat repeated
    // mismatches as a hardware fault (matching ccminer/T-Rex patterns).
    // After kKeccakMismatchFaultThreshold consecutive mismatches in a session
    // we set m_running=false; a healthy submission resets the counter.
    //
    // PR #681 follow-up §5: 3 is conservative compared to ccminer/T-Rex (5
    // for stale shares).  We pick 3 because a Nexus block winner is much
    // higher value than a pool share — the false-positive cost (taking a
    // healthy worker offline for 3 windows of mismatch noise) is a small
    // fraction of the false-negative cost (feeding the pool a winner the
    // host can't revalidate).  Do NOT tune this up without re-evaluating
    // that asymmetry.
    static constexpr std::uint32_t kKeccakMismatchFaultThreshold = 3;
    std::atomic<std::uint32_t> m_keccak_mismatch_count{0};
    std::atomic<std::uint32_t> m_keccak_mismatch_total{0};

};
}

}


#endif
