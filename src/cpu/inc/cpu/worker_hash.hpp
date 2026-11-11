#ifndef NEXUSMINER_CPU_WORKER_HASH_HPP
#define NEXUSMINER_CPU_WORKER_HASH_HPP
//a software hash channel miner for demo and test purposes

#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include "worker.hpp"
#include "block.hpp"
#include "hash/nexus_skein.hpp"
#include "hash/nexus_keccak.hpp"
#include <spdlog/spdlog.h>

namespace asio { class io_context; }

namespace nexusminer {
namespace config{ class Worker_config; }
namespace stats { class Collector; }

namespace cpu
{
struct Worker_hash_test_access;

class Worker_hash : public Worker, public std::enable_shared_from_this<Worker_hash>
{
public:

    using Worker_config = config::Worker_config;

    Worker_hash(std::shared_ptr<asio::io_context> io_context, Worker_config& config);
    ~Worker_hash();

    // Sets a new block (nexus data type) for the miner worker. The miner worker must reset the current work.
    // When  the worker finds a new block, the BlockFoundHandler has to be called with the found BlockData
    void set_block(::LLP::CBlock block, std::uint32_t nbits, Worker::Block_found_handler result) override;

    // Optimized version: accepts shared WorkPackage to eliminate repeated block data construction
    void set_block(std::shared_ptr<WorkPackage> work_package, Worker::Block_found_handler result) override;

    bool is_running() const override { return m_running.load(); }
    void update_statistics(stats::Collector& stats_collector) override;

private:
    friend struct Worker_hash_test_access;

    void run();
    void mine_loop(uint32_t thread_id, uint32_t total_threads);
    bool difficulty_check();
    std::uint64_t leading_zero_mask();  
    
    // Validation and debugging methods
    bool validate_skein_output(const NexusSkein::stateType& skeinHash) const;
    bool validate_keccak_output(uint64_t keccakHash) const;
    bool cross_validate_hashes(const NexusSkein::stateType& skeinHash, uint64_t keccakHash) const;
    void log_skein_state(const NexusSkein::stateType& skeinHash, uint64_t nonce) const;
    void log_hash_mismatch(const NexusSkein::stateType& skeinHash, uint64_t keccakHash, uint64_t nonce) const;
    void log_midstate_calculation();
 

    //Poor man's difficulty.  Report any nonces with at least this many leading zeros. Let the software perform additional filtering. 
    static constexpr int leading_zeros_required = 20;    //set lower to find more nonce candidates

    std::shared_ptr<asio::io_context> m_io_context;
    std::shared_ptr<spdlog::logger> m_logger;
    Worker_config& m_config;
    std::atomic<bool> m_stop;
    std::atomic<bool> m_running{false};  // true once set_block() has been called with valid work
    std::thread m_run_thread;
    std::vector<std::thread> m_worker_threads;  // For multi-threading support
    Worker::Block_found_handler m_found_nonce_callback;
    NexusSkein m_skein;
    Block_data m_block;
    std::mutex m_mtx;
    std::condition_variable m_cv;  // For persistent thread wake-up
    bool m_new_work = false;       // Flag to indicate new work is available
    bool m_shutdown = false;       // Flag to indicate worker should shut down
    // Option D: set true at the moment the worker dispatches the found-block
    // callback for the current template, cleared in set_block().  Checked at
    // the top of every mine_loop iteration alongside m_new_work so the worker
    // stops grinding the spent template and falls back to m_cv.wait until the
    // next set_block (which clears the flag).  Without this, after a found
    // block the worker keeps hashing the same template — at low rates on a
    // quiet channel, that means the next find rebuilds against the same
    // already-mined nonce space until the new BLOCK_DATA push lands.
    bool m_template_consumed = false;
    uint64_t m_starting_nonce = 0;
    std::string m_log_leader;
 
    void reset_statistics();
    std::uint64_t m_hash_count;
    int m_best_leading_zeros;
    int m_met_difficulty_count;

    std::uint32_t m_pool_nbits;

};
}

}


#endif
