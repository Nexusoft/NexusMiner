#ifndef NEXUSMINER_GPU_WORKER_HASH_HPP
#define NEXUSMINER_GPU_WORKER_HASH_HPP

#include <memory>
#include <atomic>
#include <string>
#include <thread>
#include "worker.hpp"
#include "LLC/types/uint1024.h"
#include <spdlog/spdlog.h>

namespace asio { class io_context; }

namespace nexusminer {
namespace config{ class Worker_config; }
namespace stats { class Collector; }

namespace gpu
{
class Worker_hash : public Worker, public std::enable_shared_from_this<Worker_hash>
{
public:

    using Worker_config = config::Worker_config;

    Worker_hash(std::shared_ptr<asio::io_context> io_context, Worker_config& config);
    ~Worker_hash();

    // Sets a new block (nexus data type) for the miner worker. The miner worker must reset the current work.
    // When  the worker finds a new block, the BlockFoundHandler has to be called with the found BlockData
    void set_block(LLP::CBlock block, std::uint32_t nbits, Worker::Block_found_handler result) override;
    void update_statistics(stats::Collector& stats_collector) override;

private:

    void run();

    std::shared_ptr<asio::io_context> m_io_context;
    std::shared_ptr<spdlog::logger> m_logger;
    Worker_config& m_config;
    Worker::Block_found_handler m_found_nonce_callback;
    std::atomic<bool> m_stop;
    std::thread m_run_thread;

    std::string m_log_leader;
    Block_data m_block;
    std::uint32_t m_pool_nbits;
    uint1024_t m_target;
    std::uint64_t m_hashes = 0;
    std::uint32_t m_intensity;
    std::uint32_t m_throughput;
    std::uint32_t m_threads_per_block;
    int m_best_leading_zeros;
    int m_met_difficulty_count;

};
}

}


#endif