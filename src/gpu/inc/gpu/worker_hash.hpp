#ifndef NEXUSMINER_GPU_WORKER_HASH_HPP
#define NEXUSMINER_GPU_WORKER_HASH_HPP

#include <memory>
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

    std::shared_ptr<asio::io_context> m_io_context;
    std::shared_ptr<spdlog::logger> m_logger;
    Worker_config& m_config;

    uint1024_t target;
    std::uint64_t hashes;
    std::uint32_t intensity;
    std::uint32_t throughput;
    std::uint32_t threads_per_block;

};
}

}


#endif