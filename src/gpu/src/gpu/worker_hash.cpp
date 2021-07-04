#include "gpu/worker_hash.hpp"
#include "config/config.hpp"
#include "stats/stats_collector.hpp"
#include "block.hpp"
#include <asio/io_context.hpp>
#include "cuda/util.h"
#include "cuda/sk1024.h"

namespace nexusminer
{
namespace gpu
{

Worker_hash::Worker_hash(std::shared_ptr<asio::io_context> io_context, Worker_config& config)
: m_io_context{std::move(io_context)}
, m_logger{spdlog::get("logger")}
, m_config{config}
, threads_per_block{896}
{	
    // Initialize the cuda device associated with this ID
    cuda_init(m_config.m_internal_id);

    // Allocate memory associated with Device Hashing
    cuda_sk1024_init(m_config.m_internal_id);

    // Compute the intensity by determining number of multiprocessors
    intensity = 2 * cuda_device_multiprocessors(m_config.m_internal_id);
    m_logger->debug("{} intensity set to {}", cuda_devicename(m_config.m_internal_id), intensity);

    // Calcluate the throughput for the cuda hash mining
    throughput = 256 * threads_per_block * intensity;
}

Worker_hash::~Worker_hash() 
{ 
    // Free the GPU device memory associated with hashing
    cuda_sk1024_free(m_config.m_internal_id);

    // Free the GPU device memory and reset them
    cuda_free(m_config.m_internal_id);
}

void Worker_hash::set_block(LLP::CBlock block, std::uint32_t nbits, Worker::Block_found_handler result)
{
}


void Worker_hash::update_statistics(stats::Collector& stats_collector)
{
}

}
}