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
, m_found_nonce_callback{}
, m_pool_nbits{0}
, m_threads_per_block{896}
{	
    // Initialize the cuda device associated with this ID
    cuda_init(m_config.m_internal_id);

    // Allocate memory associated with Device Hashing
    cuda_sk1024_init(m_config.m_internal_id);

    // Compute the intensity by determining number of multiprocessors
    m_intensity = 2 * cuda_device_multiprocessors(m_config.m_internal_id);
    m_logger->debug("{} intensity set to {}", cuda_devicename(m_config.m_internal_id), m_intensity);

    // Calcluate the throughput for the cuda hash mining
    m_throughput = 256 * m_threads_per_block * m_intensity;
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
    m_found_nonce_callback = result;
    m_block = Block_data{block};
    if (nbits != 0)
    {
        // take nbits provided by pool
        m_pool_nbits = nbits;
    }

    // Set the block for this device
    cuda_sk1024_setBlock(&m_block.nVersion, m_block.nHeight);

    /* Get the target difficulty. */
    //CBigNum target;
   // target.SetCompact(block.nBits);
  //  m_target = target.getuint1024();
    auto const nbits_cuda = m_pool_nbits != 0 ? m_pool_nbits : m_block.nBits;

    // Set the target hash on this device for the difficulty.
    cuda_sk1024_set_Target((uint64_t*)nbits_cuda);
}


void Worker_hash::update_statistics(stats::Collector& stats_collector)
{
}

}
}