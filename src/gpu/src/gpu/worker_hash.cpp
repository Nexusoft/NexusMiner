#include "gpu/worker_hash.hpp"
#include "config/config.hpp"
#include "stats/stats_collector.hpp"
#include "block.hpp"
#include <asio/io_context.hpp>
#include "cuda/util.h"
#include "cuda/sk1024.h"
#include "LLC/hash/SK.h"

namespace nexusminer
{
namespace gpu
{

Worker_hash::Worker_hash(std::shared_ptr<asio::io_context> io_context, Worker_config& config)
: m_io_context{std::move(io_context)}
, m_logger{spdlog::get("logger")}
, m_config{config}
, m_found_nonce_callback{}
, m_stop{true}
, m_log_leader{ "GPU Worker " + m_config.m_id + ": " }
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
    //make sure the run thread exits the loop
    m_stop = true;
    if (m_run_thread.joinable())
    {
        m_run_thread.join();
    }

    // Free the GPU device memory associated with hashing
    cuda_sk1024_free(m_config.m_internal_id);

    // Free the GPU device memory and reset them
    cuda_free(m_config.m_internal_id);
}

void Worker_hash::set_block(LLP::CBlock block, std::uint32_t nbits, Worker::Block_found_handler result)
{
    //stop the existing mining loop if it is running
    m_stop = true;
    if (m_run_thread.joinable())
    {
        m_run_thread.join();
    }


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

    //restart the mining loop
    m_stop = false;
    m_run_thread = std::thread(&Worker_hash::run, this);
}

void Worker_hash::run()
{
    while (!m_stop)
    {
        m_hashes = 0;

        // Do hashing on a CUDA device
        bool found = cuda_sk1024_hash(
            m_config.m_internal_id,
            reinterpret_cast<uint32_t*>(&m_block.nVersion),
            m_target,
            m_block.nNonce,
            &m_hashes,
            m_throughput,
            m_threads_per_block,
            m_block.nHeight);

        /* If a nonce with the right diffulty was found, return true and submit block. */
        if (found && !m_stop.load())
        {
            /* Calculate the number of leading zero-bits and display. */
            uint1024_t hashProof = LLC::SK1024(BEGIN(m_block.nVersion), END(m_block.nNonce));
            uint32_t nBits = hashProof.BitCount();
            uint32_t nLeadingZeroes = 1024 - nBits;
           // debug::log(0, "[MASTER] Found Hash Block ");
           // block.print();

            if (m_found_nonce_callback)
            {
                m_io_context->post([self = shared_from_this()]()
                {
                    self->m_found_nonce_callback(self->m_config.m_internal_id, std::make_unique<Block_data>(self->m_block));
                });
            }
            else
            {
                m_logger->debug(m_log_leader + "Miner callback function not set.");
            }

            m_stop = true;
        }

    }
}


void Worker_hash::update_statistics(stats::Collector& stats_collector)
{
  //  stats_collector.update_worker_stats(m_config.m_internal_id,
   //     stats::Hash{ m_hashes, m_best_leading_zeros, m_met_difficulty_count });
}

}
}