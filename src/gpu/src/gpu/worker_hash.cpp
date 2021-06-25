#include "gpu/worker_hash.hpp"
#include "config/config.hpp"
#include "stats/stats_collector.hpp"
#include "block.hpp"
#include <asio.hpp>

namespace nexusminer
{
namespace gpu
{

Worker_hash::Worker_hash(std::shared_ptr<asio::io_context> io_context, Worker_config& config)
{
	
}

Worker_hash::~Worker_hash() 
{ 
}

void Worker_hash::set_block(LLP::CBlock block, std::uint32_t nbits, Worker::Block_found_handler result)
{
}


void Worker_hash::update_statistics(stats::Collector& stats_collector)
{
}

}
}