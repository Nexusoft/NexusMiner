#include "cpu/worker_prime.hpp"
#include "config/config.hpp"
#include "stats/stats_collector.hpp"
#include "prime/prime.hpp"
#include "block.hpp"
#include <asio.hpp>

namespace nexusminer
{
namespace cpu
{
Worker_prime::Worker_prime(std::shared_ptr<asio::io_context> io_context, config::Worker_config& config)
	: m_io_context{ std::move(io_context) }
	, m_logger{ spdlog::get("logger") }
	, m_config{ config }
	, m_stop{ true }
	, m_log_leader{ "CPU Worker " + m_config.m_id + ": " }
	, m_hash_count{ 0 }
	, m_best_leading_zeros{ 0 }
	, m_met_difficulty_count{ 0 }
	, m_pool_nbits{ 0 }
{

}

Worker_prime::~Worker_prime() noexcept
{
	//make sure the run thread exits the loop
	m_stop = true;
	if (m_run_thread.joinable())
		m_run_thread.join();
}

void Worker_prime::set_block(LLP::CBlock block, std::uint32_t nbits, Worker::Block_found_handler result)
{
	//stop the existing mining loop if it is running
	m_stop = true;
	if (m_run_thread.joinable())
	{
		m_run_thread.join();
	}

	{
		std::scoped_lock<std::mutex> lck(m_mtx);
		m_found_nonce_callback = result;
		m_block = Block_data{ block };

	}
	//restart the mining loop
	m_stop = false;
	m_run_thread = std::thread(&Worker_prime::run, this);
}

void Worker_prime::run()
{
	while (!m_stop)
	{

	}
}

void Worker_prime::update_statistics(stats::Collector& stats_collector)
{
	std::scoped_lock<std::mutex> lck(m_mtx);


}


}
}