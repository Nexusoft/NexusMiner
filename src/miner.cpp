#include "miner.hpp"

#include "network/create_component.hpp"
#include "network/component.hpp"
#include "worker_manager.hpp"
#include "worker.hpp"

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>

#include <chrono>

namespace nexusminer
{

Miner::Miner() : m_io_context{std::make_shared<::asio::io_context>()}
{
}

Miner::~Miner()
{
	m_io_context->stop();
}

bool Miner::init()
{
	if (!m_config.read_config())
	{
		return false;
	}		

	m_logger = spdlog::stdout_color_mt("logger");
	m_logger->set_level(spdlog::level::debug);

	// std::err logger
	auto console_err = spdlog::stderr_color_mt("console_err");

	// network initialisation
	m_network_component = network::create_component(m_io_context);

	network::Endpoint local_endpoint{network::Transport_protocol::tcp, "127.0.0.1", 0}; 
	m_worker_manager = std::make_unique<Worker_manager>(m_config, m_network_component->get_socket_factory()->create_socket(local_endpoint));

	return true;
}

void Miner::run()
{
	network::Endpoint wallet_endpoint{network::Transport_protocol::tcp, m_config.get_wallet_ip(), m_config.get_port()};
	auto result = m_worker_manager->connect(wallet_endpoint);
	if(result != network::Result::ok)
	{
		m_logger->error("Failed to initialise socket. Result: {}", result);
		return;
	}
}

}