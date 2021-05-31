#include "miner.hpp"

#include "network/create_component.hpp"
#include "network/component.hpp"
#include "chrono/timer_factory.hpp"
#include "worker_manager.hpp"
#include "worker.hpp"
#include "worker_software_hash/worker_software_hash.hpp"

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>

#include <asio/io_context.hpp>
#include <chrono>

namespace nexusminer
{

	Miner::Miner() 
	: m_io_context{std::make_shared<::asio::io_context>()}
	, m_signals{std::make_shared<::asio::signal_set>(*m_io_context)}
	{
		m_logger = spdlog::stdout_color_mt("logger");
		m_logger->set_level(spdlog::level::debug);

		// Register to handle the signals that indicate when the server should exit.
		// It is safe to register for the same signal multiple times in a program,
		// provided all registration for the specified signal is made through Asio.
		m_signals->add(SIGINT);
		m_signals->add(SIGTERM);
#if defined(SIGQUIT)
		m_signals->add(SIGQUIT);
#endif 

		m_signals->async_wait([this](auto, auto)
		{
			m_logger->info("Shutting down NexusMiner");
			m_worker_manager->stop();
			m_io_context->stop();
			exit(1);
		});
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

		// std::err logger
		auto console_err = spdlog::stderr_color_mt("console_err");

		// timer initialisation
		chrono::Timer_factory::Sptr timer_factory = std::make_shared<chrono::Timer_factory>(m_io_context);

		// network initialisation
		m_network_component = network::create_component(m_io_context);

		network::Endpoint local_endpoint{ network::Transport_protocol::tcp, "127.0.0.1", 0 };
		m_worker_manager = std::make_unique<Worker_manager>(m_config, timer_factory, 
			m_network_component->get_socket_factory()->create_socket(local_endpoint));
		

		m_worker_manager->add_worker(std::make_shared<Worker_software_hash>(m_io_context));
		return true;
	}

	void Miner::run()
	{
		network::Endpoint wallet_endpoint{ network::Transport_protocol::tcp, m_config.get_wallet_ip(), m_config.get_port() };
		auto result = m_worker_manager->connect(wallet_endpoint);
		if (!result)
		{
			m_logger->error("Failed to initialise socket. Result: {}", result);
			return;
		}

		m_io_context->run();

	}

}