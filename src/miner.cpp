#include "miner.hpp"

#include "network/create_component.hpp"
#include "network/component.hpp"
#include "chrono/timer_factory.hpp"
#include "config/validator.hpp"
#include "worker_manager.hpp"
#include "worker.hpp"
#include "version.h"

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>

#include <asio.hpp>
#include <chrono>
#include <fstream>
#include <iostream>

namespace nexusminer
{

	Miner::Miner() 
	: m_io_context{std::make_shared<::asio::io_context>()}
	, m_signals{std::make_shared<::asio::signal_set>(*m_io_context)}
	, m_logger{ spdlog::stdout_color_mt("logger") }
	, m_config{ m_logger }
	{
		m_logger->set_pattern("[%D %H:%M:%S.%e][%^%l%$] %v");

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

	bool Miner::check_config(std::string const& miner_config_file)
	{
		m_logger->info("Running config check for {}", miner_config_file);
		std::ifstream config(miner_config_file);
		if (!config.is_open())
		{
			m_logger->critical("Unable to read {}", miner_config_file);
			return false;
		}

		config::Validator validator{};
		auto result = validator.check(miner_config_file);
		result ? m_logger->info(validator.get_check_result()) : m_logger->error(validator.get_check_result());
		return result;
	}

	bool Miner::init(std::string const& miner_config_file)
	{
		// print header
		std::cout << "NexusMiner Version " << NexusMiner_VERSION_MAJOR << "." << NexusMiner_VERSION_MINOR << "\n" << std::endl;

		if (!m_config.read_config(miner_config_file))
		{
			return false;
		}

		// logger settings
		if (!m_config.get_logfile().empty())
		{
			// initialise a new logger
			spdlog::drop("logger");
			auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
			auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(m_config.get_logfile(), true);

			m_logger = std::make_shared<spdlog::logger>(spdlog::logger("logger", { console_sink, file_sink }));
			m_logger->set_pattern("[%D %H:%M:%S.%e][%^%l%$] %v");
			spdlog::set_default_logger(m_logger);
			spdlog::flush_on(spdlog::level::info);
		}

		m_logger->set_level(static_cast<spdlog::level::level_enum>(m_config.get_log_level()));

		// timer initialisation
		chrono::Timer_factory::Sptr timer_factory = std::make_shared<chrono::Timer_factory>(m_io_context);

		// network initialisation
		m_network_component = network::create_component(m_io_context);

		auto const local_endpoint = get_local_ip();
		m_worker_manager = std::make_unique<Worker_manager>(m_io_context, m_config, timer_factory, 
			m_network_component->get_socket_factory()->create_socket(local_endpoint));
		
		return true;
	}

	void Miner::run()
	{
		auto const ip_address = m_config.get_wallet_ip();
		auto const port = m_config.get_port();
		
		network::Endpoint wallet_endpoint{network::Transport_protocol::tcp, ip_address, port};
		if(wallet_endpoint.transport_protocol() == network::Transport_protocol::none)
		{
			// resolve dns name
			wallet_endpoint = resolve_dns(ip_address, port);
			if(wallet_endpoint.transport_protocol() == network::Transport_protocol::none)
			{
				m_logger->error("Failed to resolve DNS name: {}", ip_address);
				return;
			}
		}

		auto result = m_worker_manager->connect(wallet_endpoint);
		if (!result)
		{
			m_logger->error("Failed to initialise socket. Result: {}", result);
			return;
		}

		m_io_context->run();

	}

	network::Endpoint Miner::resolve_dns(std::string const& dns_name, std::uint16_t port)
	{
		::asio::ip::tcp::resolver resolver(*m_io_context);
		::asio::error_code ec;
		auto endpoint = resolver.resolve(dns_name, std::string{std::to_string(port)}, ec);
		if(ec) 
		{
			return network::Endpoint{};
		}

		return network::Endpoint{ *endpoint.begin()};
	}

	network::Endpoint Miner::get_local_ip()
	{
		std::string local_ip = m_config.get_local_ip();
		std::for_each(local_ip.begin(), local_ip.end(), [](char& c) { c = ::tolower(c); });

		if (local_ip == "auto")
		{
			try 
			{
				asio::error_code error;
				asio::ip::udp::resolver resolver(*m_io_context);
				auto results = resolver.resolve("google.com", "80", error);
				for (auto const& endpoint : results)
				{
					asio::ip::udp::socket socket(*m_io_context);
					socket.connect(endpoint);
					asio::ip::address addr = socket.local_endpoint().address();
					return network::Endpoint{ network::Transport_protocol::tcp, addr.to_string(), 0 };
				}

				m_logger->error("Failed to resolve address google.com. Fallback to 127.0.0.1.");
				return network::Endpoint{ network::Transport_protocol::tcp, "127.0.0.1", 0 };
			}
			catch (std::exception& e) 
			{
				m_logger->error("Failed to set local_ip with auto mode. Fallback to 127.0.0.1. Exception: {}", e.what());
				return network::Endpoint{ network::Transport_protocol::tcp, "127.0.0.1", 0};
			}
		}
		else
		{
			return network::Endpoint{ network::Transport_protocol::tcp, m_config.get_local_ip(), 0 };
		}
	}
}