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

		m_signals->async_wait([this](const asio::error_code& ec, int signal_number)
		{
			if (ec) return;
			m_logger->info("Shutting down NexusMiner (signal={})", signal_number);
			// Stop the watchdog first so a deliberate, graceful shutdown is
			// never mistaken for an io_context stall while we tear down.
			if (m_liveness_watchdog)
			{
				m_liveness_watchdog->stop();
			}
			// Stop the io_context FIRST so Miner::run()'s io_context->run() returns
			// immediately, no matter what recovery/re-auth/reconnect state the
			// Worker_manager is currently in (e.g. mid SESSION_RECOVERY, awaiting a
			// GET_BLOCK, or a queued reconnect timer). Previously, the heavier
			// enter_terminal_degraded_mode() teardown ran first on this same
			// io_context thread; if any recovery-path bookkeeping took a while (or a
			// future change made it block), CTRL+C would appear to hang rather than
			// exiting promptly. Stopping first guarantees the process can always
			// exit on signal, independent of recovery state.
			m_io_context->stop();
			if (m_worker_manager)
			{
				m_worker_manager->enter_terminal_degraded_mode(signal_number);
			}
		});
	}

	Miner::~Miner()
	{
		if (m_liveness_watchdog)
		{
			m_liveness_watchdog->stop();
		}
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
		std::cout << "NexusMiner Version " << NexusMiner_VERSION << "\n" << std::endl;

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

		// Independent liveness watchdog for the shared io_context thread (see
		// liveness_watchdog.hpp). Uses its own dedicated timer so a stall
		// elsewhere on the io_context cannot be masked by shared timer state.
		m_liveness_watchdog = std::make_unique<LivenessWatchdog>(
			timer_factory, m_logger);

		// network initialisation
		m_logger->debug("Initializing network component");
		m_network_component = network::create_component(m_io_context);
		
		m_logger->debug("Getting local IP address");
		auto const local_endpoint = get_local_ip();
		std::string local_addr;
		local_endpoint.address(local_addr);
		m_logger->debug("Local endpoint: {}:{}", local_addr, local_endpoint.port());
		
		m_logger->debug("Creating worker manager");
		m_worker_manager = std::make_shared<Worker_manager>(m_io_context, m_config, timer_factory, 
			m_network_component->get_socket_factory()->create_socket(local_endpoint));
		
		return true;
	}

	void Miner::run()
	{
		auto const ip_address = m_config.get_wallet_ip();
		auto const port = m_config.get_port();
		
		m_logger->info("=== NexusMiner Port Configuration ===");
		m_logger->info("Configured wallet IP: {}", ip_address);
		m_logger->info("Configured LLP port: {} (from miner.conf)", port);
		m_logger->info("Note: This is the miningport in LLL-TAO's nexus.conf");
		m_logger->info("=====================================");
		
		network::Endpoint wallet_endpoint{network::Transport_protocol::tcp, ip_address, port};
		if(wallet_endpoint.transport_protocol() == network::Transport_protocol::none)
		{
			// resolve dns name
			m_logger->info("Resolving DNS name: {}", ip_address);
			wallet_endpoint = resolve_dns(ip_address, port);
			if(wallet_endpoint.transport_protocol() == network::Transport_protocol::none)
			{
				m_logger->error("Failed to resolve DNS name: {}", ip_address);
				return;
			}
			m_logger->info("DNS resolved to: {}", wallet_endpoint.to_string());
		}

		if (!m_worker_manager)
		{
			m_logger->error("Worker manager is not initialized");
			return;
		}
		
		auto result = m_worker_manager->connect(wallet_endpoint);
		if (!result)
		{
			m_logger->error("Failed to initialise socket. Result: {}", result);
			return;
		}

		// NodeSession now stays on the configured primary node/lane.
		// Optional secondary-node failover is handled by Worker_manager retry logic.
		if (m_config.get_enable_sim_link())
		{
			m_logger->info("[SIM Link] ENABLED — reconnect/re-auth stays on the configured primary lane");
		}
		else
		{
			m_logger->info("[SIM Link] DISABLED (single lane-scoped node session)");
		}

		if (m_liveness_watchdog)
		{
			m_liveness_watchdog->start();
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
			m_logger->error("DNS resolution failed for '{}:{}': {}", dns_name, port, ec.message());
			return network::Endpoint{};
		}

		return network::Endpoint{ *endpoint.begin()};
	}

	network::Endpoint Miner::get_local_ip()
	{
		std::string local_ip = m_config.get_local_ip();
		std::for_each(local_ip.begin(), local_ip.end(),
			[](char& c) { c = static_cast<char>(::tolower(static_cast<unsigned char>(c))); });

		if (local_ip == "auto")
		{
			// Short-circuit for localhost mining — no DNS lookup needed
			if (m_config.is_localhost_mining())
			{
				m_logger->info("Local IP: localhost mining detected — binding to 127.0.0.1 (no DNS lookup needed)");
				return network::Endpoint{ network::Transport_protocol::tcp, "127.0.0.1", 0 };
			}

			// Bind to the unspecified address (0.0.0.0) and let the OS choose
			// the source IP at connect() time based on the routing table.  This
			// is the canonical way to "auto" pick a local IP without depending
			// on an external DNS lookup, which is brittle in offline / private
			// environments.
			m_logger->debug("Local IP: 'auto' — binding to 0.0.0.0 and letting the OS pick the source address");
			return network::Endpoint{ network::Transport_protocol::tcp, "0.0.0.0", 0 };
		}
		else
		{
			m_logger->debug("Using configured local IP: {}", m_config.get_local_ip());
			return network::Endpoint{ network::Transport_protocol::tcp, m_config.get_local_ip(), 0 };
		}
	}
}
