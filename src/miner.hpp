#ifndef NEXUSMINER_MINER_HPP
#define NEXUSMINER_MINER_HPP

#include <thread>
#include <vector>
#include <memory>
#include <mutex>

#include "config.hpp"
#include <spdlog/spdlog.h>
#include <asio/signal_set.hpp>

namespace asio {
	class io_context;
}
namespace nexusminer
{
namespace network { class Component; }

class Worker_manager;

class Miner
{
public:

	Miner();
	~Miner();

	bool init();
	void run();

private:

	std::shared_ptr<::asio::io_context> m_io_context;
	std::shared_ptr<::asio::signal_set> m_signals;
	std::unique_ptr<network::Component> m_network_component;
	std::shared_ptr<Worker_manager> m_worker_manager;
	std::shared_ptr<spdlog::logger> m_logger;

	Config m_config;
};

}


#endif