#ifndef NEXUSMINER_WORKER_MANAGER_HPP
#define NEXUSMINER_WORKER_MANAGER_HPP

#include "network/connection.hpp"
#include "network/socket.hpp"
#include "network/types.hpp"
#include <spdlog/spdlog.h>
#include "chrono/timer_factory.hpp"
#include "chrono/timer.hpp"

#include <memory>

namespace LLP { class CBlock; }

namespace nexusminer 
{

class Config;
class Worker;

class Worker_manager : public std::enable_shared_from_this<Worker_manager>
{
public:

    Worker_manager(Config& config, chrono::Timer_factory::Sptr timer_factory, network::Socket::Sptr socket);

    bool connect(network::Endpoint const& wallet_endpoint);
    void add_worker(std::shared_ptr<Worker> worker);

private:

    void parse_config();
	void process_data(network::Shared_payload&& receive_buffer);	// handle network messages

    LLP::CBlock deserialize_block(network::Shared_payload data);

    chrono::Timer::Handler connection_retry_handler(network::Endpoint const& wallet_endpoint);

    Config& m_config;
	network::Socket::Sptr m_socket;
	network::Connection::Sptr m_connection;
    std::shared_ptr<spdlog::logger> m_logger;
    chrono::Timer_factory::Sptr m_timer_factory;
    chrono::Timer::Uptr m_connection_retry_timer;

    std::vector<std::shared_ptr<Worker>> m_workers;

    std::uint32_t m_current_height;
};
}

#endif
