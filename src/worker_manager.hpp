#ifndef NEXUSMINER_WORKER_MANAGER_HPP
#define NEXUSMINER_WORKER_MANAGER_HPP

#include "network/connection.hpp"
#include "network/socket.hpp"
#include "network/types.hpp"
#include <spdlog/spdlog.h>
#include "chrono/timer_factory.hpp"
#include "timer_manager.hpp"

#include <memory>

namespace LLP { class CBlock; }
namespace asio { class io_context; }

namespace nexusminer 
{

class Config;
class Worker;

class Worker_manager : public std::enable_shared_from_this<Worker_manager>
{
public:

    Worker_manager(std::shared_ptr<asio::io_context> io_context, Config& config, 
        chrono::Timer_factory::Sptr timer_factory, network::Socket::Sptr socket);

    bool connect(network::Endpoint const& wallet_endpoint);

    // stop the component and destroy all workers
    void stop();

private:

	void process_data(network::Shared_payload&& receive_buffer);	// handle network messages

    LLP::CBlock deserialize_block(network::Shared_payload data);

    void get_block();

    void create_workers();

	std::shared_ptr<::asio::io_context> m_io_context;
    Config& m_config;
	network::Socket::Sptr m_socket;
	network::Connection::Sptr m_connection;
    std::shared_ptr<spdlog::logger> m_logger;
    Timer_manager m_timer_manager;

    std::vector<std::shared_ptr<Worker>> m_workers;

    std::uint32_t m_current_height;
};
}

#endif
