#ifndef NEXUSMINER_WORKER_MANAGER_HPP
#define NEXUSMINER_WORKER_MANAGER_HPP

#include "network/connection.hpp"
#include "network/socket.hpp"
#include <spdlog/spdlog.h>

#include <memory>

namespace nexusminer 
{

class Config;
class Worker;

class Worker_manager : public std::enable_shared_from_this<Worker_manager>
{
public:

    Worker_manager(Config& config, network::Socket::Sptr socket);

    bool connect(network::Endpoint const& wallet_endpoint);


private:

    void parse_config();
	void process_data(network::Shared_payload&& receive_buffer);	// handle network messages

    Config& m_config;
	network::Socket::Sptr m_socket;
	network::Connection::Sptr m_connection;
    std::shared_ptr<spdlog::logger> m_logger;

    std::vector<std::unique_ptr<Worker>> m_workers;
};
}

#endif
