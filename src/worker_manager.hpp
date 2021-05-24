#ifndef NEXUSMINER_WORKER_MANAGER_HPP
#define NEXUSMINER_WORKER_MANAGER_HPP

#include "network/connection.hpp"
#include "network/socket.hpp"
#include "network/types.hpp"
#include <spdlog/spdlog.h>

#include <memory>

namespace LLP { class CBlock; }

namespace nexusminer 
{

class Config;
class Worker;

class Worker_manager : public std::enable_shared_from_this<Worker_manager>
{
public:

    Worker_manager(Config& config, network::Socket::Sptr socket);

    bool connect(network::Endpoint const& wallet_endpoint);
    void add_worker(std::unique_ptr<Worker> worker);

private:

    void parse_config();
	void process_data(network::Shared_payload&& receive_buffer);	// handle network messages

    LLP::CBlock deserialize_block(network::Shared_payload data);

    Config& m_config;
	network::Socket::Sptr m_socket;
	network::Connection::Sptr m_connection;
    std::shared_ptr<spdlog::logger> m_logger;

    std::vector<std::unique_ptr<Worker>> m_workers;

    std::uint32_t m_current_height;
};
}

#endif
