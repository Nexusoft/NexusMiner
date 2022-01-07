#ifndef NEXUSMINER_PROTOCOL_POOL_HPP
#define NEXUSMINER_PROTOCOL_POOL_HPP

#include "config/types.hpp"
#include "config/pool.hpp"
#include "protocol/pool_base.hpp"
#include <memory>

namespace spdlog { class logger; }
namespace nexusminer 
{
namespace network { class Connection; }
namespace stats { class Collector; }
namespace protocol
{

class Pool : public Pool_base 
{
public:

    Pool(std::shared_ptr<spdlog::logger> logger, config::Mining_mode mining_mode, config::Pool config, std::shared_ptr<stats::Collector> stats_collector);

    network::Shared_payload login(Login_handler handler) override;
	void process_messages(Packet packet, std::shared_ptr<network::Connection> connection) override;

private:

    double get_hashrate_from_workers();

    config::Mining_mode m_mining_mode;
    config::Pool m_config;
};

}
}
#endif