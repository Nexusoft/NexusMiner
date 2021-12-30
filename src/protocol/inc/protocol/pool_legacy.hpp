#ifndef NEXUSMINER_PROTOCOL_POOL_LEGACY_HPP
#define NEXUSMINER_PROTOCOL_POOL_LEGACY_HPP

#include "protocol/pool_base.hpp"
#include <memory>

namespace spdlog { class logger; }
namespace nexusminer {
namespace network { class Connection; }
namespace stats { class Collector; }
namespace protocol
{

class Pool_legacy : public Pool_base 
{
public:

    Pool_legacy(std::shared_ptr<spdlog::logger> logger, std::shared_ptr<stats::Collector> stats_collector);

    void process_messages(Packet packet, std::shared_ptr<network::Connection> connection) override;
};

}
}
#endif