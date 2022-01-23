#ifndef NEXUSMINER_PROTOCOL_POOL_BASE_HPP
#define NEXUSMINER_PROTOCOL_POOL_BASE_HPP

#include "config/types.hpp"
#include "protocol/protocol.hpp"
#include <memory>

namespace spdlog { class logger; }
namespace nexusminer {
namespace network { class Connection; }
namespace stats { class Collector; }
namespace protocol
{

class Pool_base : public Protocol 
{
public:

    Pool_base(std::shared_ptr<spdlog::logger> logger, std::shared_ptr<stats::Collector> stats_collector);
    void reset() override;
    network::Shared_payload get_work() override;
    void set_block_handler(Set_block_handler handler) override { m_set_block_handler = std::move(handler); }

protected:

    // out_param nbits. Returns data without the nbits from pool.
    network::Shared_payload extract_nbits_from_block(network::Shared_payload data, std::uint32_t& nbits);

    std::shared_ptr<spdlog::logger> m_logger;
    Set_block_handler m_set_block_handler;
    Login_handler m_login_handler;
    std::uint32_t m_current_height;
    std::shared_ptr<stats::Collector> m_stats_collector;
};

}
}
#endif