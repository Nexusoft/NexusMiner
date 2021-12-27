#ifndef NEXUSMINER_PROTOCOL_POOL_HPP
#define NEXUSMINER_PROTOCOL_POOL_HPP

#include "config/types.hpp"
#include "protocol/protocol.hpp"
#include "spdlog/spdlog.h"
#include <memory>

namespace nexusminer {
namespace network { class Connection; }
namespace stats { class Collector; }
namespace protocol
{

class Pool : public Protocol {
public:

	Pool(config::Mining_mode mining_mode, std::shared_ptr<stats::Collector> stats_collector);

    void reset() override;
    network::Shared_payload login(std::string const& account_name, Login_handler handler) override;
    network::Shared_payload get_work() override;
    network::Shared_payload submit_block(std::vector<std::uint8_t> const& block_data, 
        std::vector<std::uint8_t> const& nonce ) override;
    void set_block_handler(Set_block_handler handler) override { m_set_block_handler = std::move(handler); }

	void process_messages(Packet packet, std::shared_ptr<network::Connection> connection) override;

private:

    // out_param nbits. Returns data without the nbits from pool.
    network::Shared_payload extract_nbits_from_block(network::Shared_payload data, std::uint32_t& nbits);

    double get_hashrate_from_workers();

    std::shared_ptr<spdlog::logger> m_logger;
    config::Mining_mode m_mining_mode;
    Set_block_handler m_set_block_handler;
    Login_handler m_login_handler;
    std::uint32_t m_current_height;
    std::shared_ptr<stats::Collector> m_stats_collector;
};

}
}
#endif