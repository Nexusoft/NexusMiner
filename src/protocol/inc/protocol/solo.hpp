#ifndef NEXUSMINER_PROTOCOL_SOLO_HPP
#define NEXUSMINER_PROTOCOL_SOLO_HPP

#include "protocol/protocol.hpp"
#include "spdlog/spdlog.h"

namespace nexusminer {
namespace network { class Connection; }
namespace protocol
{

class Solo : public Protocol {
public:

    Solo(std::uint8_t channel);

    void reset() override;
    network::Shared_payload login(std::string const& account_name, Login_handler handler) override;
    network::Shared_payload get_work() override;
    network::Shared_payload submit_block(std::vector<std::uint8_t> const& block_data, 
        std::vector<std::uint8_t> const& nonce ) override;
    void set_block_handler(Set_block_handler handler) override { m_set_block_handler = std::move(handler); }

    void process_messages(Packet packet, std::shared_ptr<network::Connection> connection) override;

private:

    std::uint8_t m_channel;
    std::shared_ptr<spdlog::logger> m_logger;
    std::uint32_t m_current_height;
    Set_block_handler m_set_block_handler;
};

}
}
#endif