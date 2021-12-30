#include "protocol/pool_base.hpp"
#include "packet.hpp"
#include "network/connection.hpp"
#include "stats/stats_collector.hpp"
#include "stats/types.hpp"
#include "spdlog/spdlog.h"

namespace nexusminer
{
namespace protocol
{

Pool_base::Pool_base(std::shared_ptr<spdlog::logger> logger, std::shared_ptr<stats::Collector> stats_collector)
    : m_logger{ std::move(logger) }
    , m_set_block_handler{}
    , m_login_handler{}
    , m_current_height{ 0 }
    , m_stats_collector{ std::move(stats_collector) }
{
}

void Pool_base::reset()
{
    m_current_height = 0;
}

network::Shared_payload Pool_base::login(std::string const& account_name, Login_handler handler)
{
    m_login_handler = std::move(handler);

    std::vector<std::uint8_t> username_data{ account_name.begin(), account_name.end() };
    Packet packet{ Packet::LOGIN, std::make_shared<network::Payload>(username_data) };
    return packet.get_bytes();
}

network::Shared_payload Pool_base::get_work()
{
    m_logger->info("Get new block");

    // get new block from wallet
    Packet packet{ Packet::GET_BLOCK };
    return packet.get_bytes();
}

network::Shared_payload Pool_base::submit_block(std::vector<std::uint8_t> const& block_data,
    std::vector<std::uint8_t> const& nonce)
{
    m_logger->info("Submitting Block...");

    Packet packet{ Packet::SUBMIT_BLOCK };
    packet.m_data = std::make_shared<std::vector<std::uint8_t>>(block_data);
    packet.m_data->insert(packet.m_data->end(), nonce.begin(), nonce.end());
    packet.m_length = 72;

    return packet.get_bytes();
}

network::Shared_payload Pool_base::extract_nbits_from_block(network::Shared_payload data, std::uint32_t& nbits)
{
    nbits = bytes2uint(std::vector<unsigned char>(data->begin(), data->begin() + 4));
    return std::make_shared<network::Payload>(data->begin() + 4, data->end());
}

}
}