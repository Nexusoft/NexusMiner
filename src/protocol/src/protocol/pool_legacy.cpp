#include "protocol/pool_legacy.hpp"
#include "packet.hpp"
#include "network/connection.hpp"
#include "stats/stats_collector.hpp"
#include "spdlog/spdlog.h"

namespace nexusminer
{
namespace protocol
{

Pool_legacy::Pool_legacy(std::shared_ptr<spdlog::logger> logger, config::Pool config, std::shared_ptr<stats::Collector> stats_collector)
    : Pool_base{ std::move(logger), std::move(stats_collector) }
    , m_config{config}
{
}

network::Shared_payload Pool_legacy::login(Login_handler handler)
{
    m_login_handler = std::move(handler);

    std::vector<std::uint8_t> username_data{ m_config.m_username.begin(), m_config.m_username.end() };
    Packet packet{ Packet::LOGIN, std::make_shared<network::Payload>(username_data) };
    return packet.get_bytes();
}

network::Shared_payload Pool_legacy::submit_block(std::vector<std::uint8_t> const& block_data, std::uint64_t nonce)
{
    m_logger->info("Submitting Block...");

    Packet packet{ Packet::SUBMIT_BLOCK };
    packet.m_data = std::make_shared<std::vector<std::uint8_t>>(block_data);
    auto const nonce_data = uint2bytes64(nonce);
    packet.m_data->insert(packet.m_data->end(), nonce_data.begin(), nonce_data.end());
    packet.m_length = 72;

    return packet.get_bytes();
}

void Pool_legacy::process_messages(Packet packet, std::shared_ptr<network::Connection> connection)
{
    if (packet.m_header == Packet::LOGIN_SUCCESS)
    {
        m_logger->info("Login to Pool successful");
        if (m_login_handler)
        {
            m_login_handler(true);
        }
    }
    else if (packet.m_header == Packet::LOGIN_FAIL)
    {
        m_logger->error("Login to Pool not successful");
        if (m_login_handler)
        {
            m_login_handler(false);
        }
    }
    // blackpool and hashpool send periodically this message to set height
    else if (packet.m_header == Packet::BLOCK_DATA)
    {
        std::uint32_t nbits{ 0U };
        auto original_block = extract_nbits_from_block(std::move(packet.m_data), nbits);
        auto block = deserialize_block(std::move(original_block));
        if (block.nHeight > m_current_height)
        {
            m_logger->info("Nexus Network: New height {}", block.nHeight);
            m_current_height = block.nHeight;
        }

        if (m_set_block_handler)
        {
            m_set_block_handler(block, nbits);
        }
        else
        {
            m_logger->error("No Block handler set");
        }
    }
    else if (packet.m_header == Packet::ACCEPT)
    {
        stats::Global global_stats{};
        global_stats.m_accepted_shares = 1;
        m_stats_collector->update_global_stats(global_stats);
        m_logger->info("Share Accepted By Pool.");
        connection->transmit(get_work());
    }
    else if (packet.m_header == Packet::REJECT)
    {
        stats::Global global_stats{};
        global_stats.m_rejected_shares = 1;
        m_stats_collector->update_global_stats(global_stats);
        m_logger->warn("Share Rejected by Pool.");
        connection->transmit(get_work());
    }
    else if (packet.m_header == Packet::BLOCK)
    {
        stats::Global global_stats{};
        global_stats.m_accepted_shares = 1;
        m_stats_collector->update_global_stats(global_stats);
        m_logger->info("Share Accepted By Pool. Found Block!");
    }
}

}
}