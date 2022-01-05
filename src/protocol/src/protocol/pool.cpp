#include "protocol/pool.hpp"
#include "packet.hpp"
#include "network/connection.hpp"
#include "stats/stats_collector.hpp"
#include "stats/types.hpp"
#include <spdlog/spdlog.h>
#include <json/json.hpp>

namespace nexusminer
{
namespace protocol
{

Pool::Pool(std::shared_ptr<spdlog::logger> logger, config::Mining_mode mining_mode, config::Pool config, std::shared_ptr<stats::Collector> stats_collector)
 : Pool_base{ std::move(logger), std::move(stats_collector) }
, m_mining_mode{ mining_mode }
, m_config{config}
{
}

network::Shared_payload Pool::login(Login_handler handler)
{
    m_login_handler = std::move(handler);

    nlohmann::json j;
    j["protocol_version"] = 1;
    j["username"] = m_config.m_username;
    j["display_name"] = m_config.m_display_name;
    auto j_string = j.dump();

    network::Payload login_data{ j_string.begin(), j_string.end() };
    Packet packet{ Packet::LOGIN, std::make_shared<network::Payload>(login_data) };
    return packet.get_bytes();
}

void Pool::process_messages(Packet packet, std::shared_ptr<network::Connection> connection)
{
    if(packet.m_header == Packet::LOGIN_SUCCESS)
    {
        m_logger->info("Login to Pool successful");
        if(m_login_handler)
        {
            m_login_handler(true);
        }
    }
    else if(packet.m_header == Packet::LOGIN_FAIL)
    {
        m_logger->error("Login to Pool not successful");
        if(m_login_handler)
        {
            m_login_handler(false);
        }
    }
    else if (packet.m_header == Packet::BLOCK_HEIGHT)
    {
        auto const height = bytes2uint(*packet.m_data);
        if (height > m_current_height)
        {
            m_logger->info("Nexus Network: New height {}", height);
            m_current_height = height;
            connection->transmit(get_work());
        }
    }
    else if(packet.m_header == Packet::BLOCK_DATA)
    {
        std::uint32_t nbits{0U};
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
    else if (packet.m_header == Packet::GET_HASHRATE)
    {
        auto const hashrate = get_hashrate_from_workers();
        Packet response{ Packet::HASHRATE, std::make_shared<network::Payload>(double2bytes(hashrate)) };
        connection->transmit(response.get_bytes());
    }
    else if(packet.m_header == Packet::ACCEPT)
    {
        stats::Global global_stats{};
        global_stats.m_accepted_shares = 1;
        m_stats_collector->update_global_stats(global_stats);
        m_logger->info("Share Accepted By Pool.");
        connection->transmit(get_work());
    }
    else if(packet.m_header == Packet::REJECT)
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
double Pool::get_hashrate_from_workers()
{
    double hashrate = 0.0;
    auto const workers = m_stats_collector->get_workers_stats();
    for (auto const& worker : workers)
    {
        if (m_mining_mode == config::Mining_mode::HASH)
        {
            auto& hash_stats = std::get<stats::Hash>(worker);
            hashrate += (hash_stats.m_hash_count / static_cast<double>(m_stats_collector->get_elapsed_time_seconds().count())) / 1.0e6;
        }
        else
        {
            auto& prime_stats = std::get<stats::Prime>(worker);
            //GISPS = Billion integers searched per second
            hashrate += (prime_stats.m_range_searched / (1.0e9 * static_cast<double>(m_stats_collector->get_elapsed_time_seconds().count())));
        }
    }

    return hashrate;
}

}
}