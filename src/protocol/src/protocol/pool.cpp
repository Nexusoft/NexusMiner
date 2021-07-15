#include "protocol/pool.hpp"
#include "packet.hpp"
#include "network/connection.hpp"
#include "stats/stats_collector.hpp"

namespace nexusminer
{
namespace protocol
{

Pool::Pool(std::shared_ptr<stats::Collector> stats_collector)
: m_logger{spdlog::get("logger")}
, m_set_block_handler{}
, m_login_handler{}
, m_current_height{0}
, m_stats_collector{ std::move(stats_collector) }
{
}

void Pool::reset()
{
   m_current_height = 0;
}

network::Shared_payload Pool::login(std::string const& account_name, Login_handler handler)
{
    m_login_handler = std::move(handler);
    Packet packet;
    packet.m_header = Packet::LOGIN;
    std::vector<std::uint8_t> username_data{account_name.begin(), account_name.end()};
    packet.m_length = username_data.size();
    packet.m_data = std::make_shared<std::vector<std::uint8_t>>(username_data);
    return packet.get_bytes();
}

network::Shared_payload Pool::get_work()
{
    m_logger->info("Nexus Network: New Block");

    // get new block from wallet
    Packet packet;
    packet.m_header = Packet::GET_BLOCK;

    return packet.get_bytes();    
}

network::Shared_payload Pool::submit_block(std::vector<std::uint8_t> const& block_data, 
        std::vector<std::uint8_t> const& nonce )
{
    m_logger->info("Submitting Block...");

    Packet PACKET;
    Packet packet;
    packet.m_header = Packet::SUBMIT_BLOCK;

    packet.m_data = std::make_shared<std::vector<std::uint8_t>>(block_data);

    packet.m_data->insert(packet.m_data->end(), nonce.begin(), nonce.end());
    packet.m_length = 72;  

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
    else if(packet.m_header == Packet::BLOCK_DATA)
    {
        std::uint32_t nbits{0U};
        auto original_block = extract_nbits_from_block(std::move(packet.m_data), nbits);
        auto block = deserialize_block(std::move(original_block));
        if (block.nHeight > m_current_height)
        {
            m_current_height = block.nHeight;
            if(m_set_block_handler)
            {
                m_set_block_handler(block, nbits);
            }
            else
            {
                m_logger->error("No Block handler set");
            }
        }
    }
    else if(packet.m_header == Packet::ACCEPT)
    {
        stats::Global global_stats{};
        global_stats.m_accepted_shares = 1;
        m_stats_collector->update_global_stats(global_stats);
        m_logger->info("Share Accepted By Pool.");
    }
    else if(packet.m_header == Packet::REJECT)
    {
        stats::Global global_stats{};
        global_stats.m_rejected_shares = 1;
        m_stats_collector->update_global_stats(global_stats);
        m_logger->warn("Share Rejected by Pool.");
    }
    else if (packet.m_header == Packet::BLOCK)
    {
        stats::Global global_stats{};
        global_stats.m_accepted_shares = 1;
        m_stats_collector->update_global_stats(global_stats);
        m_logger->info("Share Accepted By Pool. Found Block!");
    }
}

network::Shared_payload Pool::extract_nbits_from_block(network::Shared_payload data, std::uint32_t& nbits)
{
    nbits = bytes2uint(std::vector<unsigned char>(data->begin(), data->begin() + 4));
    return std::make_shared<network::Payload>(data->begin() + 4, data->end());
}

}
}