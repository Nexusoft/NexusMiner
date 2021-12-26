#include "protocol/solo.hpp"
#include "packet.hpp"
#include "network/connection.hpp"
#include "stats/stats_collector.hpp"

namespace nexusminer
{
namespace protocol
{

Solo::Solo(std::uint8_t channel, std::shared_ptr<stats::Collector> stats_collector)
: m_channel{channel}
, m_logger{spdlog::get("logger")}
, m_current_height{0}
, m_set_block_handler{}
, m_stats_collector{std::move(stats_collector)}
{
}

void Solo::reset()
{
    m_current_height = 0;
}

network::Shared_payload Solo::login(std::string const& account_name, Login_handler handler)
{
    Packet packet{ Packet::SET_CHANNEL, std::make_shared<network::Payload>(uint2bytes(m_channel)) };
    // call the login handler here because for solo mining this is always a "success"
    handler(true);
    return packet.get_bytes();
}

network::Shared_payload Solo::get_work()
{
    m_logger->info("Get new block");

    // get new block from wallet
    Packet packet{ Packet::GET_BLOCK };
    return packet.get_bytes();     
}

network::Shared_payload Solo::submit_block(std::vector<std::uint8_t> const& block_data, 
        std::vector<std::uint8_t> const& nonce )
{
    m_logger->info("Submitting Block...");

    Packet packet{ Packet::SUBMIT_BLOCK };
    packet.m_data = std::make_shared<network::Payload>(block_data);
    packet.m_data->insert(packet.m_data->end(), nonce.begin(), nonce.end());
    packet.m_length = 72;  

    return packet.get_bytes();  
}

void Solo::process_messages(Packet packet, std::shared_ptr<network::Connection> connection)  
{
    if (packet.m_header == Packet::BLOCK_HEIGHT)
    {
        auto const height = bytes2uint(*packet.m_data);
        if (height > m_current_height)
        {
            m_logger->info("Nexus Network: New height {}", height);
            m_current_height = height;
            connection->transmit(get_work());          
        }			
    }
    // Block from wallet received
    else if(packet.m_header == Packet::BLOCK_DATA)
    {
        auto block = deserialize_block(std::move(packet.m_data));
        if (block.nHeight == m_current_height)
        {
            if(m_set_block_handler)
            {
                m_set_block_handler(block, 0);
            }
            else
            {
                m_logger->error("No Block handler set");
            }
        }
        else
        {
            m_logger->warn("Block Obsolete Height = {} current_height = {}, Skipping over.", block.nHeight, m_current_height);
            connection->transmit(get_work());
        }
    }
    else if(packet.m_header == Packet::ACCEPT)
    {
        stats::Global global_stats{};
        global_stats.m_accepted_blocks = 1;
        m_stats_collector->update_global_stats(global_stats);
        m_logger->info("Block Accepted By Nexus Network.");
    }
    else if(packet.m_header == Packet::REJECT)
    {
        stats::Global global_stats{};
        global_stats.m_rejected_blocks = 1;
        m_stats_collector->update_global_stats(global_stats);
        m_logger->warn("Block Rejected by Nexus Network.");
        connection->transmit(get_work());
    }
    else
    {
        m_logger->debug("Invalid header received.");
    } 
}

}
}