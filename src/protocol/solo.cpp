#include "protocol/solo.hpp"
#include "packet.hpp"
#include "../network/connection.hpp"

namespace nexusminer
{
namespace protocol
{

Solo::Solo(std::uint8_t channel)
: m_channel{channel}
, m_logger{spdlog::get("logger")}
, m_current_height{0}
, m_set_block_handler{}
{
}

void Solo::reset()
{
    m_current_height = 0;
}

network::Shared_payload Solo::login(std::string account_name)
{
    Packet packet;
    packet.m_header = Packet::SET_CHANNEL;
    packet.m_length = 4;
    packet.m_data = std::make_shared<std::vector<std::uint8_t>>(uint2bytes(m_channel));
    return packet.get_bytes();
}

network::Shared_payload Solo::get_work()
{
    m_logger->info("Nexus Network: New Block");

    // get new block from wallet
    Packet packet;
    packet.m_header = Packet::GET_BLOCK;

    return packet.get_bytes();     
}

network::Shared_payload Solo::submit_block(std::vector<std::uint8_t> const& block_data, 
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

void Solo::process_messages(Packet packet, std::shared_ptr<network::Connection> connection)  
{
    if (packet.m_header == Packet::BLOCK_HEIGHT)
    {
        auto const height = bytes2uint(*packet.m_data);
        //m_logger->debug("Height Received {}", height);

        if (height > m_current_height)
        {
            m_current_height = height;
            connection->transmit(get_work());          
        }			
    }
    // Block from wallet received
    else if(packet.m_header == Packet::BLOCK_DATA)
    {
        auto block = deserialize_block(packet.m_data);
        if (block.nHeight == m_current_height)
        {
            if(m_set_block_handler)
            {
                m_set_block_handler(block);
            }
            else
            {
                m_logger->error("No Block handler set");
            }
        }
        else
        {
            m_logger->warn("Block Obsolete Height = {}, Skipping over.", block.nHeight);
        }
    }

    else
    {
        m_logger->error("Invalid header received.");
    } 
}

}
}