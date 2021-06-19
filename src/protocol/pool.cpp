#include "protocol/pool.hpp"
#include "packet.hpp"
#include "../network/connection.hpp"


namespace nexusminer
{
namespace protocol
{

Pool::Pool()
: m_logger{spdlog::get("logger")}
, m_set_block_handler{}
, m_login_handler{}
{
}

void Pool::reset()
{

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
    if(packet.m_header == Packet::LOGIN_SKMINER_SUCCESS)
    {
        m_logger->info("Login to Pool successful");
        if(m_login_handler)
        {
            m_login_handler(true);
        }
    }
    else if(packet.m_header == Packet::LOGIN_SKMINER_FAIL)
    {
        m_logger->error("Login to Pool not successful");
        if(m_login_handler)
        {
            m_login_handler(false);
        }
    }
}

}
}