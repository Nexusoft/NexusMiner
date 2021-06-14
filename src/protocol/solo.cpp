#include "protocol/solo.hpp"
#include "packet.hpp"

namespace nexusminer
{
namespace protocol
{

Solo::Solo(std::uint8_t channel)
: m_channel{channel}
, m_logger{spdlog::get("logger")}
{

}

network::Shared_payload Solo::login()
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

}
}