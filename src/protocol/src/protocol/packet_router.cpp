#include "protocol/packet_router.hpp"
#include "LLP/miner_opcodes.hpp"

namespace nexusminer {
namespace protocol {

void PacketRouter::register_handler(uint16_t legacy_opcode, Handler handler)
{
    m_handlers[legacy_opcode] = std::move(handler);
}

void PacketRouter::register_raw_handler(uint16_t raw_opcode, Handler handler)
{
    m_raw_handlers[raw_opcode] = std::move(handler);
}

bool PacketRouter::dispatch(Packet const& packet, const std::shared_ptr<network::Connection>& connection) const
{
    // 1. Try raw handler first (for opcodes with no legacy mirror)
    {
        auto it = m_raw_handlers.find(packet.m_header);
        if (it != m_raw_handlers.end()) {
            it->second(packet, connection);
            return true;
        }
    }

    // 2. Canonicalize: convert uint16_t stateless opcode to its legacy 8-bit mirror
    uint16_t canonical = packet.m_header;
    if (packet.m_is_uint16_opcode) {
        if (!LLP::IsStatelessOpcode(static_cast<uint16_t>(packet.m_header))) {
            return false;
        }
        canonical = LLP::UnmirrorOpcode(static_cast<uint16_t>(packet.m_header));
    }

    auto it = m_handlers.find(canonical);
    if (it != m_handlers.end()) {
        it->second(packet, connection);
        return true;
    }

    return false;
}

std::size_t PacketRouter::handler_count() const
{
    return m_handlers.size() + m_raw_handlers.size();
}

}  // namespace protocol
}  // namespace nexusminer
