#ifndef NEXUSMINER_PROTOCOL_PACKET_BUILDER_HPP
#define NEXUSMINER_PROTOCOL_PACKET_BUILDER_HPP

#include "network/types.hpp"
#include "protocol_lane.hpp"
#include <cstdint>
#include <vector>

namespace nexusminer {
namespace protocol {

/**
 * @brief Central utility for building lane-aware outbound LLP packets
 *
 * Constructs correctly-framed outbound packets for both protocol lanes:
 *   - Legacy:    [uint8  opcode][uint32 BE length][payload]
 *   - Stateless: [uint16 BE (0xD000|opcode)][uint32 BE length][payload]
 *
 * This is the single place where outbound packet framing is decided.
 * All outbound sends in Solo / Timer_manager / Worker_manager MUST use
 * PacketBuilder instead of constructing Packet{static_cast<...>} directly.
 */
class PacketBuilder {
public:
    /**
     * @brief Build a header-only packet (no payload)
     *
     * @param lane         ProtocolLane::LEGACY or ProtocolLane::STATELESS
     * @param legacy_opcode  The canonical 8-bit LLP opcode (e.g. GET_BLOCK = 129)
     * @return Wire-encoded bytes, or empty shared_ptr on error
     */
    static network::Shared_payload build(ProtocolLane lane, uint8_t legacy_opcode);

    /**
     * @brief Build a packet with a payload
     *
     * @param lane           ProtocolLane::LEGACY or ProtocolLane::STATELESS
     * @param legacy_opcode  The canonical 8-bit LLP opcode
     * @param payload        Bytes to append after the length field
     * @return Wire-encoded bytes, or empty shared_ptr on error
     */
    static network::Shared_payload build(ProtocolLane lane, uint8_t legacy_opcode,
                                         const std::vector<uint8_t>& payload);
};

} // namespace protocol
} // namespace nexusminer

#endif // NEXUSMINER_PROTOCOL_PACKET_BUILDER_HPP
