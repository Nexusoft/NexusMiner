#ifndef NEXUSMINER_PROTOCOL_LANE_HPP
#define NEXUSMINER_PROTOCOL_LANE_HPP

#include <cstdint>
#include <cstdio>
#include <string>

namespace nexusminer {

/**
 * Protocol Lane: Determines packet framing and behavior based on connection port
 * 
 * STRICT PORT-LANE SEPARATION:
 * - Port 8323: Legacy lane (8-bit header, polling behavior)
 * - Port 9323 and others: Stateless lane (16-bit header, push behavior)
 * - NO fallback between lanes
 */
enum class ProtocolLane : uint8_t {
    UNKNOWN = 0,    // Lane not yet determined (error state)
    LEGACY = 1,     // Port 8323: 8-bit header, polling (GET_ROUND/GET_BLOCK)
    STATELESS = 2   // Port 9323+: 16-bit header, push (STATELESS_GET_BLOCK pushes)
};

// Port constants for lane determination
namespace ProtocolPorts {
    static constexpr uint16_t STATELESS_PORT     = 9323;
    static constexpr uint16_t LEGACY_PORT        = 8323;
    // Forward-reserved TLS mining ports — require the node to have PORT_SSL != 0 before use.
    // The LLL-TAO MiningServerFactory currently sets PORT_SSL = 0 (not yet implemented).
    // Note: 9326 is reserved for the P2P node; SSL mining ports follow the node's 9325/8325 convention.
    static constexpr uint16_t STATELESS_SSL_PORT = 9325;  // Reserved TLS mining (mirrors node PORT_SSL intent)
    static constexpr uint16_t LEGACY_SSL_PORT    = 8325;  // Reserved legacy TLS mining
}

/**
 * Determine protocol lane from remote port
 * @param port Remote server port
 * @return LEGACY if port == 8323, STATELESS otherwise
 */
inline ProtocolLane determine_lane_from_port(uint16_t port) {
    return (port == ProtocolPorts::LEGACY_PORT) ? ProtocolLane::LEGACY : ProtocolLane::STATELESS;
}

/**
 * Get human-readable lane name for logging
 */
inline const char* get_lane_name(ProtocolLane lane) {
    switch(lane) {
        case ProtocolLane::LEGACY: return "Legacy";
        case ProtocolLane::STATELESS: return "Stateless";
        default: return "Unknown";
    }
}

/**
 * Wire-format header size for a given lane.
 *   STATELESS: [uint16 opcode][uint32 length] = 6 bytes
 *   LEGACY:    [uint8  opcode][uint32 length] = 5 bytes
 *
 * Use this instead of inline ternaries so the framing constant lives in one place.
 */
inline constexpr std::size_t lane_header_size(ProtocolLane lane) noexcept {
    return (lane == ProtocolLane::STATELESS) ? 6u : 5u;
}

/**
 * Format a canonical 8-bit LLP opcode for diagnostic logging, respecting the
 * active lane's wire encoding.
 *
 *   STATELESS: "0xD085" (mirror-mapped 16-bit)
 *   LEGACY:    "0x85"   (raw 8-bit)
 *
 * Returns a short, fixed-width hex string suitable for log messages.
 * Eliminates scattered ternary expressions like:
 *     m_protocol_lane == ProtocolLane::STATELESS ? "stateless 0xD085" : "legacy 0x85"
 */
inline std::string format_lane_opcode(ProtocolLane lane, uint8_t legacy_opcode) {
    if (lane == ProtocolLane::STATELESS) {
        // Mirror-map to 16-bit stateless opcode (0xD000 | legacy_opcode)
        char buf[8];
        std::snprintf(buf, sizeof(buf), "0x%04X", 0xD000u | static_cast<unsigned>(legacy_opcode));
        return buf;
    }
    char buf[6];
    std::snprintf(buf, sizeof(buf), "0x%02X", static_cast<unsigned>(legacy_opcode));
    return buf;
}

} // namespace nexusminer

#endif // NEXUSMINER_PROTOCOL_LANE_HPP
