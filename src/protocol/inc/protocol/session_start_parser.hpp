/**
 * @file session_start_parser.hpp
 * @brief SESSION_START packet parser for NexusMiner
 *
 * Parses SESSION_START packets received from the LLL-TAO node after successful
 * MINER_AUTH_RESULT. This packet contains session parameters including success status,
 * session ID, timeout, and optional genesis hash.
 *
 * ACTUAL Wire Format (from LLL-TAO node stateless_miner_connection.cpp):
 *   [success (1 byte: 0x01)]
 *   [session_id (4 bytes, LE)]
 *   [timeout (4 bytes, LE)]
 *   [genesis_hash (32 bytes, optional)]
 *
 * Minimum size: 9 bytes (success + session_id + timeout)
 * Full format: 41 bytes (success + session_id + timeout + genesis_hash)
 *
 * NOTE: This format differs from earlier documentation. The node prepends a success
 * byte (0x01) and session_id field before the timeout. The session_key field mentioned
 * in earlier docs is not present in the actual wire format.
 */

#ifndef NEXUSMINER_SESSION_START_PARSER_HPP
#define NEXUSMINER_SESSION_START_PARSER_HPP

#include <cstdint>
#include <vector>
#include <optional>
#include "protocol/session_semantic_types.hpp"

namespace nexusminer {
namespace protocol {

/**
 * @brief Parsed SESSION_START packet data
 */
struct SessionStartData {
    uint8_t  success;                                ///< Success byte (0x01 = success)
    SessionId session_id;                            ///< Session ID from node (4 bytes LE)
    uint32_t timeout_seconds;                        ///< Session timeout in seconds (4 bytes LE)
    std::optional<std::vector<uint8_t>> genesis_hash; ///< Optional genesis hash (32 bytes)

    /// @brief Check if genesis hash is present
    bool has_genesis_hash() const { return genesis_hash.has_value(); }
};

/**
 * @brief Parse SESSION_START packet from wire format
 *
 * Validates packet structure and extracts success byte, session ID, timeout, and optional genesis hash.
 *
 * @param data Packet data (not including opcode/header, just payload)
 * @param length Packet data length
 * @return Parsed data on success, std::nullopt on parse error
 *
 * @note Returns nullopt if:
 *       - data is null
 *       - length < 9 (minimum: success + session_id + timeout)
 *       - length is 10-40 (invalid: partial genesis_hash)
 *       - length > 41 (warning: excess data ignored)
 */
inline std::optional<SessionStartData> parse_session_start(const uint8_t* data, size_t length) {
    // Validate minimum packet size: 1 (success) + 4 (session_id) + 4 (timeout) = 9 bytes
    if (!data || length < 9) {
        return std::nullopt;
    }

    SessionStartData result;

    // Parse success byte (offset 0)
    result.success = data[0];

    // Parse session_id (4 bytes, little-endian, offset 1-4)
    result.session_id = SessionId{static_cast<uint32_t>(data[1]) |
                       (static_cast<uint32_t>(data[2]) << 8) |
                       (static_cast<uint32_t>(data[3]) << 16) |
                       (static_cast<uint32_t>(data[4]) << 24)};

    // Parse timeout (4 bytes, little-endian, offset 5-8)
    result.timeout_seconds = static_cast<uint32_t>(data[5]) |
                            (static_cast<uint32_t>(data[6]) << 8) |
                            (static_cast<uint32_t>(data[7]) << 16) |
                            (static_cast<uint32_t>(data[8]) << 24);

    // Extract optional genesis hash (32 bytes at offset 9)
    if (length >= 41) {
        std::vector<uint8_t> genesis(data + 9, data + 41);
        result.genesis_hash = std::move(genesis);
    } else if (length > 9) {
        // Partial genesis data (invalid)
        return std::nullopt;
    }

    return result;
}

/**
 * @brief Parse SESSION_START packet from vector
 *
 * Convenience overload for std::vector<uint8_t>
 *
 * @param data Packet data vector
 * @return Parsed data on success, std::nullopt on parse error
 */
inline std::optional<SessionStartData> parse_session_start(const std::vector<uint8_t>& data) {
    return parse_session_start(data.data(), data.size());
}

/**
 * @brief Calculate keepalive interval in hours from session timeout
 *
 * Uses KEEPALIVE_SAFETY_DIVISOR (=2) to ensure 2 keepalive pings per timeout window.
 * Minimum keepalive interval is 1 hour.
 *
 * @param timeout_seconds Session timeout in seconds
 * @param safety_divisor Safety divisor (default: 2 for 2 pings per window)
 * @return Keepalive interval in hours (minimum 1)
 */
inline uint16_t calculate_keepalive_hours(uint32_t timeout_seconds, uint32_t safety_divisor = 2) {
    if (timeout_seconds == 0) {
        return 1;  // Minimum keepalive interval
    }
    uint32_t keepalive_hours = timeout_seconds / (safety_divisor * 3600u);
    return static_cast<uint16_t>(std::max(1u, keepalive_hours));
}

} // namespace protocol
} // namespace nexusminer

#endif // NEXUSMINER_SESSION_START_PARSER_HPP
