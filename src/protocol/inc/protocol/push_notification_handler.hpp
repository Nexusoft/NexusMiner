#ifndef NEXUSMINER_PROTOCOL_PUSH_NOTIFICATION_HANDLER_HPP
#define NEXUSMINER_PROTOCOL_PUSH_NOTIFICATION_HANDLER_HPP

#include "protocol/mining_template_interface.hpp"
#include "protocol/height_tracker.hpp"
#include "protocol_lane.hpp"
#include "packet.hpp"
#include "network/connection.hpp"
#include "spdlog/spdlog.h"
#include <functional>
#include <memory>

namespace nexusminer {
namespace protocol {

/**
 * @brief Unified handler for push notification opcodes
 *
 * Consolidates the identical logic previously duplicated across four handlers:
 *   Legacy:    PRIME_BLOCK_AVAILABLE (0xD9), HASH_BLOCK_AVAILABLE (0xDA)
 *   Stateless: STATELESS_PRIME_BLOCK_AVAILABLE (0xD0D9), STATELESS_HASH_BLOCK_AVAILABLE (0xD0DA)
 *
 * Three payload sizes are accepted:
 *   12-byte  (compact): unified_height + channel_height + difficulty
 *   140-byte (v1 extended, backward-compat): 12-byte header + 128-byte hashPrevBlock
 *   148-byte (v2 extended, new full-picture):
 *     [0..3]    unified_height
 *     [4..7]    channel_height   (own channel: prime or hash)
 *     [8..11]   difficulty (nBits)
 *     [12..15]  other_channel_height  (other PoW channel)
 *     [16..19]  stake_height
 *     [20..147] hashBestChain (uint1024_t, 128 bytes LE)
 */
class PushNotificationHandler {
public:
    PushNotificationHandler(
        std::shared_ptr<spdlog::logger> logger,
        const std::uint8_t& current_channel
    );

    /**
     * @brief Handle a block-available push notification
     *
     * Since the NODE now auto-sends BLOCK_DATA after every PUSH notification,
     * the miner no longer needs to send GET_BLOCK in response to PUSH.
     * This handler processes the push metadata (heights, tip anchors, liveness)
     * but does NOT trigger a GET_BLOCK request.
     *
     * @param packet            Received LLP packet (must have 12-byte, 140-byte, or 148-byte payload)
     * @param expected_channel  mining::CHANNEL_PRIME or mining::CHANNEL_HASH
     * @param lane              ProtocolLane::LEGACY or ProtocolLane::STATELESS
     * @param template_interface  Pointer to the active MiningTemplateInterface (may be nullptr)
     * @param height_tracker    Pointer to the active HeightTracker (may be nullptr; used for reads only)
     * @param update_height_fn  Callback invoked with (unified_height, channel_height, difficulty_nbits)
     *                          to update both HeightTracker and ClientChannelManager atomically.
     *                          If null, the update is skipped.
     * @return true  if the push was substantively processed (heights/state updated) —
     *               same-channel pushes or cross-channel tip advances
     *         false if only push liveness was recorded (cross-channel, no height change),
     *               or the payload was invalid
     */
    bool handle_push_notification(
        const Packet& packet,
        std::uint32_t expected_channel,
        ProtocolLane lane,
        MiningTemplateInterface* template_interface,
        HeightTracker* height_tracker,
        std::function<void(uint32_t, uint32_t, uint32_t)> update_height_fn
    );

private:
    std::shared_ptr<spdlog::logger> m_logger;
    const std::uint8_t& m_current_channel;

    static constexpr std::size_t PAYLOAD_SIZE_COMPACT       = 12;   // legacy push (12 bytes)
    static constexpr std::size_t PAYLOAD_SIZE_EXTENDED_V1   = 140;  // v1 extended (12 metadata + 128 hashPrevBlock)
    static constexpr std::size_t PAYLOAD_SIZE_EXTENDED       = 148;  // v2 extended full-picture (12 metadata + 8 cross-channel heights + 128 hashBestChain)
    static constexpr std::size_t PAYLOAD_SIZE = PAYLOAD_SIZE_COMPACT; // backward-compat alias
    static constexpr std::size_t UNIFIED_HEIGHT_OFFSET       = 0;
    static constexpr std::size_t CHANNEL_HEIGHT_OFFSET       = 4;
    static constexpr std::size_t DIFFICULTY_OFFSET           = 8;
    static constexpr std::size_t OTHER_CHANNEL_HEIGHT_OFFSET = 12;  // other PoW channel height in 148-byte payload
    static constexpr std::size_t STAKE_HEIGHT_OFFSET         = 16;  // stake channel height in 148-byte payload
    static constexpr std::size_t HASH_PREV_BLOCK_OFFSET_V1   = 12;  // hashBestChain offset in 140-byte payload
    static constexpr std::size_t HASH_PREV_BLOCK_OFFSET      = 20;  // hashBestChain offset in 148-byte payload
    static constexpr std::size_t HASH_BEST_CHAIN_SIZE_BYTES  = 128; // uint1024_t serialised size
    static constexpr std::size_t HASH_LOG_PREVIEW_BYTES      = 8;   // how many bytes to log for preview

    static const char* channel_name(std::uint32_t channel);
};

} // namespace protocol
} // namespace nexusminer

#endif // NEXUSMINER_PROTOCOL_PUSH_NOTIFICATION_HANDLER_HPP
