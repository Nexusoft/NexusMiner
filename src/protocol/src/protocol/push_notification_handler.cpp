#include "protocol/push_notification_handler.hpp"
#include "LLP/utils.hpp"
#include "mining/client_block.h"

namespace nexusminer {
namespace protocol {

PushNotificationHandler::PushNotificationHandler(
    std::shared_ptr<spdlog::logger> logger,
    const std::uint8_t& current_channel)
: m_logger{std::move(logger)}
, m_current_channel{current_channel}
{
}

const char* PushNotificationHandler::channel_name(std::uint32_t channel)
{
    return (channel == mining::CHANNEL_PRIME) ? "Prime"
         : (channel == mining::CHANNEL_HASH)  ? "Hash"
         : (channel == mining::CHANNEL_STAKE) ? "Stake"
         : "Unknown";
}

bool PushNotificationHandler::handle_push_notification(
    const Packet& packet,
    std::uint32_t expected_channel,
    ProtocolLane lane,
    MiningTemplateInterface* template_interface,
    HeightTracker* height_tracker,
    std::function<void(uint32_t, uint32_t, uint32_t)> update_height_fn)
{
    const char* ch_name = channel_name(expected_channel);

    bool work_requested = false;

    // Log reception with opcode
    if (lane == ProtocolLane::STATELESS) {
        m_logger->info("[Solo Push] ✉️  STATELESS_{}_BLOCK_AVAILABLE (0x{:04X}) received",
                       ch_name, packet.m_header);
    } else {
        m_logger->info("[Solo Push] ✉️  {}_BLOCK_AVAILABLE received", ch_name);
    }

    /* Validate payload (must be 12, 140, or 148 bytes) */
    const bool is_compact      = (packet.m_length == PAYLOAD_SIZE_COMPACT);
    const bool is_extended     = (packet.m_length == PAYLOAD_SIZE_EXTENDED);      // 148-byte full-picture (new)
    const bool is_extended_v1  = (packet.m_length == PAYLOAD_SIZE_EXTENDED_V1);   // 140-byte (old, backward-compat)

    if (!packet.m_data || (!is_compact && !is_extended && !is_extended_v1))
    {
        m_logger->error("[Solo Push] Invalid payload: {} bytes (expected {}, {}, or {})",
                       packet.m_length, PAYLOAD_SIZE_COMPACT, PAYLOAD_SIZE_EXTENDED_V1, PAYLOAD_SIZE_EXTENDED);
        return false;
    }

    /* Parse unified height early — needed before the channel check to detect
     * cross-channel tip advances and reset the GET_BLOCK dedup state.  A block
     * on ANY channel (Prime, Hash, or Stake) advances the canonical unified tip,
     * which means all miners need a fresh template with the new hashPrevBlock. */
    uint32_t notification_unified_height = bytes2uint(*packet.m_data, UNIFIED_HEIGHT_OFFSET);

    /* Parse channel_height and difficulty early — needed in both same-channel and
     * cross-channel paths.  Bug 1 fix: cross-channel path must call update_height_fn
     * so HeightTracker/ClientChannelManager are updated on every push, not just
     * same-channel ones.  Without this, repeated cross-channel pushes at the same
     * unified height silently skip state updates due to stale snapshot. */
    uint32_t notification_channel_height = bytes2uint(*packet.m_data, CHANNEL_HEIGHT_OFFSET);
    uint32_t notification_difficulty     = bytes2uint(*packet.m_data, DIFFICULTY_OFFSET);

    /* Validate channel — since node now broadcasts BOTH channels on every push update,
     * receiving a push for the non-subscribed channel is expected.  Record push liveness
     * so degraded-mode recovery knows the node is actively communicating.
     *
     * IMPORTANT: Every unified height advance changes the canonical tip anchor
     * (hashPrevBlock).  Cross-channel blocks (Hash/Stake for a Prime miner) advance
     * unified height just like same-channel blocks, so we MUST request a fresh
     * template to embed the correct hashPrevBlock in the next mined block. */
    if (m_current_channel != expected_channel)
    {
        if (height_tracker) {
            height_tracker->OnPushLiveness();
        }

        // If the unified tip advanced on another channel, request a fresh template.
        // Every unified height movement changes hashPrevBlock — the mined block
        // template must contain the updated hash to be valid.
        bool tip_advanced = false;
        if (notification_unified_height > 0 && height_tracker) {
            auto snap = height_tracker->GetSnapshot();
            if (notification_unified_height > snap.unified_height) {
                tip_advanced = true;
                m_logger->info("[Solo Push] ⚡ Cross-channel tip advance: unified {} → {} — "
                               "node will auto-send fresh template (hashPrevBlock changed)",
                               snap.unified_height, notification_unified_height);

                // Bug 1 fix: update height state so next cross-channel push at the same
                // unified height does not see a stale snapshot and miss state updates.
                if (update_height_fn) {
                    update_height_fn(notification_unified_height,
                                     notification_channel_height,
                                     notification_difficulty);
                }

                // Bug 2 fix: store hashBestChain from 148-byte payloads so tip-anchor
                // tracking works correctly for cross-channel extended pushes.
                if (is_extended && height_tracker)
                {
                    std::size_t hash_end = HASH_PREV_BLOCK_OFFSET + HASH_BEST_CHAIN_SIZE_BYTES;
                    if (packet.m_data->size() >= hash_end)
                    {
                        std::vector<uint8_t> hash_bytes(
                            packet.m_data->begin() + HASH_PREV_BLOCK_OFFSET,
                            packet.m_data->begin() + hash_end);
                        uint1024_t cross_hash{};
                        cross_hash.SetBytes(hash_bytes);
                        height_tracker->UpdatePushTipAnchor(cross_hash);
                    }
                }

                // NODE auto-sends BLOCK_DATA after PUSH — no GET_BLOCK request needed.
                // Heights are updated and tip anchor recorded; the miner will receive
                // the fresh template automatically from the node.
                work_requested = true;  // heights/state were substantively updated
            }
        }

        // Bug 3 fix: log at info only when tip advanced; liveness-only is debug.
        if (tip_advanced) {
            // tip-advance log already emitted above
        } else {
            m_logger->debug("[Solo Push] ℹ️  {} push received on {} lane (mining {} channel) — refreshed push liveness",
                           ch_name,
                           (lane == ProtocolLane::STATELESS) ? "stateless" : "legacy",
                           (m_current_channel == mining::CHANNEL_PRIME) ? "Prime" :
                           (m_current_channel == mining::CHANNEL_HASH)  ? "Hash"  : "Unknown");
        }
        return work_requested;
    }

    m_logger->info("[Solo Push] {} payload received ({} bytes)",
                   is_extended ? "Extended v2 full-picture" : (is_extended_v1 ? "Extended v1 stateless" : "Compact legacy"),
                   packet.m_length);

    /* Parse notification (big-endian) — already parsed before the channel check;
     * alias for readability in the same-channel path. */
    const uint32_t unified_height  = notification_unified_height;
    const uint32_t channel_height  = notification_channel_height;
    const uint32_t difficulty      = notification_difficulty;

    if (lane == ProtocolLane::STATELESS) {
        m_logger->info("[Solo Push]   Unified: {}, {}: {}, Diff: 0x{:08x}",
                       unified_height, ch_name, channel_height, difficulty);
    } else {
        m_logger->info("[Solo Push]   Unified height: {}", unified_height);
        m_logger->info("[Solo Push]   {} height: {}", ch_name, channel_height);
        m_logger->info("[Solo Push]   Difficulty: 0x{:08x}", difficulty);
    }

    // Parse cross-channel heights and hashBestChain for 148-byte full-picture payload
    uint1024_t notification_hash_prev_block{0};
    bool has_hash_prev_block = false;

    if (is_extended)
    {
        // bytes [12-15]: other PoW channel height
        // bytes [16-19]: stake channel height
        uint32_t other_channel_height = bytes2uint(*packet.m_data, OTHER_CHANNEL_HEIGHT_OFFSET);
        uint32_t stake_height         = bytes2uint(*packet.m_data, STAKE_HEIGHT_OFFSET);

        // Derive prime/hash from own channel + other channel
        uint32_t prime_h = (expected_channel == mining::CHANNEL_PRIME) ? channel_height : other_channel_height;
        uint32_t hash_h  = (expected_channel == mining::CHANNEL_HASH)  ? channel_height : other_channel_height;

        m_logger->info("[Solo Push]   Full height picture: prime={} hash={} stake={}",
                       prime_h, hash_h, stake_height);

        if (height_tracker)
        {
            height_tracker->OnPushFullPicture(unified_height, prime_h, hash_h, stake_height);
        }

        // bytes [20-147]: hashBestChain (128 bytes, little-endian uint1024_t)
        std::size_t hash_end = HASH_PREV_BLOCK_OFFSET + HASH_BEST_CHAIN_SIZE_BYTES;
        if (packet.m_data->size() >= hash_end)
        {
            std::vector<uint8_t> hash_bytes(packet.m_data->begin() + HASH_PREV_BLOCK_OFFSET,
                                            packet.m_data->begin() + hash_end);
            notification_hash_prev_block.SetBytes(hash_bytes);
            has_hash_prev_block = true;

            // Log first HASH_LOG_PREVIEW_BYTES bytes as hex for cross-reference with node Guard 2 logs
            std::string prev_hash_hex;
            for (std::size_t i = HASH_PREV_BLOCK_OFFSET; i < std::min(packet.m_data->size(), HASH_PREV_BLOCK_OFFSET + HASH_LOG_PREVIEW_BYTES); ++i) {
                char buf[3];
                snprintf(buf, sizeof(buf), "%02x", (*packet.m_data)[i]);
                prev_hash_hex += buf;
            }
            m_logger->info("[Solo Push]   hashBestChain (first {} bytes): {}... (128 bytes, can pre-validate staleness)",
                           HASH_LOG_PREVIEW_BYTES, prev_hash_hex);
        }
    }
    else if (is_extended_v1)
    {
        // bytes [12-139]: hashPrevBlock (128 bytes, little-endian uint1024_t)
        // Extract the 128-byte hash for comparison with current template
        std::vector<uint8_t> hash_bytes(packet.m_data->begin() + HASH_PREV_BLOCK_OFFSET_V1,
                                        packet.m_data->begin() + HASH_PREV_BLOCK_OFFSET_V1 + HASH_BEST_CHAIN_SIZE_BYTES);
        notification_hash_prev_block.SetBytes(hash_bytes);
        has_hash_prev_block = true;

        // Log first HASH_LOG_PREVIEW_BYTES bytes as hex for cross-reference with node Guard 2 logs
        std::string prev_hash_hex;
        for (std::size_t i = HASH_PREV_BLOCK_OFFSET_V1; i < std::min(packet.m_data->size(), HASH_PREV_BLOCK_OFFSET_V1 + HASH_LOG_PREVIEW_BYTES); ++i) {
            char buf[3];
            snprintf(buf, sizeof(buf), "%02x", (*packet.m_data)[i]);
            prev_hash_hex += buf;
        }
        m_logger->info("[Solo Push]   hashPrevBlock (first {} bytes): {}... (128 bytes, can pre-validate staleness)",
                       HASH_LOG_PREVIEW_BYTES, prev_hash_hex);
    }

    /* Update heights via unified callback (updates HeightTracker + ClientChannelManager) */
    if (update_height_fn) {
        update_height_fn(unified_height, channel_height, difficulty);
    }
    // height_tracker is used only for reads (ExplainMismatch, GetSnapshot for staleness).
    // Both update_height_fn and height_tracker* are expected to be non-null in production
    // (Solo always provides both); drift detection is advisory and safe to skip if null.
    if (height_tracker) {
        if (has_hash_prev_block) {
            height_tracker->UpdatePushTipAnchor(notification_hash_prev_block);
        }
        std::string drift_msg = height_tracker->ExplainMismatch();
        if (!drift_msg.empty()) {
            m_logger->info("{}", drift_msg);
        }
    }

    /* ═══════════════════════════════════════════════════════════════════════
     * TEMPLATE REFRESH — node auto-sends BLOCK_DATA after PUSH
     * ═══════════════════════════════════════════════════════════════════════
     * The NODE now auto-sends BLOCK_DATA after every PUSH notification.
     * The miner no longer needs to send a GET_BLOCK request in response.
     * Heights and state have been updated above; the fresh template will
     * arrive automatically from the node.
     * ═════════════════════════════════════════════════════════════════════ */
    if (template_interface && template_interface->has_valid_template())
    {
        auto snap = height_tracker ? height_tracker->GetSnapshot() : HeightTracker::Snapshot{};

        // ─── Channel staleness: [REMOVED] ─────────────────────────────────
        // is_template_stale() removed: structurally false under canonical-only
        // semantics (OnBlockDataReceived auto-advances target = channel + 1).
        // Channel staleness detection is now exclusively in PUSH, GET_ROUND,
        // and Health Monitor trigger paths.

        // ─── Same-height tip replacement (reorg at same channel height) ──────
        // A hash mismatch at the same channel height means the tip anchor was
        // replaced (same-height reorg).
        // The node auto-sends BLOCK_DATA after every PUSH; that fresh template
        // will atomically replace m_current_template inside read_template().
        // Mark it as replacement-pending (rather than discarding) so workers
        // keep mining the about-to-be-superseded template through the ~1-2s
        // round-trip instead of stalling into a HEALTH_NO_TEMPLATE recovery
        // epoch. If the replacement BLOCK_DATA never arrives, the worker
        // health monitor will time it out and fall back to the normal
        // discard + GET_BLOCK retry path.
        if (has_hash_prev_block)
        {
            auto const* tmpl = template_interface->get_current_template();
            if (tmpl &&
                snap.has_same_height_push_tip_replacement(tmpl->block.hashPrevBlock,
                                                          tmpl->nChannelHeight))
            {
                m_logger->info("[Solo Push] Same-height tip update — marking template "
                               "replacement-pending; workers continue mining until BLOCK_DATA arrives");
                template_interface->mark_replacement_pending("same_height_tip_update");
            }
        }

        // NODE auto-sends BLOCK_DATA — no GET_BLOCK request needed.
        m_logger->info("[Solo Push] {} PUSH processed — node will auto-send fresh template (unified tip moved → hashPrevBlock changed)",
                       ch_name);
        work_requested = true;
    }
    else
    {
        /* No template yet — node will auto-send block data */
        m_logger->info("[Solo Push] No template — node will auto-send {} block data", ch_name);
        work_requested = true;
    }

    return work_requested;
}

} // namespace protocol
} // namespace nexusminer
