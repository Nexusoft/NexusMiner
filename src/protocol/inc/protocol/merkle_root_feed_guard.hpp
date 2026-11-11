#ifndef NEXUSMINER_PROTOCOL_MERKLE_ROOT_FEED_GUARD_HPP
#define NEXUSMINER_PROTOCOL_MERKLE_ROOT_FEED_GUARD_HPP

/**
 * @file merkle_root_feed_guard.hpp
 * @brief Suppresses duplicate worker feeds when the same hashMerkleRoot
 *        arrives within a configurable time window, and also suppresses
 *        same-height feeds (regardless of merkle root) within a shorter
 *        window to prevent "Double Feed" restarts from mempool-variant
 *        templates.
 *
 * Architecture:
 *   When BLOCK_DATA is received and passes validation, the guard records
 *   the template's hashMerkleRoot, hashPrevBlock, height, and a
 *   steady_clock timestamp.  Two suppression tiers are checked:
 *
 *   1. **Same merkle-root suppression (2 s)**: identical (height, merkleRoot)
 *      pair within SUPPRESSION_WINDOW_SECONDS → suppress.
 *
 *   2. **Same-height suppression (500 ms)**: same height but different merkle
 *      root within HEIGHT_ONLY_SUPPRESSION_MS → suppress.  This catches
 *      mempool-variant duplicates where the node rebuilds the template with
 *      a different transaction set at the same chain tip.
 *
 *   A hashPrevBlock change at the same height (same-height reorg) bypasses
 *   both tiers so the miner always follows the correct chain tip.
 *
 *   This prevents workers from being restarted on redundant templates
 *   that would produce identical mining work, which wastes hash cycles.
 *
 * Thread safety:
 *   Single-strand (io_context thread) — no internal mutex.
 */

#include <chrono>
#include <cstdint>
#include "LLC/types/uint1024.h"

namespace nexusminer {
namespace protocol {

class MerkleRootFeedGuard {
public:
    /// Time window within which a duplicate hashMerkleRoot suppresses the feed.
    static constexpr int64_t SUPPRESSION_WINDOW_SECONDS = 2;

    /// Time window within which the same unified height (even with a different
    /// merkle root) suppresses the feed.  This catches mempool-variant duplicates.
    static constexpr int64_t HEIGHT_ONLY_SUPPRESSION_MS = 500;

    MerkleRootFeedGuard() = default;

    /**
     * @brief Check whether the feed should proceed, and record the state.
     *
     * Tier 1: If the same (height, hashMerkleRoot) pair was last fed within
     *         SUPPRESSION_WINDOW_SECONDS → suppress.
     * Tier 2: If the same height was last fed within HEIGHT_ONLY_SUPPRESSION_MS
     *         and hashPrevBlock is unchanged → suppress.
     *
     * A hashPrevBlock change (same-height reorg) always allows the feed.
     * A height change always allows the feed.
     *
     * @param hash_merkle_root  hashMerkleRoot from the incoming template
     * @param unified_height    Unified blockchain height (0 = ignore height key)
     * @param hash_prev_block   hashPrevBlock from the incoming template (zero = ignore)
     * @return true if the feed should proceed, false if suppressed
     */
    bool should_feed(const uint512_t& hash_merkle_root,
                     uint32_t unified_height = 0,
                     const uint1024_t& hash_prev_block = uint1024_t{})
    {
        auto now = std::chrono::steady_clock::now();

        // Height change always allows through.
        // unified_height == 0 means the caller opted out of height tracking.
        bool height_tracking_active = (unified_height != 0);
        bool height_changed = height_tracking_active && (unified_height != m_last_fed_height);

        // Same-height reorg (hashPrevBlock changed) always allows through
        bool reorg_at_same_height = false;
        if (!height_changed &&
            hash_prev_block != uint1024_t{} &&
            m_last_fed_hash_prev_block != uint1024_t{} &&
            hash_prev_block != m_last_fed_hash_prev_block)
        {
            reorg_at_same_height = true;
        }

        if (!height_changed && !reorg_at_same_height &&
            m_last_fed_time != std::chrono::steady_clock::time_point{})
        {
            // Tier 1: same (merkleRoot, height) within 2s → suppress
            if (hash_merkle_root != uint512_t{} &&
                hash_merkle_root == m_last_fed_merkle_root &&
                (!height_tracking_active || unified_height == m_last_fed_height))
            {
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    now - m_last_fed_time).count();
                if (elapsed < SUPPRESSION_WINDOW_SECONDS) {
                    ++m_suppressed_count;
                    return false;  // suppress — identical template
                }
            }

            // Tier 2: same height, different merkle root, within 500ms → suppress
            // This catches mempool-variant duplicates at the same chain tip.
            if (height_tracking_active && unified_height == m_last_fed_height) {
                auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - m_last_fed_time).count();
                if (elapsed_ms < HEIGHT_ONLY_SUPPRESSION_MS) {
                    ++m_suppressed_count;
                    return false;  // suppress — same height, mempool variant
                }
            }
        }

        // New merkle root, different height, reorg, or window expired — record and allow
        m_last_fed_merkle_root = hash_merkle_root;
        m_last_fed_height = unified_height;
        m_last_fed_hash_prev_block = hash_prev_block;
        m_last_fed_time = now;
        return true;
    }

    /**
     * @brief Clear all state (e.g. on session reset).
     */
    void reset()
    {
        m_last_fed_merkle_root = uint512_t{};
        m_last_fed_height = 0;
        m_last_fed_hash_prev_block = uint1024_t{};
        m_last_fed_time = {};
        m_suppressed_count = 0;
    }

    /// Number of feeds suppressed since last reset (diagnostic).
    uint32_t suppressed_count() const { return m_suppressed_count; }

    /// The last hashMerkleRoot that was fed (diagnostic).
    const uint512_t& last_fed_merkle_root() const { return m_last_fed_merkle_root; }

private:
    uint512_t m_last_fed_merkle_root{};
    uint32_t m_last_fed_height{0};
    uint1024_t m_last_fed_hash_prev_block{};
    std::chrono::steady_clock::time_point m_last_fed_time{};
    uint32_t m_suppressed_count{0};
};

} // namespace protocol
} // namespace nexusminer

#endif // NEXUSMINER_PROTOCOL_MERKLE_ROOT_FEED_GUARD_HPP
