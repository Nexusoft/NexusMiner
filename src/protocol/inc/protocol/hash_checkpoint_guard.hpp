#ifndef NEXUSMINER_PROTOCOL_HASH_CHECKPOINT_GUARD_HPP
#define NEXUSMINER_PROTOCOL_HASH_CHECKPOINT_GUARD_HPP

/**
 * @file hash_checkpoint_guard.hpp
 * @brief Advisory HashCheckpoint Guard for hashPrevBlock reorg monitoring
 *
 * Architecture:
 *   The NODE is always authoritative — this guard NEVER blocks template acceptance.
 *   It maintains a rolling window of recent canonical hashPrevBlock values to:
 *     1. Distinguish shallow reorgs (new hash matches a recent checkpoint) from
 *        deep reorgs (new hash is unknown).
 *     2. Provide reorg_depth_estimate() for Colin diagnostics.
 *     3. Log tiered advisory warnings (INFO/WARN) based on consecutive mismatch count.
 *
 *   HashCheckpoints cannot go through the same reorg process as hashPrevBlock —
 *   they are immutable once recorded. The guard uses these checkpoints to detect
 *   when the node is processing a reorg vs normal chain tip advancement.
 *
 * Thread safety:
 *   This class is designed to be used from a single strand (io_context thread).
 *   No internal mutex — callers must ensure single-writer semantics.
 */

#include <cstdint>
#include <deque>
#include <string>
#include "LLC/types/uint1024.h"

namespace nexusminer {
namespace protocol {

class HashCheckpointGuard {
public:
    /// Maximum number of checkpoints to retain in the rolling window.
    static constexpr std::size_t MAX_CHECKPOINT_HISTORY = 10;

    /**
     * @brief Result of a checkpoint check against incoming template hashPrevBlock
     */
    struct CheckResult {
        bool is_mismatch{false};         ///< True when template hashPrevBlock differs from latest checkpoint
        bool is_shallow_reorg{false};    ///< True when mismatch matches a RECENT checkpoint (depth 1-9)
        uint32_t reorg_depth{0};         ///< Estimated reorg depth (0 = match or unknown, 1-9 = shallow, 10+ = deep)
        uint32_t consecutive_count{0};   ///< Running count of consecutive mismatches
    };

    HashCheckpointGuard() = default;

    /**
     * @brief Record a new canonical hashPrevBlock checkpoint
     *
     * Called unconditionally when BLOCK_DATA is received, regardless of whether
     * the template passes validation. This ensures the checkpoint history tracks
     * reality during reorgs.
     *
     * Deduplicates consecutive identical values (normal same-tip channel advance).
     *
     * @param hash_prev_block  hashPrevBlock from the freshly decoded BLOCK_DATA
     */
    void record_checkpoint(const uint1024_t& hash_prev_block)
    {
        if (hash_prev_block == uint1024_t{})
            return;

        // Dedup consecutive duplicates (same tip, different channel block)
        if (!m_checkpoints.empty() && m_checkpoints.back() == hash_prev_block)
            return;

        m_checkpoints.push_back(hash_prev_block);
        if (m_checkpoints.size() > MAX_CHECKPOINT_HISTORY)
            m_checkpoints.pop_front();
    }

    /**
     * @brief Check a template's hashPrevBlock against the checkpoint history
     *
     * This is an ADVISORY check — the result is used for logging and diagnostics
     * only, NEVER for blocking template acceptance.
     *
     * @param template_hash_prev_block  hashPrevBlock from the incoming template
     * @return CheckResult with mismatch status and reorg depth estimate
     */
    CheckResult check(const uint1024_t& template_hash_prev_block)
    {
        CheckResult result;

        if (m_checkpoints.empty() || template_hash_prev_block == uint1024_t{})
            return result;

        const auto& latest = m_checkpoints.back();

        // No mismatch — template matches the latest checkpoint
        if (template_hash_prev_block == latest) {
            m_consecutive_mismatch = 0;
            return result;
        }

        // Mismatch detected
        result.is_mismatch = true;
        ++m_consecutive_mismatch;
        result.consecutive_count = m_consecutive_mismatch;

        // Search the rolling window (newest-first, skip the latest which we already checked)
        for (std::size_t i = m_checkpoints.size(); i-- > 0; ) {
            if (m_checkpoints[i] == template_hash_prev_block) {
                // Found in history — shallow reorg
                result.is_shallow_reorg = true;
                // Depth = distance from current tip to the matched checkpoint
                result.reorg_depth = static_cast<uint32_t>(m_checkpoints.size() - 1 - i);
                return result;
            }
        }

        // Not found in any checkpoint — deep reorg or entirely new chain segment
        result.reorg_depth = static_cast<uint32_t>(m_checkpoints.size());
        return result;
    }

    /**
     * @brief Reset the consecutive mismatch counter
     *
     * Called when a template is successfully adopted and fed to workers.
     */
    void reset_consecutive() { m_consecutive_mismatch = 0; }

    /**
     * @brief Get current consecutive mismatch count (diagnostic)
     */
    uint32_t consecutive_mismatch_count() const { return m_consecutive_mismatch; }

    /**
     * @brief Estimate the current reorg depth based on the latest check
     *
     * Returns 0 when no mismatch is active. Returns 1-9 for shallow reorgs
     * (template matches a recent checkpoint). Returns checkpoint_count for
     * deep/unknown reorgs.
     *
     * Used by Colin diagnostics to report reorg activity.
     */
    uint32_t reorg_depth_estimate() const { return m_last_reorg_depth; }

    /**
     * @brief Number of checkpoints currently stored
     */
    std::size_t checkpoint_count() const { return m_checkpoints.size(); }

    /**
     * @brief Check if a hash appears anywhere in the checkpoint history
     *
     * Used by Colin diagnostics to color-code hashPrevBlock values.
     */
    bool is_known_checkpoint(const uint1024_t& hash) const
    {
        for (const auto& cp : m_checkpoints) {
            if (cp == hash)
                return true;
        }
        return false;
    }

    /**
     * @brief Get the most recent checkpoint (or zero if empty)
     */
    const uint1024_t& latest_checkpoint() const
    {
        static const uint1024_t zero{};
        return m_checkpoints.empty() ? zero : m_checkpoints.back();
    }

    /**
     * @brief Clear all state (e.g. on session reset)
     */
    void reset()
    {
        m_checkpoints.clear();
        m_consecutive_mismatch = 0;
        m_last_reorg_depth = 0;
    }

    /**
     * @brief Perform a full advisory check and update internal reorg depth tracking
     *
     * Combines check() + internal state update. Returns the CheckResult for
     * the caller to use for logging decisions.
     *
     * @param template_hash_prev_block  hashPrevBlock from the incoming template
     * @return CheckResult with mismatch status and reorg depth estimate
     */
    CheckResult evaluate(const uint1024_t& template_hash_prev_block)
    {
        auto result = check(template_hash_prev_block);
        m_last_reorg_depth = result.is_mismatch ? result.reorg_depth : 0;
        return result;
    }

private:
    /// Rolling window of recent canonical hashPrevBlock values (newest at back).
    /// These are immutable checkpoints — once recorded, they represent confirmed
    /// chain tips that the node has built on.
    std::deque<uint1024_t> m_checkpoints;

    /// Running count of consecutive hashPrevBlock mismatches since the last match.
    uint32_t m_consecutive_mismatch{0};

    /// Last evaluated reorg depth (for diagnostics)
    uint32_t m_last_reorg_depth{0};
};

} // namespace protocol
} // namespace nexusminer

#endif // NEXUSMINER_PROTOCOL_HASH_CHECKPOINT_GUARD_HPP
