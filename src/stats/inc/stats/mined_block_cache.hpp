#ifndef NEXUSMINER_STATS_MINED_BLOCK_CACHE_HPP
#define NEXUSMINER_STATS_MINED_BLOCK_CACHE_HPP

#include "LLC/types/uint1024.h"
#include <chrono>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace nexusminer {
namespace stats {

/// Record of a single mined block accepted by the Nexus network.
struct MinedBlockRecord
{
    uint32_t height{0};
    uint1024_t hash_prev_block{0};       // 128-byte hashPrevBlock
    uint32_t channel{0};                 // 1 = Prime, 2 = Hash
    uint64_t nonce{0};
    std::chrono::steady_clock::time_point accepted_at{};
    uint32_t confirmations{0};           // updated on every new template

    /// Human-readable channel name.
    std::string channel_name() const
    {
        switch (channel) {
            case 1: return "Prime";
            case 2: return "Hash";
            default: return "Unknown";
        }
    }

    /// Status emoji for display.
    /// ⛏ = freshly mined (< 5 confirmations)
    /// ✅ = confirmed (>= 5 confirmations)
    std::string status_emoji() const
    {
        return confirmations >= 5 ? "✅" : "⛏";
    }

    /// One-line summary for logging.
    std::string summary_line() const
    {
        auto prev_bytes = hash_prev_block.GetBytes();
        std::string prev_hex;
        for (size_t i = 0; i < std::min(prev_bytes.size(), size_t(16)); ++i) {
            char buf[3];
            snprintf(buf, sizeof(buf), "%02x", prev_bytes[i]);
            prev_hex += buf;
        }
        return status_emoji() + " height=" + std::to_string(height)
             + " ch=" + channel_name()
             + " conf=" + std::to_string(confirmations)
             + " prev=" + prev_hex + "...";
    }
};

/**
 * Three-tier block confirmation cache.
 *
 * Tier 1 (Hot):     Last 5 mined blocks — confirmation tracking active.
 * Tier 2 (Warm):    Up to 100 confirmed blocks — hash_prev_block + height + channel.
 * Tier 3 (Archive): Overflow from Tier 2 — capped at 1000 entries (FIFO eviction).
 *
 * Thread safety: all methods must be called on the asio I/O thread
 * (same serialisation guarantee as the rest of Worker_manager).
 */
class MinedBlockCache
{
public:
    static constexpr size_t TIER1_MAX = 5;
    static constexpr size_t TIER2_MAX = 100;
    static constexpr size_t TIER3_MAX = 1000;
    static constexpr uint32_t CONFIRMATION_THRESHOLD = 5;

    /// Record a newly accepted block into Tier 1.
    void record_accepted_block(uint32_t height, uint1024_t const& hash_prev_block,
                               uint32_t channel, uint64_t nonce)
    {
        MinedBlockRecord rec;
        rec.height = height;
        rec.hash_prev_block = hash_prev_block;
        rec.channel = channel;
        rec.nonce = nonce;
        rec.accepted_at = std::chrono::steady_clock::now();
        rec.confirmations = 0;

        m_tier1.push_front(rec);

        // If Tier 1 overflows, graduate the oldest to Tier 2.
        while (m_tier1.size() > TIER1_MAX) {
            m_tier2.push_front(m_tier1.back());
            m_tier1.pop_back();
        }

        // If Tier 2 overflows, move the oldest to Tier 3 (archive).
        while (m_tier2.size() > TIER2_MAX) {
            m_tier3.push_front(m_tier2.back());
            m_tier2.pop_back();
        }

        // Cap Tier 3 to prevent unbounded growth over long-running sessions.
        while (m_tier3.size() > TIER3_MAX) {
            m_tier3.pop_back();
        }
    }

    /// Update confirmation counts using the current chain height.
    /// Height-gated: only recalculates when the chain has actually advanced
    /// past the last known height, preventing redundant updates when multiple
    /// templates arrive at the same height (e.g. GET_BLOCK retries).
    void update_confirmations(uint32_t current_chain_height)
    {
        if (current_chain_height <= m_last_confirmation_height)
            return;
        m_last_confirmation_height = current_chain_height;

        for (auto& rec : m_tier1) {
            if (current_chain_height >= rec.height)
                rec.confirmations = current_chain_height - rec.height + 1;
        }
        // Promote fully confirmed Tier 1 blocks to Tier 2.
        // Walk from back (oldest) to front; stop at first unconfirmed.
        while (!m_tier1.empty() &&
               m_tier1.back().confirmations >= CONFIRMATION_THRESHOLD)
        {
            // Only promote if the block is NOT the most recent (front).
            // The front block stays in Tier 1 until a newer block pushes it down.
            if (m_tier1.size() <= 1) break;

            m_tier2.push_front(m_tier1.back());
            m_tier1.pop_back();

            while (m_tier2.size() > TIER2_MAX) {
                m_tier3.push_front(m_tier2.back());
                m_tier2.pop_back();
            }
            // Cap Tier 3 to prevent unbounded growth.
            while (m_tier3.size() > TIER3_MAX) {
                m_tier3.pop_back();
            }
        }
    }

    // ── Accessors ─────────────────────────────────────────────────────────

    /// Tier 1 (hot): most recent first.
    std::deque<MinedBlockRecord> const& tier1() const { return m_tier1; }

    /// Tier 2 (warm): most recent first.
    std::deque<MinedBlockRecord> const& tier2() const { return m_tier2; }

    /// Tier 3 (archive): most recent first.
    std::deque<MinedBlockRecord> const& tier3() const { return m_tier3; }

    /// Total blocks tracked across all tiers.
    size_t total_blocks() const { return m_tier1.size() + m_tier2.size() + m_tier3.size(); }

    /// Build a multi-line log summary of all tiers.
    std::string format_cache_summary() const
    {
        std::string out;
        out += "── Mined Block Cache ──────────────────────────────\n";

        if (m_tier1.empty() && m_tier2.empty() && m_tier3.empty()) {
            out += "  (no blocks mined yet)\n";
            return out;
        }

        // Tier 1 — Hot
        out += "  Tier 1 (Hot — last " + std::to_string(TIER1_MAX) + "):\n";
        if (m_tier1.empty()) {
            out += "    (empty)\n";
        } else {
            for (auto const& rec : m_tier1) {
                out += "    " + rec.summary_line() + "\n";
            }
        }

        // Tier 2 — Warm
        out += "  Tier 2 (Confirmed — up to " + std::to_string(TIER2_MAX) + "): "
             + std::to_string(m_tier2.size()) + " blocks";
        if (!m_tier2.empty()) {
            out += " [newest: h=" + std::to_string(m_tier2.front().height)
                 + " oldest: h=" + std::to_string(m_tier2.back().height) + "]";
        }
        out += "\n";

        // Show channels mined in Tier 2
        if (!m_tier2.empty()) {
            uint32_t prime_count = 0, hash_count = 0;
            for (auto const& rec : m_tier2) {
                if (rec.channel == 1) ++prime_count;
                else if (rec.channel == 2) ++hash_count;
            }
            out += "    Channels: Prime=" + std::to_string(prime_count)
                 + " Hash=" + std::to_string(hash_count) + "\n";
        }

        // Tier 3 — Archive
        out += "  Tier 3 (Archive): "
             + std::to_string(m_tier3.size()) + " blocks\n";

        return out;
    }

private:
    std::deque<MinedBlockRecord> m_tier1;  // Hot:    ≤5  blocks
    std::deque<MinedBlockRecord> m_tier2;  // Warm:   ≤100 blocks
    std::deque<MinedBlockRecord> m_tier3;  // Archive: ≤1000 blocks
    uint32_t m_last_confirmation_height{0};  // Height-gate for update_confirmations()
};

}  // namespace stats
}  // namespace nexusminer

#endif
