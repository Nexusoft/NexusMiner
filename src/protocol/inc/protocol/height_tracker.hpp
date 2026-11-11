#ifndef NEXUSMINER_PROTOCOL_HEIGHT_TRACKER_HPP
#define NEXUSMINER_PROTOCOL_HEIGHT_TRACKER_HPP

#include <algorithm>
#include <cstdint>
#include <chrono>
#include <mutex>
#include <string>
#include <optional>
#include "LLC/types/uint1024.h"
#include "protocol/session_semantic_types.hpp"

namespace nexusminer {
namespace protocol {

struct UnifiedHeight {
    uint32_t value{0};

    constexpr uint32_t get() const noexcept { return value; }
    constexpr explicit operator uint32_t() const noexcept { return value; }
};

struct ChannelHeight {
    uint32_t value{0};

    constexpr uint32_t get() const noexcept { return value; }
    constexpr explicit operator uint32_t() const noexcept { return value; }
};

// Type marker helpers used by submission guards. They intentionally answer
// whether a height was wrapped as ChannelHeight rather than validating the
// runtime value against chain state.
constexpr bool is_channel_height(ChannelHeight channel_height) noexcept {
    (void)channel_height;
    return true;
}
constexpr bool is_channel_height(uint32_t raw_height) noexcept {
    (void)raw_height;
    return false;
}

/**
 * @brief Centralized height tracking utility (single source of truth)
 *
 * Tracks unified blockchain height, channel-specific height, difficulty, and
 * template target height. Provides drift diagnostics to detect and log
 * creeping height mismatches between the miner's displayed/used heights.
 *
 * Architecture:
 *   CanonicalChainState   — updated exclusively by OnBlockDataReceived()
 *                           (BLOCK_DATA / STATELESS_GET_BLOCK).  Monotonically
 *                           advancing.  Drives all mining decisions.
 *   DiagnosticObserverState — updated by push notifications, keepalive ACKs,
 *                           and GET_ROUND responses.  Read-only for Colin
 *                           diagnostics.  Never drives mining decisions.
 *
 * Invariant: only OnBlockDataReceived() may update canonical chain state.
 * Push notifications, GET_ROUND, and keepalive ACKs update
 * DiagnosticObserverState only.
 *
 * GetSnapshot() backward-compat composition:
 *   channel_height = max(canonical, push, round)
 *   unified_height = max(canonical, push, round)
 * This preserves push-driven staleness detection while keepalive can never
 * regress the heights used for mining decisions.
 *
 * Thread-safe: all public methods are protected by an internal mutex.
 * Snapshot() returns a plain-struct copy for lockless reads by callers.
 */
class HeightTracker {
public:
    /**
     * @brief Source of the last height update
     */
    enum class UpdateSource {
        NONE,             ///< No update received yet
        PUSH,             ///< Updated by a push notification (BLOCK_AVAILABLE opcode)
        GET_ROUND,        ///< Updated by a GET_ROUND / NEW_ROUND response
        TEMPLATE,         ///< Updated by a received mining template
        KEEPALIVE,        ///< Updated by a unified keepalive response (both legacy and stateless paths)
    };

    // ─── Canonical chain state (BLOCK_DATA only) ──────────────────────────
    /**
     * @brief Authoritative mining state updated exclusively by OnBlockDataReceived()
     *        and UpdateWithHashPrevBlock().
     *
     * Anchored to the decoded 216-byte Tritium block from BLOCK_DATA /
     * STATELESS_GET_BLOCK.  Heights are monotonically advancing.  This is the
     * ONLY state that drives mining decisions (template staleness, tip-moved,
     * drift detection).
     */
    struct CanonicalChainState {
        uint32_t canonical_unified_height{0};   ///< block.nHeight from BLOCK_DATA
        uint32_t canonical_channel_height{0};   ///< nChannelHeight from BLOCK_DATA metadata prefix
        uint32_t canonical_difficulty_nbits{0}; ///< nBits from BLOCK_DATA metadata prefix
        uint32_t canonical_prime_height{0};     ///< Prime channel height (canonical, from BLOCK_DATA when channel==1)
        uint32_t canonical_hash_height{0};      ///< Hash channel height (canonical, from BLOCK_DATA when channel==2)
        uint32_t canonical_channel_target{0};   ///< channel_height + 1 (the block we're mining for)
        uint1024_t canonical_hash_prev_block{}; ///< hashPrevBlock from BLOCK_DATA (fork detection anchor)
        std::chrono::steady_clock::time_point canonical_received_at{}; ///< When this canonical state was set

        /// True when canonical state has been set at least once from a BLOCK_DATA receipt
        bool is_initialized() const { return canonical_unified_height > 0; }

        /**
         * @brief True when canonical channel target has been met by the chain
         */
        bool is_canonically_stale() const {
            return (canonical_channel_height > 0 && canonical_channel_target > 0 &&
                    canonical_channel_height >= canonical_channel_target);
        }

        /**
         * @brief HEIGHT_DRIFT check using only canonical state
         *
         * Compares canonical_unified_height against canonical_channel_target using
         * signed arithmetic. Returns 0 when they are equal, which can occur when
         * the template's unified height happens to equal the channel target (e.g.,
         * in single-channel contexts or diagnostic checks). Non-zero values indicate
         * the magnitude of the difference between the two dimensions.
         *
         * Note: In a multi-channel blockchain, canonical_unified_height and
         * canonical_channel_target represent different height dimensions, so a
         * non-zero result is normal. This is primarily useful for Colin diagnostics.
         */
        int32_t height_drift_from_canonical() const {
            if (canonical_channel_target == 0 || canonical_unified_height == 0)
                return 0;
            return static_cast<int32_t>(canonical_unified_height) -
                   static_cast<int32_t>(canonical_channel_target);
        }
    };

    /**
     * @brief Diagnostic and telemetry state — for Colin agent observation ONLY
     *
     * Updated by push notifications, keepalive ACKs, and GET_ROUND responses.
     * MUST NOT be used to make any mining decisions (template validation,
     * staleness detection, worker dispatch, fork detection for hard stops).
     */
    struct DiagnosticObserverState {
        // ── Heights from push notifications (BLOCK_AVAILABLE) ─────────────────
        uint32_t push_unified_height{0};
        uint32_t push_channel_height{0};
        uint32_t push_difficulty_nbits{0};
        uint1024_t push_hash_prev_block{};     ///< hashPrevBlock from latest extended push payload
        std::chrono::steady_clock::time_point last_push_at{};

        // Push-derived cross-channel heights (only populated from 148-byte push payload)
        uint32_t push_prime_height{0};
        uint32_t push_hash_height{0};
        uint32_t push_stake_height{0};

        // ── Heights from GET_ROUND / NEW_ROUND responses ────────────────────────
        // Full 16-byte picture: unified + prime + hash + stake (no difficulty).
        uint32_t round_unified_height{0};
        uint32_t round_prime_height{0};
        uint32_t round_hash_height{0};
        uint32_t round_stake_height{0};
        uint32_t round_channel_height{0};  ///< Derived active-channel height (from prime/hash/stake per m_channel)
        uint32_t round_difficulty_nbits{0}; ///< Always 0 — 16-byte format carries no difficulty
        std::chrono::steady_clock::time_point last_round_at{};

        // ── Keepalive telemetry (SESSION_KEEPALIVE ACK) ─────────────────────────
        uint32_t keepalive_unified_height{0};
        uint32_t keepalive_prime_height{0};
        uint32_t keepalive_hash_height{0};
        uint32_t keepalive_stake_height{0};
        uint32_t keepalive_hash_tip_lo32{0};   ///< Lo32 of node's hashBestChain
        uint32_t keepalive_fork_score{0};       ///< Current fork score from node
        uint32_t keepalive_peak_fork_score{0};  ///< High-water mark (diagnostic canary only)
        std::chrono::steady_clock::time_point last_keepalive_ack_at{};

        /// True when the node reported any fork divergence (diagnostic canary — do NOT use for hard stops)
        bool is_fork_canary_active() const { return keepalive_peak_fork_score > 0; }

        /**
         * @brief True when TipSync mismatch is detected (diagnostic warning only)
         *
         * Fires normally during the 0-4 second window after a new block before
         * the miner receives the new template.  NOT a reason to stop workers.
         */
        bool is_tip_sync_mismatch(uint32_t canonical_hash_prev_lo32) const {
            if (keepalive_hash_tip_lo32 == 0 || canonical_hash_prev_lo32 == 0)
                return false;
            return keepalive_hash_tip_lo32 != canonical_hash_prev_lo32;
        }

        /**
         * @brief True when at least one diagnostic source has provided data.
         *
         * Diagnostic equivalent of CanonicalChainState::is_initialized().
         * Returns true as soon as any of push notifications, GET_ROUND responses,
         * or keepalive ACKs have been received.
         */
        bool is_initialized() const {
            return push_unified_height > 0 ||
                   round_unified_height > 0 ||
                   keepalive_unified_height > 0;
        }

        /**
         * @brief Time of the most recent diagnostic update from any source.
         *
         * Diagnostic equivalent of CanonicalChainState::canonical_received_at.
         * Returns the latest timestamp across push, GET_ROUND, and keepalive
         * sources. Returns a default-constructed (epoch) time_point when no
         * source has been received yet.
         */
        std::chrono::steady_clock::time_point latest_received_at() const {
            auto t = std::chrono::steady_clock::time_point{};
            if (last_push_at > t)          t = last_push_at;
            if (last_round_at > t)         t = last_round_at;
            if (last_keepalive_ack_at > t) t = last_keepalive_ack_at;
            return t;
        }
    };

    /**
     * @brief Immutable snapshot of tracker state (thread-safe to copy)
     *
     * All mining-decision heights (unified_height, channel_height, channel_target)
     * are sourced exclusively from BLOCK_DATA (canonical). Push, GET_ROUND, and
     * keepalive heights are kept in separate diagnostic fields for operator
     * visibility but are never used for staleness or validation decisions.
     *
     * Height semantics key:
     *   TIP    = current chain state (last completed block)
     *   TARGET = next block to mine (tip + 1)
     */
    struct Snapshot {
        SessionEpoch session_epoch{};        ///< Authoritative session epoch captured with this snapshot
        uint32_t unified_height{0};          ///< Unified blockchain TIP — Source: BLOCK_DATA (canonical only)
        uint32_t channel_height{0};          ///< Channel-specific TIP — Source: BLOCK_DATA (canonical only)
        uint32_t push_channel_height{0};     ///< Channel TIP from latest push notification — Source: PUSH (diagnostic)
        uint32_t difficulty_nbits{0};        ///< Compact nBits difficulty
        uint32_t channel_target{0};          ///< Channel TARGET (tip+1) — Source: BLOCK_DATA/OnTemplateReceived (canonical only)
        uint32_t channel{0};                 ///< Mining channel (1=Prime, 2=Hash)
        uint32_t template_unified_height{0}; ///< Unified TIP at time of last template receipt — Source: BLOCK_DATA (canonical)
        UnifiedHeight unified_block_height{};       ///< Typed alias of unified_height (BLOCK_DATA TIP)
        ChannelHeight channel_tip_height{};         ///< Typed alias of channel_height (BLOCK_DATA TIP)
        ChannelHeight template_channel_target{};    ///< Typed alias of channel_target (BLOCK_DATA TARGET = tip + 1)
        UnifiedHeight template_block_height{};      ///< Typed alias of template_unified_height
        uint1024_t hash_prev_block{};         ///< hashPrevBlock captured at template parse time (tip anchor)
        uint1024_t push_hash_prev_block{};    ///< hashPrevBlock from latest extended push (pre-adoption tip hint)
        UpdateSource last_update_source{UpdateSource::NONE};

        // ── Per-channel heights (diagnostic only — from PUSH/GET_ROUND/keepalive) ──
        uint32_t prime_height{0};   ///< Prime channel TIP — Source: max(GET_ROUND, keepalive, PUSH) (diagnostic)
        uint32_t hash_height{0};    ///< Hash channel TIP  — Source: max(GET_ROUND, keepalive, PUSH) (diagnostic)
        uint32_t stake_height{0};   ///< Stake channel TIP — Source: max(GET_ROUND, keepalive, PUSH) (diagnostic)
        ChannelHeight prime_channel_height{}; ///< Typed alias of prime_height (diagnostic TIP)
        ChannelHeight hash_channel_height{};  ///< Typed alias of hash_height (diagnostic TIP)
        ChannelHeight stake_channel_height{}; ///< Typed alias of stake_height (diagnostic TIP)

        // ── Push-derived cross-channel heights (from 148-byte full-picture push) ─
        uint32_t push_prime_height{0};  ///< Prime TIP from latest 148-byte push payload — Source: PUSH (diagnostic)
        uint32_t push_hash_height{0};   ///< Hash TIP from latest 148-byte push payload — Source: PUSH (diagnostic)
        uint32_t push_stake_height{0};  ///< Stake TIP from latest 148-byte push payload — Source: PUSH (diagnostic)

        // ── Diagnostic unified/channel heights from non-canonical sources ────
        // These are NEVER used for mining decisions — only for drift computation
        // and operator diagnostics (Colin Height Source Dashboard).
        uint32_t push_unified_height{0};   ///< Unified TIP — Source: PUSH (diagnostic)
        uint32_t round_unified_height{0};  ///< Unified TIP — Source: GET_ROUND (diagnostic)
        uint32_t round_channel_height{0};  ///< Channel TIP — Source: GET_ROUND (diagnostic)

        // ── Fork detection (diagnostic only — from keepalive ACKs) ─────────────
        uint32_t hash_tip_lo32{0};   ///< Lo32 of node's hashBestChain from last keepalive response
        uint32_t fork_score{0};      ///< Latest fork_score from keepalive response (0 = healthy)
        uint32_t peak_fork_score{0}; ///< Highest fork_score seen since start (persistent canary)

        // ── Keepalive timing ───────────────────────────────────────────────────
        std::chrono::steady_clock::time_point last_keepalive_ack_at{}; ///< Time of last OnKeepaliveResponse() call

        /// Time of last actual push notification (PRIME/HASH_BLOCK_AVAILABLE opcode ONLY).
        /// NOT updated by keepalive ACKs or GET_ROUND responses.
        /// Use this field — not last_height_update — for session liveness decisions
        /// (escape ladder push_recent, retry_connect guard). This is the canonical
        /// "is the node pushing to us?" signal.
        std::chrono::steady_clock::time_point last_push_notification_at{};

        /// Time of last push update (set ONLY by OnPushNotification — NOT by keepalive or GET_ROUND).
        /// Serves the temporal post-push guard in check_template_health(): comparing
        /// last_template_update >= last_height_update answers "was the template received
        /// after the last push?" (doom-loop prevention). Use last_push_notification_at —
        /// not this field — for session liveness decisions (escape ladder, retry_connect).
        std::chrono::steady_clock::time_point last_height_update{};
        /// Time of last template update
        std::chrono::steady_clock::time_point last_template_update{};

        /**
         * @brief Expected channel TARGET that the next template should target
         *
         * Returns channel_height + 1 when channel_height > 0 (the next block
         * to be mined on this channel). Returns 0 if no chain data is available.
         * Source: BLOCK_DATA (canonical TIP + 1 = TARGET)
         */
        uint32_t expected_template_target() const {
            return (channel_height > 0) ? (channel_height + 1) : 0;
        }

        /**
         * @brief [REMOVED] is_template_stale() was structurally false under
         * canonical-only semantics: OnBlockDataReceived atomically sets
         * canonical_channel_target = canonical_channel_height + 1, making
         * channel_height >= channel_target impossible.  Staleness detection
         * lives in PUSH, GET_ROUND, and Health Monitor trigger paths instead.
         * Removed to eliminate dead code and the 9 pre-existing test failures
         * that relied on the (now impossible) true-return path.
         */

        /**
         * @brief Get template age in seconds since canonical receipt
         *
         * Returns the age of the current template based on when it was received
         * via OnBlockDataReceived(). Returns 0 if no template has been received yet.
         *
         * @return Age in seconds, or 0 if canonical_received_at is uninitialized
         */
        uint64_t get_template_age_seconds() const {
            if (canonical_received_at == std::chrono::steady_clock::time_point{}) {
                return 0;
            }
            auto age = std::chrono::steady_clock::now() - canonical_received_at;
            return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(age).count());
        }

        /**
         * @brief True when template age exceeds the stale threshold (600s)
         *
         * Push-driven era last-resort timeout: node delivers a fresh template within
         * ~2s of every tip advance, so 600s only fires as a dead-connection detector.
         * Safely above 5-minute Prime block maximum.
         *
         * @return true if template age exceeds 600 seconds
         */
        bool is_template_age_stale() const {
            constexpr uint64_t MAX_TEMPLATE_AGE_SECONDS = 600;
            return get_template_age_seconds() > MAX_TEMPLATE_AGE_SECONDS;
        }

        /**
         * @brief True when template age exceeds the warning threshold (300s)
         *
         * Proactive warning threshold to detect degraded connectivity before
         * hitting the hard 600s timeout. Fires after Prime block window expires.
         *
         * @return true if template age exceeds 300 seconds
         */
        bool is_template_age_old() const {
            constexpr uint64_t WARNING_TEMPLATE_AGE_SECONDS = 300;
            return get_template_age_seconds() > WARNING_TEMPLATE_AGE_SECONDS;
        }

        /**
         * @brief True when the unified tip has moved beyond the height at which
         *        the current template was issued (hashPrevBlock is stale).
         *
         * Returns true when unified_height has advanced past template_unified_height,
         * i.e. another channel found a block after this template was received.
         * This does NOT imply channel staleness — the channel may still be valid.
         * Both values must be non-zero to avoid false positives at startup.
         * Source: both sides are BLOCK_DATA canonical TIP.
         */
        bool is_tip_moved() const {
            return (template_unified_height > 0 && unified_height > template_unified_height);
        }

        /**
         * @brief True when the latest extended push already proves a would-be live template is obsolete
         *
         * This is the last adoption gate for the "already obsolete on arrival" race:
         * an extended push has told us the current canonical tip anchor for this
         * channel height, but a template targeting that same height still builds on a
         * different hashPrevBlock. In that case the template must not be fed live.
         */
        bool has_same_height_push_tip_replacement(const uint1024_t& template_hash_prev_block,
                                                  uint32_t template_channel_target) const {
            return push_hash_prev_block != uint1024_t{} &&
                   push_channel_height > 0 &&
                   template_channel_target > 0 &&
                   template_channel_target == (push_channel_height + 1) &&
                   template_hash_prev_block != push_hash_prev_block;
        }

        /**
         * @brief Number of blocks the template target trails the current channel tip
         *
         * Returns 0 when the template is current or when either height is unset.
         * Returns 1 for the normal "previous block" case after a single channel
         * advance, 2+ when the miner has fallen multiple channel blocks behind.
         */
        uint32_t blocks_behind() const {
            uint32_t expected = expected_template_target();
            if (expected == 0 || channel_target == 0 || channel_target >= expected) {
                return 0;
            }
            return expected - channel_target;
        }

        /**
         * @brief Compute the drift between expected and actual template target
         *
         * A positive delta means the template target is ahead of expected (fine).
         * A negative delta means the template is behind (stale or confused).
         * Returns nullopt when either value is zero (not yet set).
         */
        std::optional<int32_t> drift_delta() const {
            uint32_t expected = expected_template_target();
            if (expected == 0 || channel_target == 0) {
                return std::nullopt;
            }
            return static_cast<int32_t>(channel_target) -
                   static_cast<int32_t>(expected);
        }

        /// True when the chain has reported any fork divergence since startup.
        bool is_fork_active() const { return peak_fork_score > 0; }

        /**
         * @brief Compute how far the diagnostic push/round height has drifted
         *        ahead of canonical BLOCK_DATA height.
         *
         * Returns the signed difference between the best diagnostic unified TIP
         * (max of push, round) and the canonical BLOCK_DATA unified TIP.
         * A large positive value means push/round data is ahead (BLOCK_DATA
         * hasn't caught up yet — normal during slow BLOCK_DATA latency);
         * zero means canonical is caught up.
         *
         * Source: max(PUSH, GET_ROUND) diagnostic TIP − BLOCK_DATA canonical TIP
         */
        int32_t height_drift_from_canonical() const {
            // Use diagnostic push/round heights for drift — unified_height is now canonical-only
            uint32_t diagnostic_unified = std::max({push_unified_height,
                                                     round_unified_height});
            return static_cast<int32_t>(diagnostic_unified) -
                   static_cast<int32_t>(canonical_unified_height);
        }

        // ── Canonical reference (for drift computation and fork detection) ───
        uint32_t canonical_unified_height{0};
        uint32_t canonical_channel_height{0};
        uint1024_t canonical_hash_prev_block{}; ///< From canonical state (BLOCK_DATA decoded Tritium block)
        std::chrono::steady_clock::time_point canonical_received_at{}; ///< When canonical template was received (for age calculation)
    };

    HeightTracker() = default;

    // =========================================================================
    // Canonical update (BLOCK_DATA / STATELESS_GET_BLOCK only)
    // =========================================================================

    /**
     * @brief Set canonical chain state from a received BLOCK_DATA template
     *
     * This is the primary method that updates CanonicalChainState.
     * Called once per successfully parsed BLOCK_DATA receipt (via OnTemplateMetadata
     * or directly). Monotonically advances canonical heights — never regresses.
     *
     * @param block_unified_height   block.nHeight from parsed BLOCK_DATA
     * @param metadata_channel_height nChannelHeight from 12-byte metadata prefix
     * @param metadata_nbits          nBits from 12-byte metadata prefix
     * @param hash_prev_block         hashPrevBlock (pass uint1024_t{} if not yet parsed)
     */
    void OnBlockDataReceived(uint32_t block_unified_height,
                             uint32_t metadata_channel_height,
                             uint32_t metadata_nbits,
                             const uint1024_t& hash_prev_block);

    // =========================================================================
    // Update methods (called from protocol handlers)
    // =========================================================================

    /**
     * @brief Update heights from a push notification payload
     *
     * Writes to DiagnosticObserverState only (push_* fields).
     * Call this after parsing the 12-byte BLOCK_AVAILABLE payload.
     *
     * @param unified_height  Unified blockchain height from bytes [0..3]
     * @param channel_height  Channel-specific height from bytes [4..7]
     * @param nbits           Difficulty in compact nBits from bytes [8..11]
     */
    void OnPushNotification(uint32_t unified_height, uint32_t channel_height,
                            uint32_t nbits);

    /**
     * @brief Update all-channel heights from a 148-byte full-picture push notification
     *
     * Writes to DiagnosticObserverState push_prime/hash/stake_height fields.
     * Does NOT write to CanonicalChainState.
     *
     * @param unified_height  Unified blockchain height
     * @param prime_height    Prime channel height
     * @param hash_height     Hash channel height
     * @param stake_height    Stake channel height
     */
    void OnPushFullPicture(uint32_t unified_height, uint32_t prime_height,
                           uint32_t hash_height, uint32_t stake_height);

    /**
     * @brief Record the tip anchor advertised by the latest extended push payload
     *
     * This is diagnostic/pre-adoption state only: unlike OnBlockDataReceived(), it
     * does not claim canonical ownership. It simply preserves the newest push-side
     * hashPrevBlock so first-template adoption can reject a template that is already
     * obsolete before workers resume.
     */
    void UpdatePushTipAnchor(const uint1024_t& hash_prev_block);

    /**
     * @brief Clear the pre-adoption push tip-anchor hint once it is no longer relevant
     *
     * This consumes the push-side replacement hint after a fresh template is adopted
     * or when a new session epoch invalidates prior-session ingress state.
     */
    void ClearPushTipAnchor();

    /**
     * @brief Record receipt of a BLOCK_AVAILABLE push for liveness only
     *
     * Updates the push timestamp without changing push-derived heights or
     * last_height_update. Use this when a validated push is informational
     * (e.g. non-subscribed channel broadcast) but should still prove the node
     * is alive for degraded-mode recovery decisions.
     */
    void OnPushLiveness();

    /**
     * @brief Update heights from a GET_ROUND / NEW_ROUND response
     *
     * Writes to DiagnosticObserverState only (round_* fields).
     * The 16-byte packet carries the full height picture (no difficulty).
     *
     * @param unified_height  Unified blockchain height
     * @param prime_height    Prime channel height
     * @param hash_height     Hash channel height
     * @param stake_height    Stake channel height
     */
    void OnGetRound(uint32_t unified_height, uint32_t prime_height,
                    uint32_t hash_height, uint32_t stake_height);

    /**
     * @brief Monotonic height update from BLOCK_DATA / STATELESS_GET_BLOCK metadata
     *
     * Delegates to OnBlockDataReceived() for backward compatibility.
     *
     * @param unified_height  Unified height from template metadata
     * @param channel_height  Channel height from template metadata
     * @param nbits           Difficulty from template metadata
     */
    void OnTemplateMetadata(uint32_t unified_height, uint32_t channel_height,
                            uint32_t nbits);

    /**
     * @brief Record that a new mining template has been received
     *
     * Updates canonical_channel_target — only advanced, never regressed.
     * A stale GET_BLOCK response cannot regress the target below what a
     * prior OnBlockDataReceived() has established.
     *
     * @param channel                Mining channel (1=Prime, 2=Hash)
     * @param template_channel_target Block nHeight from the template header
     *                               (represents the channel height this template
     *                               is targeting, i.e. next block to mine)
     */
    void OnTemplateReceived(uint32_t channel, uint32_t template_channel_target);

    // AdvanceChannelTarget() REMOVED — Bug #6 fix.
    // Push-driven advancement of channel_target corrupted the canonical invariant
    // (canonical_channel_target = canonical_channel_height + 1) by allowing PUSH
    // data to inflate canonical_channel_target beyond canonical_channel_height.
    // Post-refactor, canonical_channel_target is set ONLY by OnBlockDataReceived()
    // and OnTemplateReceived().  The doom-loop that AdvanceChannelTarget() prevented
    // cannot occur when is_template_stale() uses canonical-only heights.

    /**
     * @brief Record the hashPrevBlock from the most recently received template
     *
     * Call this after parsing a new template to capture the chain tip anchor.
     * Updates both the canonical state (canonical_hash_prev_block) and the
     * snapshot's hash_prev_block field.
     * Used to detect tip changes between successive templates (StakeMinter pattern).
     *
     * @param h  hashPrevBlock from the template's block header
     */
    void UpdateWithHashPrevBlock(const uint1024_t& h);

    /**
     * @brief Update diagnostic state from a unified 32-byte keepalive response.
     *
     * Writes to DiagnosticObserverState only (keepalive_* fields).
     * Does NOT update CanonicalChainState — keepalive ACKs must never
     * regress the authoritative heights used for mining decisions.
     *
     * Used for BOTH legacy SESSION_KEEPALIVE (port 8323) and stateless
     * KEEPALIVE_V2_ACK (port 9323) — they now share the same wire format.
     *
     * @param unified_height     Node's unified blockchain height
     * @param prime_height       Node's Prime channel height
     * @param hash_height        Node's Hash channel height
     * @param stake_height       Node's Stake channel height
     * @param hash_tip_lo32      Lo32 of node's hashBestChain (fork cross-check)
     * @param fork_score         Fork divergence score (0=healthy)
     */
    void OnKeepaliveResponse(uint32_t unified_height,
                              uint32_t prime_height,
                              uint32_t hash_height,
                              uint32_t stake_height,
                              uint32_t hash_tip_lo32,
                              uint32_t fork_score);

    // =========================================================================
    // Read methods
    // =========================================================================

    /**
     * @brief Return an immutable snapshot of the current state
     *
     * Backward-compatible composition:
     *   unified_height = max(canonical, push, round)
     *   channel_height = max(canonical, push, round)
     *
     * The copy is taken under the internal lock; the returned struct can be
     * used freely without holding any lock.
     */
    Snapshot GetSnapshot() const;

    /**
     * @brief Update the authoritative session epoch associated with future snapshots
     *
     * Used to reject stale snapshots/submissions after a session restart.
     *
     * When the epoch changes, the keepalive ACK timestamp is invalidated so that
     * stale-epoch keepalive responses cannot falsely signal liveness in the new
     * epoch's escape ladder (check_template_health ack_recent computation).
     */
    void set_session_epoch(SessionEpoch session_epoch);

    /**
     * @brief Return a snapshot of canonical chain state only
     *
     * Contains only the authoritative state from OnBlockDataReceived().
     */
    CanonicalChainState GetCanonicalSnapshot() const;

    /**
     * @brief Return a snapshot of diagnostic observer state only
     *
     * Contains push, GET_ROUND, and keepalive telemetry data.
     */
    DiagnosticObserverState GetDiagnosticSnapshot() const;

    /**
     * @brief Produce a human-readable explanation of any height mismatch
     *
     * Returns an empty string when heights are consistent.
     * Suitable for inclusion in info/warn log messages.
     */
    std::string ExplainMismatch() const;

private:
    mutable std::mutex m_mutex;

    // ── Canonical state (BLOCK_DATA only) ──────────────────────────────────
    CanonicalChainState m_canonical;

    // ── Diagnostic state (push / keepalive / GET_ROUND) ────────────────────
    DiagnosticObserverState m_diagnostic;

    // ── Template / shared state ────────────────────────────────────────────
    // m_channel_target REMOVED — Bug #1 fix.
    // Was a legacy field written by OnTemplateReceived() independently of
    // canonical_channel_target.  Now all channel_target writes go through
    // canonical_channel_target exclusively.
    uint32_t m_channel{0};
    uint32_t m_template_unified_height{0};
    UpdateSource m_last_update_source{UpdateSource::NONE};
    std::chrono::steady_clock::time_point m_last_height_update{};
    std::chrono::steady_clock::time_point m_last_template_update{};
    SessionEpoch m_session_epoch{};

    // Latest non-zero difficulty from any non-keepalive source (push, GET_ROUND, block data).
    // Difficulty doesn't suffer from the height-regression problem, so the latest value wins.
    uint32_t m_latest_difficulty_nbits{0};

    /// Build a Snapshot from canonical + diagnostic (must be called under m_mutex).
    Snapshot build_snapshot_locked() const;

    static const char* source_name(UpdateSource src);
};

} // namespace protocol
} // namespace nexusminer

#endif // NEXUSMINER_PROTOCOL_HEIGHT_TRACKER_HPP
