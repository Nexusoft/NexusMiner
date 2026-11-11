#include "protocol/height_tracker.hpp"
#include <algorithm>
#include <sstream>

namespace nexusminer {
namespace protocol {

const char* HeightTracker::source_name(UpdateSource src) {
    switch (src) {
        case UpdateSource::PUSH:      return "PUSH";
        case UpdateSource::GET_ROUND: return "GET_ROUND";
        case UpdateSource::TEMPLATE:  return "TEMPLATE";
        case UpdateSource::KEEPALIVE: return "KEEPALIVE";
        default:                      return "NONE";
    }
}

// ── OnPushNotification: updates DiagnosticObserverState push fields ONLY ──────
// Does NOT touch canonical state. Push-derived heights are diagnostic-only and
// appear in Snapshot via push_channel_height, push_unified_height fields for
// drift computation and Colin diagnostics. They are NOT included in
// Snapshot::unified_height / channel_height (which are canonical-only).
void HeightTracker::OnPushNotification(uint32_t unified_height,
                                        uint32_t channel_height,
                                        uint32_t nbits)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_diagnostic.push_unified_height = unified_height;
    m_diagnostic.push_channel_height = channel_height;
    m_diagnostic.push_difficulty_nbits = nbits;
    if (nbits != 0)
        m_latest_difficulty_nbits = nbits;

    m_last_update_source = UpdateSource::PUSH;
    auto now = std::chrono::steady_clock::now();
    m_last_height_update = now;
    m_diagnostic.last_push_at = now;
}

void HeightTracker::UpdatePushTipAnchor(const uint1024_t& hash_prev_block)
{
    if (hash_prev_block == uint1024_t{}) {
        return;
    }

    std::lock_guard<std::mutex> lock(m_mutex);
    m_diagnostic.push_hash_prev_block = hash_prev_block;
}

void HeightTracker::ClearPushTipAnchor()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_diagnostic.push_hash_prev_block = uint1024_t{};
}

void HeightTracker::OnPushLiveness()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_diagnostic.last_push_at = std::chrono::steady_clock::now();
}

// ── OnPushFullPicture: updates DiagnosticObserverState push cross-channel heights ──
// Called from 148-byte BLOCK_AVAILABLE payloads to record the full height picture.
// Does NOT touch canonical state or push_channel_height/push_unified_height.
void HeightTracker::OnPushFullPicture(uint32_t unified_height,
                                       uint32_t prime_height,
                                       uint32_t hash_height,
                                       uint32_t stake_height)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_diagnostic.push_unified_height = std::max(m_diagnostic.push_unified_height, unified_height);
    m_diagnostic.push_prime_height   = std::max(m_diagnostic.push_prime_height,   prime_height);
    m_diagnostic.push_hash_height    = std::max(m_diagnostic.push_hash_height,    hash_height);
    m_diagnostic.push_stake_height   = std::max(m_diagnostic.push_stake_height,   stake_height);
    m_diagnostic.last_push_at        = std::chrono::steady_clock::now();
}

// ── OnGetRound: updates DiagnosticObserverState round fields ONLY ─────────────
// 16-byte full-height-picture format: unified + prime + hash + stake (no difficulty).
void HeightTracker::OnGetRound(uint32_t unified_height,
                                uint32_t prime_height,
                                uint32_t hash_height,
                                uint32_t stake_height)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    // ✅ MONOTONIC GUARD: Only advance round_unified_height — a stale or
    // out-of-order GET_ROUND response must NOT regress the composed snapshot's
    // unified_height (which uses max(canonical, push, round)).  Without this
    // guard, a lagging GET_ROUND can overwrite a higher value from a prior
    // response, causing the Stake/cross-channel detection to see a false
    // regression and triggering unnecessary template discard + GET_BLOCK.
    if (unified_height > m_diagnostic.round_unified_height)
        m_diagnostic.round_unified_height = unified_height;
    // ✅ MONOTONIC GUARD: Only advance per-channel round heights — a stale or
    // out-of-order GET_ROUND response must NOT regress diagnostic state, matching
    // the pattern used by OnTemplateReceived() for channel_target.
    if (prime_height > m_diagnostic.round_prime_height)
        m_diagnostic.round_prime_height = prime_height;
    if (hash_height > m_diagnostic.round_hash_height)
        m_diagnostic.round_hash_height = hash_height;
    if (stake_height > m_diagnostic.round_stake_height)
        m_diagnostic.round_stake_height = stake_height;
    // 16-byte GET_ROUND carries no difficulty — always zero.
    m_diagnostic.round_difficulty_nbits = 0;
    // Derive active-channel height with monotonic guard.
    // Channel 1 = Prime, 2 = Hash; anything else → 0.
    uint32_t new_channel_height = 0;
    if (m_channel == 1)
        new_channel_height = prime_height;
    else if (m_channel == 2)
        new_channel_height = hash_height;

    if (new_channel_height > m_diagnostic.round_channel_height)
        m_diagnostic.round_channel_height = new_channel_height;

    m_last_update_source = UpdateSource::GET_ROUND;
    auto now = std::chrono::steady_clock::now();
    m_diagnostic.last_round_at = now;
}

// ── OnTemplateMetadata: backward-compat wrapper → delegates to OnBlockDataReceived ──
// Kept for existing call sites in update_height_state(TEMPLATE). The hash_prev_block
// will be set separately via UpdateWithHashPrevBlock() after read_template() succeeds.
void HeightTracker::OnTemplateMetadata(uint32_t unified_height,
                                        uint32_t channel_height,
                                        uint32_t nbits)
{
    OnBlockDataReceived(unified_height, channel_height, nbits, uint1024_t{});
}

// ── OnBlockDataReceived: THE canonical update (BLOCK_DATA / STATELESS_GET_BLOCK) ─
// Only method that writes to m_canonical. Advances monotonically — never regresses.
void HeightTracker::OnBlockDataReceived(uint32_t block_unified_height,
                                         uint32_t metadata_channel_height,
                                         uint32_t metadata_nbits,
                                         const uint1024_t& hash_prev_block)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    // Monotonic — only advance canonical heights
    if (block_unified_height > m_canonical.canonical_unified_height)
        m_canonical.canonical_unified_height = block_unified_height;

    if (metadata_channel_height > m_canonical.canonical_channel_height) {
        m_canonical.canonical_channel_height = metadata_channel_height;
        // Advance channel_target to channel_height + 1 (only if higher)
        uint32_t new_target = metadata_channel_height + 1;
        if (new_target > m_canonical.canonical_channel_target)
            m_canonical.canonical_channel_target = new_target;
    }

    if (metadata_nbits != 0)
        m_canonical.canonical_difficulty_nbits = metadata_nbits;

    if (hash_prev_block != uint1024_t{})
        m_canonical.canonical_hash_prev_block = hash_prev_block;

    m_canonical.canonical_received_at = std::chrono::steady_clock::now();
    m_last_update_source = UpdateSource::TEMPLATE;
}

void HeightTracker::OnTemplateReceived(uint32_t channel,
                                        uint32_t template_channel_target)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_channel = channel;
    // Write directly to canonical_channel_target — Bug #1 fix.
    // Previously this wrote to the legacy m_channel_target field while
    // OnBlockDataReceived() wrote to canonical_channel_target, creating
    // dual-write state corruption masked by a max() in GetSnapshot().
    // Now there is a SINGLE channel_target field: canonical_channel_target.
    if (template_channel_target > m_canonical.canonical_channel_target) {
        m_canonical.canonical_channel_target = template_channel_target;
        // Clear fork scores after successful recovery — a fresh template
        // that advances the target means the fork is resolved.
        m_diagnostic.keepalive_fork_score = 0;
        m_diagnostic.keepalive_peak_fork_score = 0;
    }
    // Bug #2 fix: capture tip from canonical (not composite).
    // Previously used build_snapshot_locked().unified_height which is
    // max(canonical, push, round). A stale push could inflate this value,
    // causing is_tip_moved() to miss real tip movements.
    m_template_unified_height = m_canonical.canonical_unified_height;
    m_last_update_source = UpdateSource::TEMPLATE;
    m_last_template_update = std::chrono::steady_clock::now();
}

// AdvanceChannelTarget() REMOVED — Bug #6 fix.
// See height_tracker.hpp comment for rationale.

void HeightTracker::UpdateWithHashPrevBlock(const uint1024_t& h)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    // Clear fork scores when hashPrevBlock changes (tip has moved).
    // This resolves persistent fork_score=1 when the miner receives a new
    // template that syncs to the current chain tip, even if channel_target
    // hasn't advanced (e.g., when another channel found a block).
    bool tip_changed = (m_canonical.canonical_hash_prev_block != uint1024_t(0) &&
                        m_canonical.canonical_hash_prev_block != h);

    m_canonical.canonical_hash_prev_block = h;

    if (tip_changed) {
        // Template with new hashPrevBlock means we've synced to the new tip
        m_diagnostic.keepalive_fork_score = 0;
        m_diagnostic.keepalive_peak_fork_score = 0;
    }
}

// ── OnKeepaliveResponse: updates DiagnosticObserverState keepalive fields ONLY ─
// Does NOT call sync_channel_height_locked() — keepalive data must NOT regress
// canonical channel_height or corrupt is_template_stale() / fork detection.
void HeightTracker::OnKeepaliveResponse(uint32_t unified_height,
                                         uint32_t prime_height,
                                         uint32_t hash_height,
                                         uint32_t stake_height,
                                         uint32_t hash_tip_lo32,
                                         uint32_t fork_score)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    // Diagnostic only — keepalive ACKs never update canonical state.
    m_diagnostic.keepalive_unified_height = unified_height;
    m_diagnostic.keepalive_prime_height   = prime_height;
    m_diagnostic.keepalive_hash_height    = hash_height;
    m_diagnostic.keepalive_stake_height   = stake_height;
    m_diagnostic.keepalive_hash_tip_lo32      = hash_tip_lo32;
    m_diagnostic.keepalive_fork_score         = fork_score;
    if (fork_score > m_diagnostic.keepalive_peak_fork_score)
        m_diagnostic.keepalive_peak_fork_score = fork_score;
    m_last_update_source = UpdateSource::KEEPALIVE;
    auto now = std::chrono::steady_clock::now();
    m_diagnostic.last_keepalive_ack_at = now;
}

HeightTracker::Snapshot HeightTracker::build_snapshot_locked() const {
    Snapshot s;

    s.session_epoch = m_session_epoch;

    // Canonical-only unified/channel heights (BLOCK_DATA TIP).
    // Previously composed as max(canonical, push, round) — this caused
    // is_template_stale() false positives when PUSH raced ahead of BLOCK_DATA,
    // and made HEIGHT_DRIFT diagnostics unreliable. Now strictly canonical.
    // Push/round heights are kept in diagnostic fields for operator visibility.
    s.unified_height = m_canonical.canonical_unified_height;     // Source: BLOCK_DATA (TIP)
    s.channel_height = m_canonical.canonical_channel_height;     // Source: BLOCK_DATA (TIP)
    s.push_channel_height = m_diagnostic.push_channel_height;
    s.unified_block_height = UnifiedHeight{s.unified_height};
    s.channel_tip_height = ChannelHeight{s.channel_height};

    // Difficulty: latest non-zero from any non-keepalive source
    s.difficulty_nbits = m_latest_difficulty_nbits;

    // Bug #1 fix: single source of truth for channel_target.
    // Previously: max(m_channel_target, m_canonical.canonical_channel_target)
    // Now: canonical_channel_target only (m_channel_target eliminated).
    s.channel_target = m_canonical.canonical_channel_target;
    s.template_channel_target = ChannelHeight{s.channel_target};
    s.channel = m_channel;
    s.template_unified_height = m_template_unified_height;
    s.template_block_height = UnifiedHeight{s.template_unified_height};
    s.hash_prev_block = m_canonical.canonical_hash_prev_block;
    s.push_hash_prev_block = m_diagnostic.push_hash_prev_block;
    s.last_update_source = m_last_update_source;

    // Per-channel heights: max of keepalive, GET_ROUND, and push full-picture diagnostic sources.
    // All are diagnostic-only and must never regress canonical mining decisions.
    s.prime_height = std::max({m_diagnostic.keepalive_prime_height,
                                m_diagnostic.round_prime_height,
                                m_diagnostic.push_prime_height});
    s.hash_height  = std::max({m_diagnostic.keepalive_hash_height,
                                m_diagnostic.round_hash_height,
                                m_diagnostic.push_hash_height});
    s.stake_height = std::max({m_diagnostic.keepalive_stake_height,
                                m_diagnostic.round_stake_height,
                                m_diagnostic.push_stake_height});
    s.prime_channel_height = ChannelHeight{s.prime_height};
    s.hash_channel_height = ChannelHeight{s.hash_height};
    s.stake_channel_height = ChannelHeight{s.stake_height};

    // Raw push-derived cross-channel heights (from 148-byte full-picture payload)
    s.push_prime_height = m_diagnostic.push_prime_height;
    s.push_hash_height  = m_diagnostic.push_hash_height;
    s.push_stake_height = m_diagnostic.push_stake_height;

    // Diagnostic unified/channel heights from non-canonical sources
    // (for height_drift_from_canonical() and Colin Height Source Dashboard)
    s.push_unified_height  = m_diagnostic.push_unified_height;   // Source: PUSH (TIP)
    s.round_unified_height = m_diagnostic.round_unified_height;  // Source: GET_ROUND (TIP)
    s.round_channel_height = m_diagnostic.round_channel_height;  // Source: GET_ROUND (TIP)

    // Fork detection fields — diagnostic only
    s.hash_tip_lo32 = m_diagnostic.keepalive_hash_tip_lo32;
    s.fork_score = m_diagnostic.keepalive_fork_score;
    s.peak_fork_score = m_diagnostic.keepalive_peak_fork_score;

    // Timing
    s.last_keepalive_ack_at = m_diagnostic.last_keepalive_ack_at;
    s.last_push_notification_at = m_diagnostic.last_push_at;
    s.last_height_update = m_last_height_update;
    s.last_template_update = m_last_template_update;

    // Canonical reference for drift computation and fork detection
    s.canonical_unified_height = m_canonical.canonical_unified_height;
    s.canonical_channel_height = m_canonical.canonical_channel_height;
    s.canonical_hash_prev_block = m_canonical.canonical_hash_prev_block;
    s.canonical_received_at = m_canonical.canonical_received_at;

    return s;
}

HeightTracker::Snapshot HeightTracker::GetSnapshot() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return build_snapshot_locked();
}

void HeightTracker::set_session_epoch(SessionEpoch session_epoch)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    // When the session epoch advances, invalidate the keepalive timestamp so
    // stale-epoch keepalive ACKs cannot falsely signal liveness in the new
    // epoch's escape ladder (check_template_health ack_recent computation).
    // Without this, a keepalive received for epoch N can keep ack_recent=true
    // during epoch N+1, suppressing Stage-0 fast reconnect when the connection
    // is actually dead.
    if (session_epoch != m_session_epoch && !session_epoch.is_default()) {
        m_diagnostic.last_keepalive_ack_at = {};
        m_diagnostic.push_hash_prev_block = uint1024_t{};
    }
    m_session_epoch = session_epoch;
}

HeightTracker::CanonicalChainState HeightTracker::GetCanonicalSnapshot() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_canonical;
}

HeightTracker::DiagnosticObserverState HeightTracker::GetDiagnosticSnapshot() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_diagnostic;
}

std::string HeightTracker::ExplainMismatch() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    const Snapshot s = build_snapshot_locked();

    // Nothing to explain if we have no data yet
    if (s.channel_height == 0 && s.channel_target == 0) {
        return {};
    }

    std::ostringstream oss;

    // is_template_stale() removed (structurally false under canonical-only semantics).

    // Check drift between expected and actual template target
    uint32_t expected = s.expected_template_target();
    if (expected > 0 && s.channel_target > 0 && expected != s.channel_target) {
        int32_t delta = static_cast<int32_t>(s.channel_target) -
                        static_cast<int32_t>(expected);
        oss << "[HeightTracker] DRIFT: channel_target=" << s.channel_target
            << " expected=" << expected
            << " delta=" << delta
            << " (canonical_channel_tip=" << s.channel_height << ")"
            << " source=" << source_name(s.last_update_source);
        return oss.str();
    }

    return {};
}

} // namespace protocol
} // namespace nexusminer
