#include "colin_agent.hpp"

#include <asio/io_context.hpp>

#include <cstdlib>
#include <ctime>

namespace {
    // ANSI color palette for hashPrevBlock history (5 slots)
    static const char* const HASH_COLORS[5] = {
        "\033[36m",   // 0 → Cyan      (most recent)
        "\033[33m",   // 1 → Yellow
        "\033[35m",   // 2 → Magenta
        "\033[32m",   // 3 → Green
        "\033[94m",   // 4 → Bright Blue
    };
    static const char* const ANSI_RESET = "\033[0m";
    static const char* const ANSI_BOLD  = "\033[1m";
} // anonymous namespace

// Returns first 8 bytes of a uint1024_t as "aabbccdd..." hex preview (Template Anchor style)
static std::string format_hash_anchor(const uint1024_t& h)
{
    auto bytes = h.GetBytes();
    std::string s;
    s.reserve(19);
    for (size_t i = 0; i < std::min(bytes.size(), size_t(8)); ++i) {
        char buf[3];
        snprintf(buf, sizeof(buf), "%02x", bytes[i]);
        s += buf;
    }
    s += "...";
    return s;
}

namespace nexusminer
{

// Warning-catalog threshold constants
static constexpr uint32_t WARN_CONNECTION_RETRIES = 100;
static constexpr uint64_t WARN_TEMPLATE_AGE_SECONDS = 150;
static constexpr int64_t DIAG_KEEPALIVE_ACK_STALE_SECONDS = 32400;  // 9h without keepalive ACK (diagnostic only); 1.5× default 6h keepalive cadence
static constexpr int64_t DIAG_SESSION_STATUS_ACK_STALE_SECONDS = 750;  // 12.5 min; 2.5× 5 min SESSION_STATUS_ACK cadence
static constexpr int32_t WARN_CANONICAL_DRIFT_THRESHOLD = 500;    // blocks ahead before warning
static constexpr uint64_t WARN_DIAGNOSTIC_STALE_SECONDS = 180;    // 3 min without any diagnostic update
static constexpr uint64_t WARN_DIAGNOSTIC_INIT_GRACE_SECONDS = 30; // grace period before warning about uninit diagnostic

ColinAgent::ColinAgent(
    std::shared_ptr<asio::io_context> io_context,
    const DualConnectionManager* dcm,
    std::shared_ptr<stats::Collector> stats,
    std::shared_ptr<spdlog::logger> logger,
    uint32_t report_interval_seconds)
    : m_io_context{std::move(io_context)}
    , m_dcm{dcm}
    , m_stats{std::move(stats)}
    , m_logger{std::move(logger)}
    , m_interval_s{report_interval_seconds > 0 ? report_interval_seconds : 60u}
    , m_timer{*m_io_context}
{
}

void ColinAgent::start()
{
    m_running = true;
    schedule_next();
    m_logger->info("[Colin] Diagnostic agent started (report every {}s)", m_interval_s);
}

void ColinAgent::stop()
{
    m_running = false;
    m_timer.cancel();
    m_logger->info("[Colin] Diagnostic agent stopped");
}

void ColinAgent::schedule_next()
{
    if (!m_running) return;
    m_timer.expires_after(std::chrono::seconds(m_interval_s));
    m_timer.async_wait([this](const asio::error_code& ec) {
        if (ec || !m_running) return;
        run_diagnostics();
        schedule_next();
    });
}

// ── Warning catalog (static helpers) ─────────────────────────────────────────

std::string ColinAgent::check_base_not_prime(const std::string& log_line)
{
    if (log_line.find("BASE IS NOT PRIME") != std::string::npos)
        return "Node PrimeCheck rejected hashPrime base → verify nNonce LE encoding (PR #180)";
    return {};
}

std::string ColinAgent::check_malformed_packet(const std::string& log_line)
{
    if (log_line.find("MALFORMED PACKET DETECTED") != std::string::npos ||
        log_line.find("MALFORMED") != std::string::npos)
        return "Likely node sent null BLOCK_DATA → check node new_block() retry (PR #283)";
    return {};
}

std::string ColinAgent::check_block_rejected_none(const std::string& log_line)
{
    if (log_line.find("BLOCK REJECTED") != std::string::npos &&
        log_line.find("reason=NONE") != std::string::npos)
        return "Node supplied no rejection reason → check node logs for the submission failure";
    return {};
}

std::string ColinAgent::check_no_offsets_found(const std::string& log_line)
{
    if (log_line.find("NO OFFSETS FOUND") != std::string::npos)
        return "Cunningham chain vOffsets empty: base not prime — chain search failed";
    return {};
}

std::string ColinAgent::check_connection_retries(uint32_t retry_count)
{
    if (retry_count > WARN_CONNECTION_RETRIES)
        return "Connection retries high (" + std::to_string(retry_count) +
               ") → check network / node restart";
    return {};
}

std::string ColinAgent::check_template_age(uint64_t age_seconds)
{
    if (age_seconds > WARN_TEMPLATE_AGE_SECONDS)
        return "Template age " + std::to_string(age_seconds) +
               "s — approaching emergency timeout → verify push notifications working";
    return {};
}

std::string ColinAgent::check_mining_stopped(bool degraded_mode)
{
    if (degraded_mode)
        return "MINING STOPPED: workers in degraded mode → check template delivery path";
    return {};
}

std::string ColinAgent::check_fork_score(uint32_t fork_score, uint32_t peak_fork_score)
{
    if (fork_score > 0)
        return "FORK CANARY active: fork_score=" + std::to_string(fork_score) +
               " peak=" + std::to_string(peak_fork_score) +
               " — miner may be on a divergent chain";
    return {};
}

std::string ColinAgent::check_canonical_drift(int32_t drift)
{
    if (drift == 0) return {};
    if (std::abs(drift) > WARN_CANONICAL_DRIFT_THRESHOLD)
        return "Canonical height drift=" + std::to_string(drift) +
               " (|drift|>" + std::to_string(WARN_CANONICAL_DRIFT_THRESHOLD) +
               ") — unified vs channel_target skew outside expected range";
    return {};
}

std::string ColinAgent::check_diagnostic_initialized(bool is_initialized,
                                                      uint64_t elapsed_seconds)
{
    if (is_initialized) return {};
    if (elapsed_seconds < WARN_DIAGNOSTIC_INIT_GRACE_SECONDS) return {};
    return "DiagnosticObserver uninitialized after " + std::to_string(elapsed_seconds) +
           "s — no push notification, GET_ROUND, or keepalive ACK received yet"
           " — check node connectivity";
}

std::string ColinAgent::check_diagnostic_freshness(uint64_t latest_age_seconds,
                                                    bool is_initialized)
{
    if (!is_initialized) return {};  // not yet initialized — separate check covers this
    if (latest_age_seconds <= WARN_DIAGNOSTIC_STALE_SECONDS) return {};
    return "Diagnostic observer silent for " + std::to_string(latest_age_seconds) +
           "s (no push/GET_ROUND/keepalive) — node may have dropped the session"
           " or push notifications have stopped";
}

std::string ColinAgent::check_failover_active(bool using_failover, uint64_t active_seconds,
                                              const std::string& active_ep,
                                              const std::string& standby_ep)
{
    if (!using_failover) return {};
    return "Failover active for " + std::to_string(active_seconds) + "s on " + active_ep +
           " — primary " + standby_ep + " unreachable";
}

// ── Main diagnostic run ───────────────────────────────────────────────────────

void ColinAgent::run_diagnostics()
{
    stats::Global gs{};
    if (m_stats)
        gs = m_stats->get_global_stats();

    // Save snapshot for trend analysis (keep last 10)
    DiagSnapshot snap;
    snap.blocks_accepted    = gs.m_accepted_blocks;
    snap.blocks_rejected    = gs.m_rejected_blocks;
    snap.connection_retries = gs.m_connection_retries;
    snap.timestamp          = std::chrono::steady_clock::now();
    m_history.push_back(snap);
    if (m_history.size() > 10) m_history.pop_front();

    // Update hashPrevBlock history ring buffer (max 5, dedup consecutive duplicates)
    if (m_height_tracker) {
        auto canonical = m_height_tracker->GetCanonicalSnapshot();
        if (canonical.canonical_hash_prev_block != uint1024_t{}) {
            std::string hex = canonical.canonical_hash_prev_block.GetHex();
            if (m_prev_hash_history.empty() || m_prev_hash_history.back() != hex) {
                m_prev_hash_history.push_back(hex);
                if (m_prev_hash_history.size() > 5) m_prev_hash_history.pop_front();
            }
        }
    }

    // Build warnings list
    std::vector<std::string> warnings;
    std::vector<std::string> recommendations;

    {
        auto w = check_connection_retries(gs.m_connection_retries);
        if (!w.empty()) warnings.push_back(w);
    }
    {
        auto w = check_mining_stopped(gs.m_degraded_mode);
        if (!w.empty()) warnings.push_back(w);
    }

    if (!m_dcm || !m_dcm->any_lane_alive())
        warnings.push_back("BOTH lanes unavailable — no template delivery possible");

    if (m_dcm && !m_dcm->is_stateless_alive())
        recommendations.push_back("Check primary (stateless:9323) connectivity");
    // Secondary down: only recommend when primary is ALSO down.
    // When primary is alive, secondary being down is normal single-lane operation.
    if (m_dcm && !m_dcm->is_legacy_alive() && !m_dcm->is_stateless_alive())
        recommendations.push_back("Check secondary (legacy:8323) connectivity");

    if (m_height_tracker) {
        auto ht_snap = m_height_tracker->GetSnapshot();
        {
            auto w = check_fork_score(ht_snap.fork_score, ht_snap.peak_fork_score);
            if (!w.empty()) {
                warnings.push_back(w);
                recommendations.push_back("Check node chain sync; consider restart if fork_score persists");
            }
        }
        if (ht_snap.last_keepalive_ack_at != std::chrono::steady_clock::time_point{}) {
            auto since_ack = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - ht_snap.last_keepalive_ack_at).count();
            if (since_ack > DIAG_KEEPALIVE_ACK_STALE_SECONDS) {
                // Diagnostic only — keepalive ACK is not authoritative for session liveness
                recommendations.push_back("Keepalive ACK stale (" + std::to_string(since_ack) +
                                          "s) — diagnostic only, PUSH is authoritative");
            }
        }

        // ── New hooks: diagnostic observer health ─────────────────────────────
        {
            auto diag = m_height_tracker->GetDiagnosticSnapshot();
            uint64_t elapsed_s = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::steady_clock::now() - m_start_time).count());
            {
                auto w = check_diagnostic_initialized(diag.is_initialized(), elapsed_s);
                if (!w.empty()) {
                    warnings.push_back(w);
                    recommendations.push_back(
                        "Ensure node is sending push notifications and keepalive ACKs");
                }
            }
            {
                auto latest = diag.latest_received_at();
                uint64_t age_s = 0;
                if (latest != std::chrono::steady_clock::time_point{}) {
                    age_s = static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::seconds>(
                            std::chrono::steady_clock::now() - latest).count());
                }
                auto w = check_diagnostic_freshness(age_s, diag.is_initialized());
                if (!w.empty()) {
                    warnings.push_back(w);
                    recommendations.push_back(
                        "Check node block propagation; verify session keepalive is active");
                }
            }
        }
    }

    if (m_failover_source) {
        auto fs = m_failover_source();
        if (fs.has_failover_configured) {
            auto w = check_failover_active(fs.using_failover, fs.failover_active_seconds,
                                           fs.active_endpoint_str, fs.standby_endpoint_str);
            if (!w.empty()) {
                warnings.push_back(w);
                if (fs.failover_active_seconds > 300) {
                    warnings.push_back("Extended failover: primary has been unreachable for " +
                                       std::to_string(fs.failover_active_seconds / 60) + " min — check primary node");
                    recommendations.push_back("Inspect primary node " + fs.standby_endpoint_str + " — restart or check network");
                }
            }
        }
    }

    emit_report(warnings, recommendations, gs);
}
void ColinAgent::emit_report(
    std::vector<std::string>& warnings,
    std::vector<std::string>& recommendations,
    const stats::Global& gs)
{
    // Timestamp
    auto now_t = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now_t);
    char ts_buf[32];
    std::strftime(ts_buf, sizeof(ts_buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now_c));

    // Fetch all HeightTracker snapshots once; reused throughout emit_report()
    nexusminer::protocol::HeightTracker::Snapshot            ht_snap{};
    nexusminer::protocol::HeightTracker::CanonicalChainState canonical{};
    nexusminer::protocol::HeightTracker::DiagnosticObserverState diag{};
    if (m_height_tracker) {
        ht_snap   = m_height_tracker->GetSnapshot();
        canonical = m_height_tracker->GetCanonicalSnapshot();
        diag      = m_height_tracker->GetDiagnosticSnapshot();
    }

    m_logger->info("{}[Colin] ⛏️  ════ DIAGNOSTIC REPORT [{}] ════ ⛏️{}", ANSI_BOLD, ts_buf, ANSI_RESET);
    m_logger->info("[Colin]  {} BLOCKS  Accepted: {}  Rejected: {}  Retries: {}",
        gs.m_degraded_mode ? "❌" : "✅", gs.m_accepted_blocks, gs.m_rejected_blocks, gs.m_connection_retries);

    /* hashPrevBlock History — last 5 canonical templates */
    m_logger->info("[Colin]  🔗 ── hashPrevBlock History (last 5 templates) ──────");
    if (m_prev_hash_history.empty()) {
        m_logger->info("[Colin]    💤 (no templates received yet)");
    } else {
        for (size_t i = 0; i < m_prev_hash_history.size(); ++i) {
            size_t idx = m_prev_hash_history.size() - 1 - i; // newest first
            if (i == 0)
                m_logger->info("[Colin]    [{}] (most recent) {}{}{}", i + 1, HASH_COLORS[i], m_prev_hash_history[idx], ANSI_RESET);
            else
                m_logger->info("[Colin]    [{}] {}{}{}", i + 1, HASH_COLORS[i], m_prev_hash_history[idx], ANSI_RESET);
        }
    }

    /* HashCheckpoint Guard — reorg depth diagnostics */
    if (m_checkpoint_guard_source)
    {
        auto cpg = m_checkpoint_guard_source();
        m_logger->info("[Colin]  🛡️ ── HashCheckpoint Guard (checkpoints={}) ──────", cpg.checkpoint_count);
        if (cpg.consecutive_mismatch > 0) {
            const char* reorg_label = (cpg.reorg_depth_estimate > 0 && cpg.reorg_depth_estimate < 10)
                ? "shallow" : (cpg.reorg_depth_estimate >= 10 ? "deep" : "unknown");
            m_logger->warn("[Colin]    ⚠️  Active reorg: depth_est={} type={} consecutive_mismatch={}",
                cpg.reorg_depth_estimate, reorg_label, cpg.consecutive_mismatch);
            warnings.push_back("HashCheckpoint Guard: active reorg (depth=" +
                std::to_string(cpg.reorg_depth_estimate) + ", mismatches=" +
                std::to_string(cpg.consecutive_mismatch) + ")");
        } else {
            m_logger->info("[Colin]    ✅ No active reorg (chain tip stable)");
        }
    }

    /* SESSION_STATUS_ACK section — node lane-health report */
    if (m_status_source)
    {
        auto [ack, ack_time] = m_status_source();
        if (ack.session_id != 0)
        {
            auto age_s = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - ack_time).count();
            m_logger->info("[Colin]  🔐 ── Node Lane Health (SESSION_STATUS_ACK, {}s ago) ──", age_s);
            m_logger->info("[Colin]    Primary alive:   {}", ack.IsPrimaryAlive()   ? "✅" : "❌");
            m_logger->info("[Colin]    Secondary alive: {}", ack.IsSecondaryAlive() ? "✅" : "❌");
            m_logger->info("[Colin]    SIM Link active: {}", ack.IsSimLinkActive()  ? "✅" : "❌");
            m_logger->info("[Colin]    Authenticated:   {}", ack.IsAuthenticated()  ? "✅" : "❌");
            m_logger->info("[Colin]    Node uptime:     {}s", ack.uptime_seconds);
            if (age_s > DIAG_SESSION_STATUS_ACK_STALE_SECONDS)
                warnings.push_back("No SESSION_STATUS_ACK for >" + std::to_string(age_s) +
                                   "s — node may have dropped session or lane is silent");
        }
    }

    if (gs.m_degraded_mode)
        m_logger->warn("[Colin]  ❌ MINING STOPPED — workers in degraded mode");
    if (m_height_tracker)
    {
        m_logger->info("[Colin]  📊 Heights │ unified_tip={} channel_tip={}  (BLOCK_DATA canonical)  prime={} hash={} stake={} (diagnostic)",
            ht_snap.unified_height,
            ht_snap.channel_height,
            ht_snap.prime_height,
            ht_snap.hash_height,    ht_snap.stake_height);
        if (ht_snap.push_prime_height > 0 || ht_snap.push_hash_height > 0) {
            m_logger->info("[Colin]  📡 Push Heights │ prime={} hash={} stake={}  (from 148-byte BLOCK_AVAILABLE)",
                ht_snap.push_prime_height, ht_snap.push_hash_height, ht_snap.push_stake_height);
        }
        if (ht_snap.peak_fork_score > 0)
            m_logger->warn("[Colin]  🔱 FORK CANARY active: peak_fork_score={} current_fork_score={}",
                ht_snap.peak_fork_score, ht_snap.fork_score);

        // Template feed source: confirms whether BLOCK_DATA metadata feed is working
        if (ht_snap.channel_target > 0) {
            const char* src = "NONE";
            const char* label = "";
            switch (ht_snap.last_update_source) {
                case nexusminer::protocol::HeightTracker::UpdateSource::PUSH:
                    src = "PUSH"; label = " (BLOCK_DATA metadata)"; break;
                case nexusminer::protocol::HeightTracker::UpdateSource::GET_ROUND:
                    src = "GET_ROUND"; label = " (fallback)"; break;
                case nexusminer::protocol::HeightTracker::UpdateSource::TEMPLATE:
                    src = "TEMPLATE"; label = " (BLOCK_DATA metadata → template received)"; break;
                case nexusminer::protocol::HeightTracker::UpdateSource::KEEPALIVE:
                    src = "KEEPALIVE"; label = " (keepalive fallback)"; break;
                default: break;
            }
            m_logger->info("[Colin]  📦 Template Feed │ last_update_source={}{} channel_target={}",
                src, label, ht_snap.channel_target);
        }
    }

    // Keepalive ACK health (Gap 4 — visibility into silent-death scenario)
    if (m_height_tracker) {
        if (ht_snap.last_keepalive_ack_at != std::chrono::steady_clock::time_point{}) {
            auto keepalive_age_s = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - ht_snap.last_keepalive_ack_at).count();
            if (keepalive_age_s < DIAG_KEEPALIVE_ACK_STALE_SECONDS) {
                m_logger->info("[Colin]  💓 Keepalive │ last ACK {}s ago ✅", keepalive_age_s);
            } else {
                m_logger->warn("[Colin]  💓 Keepalive │ last ACK {}s ago ⚠️  (>{}s — silent death risk)", keepalive_age_s, DIAG_KEEPALIVE_ACK_STALE_SECONDS);
            }
        } else {
            m_logger->info("[Colin]  💓 Keepalive │ no ACK received yet (session just started or legacy node)");
        }
    }

    // ── Failover Node Status ──────────────────────────────────────────────────
    if (m_failover_source) {
        auto fs = m_failover_source();
        if (fs.has_failover_configured) {
            if (!fs.using_failover) {
                m_logger->info("[Colin]  🟢 FailoverNode │ PRIMARY active: {}  │  standby: {}  │  fails: {}/{}",
                    fs.active_endpoint_str, fs.standby_endpoint_str,
                    fs.primary_fail_count, fs.failover_max_retries);
                if (!fs.secondary_ip.empty()) {
                    m_logger->info("[Colin]  🔗 SIM Link    │ Secondary lane IP: {}", fs.secondary_ip);
                }
            } else {
                m_logger->warn("[Colin]  🔀 FailoverNode │ ⚠️  FAILOVER ACTIVE: {}  │  primary DOWN: {}  │  active for: {}s",
                    fs.active_endpoint_str, fs.standby_endpoint_str,
                    fs.failover_active_seconds);
                if (!fs.secondary_ip.empty()) {
                    m_logger->info("[Colin]  🔗 SIM Link    │ Secondary lane IP: {}", fs.secondary_ip);
                }
                if (fs.failover_active_seconds > 300)
                    m_logger->warn("[Colin]  🔀 FailoverNode │ Extended failover: primary unreachable for {}min",
                        fs.failover_active_seconds / 60);
            }
        } else {
            m_logger->debug("[Colin]  FailoverNode │ not configured (single-node mode)");
        }
    }

    // ── Canonical Chain State section (collapsed: hashPrevBlock + nBits only) ──
    if (m_height_tracker) {
        m_logger->info("[Colin]  🔗 ── Canonical Chain State ────────────────────────");
        if (canonical.is_initialized()) {
            if (canonical.canonical_hash_prev_block != uint1024_t{}) {
                std::string canonical_hex = canonical.canonical_hash_prev_block.GetHex();
                const char* canon_color = ANSI_RESET;
                for (size_t i = 0; i < m_prev_hash_history.size(); ++i) {
                    size_t idx = m_prev_hash_history.size() - 1 - i;  // newest first
                    if (m_prev_hash_history[idx] == canonical_hex) {
                        canon_color = HASH_COLORS[i];
                        break;
                    }
                }
                m_logger->info("[Colin]    ✅ Canonical │ hashPrevBlock={}{}{}",
                    canon_color, canonical_hex, ANSI_RESET);
            }
            m_logger->info("[Colin]    ✅ Canonical │ nBits=0x{:08x}", canonical.canonical_difficulty_nbits);
        } else {
            m_logger->info("[Colin]    💤 Canonical │ not yet initialized (no BLOCK_DATA received)");
        }
    }

    // ── Tip Hash Anchor ──────────────────────────────────────────────────────
    // Full 128-byte hashPrevBlock comparison: canonical (BLOCK_DATA anchor) vs
    // template snapshot. These must agree for the miner to be on the correct tip.
    // This is the initial implementation — full fork resolution to follow.
    if (m_height_tracker) {
        m_logger->info("[Colin]  ⚓ ── Tip Hash Anchor ────────────────────────────");
        if (!canonical.is_initialized()) {
            m_logger->info("[Colin]    💤 Tip Hash Anchor │ canonical not yet initialized (no BLOCK_DATA received)");
        } else if (ht_snap.hash_prev_block == uint1024_t{}) {
            m_logger->info("[Colin]    💤 Tip Hash Anchor │ no template hashPrevBlock yet (waiting for first BLOCK_DATA)");
        } else {
            // Show both anchors (short preview only — full hash in Canonical Chain State section)
            m_logger->info("[Colin]    ⚓ Canonical │ {}  (unified={}, from BLOCK_DATA)",
                format_hash_anchor(canonical.canonical_hash_prev_block),
                canonical.canonical_unified_height);
            m_logger->info("[Colin]    ⚓ Template  │ {}  (from last template)",
                format_hash_anchor(ht_snap.hash_prev_block));

            if (canonical.canonical_hash_prev_block == ht_snap.hash_prev_block) {
                m_logger->info("[Colin]    ✅ Tip Hash Anchor │ in sync — template and canonical agree on chain tip");
            } else {
                m_logger->warn("[Colin]    ⚠️  Tip Hash Anchor │ TIP DIVERGED — template hashPrevBlock != canonical (full 128-byte mismatch)");
                m_logger->warn("[Colin]    ⚠️  Tip Hash Anchor │ Fork resolution needed — miner template is behind the canonical tip");
                warnings.push_back("Tip Hash Anchor: template hashPrevBlock != canonical — miner may be on stale/forked tip");
            }
        }
    }

    /* Canonical Drift section — uses height_drift_from_canonical() to detect
     * when push/round heights are running ahead of the canonical BLOCK_DATA path.
     * Also reports DiagnosticObserverState::is_initialized() and latest_received_at()
     * so operators can see whether all three diagnostic sources are feeding data. */
    if (m_height_tracker)
    {
        m_logger->info("[Colin]  📊 ── Canonical vs Diagnostic State ──────────────");

        // Canonical initialization status
        if (canonical.is_initialized()) {
            auto canonical_age_s = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - canonical.canonical_received_at).count();
            m_logger->info("[Colin]    ✅ Canonical │ initialized  unified={} channel={} target={} ({}s ago)",
                canonical.canonical_unified_height, canonical.canonical_channel_height,
                canonical.canonical_channel_target, canonical_age_s);
        } else {
            m_logger->warn("[Colin]    ⚠️  Canonical │ NOT initialized (no BLOCK_DATA received yet)");
        }

        // Diagnostic initialization status
        if (diag.is_initialized()) {
            auto diag_latest = diag.latest_received_at();
            auto diag_age_s = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - diag_latest).count();
            m_logger->info("[Colin]  📡 ── Diagnostic Observer State ─────────────────────");
            m_logger->info("[Colin]    ✅ Diagnostic │ initialized  push_unified={} round_unified={} keepalive_unified={} (latest {}s ago)",
                diag.push_unified_height, diag.round_unified_height,
                diag.keepalive_unified_height, diag_age_s);
            if (diag.round_unified_height > 0) {
                m_logger->info("[Colin]    📡 GET_ROUND heights │ prime={} hash={} stake={}",
                    diag.round_prime_height, diag.round_hash_height, diag.round_stake_height);
            }
        } else {
            m_logger->info("[Colin]  📡 ── Diagnostic Observer State ─────────────────────");
            m_logger->warn("[Colin]    ⚠️  Diagnostic │ NOT initialized (no push/round/keepalive data yet)");
        }

        // Height drift: how far composed snapshot heights have drifted from canonical
        if (canonical.is_initialized()) {
            int32_t drift = ht_snap.height_drift_from_canonical();
            if (drift == 0) {
                m_logger->info("[Colin]    ✅ HeightDrift │ 0 (canonical caught up with push/round)");
            } else if (drift > 0 && drift <= WARN_CANONICAL_DRIFT_THRESHOLD) {
                m_logger->info("[Colin]    📊 HeightDrift │ +{} (push/round ahead — normal during BLOCK_DATA latency)", drift);
            } else if (drift > WARN_CANONICAL_DRIFT_THRESHOLD) {
                m_logger->warn("[Colin]    ⚠️  HeightDrift │ +{} (push/round significantly ahead — BLOCK_DATA may be delayed)", drift);
                auto w = check_canonical_drift(drift);
                if (!w.empty()) warnings.push_back(w);
            } else {
                // Negative drift should not happen (canonical > composed) — log as anomaly
                m_logger->warn("[Colin]    ❌ HeightDrift │ {} (anomaly — canonical ahead of composed snapshot)", drift);
            }
        }

        // Diagnostic staleness: warn if all diagnostic sources have gone silent
        if (diag.is_initialized()) {
            auto diag_latest = diag.latest_received_at();
            auto diag_age_s = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - diag_latest).count();
            auto w = check_diagnostic_freshness(static_cast<uint64_t>(diag_age_s), diag.is_initialized());
            if (!w.empty()) {
                m_logger->warn("[Colin]    DiagStale │ ⚠ no diagnostic update for {}s", diag_age_s);
                warnings.push_back(w);
            }
        }

        // ── Per-Source Height Dashboard (TIP/TARGET taxonomy) ──────────────
        // Shows each height source independently so operators can identify
        // which source is lagging or racing ahead.  This is the primary
        // troubleshooting tool for height regression tracking.
        m_logger->info("[Colin]  🗂️  ── Height Source Dashboard (TIP / TARGET) ──────────");
        m_logger->info("[Colin]    CANONICAL (BLOCK_DATA) │ unified_tip={}  channel_tip={}  channel_target={}",
            canonical.canonical_unified_height, canonical.canonical_channel_height,
            canonical.canonical_channel_target);
        m_logger->info("[Colin]    PUSH (BLOCK_AVAILABLE)  │ unified_tip={}  channel_tip={}",
            diag.push_unified_height, diag.push_channel_height);
        m_logger->info("[Colin]    GET_ROUND               │ unified_tip={}  channel_tip={}  prime={}  hash={}  stake={}",
            diag.round_unified_height, diag.round_channel_height,
            diag.round_prime_height, diag.round_hash_height, diag.round_stake_height);
        m_logger->info("[Colin]    KEEPALIVE               │ unified_tip={}  prime={}  hash={}  stake={}",
            diag.keepalive_unified_height,
            diag.keepalive_prime_height, diag.keepalive_hash_height, diag.keepalive_stake_height);

        // Trigger analysis: which sources are ahead of canonical?
        if (canonical.is_initialized() && diag.is_initialized()) {
            int32_t push_ahead = static_cast<int32_t>(diag.push_unified_height) -
                                  static_cast<int32_t>(canonical.canonical_unified_height);
            int32_t round_ahead = static_cast<int32_t>(diag.round_unified_height) -
                                   static_cast<int32_t>(canonical.canonical_unified_height);
            if (push_ahead > 0 || round_ahead > 0) {
                m_logger->info("[Colin]    TRIGGER DRIFT │ push_ahead={} round_ahead={} (waiting for BLOCK_DATA)",
                    push_ahead > 0 ? push_ahead : 0, round_ahead > 0 ? round_ahead : 0);
            } else {
                m_logger->info("[Colin]    TRIGGER DRIFT │ 0 (all sources in sync)");
            }
        }
    }

    if (!warnings.empty())
    {
        m_logger->warn("[Colin]  🚨 ── Warnings ──────────────────────────────────");
        for (const auto& w : warnings)
            m_logger->warn("[Colin]    ⚠️  • {}", w);
    }

    if (!recommendations.empty())
    {
        m_logger->info("[Colin]  💡 ── Recommendations ────────────────────────────");
        for (const auto& r : recommendations)
            m_logger->info("[Colin]    💡 • {}", r);
    }

    /* Colin AI Miner — Node Diagnostic Section (from last received PingFrame) */
    if (m_ping_source)
    {
        const auto ping = m_ping_source();
        if (ping.valid)
        {
            m_logger->info("[Colin]  📡 ── Node Diagnostics (last PING_DIAG) ──────────");
            m_logger->info("[Colin]    PING_DIAG Seq #{}", ping.sequence);

            /* Template push rates with drought warnings */
            if (ping.prime_pushes_30s == 0 && ping.hash_pushes_30s == 0)
                m_logger->error("[Colin]    Prime pushes/30s: 0   Hash pushes/30s: 0  ⚠ TOTAL DROUGHT");
            else if (ping.prime_pushes_30s == 0)
                m_logger->warn("[Colin]    Prime pushes/30s: 0 ⚠   Hash pushes/30s: {}",
                    ping.hash_pushes_30s);
            else if (ping.hash_pushes_30s == 0)
                m_logger->warn("[Colin]    Prime pushes/30s: {}   Hash pushes/30s: 0 ⚠",
                    ping.prime_pushes_30s);
            else
                m_logger->info("[Colin]    Prime pushes/30s: {}   Hash pushes/30s: {}",
                    ping.prime_pushes_30s, ping.hash_pushes_30s);

            /* Submission accept/reject rate */
            if (ping.blocks_submitted > 0)
            {
                uint32_t reject_pct = static_cast<uint32_t>(
                    static_cast<uint64_t>(ping.blocks_rejected) * 100u / ping.blocks_submitted);
                m_logger->info("[Colin]    Submissions: {} sent / {} accepted / {} rejected ({}%)",
                    ping.blocks_submitted, ping.blocks_accepted,
                    ping.blocks_rejected, reject_pct);
            }
            else
            {
                m_logger->info("[Colin]    Submissions: 0 sent (no submissions this interval)");
            }

            /* Decoded node health flags */
            if (ping.health_flags == 0)
            {
                m_logger->info("[Colin]    ✅ Node Health: OK");
            }
            else
            {
                /* Build human-readable flag list */
                std::string flag_str;
                auto append = [&](const char* name) {
                    if (!flag_str.empty()) flag_str += " | ";
                    flag_str += name;
                };
                if (ping.health_flags & ::LLP::ReceivedPingFrame::NFLAG_NODE_SYNCING)
                    append("NODE_SYNCING");
                if (ping.health_flags & ::LLP::ReceivedPingFrame::NFLAG_FIRST_CONNECT)
                    append("FIRST_CONNECT");
                if (ping.health_flags & ::LLP::ReceivedPingFrame::NFLAG_HIGH_REJECT_RATE)
                    append("HIGH_REJECT_RATE");
                if (ping.health_flags & ::LLP::ReceivedPingFrame::NFLAG_CHANNEL_MISMATCH)
                    append("CHANNEL_MISMATCH");
                if (ping.health_flags & ::LLP::ReceivedPingFrame::NFLAG_DEDUP_HIT)
                    append("DEDUP_HIT");
                if (ping.health_flags & ::LLP::ReceivedPingFrame::NFLAG_SIM_LINK_ACTIVE)
                    append("SIM_LINK_ACTIVE");
                if (ping.health_flags & ::LLP::ReceivedPingFrame::NFLAG_RATE_LIMITED)
                    append("RATE_LIMITED");
                if (ping.health_flags & ::LLP::ReceivedPingFrame::NFLAG_STALE_TEMPLATE)
                    append("STALE_TEMPLATE");
                m_logger->warn("[Colin]    ⚠️  Node Health: {}", flag_str);
            }
        }
    }

    /* Active Template section (BLOCK_DATA canonical source) */
    if (m_template_source)
    {
        auto ts = m_template_source();
        m_logger->info("[Colin]  📦 ── Active Template (BLOCK_DATA canonical) ─────");
        if (ts.has_valid_template)
        {
            m_logger->info("[Colin]    State:          {}", ts.state_name ? ts.state_name : "?");
            m_logger->info("[Colin]    unified_height: {}  (block.nHeight — canonical for ProofHash)", ts.unified_height);
            m_logger->info("[Colin]    nBits:          0x{:08x}", ts.nBits);
            m_logger->info("[Colin]    channel_height: {}  (nChannelHeight — staleness metadata only)", ts.channel_height);
            m_logger->info("[Colin]    channel:        {} ({})", ts.channel, ts.channel == 1 ? "Prime" : "Hash");
            m_logger->info("[Colin]    age:            {}s", ts.age_seconds);

            // Cross-check: compare template unified height with HeightTracker snapshot.
            // By design, template.block.nHeight == HeightTracker.unified_height + 1 (template
            // mines the NEXT block). A drift of 0 is expected.
            // A drift of -1 is normal during the inter-push interval (push advances unified_height
            // before the new template arrives). Only warn on |drift| > 1.
            if (m_height_tracker)
            {
                if (ht_snap.unified_height != 0 && ts.unified_height != 0)
                {
                    // Compare template.block.nHeight (TARGET) against canonical unified TIP + 1.
                    // Both sides are BLOCK_DATA canonical — drift should be 0 when in sync.
                    int64_t height_drift = static_cast<int64_t>(ts.unified_height)
                                         - static_cast<int64_t>(ht_snap.unified_height + 1);
                    if (height_drift == 0)
                    {
                        m_logger->debug("[Colin]    HEIGHT_DRIFT: none (canonical_unified_tip={} + 1 == template.nHeight={})",
                            ht_snap.unified_height, ts.unified_height);
                    }
                    else if (height_drift == -1)
                    {
                        m_logger->info("[Colin]    HEIGHT_DRIFT: -1 (canonical_unified_tip={} vs template.block.nHeight={} — BLOCK_DATA pending, normal inter-push interval)",
                            ht_snap.unified_height, ts.unified_height);
                    }
                    else
                    {
                        m_logger->warn("[Colin]    ⚠ HEIGHT_DRIFT: canonical_unified_tip={} vs template.block.nHeight={} (drift={}; expected 0 or -1)",
                            ht_snap.unified_height, ts.unified_height, height_drift);
                    }
                }
            }

            if (ts.age_seconds > WARN_TEMPLATE_AGE_SECONDS)
                m_logger->warn("[Colin]    ⏱️  TEMPLATE AGING: {}s — approaching emergency timeout", ts.age_seconds);
        }
        else
        {
            m_logger->warn("[Colin]  ❌ NO VALID TEMPLATE — workers have no work to do!");
        }
        m_logger->info("[Colin]    Lifetime — rcvd:{} valid:{} rejected:{} stale:{} fed:{} expiredAge:{} expiredHt:{}",
            ts.templates_received, ts.templates_validated, ts.templates_rejected,
            ts.templates_stale, ts.templates_fed,
            ts.templates_expired_age, ts.templates_expired_height);
    }

    /* Miner Telemetry section (from ColinPingHandler PONG outbound) */
    if (m_pong_telemetry_source)
    {
        auto pt = m_pong_telemetry_source();
        m_logger->info("[Colin]  📡 ── Miner Telemetry (PONG outbound) ─────────────");
        m_logger->info("[Colin]    Ping count: {}  Last RTT: {} µs", pt.ping_count, pt.last_rtt_us);
        if (pt.hash_rate_khs > 0)
            m_logger->info("[Colin]    Hash-rate: {} kH/s  Threads: {}  Queue: {}",
                pt.hash_rate_khs, pt.thread_count, pt.queue_depth);
        if (pt.temp_cdeg > 0)
            m_logger->info("[Colin]    Temp: {:.1f} °C", pt.temp_cdeg / 10.0f);
        if (pt.health_flags != 0)
            m_logger->warn("[Colin]    Health flags: 0x{:02x}", pt.health_flags);
    }

    /* Mined Block History — Top 5 hashPrevBlock cache (Tier 1) */
    if (m_mined_block_cache_source)
    {
        auto blocks = m_mined_block_cache_source();
        m_logger->info("[Colin]  🏆 ── Mined Block hashPrevBlock History (Top 5) ─────");
        if (blocks.empty())
        {
            if (gs.m_accepted_blocks == 0) {
                m_logger->info("[Colin]  💤   (no blocks mined yet)");
            } else {
                m_logger->warn("[Colin]  ⚠️    accepted blocks={} but detailed mined-block history is unavailable in the current cache",
                               gs.m_accepted_blocks);
            }
        }
        else
        {
            for (size_t i = 0; i < blocks.size(); ++i)
            {
                const auto& b = blocks[i];
                m_logger->info("[Colin]    {} #{} height={} channel={} confirmations={}",
                    b.status_emoji, i + 1, b.height, b.channel_name, b.confirmations);
                const char* blk_color = ANSI_RESET;
                for (size_t j = 0; j < m_prev_hash_history.size(); ++j) {
                    size_t history_idx = m_prev_hash_history.size() - 1 - j;
                    if (m_prev_hash_history[history_idx] == b.hash_prev_block_hex) {
                        blk_color = HASH_COLORS[j];
                        break;
                    }
                }
                m_logger->info("[Colin]       hashPrevBlock={}{}{}",
                    blk_color, b.hash_prev_block_hex, ANSI_RESET);
            }
        }
    }

    m_logger->info("{}[Colin] ⛏️  ═══════════════════════════════════════════════════ ⛏️{}", ANSI_BOLD, ANSI_RESET);
}

} // namespace nexusminer
