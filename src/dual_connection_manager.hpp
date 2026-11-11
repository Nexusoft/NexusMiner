#ifndef NEXUSMINER_DUAL_CONNECTION_MANAGER_HPP
#define NEXUSMINER_DUAL_CONNECTION_MANAGER_HPP

#include "protocol_lane.hpp"
#include <chrono>
#include <cstdint>
#include <string>
#include <utility>

namespace nexusminer
{

/// Lightweight coordinator that tracks live/dead state and one-shot bypass flags
/// for the two SIM Link lanes (stateless 9323 + legacy 8323).
///
/// This class does NOT own connections — Worker_manager owns the actual sockets
/// and connections.  DualConnectionManager only provides bookkeeping so that
/// Worker_manager can make the right forwarding/recovery decisions.
///
/// ## NO CROSS-LANE RULE
///
/// The miner's protocol lane is determined once at initial connection time (from the
/// remote port via determine_lane_from_port()) and is **immutable** for the session
/// lifetime.  All recovery and failover operations MUST use the same protocol lane.
///
/// - Primary retry:  same node, same lane, same port.
/// - Failover:       different node, same lane, same port — with full RE-AUTH sequence.
///
/// When a lane fails its bypass is armed on the **SAME** lane (not the opposite).
/// This prepares the reconnect path on the correct lane without ever crossing to the
/// other protocol.
class DualConnectionManager
{
public:
    DualConnectionManager() = default;

    // ── Lane liveness ────────────────────────────────────────────────────────
    void set_stateless_alive(bool alive)
    {
        m_stateless_alive = alive;
        if (alive) m_last_alive_at = std::chrono::steady_clock::now();
    }
    void set_legacy_alive(bool alive)
    {
        m_legacy_alive = alive;
        if (alive) m_last_alive_at = std::chrono::steady_clock::now();
    }

    bool is_stateless_alive() const { return m_stateless_alive; }
    bool is_legacy_alive()    const { return m_legacy_alive;    }
    bool any_lane_alive()     const { return m_stateless_alive || m_legacy_alive; }

    // ── Aggregate liveness heartbeat ─────────────────────────────────────────
    /// Timestamp of the most recent moment at least one lane was known-alive.
    /// Seeded to construction time (lanes are assumed alive until proven
    /// otherwise) and refreshed both on every alive-transition (see
    /// set_stateless_alive(true)/set_legacy_alive(true)/on_lane_recovered())
    /// and via an explicit periodic touch_alive_heartbeat() call from the
    /// caller's health-check tick, so it behaves like a heartbeat rather than
    /// a one-shot edge trigger. This feeds Worker_manager's worst-case outage
    /// aggregator (max of "since last valid template", "since last push", and
    /// "since any lane was alive") used to close compounding-failure gaps
    /// where connectivity and template staleness overlap.
    std::chrono::steady_clock::time_point last_alive_at() const { return m_last_alive_at; }

    /// Explicitly refresh the heartbeat if at least one lane is currently
    /// alive. Intended to be called on a periodic health-check tick so the
    /// heartbeat reflects sustained liveness, not just alive transitions.
    void touch_alive_heartbeat()
    {
        if (any_lane_alive())
        {
            m_last_alive_at = std::chrono::steady_clock::now();
        }
    }

    // ── Mining lane (immutable once set) ─────────────────────────────────────
    /// Set the lane the miner is actively mining on.  Called once during initial
    /// connection from determine_lane_from_port() and NEVER changed afterwards.
    /// All recovery and failover operations must respect this lane.
    void set_mining_lane(ProtocolLane lane) { m_mining_lane = lane; }

    /// Returns the lane the miner is actively mining on.
    ProtocolLane mining_lane() const { return m_mining_lane; }

    // ── One-shot bypass for GET_BLOCK rate limiter (same-lane recovery) ──────
    /// Arm a one-shot bypass for the given lane.
    /// The first call to consume_bypass() for that lane after arming returns true
    /// and clears the flag (reset after each use, one bypass per lane-failure event).
    void arm_bypass(ProtocolLane lane)
    {
        if (lane == ProtocolLane::STATELESS) m_stateless_bypass_armed = true;
        else                                 m_legacy_bypass_armed    = true;
    }

    /// Returns true and clears the flag if a one-shot bypass was armed for this lane.
    bool consume_bypass(ProtocolLane lane)
    {
        if (lane == ProtocolLane::STATELESS && m_stateless_bypass_armed)
        {
            m_stateless_bypass_armed = false;
            return true;
        }
        if (lane == ProtocolLane::LEGACY && m_legacy_bypass_armed)
        {
            m_legacy_bypass_armed = false;
            return true;
        }
        return false;
    }

    // ── Lane failure event ───────────────────────────────────────────────────
    /// Called when a lane drops (MALFORMED, connection_closed, etc.).
    ///
    /// Arms the bypass on the **SAME** lane so that when it reconnects it can
    /// request a fresh template immediately without triggering the node's rate
    /// limiter.
    ///
    /// IMPORTANT — NO CROSS-LANE: the bypass is NEVER armed on the opposite lane.
    /// Recovery always stays on the lane that failed.
    void on_lane_failed(ProtocolLane dead_lane)
    {
        if (dead_lane == ProtocolLane::STATELESS)
        {
            m_stateless_alive = false;
            arm_bypass(ProtocolLane::STATELESS);  // Same lane: ready for reconnect
        }
        else
        {
            m_legacy_alive = false;
            arm_bypass(ProtocolLane::LEGACY);     // Same lane: ready for reconnect
        }
    }

    /// Called when a previously-dead lane has re-authenticated and is ready.
    void on_lane_recovered(ProtocolLane recovered_lane)
    {
        if (recovered_lane == ProtocolLane::STATELESS)
            m_stateless_alive = true;
        else
            m_legacy_alive = true;
        m_last_alive_at = std::chrono::steady_clock::now();
    }

    // ── Failover tracking ────────────────────────────────────────────────────
    /// Update the current failover state and active node endpoint.
    /// @param active True if failover is active, false if primary is active
    /// @param endpoint The current active node IP address (e.g., "192.168.1.10")
    ///
    /// NOTE: Failover changes the NODE endpoint only — the protocol lane (m_mining_lane)
    /// never changes.  The failover node must be contacted on the same port/lane as the
    /// primary.
    void set_failover_active(bool active, std::string endpoint)
    {
        m_using_failover = active;
        m_active_node_ip = std::move(endpoint);
    }

    bool is_using_failover() const { return m_using_failover; }
    std::string const& get_active_node_ip() const { return m_active_node_ip; }

private:
    bool m_stateless_alive{false};
    bool m_legacy_alive{false};

    // Seeded to construction time so a freshly-started miner (no lane has
    // failed yet) is not immediately treated as having a long-running outage
    // by the worst-case outage aggregator; see last_alive_at() above.
    std::chrono::steady_clock::time_point m_last_alive_at{std::chrono::steady_clock::now()};

    // The lane the miner is actively mining on — set once at connection time, never changed.
    ProtocolLane m_mining_lane{ProtocolLane::UNKNOWN};

    // One-shot bypass flags (per-lane; armed on failure of the SAME lane)
    bool m_stateless_bypass_armed{false};
    bool m_legacy_bypass_armed{false};

    // Failover state (node endpoint changes; lane never changes)
    bool m_using_failover{false};
    std::string m_active_node_ip;
};

} // namespace nexusminer

#endif // NEXUSMINER_DUAL_CONNECTION_MANAGER_HPP
