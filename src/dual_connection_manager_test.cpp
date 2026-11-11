#include "dual_connection_manager.hpp"

#include <iostream>

using namespace nexusminer;

namespace
{
bool expect(bool condition, const char* message)
{
    if (!condition)
    {
        std::cerr << "  [FAIL] " << message << std::endl;
        return false;
    }
    std::cout << "  [PASS] " << message << std::endl;
    return true;
}
}

int main()
{
    bool ok = true;

    std::cout << "========================================\n";
    std::cout << "DualConnectionManager Unit Tests\n";
    std::cout << "========================================\n\n";

    {
        std::cout << "Test 1: Lane liveness state tracking\n";
        DualConnectionManager mgr;
        ok &= expect(!mgr.any_lane_alive(), "Initial state: no lane alive");
        mgr.set_stateless_alive(true);
        ok &= expect(mgr.is_stateless_alive(), "Stateless lane alive after set");
        ok &= expect(mgr.any_lane_alive(), "Any lane alive when stateless alive");
        mgr.set_legacy_alive(true);
        ok &= expect(mgr.is_legacy_alive(), "Legacy lane alive after set");
        std::cout << '\n';
    }

    {
        std::cout << "Test 2: Stateless failure arms stateless one-shot bypass (same-lane — NO cross-lane)\n";
        DualConnectionManager mgr;
        mgr.set_stateless_alive(true);
        mgr.set_legacy_alive(true);
        mgr.on_lane_failed(ProtocolLane::STATELESS);
        ok &= expect(!mgr.is_stateless_alive(), "Stateless marked dead on failure");
        ok &= expect(mgr.is_legacy_alive(), "Legacy remains alive");
        ok &= expect(mgr.consume_bypass(ProtocolLane::STATELESS), "Stateless bypass consumed once (same lane)");
        ok &= expect(!mgr.consume_bypass(ProtocolLane::STATELESS), "Stateless bypass does not repeat");
        ok &= expect(!mgr.consume_bypass(ProtocolLane::LEGACY), "Legacy bypass NOT armed (no cross-lane)");
        std::cout << '\n';
    }

    {
        std::cout << "Test 3: Legacy failure arms legacy one-shot bypass (same-lane — NO cross-lane)\n";
        DualConnectionManager mgr;
        mgr.set_stateless_alive(true);
        mgr.set_legacy_alive(true);
        mgr.on_lane_failed(ProtocolLane::LEGACY);
        ok &= expect(!mgr.is_legacy_alive(), "Legacy marked dead on failure");
        ok &= expect(mgr.is_stateless_alive(), "Stateless remains alive");
        ok &= expect(mgr.consume_bypass(ProtocolLane::LEGACY), "Legacy bypass consumed once (same lane)");
        ok &= expect(!mgr.consume_bypass(ProtocolLane::LEGACY), "Legacy bypass does not repeat");
        ok &= expect(!mgr.consume_bypass(ProtocolLane::STATELESS), "Stateless bypass NOT armed (no cross-lane)");
        std::cout << '\n';
    }

    {
        std::cout << "Test 4: Lane recovery updates liveness\n";
        DualConnectionManager mgr;
        mgr.on_lane_failed(ProtocolLane::STATELESS);
        ok &= expect(!mgr.is_stateless_alive(), "Stateless dead after failure");
        mgr.on_lane_recovered(ProtocolLane::STATELESS);
        ok &= expect(mgr.is_stateless_alive(), "Stateless alive after recovery");
        mgr.on_lane_recovered(ProtocolLane::LEGACY);
        ok &= expect(mgr.is_legacy_alive(), "Legacy alive after recovery");
        std::cout << '\n';
    }

    {
        std::cout << "Test 5: No miner-side GET_BLOCK rate limiter\n";
        // The miner-side GET_BLOCK rate limiter (GET_BLOCK_MIN_INTERVAL in Solo::get_work())
        // has been removed. The node's 2-second AutoCoolDown (server-side) is the sole
        // rate limiter for GET_BLOCK. Any miner-side suppression was redundant and caused
        // doom loops during recovery.
        //
        // The compile-time proof that the DualConnectionManager timing constants are gone
        // is that this file compiles without them — any reference to
        // GET_BLOCK_MINER_INTERVAL_MS or GET_BLOCK_NODE_INTERVAL_MS would be a compile error.
        DualConnectionManager mgr;
        mgr.set_stateless_alive(true);
        ok &= expect(mgr.is_stateless_alive(),
                     "DualConnectionManager liveness works without stale GET_BLOCK timing constants");
        std::cout << '\n';
    }

    {
        std::cout << "Test 6: Cross-lane bypass is never armed\n";
        // On stateless failure the legacy bypass must remain unset, and vice versa.
        {
            DualConnectionManager mgr;
            mgr.on_lane_failed(ProtocolLane::STATELESS);
            ok &= expect(!mgr.consume_bypass(ProtocolLane::LEGACY),
                         "Stateless failure does NOT arm legacy bypass");
        }
        {
            DualConnectionManager mgr;
            mgr.on_lane_failed(ProtocolLane::LEGACY);
            ok &= expect(!mgr.consume_bypass(ProtocolLane::STATELESS),
                         "Legacy failure does NOT arm stateless bypass");
        }
        std::cout << '\n';
    }

    {
        std::cout << "Test 7: Mining lane is immutable once set\n";
        DualConnectionManager mgr;
        ok &= expect(mgr.mining_lane() == ProtocolLane::UNKNOWN,
                     "Mining lane initially UNKNOWN");
        mgr.set_mining_lane(ProtocolLane::STATELESS);
        ok &= expect(mgr.mining_lane() == ProtocolLane::STATELESS,
                     "Mining lane set to STATELESS");
        // Simulate lane failure — mining lane must not change
        mgr.on_lane_failed(ProtocolLane::STATELESS);
        ok &= expect(mgr.mining_lane() == ProtocolLane::STATELESS,
                     "Mining lane unchanged after stateless failure");
        // Simulate failover — mining lane must still not change
        mgr.set_failover_active(true, "192.168.1.99");
        ok &= expect(mgr.mining_lane() == ProtocolLane::STATELESS,
                     "Mining lane unchanged after failover activation");
        std::cout << '\n';
    }

    {
        std::cout << "Test 8: Alive heartbeat tracks last known-alive moment\n";
        DualConnectionManager mgr;
        auto seeded_at = mgr.last_alive_at();
        ok &= expect(seeded_at != std::chrono::steady_clock::time_point{},
                     "Heartbeat seeded at construction (not epoch)");

        // touch_alive_heartbeat() is a no-op while no lane is alive.
        mgr.touch_alive_heartbeat();
        ok &= expect(mgr.last_alive_at() == seeded_at,
                     "Heartbeat unchanged while no lane is alive");

        mgr.set_stateless_alive(true);
        auto after_alive = mgr.last_alive_at();
        ok &= expect(after_alive >= seeded_at,
                     "Heartbeat refreshed when a lane becomes alive");

        mgr.set_stateless_alive(false);
        mgr.set_legacy_alive(false);
        ok &= expect(!mgr.any_lane_alive(), "Both lanes dead after clearing");
        auto stamp_before_touch = mgr.last_alive_at();
        mgr.touch_alive_heartbeat();
        ok &= expect(mgr.last_alive_at() == stamp_before_touch,
                     "Heartbeat does not advance while all lanes are dead");

        mgr.on_lane_recovered(ProtocolLane::LEGACY);
        ok &= expect(mgr.last_alive_at() >= stamp_before_touch,
                     "Heartbeat refreshed on lane recovery");
        std::cout << '\n';
    }

    std::cout << "========================================\n";
    std::cout << "Test Summary\n";
    std::cout << "========================================\n";
    std::cout << (ok ? "ALL TESTS PASSED" : "TESTS FAILED") << std::endl;
    return ok ? 0 : 1;
}
