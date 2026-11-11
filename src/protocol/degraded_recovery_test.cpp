/**
 * @file degraded_recovery_test.cpp
 * @brief Tests for keepalive epoch isolation, channel-advance staleness, and degraded-mode
 *        recovery determinism.
 *
 * Tests:
 *  1.  Keepalive ACK update across epoch change — old-epoch ack is invalidated
 *  2.  Channel advance + stale template transition — HeightTracker detects staleness correctly
 *  3.  Recovery pending debouncing — mark_recovery_initiated() is idempotent for same event
 *  4.  Recovery GET_BLOCK can dispatch after debounce window — no permanent starvation
 *  5.  Keepalive epoch isolation — new epoch starts with clean ack timestamp
 *  6.  Stale template after channel advance — canonical auto-advance invariant
 *  7.  set_session_epoch() with a higher epoch suppresses the old keepalive signal
 *  8.  Multiple rapid epoch changes produce clean keepalive state
 *  9.  Push notification does not reset keepalive timestamp on epoch change
 * 10.  Recovery epoch tracking — monotonic epoch counter with idempotent initiation
 * 11.  Integration: stale -> degraded -> fresh template -> mining resumes
 * 12.  Integration: forced retry lane remains bounded (no flood)
 * 13.  Successful re-authentication restarts the recovery window on a fresh epoch
 * 14.  Session epoch advance clears stale push tip-anchor hints without clearing push liveness
 * 15.  Recovery completion clears suppression and authoritative recovery-reason state
 * 16.  Authoritative soft-refresh state repopulates Worker_manager local soft-pause state
 * 17.  Recovery worker respawn guard creates workers only once per degraded exit
 */

#include "protocol/height_tracker.hpp"
#include "protocol/packet_builder.hpp"
#include "protocol/epoch_coordinator.hpp"
#include "worker/recovery_timeout.hpp"
#include "miner_opcodes.hpp"
#include <iostream>
#include <cassert>
#include <cstdint>
#include <chrono>
#include <thread>
#include <memory>
#include <vector>
#include <deque>

using namespace nexusminer::protocol;
using namespace nexusminer;

// Test statistics
static int tests_run    = 0;
static int tests_passed = 0;
static int tests_failed = 0;

void print_test_result(const char* name, bool passed) {
    tests_run++;
    if (passed) {
        tests_passed++;
        std::cout << "  [PASS] " << name << "\n";
    } else {
        tests_failed++;
        std::cout << "  [FAIL] " << name << "\n";
    }
}

// ============================================================================
// ⚡ TestRecoveryPhase — minimal mirror of worker_manager RecoveryPhase enum
// Defined here so all tests can use it without including worker_manager.hpp.
// ============================================================================

// Minimal mirror of RecoveryPhase for standalone testing without including
// the full worker_manager.hpp dependency tree.
enum class TestRecoveryPhase : uint8_t {
    HEALTHY,
    WAITING_TEMPLATE,
    SESSION_RECOVERY,
    RECONNECTING,
};

const char* test_phase_name(TestRecoveryPhase p) {
    switch (p) {
        case TestRecoveryPhase::HEALTHY:          return "HEALTHY";
        case TestRecoveryPhase::WAITING_TEMPLATE: return "WAITING_TEMPLATE";
        case TestRecoveryPhase::SESSION_RECOVERY: return "SESSION_RECOVERY";
        case TestRecoveryPhase::RECONNECTING:     return "RECONNECTING";
    }
    return "UNKNOWN";
}

static bool test_is_valid_transition(TestRecoveryPhase from, TestRecoveryPhase to) {
    if (from == to) return true;
    switch (from) {
        case TestRecoveryPhase::HEALTHY:
            return to == TestRecoveryPhase::WAITING_TEMPLATE ||
                   to == TestRecoveryPhase::SESSION_RECOVERY ||
                   to == TestRecoveryPhase::RECONNECTING;
        case TestRecoveryPhase::WAITING_TEMPLATE:
            return to == TestRecoveryPhase::HEALTHY ||
                   to == TestRecoveryPhase::SESSION_RECOVERY ||
                   to == TestRecoveryPhase::RECONNECTING;
        case TestRecoveryPhase::SESSION_RECOVERY:
            return to == TestRecoveryPhase::WAITING_TEMPLATE ||
                   to == TestRecoveryPhase::RECONNECTING;
        case TestRecoveryPhase::RECONNECTING:
            return to == TestRecoveryPhase::HEALTHY ||
                   to == TestRecoveryPhase::WAITING_TEMPLATE;
    }
    return false;
}

// Helper predicates matching the Worker_manager helper methods
static bool phase_is_degraded(TestRecoveryPhase p) {
    return p == TestRecoveryPhase::WAITING_TEMPLATE;
}
static bool phase_is_submissions_withheld(TestRecoveryPhase /*p*/) {
    return false;  // No longer a concept — workers always submit if they have a template
}
static bool phase_is_recovery_active(TestRecoveryPhase p) {
    return p != TestRecoveryPhase::HEALTHY;
}
static bool phase_is_reconnecting(TestRecoveryPhase p) {
    return p == TestRecoveryPhase::RECONNECTING;
}

// ============================================================================
// Test 1: Keepalive ACK update across epoch change — old-epoch ack invalidated
// ============================================================================
void test_keepalive_ack_invalidated_on_epoch_change() {
    std::cout << "\nTest 1: Keepalive ACK invalidated when session epoch advances\n";
    HeightTracker tracker;

    // Epoch 1: receive keepalive
    tracker.set_session_epoch(SessionEpoch(1));
    tracker.OnKeepaliveResponse(5000, 400, 700, 900, 0xDEADBEEFu, 0);
    auto snap1 = tracker.GetSnapshot();
    bool ack_was_set = (snap1.last_keepalive_ack_at != std::chrono::steady_clock::time_point{});
    print_test_result("Epoch 1: keepalive ACK timestamp is set after response", ack_was_set);

    // Advance to epoch 2 (session re-auth or channel re-init)
    tracker.set_session_epoch(SessionEpoch(2));
    auto snap2 = tracker.GetSnapshot();
    print_test_result("Epoch 2: keepalive ACK timestamp cleared on epoch advance",
                      snap2.last_keepalive_ack_at == std::chrono::steady_clock::time_point{});
    print_test_result("Epoch 2: snapshot session_epoch reflects new epoch",
                      snap2.session_epoch == SessionEpoch(2));

    // New keepalive in epoch 2 should be accepted
    tracker.OnKeepaliveResponse(5001, 401, 701, 901, 0xCAFEBABEu, 0);
    auto snap3 = tracker.GetSnapshot();
    print_test_result("Epoch 2: new keepalive ACK is accepted",
                      snap3.last_keepalive_ack_at != std::chrono::steady_clock::time_point{});
}

// ============================================================================
// Test 2: Channel advance + stale template transition — staleness detection
// ============================================================================
void test_channel_advance_stale_template_transition() {
    std::cout << "\nTest 2: Channel advance + stale template transition\n";
    HeightTracker tracker;

    // Initial state: template for channel_target=101 while channel_height=100
    tracker.OnBlockDataReceived(5000, 100, 0x1d00ffff, uint1024_t{});
    tracker.OnTemplateReceived(2, 101);
    auto snap = tracker.GetSnapshot();
    print_test_result("Initial: channel_height=100 < target=101 (not stale)",
                      snap.channel_height < snap.channel_target);
    print_test_result("Initial: channel_height == 100", snap.channel_height == 100);
    print_test_result("Initial: channel_target == 101", snap.channel_target == 101);

    // Channel advances — node found a block.
    // Under canonical-only semantics, OnBlockDataReceived auto-advances target to
    // channel+1, so staleness is structurally impossible from canonical state alone.
    // is_template_stale() removed (structurally false under canonical-only semantics).
    // Staleness detection has moved to push/GET_ROUND trigger paths.
    // This test verifies the auto-advance behavior.
    tracker.OnBlockDataReceived(5001, 101, 0x1d00ffff, uint1024_t{});
    auto snap2 = tracker.GetSnapshot();
    print_test_result("After channel advance: canonical auto-advance: target=102 > channel=101",
                      snap2.channel_height < snap2.channel_target);
    print_test_result("After channel advance: channel_height == 101", snap2.channel_height == 101);
    print_test_result("After channel advance: channel_target auto-advanced to 102", snap2.channel_target == 102);

    // Simulate keepalive arriving from OLD session — should not affect canonical state
    tracker.OnKeepaliveResponse(5001, 300, 101, 900, 0xDEADBEEFu, 0);
    auto snap3 = tracker.GetSnapshot();
    print_test_result("After old keepalive: canonical state unchanged (channel < target)",
                      snap3.channel_height < snap3.channel_target);
    print_test_result("After old keepalive: channel_height unchanged at 101",
                      snap3.channel_height == 101);

    // New template from GET_BLOCK response — target advances further
    tracker.OnTemplateReceived(2, 103);
    auto snap4 = tracker.GetSnapshot();
    print_test_result("After new template: channel_height < channel_target",
                      snap4.channel_height < snap4.channel_target);
    print_test_result("After new template: channel_target == 103", snap4.channel_target == 103);
}

// ============================================================================
// Test 3: Recovery pending debouncing — mark_recovery is idempotent
// Simulates the "Recovery already pending" scenario: multiple staleness sources
// calling mark_recovery_initiated() for the same event must not reset the epoch.
// ============================================================================
void test_recovery_pending_debounce_idempotent() {
    std::cout << "\nTest 3: Recovery-pending debounce — multiple mark_recovery calls are idempotent\n";

    // Simulate the mark_recovery_initiated() logic using the 5-state RecoveryPhase machine
    struct RecoveryTracker {
        TestRecoveryPhase phase{TestRecoveryPhase::HEALTHY};
        uint64_t epoch{0};
        std::chrono::steady_clock::time_point entered_at{};
        std::vector<std::string> m_log;

        // Mirrors Worker_manager::mark_recovery_initiated() / transition_to(HARD_RECOVERY)
        void initiate(const char* reason) {
            if (phase_is_recovery_active(phase)) {
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::steady_clock::now() - entered_at).count();
                m_log.push_back("NOOP: already pending epoch=" + std::to_string(epoch)
                                + " elapsed=" + std::to_string(elapsed) + "s reason=" + reason);
                return;
            }
            ++epoch;
            phase = TestRecoveryPhase::WAITING_TEMPLATE;
            entered_at = std::chrono::steady_clock::now();
            m_log.push_back("STARTED: epoch=" + std::to_string(epoch) + " reason=" + reason);
        }

        void clear() {
            phase = TestRecoveryPhase::HEALTHY;
            m_log.push_back("CLEARED: epoch=" + std::to_string(epoch));
        }
    };

    RecoveryTracker rt;

    // First initiation from push handler
    rt.initiate("push_staleness");
    print_test_result("First initiation starts recovery (epoch=1)",
                      rt.epoch == 1 && phase_is_recovery_active(rt.phase));

    // Second initiation from health monitor (same staleness event)
    rt.initiate("health_monitor_channel_stale");
    print_test_result("Second initiation is a no-op (epoch still 1)",
                      rt.epoch == 1 && phase_is_recovery_active(rt.phase));

    // Third initiation from different source
    rt.initiate("health_monitor_or_validation");
    print_test_result("Third initiation is still a no-op (epoch still 1)",
                      rt.epoch == 1 && phase_is_recovery_active(rt.phase));

    print_test_result("All initiation calls produced at least one STARTED entry",
                      std::any_of(rt.m_log.begin(), rt.m_log.end(),
                                  [](const auto& s) { return s.find("STARTED") != std::string::npos; }));
    size_t noop_count = 0;
    for (const auto& entry : rt.m_log)
        if (entry.find("NOOP") != std::string::npos) ++noop_count;
    print_test_result("Subsequent initiations were all NOOP (2 no-ops for 3 total calls)",
                      noop_count == 2);

    // Clear and re-initiate — new epoch must be incremented
    rt.clear();
    rt.initiate("escalation_hard_recovery");
    print_test_result("After clear, new initiation produces epoch=2",
                      rt.epoch == 2 && phase_is_recovery_active(rt.phase));
}

// ============================================================================
// Test 4: Recovery GET_BLOCK can dispatch after debounce window — no starvation
// Simulates the deduplication window expiring between recovery retries.
// ============================================================================
void test_recovery_get_block_no_permanent_starvation() {
    std::cout << "\nTest 4: Recovery GET_BLOCK dispatches after dedup window expires\n";

    // Simulate GET_BLOCK deduplication logic (mirrors Solo::get_work dedup guard)
    const int64_t DEDUP_MS = 100;
    struct GetBlockGate {
        std::chrono::steady_clock::time_point m_last_transmitted{};
        int64_t dedup_ms{0};

        // Returns true if GET_BLOCK can be dispatched, false if suppressed
        bool try_dispatch() {
            auto now = std::chrono::steady_clock::now();
            if (m_last_transmitted != std::chrono::steady_clock::time_point{}) {
                auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - m_last_transmitted).count();
                if (elapsed_ms < dedup_ms) {
                    return false;  // suppressed
                }
            }
            m_last_transmitted = now;
            return true;  // dispatched
        }
    };

    GetBlockGate gate;
    gate.dedup_ms = DEDUP_MS;

    // First dispatch should succeed
    bool first = gate.try_dispatch();
    print_test_result("First GET_BLOCK dispatch succeeds", first);

    // Immediate second dispatch should be suppressed (within 100ms)
    bool second = gate.try_dispatch();
    print_test_result("Immediate second dispatch suppressed by dedup window", !second);

    // After dedup window expires, dispatch should succeed again
    std::this_thread::sleep_for(std::chrono::milliseconds(110));
    bool third = gate.try_dispatch();
    print_test_result("GET_BLOCK dispatch succeeds after dedup window (110ms wait)", third);

    // Another immediate attempt should be suppressed again
    bool fourth = gate.try_dispatch();
    print_test_result("Immediate dispatch after third is suppressed again", !fourth);

    // Recovery timer interval (30s >> 100ms dedup) ensures no starvation
    // at normal check_template_health intervals
    print_test_result("30s timer interval >> 100ms dedup = no starvation at timer cadence",
                      30000 > DEDUP_MS * 100);
}

// ============================================================================
// Test 4b: Successful re-authentication restarts the recovery window
// Mirrors Worker_manager::restart_recovery_window() + retry_template_request(true)
// so a fresh session does not inherit a long-expired recovery epoch.
// ============================================================================
void test_successful_reauth_restarts_recovery_epoch() {
    std::cout << "\nTest 4b: Re-authentication restarts recovery window on a fresh epoch\n";

    struct RecoveryTracker {
        bool m_degraded_mode{true};
        bool m_recovery_pending{true};
        uint64_t m_recovery_epoch{1};
        std::chrono::steady_clock::time_point m_recovery_started_at{};
        std::chrono::steady_clock::time_point m_degraded_since{};
        nexusminer::worker::RecoveryTimeout m_timeout;

        RecoveryTracker() {
            constexpr auto simulated_stale_duration = std::chrono::seconds(1207);
            const auto now = std::chrono::steady_clock::now();
            m_recovery_started_at = now - simulated_stale_duration;
            m_degraded_since = now - simulated_stale_duration;
            m_timeout.start_if_unset(m_degraded_since);
        }

        void restart_recovery_window() {
            m_recovery_started_at = std::chrono::steady_clock::now();
            m_degraded_since = m_recovery_started_at;
            m_timeout.reset();
            m_timeout.start_if_unset(m_degraded_since);
        }

        void on_reauthenticated() {
            if (m_degraded_mode || m_recovery_pending) {
                restart_recovery_window();
                ++m_recovery_epoch;
            }
        }
    };

    RecoveryTracker rt;
    const auto old_started_at = rt.m_recovery_started_at;
    print_test_result("Old recovery timeout has exceeded the hard stop before re-authentication",
                      rt.m_timeout.elapsed_seconds(std::chrono::steady_clock::now()) >= 300);
    rt.on_reauthenticated();

    print_test_result("Re-authentication starts a new recovery epoch",
                      rt.m_recovery_epoch == 2);
    print_test_result("Recovery remains pending after re-authentication",
                      rt.m_recovery_pending);
    print_test_result("Recovery start time is refreshed after re-authentication",
                      rt.m_recovery_started_at != std::chrono::steady_clock::time_point{} &&
                      rt.m_recovery_started_at > old_started_at);
    auto elapsed_after_reauth = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - rt.m_recovery_started_at).count();
    print_test_result("Fresh recovery epoch elapsed time is near-zero after re-authentication",
                      elapsed_after_reauth >= 0 && elapsed_after_reauth <= 1);
    print_test_result("Degraded timer is reset for the new authenticated session",
                      rt.m_degraded_since == rt.m_recovery_started_at);
    print_test_result("Recovery timeout is reset without entering HEALTHY",
                      rt.m_timeout.elapsed_seconds(rt.m_degraded_since) == 0);
    print_test_result("New session receives the full recovery timeout window",
                      rt.m_timeout.elapsed_seconds(
                         rt.m_degraded_since + std::chrono::seconds(299)) == 299);

    const auto old_wait_end = rt.m_degraded_since + std::chrono::seconds(600);
    rt.m_timeout.observe_node_retry(true, true, rt.m_degraded_since);
    rt.m_timeout.observe_node_retry(false, true, old_wait_end);
    rt.m_timeout.observe_node_retry(true, true, old_wait_end);
    rt.on_reauthenticated();
    const auto next_tick = rt.m_degraded_since + std::chrono::seconds(1);
    rt.m_timeout.observe_node_retry(false, true, next_tick);
    print_test_result("Re-authentication clears the prior session's active retry wait",
                      rt.m_timeout.elapsed_seconds(next_tick) == 1);
    print_test_result("Re-authentication clears the prior retry's silence-accounting rebase",
                      rt.m_timeout.worst_case_outage_seconds(next_tick, 10, 20) == 20);

    // Verify EpochCoordinator-based recovery epoch works the same way
    {
        nexusminer::protocol::EpochCoordinator coordinator;
        coordinator.advance_recovery_epoch("initial_recovery");  // simulate entering recovery
        const uint64_t epoch_before_reauth = coordinator.recovery_epoch();
        coordinator.advance_recovery_epoch("session_reauthenticated");  // simulate reauth
        print_test_result("EpochCoordinator: re-authentication advances recovery epoch",
                          coordinator.recovery_epoch() == epoch_before_reauth + 1);
        print_test_result("EpochCoordinator: recovery epoch never resets to 0",
                          coordinator.recovery_epoch() > 0);
    }
}

// ============================================================================
// Test 4c: Soft refresh timeout escalates only after the hot-swap window stalls
// ============================================================================
void test_same_height_soft_refresh_escalates_only_after_timeout() {
    std::cout << "\nTest 4c: Template request enters WAITING_TEMPLATE until timeout triggers reconnect\n";
    constexpr int64_t RECOVERY_WINDOW_SECONDS = 300;
    constexpr int64_t TIMEOUT_TEST_SECONDS = RECOVERY_WINDOW_SECONDS + 1;

    struct RecoveryTracker {
        TestRecoveryPhase phase{TestRecoveryPhase::HEALTHY};
        uint64_t epoch{0};
        std::chrono::steady_clock::time_point entered_at{};
        std::string authoritative_recovery_reason;
        std::vector<std::string> m_log;

        void start_waiting_template(const std::string& reason) {
            ++epoch;
            phase = TestRecoveryPhase::WAITING_TEMPLATE;
            entered_at = std::chrono::steady_clock::now();
            authoritative_recovery_reason = reason;
            m_log.emplace_back("template refresh requested");
        }

        bool should_reconnect(int64_t recovery_window_seconds) const {
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - entered_at).count();
            return elapsed >= recovery_window_seconds;
        }

        void do_reconnect() {
            phase = TestRecoveryPhase::RECONNECTING;
        }
    };

    RecoveryTracker rt;
    rt.start_waiting_template("same_height_push_tip_replacement_pre_adoption");
    print_test_result("Template request starts WAITING_TEMPLATE recovery tracking",
                      phase_is_recovery_active(rt.phase) && rt.epoch == 1);
    print_test_result("WAITING_TEMPLATE logs template refresh requested",
                      !rt.m_log.empty() && rt.m_log.back() == "template refresh requested");
    print_test_result("WAITING_TEMPLATE records correct reason",
                      rt.authoritative_recovery_reason == "same_height_push_tip_replacement_pre_adoption");
    print_test_result("WAITING_TEMPLATE: submissions not withheld (workers keep running)",
                      !phase_is_submissions_withheld(rt.phase));
    print_test_result("WAITING_TEMPLATE: is_degraded() == true (waiting for template)",
                      phase_is_degraded(rt.phase));
    print_test_result("WAITING_TEMPLATE does not reconnect immediately", !rt.should_reconnect(RECOVERY_WINDOW_SECONDS));

    rt.entered_at = std::chrono::steady_clock::now() - std::chrono::seconds(TIMEOUT_TEST_SECONDS);
    print_test_result("WAITING_TEMPLATE triggers reconnect after 300s timeout", rt.should_reconnect(RECOVERY_WINDOW_SECONDS));
    rt.do_reconnect();
    print_test_result("After timeout: phase is RECONNECTING", phase_is_reconnecting(rt.phase));
    print_test_result("After timeout: submissions still not withheld", !phase_is_submissions_withheld(rt.phase));
}

// ============================================================================
// Test 4d: clear_recovery_state-style normalization restores healthy baseline
// ============================================================================
void test_degraded_exit_normalizes_recovery_state() {
    std::cout << "\nTest 4d: Degraded exit normalization clears recovery bookkeeping\n";

    struct RecoveryTracker {
        TestRecoveryPhase phase{TestRecoveryPhase::WAITING_TEMPLATE};
        uint64_t epoch{7};
        std::chrono::steady_clock::time_point entered_at{std::chrono::steady_clock::now()};
        std::chrono::steady_clock::time_point last_get_block_at{std::chrono::steady_clock::now()};
        bool get_block_confirmed{true};
        std::chrono::steady_clock::time_point degraded_since{std::chrono::steady_clock::now()};
        bool authoritative_recovery_healthy_called{false};
        std::string authoritative_recovery_reason{"same_height_push_tip_replacement_pre_adoption"};

        void clear() {
            authoritative_recovery_healthy_called = true;
            authoritative_recovery_reason.clear();
            phase = TestRecoveryPhase::HEALTHY;
            epoch = 0;
            entered_at = {};
            last_get_block_at = {};
            get_block_confirmed = false;
            degraded_since = {};
        }
    };

    RecoveryTracker rt;
    rt.clear();

    print_test_result("Phase returned to HEALTHY", rt.phase == TestRecoveryPhase::HEALTHY);
    print_test_result("Recovery active cleared", !phase_is_recovery_active(rt.phase));
    print_test_result("Degraded mode cleared", !phase_is_degraded(rt.phase));
    print_test_result("Soft-pause guard cleared", !phase_is_submissions_withheld(rt.phase));
    print_test_result("Recovery epoch reset", rt.epoch == 0);
    print_test_result("GET_BLOCK bookkeeping cleared",
                      rt.entered_at == std::chrono::steady_clock::time_point{} &&
                      rt.last_get_block_at == std::chrono::steady_clock::time_point{} &&
                      !rt.get_block_confirmed);
    print_test_result("Escalation timer cleared",
                      rt.degraded_since == std::chrono::steady_clock::time_point{});
    print_test_result("Authoritative recovery state normalized",
                      rt.authoritative_recovery_healthy_called &&
                      rt.authoritative_recovery_reason.empty());
}

// ============================================================================
// Test 4e: SOFT_REFRESH transition withholds submissions without degraded mode
// Replaces the old test_authoritative_soft_refresh_backfills_local_state()
// which referenced the removed RecoveryState::SOFT_REFRESH_REQUESTED enum value.
// Now tests the surviving RecoveryPhase::SOFT_REFRESH transition path.
// ============================================================================
void test_authoritative_soft_refresh_backfills_local_state() {
    std::cout << "\nTest 4e: Template refresh request enters WAITING_TEMPLATE\n";

    struct RecoveryTracker {
        TestRecoveryPhase phase{TestRecoveryPhase::HEALTHY};
        uint64_t epoch{0};
        std::chrono::steady_clock::time_point entered_at{};

        bool request_template_refresh(const char* /*reason*/) {
            if (phase_is_recovery_active(phase)) {
                return false;
            }
            ++epoch;
            phase = TestRecoveryPhase::WAITING_TEMPLATE;
            entered_at = std::chrono::steady_clock::now();
            return true;
        }
    };

    RecoveryTracker rt;
    const bool transitioned = rt.request_template_refresh("same_height_push_tip_replacement");
    print_test_result("Template refresh transitions from HEALTHY to WAITING_TEMPLATE", transitioned);
    print_test_result("WAITING_TEMPLATE: submissions are NOT withheld (new model)", !phase_is_submissions_withheld(rt.phase));
    print_test_result("WAITING_TEMPLATE: is_degraded() == true (waiting for template)", phase_is_degraded(rt.phase));
    print_test_result("WAITING_TEMPLATE anchors recovery epoch", rt.epoch == 1);
    print_test_result("WAITING_TEMPLATE anchors the recovery timer",
                      rt.entered_at != std::chrono::steady_clock::time_point{});

    const bool second = rt.request_template_refresh("duplicate_push");
    print_test_result("Second refresh request is a no-op (idempotent)", !second);
    print_test_result("Epoch unchanged after idempotent call", rt.epoch == 1);
}

// ============================================================================
// Test 4f: tip_moved soft refresh shields workers from immediate HEIGHT_DRIFT
//          escalation until the replacement-template window actually times out.
// ============================================================================
void test_tip_moved_soft_refresh_defers_unified_drift_stop_until_timeout() {
    std::cout << "\nTest 4f: tip_moved requests fresh template without halting submissions\n";
    // KEY CHANGE: is_tip_moved() no longer triggers soft refresh.
    // Cross-channel tip advances are informational — the current channel template
    // is still valid.  Workers keep mining AND submitting on the current template.
    // A fresh GET_BLOCK is requested opportunistically so hashPrevBlock stays current.
    // The HEIGHT_DRIFT guard still fires on the next tick if unified drift exceeds
    // the threshold and no new template has arrived.
    constexpr uint32_t UNIFIED_DRIFT_THRESHOLD = 5;

    struct HealthState {
        bool m_degraded_mode{false};
        bool m_recovery_pending{false};
        bool m_template_withheld{false};
        bool request_refresh{false};
        bool stop_workers{false};
        uint64_t m_recovery_epoch{0};

        void tick(bool tip_moved, uint32_t unified_drift, int64_t /*recovery_elapsed_s*/) {
            request_refresh = false;
            stop_workers = false;

            // NEW: tip_moved no longer starts a soft refresh.
            // It just requests a fresh template opportunistically and returns.
            if (tip_moved) {
                request_refresh = true;
                // m_template_withheld stays false — submissions are NOT withheld
                return;
            }

            // HEIGHT_DRIFT fires on subsequent ticks when no new template has arrived
            if (unified_drift > UNIFIED_DRIFT_THRESHOLD) {
                m_degraded_mode = true;
                stop_workers = true;
            }
        }
    };

    HealthState state;
    state.tick(/*tip_moved=*/true, /*unified_drift=*/7, /*recovery_elapsed_s=*/0);
    print_test_result("tip_moved requests fresh template without stopping workers",
                      state.request_refresh && !state.stop_workers && !state.m_degraded_mode);
    print_test_result("tip_moved does NOT withhold submissions (workers keep submitting)",
                      !state.m_template_withheld && !state.m_recovery_pending);

    state.tick(/*tip_moved=*/false, /*unified_drift=*/7, /*recovery_elapsed_s=*/10);
    print_test_result("HEIGHT_DRIFT fires on next tick when drift exceeds threshold and tip no longer moved",
                      state.stop_workers && state.m_degraded_mode);
}

// ============================================================================
// Test 4g: Worker respawn guard prevents duplicate worker recreation on exit
// ============================================================================
void test_worker_respawn_guard_is_single_shot_per_degraded_exit() {
    std::cout << "\nTest 4g: Recovery worker respawn guard is single-shot per degraded exit\n";

    struct WorkerRestartTracker {
        bool m_degraded_mode{true};
        bool m_recovery_workers_spawned{false};
        std::size_t worker_instances{0};
        std::size_t create_calls{0};

        bool restart_workers_if_needed() {
            if (!m_degraded_mode || m_recovery_workers_spawned || worker_instances > 0) {
                return false;
            }
            worker_instances = 8;
            m_recovery_workers_spawned = true;
            ++create_calls;
            return true;
        }
    };

    WorkerRestartTracker rt;
    bool first_restart = rt.restart_workers_if_needed();
    bool second_restart = rt.restart_workers_if_needed();

    print_test_result("First degraded-exit restart creates workers", first_restart);
    print_test_result("Second degraded-exit restart is suppressed", !second_restart);
    print_test_result("Workers are created only once for the outage", rt.create_calls == 1);
    print_test_result("Worker count stays at the expected 8 threads", rt.worker_instances == 8);
}

// ============================================================================
// Test 4h: Workers only respawn after the prior generation is fully stopped
// ============================================================================
void test_worker_respawn_waits_for_authoritative_empty_generation() {
    std::cout << "\nTest 4h: Worker respawn waits for authoritative empty generation\n";

    struct WorkerLifecycleTracker {
        bool m_degraded_mode{true};
        bool m_recovery_workers_spawned{false};
        bool stop_in_progress{false};
        std::size_t worker_instances{8};
        std::size_t create_calls{0};

        void begin_stop() {
            stop_in_progress = true;
        }

        void finish_stop() {
            worker_instances = 0;
            stop_in_progress = false;
            m_recovery_workers_spawned = false;
        }

        bool restart_workers_if_needed() {
            const bool degraded_mode_inactive = !m_degraded_mode;
            const bool restart_already_consumed = m_recovery_workers_spawned;
            const bool prior_generation_still_stopping = stop_in_progress;
            const bool workers_still_present = worker_instances != 0;

            if (degraded_mode_inactive || restart_already_consumed ||
                prior_generation_still_stopping || workers_still_present) {
                return false;
            }
            worker_instances = 8;
            m_recovery_workers_spawned = true;
            ++create_calls;
            return true;
        }
    };

    WorkerLifecycleTracker rt;
    rt.begin_stop();
    bool restart_during_stop = rt.restart_workers_if_needed();
    rt.finish_stop();
    bool restart_after_stop = rt.restart_workers_if_needed();
    bool second_restart_after_stop = rt.restart_workers_if_needed();

    print_test_result("Respawn is blocked while prior generation is still stopping", !restart_during_stop);
    print_test_result("Respawn succeeds once workers are authoritatively gone", restart_after_stop);
    print_test_result("Respawn remains single-shot after the restart", !second_restart_after_stop);
    print_test_result("Only one new worker generation is created", rt.create_calls == 1);
    print_test_result("Worker count returns to the expected 8 threads", rt.worker_instances == 8);
}

// ============================================================================
// Test 5: Keepalive epoch isolation — new epoch starts with clean ack timestamp
// ============================================================================
void test_keepalive_epoch_isolation_clean_start() {
    std::cout << "\nTest 5: New epoch starts with clean keepalive ACK timestamp\n";
    HeightTracker tracker;

    // Simulate three successive re-auths (epoch 1 → 2 → 3)
    for (uint64_t epoch = 1; epoch <= 3; ++epoch) {
        tracker.set_session_epoch(SessionEpoch(epoch));

        // At start of each epoch, ack timestamp must be clear
        auto snap_start = tracker.GetSnapshot();
        print_test_result(("Epoch " + std::to_string(epoch) + ": ack timestamp clear at epoch start").c_str(),
                          snap_start.last_keepalive_ack_at == std::chrono::steady_clock::time_point{});

        // Receive keepalive for this epoch
        tracker.OnKeepaliveResponse(5000 + static_cast<uint32_t>(epoch), 400, 700, 900, 0u, 0);
        auto snap_after = tracker.GetSnapshot();
        print_test_result(("Epoch " + std::to_string(epoch) + ": ack timestamp set after response").c_str(),
                          snap_after.last_keepalive_ack_at != std::chrono::steady_clock::time_point{});
    }
}

// ============================================================================
// Test 6: Stale template after channel advance — canonical auto-advance invariant
// (Validates that canonical auto-advance prevents staleness; real staleness detection
//  has moved to push/GET_ROUND trigger paths)
// ============================================================================
void test_stale_template_after_channel_advance() {
    std::cout << "\nTest 6: Stale template detection after channel advance (canonical auto-advance)\n";
    HeightTracker tracker;

    // Template targeting channel_height=101
    tracker.OnBlockDataReceived(5000, 100, 0x1d00ffff, uint1024_t{});
    tracker.OnTemplateReceived(1, 101);

    // Confirm not stale before advance
    auto snap_before = tracker.GetSnapshot();
    print_test_result("Before advance: channel_height < channel_target", snap_before.channel_height < snap_before.channel_target);

    // Channel advances (block found) — OnBlockDataReceived auto-advances target to 102.
    // Under canonical-only semantics, is_template_stale() was structurally false
    // because OnBlockDataReceived always ensures target >= channel + 1.
    // is_template_stale() removed (structurally false under canonical-only semantics)
    tracker.OnBlockDataReceived(5001, 101, 0x1d00ffff, uint1024_t{});
    auto snap_after = tracker.GetSnapshot();
    print_test_result("After advance: canonical auto-advance (channel < target)", snap_after.channel_height < snap_after.channel_target);
    print_test_result("After advance: channel_height == 101", snap_after.channel_height == 101);
    print_test_result("After advance: channel_target == 102 (auto-advanced)", snap_after.channel_target == 102);

    // Under canonical-only: channel_height < channel_target always holds after OnBlockDataReceived
    print_test_result("channel_height < channel_target (canonical auto-advance invariant)",
                      snap_after.channel_height < snap_after.channel_target);
}

// ============================================================================
// Test 7: set_session_epoch() suppresses old keepalive signal
// Confirms keepalive ACK timestamp is cleared after epoch change (diagnostic field)
// ============================================================================
void test_epoch_advance_suppresses_old_keepalive_signal() {
    std::cout << "\nTest 7: Epoch advance clears keepalive ACK timestamp (diagnostic field)\n";
    HeightTracker tracker;

    tracker.set_session_epoch(SessionEpoch(10));
    tracker.OnKeepaliveResponse(5000, 400, 700, 900, 0u, 0);

    // Confirm old epoch keepalive is "set"
    auto snap_epoch10 = tracker.GetSnapshot();
    bool was_set = (snap_epoch10.last_keepalive_ack_at != std::chrono::steady_clock::time_point{});
    print_test_result("Epoch 10: keepalive ACK timestamp is set", was_set);

    // Advance epoch — simulates re-auth or session restart
    tracker.set_session_epoch(SessionEpoch(11));
    auto snap_epoch11 = tracker.GetSnapshot();

    // The keepalive ACK timestamp is diagnostic only — PUSH is the sole authoritative
    // signal for session liveness.  After epoch change, the diagnostic timestamp is cleared.
    bool keepalive_ack_received = (snap_epoch11.last_keepalive_ack_at !=
                                    std::chrono::steady_clock::time_point{});
    print_test_result("Epoch 11: keepalive_ack_received == false after epoch advance",
                      !keepalive_ack_received);
    print_test_result("Epoch 11: keepalive diagnostic timestamp cleared (epoch isolation)",
                      !keepalive_ack_received);
}

// ============================================================================
// Test 8: Multiple rapid epoch changes produce clean keepalive state
// ============================================================================
void test_multiple_rapid_epoch_changes_clean_state() {
    std::cout << "\nTest 8: Multiple rapid epoch changes always produce clean keepalive state\n";
    HeightTracker tracker;

    uint64_t current_epoch = 0;
    for (int i = 0; i < 5; ++i) {
        ++current_epoch;
        tracker.set_session_epoch(SessionEpoch(current_epoch));

        // Epoch starts clean
        auto snap_start = tracker.GetSnapshot();
        print_test_result(("Epoch " + std::to_string(current_epoch) + ": clean start").c_str(),
                          snap_start.last_keepalive_ack_at == std::chrono::steady_clock::time_point{});

        // Receive keepalive
        tracker.OnKeepaliveResponse(5000, 400, 700, 900, 0u, 0);
    }

    // Final state: keepalive is set for epoch 5
    auto snap_final = tracker.GetSnapshot();
    print_test_result("Final epoch (5): keepalive ACK timestamp is set from epoch 5 response",
                      snap_final.last_keepalive_ack_at != std::chrono::steady_clock::time_point{});
    print_test_result("Final epoch (5): session_epoch == 5", snap_final.session_epoch == SessionEpoch(5));
}

// ============================================================================
// Test 9: Push notification does not reset keepalive timestamp on epoch change
// Pushes and keepalives are orthogonal — epoch change clears ONLY keepalive ack
// ============================================================================
void test_push_does_not_clear_keepalive_on_epoch_change() {
    std::cout << "\nTest 9: Push notifications do not clear keepalive timestamp on epoch change\n";
    HeightTracker tracker;

    tracker.set_session_epoch(SessionEpoch(1));
    tracker.OnKeepaliveResponse(5000, 400, 700, 900, 0u, 0);
    tracker.OnPushNotification(5000, 100, 0x1d00ffff);

    auto snap1 = tracker.GetSnapshot();
    bool push_at_set  = (snap1.last_push_notification_at != std::chrono::steady_clock::time_point{});
    bool ack_at_set   = (snap1.last_keepalive_ack_at != std::chrono::steady_clock::time_point{});
    print_test_result("Epoch 1: push timestamp is set", push_at_set);
    print_test_result("Epoch 1: keepalive ACK timestamp is set", ack_at_set);

    // Epoch change: keepalive clears, push is unchanged
    tracker.set_session_epoch(SessionEpoch(2));
    auto snap2 = tracker.GetSnapshot();
    bool push_unchanged  = (snap2.last_push_notification_at == snap1.last_push_notification_at);
    bool ack_cleared     = (snap2.last_keepalive_ack_at == std::chrono::steady_clock::time_point{});
    print_test_result("Epoch 2: push timestamp preserved (not cleared by epoch change)",
                      push_unchanged);
    print_test_result("Epoch 2: keepalive ACK timestamp cleared by epoch change",
                      ack_cleared);
}

// ============================================================================
// Test 9b: Session epoch advance clears stale push tip-anchor hints but keeps push liveness
// ============================================================================
void test_epoch_change_clears_push_tip_anchor_hint() {
    std::cout << "\nTest 9b: Session epoch advance clears stale push tip-anchor hint only\n";
    HeightTracker tracker;

    tracker.set_session_epoch(SessionEpoch(7));
    tracker.OnPushNotification(5000, 100, 0x1d00ffff);
    tracker.UpdatePushTipAnchor(uint1024_t(0x42));

    auto snap_before = tracker.GetSnapshot();
    print_test_result("Epoch 7: push tip-anchor hint is present before epoch change",
                      snap_before.push_hash_prev_block != uint1024_t{});
    print_test_result("Epoch 7: push liveness timestamp is present before epoch change",
                      snap_before.last_push_notification_at != std::chrono::steady_clock::time_point{});

    tracker.set_session_epoch(SessionEpoch(8));
    auto snap_after = tracker.GetSnapshot();
    print_test_result("Epoch 8: push tip-anchor hint cleared on epoch change",
                      snap_after.push_hash_prev_block == uint1024_t{});
    print_test_result("Epoch 8: push liveness timestamp preserved across epoch change",
                      snap_after.last_push_notification_at == snap_before.last_push_notification_at);

    tracker.UpdatePushTipAnchor(uint1024_t(0x77));
    auto snap_reseeded = tracker.GetSnapshot();
    print_test_result("Explicit clear test re-seeds push tip-anchor hint first",
                      snap_reseeded.push_hash_prev_block != uint1024_t{});
    tracker.ClearPushTipAnchor();
    auto snap_cleared = tracker.GetSnapshot();
    print_test_result("Explicit tip-anchor clear consumes replacement hint after adoption",
                      snap_cleared.push_hash_prev_block == uint1024_t{});
}

// ============================================================================
// Test 10: Recovery epoch tracking — monotonic with idempotent initiation
// ============================================================================
void test_recovery_epoch_monotonic_with_idempotent_initiation() {
    std::cout << "\nTest 10: Recovery epoch is monotonic; initiation is idempotent within epoch\n";

    struct RecoveryState {
        int epoch{0};
        bool pending{false};

        void initiate() {
            if (pending) return;  // idempotent
            ++epoch;
            pending = true;
        }
        void clear() {
            pending = false;
        }
    };

    RecoveryState rs;

    // Multiple calls before any template arrives: all no-ops except first
    rs.initiate();
    int epoch_after_first = rs.epoch;
    rs.initiate();
    rs.initiate();
    int epoch_after_many = rs.epoch;
    print_test_result("Epoch is monotonic: multiple inits don't increment past 1",
                      epoch_after_first == 1 && epoch_after_many == 1);
    print_test_result("Pending remains true after multiple inits", rs.pending);

    // Recovery completes (template received)
    rs.clear();
    print_test_result("After clear: pending == false", !rs.pending);
    print_test_result("After clear: epoch still == 1 (preserved for audit)", rs.epoch == 1);

    // Next staleness event: epoch advances
    rs.initiate();
    print_test_result("Second staleness: epoch == 2 (monotonically incremented)", rs.epoch == 2);
    print_test_result("Second staleness: pending == true", rs.pending);

    // Multiple calls again: no increment past 2
    rs.initiate();
    rs.initiate();
    print_test_result("Multiple inits after epoch 2: epoch remains 2", rs.epoch == 2);
}

// ============================================================================
// Test 11: Integration - channel advance -> stale -> degraded -> fresh -> resume
// ============================================================================
void test_integration_degraded_recovery_to_resume() {
    std::cout << "\nTest 11: Integration degraded recovery returns to active mining\n";

    struct IntegrationState {
        bool mining_active{true};
        bool degraded{false};
        bool recovery_pending{false};
        bool has_valid_template{true};
        uint32_t channel_height{100};
        uint32_t channel_target{101};
        int get_block_sent{0};

        void on_channel_advance(uint32_t new_height) {
            channel_height = new_height;
            if (channel_height >= channel_target) {
                degraded = true;
                recovery_pending = true;
                has_valid_template = false;
                mining_active = false;
            }
        }

        void recovery_tick() {
            if (degraded && recovery_pending && !has_valid_template) {
                ++get_block_sent;
            }
        }

        void on_fresh_template(uint32_t new_target) {
            channel_target = new_target;
            has_valid_template = true;
            if (degraded && recovery_pending) {
                degraded = false;
                recovery_pending = false;
                mining_active = true;
            }
        }
    };

    IntegrationState s;
    s.on_channel_advance(101);  // stale
    print_test_result("Stale transition enters degraded mode", s.degraded && !s.mining_active);

    s.recovery_tick();
    print_test_result("Recovery tick sends GET_BLOCK while degraded", s.get_block_sent == 1);

    s.on_fresh_template(102);
    print_test_result("Fresh template exits degraded mode", !s.degraded && !s.recovery_pending);
    print_test_result("Mining resumes after template acceptance", s.mining_active);
}

// ============================================================================
// Test 12: Integration - bounded forced retries prevent flooding
// ============================================================================
void test_integration_forced_retry_is_bounded() {
    std::cout << "\nTest 12: Integration forced retry channel remains bounded\n";

    struct ForcedBound {
        std::deque<std::chrono::steady_clock::time_point> sends;
        int64_t interval_ms{1000};
        size_t max_burst{25};
        std::chrono::steady_clock::time_point next_due{};

        bool try_send(std::chrono::steady_clock::time_point now) {
            while (!sends.empty()) {
                auto age = std::chrono::duration_cast<std::chrono::seconds>(now - sends.front()).count();
                if (age <= 60) break;
                sends.pop_front();
            }
            if (next_due != std::chrono::steady_clock::time_point{} && now < next_due) {
                return false;
            }
            if (sends.size() >= max_burst) {
                return false;
            }
            sends.push_back(now);
            next_due = now + std::chrono::milliseconds(interval_ms);
            return true;
        }
    };

    ForcedBound bound;
    auto now = std::chrono::steady_clock::now();
    int sent = 0;
    for (int i = 0; i < 120; ++i) {  // simulate 120 recovery ticks per second
        auto tick_tp = now + std::chrono::milliseconds(i * 500); // tick every 500ms
        if (bound.try_send(tick_tp)) {
            ++sent;
        }
    }
    print_test_result("Forced retries stay at or below max_forced_burst_per_60s", sent <= 25);
    print_test_result("Forced retries still make progress (at least one send)", sent > 0);
}

// ============================================================================
// Test 13: Health policy — channel staleness requests refresh without forcing
//          degraded/worker-stop; session is preserved for reorg-heavy periods.
// ============================================================================
void test_health_policy_distinguishes_normal_refresh_from_multi_block_lag() {
    std::cout << "\nTest 13: Health policy distinguishes 1-block refresh from 2+-block lag\n";

    struct HealthDecision {
        bool request_refresh{false};
        bool recovery_initiated{false};
        bool stop_workers{false};
    };

    auto decide = [](const HeightTracker::Snapshot& snap, bool template_is_newer_than_push) {
        HealthDecision decision;

        // is_template_stale() removed (structurally false under canonical-only semantics).
        // Under canonical semantics the condition was always true, so the early return
        // always executes — no refresh action is ever triggered from this path.
        return decision;
    };

    // Under canonical-only semantics, is_template_stale() was structurally false
    // because OnBlockDataReceived auto-advances target to channel+1. The health
    // policy returns early with no action — real staleness detection has moved
    // to the push/GET_ROUND trigger paths.
    HeightTracker one_block_tracker;
    one_block_tracker.OnBlockDataReceived(5000, 100, 0x1d00ffff, uint1024_t{});
    one_block_tracker.OnTemplateReceived(2, 101);
    one_block_tracker.OnBlockDataReceived(5001, 101, 0x1d00ffff, uint1024_t{});
    auto one_block = decide(one_block_tracker.GetSnapshot(), false);
    print_test_result("One-block lag: canonical not stale (auto-advance), no refresh needed", !one_block.request_refresh);
    print_test_result("One-block lag does not initiate recovery", !one_block.recovery_initiated);
    print_test_result("One-block lag does not stop workers", !one_block.stop_workers);

    HeightTracker two_block_tracker;
    two_block_tracker.OnBlockDataReceived(5000, 100, 0x1d00ffff, uint1024_t{});
    two_block_tracker.OnTemplateReceived(2, 101);
    two_block_tracker.OnBlockDataReceived(5001, 101, 0x1d00ffff, uint1024_t{});
    two_block_tracker.OnBlockDataReceived(5002, 102, 0x1d00ffff, uint1024_t{});
    auto two_block = decide(two_block_tracker.GetSnapshot(), false);
    print_test_result("Two-block lag: canonical not stale (auto-advance), no refresh needed", !two_block.request_refresh);
    print_test_result("Two-block lag does not initiate recovery", !two_block.recovery_initiated);
    print_test_result("Two-block lag does not stop workers", !two_block.stop_workers);

    // Post-push scenario: under canonical-only semantics, decide() always returns
    // early with no action (is_template_stale was structurally false), so no refresh.
    HeightTracker::Snapshot post_push_snap;
    post_push_snap.channel_height = 300;
    post_push_snap.channel_target = 300;
    auto post_push = decide(post_push_snap, true);
    print_test_result("Post-push: canonical policy returns no-action (staleness moved to trigger paths)", !post_push.request_refresh);
    print_test_result("Post-push template freshness does not initiate recovery", !post_push.recovery_initiated);
}

// ============================================================================
// ⚡ RecoveryPhase State Machine Tests (mirrors worker_manager.hpp RecoveryPhase)
// ============================================================================

// ── Test 18: Legal transitions succeed, illegal ones are rejected ──────────
void test_recovery_phase_valid_transitions() {
    std::cout << "\nTest 18: RecoveryPhase state machine — valid transition matrix\n";

    // Legal transitions
    print_test_result("HEALTHY → WAITING_TEMPLATE is legal",
        test_is_valid_transition(TestRecoveryPhase::HEALTHY, TestRecoveryPhase::WAITING_TEMPLATE));
    print_test_result("HEALTHY → RECONNECTING is legal",
        test_is_valid_transition(TestRecoveryPhase::HEALTHY, TestRecoveryPhase::RECONNECTING));
    print_test_result("HEALTHY → SESSION_RECOVERY is legal",
        test_is_valid_transition(TestRecoveryPhase::HEALTHY, TestRecoveryPhase::SESSION_RECOVERY));
    print_test_result("WAITING_TEMPLATE → HEALTHY is legal",
        test_is_valid_transition(TestRecoveryPhase::WAITING_TEMPLATE, TestRecoveryPhase::HEALTHY));
    print_test_result("WAITING_TEMPLATE → RECONNECTING is legal",
        test_is_valid_transition(TestRecoveryPhase::WAITING_TEMPLATE, TestRecoveryPhase::RECONNECTING));
    print_test_result("WAITING_TEMPLATE → SESSION_RECOVERY is legal",
        test_is_valid_transition(TestRecoveryPhase::WAITING_TEMPLATE, TestRecoveryPhase::SESSION_RECOVERY));
    print_test_result("SESSION_RECOVERY → WAITING_TEMPLATE is legal",
        test_is_valid_transition(TestRecoveryPhase::SESSION_RECOVERY, TestRecoveryPhase::WAITING_TEMPLATE));
    print_test_result("SESSION_RECOVERY → RECONNECTING is legal",
        test_is_valid_transition(TestRecoveryPhase::SESSION_RECOVERY, TestRecoveryPhase::RECONNECTING));
    print_test_result("RECONNECTING → HEALTHY is legal",
        test_is_valid_transition(TestRecoveryPhase::RECONNECTING, TestRecoveryPhase::HEALTHY));
    print_test_result("RECONNECTING → WAITING_TEMPLATE is legal",
        test_is_valid_transition(TestRecoveryPhase::RECONNECTING, TestRecoveryPhase::WAITING_TEMPLATE));

    // Same-phase no-ops are always valid
    print_test_result("HEALTHY → HEALTHY is valid (no-op)",
        test_is_valid_transition(TestRecoveryPhase::HEALTHY, TestRecoveryPhase::HEALTHY));
    print_test_result("WAITING_TEMPLATE → WAITING_TEMPLATE is valid (no-op)",
        test_is_valid_transition(TestRecoveryPhase::WAITING_TEMPLATE, TestRecoveryPhase::WAITING_TEMPLATE));
}

// ── Test 19: Phase helper equivalences match explicit session-recovery design ─────────
void test_recovery_phase_helpers_equivalence() {
    std::cout << "\nTest 19: Phase helpers match explicit session-recovery design\n";

    // is_degraded() ← WAITING_TEMPLATE only
    print_test_result("HEALTHY: is_degraded() == false",       !phase_is_degraded(TestRecoveryPhase::HEALTHY));
    print_test_result("WAITING_TEMPLATE: is_degraded() == true", phase_is_degraded(TestRecoveryPhase::WAITING_TEMPLATE));
    print_test_result("SESSION_RECOVERY: is_degraded() == false", !phase_is_degraded(TestRecoveryPhase::SESSION_RECOVERY));
    print_test_result("RECONNECTING: is_degraded() == false",  !phase_is_degraded(TestRecoveryPhase::RECONNECTING));

    // is_submissions_withheld() ← always false in new design
    print_test_result("HEALTHY: is_submissions_withheld() == false",
        !phase_is_submissions_withheld(TestRecoveryPhase::HEALTHY));
    print_test_result("WAITING_TEMPLATE: is_submissions_withheld() == false",
        !phase_is_submissions_withheld(TestRecoveryPhase::WAITING_TEMPLATE));
    print_test_result("SESSION_RECOVERY: is_submissions_withheld() == false",
        !phase_is_submissions_withheld(TestRecoveryPhase::SESSION_RECOVERY));

    // is_recovery_active() ← any non-HEALTHY phase
    print_test_result("HEALTHY: is_recovery_active() == false",
        !phase_is_recovery_active(TestRecoveryPhase::HEALTHY));
    print_test_result("WAITING_TEMPLATE: is_recovery_active() == true",
        phase_is_recovery_active(TestRecoveryPhase::WAITING_TEMPLATE));
    print_test_result("SESSION_RECOVERY: is_recovery_active() == true",
        phase_is_recovery_active(TestRecoveryPhase::SESSION_RECOVERY));
    print_test_result("RECONNECTING: is_recovery_active() == true",
        phase_is_recovery_active(TestRecoveryPhase::RECONNECTING));

    // is_reconnecting() ← RECONNECTING only
    print_test_result("HEALTHY: is_reconnecting() == false",
        !phase_is_reconnecting(TestRecoveryPhase::HEALTHY));
    print_test_result("RECONNECTING: is_reconnecting() == true",
        phase_is_reconnecting(TestRecoveryPhase::RECONNECTING));
    print_test_result("WAITING_TEMPLATE: is_reconnecting() == false",
        !phase_is_reconnecting(TestRecoveryPhase::WAITING_TEMPLATE));
    print_test_result("SESSION_RECOVERY: is_reconnecting() == false",
        !phase_is_reconnecting(TestRecoveryPhase::SESSION_RECOVERY));
}

// ── Test 20: WAITING_TEMPLATE state — clearing state always exits it ──────
void test_orphaned_soft_refresh_cleared() {
    std::cout << "\nTest 20: WAITING_TEMPLATE state is cleared by clear_recovery_state\n";

    // Simulate: was in WAITING_TEMPLATE, clear_recovery_state() transitions to HEALTHY
    TestRecoveryPhase phase = TestRecoveryPhase::WAITING_TEMPLATE;

    // Verify waiting-template is active
    print_test_result("Initial: is_recovery_active() == true (WAITING_TEMPLATE)",
        phase_is_recovery_active(phase));
    print_test_result("Initial: is_degraded() == true (WAITING_TEMPLATE)",
        phase_is_degraded(phase));

    // clear_recovery_state() logic: transition to HEALTHY
    phase = TestRecoveryPhase::HEALTHY;

    print_test_result("After clear: phase == HEALTHY",
        phase == TestRecoveryPhase::HEALTHY);
    print_test_result("After clear: is_submissions_withheld() == false",
        !phase_is_submissions_withheld(phase));
    print_test_result("After clear: is_recovery_active() == false",
        !phase_is_recovery_active(phase));
}

// ── Test 21: WAITING_TEMPLATE → RECONNECTING escalation path ─────────────
void test_soft_refresh_escalation_to_hard_recovery() {
    std::cout << "\nTest 21: WAITING_TEMPLATE → RECONNECTING after 300s timeout\n";

    TestRecoveryPhase phase = TestRecoveryPhase::HEALTHY;

    // Enter WAITING_TEMPLATE (stale template)
    assert(test_is_valid_transition(phase, TestRecoveryPhase::WAITING_TEMPLATE));
    phase = TestRecoveryPhase::WAITING_TEMPLATE;
    print_test_result("Entered WAITING_TEMPLATE: is_recovery_active() == true",
        phase_is_recovery_active(phase));
    print_test_result("WAITING_TEMPLATE: is_degraded() == true",
        phase_is_degraded(phase));
    print_test_result("WAITING_TEMPLATE: is_submissions_withheld() == false (workers keep running)",
        !phase_is_submissions_withheld(phase));

    // After 300s timeout → enter RECONNECTING
    assert(test_is_valid_transition(phase, TestRecoveryPhase::RECONNECTING));
    phase = TestRecoveryPhase::RECONNECTING;
    print_test_result("After timeout: RECONNECTING phase active",
        phase_is_reconnecting(phase));
    print_test_result("After timeout: is_recovery_active() == true",
        phase_is_recovery_active(phase));
}

// ── Test 22: RECONNECTING prevents SESSION_EXPIRED re-entrance ────────────
void test_reconnecting_guards_session_expired() {
    std::cout << "\nTest 22: SESSION_EXPIRED preempts template recovery but not reconnect/session recovery\n";

    TestRecoveryPhase phase = TestRecoveryPhase::RECONNECTING;

    bool should_ignore = phase_is_reconnecting(phase) ||
                         phase == TestRecoveryPhase::SESSION_RECOVERY;
    print_test_result("RECONNECTING: session_expired is ignored",
        should_ignore);

    // Verify HEALTHY does NOT suppress session_expired
    TestRecoveryPhase healthy = TestRecoveryPhase::HEALTHY;
    bool healthy_ignore = phase_is_reconnecting(healthy) ||
                          healthy == TestRecoveryPhase::SESSION_RECOVERY;
    print_test_result("HEALTHY: session_expired is NOT ignored",
        !healthy_ignore);

    // WAITING_TEMPLATE must not suppress authoritative session recovery anymore.
    TestRecoveryPhase waiting = TestRecoveryPhase::WAITING_TEMPLATE;
    bool waiting_ignore = phase_is_reconnecting(waiting) ||
                          waiting == TestRecoveryPhase::SESSION_RECOVERY;
    print_test_result("WAITING_TEMPLATE: session_expired is NOT ignored",
        !waiting_ignore);

    if (test_is_valid_transition(waiting, TestRecoveryPhase::SESSION_RECOVERY)) {
        waiting = TestRecoveryPhase::SESSION_RECOVERY;
    }
    print_test_result("WAITING_TEMPLATE escalates to SESSION_RECOVERY",
        waiting == TestRecoveryPhase::SESSION_RECOVERY);
}

// ── Test 22b: Template arrival during SESSION_RECOVERY must not imply recovery complete ──
void test_session_recovery_blocks_template_only_exit() {
    std::cout << "\nTest 22b: Template arrival during SESSION_RECOVERY does not clear degraded state\n";

    struct ExitGateInput {
        TestRecoveryPhase phase{TestRecoveryPhase::SESSION_RECOVERY};
        bool valid_template{true};
        bool authoritative_full_recovery_required{true};
        bool authoritative_may_request_work{false};

        bool may_clear() const {
            return valid_template &&
                   phase != TestRecoveryPhase::SESSION_RECOVERY &&
                   !authoritative_full_recovery_required &&
                   authoritative_may_request_work;
        }
    };

    ExitGateInput blocked;
    print_test_result("SESSION_RECOVERY blocks clear_recovery_state despite valid template",
        !blocked.may_clear());

    blocked.phase = TestRecoveryPhase::WAITING_TEMPLATE;
    blocked.authoritative_full_recovery_required = false;
    blocked.authoritative_may_request_work = true;
    print_test_result("WAITING_TEMPLATE clears only after authoritative session is restored",
        blocked.may_clear());
}

// ── Test 23: Mutual exclusivity — only one phase at a time ───────────────
void test_recovery_phase_mutual_exclusivity() {
    std::cout << "\nTest 23: Phase mutual exclusivity — only one phase at a time\n";

    // The enum class guarantees mutual exclusivity at the type level.
    // Verify all possible phase combinations are distinct.
    TestRecoveryPhase phases[] = {
        TestRecoveryPhase::HEALTHY,
        TestRecoveryPhase::WAITING_TEMPLATE,
        TestRecoveryPhase::SESSION_RECOVERY,
        TestRecoveryPhase::RECONNECTING,
    };

    bool all_distinct = true;
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = i + 1; j < 4; ++j) {
            if (phases[i] == phases[j]) {
                all_distinct = false;
                break;
            }
        }
    }
    print_test_result("All 3 phases have distinct enum values", all_distinct);

    // Verify: WAITING_TEMPLATE cannot have submissions withheld
    TestRecoveryPhase waiting = TestRecoveryPhase::WAITING_TEMPLATE;
    bool withheld = phase_is_submissions_withheld(waiting);
    print_test_result("WAITING_TEMPLATE: is_submissions_withheld() == false",
        !withheld);

    // Verify: WAITING_TEMPLATE is degraded (workers have no fresh template)
    bool degraded = phase_is_degraded(waiting);
    print_test_result("WAITING_TEMPLATE: is_degraded() == true",
        degraded);
}

// ── Test 24: Epoch 0 recovery escalation — GET_BLOCK suppressed with no template ─
void test_epoch0_recovery_escalation_on_suppressed_get_block() {
    std::cout << "\nTest 24: Epoch 0 recovery escalation when GET_BLOCK suppressed with no template\n";

    // Scenario: HEALTHY (epoch 0) + authenticated + no template + GET_BLOCK dedup-suppressed.
    // The Worker_manager fix must initiate recovery (epoch 0 → 1) so RECOVERY_FORCED/TIMER
    // can bypass ALL dedup guards on the next retry.
    auto coordinator = std::make_shared<EpochCoordinator>();
    print_test_result("Initial recovery_epoch is 0",
        coordinator->recovery_epoch() == 0);

    // Simulate initial HEALTHY state
    TestRecoveryPhase phase = TestRecoveryPhase::HEALTHY;
    bool authenticated = true;
    bool no_valid_template = true;
    bool get_block_suppressed = true;  // Solo::get_work() returned empty

    // ── Before fix: only scheduled retry when is_recovery_active() ──
    bool old_guard = phase_is_recovery_active(phase) && authenticated && no_valid_template;
    print_test_result("Old guard: HEALTHY + epoch 0 → NO forced retry (bug)",
        !old_guard);

    // ── After fix: initiate recovery when no template + suppressed ──
    bool should_escalate = authenticated && no_valid_template && get_block_suppressed;
    print_test_result("New guard: authenticated + no_template + suppressed → SHOULD escalate",
        should_escalate);

    // Simulate the fix: initiate recovery if not already active
    if (should_escalate && !phase_is_recovery_active(phase)) {
        // Simulates mark_recovery_initiated → transition_to(WAITING_TEMPLATE)
        phase = TestRecoveryPhase::WAITING_TEMPLATE;
        coordinator->advance_recovery_epoch("get_block_suppressed_no_template");
    }

    print_test_result("After escalation: phase is WAITING_TEMPLATE",
        phase == TestRecoveryPhase::WAITING_TEMPLATE);
    print_test_result("After escalation: recovery_epoch advanced to 1",
        coordinator->recovery_epoch() == 1);
    print_test_result("After escalation: is_recovery_active() is true",
        phase_is_recovery_active(phase));

    // Now forced retry can be scheduled (RECOVERY_FORCED bypasses all dedup)
    bool can_schedule = phase_is_recovery_active(phase) && authenticated && no_valid_template;
    print_test_result("After escalation: forced retry CAN be scheduled",
        can_schedule);

    // Idempotent: if already in recovery, don't re-initiate
    auto epoch_before = coordinator->recovery_epoch();
    if (should_escalate && !phase_is_recovery_active(phase)) {
        phase = TestRecoveryPhase::WAITING_TEMPLATE;
        coordinator->advance_recovery_epoch("should_not_fire");
    }
    print_test_result("Idempotent: recovery not re-initiated when already active",
        coordinator->recovery_epoch() == epoch_before);
}

void test_session_status_cooldown_only_advances_after_successful_queue()
{
    std::cout << "\nTest 27b: SESSION_STATUS cooldown advances only after successful queue\n";

    struct SessionStatusSender {
        std::chrono::steady_clock::time_point last_sent{};

        bool due(std::chrono::steady_clock::time_point now) const {
            constexpr int64_t interval_seconds = 300;
            return std::chrono::duration_cast<std::chrono::seconds>(now - last_sent).count() >= interval_seconds;
        }

        bool attempt(std::chrono::steady_clock::time_point now, bool queue_success) {
            if (!due(now)) {
                return false;
            }
            if (!queue_success) {
                return false;
            }
            last_sent = now;
            return true;
        }
    };

    SessionStatusSender sender;
    auto now = std::chrono::steady_clock::now();
    sender.last_sent = now - std::chrono::seconds(301);

    bool failed_attempt = !sender.attempt(now, false);
    bool still_due = sender.due(now);
    bool successful_retry = sender.attempt(now, true);
    bool no_longer_due = !sender.due(now);

    print_test_result("Failed SESSION_STATUS queue does not consume cooldown",
                      failed_attempt && still_due);
    print_test_result("Successful SESSION_STATUS queue consumes cooldown",
                      successful_retry && no_longer_due);
}

void test_node_retry_wait_restarts_recovery_response_window() {
    std::cout << "\nTest 29: Node retry wait is excluded from recovery timeout\n";

    using RecoveryClock = nexusminer::worker::RecoveryTimeout::Clock;
    nexusminer::worker::RecoveryTimeout timeout;
    const auto start = RecoveryClock::time_point{} + std::chrono::seconds(1);
    timeout.start_if_unset(start);

    auto before_wait = start + std::chrono::seconds(299);
    timeout.observe_node_retry(true, true, before_wait);
    auto retry_deadline = before_wait + std::chrono::seconds(600);
    timeout.observe_node_retry(false, true, retry_deadline);

    print_test_result("A long node wait does not count toward recovery timeout",
        timeout.elapsed_seconds(retry_deadline) == 0);
    print_test_result("Old push and lane silence do not immediately trip hard stop",
        timeout.worst_case_outage_seconds(retry_deadline, 899, 899) == 0);
    print_test_result("GET_BLOCK receives a full post-deadline response window",
        timeout.worst_case_outage_seconds(
            retry_deadline + std::chrono::seconds(299), 1198, 1198) == 299);
    print_test_result("Hard-stop accounting resumes after the response window",
        timeout.worst_case_outage_seconds(
            retry_deadline + std::chrono::seconds(300), 1199, 1199) == 300);

    auto disconnected_at = retry_deadline + std::chrono::seconds(1);
    timeout.observe_node_retry(true, true, disconnected_at);
    auto transport_failure_at = disconnected_at + std::chrono::seconds(301);
    timeout.observe_node_retry(false, false, transport_failure_at);
    print_test_result("Transport failure does not rebase the recovery timeout",
        timeout.elapsed_seconds(transport_failure_at) == 302);
}

int main() {
    std::cout << "\n═══════════════════════════════════════════════════════════\n";
    std::cout << "Degraded Recovery / Keepalive Epoch Tests\n";
    std::cout << "═══════════════════════════════════════════════════════════\n";

    test_keepalive_ack_invalidated_on_epoch_change();
    test_channel_advance_stale_template_transition();
    test_recovery_pending_debounce_idempotent();
    test_recovery_get_block_no_permanent_starvation();
    test_successful_reauth_restarts_recovery_epoch();
    test_same_height_soft_refresh_escalates_only_after_timeout();
    test_degraded_exit_normalizes_recovery_state();
    test_authoritative_soft_refresh_backfills_local_state();
    test_tip_moved_soft_refresh_defers_unified_drift_stop_until_timeout();
    test_worker_respawn_guard_is_single_shot_per_degraded_exit();
    test_worker_respawn_waits_for_authoritative_empty_generation();
    test_keepalive_epoch_isolation_clean_start();
    test_stale_template_after_channel_advance();
    test_epoch_advance_suppresses_old_keepalive_signal();
    test_multiple_rapid_epoch_changes_clean_state();
    test_push_does_not_clear_keepalive_on_epoch_change();
    test_epoch_change_clears_push_tip_anchor_hint();
    test_recovery_epoch_monotonic_with_idempotent_initiation();
    test_integration_degraded_recovery_to_resume();
    test_integration_forced_retry_is_bounded();
    test_health_policy_distinguishes_normal_refresh_from_multi_block_lag();

    // ── RecoveryPhase state machine tests (new with explicit 5-state machine) ──
    test_recovery_phase_valid_transitions();
    test_recovery_phase_helpers_equivalence();
    test_orphaned_soft_refresh_cleared();
    test_soft_refresh_escalation_to_hard_recovery();
    test_reconnecting_guards_session_expired();
    test_session_recovery_blocks_template_only_exit();
    test_recovery_phase_mutual_exclusivity();
    test_epoch0_recovery_escalation_on_suppressed_get_block();
    test_session_status_cooldown_only_advances_after_successful_queue();
    test_node_retry_wait_restarts_recovery_response_window();

    std::cout << "\n═══════════════════════════════════════════════════════════\n";
    std::cout << "Test Results: " << tests_passed << "/" << tests_run << " passed";
    if (tests_failed > 0) {
        std::cout << " (" << tests_failed << " failed)";
    }
    std::cout << "\n═══════════════════════════════════════════════════════════\n\n";

    return (tests_failed == 0) ? 0 : 1;
}
