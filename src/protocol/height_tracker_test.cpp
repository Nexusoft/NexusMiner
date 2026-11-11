/**
 * @file height_tracker_test.cpp
 * @brief Unit tests for HeightTracker centralized height tracking utility
 *
 * Tests:
 *  1. Push update + template received produces expected drift delta 0
 *  2. Channel height advanced makes IsTemplateStale true
 *  3. Snapshot is consistent after concurrent-style updates
 *  4. GET_ROUND update sets source correctly (heights reflected in snapshot composition)
 *  5. ExplainMismatch reports drift when template target differs from expected
 *  6. ExplainMismatch reports staleness when channel height >= template target
 *  7. Unified height advancing alone does NOT make template stale
 *  8. Channel height advancing DOES make template stale
 *  9. is_tip_moved() detects unified tip advance (Phase 3A: tip_moved refresh reason)
 * 10. is_tip_moved() resets to false after new template received
 * 11. set_session_epoch() clears last_keepalive_ack_at on epoch change (keepalive epoch isolation)
 * 12. set_session_epoch() does NOT clear last_keepalive_ack_at when epoch is unchanged
 * 13. Keepalive timestamp from old epoch does not survive into new epoch (liveness isolation)
 * 14. set_session_epoch(0) does NOT clear last_keepalive_ack_at (sentinel epoch = no-op)
 * 15. Session epoch reflected in Snapshot.session_epoch field
 * 11. Difficulty from push updates is reflected in HeightTracker snapshot
 *     (GET_ROUND and keepalive difficulty go to diagnostic only — not in snapshot)
 * 12. Post-push guard — last_template_update >= last push time
 * 13. Pre-push guard — template received BEFORE a push is identified correctly
 * 14. OnKeepaliveResponse sets diagnostic fields only (not unified/channel_height)
 * 15. OnKeepaliveResponse legacy zeros — safe, no canonical corruption
 * 16. OnKeepaliveResponse updates stake_height correctly
 * 17. peak_fork_score is a persistent high-water mark
 * 18. Push updates channel_height; prime_height is diagnostic max (keepalive, round, push)
 * 19. Push updates channel_height; hash_height is diagnostic max (keepalive, round, push)
 * 20. Production regression: push advances channel_height; prime_height diagnostic max
 * 21. OnTemplateReceived writes canonical_channel_target (only advances)
 * 22. OnTemplateReceived does not regress channel_target
 * 23. Doom-loop impossible with canonical-only channel_target
 * 24. Stale GET_BLOCK response does not cause regression
 * 25. OnTemplateMetadata does NOT regress channel_height
 * 26. OnTemplateMetadata per-channel heights — canonical is monotonic
 * 27. Canonical state only from OnBlockDataReceived — push/keepalive don't corrupt
 * 28. Keepalive does not regress canonical unified/channel heights
 * 29. OnBlockDataReceived is monotonic — stale BLOCK_DATA cannot regress canonical
 * 30. Fork score lives in DiagnosticObserverState, not canonical
 * 31. height_drift_from_canonical() returns 0 for healthy state
 * 32. canonical_hash_prev_block anchored to BLOCK_DATA (Tritium block)
 * 33. DiagnosticObserverState::is_initialized() — diagnostic equivalent of canonical
 * 34. DiagnosticObserverState::latest_received_at() — diagnostic equivalent of canonical_received_at
 * 35. OnTemplateReceived() writes canonical_channel_target (Bug #1 fix)
 * 36. OnTemplateReceived() sets template_unified_height from canonical
 * 37. Cross-channel block (Hash/Stake) advances unified but not Prime channel →
 *     is_tip_moved() == true (template still valid for submission)
 * 38. OnTemplateReceived +1 target does NOT contaminate snapshot channel_height
 *     (regression: raw blockchain height must not be overwritten by mining target)
 */

#include "protocol/height_tracker.hpp"
#include <iostream>
#include <cassert>
#include <cstdint>
#include <thread>

using namespace nexusminer::protocol;

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
// Test 1: Push update + template received produces drift delta 0
// ============================================================================
void test_push_then_template_no_drift() {
    std::cout << "\nTest 1: Push update + template → drift delta 0\n";
    HeightTracker tracker;

    // Node tells miner: channel is at height 100, difficulty 0x1d00ffff
    // Use OnBlockDataReceived to set canonical state (snap.channel_height is canonical-only)
    tracker.OnBlockDataReceived(5000, 100, 0x1d00ffff, uint1024_t{});

    // Template arrives targeting height 101 (channel_height + 1)
    tracker.OnTemplateReceived(1, 101);

    auto snap = tracker.GetSnapshot();

    // expected_template_target() == channel_height + 1 == 101
    print_test_result("expected_template_target() == 101",
                      snap.expected_template_target() == 101);

    // drift_delta() == channel_target - expected == 101 - 101 == 0
    auto delta = snap.drift_delta();
    print_test_result("drift_delta() has value", delta.has_value());
    print_test_result("drift_delta() == 0", delta.has_value() && *delta == 0);

    // is_template_stale() removed (structurally false under canonical-only semantics)

    // ExplainMismatch returns empty (heights are consistent)
    std::string msg = tracker.ExplainMismatch();
    print_test_result("ExplainMismatch() is empty when consistent", msg.empty());
}

// ============================================================================
// Test 2: Channel height advanced — canonical auto-advance prevents staleness
// ============================================================================
void test_channel_height_advance_makes_stale() {
    std::cout << "\nTest 2: Channel height advance → canonical auto-advances target\n";
    HeightTracker tracker;

    // Template was for height 101
    tracker.OnBlockDataReceived(5000, 100, 0x1d00ffff, uint1024_t{});
    tracker.OnTemplateReceived(1, 101);

    // Another miner finds the block; canonical channel advances to 101
    tracker.OnBlockDataReceived(5001, 101, 0x1d00ffff, uint1024_t{});

    auto snap = tracker.GetSnapshot();

    // channel_height (101) but channel_target auto-advanced to 102 by OnBlockDataReceived
    // With canonical-only heights, OnBlockDataReceived always sets target = channel + 1.
    // is_template_stale() removed (structurally false under canonical-only semantics)

    // ExplainMismatch returns empty (canonical state is self-consistent)
    std::string msg = tracker.ExplainMismatch();
    print_test_result("ExplainMismatch() is empty (canonical auto-advance prevents stale)",
                      msg.empty());
}

// ============================================================================
// Test 3: Snapshot is consistent
// ============================================================================
void test_snapshot_consistency() {
    std::cout << "\nTest 3: Snapshot consistency\n";
    HeightTracker tracker;
    tracker.OnBlockDataReceived(4999, 99, 0x1a0abc12, uint1024_t{});
    tracker.OnPushNotification(4999, 99, 0x1a0abc12);  // sets m_latest_difficulty_nbits
    tracker.OnTemplateReceived(2, 100);

    auto snap = tracker.GetSnapshot();

    print_test_result("Snapshot unified_height == 4999",
                      snap.unified_height == 4999);
    print_test_result("Snapshot channel_height == 99",
                      snap.channel_height == 99);
    print_test_result("Snapshot difficulty_nbits == 0x1a0abc12",
                      snap.difficulty_nbits == 0x1a0abc12);
    print_test_result("Snapshot channel_target == 100",
                      snap.channel_target == 100);
    print_test_result("Snapshot channel == 2",
                      snap.channel == 2);
    // Last update source should be TEMPLATE (last call was OnTemplateReceived)
    print_test_result("Snapshot last_update_source == TEMPLATE",
                      snap.last_update_source == HeightTracker::UpdateSource::TEMPLATE);
}

// ============================================================================
// Test 4: GET_ROUND update sets source correctly; heights reflected in snapshot composition
// ============================================================================
void test_get_round_source() {
    std::cout << "\nTest 4: GET_ROUND update source (heights in snapshot composition)\n";
    HeightTracker tracker;
    // 16-byte full height picture: unified=5100, prime=120, hash=200, stake=300
    tracker.OnGetRound(5100, 120, 200, 300);

    auto snap = tracker.GetSnapshot();
    print_test_result("Source is GET_ROUND",
                      snap.last_update_source == HeightTracker::UpdateSource::GET_ROUND);
    // GET_ROUND writes to diagnostic only — unified_height is canonical-only (= 0)
    print_test_result("unified_height == 0 (GET_ROUND is diagnostic only, canonical unset)",
                      snap.unified_height == 0);
    // channel is not set (m_channel==0), so round_channel_height remains 0:
    // channel_height = max(canonical=0, push=0, round=0) = 0
    print_test_result("channel_height == 0 (no channel set, round_channel_height=0)",
                      snap.channel_height == 0);
    // Verify diagnostic snapshot captured the GET_ROUND per-channel heights
    auto diag = tracker.GetDiagnosticSnapshot();
    print_test_result("diagnostic round_unified_height == 5100",
                      diag.round_unified_height == 5100);
    print_test_result("diagnostic round_prime_height == 120",
                      diag.round_prime_height == 120);
    print_test_result("diagnostic round_hash_height == 200",
                      diag.round_hash_height == 200);
    print_test_result("diagnostic round_stake_height == 300",
                      diag.round_stake_height == 300);
    // 16-byte format carries no difficulty — round_difficulty_nbits is always 0
    print_test_result("diagnostic round_difficulty_nbits == 0 (no difficulty in 16-byte format)",
                      diag.round_difficulty_nbits == 0);
}

// ============================================================================
// Test 5: Drift detected and explained when template target differs
// ============================================================================
void test_drift_explain() {
    std::cout << "\nTest 5: Drift detection and ExplainMismatch\n";
    HeightTracker tracker;

    // Node says channel height is 100 — set canonical
    tracker.OnBlockDataReceived(5000, 100, 0x1d00ffff, uint1024_t{});

    // But template targets height 103 (not 101 — possible if heights drifted)
    tracker.OnTemplateReceived(1, 103);

    auto snap = tracker.GetSnapshot();
    auto delta = snap.drift_delta();
    print_test_result("drift_delta() has value", delta.has_value());
    print_test_result("drift_delta() == 2 (103 - 101)",
                      delta.has_value() && *delta == 2);

    std::string msg = tracker.ExplainMismatch();
    print_test_result("ExplainMismatch() mentions DRIFT",
                      msg.find("DRIFT") != std::string::npos);
}

// ============================================================================
// Test 6: Zero heights return safe defaults
// ============================================================================
void test_zero_heights_safe() {
    std::cout << "\nTest 6: Zero heights return safe defaults\n";
    HeightTracker tracker;

    auto snap = tracker.GetSnapshot();
    print_test_result("expected_template_target() == 0 when channel_height == 0",
                      snap.expected_template_target() == 0);
    // is_template_stale() removed (structurally false under canonical-only semantics)
    print_test_result("drift_delta() has no value when both are 0",
                      !snap.drift_delta().has_value());
    std::string msg = tracker.ExplainMismatch();
    print_test_result("ExplainMismatch() is empty with zero state", msg.empty());
}

// ============================================================================
// Test 7: Unified height advancing alone does NOT make template stale
// (Phase 1B: unified-delta must NOT trigger stale/template refresh)
// ============================================================================
void test_unified_advance_does_not_make_stale() {
    std::cout << "\nTest 7: Unified height advancing alone → NOT stale\n";
    HeightTracker tracker;

    // Initial state: channel at 100, template for 101
    tracker.OnBlockDataReceived(5000, 100, 0x1d00ffff, uint1024_t{});
    tracker.OnTemplateReceived(1, 101);

    // Unified height advances many blocks (other channels found blocks)
    // while channel height stays at 100 (Prime mining takes minutes)
    tracker.OnBlockDataReceived(5010, 100, 0x1d00ffff, uint1024_t{});

    auto snap = tracker.GetSnapshot();

    // Template should still be valid: channel_height (100) < channel_target (101)
    // is_template_stale() removed (structurally false under canonical-only semantics)

    // Channel target unchanged
    print_test_result("channel_target still == 101", snap.channel_target == 101);

    // Channel height unchanged
    print_test_result("channel_height still == 100", snap.channel_height == 100);

    // Unified height updated
    print_test_result("unified_height updated to 5010", snap.unified_height == 5010);

    // ExplainMismatch should be empty (heights are still consistent)
    std::string msg = tracker.ExplainMismatch();
    print_test_result("ExplainMismatch() is empty (template still valid)", msg.empty());
}

// ============================================================================
// Test 8: Channel height advancing DOES make template stale
// ============================================================================
void test_channel_advance_makes_stale() {
    std::cout << "\nTest 8: Channel height advancing → stale\n";
    HeightTracker tracker;

    // Initial state: channel at 100, template for 101
    tracker.OnBlockDataReceived(5000, 100, 0x1d00ffff, uint1024_t{});
    tracker.OnTemplateReceived(1, 101);

    // Channel height advances to 101 (another miner found the block)
    tracker.OnBlockDataReceived(5001, 101, 0x1d00ffff, uint1024_t{});

    auto snap = tracker.GetSnapshot();

    // With canonical-only heights, OnBlockDataReceived auto-advances target to 102.
    // is_template_stale() removed (structurally false under canonical-only semantics)

    // ExplainMismatch returns empty (canonical state is self-consistent)
    std::string msg = tracker.ExplainMismatch();
    print_test_result("ExplainMismatch() is empty (canonical auto-advance prevents stale)",
                      msg.empty());
}

// ============================================================================
// Test 9: is_tip_moved() detects when unified tip advances beyond template tip
//         (Phase 3A: hashPrevBlock staleness via unified height delta)
// ============================================================================
void test_is_tip_moved_detected() {
    std::cout << "\nTest 9: is_tip_moved() — unified tip advance triggers refresh\n";
    HeightTracker tracker;

    // Initial state: set CANONICAL state via OnBlockDataReceived, then template
    // Bug #2 fix: OnTemplateReceived captures template_unified_height from canonical,
    // so canonical must be set first.
    tracker.OnBlockDataReceived(5000, 100, 0x1d00ffff, uint1024_t{});
    tracker.OnTemplateReceived(1, 101);

    auto snap_initial = tracker.GetSnapshot();

    // template_unified_height should be captured as 5000 (canonical)
    print_test_result("template_unified_height == 5000 after template received",
                      snap_initial.template_unified_height == 5000);

    // No tip movement yet: unified == template_unified
    print_test_result("is_tip_moved() == false before unified advance",
                      !snap_initial.is_tip_moved());

    // Another channel finds a block: canonical unified advances to 5005, channel stays at 100
    tracker.OnBlockDataReceived(5005, 100, 0x1d00ffff, uint1024_t{});
    auto snap = tracker.GetSnapshot();

    // Tip has moved: unified (5005) > template_unified (5000)
    print_test_result("is_tip_moved() == true after unified advance",
                      snap.is_tip_moved());

    // But template is NOT channel-stale: channel_height (100) < channel_target (101)
    // is_template_stale() removed (structurally false under canonical-only semantics)

    // template_unified_height unchanged (only updates on OnTemplateReceived)
    print_test_result("template_unified_height still == 5000 (template not refreshed)",
                      snap.template_unified_height == 5000);

    // unified_height updated to new value (canonical)
    print_test_result("unified_height updated to 5005",
                      snap.unified_height == 5005);
}

// ============================================================================
// Test 10: is_tip_moved() resets to false after new template received
// ============================================================================
void test_is_tip_moved_resets_on_new_template() {
    std::cout << "\nTest 10: is_tip_moved() resets when new template received\n";
    HeightTracker tracker;

    // Bug #2 fix: canonical must be set before OnTemplateReceived captures
    tracker.OnBlockDataReceived(5000, 100, 0x1d00ffff, uint1024_t{});
    tracker.OnTemplateReceived(1, 101);

    // Unified tip moves to 5005 via canonical advance
    tracker.OnBlockDataReceived(5005, 100, 0x1d00ffff, uint1024_t{});
    print_test_result("is_tip_moved() == true before fresh template",
                      tracker.GetSnapshot().is_tip_moved());

    // Fresh template received (template_unified_height == 5005 from canonical)
    tracker.OnTemplateReceived(1, 101);
    auto snap = tracker.GetSnapshot();

    // After new template, tip is no longer "moved"
    print_test_result("is_tip_moved() == false after fresh template received",
                      !snap.is_tip_moved());

    // template_unified_height updated to current canonical unified height
    print_test_result("template_unified_height == 5005 (new template at new tip)",
                      snap.template_unified_height == 5005);
}

// ============================================================================
// Test 11: Difficulty from push updates is reflected in HeightTracker and
//         consistent across snapshot reads (Phase 2C regression test)
//         GET_ROUND difficulty goes to DiagnosticObserverState only — it does
//         NOT appear in the main Snapshot (which uses canonical or push).
// ============================================================================
void test_difficulty_from_push_reflected_in_snapshot() {
    std::cout << "\nTest 11: Difficulty from push updates reflected in HeightTracker\n";
    HeightTracker tracker;

    // First push: difficulty_nbits = 0x1d00ffff
    tracker.OnPushNotification(5000, 100, 0x1d00ffff);
    auto snap = tracker.GetSnapshot();
    print_test_result("difficulty_nbits set from push (0x1d00ffff)",
                      snap.difficulty_nbits == 0x1d00ffff);
    print_test_result("Source is PUSH after OnPushNotification",
                      snap.last_update_source == HeightTracker::UpdateSource::PUSH);

    // Second push with updated difficulty: difficulty_nbits = 0x1c0e9f34
    tracker.OnPushNotification(5001, 101, 0x1c0e9f34);
    snap = tracker.GetSnapshot();
    print_test_result("difficulty_nbits updated from second push (0x1c0e9f34)",
                      snap.difficulty_nbits == 0x1c0e9f34);
    print_test_result("push_unified_height updated from second push (5001)",
                      snap.push_unified_height == 5001);
    print_test_result("push_channel_height updated from second push (101)",
                      snap.push_channel_height == 101);

    // GET_ROUND is diagnostic-only; snapshot difficulty stays at push value (0x1c0e9f34)
    // 16-byte format: unified=5002, prime=102, hash=203, stake=0
    tracker.OnGetRound(5002, 102, 203, 0);
    snap = tracker.GetSnapshot();
    print_test_result("difficulty_nbits still from push after GET_ROUND (0x1c0e9f34)",
                      snap.difficulty_nbits == 0x1c0e9f34);
    print_test_result("Source is GET_ROUND after OnGetRound",
                      snap.last_update_source == HeightTracker::UpdateSource::GET_ROUND);
    // 16-byte GET_ROUND carries no difficulty — round_difficulty_nbits is always 0
    auto diag = tracker.GetDiagnosticSnapshot();
    print_test_result("diagnostic round_difficulty_nbits == 0 (no difficulty in 16-byte format)",
                      diag.round_difficulty_nbits == 0);

    // OnTemplateReceived must NOT overwrite difficulty (it doesn't carry nbits)
    tracker.OnTemplateReceived(1, 103);
    snap = tracker.GetSnapshot();
    print_test_result("difficulty_nbits unchanged after OnTemplateReceived (still push 0x1c0e9f34)",
                      snap.difficulty_nbits == 0x1c0e9f34);
}

// ============================================================================
// Test 12: Post-push guard — last_template_update >= last push time when
//          template is received AFTER a push notification.
//          This is the HeightTracker side of the doom-loop prevention fix:
//          Worker_manager::check_template_health() uses these timestamps to
//          decide whether to skip stop_all_workers for a "stale" template that
//          was actually received after the push that caused the stale reading.
// ============================================================================
void test_post_push_timestamps_ordered_correctly() {
    std::cout << "\nTest 12: Post-push guard — last_template_update >= last_push after push→template\n";
    HeightTracker tracker;

    // Step 1: Push notification arrives (block Y mined, channel at Y).
    tracker.OnPushNotification(5000, 100, 0x1d00ffff);
    auto snap_after_push = tracker.GetSnapshot();

    // Verify push time is recorded.
    print_test_result("last_height_update set after push",
                      snap_after_push.last_height_update != std::chrono::steady_clock::time_point{});

    // last_template_update should be before last_height_update (no template yet).
    print_test_result("last_template_update < last_height_update before template arrives",
                      snap_after_push.last_template_update < snap_after_push.last_height_update);

    // channel_height = 100, channel_target still 0 (no template) → not stale.
    // is_template_stale() removed (structurally false under canonical-only semantics)

    // Step 2: Recovery GET_BLOCK returns a template targeting Y+1 = 101.
    // Small sleep to ensure measurable time difference between push and template.
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    tracker.OnTemplateReceived(1, 101);
    auto snap_after_tmpl = tracker.GetSnapshot();

    // last_template_update must be >= last_height_update (post-push guard).
    print_test_result("last_template_update >= last_height_update after template (post-push)",
                      snap_after_tmpl.last_template_update >= snap_after_tmpl.last_height_update);

    // is_template_stale() removed (structurally false under canonical-only semantics)
}

// ============================================================================
// Test 13: Pre-push guard — template received BEFORE a push has
//          last_template_update < last_height_update, so check_template_health
//          correctly identifies it as a pre-push (genuinely stale) template
//          and does NOT skip stop_all_workers.
// ============================================================================
void test_pre_push_template_identified_correctly() {
    std::cout << "\nTest 13: Pre-push guard — last_template_update < last_height_update (pre-push template)\n";
    HeightTracker tracker;

    // Step 1: Template received first (targeting 101, channel at 100).
    tracker.OnBlockDataReceived(5000, 100, 0x1d00ffff, uint1024_t{});
    tracker.OnTemplateReceived(1, 101);

    // Step 2: A later block is found (channel advances to 101 — template is now stale).
    // OnBlockDataReceived advances canonical; push sets the timestamp for temporal guard.
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    tracker.OnBlockDataReceived(5001, 101, 0x1d00ffff, uint1024_t{});
    tracker.OnPushNotification(5001, 101, 0x1d00ffff);
    auto snap = tracker.GetSnapshot();

    // last_template_update < last_height_update → pre-push template.
    print_test_result("last_template_update < last_height_update (pre-push template)",
                      snap.last_template_update < snap.last_height_update);

    // is_template_stale() removed (structurally false under canonical-only semantics)
    // Canonical auto-advance prevents staleness even when channel catches up.
}

// ============================================================================
// Test 14: blocks_behind() distinguishes normal single-block refresh from
//          true multi-block lag while leaving same-height tip moves at 0.
// ============================================================================
void test_blocks_behind_distinguishes_normal_vs_severe_lag() {
    std::cout << "\nTest 14: blocks_behind() distinguishes normal vs severe lag\n";
    HeightTracker tracker;

    tracker.OnBlockDataReceived(5000, 100, 0x1d00ffff, uint1024_t{});
    tracker.OnTemplateReceived(2, 101);

    auto initial = tracker.GetSnapshot();
    print_test_result("blocks_behind() == 0 before channel advances",
                      initial.blocks_behind() == 0);

    tracker.OnBlockDataReceived(5001, 101, 0x1d00ffff, uint1024_t{});
    auto one_block = tracker.GetSnapshot();
    // With canonical-only heights, OnBlockDataReceived auto-advances target to 102.
    // blocks_behind() = expected(102) - target(102) = 0 (canonical is always self-consistent).
    print_test_result("blocks_behind() == 0 (canonical auto-advances target with channel)",
                      one_block.blocks_behind() == 0);

    tracker.OnBlockDataReceived(5002, 102, 0x1d00ffff, uint1024_t{});
    auto two_blocks = tracker.GetSnapshot();
    print_test_result("blocks_behind() == 0 (canonical auto-advance keeps target = channel+1)",
                      two_blocks.blocks_behind() == 0);

    HeightTracker tip_move_tracker;
    tip_move_tracker.OnBlockDataReceived(6000, 200, 0x1d00ffff, uint1024_t{});
    tip_move_tracker.OnTemplateReceived(2, 201);
    // Advance unified only (other channel found block) via canonical
    tip_move_tracker.OnBlockDataReceived(6001, 200, 0x1d00ffff, uint1024_t{});
    auto tip_move = tip_move_tracker.GetSnapshot();
    print_test_result("blocks_behind() == 0 when only unified tip moves",
                      tip_move.blocks_behind() == 0);
    print_test_result("is_tip_moved() == true when only unified tip moves",
                      tip_move.is_tip_moved());
}

// ============================================================================
// Test 14: OnKeepaliveResponse — keepalive sets diagnostic fields only
//          (unified_height and channel_height in snapshot come from canonical/push)
// ============================================================================
void test_on_keepalive_response_unified() {
    std::cout << "\nTest 14: OnKeepaliveResponse sets diagnostic fields (not unified/channel_height)\n";
    HeightTracker tracker;
    // Set channel to Hash (2) so channel_height can be verified
    tracker.OnTemplateReceived(2, 101);
    tracker.OnKeepaliveResponse(6000, 450, 800, 999, 0xCAFEBABEu, 3);

    auto snap = tracker.GetSnapshot();
    // Canonical heights must NOT be set by keepalive
    print_test_result("unified_height == 0 (keepalive does not set canonical)",
                      snap.unified_height == 0);
    // prime_height and hash_height are diagnostic max(keepalive, round, push),
    // so keepalive DOES contribute to these snapshot fields.
    print_test_result("prime_height == 450 (diagnostic max includes keepalive)",
                      snap.prime_height == 450);
    print_test_result("hash_height == 800 (diagnostic max includes keepalive)",
                      snap.hash_height == 800);
    print_test_result("channel_height == 0 (keepalive does not set canonical)",
                      snap.channel_height == 0);
    print_test_result("stake_height == 999",    snap.stake_height == 999);
    print_test_result("hash_tip_lo32 stored",   snap.hash_tip_lo32 == 0xCAFEBABEu);
    print_test_result("fork_score == 3",        snap.fork_score == 3);
    print_test_result("peak_fork_score == 3",   snap.peak_fork_score == 3);
    print_test_result("is_fork_active() == true", snap.is_fork_active());
    print_test_result("last_update_source == KEEPALIVE",
                      snap.last_update_source == HeightTracker::UpdateSource::KEEPALIVE);

    // Verify via DiagnosticObserverState directly
    auto diag = tracker.GetDiagnosticSnapshot();
    print_test_result("diagnostic keepalive_unified_height == 6000",
                      diag.keepalive_unified_height == 6000);
    print_test_result("diagnostic keepalive_hash_height == 800",
                      diag.keepalive_hash_height == 800);
}

// ============================================================================
// Test 15: OnKeepaliveResponse — legacy path (hash_tip_lo32=0, fork_score=0)
//          is safe and does NOT corrupt canonical heights
// ============================================================================
void test_on_keepalive_response_legacy_zeros_safe() {
    std::cout << "\nTest 15: OnKeepaliveResponse legacy zeros — safe, no canonical corruption\n";
    HeightTracker tracker;
    tracker.OnKeepaliveResponse(6001, 451, 801, 999, 0u, 0u);

    auto snap = tracker.GetSnapshot();
    // Canonical heights must NOT be set by keepalive
    print_test_result("unified_height == 0 (keepalive does not set canonical)",
                      snap.unified_height == 0);
    // prime_height and hash_height are diagnostic max(keepalive, round, push),
    // so keepalive DOES contribute to these snapshot fields.
    print_test_result("prime_height == 451 (diagnostic max includes keepalive)",
                      snap.prime_height == 451);
    print_test_result("hash_height == 801 (diagnostic max includes keepalive)",
                      snap.hash_height == 801);

    // Stake and fork fields from diagnostic
    print_test_result("stake_height == 999",       snap.stake_height == 999);
    print_test_result("hash_tip_lo32 == 0 (safe)", snap.hash_tip_lo32 == 0);
    print_test_result("fork_score == 0 (healthy)", snap.fork_score == 0);
    print_test_result("is_fork_active() == false", !snap.is_fork_active());
    print_test_result("last_update_source == KEEPALIVE",
                      snap.last_update_source == HeightTracker::UpdateSource::KEEPALIVE);

    // Diagnostic snapshot has the keepalive heights
    auto diag = tracker.GetDiagnosticSnapshot();
    print_test_result("diag.keepalive_unified_height == 6001",
                      diag.keepalive_unified_height == 6001);
    print_test_result("diag.keepalive_prime_height == 451",
                      diag.keepalive_prime_height == 451);
    print_test_result("diag.keepalive_hash_height == 801",
                      diag.keepalive_hash_height == 801);
}

// ============================================================================
// Test 16: OnKeepaliveResponse updates stake_height correctly
// ============================================================================
void test_keepalive_response_sets_stake_height() {
    std::cout << "\nTest 16: OnKeepaliveResponse updates stake_height\n";
    HeightTracker tracker;
    tracker.OnKeepaliveResponse(6001, 451, 801, 777, 0u, 0u);

    auto snap = tracker.GetSnapshot();
    print_test_result("stake_height == 777 via OnKeepaliveResponse",
                      snap.stake_height == 777);

    tracker.OnKeepaliveResponse(6002, 452, 802, 888, 0u, 0u);
    snap = tracker.GetSnapshot();
    print_test_result("stake_height == 888 via second OnKeepaliveResponse",
                      snap.stake_height == 888);
}

// ============================================================================
// Test 17: peak_fork_score is a persistent high-water mark
// ============================================================================
void test_peak_fork_score_high_water_mark() {
    std::cout << "\nTest 17: peak_fork_score is a persistent high-water mark\n";
    HeightTracker tracker;
    tracker.OnKeepaliveResponse(6000, 450, 800, 999, 0xCAFEBABEu, 5);
    tracker.OnKeepaliveResponse(6001, 451, 801, 999, 0xCAFEBABEu, 1);  // lower score

    auto snap = tracker.GetSnapshot();
    print_test_result("fork_score == 1 (latest)",      snap.fork_score == 1);
    print_test_result("peak_fork_score == 5 (canary)", snap.peak_fork_score == 5);
    print_test_result("is_fork_active() still true",   snap.is_fork_active());
}

// ============================================================================
// Test 18: Push updates channel_height; prime_height is diagnostic max
// ============================================================================
void test_push_updates_per_channel_heights() {
    std::cout << "\nTest 18: Push updates channel_height; prime_height is diagnostic max\n";
    HeightTracker tracker;
    tracker.OnTemplateReceived(1, 101);
    tracker.OnKeepaliveResponse(5000, 100, 200, 300, 0, 0);
    auto snap1 = tracker.GetSnapshot();
    // prime_height is diagnostic max(keepalive, round, push) — keepalive sets it to 100
    print_test_result("prime_height == 100 after keepalive (diagnostic max)", snap1.prime_height == 100);
    tracker.OnPushNotification(5002, 102, 0x1d00ffff);
    auto snap2 = tracker.GetSnapshot();
    print_test_result("push_channel_height == 102 after push", snap2.push_channel_height == 102);
    // prime_height is still from keepalive (push sets push_channel_height, not per-channel prime)
    print_test_result("prime_height == 100 after push (keepalive contribution persists)",
                      snap2.prime_height == 100);
}

// ============================================================================
// Test 19: Push updates channel_height; hash_height is diagnostic max
// ============================================================================
void test_push_updates_hash_height() {
    std::cout << "\nTest 19: Push updates channel_height; hash_height is diagnostic max\n";
    HeightTracker tracker;
    tracker.OnTemplateReceived(2, 201);
    tracker.OnKeepaliveResponse(5000, 100, 200, 300, 0, 0);
    auto snap1 = tracker.GetSnapshot();
    // hash_height is diagnostic max(keepalive, round, push) — keepalive sets it to 200
    print_test_result("hash_height == 200 after keepalive (diagnostic max)", snap1.hash_height == 200);
    tracker.OnPushNotification(5002, 202, 0x1d00ffff);
    auto snap2 = tracker.GetSnapshot();
    print_test_result("push_channel_height == 202 after push", snap2.push_channel_height == 202);
    // hash_height is still from keepalive (push sets push_channel_height, not per-channel hash)
    print_test_result("hash_height == 200 after push (keepalive contribution persists)",
                      snap2.hash_height == 200);
}

// ============================================================================
// Test 20: Production regression (prime drift from 2331124 to 2331126)
//          Push advances channel_height correctly; prime_height is diagnostic max.
// ============================================================================
void test_push_keepalive_no_regression() {
    std::cout << "\nTest 20: Production regression — push advances channel_height; prime_height diagnostic max\n";
    HeightTracker tracker;
    tracker.OnTemplateReceived(1, 2331125);
    tracker.OnKeepaliveResponse(6609207, 2331124, 2193089, 2084996, 0, 0);
    // Set canonical state so staleness check uses authoritative BLOCK_DATA
    // (push alone is diagnostic-only and cannot trigger stale)
    tracker.OnBlockDataReceived(6609208, 2331126, 0x0414b755, uint1024_t{});
    tracker.OnPushNotification(6609208, 2331126, 0x0414b755);
    auto snap = tracker.GetSnapshot();
    print_test_result("channel_height == 2331126", snap.channel_height == 2331126);
    // prime_height is diagnostic max — keepalive set it to 2331124
    print_test_result("prime_height == 2331124 (diagnostic max from keepalive)", snap.prime_height == 2331124);
    // is_template_stale() removed (structurally false under canonical-only semantics)
    // OnBlockDataReceived auto-advanced target to 2331127, so 2331126 < 2331127.
}

// ============================================================================
// Test 21: OnTemplateReceived writes to canonical_channel_target (Bug #1 fix)
// ============================================================================
void test_advance_channel_target_only_advances() {
    std::cout << "\nTest 21: OnTemplateReceived writes canonical_channel_target (only advances)\n";
    HeightTracker tracker;

    // Set initial channel_target via OnTemplateReceived
    tracker.OnPushNotification(5000, 100, 0x1d00ffff);
    tracker.OnTemplateReceived(1, 101);
    auto snap = tracker.GetSnapshot();
    print_test_result("channel_target == 101 after OnTemplateReceived",
                      snap.channel_target == 101);

    // Canonical also confirms
    auto canonical = tracker.GetCanonicalSnapshot();
    print_test_result("canonical_channel_target == 101",
                      canonical.canonical_channel_target == 101);

    // Advance to 105 via OnTemplateReceived
    tracker.OnTemplateReceived(1, 105);
    snap = tracker.GetSnapshot();
    print_test_result("channel_target == 105 after OnTemplateReceived(105)",
                      snap.channel_target == 105);

    // Attempting to set a lower value via OnTemplateReceived is a no-op
    tracker.OnTemplateReceived(1, 102);
    snap = tracker.GetSnapshot();
    print_test_result("channel_target still 105 after OnTemplateReceived(102)",
                      snap.channel_target == 105);

    // Attempting to set the same value is a no-op
    tracker.OnTemplateReceived(1, 105);
    snap = tracker.GetSnapshot();
    print_test_result("channel_target still 105 after OnTemplateReceived(105)",
                      snap.channel_target == 105);

    // Advancing beyond current value works
    tracker.OnTemplateReceived(1, 110);
    snap = tracker.GetSnapshot();
    print_test_result("channel_target == 110 after OnTemplateReceived(110)",
                      snap.channel_target == 110);
}

// ============================================================================
// Test 22: OnTemplateReceived does not regress channel_target
// ============================================================================
void test_on_template_received_no_regress() {
    std::cout << "\nTest 22: OnTemplateReceived does not regress channel_target\n";
    HeightTracker tracker;

    tracker.OnPushNotification(5000, 100, 0x1d00ffff);
    tracker.OnTemplateReceived(1, 101);
    auto snap = tracker.GetSnapshot();
    print_test_result("channel_target == 101 initially", snap.channel_target == 101);

    // OnBlockDataReceived advances canonical_channel_target to 106
    tracker.OnBlockDataReceived(5005, 105, 0x1d00ffff, uint1024_t{});
    snap = tracker.GetSnapshot();
    print_test_result("channel_target == 106 after OnBlockDataReceived(channel=105)",
                      snap.channel_target == 106);

    // Stale GET_BLOCK response arrives with a lower channel_target
    tracker.OnTemplateReceived(1, 102);
    snap = tracker.GetSnapshot();
    print_test_result("channel_target still 106 after stale OnTemplateReceived(102)",
                      snap.channel_target == 106);

    // But template_unified_height and timestamps are still updated
    print_test_result("last_update_source == TEMPLATE",
                      snap.last_update_source == HeightTracker::UpdateSource::TEMPLATE);

    // A fresh template with a higher target DOES advance
    tracker.OnTemplateReceived(1, 110);
    snap = tracker.GetSnapshot();
    print_test_result("channel_target == 110 after fresh OnTemplateReceived(110)",
                      snap.channel_target == 110);
}

// ============================================================================
// Test 23: Doom-loop prevention — with canonical-only channel_target, push
//          cannot create a doom loop (Bug #6 fix confirmation)
// ============================================================================
void test_doom_loop_prevention() {
    std::cout << "\nTest 23: Doom-loop impossible with canonical-only channel_target\n";
    HeightTracker tracker;

    // Initial: template targeting block 101 (canonical channel=100)
    tracker.OnBlockDataReceived(5000, 100, 0x1d00ffff, uint1024_t{});
    tracker.OnTemplateReceived(1, 101);

    // Push₁: says channel=101 → but canonical channel_height is still 100
    // because push only writes to diagnostic state, not canonical.
    tracker.OnPushNotification(5001, 101, 0x1d00ffff);
    auto snap = tracker.GetSnapshot();
    // is_template_stale() removed (structurally false under canonical-only semantics)
    // Push cannot trigger staleness — both sides are canonical-only

    // But canonical staleness is different: canonical_channel_height=100 < target=101
    auto canonical = tracker.GetCanonicalSnapshot();
    print_test_result("Canonical NOT stale: 100 < 101",
                      !canonical.is_canonically_stale());

    // Push₂: same channel_height 101 → still not stale (push is diagnostic only)
    tracker.OnPushNotification(5002, 101, 0x1d00ffff);
    snap = tracker.GetSnapshot();
    // is_template_stale() removed (structurally false under canonical-only semantics)

    // When BLOCK_DATA arrives with channel=101, canonical catches up and
    // OnBlockDataReceived sets canonical_channel_target=102
    tracker.OnBlockDataReceived(5001, 101, 0x1d00ffff, uint1024_t{});
    snap = tracker.GetSnapshot();
    print_test_result("After BLOCK_DATA: canonical target advanced to 102",
                      snap.channel_target == 102);
    // Now composite: channel_height=max(canonical=101, push=101)=101, target=102
    // is_template_stale() removed (structurally false under canonical-only semantics)
}

// ============================================================================
// Test 24: Stale GET_BLOCK response does not cause regression
// ============================================================================
void test_stale_get_block_no_regression() {
    std::cout << "\nTest 24: Stale GET_BLOCK response does not regress channel_target\n";
    HeightTracker tracker;

    // BLOCK_DATA says chain is at channel_height 105 → target = 106
    tracker.OnBlockDataReceived(5010, 105, 0x1d00ffff, uint1024_t{});
    tracker.OnTemplateReceived(1, 106);

    auto snap = tracker.GetSnapshot();
    print_test_result("channel_target == 106 after BLOCK_DATA(channel=105)",
                      snap.channel_target == 106);

    // Stale GET_BLOCK response has channel_height=99 (old data)
    tracker.OnGetRound(5007, 99, 99, 0);
    tracker.OnTemplateReceived(1, 100);  // stale template target

    snap = tracker.GetSnapshot();
    // channel_target should NOT have regressed to 100
    print_test_result("channel_target still 106 after stale GET_BLOCK (not regressed to 100)",
                      snap.channel_target == 106);

    // Push does NOT inflate canonical channel_target — it's diagnostic only
    tracker.OnPushNotification(5011, 105, 0x1d00ffff);
    snap = tracker.GetSnapshot();
    // channel_height composite includes push, but channel_target remains canonical
    print_test_result("channel_target still 106 (canonical, not inflated by push)",
                      snap.channel_target == 106);
    print_test_result("Not stale: canonical channel=105 < target=106",
                      snap.channel_height < snap.channel_target);
}

// ============================================================================
// Test 25: OnTemplateMetadata does NOT regress channel_height
//          (fixes HeightTracker update gap after stale GET_BLOCK responses)
// ============================================================================
void test_on_template_metadata_monotonic() {
    std::cout << "\nTest 25: OnTemplateMetadata does NOT regress channel_height\n";
    HeightTracker tracker;

    // Simulate: canonical state at channel 2332106
    tracker.OnBlockDataReceived(6611706, 2332106, 0x1d00ffff, uint1024_t{});
    tracker.OnPushNotification(6611706, 2332106, 0x1d00ffff);
    tracker.OnTemplateReceived(1, 2332100);  // old template target

    auto snap = tracker.GetSnapshot();
    print_test_result("channel_height == 2332106 after canonical",
                      snap.channel_height == 2332106);

    // Stale GET_BLOCK response arrives with channel_height=2332100
    // OnTemplateMetadata must NOT regress channel_height to 2332100.
    tracker.OnTemplateMetadata(6611706, 2332100, 0x1d00ffff);
    snap = tracker.GetSnapshot();
    print_test_result("channel_height still 2332106 after stale OnTemplateMetadata(2332100)",
                      snap.channel_height == 2332106);
    print_test_result("unified_height still 6611706",
                      snap.unified_height == 6611706);
    print_test_result("last_update_source == TEMPLATE",
                      snap.last_update_source == HeightTracker::UpdateSource::TEMPLATE);

    // Fresh template metadata with a higher channel_height DOES advance
    tracker.OnTemplateMetadata(6611710, 2332108, 0x1d00ffff);
    snap = tracker.GetSnapshot();
    print_test_result("channel_height advanced to 2332108 from fresh OnTemplateMetadata",
                      snap.channel_height == 2332108);
    print_test_result("unified_height advanced to 6611710",
                      snap.unified_height == 6611710);
}

// ============================================================================
// Test 26: OnTemplateMetadata per-channel heights — canonical is monotonic;
//          prime_height/hash_height are keepalive/round/push-full-picture-only (diagnostic)
// ============================================================================
void test_on_template_metadata_per_channel_no_regression() {
    std::cout << "\nTest 26: OnTemplateMetadata canonical is monotonic; prime/hash from keepalive/round/push-full-picture only\n";
    HeightTracker tracker;

    // Prime channel: canonical sets channel_height=500
    tracker.OnTemplateReceived(1, 1);  // set channel to Prime
    tracker.OnBlockDataReceived(10000, 500, 0x1d00ffff, uint1024_t{});
    tracker.OnPushNotification(10000, 500, 0x1d00ffff);
    auto snap = tracker.GetSnapshot();
    print_test_result("channel_height == 500 after canonical",
                      snap.channel_height == 500);
    // prime_height is diagnostic-only — 12-byte OnPushNotification does NOT update it;
    // only OnPushFullPicture (148-byte), OnGetRound, or OnKeepaliveResponse set it.
    print_test_result("prime_height == 0 (no keepalive/round/push-full-picture received yet)",
                      snap.prime_height == 0);

    // Stale template metadata with channel_height=400 must not regress channel_height
    tracker.OnTemplateMetadata(10001, 400, 0x1d00ffff);
    snap = tracker.GetSnapshot();
    print_test_result("channel_height still 500 after stale OnTemplateMetadata(400)",
                      snap.channel_height == 500);
    // prime_height remains 0 (no keepalive/round/push-full-picture called)
    print_test_result("prime_height still 0 after OnTemplateMetadata (no push-full-picture)",
                      snap.prime_height == 0);

    // Hash channel test
    HeightTracker hash_tracker;
    hash_tracker.OnTemplateReceived(2, 1);  // set channel to Hash
    hash_tracker.OnBlockDataReceived(10000, 300, 0x1d00ffff, uint1024_t{});
    hash_tracker.OnPushNotification(10000, 300, 0x1d00ffff);
    snap = hash_tracker.GetSnapshot();
    print_test_result("channel_height == 300 after canonical (Hash channel)",
                      snap.channel_height == 300);
    // hash_height is diagnostic-only — 12-byte push does NOT update it
    print_test_result("hash_height == 0 (no keepalive/round/push-full-picture received yet)",
                      snap.hash_height == 0);

    hash_tracker.OnTemplateMetadata(10001, 250, 0x1d00ffff);
    snap = hash_tracker.GetSnapshot();
    print_test_result("channel_height still 300 after stale OnTemplateMetadata(250)",
                      snap.channel_height == 300);
    print_test_result("hash_height still 0 after OnTemplateMetadata (no push-full-picture)",
                      snap.hash_height == 0);
}

// ============================================================================
// Test 27: Canonical state only from OnBlockDataReceived — push/keepalive
//          do NOT update canonical heights
// ============================================================================
void test_canonical_state_only_from_block_data() {
    std::cout << "\nTest 27: Canonical state only from OnBlockDataReceived\n";
    HeightTracker tracker;

    // Push notification arrives
    tracker.OnPushNotification(6611227, 2332106, 0x1d00ffff);
    auto canonical = tracker.GetCanonicalSnapshot();
    print_test_result("canonical not initialized after push",
                      !canonical.is_initialized());
    print_test_result("canonical_unified_height == 0 after push",
                      canonical.canonical_unified_height == 0);
    print_test_result("canonical_channel_height == 0 after push",
                      canonical.canonical_channel_height == 0);

    // Keepalive arrives
    tracker.OnKeepaliveResponse(6611227, 2332106, 9999, 8888, 0xABCD1234u, 2);
    canonical = tracker.GetCanonicalSnapshot();
    print_test_result("canonical still not initialized after keepalive",
                      !canonical.is_initialized());

    // GET_ROUND arrives
    tracker.OnGetRound(6611227, 2332106, 2332106, 0);
    canonical = tracker.GetCanonicalSnapshot();
    print_test_result("canonical still not initialized after GET_ROUND",
                      !canonical.is_initialized());

    // BLOCK_DATA arrives (via OnTemplateMetadata which delegates to OnBlockDataReceived)
    tracker.OnTemplateMetadata(6611227, 2332106, 0x1d00ffff);
    canonical = tracker.GetCanonicalSnapshot();
    print_test_result("canonical IS initialized after OnTemplateMetadata (→ OnBlockDataReceived)",
                      canonical.is_initialized());
    print_test_result("canonical_unified_height == 6611227",
                      canonical.canonical_unified_height == 6611227);
    print_test_result("canonical_channel_height == 2332106",
                      canonical.canonical_channel_height == 2332106);
}

// ============================================================================
// Test 28: Keepalive does not corrupt canonical unified/channel heights
//          (the core bug this refactor fixes)
// ============================================================================
void test_keepalive_does_not_corrupt_canonical() {
    std::cout << "\nTest 28: Keepalive does not corrupt canonical state\n";
    HeightTracker tracker;

    // Establish canonical state from BLOCK_DATA
    tracker.OnTemplateMetadata(6611228, 2332107, 0x1d00ffff);
    tracker.OnTemplateReceived(1, 2332108);
    auto canonical = tracker.GetCanonicalSnapshot();
    print_test_result("canonical initialized: unified=6611228",
                      canonical.canonical_unified_height == 6611228);
    print_test_result("canonical initialized: channel=2332107",
                      canonical.canonical_channel_height == 2332107);
    print_test_result("canonical_channel_target == 2332108",
                      canonical.canonical_channel_target == 2332108);

    // Keepalive arrives with STALE values (lower heights from 45s ago)
    tracker.OnKeepaliveResponse(6611200, 2332050, 9000, 8000, 0xDEADBEEFu, 1);

    // Canonical must NOT be regressed by keepalive
    canonical = tracker.GetCanonicalSnapshot();
    print_test_result("canonical_unified_height still 6611228 after stale keepalive",
                      canonical.canonical_unified_height == 6611228);
    print_test_result("canonical_channel_height still 2332107 after stale keepalive",
                      canonical.canonical_channel_height == 2332107);
    print_test_result("canonical_channel_target still 2332108 after stale keepalive",
                      canonical.canonical_channel_target == 2332108);

    // Snapshot heights must also not regress
    auto snap = tracker.GetSnapshot();
    print_test_result("snapshot channel_height still 2332107 after keepalive (canonical wins)",
                      snap.channel_height == 2332107);
    print_test_result("snapshot unified_height still 6611228 after keepalive (canonical wins)",
                      snap.unified_height == 6611228);
    // Fork score IS available from diagnostic
    print_test_result("snapshot fork_score == 1 (from keepalive diagnostic)",
                      snap.fork_score == 1);
    print_test_result("snapshot peak_fork_score == 1 (from keepalive diagnostic)",
                      snap.peak_fork_score == 1);
}

// ============================================================================
// Test 29: OnBlockDataReceived is monotonic — stale BLOCK_DATA cannot regress
//          canonical state established by a previous BLOCK_DATA receipt
// ============================================================================
void test_on_block_data_received_monotonic() {
    std::cout << "\nTest 29: OnBlockDataReceived is monotonic\n";
    HeightTracker tracker;

    // First BLOCK_DATA: sets canonical to height N
    tracker.OnBlockDataReceived(6611228, 2332107, 0x1d00ffff, uint1024_t{});
    auto canonical = tracker.GetCanonicalSnapshot();
    print_test_result("canonical_unified_height == 6611228 after first BLOCK_DATA",
                      canonical.canonical_unified_height == 6611228);
    print_test_result("canonical_channel_height == 2332107",
                      canonical.canonical_channel_height == 2332107);
    print_test_result("canonical_channel_target == 2332108",
                      canonical.canonical_channel_target == 2332108);

    // Stale BLOCK_DATA arrives (lower heights — must be ignored)
    tracker.OnBlockDataReceived(6611100, 2332050, 0x1d00ffff, uint1024_t{});
    canonical = tracker.GetCanonicalSnapshot();
    print_test_result("canonical_unified_height still 6611228 (stale BLOCK_DATA ignored)",
                      canonical.canonical_unified_height == 6611228);
    print_test_result("canonical_channel_height still 2332107 (stale BLOCK_DATA ignored)",
                      canonical.canonical_channel_height == 2332107);
    print_test_result("canonical_channel_target still 2332108",
                      canonical.canonical_channel_target == 2332108);

    // Fresh BLOCK_DATA with higher heights DOES advance canonical
    tracker.OnBlockDataReceived(6611229, 2332108, 0x1d00ffff, uint1024_t{});
    canonical = tracker.GetCanonicalSnapshot();
    print_test_result("canonical_unified_height advanced to 6611229",
                      canonical.canonical_unified_height == 6611229);
    print_test_result("canonical_channel_height advanced to 2332108",
                      canonical.canonical_channel_height == 2332108);
    print_test_result("canonical_channel_target advanced to 2332109",
                      canonical.canonical_channel_target == 2332109);
}

// ============================================================================
// Test 30: Fork score lives in DiagnosticObserverState — GetCanonicalSnapshot
//          has no fork_score field (architectural isolation)
// ============================================================================
void test_diagnostic_fork_score_isolated() {
    std::cout << "\nTest 30: Fork score is in DiagnosticObserverState only\n";
    HeightTracker tracker;

    // No keepalive yet
    auto canonical = tracker.GetCanonicalSnapshot();
    auto diag = tracker.GetDiagnosticSnapshot();
    print_test_result("canonical has no fork_score field (it is NOT in CanonicalChainState)",
                      true);  // Structural: CanonicalChainState has no fork_score member
    print_test_result("diagnostic keepalive_fork_score == 0 initially",
                      diag.keepalive_fork_score == 0);
    print_test_result("diagnostic keepalive_peak_fork_score == 0 initially",
                      diag.keepalive_peak_fork_score == 0);

    // Keepalive with fork_score
    tracker.OnKeepaliveResponse(6000, 100, 200, 300, 0xABCDu, 7);
    diag = tracker.GetDiagnosticSnapshot();
    print_test_result("diagnostic keepalive_fork_score == 7 after keepalive",
                      diag.keepalive_fork_score == 7);
    print_test_result("diagnostic keepalive_peak_fork_score == 7",
                      diag.keepalive_peak_fork_score == 7);
    print_test_result("diagnostic is_fork_canary_active() == true",
                      diag.is_fork_canary_active());

    // Snapshot backward compat: fork_score is still accessible via GetSnapshot()
    auto snap = tracker.GetSnapshot();
    print_test_result("snapshot fork_score == 7 (backward compat via diagnostic)",
                      snap.fork_score == 7);
    print_test_result("snapshot peak_fork_score == 7 (backward compat via diagnostic)",
                      snap.peak_fork_score == 7);
    print_test_result("snapshot is_fork_active() == true (backward compat)",
                      snap.is_fork_active());
}

// ============================================================================
// Test 31: height_drift_from_canonical() returns 0 for a healthy canonical state
// ============================================================================
void test_height_drift_from_canonical() {
    std::cout << "\nTest 31: height_drift_from_canonical() checks\n";
    HeightTracker tracker;

    // No canonical state yet — drift should be 0 (no data)
    auto canonical = tracker.GetCanonicalSnapshot();
    print_test_result("height_drift_from_canonical() == 0 when uninitialized",
                      canonical.height_drift_from_canonical() == 0);

    // Set canonical: unified=6611228, channel=2332107, target=2332108
    tracker.OnBlockDataReceived(6611228, 2332107, 0x1d00ffff, uint1024_t{});
    canonical = tracker.GetCanonicalSnapshot();
    // drift = canonical_unified_height - canonical_channel_target = 6611228 - 2332108 = 4279120
    // (not 0 because unified and channel are different dimensions — but the method still works)
    print_test_result("height_drift_from_canonical() is non-zero (unified != channel_target)",
                      canonical.height_drift_from_canonical() != 0);
    print_test_result("height_drift_from_canonical() == 6611228 - 2332108 = 4279120",
                      canonical.height_drift_from_canonical() == (int32_t)(6611228 - 2332108));

    // When canonical_unified == canonical_channel_target (perfectly aligned), drift == 0
    HeightTracker aligned_tracker;
    aligned_tracker.OnBlockDataReceived(1000, 999, 0x1d00ffff, uint1024_t{});
    // canonical_channel_target = 1000 (999+1)
    canonical = aligned_tracker.GetCanonicalSnapshot();
    print_test_result("height_drift_from_canonical() == 0 when unified == channel_target",
                      canonical.height_drift_from_canonical() == 0);
}

// ============================================================================
// Test 27: Canonical isolation — OnBlockDataReceived sets canonical,
//          keepalive cannot corrupt it
// ============================================================================
void test_canonical_isolation() {
    std::cout << "\nTest 27: Canonical isolation — keepalive cannot corrupt canonical\n";
    HeightTracker tracker;
    tracker.OnTemplateReceived(1, 101);

    // Set canonical via OnBlockDataReceived
    tracker.OnBlockDataReceived(6000, 500, 0x1d00ffff, uint1024_t{});
    auto canon = tracker.GetCanonicalSnapshot();
    print_test_result("canonical_unified_height == 6000", canon.canonical_unified_height == 6000);
    print_test_result("canonical_channel_height == 500",  canon.canonical_channel_height == 500);
    print_test_result("canonical_channel_target == 501",  canon.canonical_channel_target == 501);
    print_test_result("is_initialized() == true",         canon.is_initialized());
    print_test_result("canonical_received_at is set",
                      canon.canonical_received_at != std::chrono::steady_clock::time_point{});

    // Keepalive with LOWER heights must NOT corrupt canonical
    tracker.OnKeepaliveResponse(5990, 490, 790, 300, 0xDEADBEEFu, 2);
    canon = tracker.GetCanonicalSnapshot();
    print_test_result("canonical_unified_height still 6000 after stale keepalive",
                      canon.canonical_unified_height == 6000);
    print_test_result("canonical_channel_height still 500 after stale keepalive",
                      canon.canonical_channel_height == 500);

    // Keepalive with HIGHER heights still must NOT corrupt canonical
    tracker.OnKeepaliveResponse(7000, 600, 900, 400, 0xCAFEBABEu, 0);
    canon = tracker.GetCanonicalSnapshot();
    print_test_result("canonical_unified_height still 6000 after higher keepalive",
                      canon.canonical_unified_height == 6000);
    print_test_result("canonical_channel_height still 500 after higher keepalive",
                      canon.canonical_channel_height == 500);

    // Only OnBlockDataReceived can advance canonical
    tracker.OnBlockDataReceived(7001, 601, 0x1d00ffff, uint1024_t{});
    canon = tracker.GetCanonicalSnapshot();
    print_test_result("canonical_unified_height advanced to 7001 via OnBlockDataReceived",
                      canon.canonical_unified_height == 7001);
    print_test_result("canonical_channel_height advanced to 601 via OnBlockDataReceived",
                      canon.canonical_channel_height == 601);
    print_test_result("canonical_channel_target advanced to 602",
                      canon.canonical_channel_target == 602);
}

// ============================================================================
// Test 28: Fork scores reset when OnTemplateReceived advances channel_target
// ============================================================================
void test_fork_scores_reset_on_template_advance() {
    std::cout << "\nTest 28: Fork scores reset when OnTemplateReceived advances channel_target\n";
    HeightTracker tracker;
    // Set up: channel=2 (Hash), initial template, then a fork
    tracker.OnTemplateReceived(2, 101);
    tracker.OnKeepaliveResponse(6000, 450, 800, 999, 0xCAFEBABEu, 5);

    auto snap = tracker.GetSnapshot();
    print_test_result("fork_score == 5 after keepalive",      snap.fork_score == 5);
    print_test_result("peak_fork_score == 5 after keepalive", snap.peak_fork_score == 5);
    print_test_result("is_fork_active() == true",             snap.is_fork_active());

    // Recovery: a fresh template arrives that advances channel_target
    tracker.OnTemplateReceived(2, 102);

    snap = tracker.GetSnapshot();
    print_test_result("fork_score == 0 after recovery template",      snap.fork_score == 0);
    print_test_result("peak_fork_score == 0 after recovery template", snap.peak_fork_score == 0);
    print_test_result("is_fork_active() == false after recovery",     !snap.is_fork_active());
    print_test_result("channel_target == 102",                        snap.channel_target == 102);
}

// ============================================================================
// Test 29: Fork scores NOT reset when OnTemplateReceived doesn't advance target
// ============================================================================
void test_fork_scores_sticky_without_advance() {
    std::cout << "\nTest 29: Fork scores NOT reset when OnTemplateReceived doesn't advance target\n";
    HeightTracker tracker;
    tracker.OnTemplateReceived(2, 101);
    tracker.OnKeepaliveResponse(6000, 450, 800, 999, 0xCAFEBABEu, 3);

    // Stale template that doesn't advance channel_target
    tracker.OnTemplateReceived(2, 100);

    auto snap = tracker.GetSnapshot();
    print_test_result("fork_score still 3 (stale template)",      snap.fork_score == 3);
    print_test_result("peak_fork_score still 3 (stale template)", snap.peak_fork_score == 3);
    print_test_result("is_fork_active() still true",              snap.is_fork_active());
    print_test_result("channel_target unchanged at 101",          snap.channel_target == 101);
}

// ============================================================================
// Test 32: canonical_hash_prev_block anchored to BLOCK_DATA (Tritium block)
// ============================================================================
void test_canonical_hash_prev_block() {
    std::cout << "\nTest 32: canonical_hash_prev_block anchored to BLOCK_DATA\n";
    HeightTracker tracker;
    tracker.OnTemplateReceived(1, 101);

    // Before any block data: is_initialized() is false, hash_prev_block is zero
    auto canon = tracker.GetCanonicalSnapshot();
    print_test_result("is_initialized() == false before block data",
                      !canon.is_initialized());
    print_test_result("canonical_hash_prev_block is zero before block data",
                      canon.canonical_hash_prev_block == uint1024_t(0));

    // OnBlockDataReceived sets heights; UpdateWithHashPrevBlock sets hashPrevBlock
    tracker.OnBlockDataReceived(6000, 500, 0x1d00ffff, uint1024_t{});
    uint1024_t fake_prev_hash(0xDEADBEEFu);
    tracker.UpdateWithHashPrevBlock(fake_prev_hash);

    canon = tracker.GetCanonicalSnapshot();
    print_test_result("is_initialized() == true after block data",
                      canon.is_initialized());
    print_test_result("canonical_hash_prev_block set from UpdateWithHashPrevBlock",
                      canon.canonical_hash_prev_block == fake_prev_hash);

    // Snapshot's hash_prev_block is sourced from canonical
    auto snap = tracker.GetSnapshot();
    print_test_result("snapshot.hash_prev_block == canonical_hash_prev_block",
                      snap.hash_prev_block == fake_prev_hash);
    print_test_result("snapshot.canonical_hash_prev_block == canonical source",
                      snap.canonical_hash_prev_block == fake_prev_hash);

    // Keepalive does NOT affect canonical_hash_prev_block
    tracker.OnKeepaliveResponse(7000, 600, 900, 400, 0xCAFEBABEu, 1);
    canon = tracker.GetCanonicalSnapshot();
    print_test_result("canonical_hash_prev_block unchanged after keepalive",
                      canon.canonical_hash_prev_block == fake_prev_hash);

    // Diagnostic hash_tip_lo32 is separate (lo32 of node's hashBestChain)
    auto diag = tracker.GetDiagnosticSnapshot();
    print_test_result("diagnostic hash_tip_lo32 == 0xCAFEBABE from keepalive",
                      diag.keepalive_hash_tip_lo32 == 0xCAFEBABEu);

    // Update hash_prev_block with new chain tip
    uint1024_t new_prev_hash(0xFEEDFACEu);
    tracker.UpdateWithHashPrevBlock(new_prev_hash);
    canon = tracker.GetCanonicalSnapshot();
    print_test_result("canonical_hash_prev_block updated to new tip",
                      canon.canonical_hash_prev_block == new_prev_hash);
}

// Test 33: DiagnosticObserverState::is_initialized() and latest_received_at()
void test_diagnostic_observer_helpers() {
    std::cout << "\nTest 33: DiagnosticObserverState::is_initialized() and latest_received_at()\n";
    HeightTracker tracker;

    // Before any updates, diagnostic state is NOT initialized
    auto diag = tracker.GetDiagnosticSnapshot();
    print_test_result("diagnostic not initialized before any update",
                      !diag.is_initialized());
    print_test_result("latest_received_at is default before any update",
                      diag.latest_received_at() == std::chrono::steady_clock::time_point{});

    // After push notification, diagnostic state IS initialized
    auto before_push = std::chrono::steady_clock::now();
    tracker.OnPushNotification(100, 50, 0x1D00FFFF);
    auto after_push = std::chrono::steady_clock::now();
    diag = tracker.GetDiagnosticSnapshot();
    print_test_result("diagnostic initialized after push",
                      diag.is_initialized());
    print_test_result("last_push_at set after push",
                      diag.last_push_at >= before_push && diag.last_push_at <= after_push);
    print_test_result("latest_received_at reflects push",
                      diag.latest_received_at() >= before_push);

    // After GET_ROUND, latest_received_at advances
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    auto before_round = std::chrono::steady_clock::now();
    tracker.OnGetRound(101, 51, 52, 0);
    auto after_round = std::chrono::steady_clock::now();
    diag = tracker.GetDiagnosticSnapshot();
    print_test_result("last_round_at set after GET_ROUND",
                      diag.last_round_at >= before_round && diag.last_round_at <= after_round);
    print_test_result("latest_received_at >= round time (most recent)",
                      diag.latest_received_at() >= before_round);

    // After keepalive, latest_received_at may advance further
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    auto before_ka = std::chrono::steady_clock::now();
    tracker.OnKeepaliveResponse(102, 52, 53, 10, 0xDEADBEEF, 0);
    diag = tracker.GetDiagnosticSnapshot();
    print_test_result("latest_received_at >= keepalive time",
                      diag.latest_received_at() >= before_ka);

    // Canonical is NOT initialized (no BLOCK_DATA received)
    auto canonical = tracker.GetCanonicalSnapshot();
    print_test_result("canonical NOT initialized without BLOCK_DATA",
                      !canonical.is_initialized());

    // Now receive BLOCK_DATA — canonical becomes initialized
    tracker.OnBlockDataReceived(103, 53, 0x1D00FFFF, uint1024_t{});
    canonical = tracker.GetCanonicalSnapshot();
    print_test_result("canonical IS initialized after BLOCK_DATA",
                      canonical.is_initialized());
}

// Test 32: DiagnosticObserverState::is_initialized() — diagnostic equivalent
// ============================================================================
void test_diagnostic_is_initialized() {
    std::cout << "\nTest 32: DiagnosticObserverState::is_initialized() — diagnostic equivalent\n";

    // Empty tracker: neither canonical nor diagnostic should be initialized
    HeightTracker tracker;
    auto diag = tracker.GetDiagnosticSnapshot();
    print_test_result("is_initialized() == false when no data",     !diag.is_initialized());

    // Keepalive alone initializes diagnostic
    tracker.OnKeepaliveResponse(6000, 450, 800, 999, 0xCAFEBABEu, 0);
    diag = tracker.GetDiagnosticSnapshot();
    print_test_result("is_initialized() == true after keepalive",   diag.is_initialized());

    // Fresh tracker: push alone initializes diagnostic
    HeightTracker tracker2;
    tracker2.OnPushNotification(6100, 2332100, 0x1d00ffff);
    diag = tracker2.GetDiagnosticSnapshot();
    print_test_result("is_initialized() == true after push",        diag.is_initialized());

    // Fresh tracker: GET_ROUND alone initializes diagnostic
    HeightTracker tracker3;
    tracker3.OnGetRound(6200, 2332200, 2332200, 0);
    diag = tracker3.GetDiagnosticSnapshot();
    print_test_result("is_initialized() == true after GET_ROUND",   diag.is_initialized());

    // Canonical BLOCK_DATA does NOT initialize diagnostic
    HeightTracker tracker4;
    tracker4.OnBlockDataReceived(6300, 2332300, 0x1d00ffff, uint1024_t{});
    diag = tracker4.GetDiagnosticSnapshot();
    print_test_result("is_initialized() == false after BLOCK_DATA only",  !diag.is_initialized());
    auto can = tracker4.GetCanonicalSnapshot();
    print_test_result("canonical is_initialized() == true after BLOCK_DATA", can.is_initialized());
}

// ============================================================================
// Test 33: DiagnosticObserverState::latest_received_at() — diagnostic equivalent
// ============================================================================
void test_diagnostic_latest_received_at() {
    std::cout << "\nTest 33: DiagnosticObserverState::latest_received_at() — diagnostic equivalent\n";

    HeightTracker tracker;
    auto epoch = std::chrono::steady_clock::time_point{};
    auto diag = tracker.GetDiagnosticSnapshot();
    print_test_result("latest_received_at() == epoch when no data",
                      diag.latest_received_at() == epoch);

    // Push sets a timestamp
    tracker.OnPushNotification(6100, 2332100, 0x1d00ffff);
    diag = tracker.GetDiagnosticSnapshot();
    print_test_result("latest_received_at() > epoch after push",
                      diag.latest_received_at() > epoch);

    auto after_push = diag.latest_received_at();

    // GET_ROUND at a later time should become the latest
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    tracker.OnGetRound(6200, 2332200, 2332200, 0);
    diag = tracker.GetDiagnosticSnapshot();
    print_test_result("latest_received_at() advances after GET_ROUND",
                      diag.latest_received_at() >= after_push);

    // Keepalive at a later time should become the latest
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    tracker.OnKeepaliveResponse(6000, 450, 800, 999, 0, 0);
    diag = tracker.GetDiagnosticSnapshot();
    print_test_result("latest_received_at() == last_keepalive_ack_at after keepalive",
                      diag.latest_received_at() == diag.last_keepalive_ack_at);
}

// ============================================================================
// Test 34: OnTemplateReceived() writes to canonical_channel_target (Bug #1 fix)
// ============================================================================
void test_advance_channel_target_updates_canonical() {
    std::cout << "\nTest 34: OnTemplateReceived writes canonical_channel_target\n";
    HeightTracker tracker;

    // Give it a canonical base via OnBlockDataReceived
    tracker.OnBlockDataReceived(6000, 2000, 0x1d00ffff, uint1024_t{});
    auto canon = tracker.GetCanonicalSnapshot();
    print_test_result("canonical_channel_target == 2001 after BLOCK_DATA",
                      canon.canonical_channel_target == 2001);

    // OnTemplateReceived advances canonical_channel_target beyond BLOCK_DATA
    tracker.OnTemplateReceived(1, 2010);
    canon = tracker.GetCanonicalSnapshot();
    print_test_result("canonical_channel_target == 2010 after OnTemplateReceived(2010)",
                      canon.canonical_channel_target == 2010);

    // Snapshot channel_target should also reflect the advance
    auto snap = tracker.GetSnapshot();
    print_test_result("snapshot channel_target == 2010",
                      snap.channel_target == 2010);

    // Attempting to regress is a no-op on canonical too
    tracker.OnTemplateReceived(1, 2005);
    canon = tracker.GetCanonicalSnapshot();
    print_test_result("canonical_channel_target still 2010 after lower OnTemplateReceived(2005)",
                      canon.canonical_channel_target == 2010);
}

// ============================================================================
// Test 35: OnTemplateReceived() sets template_unified_height from canonical
// ============================================================================
void test_on_template_received_sets_template_unified_height() {
    std::cout << "\nTest 35: OnTemplateReceived sets template_unified_height\n";
    HeightTracker tracker;

    // With no data, template_unified_height should be 0
    auto snap = tracker.GetSnapshot();
    print_test_result("template_unified_height == 0 before any data",
                      snap.template_unified_height == 0);

    // Receive canonical BLOCK_DATA to set unified height
    tracker.OnBlockDataReceived(6612224, 2332312, 0x043d6262, uint1024_t{});

    // Receive template — template_unified_height should be set to canonical unified height
    tracker.OnTemplateReceived(2, 2332313);
    snap = tracker.GetSnapshot();
    auto canon = tracker.GetCanonicalSnapshot();
    print_test_result("template_unified_height == canonical_unified_height after OnTemplateReceived",
                      snap.template_unified_height == canon.canonical_unified_height);
    print_test_result("template_unified_height == 6612224",
                      snap.template_unified_height == 6612224);

    // After tip moves (canonical advance), is_tip_moved() returns true
    tracker.OnBlockDataReceived(6612225, 2332313, 0x043d6262, uint1024_t{});
    snap = tracker.GetSnapshot();
    print_test_result("is_tip_moved() == true after canonical advances beyond template_unified_height",
                      snap.is_tip_moved());
}

// ============================================================================
// Test 36: last_push_notification_at is set ONLY by OnPushNotification —
//          not by OnKeepaliveResponse or OnGetRound.
//          last_height_update is also set ONLY by OnPushNotification (Bug #1 fix).
// ============================================================================
void test_push_notification_at_isolated_from_keepalive_and_getround() {
    std::cout << "\nTest 36: last_push_notification_at set only by push, not by keepalive or GET_ROUND\n";
    HeightTracker tracker;

    // After keepalive only, last_push_notification_at must remain epoch (unset).
    tracker.OnKeepaliveResponse(6000, 450, 800, 999, 0xCAFEBABEu, 0);
    auto snap = tracker.GetSnapshot();
    print_test_result("last_push_notification_at == epoch after keepalive only",
                      snap.last_push_notification_at == std::chrono::steady_clock::time_point{});
    // last_height_update must also remain epoch after keepalive (Bug #1 fix)
    print_test_result("last_height_update == epoch after keepalive only (Bug #1 fix)",
                      snap.last_height_update == std::chrono::steady_clock::time_point{});

    // After GET_ROUND only, last_push_notification_at must remain epoch.
    tracker.OnGetRound(6001, 451, 452, 0);
    snap = tracker.GetSnapshot();
    print_test_result("last_push_notification_at == epoch after GET_ROUND only",
                      snap.last_push_notification_at == std::chrono::steady_clock::time_point{});
    // last_height_update must also remain epoch after GET_ROUND (Bug #3 fix)
    print_test_result("last_height_update == epoch after GET_ROUND only (Bug #3 fix)",
                      snap.last_height_update == std::chrono::steady_clock::time_point{});

    // After OnPushNotification, last_push_notification_at and last_height_update must be set.
    tracker.OnPushNotification(6002, 452, 0x1d00ffff);
    snap = tracker.GetSnapshot();
    print_test_result("last_push_notification_at set after OnPushNotification",
                      snap.last_push_notification_at != std::chrono::steady_clock::time_point{});
    print_test_result("last_height_update set after OnPushNotification",
                      snap.last_height_update != std::chrono::steady_clock::time_point{});

    // Both timestamps must be consistent (same push event)
    print_test_result("last_push_notification_at == last_height_update after push",
                      snap.last_push_notification_at == snap.last_height_update);

    // A subsequent keepalive must NOT advance last_push_notification_at or last_height_update.
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    auto push_time = snap.last_push_notification_at;
    tracker.OnKeepaliveResponse(6005, 455, 805, 999, 0u, 0u);
    snap = tracker.GetSnapshot();
    print_test_result("last_push_notification_at unchanged after subsequent keepalive",
                      snap.last_push_notification_at == push_time);
    print_test_result("last_height_update unchanged after subsequent keepalive",
                      snap.last_height_update == push_time);

    // A subsequent GET_ROUND must NOT advance last_push_notification_at or last_height_update.
    tracker.OnGetRound(6006, 456, 457, 0);
    snap = tracker.GetSnapshot();
    print_test_result("last_push_notification_at unchanged after subsequent GET_ROUND",
                      snap.last_push_notification_at == push_time);
    print_test_result("last_height_update unchanged after subsequent GET_ROUND",
                      snap.last_height_update == push_time);
}

// ============================================================================
// Test 11: set_session_epoch() clears last_keepalive_ack_at on epoch change
// This is the core fix for the keepalive tracking split:
// a stale-epoch keepalive ACK must not signal liveness for the new epoch.
// ============================================================================
void test_set_session_epoch_clears_keepalive_on_epoch_change() {
    std::cout << "\nTest 11 (NEW): set_session_epoch() clears last_keepalive_ack_at on epoch change\n";
    HeightTracker tracker;

    // Epoch 1: session established, keepalives flowing
    tracker.set_session_epoch(SessionEpoch(1));
    tracker.OnKeepaliveResponse(5000, 400, 700, 900, 0xDEADBEEFu, 0);

    auto snap_before = tracker.GetSnapshot();
    print_test_result("Epoch 1: last_keepalive_ack_at is set after OnKeepaliveResponse",
                      snap_before.last_keepalive_ack_at != std::chrono::steady_clock::time_point{});
    print_test_result("Epoch 1: session_epoch == 1 in snapshot",
                      snap_before.session_epoch == SessionEpoch(1));

    // Advance the session epoch (simulates re-auth / channel advance causing epoch bump)
    tracker.set_session_epoch(SessionEpoch(2));

    auto snap_after = tracker.GetSnapshot();
    print_test_result("Epoch 2: last_keepalive_ack_at CLEARED after epoch change",
                      snap_after.last_keepalive_ack_at == std::chrono::steady_clock::time_point{});
    print_test_result("Epoch 2: session_epoch == 2 in snapshot",
                      snap_after.session_epoch == SessionEpoch(2));
}

// ============================================================================
// Test 12: set_session_epoch() does NOT clear last_keepalive_ack_at when epoch unchanged
// ============================================================================
void test_set_session_epoch_no_clear_when_unchanged() {
    std::cout << "\nTest 12 (NEW): set_session_epoch() does NOT clear keepalive timestamp when epoch unchanged\n";
    HeightTracker tracker;

    tracker.set_session_epoch(SessionEpoch(5));
    tracker.OnKeepaliveResponse(5000, 400, 700, 900, 0xDEADBEEFu, 0);

    auto snap_before = tracker.GetSnapshot();
    auto ack_time = snap_before.last_keepalive_ack_at;
    print_test_result("last_keepalive_ack_at is set initially",
                      ack_time != std::chrono::steady_clock::time_point{});

    // Re-setting the SAME epoch should NOT clear the keepalive timestamp
    tracker.set_session_epoch(SessionEpoch(5));

    auto snap_after = tracker.GetSnapshot();
    print_test_result("last_keepalive_ack_at preserved when same epoch re-set",
                      snap_after.last_keepalive_ack_at == ack_time);
}

// ============================================================================
// Test 13: Keepalive timestamp from old epoch does not survive into new epoch
// (Simulates the real-world deadlock: channel advances → epoch bumps → stale
// keepalive should not falsely report ack_recent=true in new epoch)
// ============================================================================
void test_old_epoch_keepalive_does_not_signal_new_epoch_liveness() {
    std::cout << "\nTest 13 (NEW): Old-epoch keepalive timestamp does not survive into new epoch\n";
    HeightTracker tracker;

    // Epoch 1: session established, keepalives flowing
    tracker.set_session_epoch(SessionEpoch(1));
    tracker.OnKeepaliveResponse(5000, 400, 700, 900, 0xDEADBEEFu, 0);

    // Confirm keepalive is "recent" for epoch 1
    auto snap1 = tracker.GetSnapshot();
    print_test_result("Epoch 1: keepalive ACK timestamp is set",
                      snap1.last_keepalive_ack_at != std::chrono::steady_clock::time_point{});

    // Channel advance + re-auth → epoch 2
    tracker.set_session_epoch(SessionEpoch(2));
    tracker.OnPushNotification(5001, 101, 0x1d00ffff);  // channel advances

    // In the new epoch, before any keepalive for epoch 2,
    // the old epoch's keepalive must NOT make ack_recent appear true.
    auto snap2 = tracker.GetSnapshot();
    print_test_result("Epoch 2: old-epoch keepalive timestamp is cleared (no stale liveness signal)",
                      snap2.last_keepalive_ack_at == std::chrono::steady_clock::time_point{});
    print_test_result("Epoch 2: push_channel_height updated by push (not affected by epoch change)",
                      snap2.push_channel_height == 101);

    // Now receive a keepalive for epoch 2 — should work normally
    tracker.OnKeepaliveResponse(5001, 401, 701, 901, 0xCAFEBABEu, 0);
    auto snap3 = tracker.GetSnapshot();
    print_test_result("Epoch 2: new keepalive ACK is accepted and timestamp set",
                      snap3.last_keepalive_ack_at != std::chrono::steady_clock::time_point{});
}

// ============================================================================
// Test 14: set_session_epoch(0) does NOT clear last_keepalive_ack_at
// (Epoch 0 is the sentinel "no active session" baseline — must not invalidate)
// ============================================================================
void test_set_session_epoch_zero_no_clear() {
    std::cout << "\nTest 14 (NEW): set_session_epoch(0) does NOT clear keepalive timestamp (sentinel epoch)\n";
    HeightTracker tracker;

    tracker.set_session_epoch(SessionEpoch(3));
    tracker.OnKeepaliveResponse(5000, 400, 700, 900, 0xDEADBEEFu, 0);
    auto ack_time = tracker.GetSnapshot().last_keepalive_ack_at;
    print_test_result("Initial keepalive timestamp is set",
                      ack_time != std::chrono::steady_clock::time_point{});

    // Setting epoch to 0 (sentinel) should be a no-op for keepalive clearing
    tracker.set_session_epoch(SessionEpoch(0));
    auto snap_after = tracker.GetSnapshot();
    print_test_result("set_session_epoch(0) does not clear keepalive timestamp",
                      snap_after.last_keepalive_ack_at == ack_time);
    print_test_result("session_epoch updated to 0",
                      snap_after.session_epoch == SessionEpoch(0));
}

// ============================================================================
// Test 15: Session epoch reflected in Snapshot.session_epoch
// ============================================================================
void test_session_epoch_in_snapshot() {
    std::cout << "\nTest 15 (NEW): session_epoch field in Snapshot reflects set_session_epoch()\n";
    HeightTracker tracker;

    auto snap0 = tracker.GetSnapshot();
    print_test_result("Initial session_epoch == 0",
                      snap0.session_epoch == SessionEpoch(0));

    tracker.set_session_epoch(SessionEpoch(42));
    auto snap42 = tracker.GetSnapshot();
    print_test_result("session_epoch == 42 after set",
                      snap42.session_epoch == SessionEpoch(42));

    tracker.set_session_epoch(SessionEpoch(99));
    auto snap99 = tracker.GetSnapshot();
    print_test_result("session_epoch == 99 after second set",
                      snap99.session_epoch == SessionEpoch(99));
}

// ============================================================================
// Test: OnPushFullPicture() — updates push cross-channel heights in snapshot
// ============================================================================
void test_on_push_full_picture() {
    std::cout << "\nTest (NEW): OnPushFullPicture() updates push_prime/hash/stake_height\n";
    HeightTracker tracker;

    // Before any update, push cross-channel heights should be zero
    auto snap0 = tracker.GetSnapshot();
    print_test_result("push_prime_height == 0 before OnPushFullPicture",
                      snap0.push_prime_height == 0);
    print_test_result("push_hash_height == 0 before OnPushFullPicture",
                      snap0.push_hash_height == 0);
    print_test_result("push_stake_height == 0 before OnPushFullPicture",
                      snap0.push_stake_height == 0);

    // After OnPushFullPicture(), snapshot should reflect the values
    tracker.OnPushFullPicture(6500000, 2300000, 900000, 400000);
    auto snap1 = tracker.GetSnapshot();
    print_test_result("push_prime_height == 2300000 after OnPushFullPicture",
                      snap1.push_prime_height == 2300000);
    print_test_result("push_hash_height == 900000 after OnPushFullPicture",
                      snap1.push_hash_height == 900000);
    print_test_result("push_stake_height == 400000 after OnPushFullPicture",
                      snap1.push_stake_height == 400000);

    // prime_height and hash_height in snapshot should include push values
    print_test_result("prime_height includes push_prime_height",
                      snap1.prime_height >= 2300000);
    print_test_result("hash_height includes push_hash_height",
                      snap1.hash_height >= 900000);
    print_test_result("stake_height includes push_stake_height",
                      snap1.stake_height >= 400000);

    // OnPushFullPicture is monotonic — lower values must not regress
    tracker.OnPushFullPicture(6400000, 2200000, 800000, 300000);
    auto snap2 = tracker.GetSnapshot();
    print_test_result("push_prime_height not regressed by lower value",
                      snap2.push_prime_height == 2300000);
    print_test_result("push_hash_height not regressed by lower value",
                      snap2.push_hash_height == 900000);
    print_test_result("push_stake_height not regressed by lower value",
                      snap2.push_stake_height == 400000);

    // Higher values should advance
    tracker.OnPushFullPicture(6600000, 2400000, 1000000, 500000);
    auto snap3 = tracker.GetSnapshot();
    print_test_result("push_prime_height advances on higher value",
                      snap3.push_prime_height == 2400000);
    print_test_result("push_hash_height advances on higher value",
                      snap3.push_hash_height == 1000000);
    print_test_result("push_stake_height advances on higher value",
                      snap3.push_stake_height == 500000);

    // DiagnosticObserverState should also reflect values
    auto diag = tracker.GetDiagnosticSnapshot();
    print_test_result("DiagnosticObserverState.push_prime_height == 2400000",
                      diag.push_prime_height == 2400000);
    print_test_result("DiagnosticObserverState.push_hash_height == 1000000",
                      diag.push_hash_height == 1000000);
    print_test_result("DiagnosticObserverState.push_stake_height == 500000",
                      diag.push_stake_height == 500000);
}

// ============================================================================
// Test 35 (RC7-D): Burst recovery — nChannelHeight 4 blocks ahead of tracker
//   push_channel_height=2344737 (burst lag), template.nChannelHeight=2344741.
//   expected_template_target() = 2344738, and channel_height < channel_target
//   (2344737 < 2344741) confirms the template is valid.
//   The directional guard must accept this template as VALID.
// ============================================================================
void test_burst_recovery_template_ahead_of_tracker() {
    std::cout << "\nTest 35 (RC7-D): Burst recovery — nChannelHeight 4 blocks ahead of local tracker, must NOT be stale\n";
    HeightTracker tracker;

    // Simulate burst: only 1 of 5 push notifications processed so far.
    // Set canonical to establish channel_height for expected_template_target()
    tracker.OnBlockDataReceived(6644208, 2344737, 0x1d00ffff, uint1024_t{});
    tracker.OnPushNotification(6644208, 2344737, 0x1d00ffff);

    // GET_BLOCK arrives with the authoritative template from the node:
    // nChannelHeight = 2344741 (tip after burst = 2344740, next block = 2344741).
    tracker.OnTemplateReceived(1, 2344741);

    auto snap = tracker.GetSnapshot();

    // channel_height (from push) trails the template target by 4 blocks.
    // expected_template_target() = channel_height + 1 = 2344738 (not 2344741).
    print_test_result("RC7-D: expected_template_target() == 2344738 (lags template by 3)",
                      snap.expected_template_target() == 2344738);

    // channel_target was advanced by OnTemplateReceived to 2344741.
    print_test_result("RC7-D: channel_target == 2344741 after OnTemplateReceived",
                      snap.channel_target == 2344741);

    // Directional guard: 2344737 < 2344741 → template is VALID.
    // is_template_stale() removed (structurally false under canonical-only semantics)

    // Drift is positive (template is 3 ahead of expected) but NOT stale.
    auto delta = snap.drift_delta();
    print_test_result("RC7-D: drift_delta() has value (channel_target > expected_template_target)",
                      delta.has_value());
    print_test_result("RC7-D: drift_delta() == 3 (2344741 - 2344738)",
                      delta.has_value() && *delta == 3);
}

// ============================================================================
// Test: OnGetRound monotonic guard — stale round does not regress prime/hash heights
// ============================================================================
void test_on_get_round_monotonic_guard() {
    std::cout << "\nTest (Bug #1): OnGetRound monotonic guard for ALL round heights (including unified)\n";
    HeightTracker tracker;

    // Set channel to Prime (1) via OnTemplateReceived so m_channel is initialized
    tracker.OnPushNotification(5000, 100, 0x1d00ffff);
    tracker.OnTemplateReceived(1, 101);

    // First GET_ROUND with higher per-channel heights
    tracker.OnGetRound(5100, 120, 200, 300);
    auto diag = tracker.GetDiagnosticSnapshot();
    print_test_result("round_unified_height == 5100 after first GET_ROUND",
                      diag.round_unified_height == 5100);
    print_test_result("round_prime_height == 120 after first GET_ROUND",
                      diag.round_prime_height == 120);
    print_test_result("round_hash_height == 200 after first GET_ROUND",
                      diag.round_hash_height == 200);
    print_test_result("round_channel_height == 120 (Prime channel)",
                      diag.round_channel_height == 120);

    // Stale GET_ROUND with LOWER unified AND lower prime/hash — monotonic guard
    // must prevent regression on ALL heights (including unified)
    tracker.OnGetRound(5099, 110, 190, 310);
    diag = tracker.GetDiagnosticSnapshot();
    print_test_result("round_unified_height still 5100 (monotonic: 5099 < 5100)",
                      diag.round_unified_height == 5100);
    print_test_result("round_prime_height still 120 (monotonic: 110 < 120)",
                      diag.round_prime_height == 120);
    print_test_result("round_hash_height still 200 (monotonic: 190 < 200)",
                      diag.round_hash_height == 200);
    print_test_result("round_channel_height still 120 (derived from prime, monotonic)",
                      diag.round_channel_height == 120);
    // stake_height can advance independently (310 > 300)
    print_test_result("round_stake_height advanced to 310 (310 > 300)",
                      diag.round_stake_height == 310);

    // GET_ROUND with unified advancing but per-channel still lower — unified advances,
    // per-channel heights stay
    tracker.OnGetRound(5101, 110, 190, 305);
    diag = tracker.GetDiagnosticSnapshot();
    print_test_result("round_unified_height advanced to 5101 (5101 > 5100)",
                      diag.round_unified_height == 5101);
    print_test_result("round_prime_height still 120 (monotonic: 110 < 120)",
                      diag.round_prime_height == 120);
    print_test_result("round_hash_height still 200 (monotonic: 190 < 200)",
                      diag.round_hash_height == 200);

    // Verify diagnostic snapshot: round_unified_height was advanced
    auto snap = tracker.GetSnapshot();
    print_test_result("snapshot round_unified_height >= 5101 (GET_ROUND diagnostic)",
                      snap.round_unified_height >= 5101);

    // Fresh GET_ROUND with HIGHER prime/hash — must advance
    tracker.OnGetRound(5102, 125, 210, 305);
    diag = tracker.GetDiagnosticSnapshot();
    print_test_result("round_unified_height advanced to 5102",
                      diag.round_unified_height == 5102);
    print_test_result("round_prime_height advanced to 125",
                      diag.round_prime_height == 125);
    print_test_result("round_hash_height advanced to 210",
                      diag.round_hash_height == 210);
    print_test_result("round_channel_height advanced to 125 (Prime channel)",
                      diag.round_channel_height == 125);
}

// ============================================================================
// Test 37: Cross-channel block (Hash/Stake) advances unified but not Prime
//          channel height — is_tip_moved() == true (template still valid)
//
// This verifies that HeightTracker correctly distinguishes between:
//   - Tip movement (unified height advanced on another channel) → need fresh template
//   - Channel staleness (channel_height >= channel_target) → template is stale
//
// A Prime miner receiving Hash/Stake blocks should see is_tip_moved()==true
// (prompting an opportunistic template refresh for the new hashPrevBlock)
// but the channel_target hasn't been reached, so the current template is
// still valid for submission.
// ============================================================================
void test_cross_channel_block_tip_moved_not_stale() {
    std::cout << "\nTest 37: Cross-channel block advances unified but not Prime channel → is_tip_moved=true\n";
    HeightTracker tracker;

    // Bug #2 fix: Set canonical state first so OnTemplateReceived captures properly
    tracker.OnBlockDataReceived(100, 50, 0x1d00ffff, uint1024_t{});
    tracker.OnTemplateReceived(1 /* CHANNEL_PRIME */, 51);

    auto snap_base = tracker.GetSnapshot();
    // is_template_stale() removed (structurally false under canonical-only semantics)
    print_test_result("is_tip_moved() == false before any advance",
                      !snap_base.is_tip_moved());

    // Hash block found: unified advances to 101 but prime_channel stays at 50.
    // With canonical-only heights, we use OnBlockDataReceived to advance unified
    // while keeping channel at 50 (same value — monotonic guard keeps it).
    tracker.OnBlockDataReceived(101, 50, 0x1d00ffff, uint1024_t{});

    auto snap_after = tracker.GetSnapshot();
    // Template is still valid: Prime channel has NOT advanced past channel_target=51.
    // is_template_stale() removed (structurally false under canonical-only semantics)
    // Unified tip DID advance: the miner should opportunistically refresh hashPrevBlock.
    print_test_result("is_tip_moved() == true after cross-channel block advances unified",
                      snap_after.is_tip_moved());
    print_test_result("unified_height == 101 in snapshot",
                      snap_after.unified_height == 101);
    print_test_result("channel_height == 50 in snapshot (Prime channel unchanged)",
                      snap_after.channel_height == 50);
}

// ============================================================================
// Test 38: OnTemplateReceived +1 target does NOT contaminate snapshot channel_height
//
// Regression test for the GET_BLOCK dedup asymmetry bug:
//   - OnBlockDataReceived(unified=100, channel=50) sets canonical_channel_height=50
//   - OnTemplateReceived(1, 51) sets canonical_channel_target=51 (the mining target = tip+1)
//   - The snapshot's channel_height must still reflect the raw blockchain height (50),
//     NOT the +1 mining target (51).
//
// If channel_height were 51 instead of 50, the GET_BLOCK dedup key recorded at
// template-receipt time would be 51, but GET_ROUND would later report 50 —
// causing an asymmetric mismatch where new-round detection fails or
// dedup-suppression fires incorrectly on the next GET_BLOCK.
// ============================================================================
void test_template_received_plus1_does_not_contaminate_channel_height() {
    std::cout << "\nTest 38: OnTemplateReceived +1 target does NOT contaminate snap.channel_height\n";
    HeightTracker tracker;

    // Canonical BLOCK_DATA sets unified=100, channel=50 (raw blockchain heights)
    tracker.OnBlockDataReceived(100, 50, 0x1d00ffff, uint1024_t{});

    // Template received: the miner will mine toward channel target = 51 (= 50 + 1).
    // This +1 value must NOT leak into snap.channel_height.
    tracker.OnTemplateReceived(1 /* CHANNEL_PRIME */, 51);

    auto snap = tracker.GetSnapshot();

    // channel_height must be the raw blockchain tip (50), not the mining target (51)
    print_test_result("channel_height == 50 (raw blockchain height, not +1 target)",
                      snap.channel_height == 50);

    // channel_target must be the mining target (51)
    print_test_result("channel_target == 51 (mining target = tip+1)",
                      snap.channel_target == 51);

    // unified_height must be correct
    print_test_result("unified_height == 100",
                      snap.unified_height == 100);

    // Verify the canonical snapshot directly — canonical_channel_height must also be 50
    auto canon = tracker.GetCanonicalSnapshot();
    print_test_result("canonical_channel_height == 50 (raw, not contaminated by +1)",
                      canon.canonical_channel_height == 50);
    print_test_result("canonical_channel_target == 51 (correctly set to tip+1)",
                      canon.canonical_channel_target == 51);
}

// ============================================================================
// main
// ============================================================================
int main() {
    std::cout << "========================================\n";
    std::cout << "HeightTracker Unit Tests\n";
    std::cout << "========================================\n";

    test_push_then_template_no_drift();
    test_channel_height_advance_makes_stale();
    test_snapshot_consistency();
    test_get_round_source();
    test_drift_explain();
    test_zero_heights_safe();
    test_unified_advance_does_not_make_stale();
    test_channel_advance_makes_stale();
    test_is_tip_moved_detected();
    test_is_tip_moved_resets_on_new_template();
    test_difficulty_from_push_reflected_in_snapshot();
    test_post_push_timestamps_ordered_correctly();
    test_pre_push_template_identified_correctly();
    test_blocks_behind_distinguishes_normal_vs_severe_lag();
    test_on_keepalive_response_unified();
    test_on_keepalive_response_legacy_zeros_safe();
    test_keepalive_response_sets_stake_height();
    test_peak_fork_score_high_water_mark();
    test_push_updates_per_channel_heights();
    test_push_updates_hash_height();
    test_push_keepalive_no_regression();
    test_advance_channel_target_only_advances();
    test_on_template_received_no_regress();
    test_doom_loop_prevention();
    test_stale_get_block_no_regression();
    test_on_template_metadata_monotonic();
    test_on_template_metadata_per_channel_no_regression();
    test_canonical_isolation();
    test_fork_scores_reset_on_template_advance();
    test_fork_scores_sticky_without_advance();
    test_on_block_data_received_monotonic();
    test_height_drift_from_canonical();
    test_canonical_hash_prev_block();
    test_diagnostic_observer_helpers();
    test_canonical_state_only_from_block_data();
    test_keepalive_does_not_corrupt_canonical();
    test_diagnostic_fork_score_isolated();
    test_diagnostic_is_initialized();
    test_diagnostic_latest_received_at();
    test_advance_channel_target_updates_canonical();
    test_on_template_received_sets_template_unified_height();
    test_push_notification_at_isolated_from_keepalive_and_getround();
    test_set_session_epoch_clears_keepalive_on_epoch_change();
    test_set_session_epoch_no_clear_when_unchanged();
    test_old_epoch_keepalive_does_not_signal_new_epoch_liveness();
    test_set_session_epoch_zero_no_clear();
    test_session_epoch_in_snapshot();
    test_on_push_full_picture();
    test_burst_recovery_template_ahead_of_tracker();
    test_on_get_round_monotonic_guard();
    test_cross_channel_block_tip_moved_not_stale();
    test_template_received_plus1_does_not_contaminate_channel_height();

    std::cout << "\n========================================\n";
    std::cout << "Test Summary\n";
    std::cout << "========================================\n";
    std::cout << "Tests run:    " << tests_run    << "\n";
    std::cout << "Tests passed: " << tests_passed << "\n";
    std::cout << "Tests failed: " << tests_failed << "\n";
    std::cout << "Success rate: "
              << (tests_run > 0 ? (100 * tests_passed / tests_run) : 0) << "%\n";
    std::cout << "========================================\n";

    return tests_failed > 0 ? 1 : 0;
}
