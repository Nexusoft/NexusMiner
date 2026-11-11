/**
 * @file fork_vs_tip_change_test.cpp
 * @brief Unit tests for fork detection vs tip change semantics
 *
 * Validates the core invariant: fork warnings (⚠) are reserved for actual
 * blockchain forks (height regression), while tip changes (⚡) are used for
 * normal chain advancement detected by GET_ROUND or PUSH.
 *
 * Tests:
 *  1. Normal tip advance does NOT trigger fork detection
 *  2. Height regression DOES trigger fork detection (true fork)
 *  3. Same-height update does NOT trigger fork detection (tip change)
 *  4. Large forward jump does NOT trigger fork detection
 *  5. Fork flag cleared after ClearForkFlag()
 *  6. Multiple sequential advances never trigger fork detection
 *  7. Regression after advance triggers fork detection
 *  8. Zero-height initial update does NOT trigger fork detection
 *  9. Single-block regression = Phantom Stake (NOT fork — cross-channel tip oscillation)
 * 10. Channel height regression alone does NOT trigger fork detection
 *     (only unified height regression triggers fork)
 * 11. GetBlockReason GET_ROUND variants bypass height dedup (tip change support)
 * 12. GetBlockReason PUSH variants bypass height dedup
 * 13. Only RECOVERY reasons bypass ALL dedup (not tip changes)
 * 14. HeightTracker tip advance is distinct from fork
 * 15. 2-block regression IS a real fork
 */

#include "mining/client_channel_manager.h"
#include "protocol/height_tracker.hpp"
#include "protocol/get_block_reason.hpp"
#include <iostream>
#include <cassert>
#include <cstdint>

using namespace nexusminer;
using namespace nexusminer::mining;
using namespace nexusminer::protocol;

// Test statistics
static int tests_run    = 0;
static int tests_passed = 0;
static int tests_failed = 0;

void print_test_result(const char* name, bool passed)
{
    tests_run++;
    if (passed) {
        tests_passed++;
        std::cout << "  [PASS] " << name << "\n";
    } else {
        tests_failed++;
        std::cout << "  [FAIL] " << name << "\n";
    }
}

// Concrete subclass for testing — uses Prime channel (1) which is the primary mining channel
class TestPrimeManager : public ClientChannelManager
{
public:
    TestPrimeManager() : ClientChannelManager(1) {}
    uint32_t GetNodeUnifiedHeight() const { return m_nNodeUnifiedHeight.load(); }
    uint32_t GetNodeChannelHeight() const { return m_nNodeChannelHeight.load(); }
};

// ============================================================================
// Test 1: Normal tip advance does NOT trigger fork detection
// ============================================================================
static void test_normal_advance_no_fork()
{
    std::cout << "\nTest 1: Normal tip advance does NOT trigger fork detection\n";

    TestPrimeManager mgr;

    // Initial state: height 5000
    mgr.UpdateFromGetRound(5000, 200);
    print_test_result("No fork on initial update", !mgr.IsForkDetected());

    // Normal advance: 5000 → 5001
    mgr.UpdateFromGetRound(5001, 201);
    print_test_result("No fork on +1 advance", !mgr.IsForkDetected());

    // Another advance: 5001 → 5002
    mgr.UpdateFromGetRound(5002, 202);
    print_test_result("No fork on +1 advance (second)", !mgr.IsForkDetected());
}

// ============================================================================
// Test 2: Height regression DOES trigger fork detection (true fork)
// ============================================================================
static void test_height_regression_is_fork()
{
    std::cout << "\nTest 2: Height regression DOES trigger fork detection (true fork)\n";

    TestPrimeManager mgr;

    // Set initial height
    mgr.UpdateFromGetRound(5000, 200);
    print_test_result("No fork initially", !mgr.IsForkDetected());

    // Regress: 5000 → 4998 (2-block rollback = true fork)
    mgr.UpdateFromGetRound(4998, 198);
    print_test_result("Fork detected on 2-block regression", mgr.IsForkDetected());
}

// ============================================================================
// Test 3: Same-height update does NOT trigger fork detection (tip change)
// ============================================================================
static void test_same_height_no_fork()
{
    std::cout << "\nTest 3: Same-height update does NOT trigger fork (tip change)\n";

    TestPrimeManager mgr;

    // Set initial height
    mgr.UpdateFromGetRound(5000, 200);
    print_test_result("No fork initially", !mgr.IsForkDetected());

    // Same unified height again (different GET_ROUND, same block = tip change/reorg at same height)
    mgr.UpdateFromGetRound(5000, 200);
    print_test_result("No fork on same-height update (tip change)", !mgr.IsForkDetected());

    // Same unified but channel advanced (another channel found a block)
    mgr.UpdateFromGetRound(5000, 201);
    print_test_result("No fork on same-unified/channel-advance", !mgr.IsForkDetected());
}

// ============================================================================
// Test 4: Large forward jump does NOT trigger fork detection
// ============================================================================
static void test_large_forward_jump_no_fork()
{
    std::cout << "\nTest 4: Large forward jump does NOT trigger fork detection\n";

    TestPrimeManager mgr;

    mgr.UpdateFromGetRound(5000, 200);
    print_test_result("No fork initially", !mgr.IsForkDetected());

    // Jump forward by 100 blocks (burst recovery scenario)
    mgr.UpdateFromGetRound(5100, 210);
    print_test_result("No fork on +100 jump", !mgr.IsForkDetected());
}

// ============================================================================
// Test 5: Fork flag cleared after ClearForkFlag()
// ============================================================================
static void test_fork_flag_clear()
{
    std::cout << "\nTest 5: Fork flag cleared after ClearForkFlag()\n";

    TestPrimeManager mgr;

    mgr.UpdateFromGetRound(5000, 200);
    mgr.UpdateFromGetRound(4998, 198);  // 2-block regression = genuine fork
    print_test_result("Fork detected", mgr.IsForkDetected());

    mgr.ClearForkFlag();
    print_test_result("Fork flag cleared", !mgr.IsForkDetected());

    // Subsequent normal advance should not show fork
    mgr.UpdateFromGetRound(5001, 201);
    print_test_result("No fork after clear + normal advance", !mgr.IsForkDetected());
}

// ============================================================================
// Test 6: Multiple sequential advances never trigger fork detection
// ============================================================================
static void test_sequential_advances_no_fork()
{
    std::cout << "\nTest 6: Multiple sequential advances never trigger fork\n";

    TestPrimeManager mgr;

    for (uint32_t h = 1000; h <= 1050; ++h) {
        mgr.UpdateFromGetRound(h, h / 10);
        if (mgr.IsForkDetected()) {
            print_test_result("No fork during sequential advance", false);
            return;
        }
    }
    print_test_result("No fork during 50 sequential advances", true);
}

// ============================================================================
// Test 7: Regression after advance triggers fork detection
// ============================================================================
static void test_regression_after_advance()
{
    std::cout << "\nTest 7: Regression after advance triggers fork detection\n";

    TestPrimeManager mgr;

    mgr.UpdateFromGetRound(5000, 200);
    mgr.UpdateFromGetRound(5001, 201);  // Normal advance
    mgr.UpdateFromGetRound(5002, 202);  // Normal advance
    print_test_result("No fork during advances", !mgr.IsForkDetected());

    // Now regress: 5002 → 5000 (2-block rollback)
    mgr.UpdateFromGetRound(5000, 200);
    print_test_result("Fork detected after regression", mgr.IsForkDetected());
}

// ============================================================================
// Test 8: Zero-height initial update does NOT trigger fork detection
// ============================================================================
static void test_zero_height_no_fork()
{
    std::cout << "\nTest 8: Zero-height initial update does NOT trigger fork\n";

    TestPrimeManager mgr;

    // First update with height 0 (edge case: node just started)
    mgr.UpdateFromGetRound(0, 0);
    print_test_result("No fork on zero-height initial", !mgr.IsForkDetected());

    // Normal advance from 0
    mgr.UpdateFromGetRound(1, 1);
    print_test_result("No fork on advance from 0", !mgr.IsForkDetected());
}

// ============================================================================
// Test 9: Single-block regression = Phantom Stake (NOT a fork)
// ============================================================================
static void test_single_block_regression()
{
    std::cout << "\nTest 9: Single-block regression = Phantom Stake (NOT fork)\n";

    TestPrimeManager mgr;

    mgr.UpdateFromGetRound(5000, 200);
    mgr.UpdateFromGetRound(4999, 200);  // 1-block regression = Phantom Stake

    print_test_result("IsForkDetected == false (phantom stake, not real fork)", !mgr.IsForkDetected());
    print_test_result("IsPhantomStakeRegression == true", mgr.IsPhantomStakeRegression());

    mgr.ClearForkFlag();
    print_test_result("Phantom flag cleared after ClearForkFlag()", !mgr.IsPhantomStakeRegression());
}

// ============================================================================
// Test 10: Channel height regression alone does NOT trigger fork
// ============================================================================
static void test_channel_regression_no_fork()
{
    std::cout << "\nTest 10: Channel height regression alone does NOT trigger fork\n";

    TestPrimeManager mgr;

    mgr.UpdateFromGetRound(5000, 200);
    // Unified advances but channel goes back (shouldn't happen normally,
    // but fork detection only tracks unified height)
    mgr.UpdateFromGetRound(5001, 199);
    print_test_result("No fork when only channel regresses but unified advances",
                      !mgr.IsForkDetected());
}

// ============================================================================
// Test 11: GET_ROUND reasons bypass height dedup (tip change support)
// ============================================================================
static void test_get_round_reasons_bypass_height_dedup()
{
    std::cout << "\nTest 11: GET_ROUND reasons bypass height dedup (tip change)\n";

    // GET_ROUND reasons are tip-change-driven, not fork-driven
    print_test_result("GET_ROUND_STALE bypasses height dedup",
        should_bypass_height_dedup(GetBlockReason::GET_ROUND_STALE));
    print_test_result("GET_ROUND_NO_TEMPLATE bypasses height dedup",
        should_bypass_height_dedup(GetBlockReason::GET_ROUND_NO_TEMPLATE));
    print_test_result("GET_ROUND_HEIGHT_PARITY bypasses height dedup",
        should_bypass_height_dedup(GetBlockReason::GET_ROUND_HEIGHT_PARITY));

    // But they should NOT bypass ALL dedup (they're not recovery)
    print_test_result("GET_ROUND_STALE does NOT bypass all dedup",
        !should_bypass_all_dedup(GetBlockReason::GET_ROUND_STALE));
    print_test_result("GET_ROUND_NO_TEMPLATE does NOT bypass all dedup",
        !should_bypass_all_dedup(GetBlockReason::GET_ROUND_NO_TEMPLATE));
    print_test_result("GET_ROUND_HEIGHT_PARITY does NOT bypass all dedup",
        !should_bypass_all_dedup(GetBlockReason::GET_ROUND_HEIGHT_PARITY));
}

// ============================================================================
// Test 12: PUSH reasons bypass height dedup
// ============================================================================
static void test_push_reasons_bypass_height_dedup()
{
    std::cout << "\nTest 12: PUSH reasons bypass height dedup\n";

    // Remaining PUSH reasons (PUSH_STALE, PUSH_NO_TEMPLATE, PUSH_CROSS_CHANNEL
    // removed — NODE auto-sends BLOCK_DATA after PUSH)
    print_test_result("PUSH_TIP_MOVED bypasses height dedup",
        should_bypass_height_dedup(GetBlockReason::PUSH_TIP_MOVED));
    print_test_result("PUSH_SAME_HEIGHT_TIP bypasses height dedup",
        should_bypass_height_dedup(GetBlockReason::PUSH_SAME_HEIGHT_TIP));

    // PUSH reasons should NOT bypass ALL dedup
    print_test_result("PUSH_TIP_MOVED does NOT bypass all dedup",
        !should_bypass_all_dedup(GetBlockReason::PUSH_TIP_MOVED));
}

// ============================================================================
// Test 13: Only RECOVERY reasons bypass ALL dedup
// ============================================================================
static void test_recovery_bypasses_all_dedup()
{
    std::cout << "\nTest 13: Only RECOVERY reasons bypass ALL dedup\n";

    print_test_result("RECOVERY_FORCED bypasses all dedup",
        should_bypass_all_dedup(GetBlockReason::RECOVERY_FORCED));
    print_test_result("RECOVERY_TIMER bypasses all dedup",
        should_bypass_all_dedup(GetBlockReason::RECOVERY_TIMER));
    print_test_result("HEALTH_NO_TEMPLATE bypasses all dedup",
        should_bypass_all_dedup(GetBlockReason::HEALTH_NO_TEMPLATE));

    // Non-recovery reasons should NOT bypass all dedup
    print_test_result("INITIAL_REQUEST does NOT bypass all dedup",
        !should_bypass_all_dedup(GetBlockReason::INITIAL_REQUEST));
    print_test_result("HEALTH_CHANNEL_ADVANCE does NOT bypass all dedup",
        !should_bypass_all_dedup(GetBlockReason::HEALTH_CHANNEL_ADVANCE));
}

// ============================================================================
// Test 14: HeightTracker tip advance is distinct from fork
// ============================================================================
static void test_height_tracker_tip_advance_not_fork()
{
    std::cout << "\nTest 14: HeightTracker tip advance is distinct from fork\n";

    HeightTracker tracker;

    // Bug #2 fix: Set canonical state so OnTemplateReceived captures properly
    tracker.OnBlockDataReceived(5000, 200, 0x1d00ffff, uint1024_t{});
    auto snap1 = tracker.GetSnapshot();
    print_test_result("Initial unified_height = 5000", snap1.unified_height == 5000);

    // Simulate template received at unified height 5000 (target = channel 201)
    tracker.OnTemplateReceived(1, 201);
    auto snap1b = tracker.GetSnapshot();
    print_test_result("template_unified_height set to 5000",
                      snap1b.template_unified_height == 5000);

    // Simulate canonical advance to 5001 (BLOCK_DATA is the canonical source)
    tracker.OnBlockDataReceived(5001, 201, 0x1d00ffff, uint1024_t{});
    auto snap2 = tracker.GetSnapshot();
    print_test_result("Unified advanced to 5001 via GET_ROUND",
                      snap2.unified_height == 5001);
    // is_tip_moved() checks unified_height > template_unified_height
    print_test_result("is_tip_moved() after GET_ROUND advance (tip > template)",
                      snap2.is_tip_moved());
    tracker.OnTemplateReceived(1, 202);
    auto snap3 = tracker.GetSnapshot();
    print_test_result("is_tip_moved() reset after template received",
                      !snap3.is_tip_moved());
}

// ============================================================================
// Test 15: 2-block regression IS a real fork
// ============================================================================
static void test_two_block_regression_is_real_fork()
{
    std::cout << "\nTest 15: 2-block regression IS a real fork\n";

    TestPrimeManager mgr;

    mgr.UpdateFromGetRound(5000, 200);
    mgr.UpdateFromGetRound(4998, 198);  // 2-block rollback

    print_test_result("IsForkDetected == true (real fork)", mgr.IsForkDetected());
    print_test_result("IsPhantomStakeRegression == false", !mgr.IsPhantomStakeRegression());
}

// ============================================================================
// Main
// ============================================================================
int main()
{
    std::cout << "═══════════════════════════════════════════════════════════\n";
    std::cout << "  Fork vs Tip Change Semantics Test\n";
    std::cout << "  ⚠ = FORK (height regression only)\n";
    std::cout << "  ⚡ = TIP CHANGE (normal advance, GET_ROUND/PUSH driven)\n";
    std::cout << "═══════════════════════════════════════════════════════════\n";

    test_normal_advance_no_fork();
    test_height_regression_is_fork();
    test_same_height_no_fork();
    test_large_forward_jump_no_fork();
    test_fork_flag_clear();
    test_sequential_advances_no_fork();
    test_regression_after_advance();
    test_zero_height_no_fork();
    test_single_block_regression();
    test_channel_regression_no_fork();
    test_get_round_reasons_bypass_height_dedup();
    test_push_reasons_bypass_height_dedup();
    test_recovery_bypasses_all_dedup();
    test_height_tracker_tip_advance_not_fork();
    test_two_block_regression_is_real_fork();

    std::cout << "\n═══════════════════════════════════════════════════════════\n";
    std::cout << "Test Results: " << tests_passed << "/" << tests_run << " passed\n";
    std::cout << "═══════════════════════════════════════════════════════════\n";

    return tests_failed > 0 ? 1 : 0;
}
