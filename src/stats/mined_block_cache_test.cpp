/**
 * @file mined_block_cache_test.cpp
 * @brief Unit tests for MinedBlockCache three-tier confirmation cache and
 *        Worker-feed dedup guard.
 *
 * Tests:
 *   MinedBlockCache:
 *    1.  record_accepted_block() stores one record correctly in Tier 1
 *    2.  After TIER1_MAX records, oldest is evicted from Tier 1 to Tier 2
 *    3.  After TIER2_MAX records in Tier 2, overflow moves to Tier 3
 *    4.  update_confirmations() increments confirmations for all Tier 1 records
 *    5.  A record reaching CONFIRMATION_THRESHOLD is promoted from Tier 1 to Tier 2
 *   5b.  Height-gated update_confirmations skips same-height calls
 *   5c.  Sole record in Tier 1 is NOT promoted even when confirmed
 *   5d.  Promotion-driven Tier 2 overflow cascades to Tier 3
 *   5e.  Full lifecycle: Tier 1 → Tier 2 (confirm) → Tier 3 (overflow)
 *    6.  tier1() returns at most 5 records in newest-first order
 *    7.  channel_name() returns correct strings
 *    8.  status_emoji() returns correct emoji below and at threshold
 *    9.  summary_line() is non-empty and contains the height
 *
 *   Worker-feed dedup guard (standalone, no asio dependency):
 *   10.  Same (height, hashPrevBlock) within 2 000 ms is suppressed
 *   11.  Same height but different hashPrevBlock (fork) is NOT suppressed
 *   12.  Same pair after > 2 000 ms is NOT suppressed (debounce expired)
 *
 *   Concurrent recovery gate:
 *   13.  Two simultaneous recovery attempts: only one executes at a time
 */

#include "stats/mined_block_cache.hpp"
#include <LLC/types/uint1024.h>

#include <cassert>
#include <chrono>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

using nexusminer::stats::MinedBlockCache;
using nexusminer::stats::MinedBlockRecord;

// ── Test harness ─────────────────────────────────────────────────────────────

static int g_run    = 0;
static int g_passed = 0;
static int g_failed = 0;

static void check(const char* name, bool ok)
{
    ++g_run;
    if (ok) {
        ++g_passed;
        std::cout << "  [PASS] " << name << "\n";
    } else {
        ++g_failed;
        std::cout << "  [FAIL] " << name << "\n";
    }
}

// ── Minimal dedup-guard state (mirrors worker_manager.hpp fields) ─────────────

struct DedupGuard
{
    std::chrono::steady_clock::time_point last_tp{};
    uint32_t last_height{0};
    uint1024_t last_prev_hash{0};
    static constexpr int64_t DEBOUNCE_MS = 2000;

    /// Returns true if the template should be passed through (not suppressed).
    bool should_pass(uint32_t height, uint1024_t const& prev_hash,
                     std::chrono::steady_clock::time_point now)
    {
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      now - last_tp).count();
        bool same = (height == last_height && prev_hash == last_prev_hash);
        if (same && ms < DEBOUNCE_MS)
            return false;  // suppress

        last_tp        = now;
        last_height    = height;
        last_prev_hash = prev_hash;
        return true;
    }
};

// ── Test 1: record_accepted_block stores one record in Tier 1 ─────────────────
static void test_record_one_block()
{
    std::cout << "\nTest 1: record_accepted_block stores one record in Tier 1\n";

    MinedBlockCache cache;
    uint1024_t prev{0};
    cache.record_accepted_block(100, prev, 1, 0xDEADBEEF);

    check("tier1() has 1 entry",       cache.tier1().size() == 1);
    check("tier2() is empty",          cache.tier2().empty());
    check("tier3() is empty",          cache.tier3().empty());
    check("height stored correctly",   cache.tier1().front().height  == 100);
    check("channel stored correctly",  cache.tier1().front().channel == 1);
    check("nonce stored correctly",    cache.tier1().front().nonce   == 0xDEADBEEF);
    check("confirmations start at 0",  cache.tier1().front().confirmations == 0);
}

// ── Test 2: after TIER1_MAX records, oldest is evicted to Tier 2 ───────────
static void test_tier1_eviction_to_tier2()
{
    std::cout << "\nTest 2: After TIER1_MAX records, oldest evicted to Tier 2\n";

    MinedBlockCache cache;
    uint1024_t prev{0};
    // Insert TIER1_MAX + 1 blocks so the first one overflows.
    for (uint32_t h = 1; h <= MinedBlockCache::TIER1_MAX + 1; ++h)
        cache.record_accepted_block(h, prev, 1, h);

    check("tier1() size == TIER1_MAX",
          cache.tier1().size() == MinedBlockCache::TIER1_MAX);
    check("tier2() has 1 entry",  cache.tier2().size() == 1);
    check("tier3() is empty",     cache.tier3().empty());

    // Tier 2 should contain the oldest block (height = 1)
    check("tier2 front is oldest block",
          cache.tier2().front().height == 1);

    // Tier 1 front is the newest block
    check("tier1 front is newest block",
          cache.tier1().front().height == MinedBlockCache::TIER1_MAX + 1);
}

// ── Test 3: after TIER2_MAX records in Tier 2, overflow to Tier 3 ────────────
static void test_tier2_overflow_to_tier3()
{
    std::cout << "\nTest 3: After TIER2_MAX records in Tier 2, overflow to Tier 3\n";

    MinedBlockCache cache;
    uint1024_t prev{0};
    // Insert TIER2_MAX + TIER1_MAX + 1 blocks total to fill Tier 2 and trigger archive
    size_t total = MinedBlockCache::TIER2_MAX + MinedBlockCache::TIER1_MAX + 1;
    for (uint32_t h = 1; h <= static_cast<uint32_t>(total); ++h)
        cache.record_accepted_block(h, prev, 1, h);

    check("tier1() == TIER1_MAX",  cache.tier1().size() == MinedBlockCache::TIER1_MAX);
    check("tier2() == TIER2_MAX",  cache.tier2().size() == MinedBlockCache::TIER2_MAX);
    check("tier3() has 1 entry",   cache.tier3().size() == 1);
    check("total_blocks correct",
          cache.total_blocks() ==
              MinedBlockCache::TIER1_MAX + MinedBlockCache::TIER2_MAX + 1);
}

// ── Test 4: update_confirmations increments for all Tier 1 records ────────────
static void test_update_confirmations()
{
    std::cout << "\nTest 4: update_confirmations increments confirmations for Tier 1\n";

    MinedBlockCache cache;
    uint1024_t prev{0};
    cache.record_accepted_block(10, prev, 2, 1);
    cache.record_accepted_block(20, prev, 2, 2);

    // Simulate chain advancing to height 22
    cache.update_confirmations(22);

    // The record at height 20 should have confirmations = 22 - 20 + 1 = 3
    // The record at height 10 should have confirmations = 22 - 10 + 1 = 13,
    // but it stays in Tier 1 only if it's the front (most recent).
    // Both records are still in Tier 1 (size 2 < TIER1_MAX).
    bool conf_ok = false;
    for (const auto& r : cache.tier1()) {
        if (r.height == 20 && r.confirmations == 3) conf_ok = true;
    }
    check("height=20 confirmations == 3", conf_ok);
}

// ── Test 5: record reaching CONFIRMATION_THRESHOLD promoted Tier 1 → Tier 2 ──
static void test_confirmation_threshold_promotion()
{
    std::cout << "\nTest 5: CONFIRMATION_THRESHOLD promotes Tier 1 → Tier 2\n";

    MinedBlockCache cache;
    uint1024_t prev{0};
    // Add 2 records so that oldest is not the sole front record
    cache.record_accepted_block(10, prev, 1, 1);
    cache.record_accepted_block(20, prev, 1, 2);

    // Advance chain height so height=10 gets >= 5 confirmations
    // height=10: confs = (10 + 5) - 10 + 1 = 6
    cache.update_confirmations(10 + MinedBlockCache::CONFIRMATION_THRESHOLD);

    // The block at height=10 (oldest in Tier 1, not front) should be promoted
    bool in_tier2 = false;
    for (const auto& r : cache.tier2())
        if (r.height == 10) in_tier2 = true;

    check("height=10 promoted to Tier 2", in_tier2);
    check("height=20 still in Tier 1",    !cache.tier1().empty() &&
                                           cache.tier1().front().height == 20);
}

// ── Test 5b: height-gated update_confirmations skips same-height calls ────────
static void test_height_gated_confirmations()
{
    std::cout << "\nTest 5b: update_confirmations is height-gated (same height = no-op)\n";

    MinedBlockCache cache;
    uint1024_t prev{0};
    cache.record_accepted_block(100, prev, 1, 1);

    // First call at height 102: confirmations = 102 - 100 + 1 = 3
    cache.update_confirmations(102);
    check("height=100 conf==3 after chain at 102",
          cache.tier1().front().confirmations == 3);

    // Add a new block at height 101 — it starts with conf=0
    cache.record_accepted_block(101, prev, 2, 2);
    check("new block at height 101 starts with conf=0",
          cache.tier1().front().confirmations == 0);

    // Call update_confirmations(102) again — same height as before.
    // The height-gate should skip recalculation, so the new block
    // at height=101 should still have conf=0.
    cache.update_confirmations(102);
    check("same-height call is no-op (new block conf still 0)",
          cache.tier1().front().confirmations == 0);

    // Advancing height to 103 should recalculate:
    // height=101: conf = 103 - 101 + 1 = 3
    // height=100: conf = 103 - 100 + 1 = 4
    cache.update_confirmations(103);
    check("higher height recalculates (h=101 conf==3)",
          cache.tier1().front().confirmations == 3);
}

// ── Test 5c: sole record in Tier 1 is NOT promoted even when confirmed ────────
static void test_sole_record_stays_in_tier1()
{
    std::cout << "\nTest 5c: Sole record in Tier 1 is NOT promoted even when confirmed\n";

    MinedBlockCache cache;
    uint1024_t prev{0};
    // Add only 1 record — it is both front and back
    cache.record_accepted_block(50, prev, 2, 0xAAAA);

    // Advance chain far past confirmation threshold
    cache.update_confirmations(50 + MinedBlockCache::CONFIRMATION_THRESHOLD + 10);

    check("tier1() still has 1 entry", cache.tier1().size() == 1);
    check("tier2() is empty",          cache.tier2().empty());
    check("sole record stays in Tier 1 (height=50)",
          cache.tier1().front().height == 50);
    check("sole record has correct confirmations",
          cache.tier1().front().confirmations ==
              MinedBlockCache::CONFIRMATION_THRESHOLD + 10 + 1);
}

// ── Test 5d: promotion-driven Tier 2 overflow cascades to Tier 3 ─────────────
static void test_promotion_driven_tier2_overflow()
{
    std::cout << "\nTest 5d: Promotion-driven Tier 2 overflow cascades to Tier 3\n";

    MinedBlockCache cache;
    uint1024_t prev{0};
    // Fill Tier 2 to capacity via record_accepted_block overflow:
    // Insert TIER1_MAX + TIER2_MAX blocks so that Tier 2 is exactly full.
    size_t total = MinedBlockCache::TIER1_MAX + MinedBlockCache::TIER2_MAX;
    for (uint32_t h = 1; h <= static_cast<uint32_t>(total); ++h)
        cache.record_accepted_block(h, prev, 1, h);

    check("tier2 is full (== TIER2_MAX)",
          cache.tier2().size() == MinedBlockCache::TIER2_MAX);
    check("tier3 is empty before promotion", cache.tier3().empty());

    // Now add one more block so we have 2 in Tier 1 (needed for promotion)
    uint32_t newest_height = static_cast<uint32_t>(total) + 1;
    cache.record_accepted_block(newest_height, prev, 2, newest_height);
    // Tier 1 still has TIER1_MAX entries; the overflow went to Tier 2 which
    // itself overflowed 1 to Tier 3.
    // But let's now trigger confirmation-based promotion:

    // Advance chain so older Tier 1 blocks (but not the newest) reach threshold.
    // Tier 1 has blocks from (total - TIER1_MAX + 2) to (total + 1).
    // The oldest block in Tier 1 is at height (total - TIER1_MAX + 2).
    uint32_t oldest_in_tier1 = cache.tier1().back().height;
    uint32_t confirm_height = oldest_in_tier1 + MinedBlockCache::CONFIRMATION_THRESHOLD;
    cache.update_confirmations(confirm_height);

    // Oldest Tier 1 block should be promoted to Tier 2, which should overflow to Tier 3
    size_t total_blocks = cache.tier1().size() + cache.tier2().size() + cache.tier3().size();
    check("total blocks preserved after promotion",
          total_blocks == MinedBlockCache::TIER1_MAX + MinedBlockCache::TIER2_MAX + 1);
    check("tier3 has overflow after promotion-driven cascade",
          cache.tier3().size() >= 1);
}

// ── Test 5e: full lifecycle Tier 1 → Tier 2 (confirm) → Tier 3 (overflow) ──
static void test_full_promotion_lifecycle()
{
    std::cout << "\nTest 5e: Full lifecycle: Tier 1 → Tier 2 (confirm) → Tier 3 (overflow)\n";

    MinedBlockCache cache;
    uint1024_t prev{0};

    // Phase 1: Add 2 blocks to Tier 1
    cache.record_accepted_block(100, prev, 1, 1);
    cache.record_accepted_block(200, prev, 2, 2);
    check("Phase 1: 2 blocks in Tier 1", cache.tier1().size() == 2);

    // Phase 2: Confirm h=100 → promote to Tier 2
    cache.update_confirmations(100 + MinedBlockCache::CONFIRMATION_THRESHOLD);
    check("Phase 2: h=100 promoted to Tier 2", cache.tier2().size() == 1);
    check("Phase 2: h=200 stays in Tier 1",
          cache.tier1().size() == 1 && cache.tier1().front().height == 200);

    // Phase 3: Fill Tier 2 to max via bulk inserts (overflow older ones out of Tier 1)
    uint32_t const fill_count = MinedBlockCache::TIER2_MAX + MinedBlockCache::TIER1_MAX;
    for (uint32_t h = 300; h < 300 + fill_count; ++h)
        cache.record_accepted_block(h, prev, 1, h);

    check("Phase 3: Tier 2 at capacity", cache.tier2().size() == MinedBlockCache::TIER2_MAX);
    check("Phase 3: Tier 3 has overflow (includes h=100)",
          cache.tier3().size() >= 1);

    // Verify h=100 ended up in Tier 3 (archived)
    bool found_in_t3 = false;
    for (auto const& r : cache.tier3())
        if (r.height == 100) found_in_t3 = true;
    check("Phase 3: h=100 is in Tier 3 (archive)", found_in_t3);
}

// ── Test 6: tier1() returns at most 5 records, newest-first ────────────────
static void test_tier1_newest_first()
{
    std::cout << "\nTest 6: tier1() returns at most 5 records, newest-first\n";

    MinedBlockCache cache;
    uint1024_t prev{0};
    for (uint32_t h = 1; h <= 10; ++h)
        cache.record_accepted_block(h, prev, 1, h);

    check("tier1() size <= 5",
          cache.tier1().size() <= MinedBlockCache::TIER1_MAX);

    // Newest-first: front has the highest height among Tier 1
    bool newest_first = !cache.tier1().empty() &&
                        cache.tier1().front().height > cache.tier1().back().height;
    check("tier1() is newest-first", newest_first);
    check("tier1() front is height 10", cache.tier1().front().height == 10);
}

// ── Test 7: channel_name() ────────────────────────────────────────────────────
static void test_channel_name()
{
    std::cout << "\nTest 7: channel_name() returns correct strings\n";

    MinedBlockRecord r;
    r.channel = 1;
    check("channel 1 == Prime",   r.channel_name() == "Prime");
    r.channel = 2;
    check("channel 2 == Hash",    r.channel_name() == "Hash");
    r.channel = 0;
    check("channel 0 == Unknown", r.channel_name() == "Unknown");
    r.channel = 99;
    check("channel 99 == Unknown", r.channel_name() == "Unknown");
}

// ── Test 8: status_emoji() ────────────────────────────────────────────────────
static void test_status_emoji()
{
    std::cout << "\nTest 8: status_emoji() below and at/above threshold\n";

    MinedBlockRecord r;
    r.confirmations = 0;
    check("0 confirmations → ⛏", r.status_emoji() == "⛏");
    r.confirmations = MinedBlockCache::CONFIRMATION_THRESHOLD - 1;
    check("threshold-1 → ⛏", r.status_emoji() == "⛏");
    r.confirmations = MinedBlockCache::CONFIRMATION_THRESHOLD;
    check("threshold → ✅", r.status_emoji() == "✅");
    r.confirmations = MinedBlockCache::CONFIRMATION_THRESHOLD + 10;
    check("threshold+10 → ✅", r.status_emoji() == "✅");
}

// ── Test 9: summary_line() is non-empty and contains the height ───────────────
static void test_summary_line()
{
    std::cout << "\nTest 9: summary_line() is non-empty and contains height\n";

    MinedBlockRecord r;
    r.height = 12345;
    r.channel = 1;
    r.confirmations = 0;
    r.hash_prev_block = uint1024_t{0};

    std::string s = r.summary_line();
    check("summary_line() is non-empty", !s.empty());
    check("summary_line() contains '12345'", s.find("12345") != std::string::npos);
}

// ── Test 10: same (height, hashPrevBlock) within 2000 ms is suppressed ────────
static void test_dedup_suppresses_same_pair_within_window()
{
    std::cout << "\nTest 10: Same (height, hashPrevBlock) within 2000 ms is suppressed\n";

    DedupGuard guard;
    uint1024_t prev{0};
    auto t0 = std::chrono::steady_clock::now();

    // First call: should pass through
    bool first = guard.should_pass(100, prev, t0);
    check("First call passes through", first);

    // Second call 500 ms later with same pair: should be suppressed
    auto t1 = t0 + std::chrono::milliseconds(500);
    bool second = guard.should_pass(100, prev, t1);
    check("Second call within 2000 ms is suppressed", !second);
}

// ── Test 11: same height, different hashPrevBlock (fork) is NOT suppressed ───
static void test_dedup_fork_not_suppressed()
{
    std::cout << "\nTest 11: Same height, different hashPrevBlock (fork) is NOT suppressed\n";

    DedupGuard guard;
    uint1024_t prev_a{0};
    uint1024_t prev_b{1};  // different

    auto t0 = std::chrono::steady_clock::now();

    bool first = guard.should_pass(100, prev_a, t0);
    check("First (height=100, prev=A) passes", first);

    // Same height but different prev hash — genuine fork must NOT be suppressed
    auto t1 = t0 + std::chrono::milliseconds(200);
    bool fork = guard.should_pass(100, prev_b, t1);
    check("Fork (height=100, prev=B) is NOT suppressed", fork);
}

// ── Test 12: same pair after > 2000 ms is NOT suppressed ─────────────────────
static void test_dedup_debounce_expired()
{
    std::cout << "\nTest 12: Same pair after > 2000 ms is NOT suppressed\n";

    DedupGuard guard;
    uint1024_t prev{0};
    auto t0 = std::chrono::steady_clock::now();

    guard.should_pass(100, prev, t0);  // record first feed

    // 2001 ms later: debounce has expired
    auto t1 = t0 + std::chrono::milliseconds(2001);
    bool passed = guard.should_pass(100, prev, t1);
    check("Same pair after > 2000 ms passes through", passed);
}

// ── Test 13: concurrent recovery gate ─────────────────────────────────────────
static void test_concurrent_recovery_gate()
{
    std::cout << "\nTest 13: Concurrent recovery gate — only one executes at a time\n";

    std::mutex m_recovery_mutex;
    int concurrent_count   = 0;  // how many threads are inside the critical section simultaneously
    int max_concurrent     = 0;
    int completions        = 0;
    std::mutex count_mutex;

    auto recovery_path = [&]() {
        // Try to acquire the recovery mutex (non-blocking for the test gate variant)
        std::lock_guard<std::mutex> lock(m_recovery_mutex);

        {
            std::lock_guard<std::mutex> cl(count_mutex);
            ++concurrent_count;
            if (concurrent_count > max_concurrent)
                max_concurrent = concurrent_count;
        }

        // Simulate work
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        {
            std::lock_guard<std::mutex> cl(count_mutex);
            --concurrent_count;
            ++completions;
        }
    };

    std::thread t1(recovery_path);
    std::thread t2(recovery_path);
    t1.join();
    t2.join();

    check("No two threads execute recovery simultaneously (max_concurrent == 1)",
          max_concurrent == 1);
    check("Both recoveries eventually complete", completions == 2);
}

// ── main ──────────────────────────────────────────────────────────────────────
int main()
{
    std::cout << "==============================================\n";
    std::cout << "MinedBlockCache + Dedup Guard Unit Tests\n";
    std::cout << "==============================================\n";

    test_record_one_block();
    test_tier1_eviction_to_tier2();
    test_tier2_overflow_to_tier3();
    test_update_confirmations();
    test_confirmation_threshold_promotion();
    test_height_gated_confirmations();
    test_sole_record_stays_in_tier1();
    test_promotion_driven_tier2_overflow();
    test_full_promotion_lifecycle();
    test_tier1_newest_first();
    test_channel_name();
    test_status_emoji();
    test_summary_line();
    test_dedup_suppresses_same_pair_within_window();
    test_dedup_fork_not_suppressed();
    test_dedup_debounce_expired();
    test_concurrent_recovery_gate();

    std::cout << "\n==============================================\n";
    std::cout << "Test Summary\n";
    std::cout << "==============================================\n";
    std::cout << "Tests run:    " << g_run    << "\n";
    std::cout << "Tests passed: " << g_passed << "\n";
    std::cout << "Tests failed: " << g_failed << "\n";
    std::cout << "Success rate: "
              << (g_run > 0 ? (100 * g_passed / g_run) : 0) << "%\n";
    std::cout << "==============================================\n";

    return g_failed > 0 ? 1 : 0;
}
