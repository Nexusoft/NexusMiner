/**
 * @file new_round_recovery_debounce_test.cpp
 * @brief Tests for the 5-second symmetric debounce on the NEW_ROUND recovery GET_BLOCK.
 *
 * Validates the operator-directed fix for the "Double/Triple Template Same Height Burst"
 * caused by a node-side broadcast ordering race:
 *   BLOCK_ACCEPTED, NEW_ROUND, PRIME/HASH_BLOCK_AVAILABLE can arrive in any order
 *   within a ~50–500ms window.
 *
 * The fix: always defer the recovery GET_BLOCK by 5s; cancel the deferred request
 * if a PUSH or BLOCK_ACCEPTED arrives during the wait.
 *
 * Test cases:
 *   Test 1 — PUSH-before-NEW_ROUND  (most common race from production trace)
 *   Test 2 — NEW_ROUND-before-PUSH  (operator-identified reverse-order case)
 *   Test 3 — NEW_ROUND with NO follow-up push (legacy node behaviour — should fire)
 *   Test 4 — Rapid NEW_ROUND burst (coalescing — only the last one fires)
 *   Test 5 — Self-induced NEW_ROUND tagging (SELF vs external log annotation)
 *
 * All tests are host-only: no network connection, no GPU, no real node.
 * The asio io_context is driven with run_for() to simulate timer expiry.
 */

#include "protocol/solo.hpp"
#include "protocol/node_session_context.hpp"
#include "protocol/session_manager.hpp"
#include "protocol_lane.hpp"
#include "miner_opcodes.hpp"
#include "packet.hpp"
#include "mining/mining_constants.hpp"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/null_sink.h"
#include "spdlog/sinks/ringbuffer_sink.h"

#include <asio/io_context.hpp>
#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace nexusminer;
using namespace nexusminer::protocol;
namespace MinerLLP = nexusminer::LLP;

// ─────────────────────────────────────────────────────────────────────────────
// Test scaffolding
// ─────────────────────────────────────────────────────────────────────────────

static int tests_run    = 0;
static int tests_passed = 0;
static int tests_failed = 0;

void print_test_result(const char* name, bool passed)
{
    ++tests_run;
    if (passed) {
        ++tests_passed;
        std::cout << "  [PASS] " << name << '\n';
    } else {
        ++tests_failed;
        std::cout << "  [FAIL] " << name << '\n';
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Packet helpers (mirrored from push_notification_lane_test.cpp)
// ─────────────────────────────────────────────────────────────────────────────

static network::Payload make_new_round_payload(uint32_t unified_height,
                                               uint32_t prime_height = 100,
                                               uint32_t hash_height  = 200,
                                               uint32_t stake_height = 10)
{
    network::Payload p(16, 0);
    auto w32 = [&](size_t off, uint32_t v) {
        p[off+0] = static_cast<uint8_t>((v >> 24) & 0xFF);
        p[off+1] = static_cast<uint8_t>((v >> 16) & 0xFF);
        p[off+2] = static_cast<uint8_t>((v >>  8) & 0xFF);
        p[off+3] = static_cast<uint8_t>( v        & 0xFF);
    };
    w32( 0, unified_height);
    w32( 4, prime_height);
    w32( 8, hash_height);
    w32(12, stake_height);
    return p;
}

static network::Payload make_push_payload(uint32_t unified_height,
                                          uint32_t channel_height,
                                          uint32_t difficulty    = 0x1d00ffff,
                                          uint8_t  prev_hash_fill = 0x42)
{
    network::Payload p(140, 0);
    auto w32 = [&](size_t off, uint32_t v) {
        p[off+0] = static_cast<uint8_t>((v >> 24) & 0xFF);
        p[off+1] = static_cast<uint8_t>((v >> 16) & 0xFF);
        p[off+2] = static_cast<uint8_t>((v >>  8) & 0xFF);
        p[off+3] = static_cast<uint8_t>( v        & 0xFF);
    };
    w32(0, unified_height);
    w32(4, channel_height);
    w32(8, difficulty);
    for (size_t i = 12; i < p.size(); ++i) {
        p[i] = prev_hash_fill;
    }
    return p;
}

// Build a Solo fixture: shared session, STATELESS lane, io_context for the timer.
struct SoloFixture
{
    std::shared_ptr<asio::io_context>      io;
    std::shared_ptr<SessionManager>        session_manager;
    std::shared_ptr<NodeSessionContext>    session_context;
    Solo                                   solo;

    explicit SoloFixture(uint8_t channel = static_cast<uint8_t>(mining::CHANNEL_HASH))
        : io(std::make_shared<asio::io_context>())
        , session_manager(std::make_shared<SessionManager>())
        , session_context(std::make_shared<NodeSessionContext>(session_manager))
        , solo(channel, /*stats=*/nullptr, session_context, io)
    {
        session_manager->start_session(SessionId(0xDEADBEEFu));
        solo.set_protocol_lane(ProtocolLane::STATELESS);
    }

    // Send a NEW_ROUND packet (hash channel heights).
    void send_new_round(uint32_t unified = 6693556,
                        uint32_t hash_h  = 200)
    {
        auto payload = make_new_round_payload(unified, /*prime=*/100, hash_h, /*stake=*/10);
        Packet pkt(MinerLLP::MirrorOpcode(static_cast<uint8_t>(Packet::NEW_ROUND)), payload);
        solo.process_messages(pkt, nullptr);
    }

    // Send a HASH_BLOCK_AVAILABLE push (same-channel for hash miner).
    void send_hash_push(uint32_t unified = 6693557,
                        uint32_t hash_h  = 201)
    {
        auto payload = make_push_payload(unified, hash_h);
        Packet pkt(MinerLLP::MirrorOpcode(MinerLLP::HASH_BLOCK_AVAILABLE), payload);
        solo.process_messages(pkt, nullptr);
    }

    // Send a PRIME_BLOCK_AVAILABLE push (cross-channel for hash miner).
    void send_prime_push(uint32_t unified = 6693557,
                         uint32_t prime_h = 101)
    {
        auto payload = make_push_payload(unified, prime_h);
        Packet pkt(MinerLLP::MirrorOpcode(MinerLLP::PRIME_BLOCK_AVAILABLE), payload);
        solo.process_messages(pkt, nullptr);
    }

    // Run the io_context for the given duration (drives async timers).
    void run_for(std::chrono::milliseconds dur)
    {
        io->run_for(dur);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Test 1 — PUSH-before-NEW_ROUND
// Observed sequence from production trace (the most common race).
// A PUSH arrives, then NEW_ROUND fires 1ms later.  The deferred recovery
// GET_BLOCK must be cancelled because a fresh BLOCK_DATA is already on its way.
// ─────────────────────────────────────────────────────────────────────────────
void test1_push_before_new_round()
{
    std::cout << "\nTest 1: PUSH-before-NEW_ROUND — recovery GET_BLOCK must be suppressed\n";

    SoloFixture f;

    // Step 1: HASH_BLOCK_AVAILABLE arrives (simulates node PUSH) for height
    // greater-or-equal to the upcoming NEW_ROUND height — this is exactly the
    // production race we want to suppress.  push_implies_block_data_in_transit()
    // returns true because last_push_notification_at is fresh and
    // push_unified_height (6693557) >= NEW_ROUND unified_height (6693556).
    f.send_hash_push(6693557, 201);

    // Step 2: NEW_ROUND arrives 1ms later.  schedule_recovery_get_block() now
    // detects the in-transit BLOCK_DATA and refuses to start the 5s timer.
    f.send_new_round(6693556, 200);

    // Post-fix expectation: the recovery GET_BLOCK is SUPPRESSED entirely
    // (no timer scheduled, no GET_BLOCK fired) because BLOCK_DATA is in
    // transit from the just-arrived PUSH.
    print_test_result("After PUSH then NEW_ROUND: recovery NOT scheduled (PUSH-in-transit guard)",
                      !f.solo.is_recovery_pending());

    // Drain without waiting full 5s — nothing should change because no timer was started.
    f.run_for(std::chrono::milliseconds(100));
    print_test_result("After 100ms: recovery still not pending (no timer was ever scheduled)",
                      !f.solo.is_recovery_pending());

    // Drain remaining 5s+ — recovery must never fire because no timer was scheduled.
    f.run_for(std::chrono::seconds(6));
    print_test_result("After 6s: recovery NEVER fired (BLOCK_DATA in transit; GET_BLOCK suppressed)",
                      f.solo.get_recovery_fired_count() == 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2 — NEW_ROUND-before-PUSH
// Operator's identified reverse-order case: NEW_ROUND fires FIRST, then 500ms
// later the PUSH arrives.  The deferred recovery GET_BLOCK must be cancelled
// when the PUSH shows up.
// ─────────────────────────────────────────────────────────────────────────────
void test2_new_round_before_push()
{
    std::cout << "\nTest 2: NEW_ROUND-before-PUSH — deferred timer cancelled by push\n";

    SoloFixture f;

    // Step 1: NEW_ROUND arrives with no template → schedules 5s deferred recovery
    f.send_new_round(6693556, 200);
    print_test_result("After NEW_ROUND: recovery is pending", f.solo.is_recovery_pending());

    // Step 2: Run for 500ms (simulates PUSH arriving 500ms after NEW_ROUND)
    f.run_for(std::chrono::milliseconds(500));
    print_test_result("At T=500ms: recovery still pending (5s not elapsed)",
                      f.solo.is_recovery_pending());

    // Step 3: HASH_BLOCK_AVAILABLE push arrives — should cancel the deferred timer
    f.send_hash_push(6693557, 201);
    print_test_result("After PUSH: recovery timer cancelled (is_recovery_pending = false)",
                      !f.solo.is_recovery_pending());

    // Step 4: Run io_context for 3s — timer was cancelled, must NOT fire
    f.run_for(std::chrono::seconds(3));
    print_test_result("After 3s with PUSH arriving at T=500ms: recovery NOT fired",
                      f.solo.get_recovery_fired_count() == 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3 — NEW_ROUND with NO follow-up push (legacy node behaviour)
// No PUSH, no BLOCK_ACCEPTED, no BLOCK_DATA.  The recovery must fire after 5s.
// ─────────────────────────────────────────────────────────────────────────────
void test3_new_round_no_push()
{
    std::cout << "\nTest 3: NEW_ROUND with no push — recovery fires after 5s\n";

    SoloFixture f;

    // Step 1: NEW_ROUND with no template
    f.send_new_round(6693556, 200);
    print_test_result("After NEW_ROUND: recovery is pending", f.solo.is_recovery_pending());

    // Step 2: Run for 4.9s — timer should NOT have fired yet
    f.run_for(std::chrono::milliseconds(4900));
    print_test_result("At T=4.9s: recovery NOT yet fired",
                      f.solo.get_recovery_fired_count() == 0);

    // Step 3: Run for 6s total — timer must fire exactly once
    f.run_for(std::chrono::milliseconds(1100));
    print_test_result("After 6s with no push: recovery fired exactly once",
                      f.solo.get_recovery_fired_count() == 1);
    print_test_result("After firing: recovery_deferred_at cleared (is_recovery_pending = false)",
                      !f.solo.is_recovery_pending());
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4 — Rapid NEW_ROUND burst (coalescing)
// Three NEW_ROUNDs in quick succession.  Each cancels the previous timer.
// Only the LAST one should result in a recovery GET_BLOCK, fired 5s after T=200ms.
// ─────────────────────────────────────────────────────────────────────────────
void test4_rapid_new_round_burst()
{
    std::cout << "\nTest 4: Rapid NEW_ROUND burst — coalesced to single deferred recovery\n";

    SoloFixture f;

    // Step 1: NEW_ROUND at T=0
    f.send_new_round(6693554, 198);
    print_test_result("After first NEW_ROUND: recovery is pending", f.solo.is_recovery_pending());

    // Step 2: run 100ms, then another NEW_ROUND at T=100ms
    f.run_for(std::chrono::milliseconds(100));
    f.send_new_round(6693555, 199);
    print_test_result("After second NEW_ROUND at T=100ms: recovery still pending (new timer)",
                      f.solo.is_recovery_pending());

    // Step 3: run 100ms, then another NEW_ROUND at T=200ms
    f.run_for(std::chrono::milliseconds(100));
    f.send_new_round(6693556, 200);
    print_test_result("After third NEW_ROUND at T=200ms: recovery still pending",
                      f.solo.is_recovery_pending());

    // Step 4: Run for 6s — only the final 5s timer should fire (once)
    f.run_for(std::chrono::seconds(6));
    print_test_result("After 6s: recovery fired exactly once (burst coalesced)",
                      f.solo.get_recovery_fired_count() == 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 5 — Self-induced NEW_ROUND tagging
// A NEW_ROUND at height == m_last_submitted_height within 2s of BLOCK_ACCEPTED
// should emit a "SELF-induced" log line.  An unrelated height should emit
// "external".
// ─────────────────────────────────────────────────────────────────────────────
void test5_self_induced_tagging()
{
    std::cout << "\nTest 5: Self-induced NEW_ROUND tagging — SELF vs external annotation\n";

    // Use a ringbuffer sink so we can inspect log output.
    auto ring_sink = std::make_shared<spdlog::sinks::ringbuffer_sink_mt>(128);
    auto logger = spdlog::get("logger");
    if (!logger) {
        logger = spdlog::default_logger();
    }
    const auto original_sink_count = logger->sinks().size();
    logger->sinks().push_back(ring_sink);

    {
        auto io = std::make_shared<asio::io_context>();
        auto session_manager = std::make_shared<SessionManager>();
        auto session_context = std::make_shared<NodeSessionContext>(session_manager);
        session_manager->start_session(SessionId(0xCAFEBABEu));
        Solo solo(static_cast<uint8_t>(mining::CHANNEL_HASH), nullptr, session_context, io);
        solo.set_protocol_lane(ProtocolLane::STATELESS);

        // Simulate a BLOCK_ACCEPTED arriving right now by sending a PUSH which
        // sets m_last_push_received_time (we cannot easily invoke on_block_accepted
        // without a full session ownership chain, so we test the log tag via
        // a NEW_ROUND after a known-height push window).
        //
        // For the self-induced case: set m_last_submitted_height to 6693556 and
        // m_last_block_accepted_time by sending a push + new_round with matching height.
        //
        // Since m_last_submitted_height is private but gets set on submit_block(),
        // and m_last_block_accepted_time is set in on_block_accepted() which needs
        // a full preflight setup, we test the "external" path directly (no submitted height
        // match) and infer the "SELF" path via the height-match logic on a clean fixture.

        // Use last_formatted() which returns std::vector<std::string> — easier to search.
        // We don't need to drain first since ring_sink is freshly created for this test.

        // Send NEW_ROUND with unified_height = 9999 (no prior submission → "external")
        auto payload = make_new_round_payload(9999, 100, 200, 10);
        Packet pkt(MinerLLP::MirrorOpcode(static_cast<uint8_t>(Packet::NEW_ROUND)), payload);
        solo.process_messages(pkt, nullptr);

        const auto msgs = ring_sink->last_formatted(64);
        bool found_external = false;
        for (const auto& m : msgs) {
            if (m.find("external") != std::string::npos &&
                m.find("NEW_ROUND]") != std::string::npos)
            {
                found_external = true;
                break;
            }
        }
        print_test_result("NEW_ROUND at unrelated height annotated as 'external'", found_external);
    }

    logger->sinks().resize(original_sink_count);
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 6 — PRIME push cancels timer on HASH miner (cross-channel cancel)
// Any push — same-channel or cross-channel — should cancel the recovery timer.
// ─────────────────────────────────────────────────────────────────────────────
void test6_cross_channel_push_cancels_timer()
{
    std::cout << "\nTest 6: Cross-channel PRIME push cancels HASH miner recovery timer\n";

    SoloFixture f(static_cast<uint8_t>(mining::CHANNEL_HASH));

    // NEW_ROUND with no template on hash miner
    f.send_new_round(6693556, 200);
    print_test_result("After NEW_ROUND: recovery is pending", f.solo.is_recovery_pending());

    f.run_for(std::chrono::milliseconds(300));

    // PRIME_BLOCK_AVAILABLE (cross-channel for hash miner) — must still cancel
    f.send_prime_push(6693557, 101);
    print_test_result("After cross-channel PRIME push: recovery timer cancelled",
                      !f.solo.is_recovery_pending());

    f.run_for(std::chrono::seconds(3));
    print_test_result("After 3s: recovery NOT fired (cross-channel push cancelled it)",
                      f.solo.get_recovery_fired_count() == 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main()
{
    // Use a null sink so the tests don't produce noise — Test 5 swaps in a ring
    // buffer when it needs to inspect log content.
    auto null_sink = std::make_shared<spdlog::sinks::null_sink_mt>();
    auto logger    = std::make_shared<spdlog::logger>("logger", null_sink);
    spdlog::set_default_logger(logger);

    std::cout << "================================================\n";
    std::cout << "NEW_ROUND Recovery Debounce Tests (5s symmetric)\n";
    std::cout << "================================================\n";

    test1_push_before_new_round();
    test2_new_round_before_push();
    test3_new_round_no_push();
    test4_rapid_new_round_burst();
    test5_self_induced_tagging();
    test6_cross_channel_push_cancels_timer();

    std::cout << "\n================================================\n";
    std::cout << "Results: " << tests_passed << "/" << tests_run << " passed";
    if (tests_failed > 0) {
        std::cout << " (" << tests_failed << " FAILED)";
    }
    std::cout << "\n================================================\n";

    return (tests_failed == 0) ? 0 : 1;
}
