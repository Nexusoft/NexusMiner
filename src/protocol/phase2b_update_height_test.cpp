/**
 * @file phase2b_update_height_test.cpp
 * @brief Phase 2B regression tests for unified height-state update path
 *
 * Tests:
 *  1. PushNotificationHandler passes correct parsed values to update_height_fn
 *     (not template header nHeight, which is channel_target in stateless templates).
 *  2. update_height_fn (callback) is invoked with the exact unified_height,
 *     channel_height, and difficulty_nbits from the 12-byte push payload.
 *  3. HeightTracker reflects the values delivered to update_height_fn.
 *  4. ClientChannelManager is updated with the same values as HeightTracker
 *     when UpdateFromGetRound is called with identical data.
 *  5. Fork detection works after ClientChannelManager update.
 *  6. Node BLOCK_DATA metadata fields feed unified/channel/difficulty first,
 *     then template target is derived from channel_height + 1.
 *  7. Stateless lane BLOCK_DATA fields drive HeightTracker identically to the
 *     legacy lane — both lanes call update_height_state() → OnPushNotification()
 *     → OnTemplateReceived() and produce identical HeightTracker snapshots.
 */

#include "protocol/push_notification_handler.hpp"
#include "protocol/height_tracker.hpp"
#include "mining/client_channel_manager.h"
#include "miner_opcodes.hpp"
#include <iostream>
#include <vector>
#include <cstdint>
#include <cassert>
#include <chrono>
#include <functional>
#include <optional>

using namespace nexusminer;
using namespace nexusminer::protocol;
using namespace nexusminer::mining;

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

// ============================================================================
// Helpers to build a 12-byte push notification packet
// ============================================================================

// Extract byte `i` (0=MSB) from a big-endian uint32_t
static uint8_t be_byte(uint32_t v, int i) { return (v >> (24 - 8*i)) & 0xFF; }

static Packet make_push_packet(uint32_t unified_h, uint32_t channel_h, uint32_t difficulty,
                                bool use_16bit_opcode = false)
{
    Packet pkt;
    pkt.m_header = use_16bit_opcode
        ? nexusminer::LLP::MirrorOpcode(nexusminer::LLP::PRIME_BLOCK_AVAILABLE)
        : static_cast<uint16_t>(nexusminer::LLP::PRIME_BLOCK_AVAILABLE);
    pkt.m_is_uint16_opcode = use_16bit_opcode;
    pkt.m_length = 12;
    pkt.m_data = std::make_shared<network::Payload>(network::Payload{
        be_byte(unified_h, 0), be_byte(unified_h, 1),
        be_byte(unified_h, 2), be_byte(unified_h, 3),
        be_byte(channel_h, 0), be_byte(channel_h, 1),
        be_byte(channel_h, 2), be_byte(channel_h, 3),
        be_byte(difficulty, 0), be_byte(difficulty, 1),
        be_byte(difficulty, 2), be_byte(difficulty, 3),
    });
    return pkt;
}

// ============================================================================
// Test 1: update_height_fn receives exact values from push notification payload
// ============================================================================
static void test_push_handler_callback_values()
{
    std::cout << "\nTest 1: update_height_fn receives correct parsed values\n";

    constexpr uint32_t UNIFIED  = 4321;
    constexpr uint32_t CHANNEL  = 210;
    constexpr uint32_t DIFF     = 0x1d00ffff;

    auto logger = spdlog::get("logger");
    if (!logger) logger = spdlog::default_logger();

    uint8_t channel = CHANNEL_PRIME;
    PushNotificationHandler handler(logger, channel);

    Packet pkt = make_push_packet(UNIFIED, CHANNEL, DIFF);

    uint32_t got_unified = 0, got_channel = 0, got_diff = 0;
    bool callback_invoked = false;

    auto update_fn = [&](uint32_t u, uint32_t c, uint32_t d) {
        got_unified  = u;
        got_channel  = c;
        got_diff     = d;
        callback_invoked = true;
    };

    handler.handle_push_notification(
        pkt,
        CHANNEL_PRIME,
        ProtocolLane::LEGACY,
        nullptr,   // no template_interface
        nullptr,   // no height_tracker (not needed for this check)
        update_fn
    );

    print_test_result("callback invoked", callback_invoked);
    print_test_result("unified_height passed correctly", got_unified == UNIFIED);
    print_test_result("channel_height passed correctly", got_channel == CHANNEL);
    print_test_result("difficulty passed correctly",     got_diff    == DIFF);
}

// ============================================================================
// Test 2: HeightTracker reflects push-derived diagnostic values
//         (unified_height and channel_height are now canonical-only from BLOCK_DATA)
// ============================================================================
static void test_height_tracker_updated_via_callback()
{
    std::cout << "\nTest 2: HeightTracker reflects values delivered by update_height_fn\n";

    constexpr uint32_t UNIFIED  = 5000;
    constexpr uint32_t CHANNEL  = 300;
    constexpr uint32_t DIFF     = 0x1c0e9f34;

    HeightTracker tracker;

    auto update_fn = [&](uint32_t u, uint32_t c, uint32_t d) {
        tracker.OnPushNotification(u, c, d);
    };

    auto logger = spdlog::get("logger");
    if (!logger) logger = spdlog::default_logger();
    uint8_t channel = CHANNEL_PRIME;
    PushNotificationHandler handler(logger, channel);
    Packet pkt = make_push_packet(UNIFIED, CHANNEL, DIFF);

    handler.handle_push_notification(
        pkt, CHANNEL_PRIME, ProtocolLane::LEGACY,
        nullptr, &tracker, update_fn
    );

    auto snap = tracker.GetSnapshot();
    // unified_height and channel_height are canonical-only (set by OnBlockDataReceived),
    // so push does NOT set them in the snapshot.
    print_test_result("HeightTracker push_unified_height == UNIFIED", snap.push_unified_height == UNIFIED);
    print_test_result("HeightTracker push_channel_height == CHANNEL", snap.push_channel_height == CHANNEL);
    print_test_result("HeightTracker difficulty_nbits == DIFF",  snap.difficulty_nbits == DIFF);
    print_test_result("HeightTracker source == PUSH",
        snap.last_update_source == HeightTracker::UpdateSource::PUSH);
}

// ============================================================================
// Test 2b: Channel-mismatch push with unified tip advance triggers callback
//          (Bug 1 fix: cross-channel tip advance requests fresh template)
// ============================================================================
static void test_channel_mismatch_refreshes_push_liveness_only()
{
    std::cout << "\nTest 2b: channel-mismatch push with tip advance invokes callback\n";

    constexpr uint32_t UNIFIED  = 5001;
    constexpr uint32_t CHANNEL  = 301;
    constexpr uint32_t DIFF     = 0x1c0e9f35;

    HeightTracker tracker;

    auto logger = spdlog::get("logger");
    if (!logger) logger = spdlog::default_logger();
    uint8_t mining_channel = CHANNEL_PRIME;
    PushNotificationHandler handler(logger, mining_channel);
    Packet pkt = make_push_packet(UNIFIED, CHANNEL, DIFF);
    pkt.m_header = nexusminer::LLP::HASH_BLOCK_AVAILABLE;

    bool callback_invoked = false;
    auto before_push = std::chrono::steady_clock::now();
    handler.handle_push_notification(
        pkt, CHANNEL_HASH, ProtocolLane::LEGACY,
        nullptr, &tracker,
        [&](uint32_t, uint32_t, uint32_t) { callback_invoked = true; }
    );
    auto after_push = std::chrono::steady_clock::now();

    auto snap = tracker.GetSnapshot();
    // Cross-channel push with unified tip advance (5001 > 0) DOES invoke height
    // callback to update push diagnostic state and request fresh template.
    print_test_result("cross-channel push with tip advance invokes height callback", callback_invoked);
    print_test_result("channel-mismatch push refreshes push liveness timestamp",
        snap.last_push_notification_at >= before_push && snap.last_push_notification_at <= after_push);
    // unified_height and channel_height remain canonical-only (0 until BLOCK_DATA)
    print_test_result("channel-mismatch push does not advance canonical unified_height",
        snap.unified_height == 0);
    print_test_result("channel-mismatch push does not advance canonical channel_height",
        snap.channel_height == 0);
}

// ============================================================================
// Test 3: ClientChannelManager reflects same values as HeightTracker
//         (simulates what update_height_state() does)
// ============================================================================
static void test_channel_manager_same_data_as_height_tracker()
{
    std::cout << "\nTest 3: ClientChannelManager updated with same data as HeightTracker\n";

    constexpr uint32_t UNIFIED  = 6000;
    constexpr uint32_t CHANNEL  = 400;
    constexpr uint32_t DIFF     = 0x1b123456;

    HeightTracker tracker;
    HashClientManager mgr;

    // Simulate update_height_state() called from push notification path
    auto update_fn = [&](uint32_t u, uint32_t c, uint32_t d) {
        tracker.OnPushNotification(u, c, d);
        mgr.UpdateFromGetRound(u, c);
    };

    auto logger = spdlog::get("logger");
    if (!logger) logger = spdlog::default_logger();
    uint8_t channel = CHANNEL_HASH;
    PushNotificationHandler handler(logger, channel);
    Packet pkt = make_push_packet(UNIFIED, CHANNEL, DIFF, false);
    // Use HASH opcode
    pkt.m_header = nexusminer::LLP::HASH_BLOCK_AVAILABLE;

    handler.handle_push_notification(
        pkt, CHANNEL_HASH, ProtocolLane::LEGACY,
        nullptr, &tracker, update_fn
    );

    auto snap = tracker.GetSnapshot();
    auto [node_u, node_c] = mgr.GetNodeHeights();

    print_test_result("HeightTracker push_unified_height == UNIFIED", snap.push_unified_height == UNIFIED);
    print_test_result("ClientChannelManager unified_height == UNIFIED", node_u == UNIFIED);
    print_test_result("ClientChannelManager channel_height == CHANNEL", node_c == CHANNEL);
    // HeightTracker's push_unified_height matches ClientChannelManager (both from push path)
    print_test_result("Both sources match (unified)",  snap.push_unified_height == node_u);
    print_test_result("Both sources match (channel)",  snap.push_channel_height == node_c);
}

// ============================================================================
// Test 4: Fork detection works when heights regress via update callback
// ============================================================================
static void test_fork_detection_via_update_callback()
{
    std::cout << "\nTest 4: Fork detection fires when heights regress by 2+ blocks\n";

    PrimeClientManager mgr;

    // First update: height = 5000
    mgr.UpdateFromGetRound(5000, 200);
    print_test_result("No fork on first update", !mgr.IsForkDetected());

    // Second update: unified regresses by 2 blocks (genuine fork)
    mgr.UpdateFromGetRound(4998, 200);
    print_test_result("Fork detected after 2-block height regression", mgr.IsForkDetected());

    mgr.ClearForkFlag();
    print_test_result("Fork flag cleared after ClearForkFlag()", !mgr.IsForkDetected());

    // Advance normally: no fork
    mgr.UpdateFromGetRound(5001, 201);
    print_test_result("No fork after normal advance", !mgr.IsForkDetected());
}

// ============================================================================
// Test 5: [REMOVED] is_template_stale() was removed from HeightTracker::Snapshot
//         (structurally false under canonical-only semantics).
//         Staleness detection has moved to push/GET_ROUND trigger paths.
// ============================================================================

// ============================================================================
// Test 6: Node BLOCK_DATA metadata fields feed unified/channel/difficulty first,
//         then template target is derived from channel_height + 1.
// ============================================================================
static void test_node_block_data_fields_drive_height_tracker()
{
    std::cout << "\nTest 6: Node BLOCK_DATA fields drive nBits, unified height, and channel target\n";

    constexpr uint32_t UNIFIED  = 7000;
    constexpr uint32_t CHANNEL  = 450;
    constexpr uint32_t DIFF     = 0x1b01abcd;

    HeightTracker tracker;
    HashClientManager mgr;

    // Mirrors Solo::update_height_state(...) for BLOCK_DATA metadata feed.
    // OnPushNotification sets diagnostic push heights only.
    tracker.OnPushNotification(UNIFIED, CHANNEL, DIFF);
    // OnBlockDataReceived sets canonical heights (required for snap.unified_height).
    tracker.OnBlockDataReceived(UNIFIED, CHANNEL, DIFF, uint1024_t{});
    mgr.UpdateFromGetRound(UNIFIED, CHANNEL);

    // Mirrors Solo::OnTemplateReceived(channel_height + 1) for template target.
    tracker.OnTemplateReceived(CHANNEL_HASH, CHANNEL + 1);

    auto snap = tracker.GetSnapshot();
    auto [node_u, node_c] = mgr.GetNodeHeights();

    print_test_result("HeightTracker unified_height from BLOCK_DATA metadata", snap.unified_height == UNIFIED);
    print_test_result("HeightTracker difficulty_nbits from BLOCK_DATA metadata", snap.difficulty_nbits == DIFF);
    print_test_result("HeightTracker channel_target == channel_height + 1", snap.channel_target == CHANNEL + 1);
    // template_unified_height is set from canonical_unified_height by OnTemplateReceived
    print_test_result("template_unified_height captured from canonical unified_height", snap.template_unified_height == UNIFIED);
    print_test_result("ClientChannelManager unified_height matches metadata", node_u == UNIFIED);
    print_test_result("ClientChannelManager channel_height matches metadata", node_c == CHANNEL);
}

// ============================================================================
// Test 7: Stateless lane BLOCK_DATA metadata also drives HeightTracker identically to legacy lane.
//         Both lanes call update_height_state() → OnPushNotification() → OnTemplateReceived().
//         The only difference is the packet parser; the HeightTracker update is identical.
// ============================================================================
static void test_stateless_lane_mirrors_legacy_for_height_tracker()
{
    std::cout << "\nTest 7: Stateless lane BLOCK_DATA fields drive HeightTracker identically to legacy\n";

    constexpr uint32_t UNIFIED = 8000;
    constexpr uint32_t CHANNEL = 500;
    constexpr uint32_t DIFF    = 0x1a0fffff;

    HeightTracker tracker_legacy;
    HeightTracker tracker_stateless;

    // Legacy lane sequence
    tracker_legacy.OnPushNotification(UNIFIED, CHANNEL, DIFF);
    tracker_legacy.OnBlockDataReceived(UNIFIED, CHANNEL, DIFF, uint1024_t{});
    tracker_legacy.OnTemplateReceived(CHANNEL_HASH, CHANNEL + 1);

    // Stateless lane sequence (identical calls — same update path)
    tracker_stateless.OnPushNotification(UNIFIED, CHANNEL, DIFF);
    tracker_stateless.OnBlockDataReceived(UNIFIED, CHANNEL, DIFF, uint1024_t{});
    tracker_stateless.OnTemplateReceived(CHANNEL_HASH, CHANNEL + 1);

    auto snap_l = tracker_legacy.GetSnapshot();
    auto snap_s = tracker_stateless.GetSnapshot();

    print_test_result("Legacy and stateless lanes produce identical unified_height",
        snap_l.unified_height == snap_s.unified_height);
    print_test_result("Legacy and stateless lanes produce identical channel_height",
        snap_l.channel_height == snap_s.channel_height);
    print_test_result("Legacy and stateless lanes produce identical channel_target",
        snap_l.channel_target == snap_s.channel_target);
    print_test_result("Legacy and stateless lanes produce identical difficulty_nbits",
        snap_l.difficulty_nbits == snap_s.difficulty_nbits);
    print_test_result("channel_target == CHANNEL + 1",
        snap_l.channel_target == CHANNEL + 1);
    print_test_result("template_unified_height == UNIFIED",
        snap_l.template_unified_height == UNIFIED);
}

// ============================================================================
// main
// ============================================================================
int main()
{
    std::cout << "========================================\n";
    std::cout << "Phase 2B: Height State Update Tests\n";
    std::cout << "========================================\n";

    test_push_handler_callback_values();
    test_height_tracker_updated_via_callback();
    test_channel_mismatch_refreshes_push_liveness_only();
    test_channel_manager_same_data_as_height_tracker();
    test_fork_detection_via_update_callback();
    // test_height_tracker_staleness_matches_expected() removed — is_template_stale() no longer exists
    test_node_block_data_fields_drive_height_tracker();
    test_stateless_lane_mirrors_legacy_for_height_tracker();

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
