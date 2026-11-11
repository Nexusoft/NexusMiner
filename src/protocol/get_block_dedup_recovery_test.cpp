/**
 * @file get_block_dedup_recovery_test.cpp
 * @brief Unit tests for GET_BLOCK deduplication and empty BLOCK_DATA recovery
 *
 * Tests:
 *  1. GET_BLOCK deduplication within 100ms window
 *  2. GET_BLOCK allowed after deduplication window expires
 *  3. Multiple rapid GET_BLOCK requests deduplicated
 *  4. GET_BLOCK deduplication across push handler and Worker_manager
 *  5. Deduplication state can be reset
 *  6. Three successive GET_BLOCK calls with proper timing
 *  7. Verify packet format for GET_BLOCK
 *  8. Degraded forced retry sends within bounded interval
 *  9. Dedup still allows periodic forced retry in degraded mode
 * 10. request_work empty schedules delayed retry (no starvation)
 * 11. ⚡ Unified Tip-Anchor Changed — dedup reset allows fresh GET_BLOCK despite recent prior request
 * 12. New recovery epoch does not inherit stale GET_BLOCK suppression state; anti-flood preserved within epoch
 * 13. Anti-flood preserved — true duplicates in same epoch/state still suppressed
 * 14. Height-based dedup bypassed when no valid template exists
 * 15. Cross-channel (Hash/Stake) block advancing unified but not channel height → dedup reset
 *     unblocks age-based GET_BLOCK retry
 * 16. Unified-only dedup allows cross-channel refresh
 * 17. GetBlockReason dedup bypass policy validation (three-tier: bypass_all, bypass_height, full)
 */

#include "protocol/packet_builder.hpp"
#include "protocol/get_block_reason.hpp"
#include "miner_opcodes.hpp"
#include <iostream>
#include <cassert>
#include <cstdint>
#include <chrono>
#include <thread>
#include <memory>
#include <deque>

using namespace nexusminer;
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
// Mock GET_BLOCK deduplication logic (mirrors Solo::get_work)
// ============================================================================
class GetBlockDeduplicator {
public:
    GetBlockDeduplicator()
        : m_authenticated(true)
        , m_reward_bound(true)
        , m_protocol_lane(ProtocolLane::STATELESS)
        , m_last_get_block_transmitted_tp{}
        , m_get_block_call_count(0)
    {}

    // Simulates Solo::get_work() payload generation without stamping dedup state.
    // The request becomes "sent" only after transmit_built_payload(true).
    network::Shared_payload build_work(GetBlockReason reason = GetBlockReason::INITIAL_REQUEST) {
        m_get_block_call_count++;

        if (!m_authenticated || !m_reward_bound) {
            return nullptr;
        }

        // GET_BLOCK rapid-burst guard (mirrors get_block_dedup_guard.hpp DEDUP_WINDOW_MS).
        // bypass_all reasons (RECOVERY_FORCED, RECOVERY_TIMER, BLOCK_ACCEPTED, PRIME_ORIGIN_TOO_LOW, HEALTH_NO_TEMPLATE) skip
        // this check entirely; bypass_height reasons still respect it.
        auto now_tp = std::chrono::steady_clock::now();
        if (!should_bypass_all_dedup(reason) &&
            m_last_get_block_transmitted_tp != std::chrono::steady_clock::time_point{}) {
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now_tp - m_last_get_block_transmitted_tp).count();
            if (elapsed_ms < 100) {  // GET_BLOCK_DEDUP_MS = 100
                std::cout << "    [Dedup] Suppressing duplicate GET_BLOCK ("
                         << elapsed_ms << "ms since last)\n";
                return nullptr;  // Suppress duplicate
            }
        }

        // Build GET_BLOCK packet
        return PacketBuilder::build(m_protocol_lane, nexusminer::LLP::GET_BLOCK);
    }

    bool transmit_built_payload(const network::Shared_payload& payload, bool transmit_ok = true) {
        if (!payload || payload->empty() || !transmit_ok) {
            return false;
        }
        m_last_get_block_transmitted_tp = std::chrono::steady_clock::now();
        std::cout << "    [Transmitted] GET_BLOCK sent successfully\n";
        return true;
    }

    // Convenience wrapper for the common build+successful-transmit path.
    network::Shared_payload get_work(GetBlockReason reason = GetBlockReason::INITIAL_REQUEST) {
        auto payload = build_work(reason);
        transmit_built_payload(payload, true);
        return payload;
    }

    void reset_timestamp() {
        m_last_get_block_transmitted_tp = {};
    }

    int get_call_count() const { return m_get_block_call_count; }
    void reset_call_count() { m_get_block_call_count = 0; }

    std::chrono::steady_clock::time_point get_last_transmitted_tp() const {
        return m_last_get_block_transmitted_tp;
    }

private:
    bool m_authenticated;
    bool m_reward_bound;
    ProtocolLane m_protocol_lane;
    std::chrono::steady_clock::time_point m_last_get_block_transmitted_tp;
    int m_get_block_call_count;
};

class DegradedForcedRetryController {
public:
    bool degraded{true};
    bool authenticated{true};
    bool no_valid_template{true};
    bool request_work_empty_once{false};
    int sent_count{0};
    int scheduled_retry_count{0};
    std::deque<std::chrono::steady_clock::time_point> forced_send_timestamps;
    std::chrono::steady_clock::time_point next_due{};
    GetBlockDeduplicator dedup;

    bool tick(std::chrono::steady_clock::time_point now) {
        if (!(degraded && authenticated && no_valid_template)) {
            return false;
        }
        if (next_due != std::chrono::steady_clock::time_point{} && now < next_due) {
            return false;
        }
        while (!forced_send_timestamps.empty()) {
            auto age_s = std::chrono::duration_cast<std::chrono::seconds>(now - forced_send_timestamps.front()).count();
            if (age_s <= 60) break;
            forced_send_timestamps.pop_front();
        }
        if (forced_send_timestamps.size() >= 25) {
            return false;
        }

        if (request_work_empty_once) {
            request_work_empty_once = false;
            schedule_retry(now);
            return false;
        }

        auto payload = dedup.get_work(GetBlockReason::RECOVERY_FORCED);
        if (payload && !payload->empty()) {
            ++sent_count;
            forced_send_timestamps.push_back(now);
            schedule_retry(now);
            return true;
        }
        schedule_retry(now);
        return false;
    }

private:
    void schedule_retry(std::chrono::steady_clock::time_point now) {
        ++scheduled_retry_count;
        next_due = now + std::chrono::milliseconds(1000 + 125);
    }
};

// ============================================================================
// Test 1: GET_BLOCK deduplication within 100ms window
// ============================================================================
void test_get_block_dedup_within_window() {
    std::cout << "\nTest 1: GET_BLOCK deduplication within 100ms window\n";

    GetBlockDeduplicator dedup;

    // First call should succeed
    auto payload1 = dedup.get_work();
    bool first_success = (payload1 != nullptr && !payload1->empty());

    // Second call within 100ms should be suppressed
    auto payload2 = dedup.get_work();
    bool second_suppressed = (payload2 == nullptr);

    // Third call within 100ms should also be suppressed
    auto payload3 = dedup.get_work();
    bool third_suppressed = (payload3 == nullptr);

    int call_count = dedup.get_call_count();
    bool correct_call_count = (call_count == 3);  // All 3 calls should be counted

    bool passed = first_success && second_suppressed && third_suppressed && correct_call_count;
    print_test_result("GET_BLOCK deduplication within 100ms", passed);
}

// ============================================================================
// Test 2: GET_BLOCK allowed after deduplication window expires
// ============================================================================
void test_get_block_after_window() {
    std::cout << "\nTest 2: GET_BLOCK allowed after deduplication window expires\n";

    GetBlockDeduplicator dedup;

    // First call should succeed
    auto payload1 = dedup.get_work();
    bool first_success = (payload1 != nullptr && !payload1->empty());

    // Wait for deduplication window to expire (110ms > 100ms)
    std::cout << "    [Wait] Sleeping for 110ms to expire dedup window...\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(110));

    // Second call after window should succeed
    auto payload2 = dedup.get_work();
    bool second_success = (payload2 != nullptr && !payload2->empty());

    bool passed = first_success && second_success;
    print_test_result("GET_BLOCK allowed after dedup window expires", passed);
}

// ============================================================================
// Test 3: Multiple rapid GET_BLOCK requests deduplicated
// ============================================================================
void test_multiple_rapid_requests() {
    std::cout << "\nTest 3: Multiple rapid GET_BLOCK requests deduplicated\n";

    GetBlockDeduplicator dedup;

    int successful_transmissions = 0;
    int suppressed_transmissions = 0;

    // First request should succeed
    auto payload = dedup.get_work();
    if (payload && !payload->empty()) {
        successful_transmissions++;
    }

    // Rapid-fire 9 more requests within 100ms
    for (int i = 0; i < 9; i++) {
        payload = dedup.get_work();
        if (payload && !payload->empty()) {
            successful_transmissions++;
        } else {
            suppressed_transmissions++;
        }
    }

    std::cout << "    [Stats] Successful: " << successful_transmissions
              << ", Suppressed: " << suppressed_transmissions << "\n";

    bool passed = (successful_transmissions == 1) && (suppressed_transmissions == 9);
    print_test_result("Multiple rapid requests deduplicated correctly", passed);
}

// ============================================================================
// Test 3b: Failed transmit must not stamp GET_BLOCK dedup state
// ============================================================================
void test_failed_transmit_does_not_stamp_dedup()
{
    std::cout << "\nTest 3b: failed transmit does not stamp GET_BLOCK dedup state\n";

    GetBlockDeduplicator dedup;
    auto payload = dedup.build_work(GetBlockReason::VALIDATION_FAILURE);
    bool built = (payload != nullptr && !payload->empty());
    bool transmit_failed = !dedup.transmit_built_payload(payload, false);
    bool unstamped = (dedup.get_last_transmitted_tp() == std::chrono::steady_clock::time_point{});

    auto retry_payload = dedup.build_work(GetBlockReason::VALIDATION_FAILURE);
    bool retry_allowed = (retry_payload != nullptr && !retry_payload->empty());

    print_test_result("Failed transmit leaves dedup timestamp unset",
                      built && transmit_failed && unstamped && retry_allowed);
}

// ============================================================================
// Test 3c: GET_ROUND handler should only mark "sent in handler" after queue success
// ============================================================================
void test_handler_send_flag_requires_successful_queue()
{
    std::cout << "\nTest 3c: GET_ROUND handler send flag requires successful queue\n";

    GetBlockDeduplicator dedup;
    bool sent_in_handler = false;

    auto payload = dedup.build_work(GetBlockReason::GET_ROUND_NO_TEMPLATE);
    bool first_built = (payload != nullptr && !payload->empty());
    if (first_built && dedup.transmit_built_payload(payload, false)) {
        sent_in_handler = true;
    }
    bool clear_after_failure = !sent_in_handler;

    auto retry_payload = dedup.build_work(GetBlockReason::GET_ROUND_NO_TEMPLATE);
    bool retry_built = (retry_payload != nullptr && !retry_payload->empty());
    bool retry_sent = dedup.transmit_built_payload(retry_payload, true);
    if (retry_sent) {
        sent_in_handler = true;
    }

    print_test_result("Failed queue leaves handler send flag clear for retry",
                      first_built && clear_after_failure && retry_built && retry_sent && sent_in_handler);
}

// ============================================================================
// Test 4: GET_BLOCK deduplication across push handler and Worker_manager
// ============================================================================
void test_dedup_across_callers() {
    std::cout << "\nTest 4: GET_BLOCK deduplication across callers\n";

    GetBlockDeduplicator dedup;

    // Simulate push handler call
    std::cout << "    [Push Handler] Requesting GET_BLOCK...\n";
    auto payload1 = dedup.get_work();
    bool push_success = (payload1 != nullptr && !payload1->empty());

    // Simulate Worker_manager call immediately after (within 100ms)
    std::cout << "    [Worker Manager] Requesting GET_BLOCK...\n";
    auto payload2 = dedup.get_work();
    bool worker_suppressed = (payload2 == nullptr);

    bool passed = push_success && worker_suppressed;
    print_test_result("Deduplication works across different callers", passed);
}

// ============================================================================
// Test 5: Deduplication resets after timestamp reset
// ============================================================================
void test_dedup_reset() {
    std::cout << "\nTest 5: Deduplication state can be reset\n";

    GetBlockDeduplicator dedup;

    // First call succeeds
    auto payload1 = dedup.get_work();
    bool first_success = (payload1 != nullptr && !payload1->empty());

    // Second call within window is suppressed
    auto payload2 = dedup.get_work();
    bool second_suppressed = (payload2 == nullptr);

    // Reset timestamp (simulating recovery completion)
    dedup.reset_timestamp();

    // Third call after reset should succeed
    auto payload3 = dedup.get_work();
    bool third_success = (payload3 != nullptr && !payload3->empty());

    bool passed = first_success && second_suppressed && third_success;
    print_test_result("Deduplication state resets correctly", passed);
}

// ============================================================================
// Test 6: Three successive GET_BLOCK calls with proper timing
// ============================================================================
void test_three_successive_calls() {
    std::cout << "\nTest 6: Three successive GET_BLOCK calls with proper timing\n";

    GetBlockDeduplicator dedup;

    // First call
    auto payload1 = dedup.get_work();
    bool first_success = (payload1 != nullptr && !payload1->empty());
    auto first_tp = dedup.get_last_transmitted_tp();

    // Wait 110ms
    std::this_thread::sleep_for(std::chrono::milliseconds(110));

    // Second call
    auto payload2 = dedup.get_work();
    bool second_success = (payload2 != nullptr && !payload2->empty());
    auto second_tp = dedup.get_last_transmitted_tp();

    // Wait 110ms
    std::this_thread::sleep_for(std::chrono::milliseconds(110));

    // Third call
    auto payload3 = dedup.get_work();
    bool third_success = (payload3 != nullptr && !payload3->empty());
    auto third_tp = dedup.get_last_transmitted_tp();

    // Verify timestamps are advancing
    bool timestamps_advance = (first_tp < second_tp) && (second_tp < third_tp);

    bool passed = first_success && second_success && third_success && timestamps_advance;
    print_test_result("Three successive calls with proper timing", passed);
}

// ============================================================================
// Test 7: Verify packet format
// ============================================================================
void test_packet_format() {
    std::cout << "\nTest 7: Verify GET_BLOCK packet format\n";

    GetBlockDeduplicator dedup;
    auto payload = dedup.get_work();

    bool valid_payload = (payload != nullptr && !payload->empty());

    // Stateless GET_BLOCK should be a 6-byte zero-length frame:
    // [0xD0][0x81][0x00][0x00][0x00][0x00]
    bool correct_size = valid_payload && (payload->size() == 6);
    bool correct_header = correct_size &&
                          ((*payload)[0] == 0xD0) &&
                          ((*payload)[1] == 0x81);
    bool correct_length = correct_size &&
                          ((*payload)[2] == 0x00) &&
                          ((*payload)[3] == 0x00) &&
                          ((*payload)[4] == 0x00) &&
                          ((*payload)[5] == 0x00);

    if (valid_payload) {
        std::cout << "    [Packet] Size: " << payload->size() << " bytes\n";
        std::cout << "    [Packet] Content: 0x";
        for (size_t i = 0; i < payload->size(); i++) {
            printf("%02X", (*payload)[i]);
        }
        std::cout << "\n";
    }

    bool passed = valid_payload && correct_size && correct_header && correct_length;
    print_test_result("GET_BLOCK packet format correct", passed);
}

// ============================================================================
// Test 8: Degraded forced retry sends within bounded interval
// ============================================================================
void test_degraded_forced_retry_sends_within_interval() {
    std::cout << "\nTest 8: Degraded forced retry sends within bounded interval\n";
    DegradedForcedRetryController controller;
    auto now = std::chrono::steady_clock::now();

    bool first_sent = controller.tick(now);
    print_test_result("Forced retry sends immediately on degraded tick", first_sent);

    bool second_sent_too_early = controller.tick(now + std::chrono::milliseconds(50));
    print_test_result("Second retry before interval is suppressed", !second_sent_too_early);

    bool third_sent = controller.tick(now + std::chrono::milliseconds(1200));
    print_test_result("Forced retry sends again after interval", third_sent);
}

// ============================================================================
// Test 9: Dedup window cannot starve degraded forced retry lane
// ============================================================================
void test_dedup_still_allows_periodic_forced_retry() {
    std::cout << "\nTest 9: Dedup still allows periodic forced retry in degraded mode\n";
    DegradedForcedRetryController controller;
    auto now = std::chrono::steady_clock::now();

    bool first_sent = controller.tick(now);
    bool immediate_retry = controller.tick(now + std::chrono::milliseconds(10));  // interval gate
    bool second_sent = controller.tick(now + std::chrono::milliseconds(1200));

    print_test_result("First forced send succeeds", first_sent);
    print_test_result("Immediate retry is suppressed by local interval", !immediate_retry);
    print_test_result("Periodic forced retry succeeds after interval despite dedup", second_sent);
}

// ============================================================================
// Test 10: request_work empty schedules delayed retry (no starvation)
// ============================================================================
void test_request_work_empty_delayed_retry_path() {
    std::cout << "\nTest 10: request_work empty triggers delayed retry path\n";
    DegradedForcedRetryController controller;
    controller.request_work_empty_once = true;
    auto now = std::chrono::steady_clock::now();

    bool first_sent = controller.tick(now);
    print_test_result("Initial empty work does not send", !first_sent);
    print_test_result("Empty work schedules retry token", controller.scheduled_retry_count == 1);

    bool retry_sent = controller.tick(now + std::chrono::milliseconds(1200));
    print_test_result("Delayed retry sends GET_BLOCK", retry_sent);
    print_test_result("No starvation after empty work transient", controller.sent_count >= 1);
}

// ============================================================================
// Test 11: Same-height tip-anchor change: dedup reset allows fresh GET_BLOCK
//
// Simulates the Unified Tip-Anchor Changed path: the canonical prev hash
// changes at the same channel height.  The old dedup timestamp refers to
// a request for the wrong canonical state and must not suppress the new one.
// ============================================================================
void test_tip_anchor_change_resets_dedup_allows_fresh_get_block() {
    std::cout << "\nTest 11: ⚡ Unified Tip-Anchor Changed — dedup reset allows fresh GET_BLOCK\n";

    GetBlockDeduplicator dedup;

    // First GET_BLOCK succeeds (current canonical tip-anchor, epoch 1)
    auto payload1 = dedup.get_work();
    bool first_success = (payload1 != nullptr && !payload1->empty());
    print_test_result("Initial GET_BLOCK for first tip-anchor succeeds", first_success);

    // Immediate second request within 100ms is suppressed (same epoch, no tip change)
    auto payload_dup = dedup.get_work();
    bool dup_suppressed = (payload_dup == nullptr);
    print_test_result("Immediate duplicate within same epoch is suppressed", dup_suppressed);

    // ⚡ Unified Tip-Anchor Changed — simulate recovery epoch reset:
    // reset_get_block_dedup_state() clears the dedup timestamp so the new canonical
    // tip-anchor's GET_BLOCK is not blocked by the old request's timestamp.
    dedup.reset_timestamp();  // mirrors Solo::reset_get_block_dedup_state()

    // Request for new canonical tip-anchor must succeed immediately despite
    // being within 100ms of the previous request.
    auto payload_recovery = dedup.get_work();
    bool recovery_allowed = (payload_recovery != nullptr && !payload_recovery->empty());
    print_test_result("GET_BLOCK after tip-anchor change succeeds (not suppressed)", recovery_allowed);
}

// ============================================================================
// Test 12: New recovery epoch does not inherit stale dedup suppression state
//
// Confirms that once the dedup state is reset at recovery epoch boundary,
// subsequent rapid-fire requests in the *new* epoch are correctly deduplicated
// again (anti-flood preserved) while the first request is allowed through.
// ============================================================================
void test_new_recovery_epoch_does_not_inherit_stale_dedup() {
    std::cout << "\nTest 12: New recovery epoch — anti-flood preserved within epoch\n";

    GetBlockDeduplicator dedup;
    auto now = std::chrono::steady_clock::now();

    // Epoch 1: first request succeeds
    auto p1 = dedup.get_work();
    bool epoch1_first_ok = (p1 != nullptr && !p1->empty());
    print_test_result("Epoch 1 first GET_BLOCK succeeds", epoch1_first_ok);

    // Epoch 1: second rapid request is suppressed
    auto p2 = dedup.get_work();
    bool epoch1_second_suppressed = (p2 == nullptr);
    print_test_result("Epoch 1 second rapid request suppressed (anti-flood)", epoch1_second_suppressed);

    // ⚡ Recovery epoch transition (Unified Tip-Anchor Changed or degraded recovery begin)
    dedup.reset_timestamp();  // mirrors Solo::reset_get_block_dedup_state()

    // Epoch 2: first request after reset succeeds (new canonical state)
    auto p3 = dedup.get_work();
    bool epoch2_first_ok = (p3 != nullptr && !p3->empty());
    print_test_result("Epoch 2 first GET_BLOCK after reset succeeds", epoch2_first_ok);

    // Epoch 2: second rapid request within same epoch is still suppressed (anti-flood preserved)
    auto p4 = dedup.get_work();
    bool epoch2_second_suppressed = (p4 == nullptr);
    print_test_result("Epoch 2 anti-flood still active — rapid duplicate suppressed", epoch2_second_suppressed);

    // Verify call count is as expected
    bool correct_count = (dedup.get_call_count() == 4);
    print_test_result("All 4 calls tracked regardless of suppression", correct_count);
}

// ============================================================================
// Test 13: Anti-flood preserved — true duplicates in same epoch still suppressed
//
// Verifies that the dedup reset mechanism does not disable flood protection:
// within a single recovery epoch, rapid duplicate requests are still suppressed.
// ============================================================================
void test_anti_flood_preserved_within_same_epoch() {
    std::cout << "\nTest 13: Anti-flood preserved for true duplicates (same epoch/state)\n";

    GetBlockDeduplicator dedup;

    // Send first request — should succeed
    auto p1 = dedup.get_work();
    bool first_ok = (p1 != nullptr && !p1->empty());

    // Fire 5 rapid duplicates — all must be suppressed
    int suppressed = 0;
    for (int i = 0; i < 5; i++) {
        auto p = dedup.get_work();
        if (!p || p->empty()) suppressed++;
    }

    bool flood_blocked = (suppressed == 5);
    print_test_result("First request in epoch succeeds", first_ok);
    print_test_result("5 rapid true duplicates are all suppressed", flood_blocked);

    // After dedup window, one more request should succeed
    std::this_thread::sleep_for(std::chrono::milliseconds(110));
    auto p_after = dedup.get_work();
    bool after_window_ok = (p_after != nullptr && !p_after->empty());
    print_test_result("After dedup window, next request succeeds again", after_window_ok);
}

// ============================================================================
// Test 14: Height-based dedup bypassed when no valid template exists
//
// Verifies that the height-based dedup guard does NOT suppress GET_BLOCK when
// no valid template is held, even if (unified, channel) heights are unchanged
// since the last transmission.  The guard must only suppress redundant
// refreshes of an already-valid template.
// ============================================================================

// Minimal mock of the height-based dedup condition added in solo.cpp,
// isolated from the time-based rapid-burst guard.
struct HeightDeduplicator {
    bool has_valid_template{false};
    uint32_t last_unified{0};
    uint32_t cur_unified{100};

    // Simulates reset_get_block_dedup_state() — clears all dedup tracking so
    // the next would_send() call is never suppressed regardless of heights.
    void reset() {
        last_unified = 0;
    }

    // Returns true when the GET_BLOCK would be transmitted (not suppressed).
    // Mirrors the solo.cpp height-based dedup condition exactly:
    //   suppress only when unified height unchanged AND a valid template exists.
    //   Channel height is intentionally excluded — hashPrevBlock changes on every
    //   unified-height advance regardless of which channel mined the block.
    bool would_send() {
        if (last_unified > 0 &&
            cur_unified == last_unified &&
            has_valid_template)
        {
            return false;  // suppressed — redundant refresh of valid template
        }
        last_unified = cur_unified;
        return true;
    }
};

void test_height_dedup_bypassed_when_no_valid_template() {
    std::cout << "\nTest 14: Height-based dedup bypassed when no valid template\n";

    HeightDeduplicator dedup;

    // First request: no prior height recorded, always passes.
    dedup.has_valid_template = false;
    bool first_ok = dedup.would_send();
    print_test_result("Initial GET_BLOCK succeeds (no prior height)", first_ok);
    // Unified height is now recorded as 100.

    // Same height, valid template present → guard fires, request suppressed.
    dedup.has_valid_template = true;
    bool suppressed_with_template = !dedup.would_send();
    print_test_result("Same unified height with valid template → suppressed", suppressed_with_template);

    // Same height, NO valid template → guard bypassed, request allowed.
    dedup.has_valid_template = false;
    bool allowed_no_template = dedup.would_send();
    print_test_result("Same unified height without valid template → allowed (bypass)", allowed_no_template);

    // Height advances, valid template present → height change unblocks guard.
    dedup.has_valid_template = true;
    dedup.cur_unified = 101;
    bool allowed_new_height = dedup.would_send();
    print_test_result("New unified height with valid template → allowed (height changed)", allowed_new_height);
}

// ============================================================================
// Test 15: Unified-only dedup allows cross-channel template refresh
//
// Simulates the scenario where a non-Prime block (Hash/Stake) advances the
// unified chain tip without changing Prime channel height.  With the old
// (unified, channel) pair guard, a Prime miner would be suppressed because
// channel height is unchanged.  With unified-only dedup, the GET_BLOCK is
// allowed whenever unified height advances — regardless of which channel mined.
// ============================================================================
void test_unified_only_dedup_allows_cross_channel_refresh() {
    std::cout << "\nTest 15: Cross-channel refresh — unified advances, channel stays same\n";

    HeightDeduplicator dedup;
    dedup.has_valid_template = true;

    // First GET_BLOCK at unified=100 (no prior state).
    bool first_ok = dedup.would_send();
    print_test_result("Initial GET_BLOCK at unified=100 succeeds", first_ok);

    // Same unified height → suppressed (valid template, no new block on any channel).
    bool same_height_suppressed = !dedup.would_send();
    print_test_result("Same unified=100 with valid template → suppressed", same_height_suppressed);

    // A Hash block arrives, advancing unified to 101 but Prime channel stays at 50.
    // The dedup must NOT suppress — hashPrevBlock has changed.
    dedup.cur_unified = 101;
    // (cur_channel would still be 50 in the old code — but it's no longer tracked here)
    bool cross_channel_allowed = dedup.would_send();
    print_test_result("Unified advances to 101 (Hash/Stake block) → allowed (hashPrevBlock changed)", cross_channel_allowed);

    // Now at unified=101, same height again → suppressed.
    bool suppressed_after_refresh = !dedup.would_send();
    print_test_result("Same unified=101 after refresh → suppressed again", suppressed_after_refresh);

    // Another non-Prime block: unified→102, Prime channel still same.
    dedup.cur_unified = 102;
    bool second_cross_channel = dedup.would_send();
    print_test_result("Unified advances to 102 (another cross-channel block) → allowed", second_cross_channel);
}

// ============================================================================
// Test 15: Cross-channel (Hash/Stake) block advancing unified but not channel
//           height → dedup reset unblocks age-based GET_BLOCK retry
//
// Scenario (Prime miner, Hash blocks advancing the chain every ~18s):
//   1. Prime block found → PUSH(unified=100, prime=50) → GET_BLOCK → m_last=100/50
//   2. Template received, valid
//   3. Hash block found → PUSH(unified=101, prime=50) arrives on Hash channel
//      → push_notification_handler detects cross-channel unified advance
//      → calls reset_dedup_fn() → m_last reset to 0/0
//   4. Template-age warning fires (480s+ old template) → retry GET_BLOCK
//      → cur_unified=100 (HeightTracker still at old value), m_last_unified=0
//      → Condition "m_last_unified > 0" is FALSE → NOT suppressed → GET_BLOCK sent ✓
//
// Without the cross-channel dedup reset in push_notification_handler.cpp the
// template-age retry would be stuck: both unified=100 and channel=50 match
// m_last_unified=100 / m_last_channel=50, so the guard fires and suppresses
// the request indefinitely — the miner can never refresh a stale template.
// ============================================================================
void test_cross_channel_unified_advance_resets_dedup() {
    std::cout << "\nTest 15: Cross-channel unified advance resets dedup, unblocking age-based retry\n";

    HeightDeduplicator dedup;

    // Initial GET_BLOCK after a Prime block (unified=100, prime_channel=50).
    dedup.cur_unified = 100;
    dedup.has_valid_template = false;
    bool first_ok = dedup.would_send();
    print_test_result("Initial GET_BLOCK succeeds (no prior heights recorded)", first_ok);

    // Template received → now valid.
    dedup.has_valid_template = true;

    // Template-age warning fires (heights unchanged) — should be suppressed.
    bool age_retry_suppressed = !dedup.would_send();
    print_test_result("Age-based retry suppressed at same heights with valid template", age_retry_suppressed);

    // Hash block found: unified advances to 101 but prime_channel stays at 50.
    // push_notification_handler detects unified advance (101 > HeightTracker=100)
    // and calls reset_dedup_fn().  HeightTracker still reports unified=100 because
    // cross-channel pushes do not update the tracker via update_height_fn.
    dedup.reset();  // Simulates reset_get_block_dedup_state()

    // Template-age warning fires again after dedup reset.
    // HeightTracker still reports unified=100 (old value), so cur_unified=100.
    // But m_last_unified=0 after reset → condition "m_last_unified > 0" is FALSE
    // → dedup does NOT fire → GET_BLOCK is sent.
    bool retry_after_reset_ok = dedup.would_send();
    print_test_result("Age-based retry succeeds after cross-channel dedup reset", retry_after_reset_ok);

    // GET_BLOCK records m_last=100/50 (HeightTracker still at 100).
    // Immediate second retry is suppressed (same heights, valid template).
    dedup.has_valid_template = true;
    bool immediate_retry_suppressed = !dedup.would_send();
    print_test_result("Immediate second retry suppressed (normal dedup protection)", immediate_retry_suppressed);

    // Another Hash block at unified=102: cross-channel push resets dedup again.
    // Next GET_BLOCK (from timer) should be unblocked.
    dedup.reset();
    bool third_retry_ok = dedup.would_send();
    print_test_result("Third retry allowed after second cross-channel dedup reset", third_retry_ok);
}

// ============================================================================
// Test: GetBlockReason dedup bypass policy validation
// ============================================================================
// Validates the three-tier dedup policy defined in get_block_reason.hpp:
//   1. bypass_all:    RECOVERY_FORCED, RECOVERY_TIMER, BLOCK_ACCEPTED, PRIME_ORIGIN_TOO_LOW, HEALTH_NO_TEMPLATE → skip everything
//   2. bypass_height: TEMPLATE_AGE_WARNING, VALIDATION_FAILURE, BLOCK_REJECTED, etc. → skip height guard
//   3. full dedup:    INITIAL_REQUEST, HEALTH_STALE_SUPPRESSED, etc. → all guards active
//   Note: HEALTH_CHANNEL_ADVANCE is in tier 2 (bypass_height)
//   because when the unified tip moves the template's hashPrevBlock is stale even
//   though the DedupGuard has the same unified height recorded.
//   HEALTH_TIP_MOVED removed — GET_ROUND handles tip changes.
//
// This is the core bug fix: TEMPLATE_AGE_WARNING must bypass height-based dedup
// so the 480s proactive refresh is not suppressed when heights are stagnant.
void test_get_block_reason_dedup_policy() {
    std::cout << "\nTest 17: GetBlockReason dedup bypass policy" << std::endl;

    // Tier 1: bypass_all — recovery retries skip everything
    print_test_result("RECOVERY_FORCED bypasses all dedup",
        should_bypass_all_dedup(GetBlockReason::RECOVERY_FORCED));
    print_test_result("RECOVERY_TIMER bypasses all dedup",
        should_bypass_all_dedup(GetBlockReason::RECOVERY_TIMER));
    print_test_result("RECOVERY_FORCED also bypasses height dedup",
        should_bypass_height_dedup(GetBlockReason::RECOVERY_FORCED));
    print_test_result("BLOCK_ACCEPTED bypasses all dedup (spent template replacement)",
        should_bypass_all_dedup(GetBlockReason::BLOCK_ACCEPTED));
    print_test_result("BLOCK_ACCEPTED also bypasses height dedup (implied by bypass_all)",
        should_bypass_height_dedup(GetBlockReason::BLOCK_ACCEPTED));
    print_test_result("PRIME_ORIGIN_TOO_LOW bypasses all dedup (dead-on-arrival template replacement)",
        should_bypass_all_dedup(GetBlockReason::PRIME_ORIGIN_TOO_LOW));
    print_test_result("PRIME_ORIGIN_TOO_LOW also bypasses height dedup (implied by bypass_all)",
        should_bypass_height_dedup(GetBlockReason::PRIME_ORIGIN_TOO_LOW));

    // HEALTH_NO_TEMPLATE: bypass_all so the health timer always gets through even
    // when a push-triggered GET_BLOCK fired within the last 100ms.
    print_test_result("HEALTH_NO_TEMPLATE bypasses all dedup (no template → burst guard counterproductive)",
        should_bypass_all_dedup(GetBlockReason::HEALTH_NO_TEMPLATE));
    print_test_result("HEALTH_NO_TEMPLATE also bypasses height dedup (implied by bypass_all)",
        should_bypass_height_dedup(GetBlockReason::HEALTH_NO_TEMPLATE));

    // Tier 2: bypass_height — age-based refresh and forced scenarios
    // THE KEY BUG FIX: TEMPLATE_AGE_WARNING bypasses height dedup
    print_test_result("TEMPLATE_AGE_WARNING bypasses height dedup (key bug fix)",
        should_bypass_height_dedup(GetBlockReason::TEMPLATE_AGE_WARNING));
    print_test_result("TEMPLATE_AGE_WARNING does NOT bypass all dedup (rapid-burst still active)",
        !should_bypass_all_dedup(GetBlockReason::TEMPLATE_AGE_WARNING));
    print_test_result("TEMPLATE_AGE_EMERGENCY bypasses height dedup",
        should_bypass_height_dedup(GetBlockReason::TEMPLATE_AGE_EMERGENCY));
    print_test_result("VALIDATION_FAILURE bypasses height dedup",
        should_bypass_height_dedup(GetBlockReason::VALIDATION_FAILURE));
    print_test_result("HEIGHT_DRIFT bypasses height dedup",
        should_bypass_height_dedup(GetBlockReason::HEIGHT_DRIFT));
    print_test_result("HEALTH_CHANNEL_STALE bypasses height dedup",
        should_bypass_height_dedup(GetBlockReason::HEALTH_CHANNEL_STALE));
    print_test_result("SESSION_REAUTH bypasses height dedup",
        should_bypass_height_dedup(GetBlockReason::SESSION_REAUTH));
    print_test_result("GET_ROUND_HEIGHT_PARITY bypasses height dedup",
        should_bypass_height_dedup(GetBlockReason::GET_ROUND_HEIGHT_PARITY));
    print_test_result("GET_ROUND_STALE bypasses height dedup",
        should_bypass_height_dedup(GetBlockReason::GET_ROUND_STALE));
    print_test_result("GET_ROUND_NO_TEMPLATE bypasses height dedup",
        should_bypass_height_dedup(GetBlockReason::GET_ROUND_NO_TEMPLATE));
    print_test_result("TEMPLATE_FEED_FAILURE bypasses height dedup",
        should_bypass_height_dedup(GetBlockReason::TEMPLATE_FEED_FAILURE));
    print_test_result("TEMPLATE_AGE_DEFERRED bypasses height dedup",
        should_bypass_height_dedup(GetBlockReason::TEMPLATE_AGE_DEFERRED));

    // BLOCK_REJECTED: node rejected our block; fresh template needed immediately.
    // Height-based guard bypassed; dedup state is reset before these calls so the
    // burst guard does not fire on the first post-rejection request.
    print_test_result("BLOCK_REJECTED bypasses height dedup",
        should_bypass_height_dedup(GetBlockReason::BLOCK_REJECTED));
    print_test_result("BLOCK_REJECTED does NOT bypass all dedup (burst guard still active on retry)",
        !should_bypass_all_dedup(GetBlockReason::BLOCK_REJECTED));

    // PUSH reasons: bypass height dedup (PUSH is authoritative) but NOT all dedup
    // (100ms rapid-burst guard still applies).
    // NOTE: PUSH_STALE, PUSH_NO_TEMPLATE, PUSH_CROSS_CHANNEL removed —
    //       NODE auto-sends BLOCK_DATA after PUSH, so no GET_BLOCK needed.
    print_test_result("PUSH_TIP_MOVED bypasses height dedup (authoritative push)",
        should_bypass_height_dedup(GetBlockReason::PUSH_TIP_MOVED));
    print_test_result("PUSH_SAME_HEIGHT_TIP bypasses height dedup (authoritative push)",
        should_bypass_height_dedup(GetBlockReason::PUSH_SAME_HEIGHT_TIP));

    // Tier 3: full dedup — non-push normal requests respect all guards
    // Note: HEALTH_CHANNEL_ADVANCE BYPASSES height dedup because
    // when the unified tip moves (e.g. Stake block on another channel), the current
    // template's hashPrevBlock becomes stale even though the DedupGuard already recorded
    // a GET_BLOCK at the same unified height.  Without the bypass, the height-match
    // guard suppresses the refresh and the miner gets stuck on a stale tip.
    // NOTE: HEALTH_TIP_MOVED removed — GET_ROUND is the backup for tip changes.
    print_test_result("HEALTH_CHANNEL_ADVANCE bypasses height dedup (template stale at same unified height)",
        should_bypass_height_dedup(GetBlockReason::HEALTH_CHANNEL_ADVANCE));
    print_test_result("HEALTH_CHANNEL_ADVANCE does NOT bypass all dedup (burst guard still active)",
        !should_bypass_all_dedup(GetBlockReason::HEALTH_CHANNEL_ADVANCE));
    print_test_result("HEALTH_STALE_SUPPRESSED does NOT bypass height dedup",
        !should_bypass_height_dedup(GetBlockReason::HEALTH_STALE_SUPPRESSED));
    print_test_result("INITIAL_REQUEST does NOT bypass height dedup",
        !should_bypass_height_dedup(GetBlockReason::INITIAL_REQUEST));

    // PUSH bypasses height but NOT all dedup (burst guard still applies)
    print_test_result("PUSH_TIP_MOVED does NOT bypass all dedup",
        !should_bypass_all_dedup(GetBlockReason::PUSH_TIP_MOVED));
    print_test_result("INITIAL_REQUEST does NOT bypass all dedup",
        !should_bypass_all_dedup(GetBlockReason::INITIAL_REQUEST));

    // reason_name() coverage — must not return "unknown" for any defined reason
    print_test_result("reason_name(TEMPLATE_AGE_WARNING) returns expected name",
        std::string(reason_name(GetBlockReason::TEMPLATE_AGE_WARNING)) == "template_age_warning");
    print_test_result("reason_name(RECOVERY_FORCED) returns expected name",
        std::string(reason_name(GetBlockReason::RECOVERY_FORCED)) == "recovery_forced");
    print_test_result("reason_name(GET_ROUND_HEIGHT_PARITY) returns expected name",
        std::string(reason_name(GetBlockReason::GET_ROUND_HEIGHT_PARITY)) == "get_round_height_parity");
    print_test_result("reason_name(BLOCK_REJECTED) returns expected name",
        std::string(reason_name(GetBlockReason::BLOCK_REJECTED)) == "block_rejected");
    print_test_result("reason_name(BLOCK_ACCEPTED) returns expected name",
        std::string(reason_name(GetBlockReason::BLOCK_ACCEPTED)) == "block_accepted");
    print_test_result("reason_name(PRIME_ORIGIN_TOO_LOW) returns expected name",
        std::string(reason_name(GetBlockReason::PRIME_ORIGIN_TOO_LOW)) == "prime_origin_too_low");
    print_test_result("reason_name(PUSH_TIP_MOVED) returns expected name",
        std::string(reason_name(GetBlockReason::PUSH_TIP_MOVED)) == "push_tip_moved");
}

// ============================================================================
// Test 18: Worker_manager recovery handler routes soft vs hard reasons correctly
// ============================================================================
namespace {
struct RecoveryHandlerHarness {
    enum class Phase : uint8_t { HEALTHY, WAITING_TEMPLATE };

    Phase phase{Phase::HEALTHY};
    uint64_t recovery_epoch{0};
    int mark_calls{0};
    int retry_calls{0};
    std::string marked_reason;

    void mark_recovery_initiated(const char* reason)
    {
        ++mark_calls;
        marked_reason = reason ? reason : "";
        if (phase == Phase::HEALTHY) {
            phase = Phase::WAITING_TEMPLATE;
            ++recovery_epoch;
        }
    }

    bool retry_template_request_returning(GetBlockReason)
    {
        ++retry_calls;
        return true;
    }

    bool invoke(GetBlockReason reason)
    {
        if (should_initiate_recovery_epoch(reason)) {
            mark_recovery_initiated(reason_name(reason));
        }
        return retry_template_request_returning(reason);
    }
};
} // namespace

void test_recovery_handler_reason_routing() {
    std::cout << "\nTest 18: Recovery handler reason-aware epoch routing" << std::endl;

    RecoveryHandlerHarness soft_feed;
    bool soft_feed_sent = soft_feed.invoke(GetBlockReason::TEMPLATE_FEED_FAILURE);
    print_test_result("TEMPLATE_FEED_FAILURE remains a soft retry", soft_feed_sent);
    print_test_result("TEMPLATE_FEED_FAILURE does not mark recovery initiated",
        soft_feed.mark_calls == 0);
    print_test_result("TEMPLATE_FEED_FAILURE leaves phase HEALTHY",
        soft_feed.phase == RecoveryHandlerHarness::Phase::HEALTHY);
    print_test_result("TEMPLATE_FEED_FAILURE leaves recovery epoch unchanged",
        soft_feed.recovery_epoch == 0);
    print_test_result("TEMPLATE_FEED_FAILURE still retries GET_BLOCK once",
        soft_feed.retry_calls == 1);

    RecoveryHandlerHarness soft_validation;
    bool soft_validation_sent = soft_validation.invoke(GetBlockReason::VALIDATION_FAILURE);
    print_test_result("VALIDATION_FAILURE remains a soft retry", soft_validation_sent);
    print_test_result("VALIDATION_FAILURE does not mark recovery initiated",
        soft_validation.mark_calls == 0);
    print_test_result("VALIDATION_FAILURE leaves phase HEALTHY",
        soft_validation.phase == RecoveryHandlerHarness::Phase::HEALTHY);
    print_test_result("VALIDATION_FAILURE leaves recovery epoch unchanged",
        soft_validation.recovery_epoch == 0);
    print_test_result("VALIDATION_FAILURE still retries GET_BLOCK once",
        soft_validation.retry_calls == 1);

    // PRIME_ORIGIN_TOO_LOW is covered here as a soft reason under the shared
    // should_initiate_recovery_epoch() soft/hard routing policy used by the
    // recovery/validation handlers (RecoveryHandlerHarness::invoke mirrors
    // that policy). This is not a full Worker_manager validation_failure_handler
    // integration test — it asserts the routing decision and side effects only.
    RecoveryHandlerHarness soft_prime_origin;
    bool soft_prime_origin_sent = soft_prime_origin.invoke(GetBlockReason::PRIME_ORIGIN_TOO_LOW);
    print_test_result("PRIME_ORIGIN_TOO_LOW soft/hard routing: remains a soft retry", soft_prime_origin_sent);
    print_test_result("PRIME_ORIGIN_TOO_LOW soft/hard routing: does not mark recovery initiated",
        soft_prime_origin.mark_calls == 0);
    print_test_result("PRIME_ORIGIN_TOO_LOW soft/hard routing: leaves phase HEALTHY",
        soft_prime_origin.phase == RecoveryHandlerHarness::Phase::HEALTHY);
    print_test_result("PRIME_ORIGIN_TOO_LOW soft/hard routing: leaves recovery epoch unchanged",
        soft_prime_origin.recovery_epoch == 0);
    print_test_result("PRIME_ORIGIN_TOO_LOW soft/hard routing: still retries GET_BLOCK once",
        soft_prime_origin.retry_calls == 1);

    RecoveryHandlerHarness hard_forced;
    bool hard_forced_sent = hard_forced.invoke(GetBlockReason::RECOVERY_FORCED);
    print_test_result("RECOVERY_FORCED schedules a retry", hard_forced_sent);
    print_test_result("RECOVERY_FORCED marks recovery initiated",
        hard_forced.mark_calls == 1);
    print_test_result("RECOVERY_FORCED advances to WAITING_TEMPLATE",
        hard_forced.phase == RecoveryHandlerHarness::Phase::WAITING_TEMPLATE);
    print_test_result("RECOVERY_FORCED advances recovery epoch",
        hard_forced.recovery_epoch == 1);
    print_test_result("RECOVERY_FORCED uses the reason-derived telemetry label",
        hard_forced.marked_reason == "recovery_forced");
    print_test_result("RECOVERY_FORCED still retries GET_BLOCK once",
        hard_forced.retry_calls == 1);

    RecoveryHandlerHarness hard_stale;
    hard_stale.invoke(GetBlockReason::GET_ROUND_STALE);
    print_test_result("GET_ROUND_STALE marks recovery initiated",
        hard_stale.mark_calls == 1);
    print_test_result("GET_ROUND_STALE uses the reason-derived telemetry label",
        hard_stale.marked_reason == "get_round_stale");
    print_test_result("GET_ROUND_STALE advances recovery epoch",
        hard_stale.recovery_epoch == 1);
    print_test_result("GET_ROUND_STALE retries GET_BLOCK once",
        hard_stale.retry_calls == 1);

    RecoveryHandlerHarness hard_no_template;
    hard_no_template.invoke(GetBlockReason::GET_ROUND_NO_TEMPLATE);
    print_test_result("GET_ROUND_NO_TEMPLATE marks recovery initiated",
        hard_no_template.mark_calls == 1);
    print_test_result("GET_ROUND_NO_TEMPLATE uses the reason-derived telemetry label",
        hard_no_template.marked_reason == "get_round_no_template");
    print_test_result("GET_ROUND_NO_TEMPLATE advances recovery epoch",
        hard_no_template.recovery_epoch == 1);
    print_test_result("GET_ROUND_NO_TEMPLATE retries GET_BLOCK once",
        hard_no_template.retry_calls == 1);
}

// ============================================================================
// Main Test Runner
// ============================================================================
int main() {
    std::cout << "\n═══════════════════════════════════════════════════════════\n";
    std::cout << "GET_BLOCK Deduplication Tests\n";
    std::cout << "═══════════════════════════════════════════════════════════\n";

    test_get_block_dedup_within_window();
    test_get_block_after_window();
    test_multiple_rapid_requests();
    test_failed_transmit_does_not_stamp_dedup();
    test_handler_send_flag_requires_successful_queue();
    test_dedup_across_callers();
    test_dedup_reset();
    test_three_successive_calls();
    test_packet_format();
    test_degraded_forced_retry_sends_within_interval();
    test_dedup_still_allows_periodic_forced_retry();
    test_request_work_empty_delayed_retry_path();
    test_tip_anchor_change_resets_dedup_allows_fresh_get_block();
    test_new_recovery_epoch_does_not_inherit_stale_dedup();
    test_anti_flood_preserved_within_same_epoch();
    test_height_dedup_bypassed_when_no_valid_template();
    test_cross_channel_unified_advance_resets_dedup();
    test_unified_only_dedup_allows_cross_channel_refresh();
    test_get_block_reason_dedup_policy();
    test_recovery_handler_reason_routing();

    std::cout << "\n═══════════════════════════════════════════════════════════\n";
    std::cout << "Test Results: " << tests_passed << "/" << tests_run << " passed";
    if (tests_failed > 0) {
        std::cout << " (" << tests_failed << " failed)";
    }
    std::cout << "\n═══════════════════════════════════════════════════════════\n\n";

    return (tests_failed == 0) ? 0 : 1;
}
