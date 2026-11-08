/**
 * @file same_tip_reconfirmation_test.cpp
 * @brief Regression test for the "same-tip re-confirmation should clear an in-flight
 *        recovery" fix in Solo::finalize_and_feed_current_template().
 *
 * Reproduces the trace reported by an operator: a worker's found-block submission
 * races a chain-tip change, is discarded, and recovery (WAITING_TEMPLATE) is
 * initiated. The node then replies to the recovery GET_BLOCK with BLOCK_DATA for
 * the SAME (unified height, hashPrevBlock) that is already loaded — i.e. the node
 * simply hadn't produced a new tip yet. The unified-height feed guard correctly
 * suppresses re-distributing this unchanged template to workers (to avoid a
 * redundant worker restart), but prior to this fix that also meant the recovery
 * state machine's only clearing path (the worker feed-handler chokepoint) never
 * ran, leaving recovery parked indefinitely until a genuinely new tip arrived or
 * the controlled-recovery hard-stop escalated it into DEGRADED_MODE.
 *
 * The fix adds a narrow Recovery_confirmed_handler, fired specifically when the
 * same-height feed guard suppresses a re-feed, so Worker_manager can clear
 * recovery on this proof-of-liveness signal without requiring a worker re-feed.
 *
 * Tests:
 *  1. First feed of a template invokes the block (worker feed) handler exactly
 *     once and does NOT fire the recovery-confirmed handler.
 *  2. A second BLOCK_DATA for the identical (height, hashPrevBlock) within the
 *     same-height cooldown does NOT re-invoke the block handler (pre-existing
 *     guard behavior, unchanged) but DOES fire the recovery-confirmed handler
 *     exactly once (the new behavior under test).
 *  3. A subsequent BLOCK_DATA for a genuinely new tip (different hashPrevBlock)
 *     bypasses the guard entirely: the block handler fires again and the
 *     recovery-confirmed handler is NOT fired for that call.
 */

#include "protocol/solo.hpp"
#include "network/connection.hpp"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/null_sink.h"

#include <asio/io_context.hpp>
#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>
#include <vector>

using namespace nexusminer;
using namespace nexusminer::protocol;

namespace {

static int tests_run = 0;
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

// Builds a 216-byte Tritium Block::Serialize() payload (matches the format consumed
// by MiningTemplateInterface::read_template() — same layout used by the existing
// mining_template_validation_test.cpp create_mock_template() helper), with a
// caller-controlled hashPrevBlock pattern so tests can simulate "same tip" vs.
// "new tip" BLOCK_DATA responses.
std::vector<uint8_t> make_block_payload(uint32_t height, uint8_t hash_prev_pattern,
                                         uint32_t nBits = 0x03a4166d, uint8_t channel = 1)
{
    std::vector<uint8_t> data(216, 0);
    size_t offset = 0;

    auto write_u32_be = [&](uint32_t value) {
        data[offset++] = (value >> 24) & 0xFF;
        data[offset++] = (value >> 16) & 0xFF;
        data[offset++] = (value >> 8) & 0xFF;
        data[offset++] = value & 0xFF;
    };

    write_u32_be(8);  // nVersion

    // hashPrevBlock (128 bytes) — fill with a distinctive repeating pattern so
    // two payloads only match when explicitly given the same pattern byte.
    for (int i = 0; i < 128; ++i) {
        data[offset++] = hash_prev_pattern;
    }

    // hashMerkleRoot (64 bytes) — non-zero pattern so merkle-root validation passes.
    for (int i = 0; i < 64; ++i) {
        data[offset++] = static_cast<uint8_t>(i + 1);
    }

    write_u32_be(static_cast<uint32_t>(channel));  // nChannel
    write_u32_be(height);                          // nHeight (target)
    write_u32_be(nBits);                           // nBits

    // nNonce (8 bytes, little-endian per node wire format) — zero.
    for (int i = 0; i < 8; ++i) {
        data[offset++] = 0;
    }

    return data;
}

} // namespace

namespace nexusminer {
namespace protocol {

// Friend harness granting the test access to Solo's private
// finalize_and_feed_current_template() chokepoint (same pattern as
// PostAdoptionSuppressionHarness in post_adoption_suppression_test.cpp).
struct SameTipReconfirmationHarness {
    static bool finalize(Solo& solo, uint32_t unified_height, uint32_t effective_channel_height,
                         const char* log_scope)
    {
        return solo.finalize_and_feed_current_template(unified_height, effective_channel_height,
                                                        log_scope, /*snapshot_round_channel_height=*/false);
    }
};

} // namespace protocol
} // namespace nexusminer

namespace {

std::unique_ptr<Solo> make_solo()
{
    auto io = std::make_shared<asio::io_context>();
    // Session context/stats are intentionally null: this test targets the
    // finalize_and_feed_current_template() chokepoint in isolation, matching
    // the construction pattern used by post_adoption_suppression_test.cpp.
    return std::make_unique<Solo>(static_cast<uint8_t>(1) /* CHANNEL_PRIME */, nullptr, nullptr, io);
}

void test_first_feed_invokes_block_handler_only()
{
    std::cout << "\nTest 1: First feed invokes block handler, not recovery-confirmed handler\n";
    auto solo = make_solo();

    int block_handler_calls = 0;
    int recovery_confirmed_calls = 0;
    solo->set_block_handler([&](::LLP::CBlock, std::uint32_t) { ++block_handler_calls; return true; });
    solo->set_recovery_confirmed_handler([&]() { ++recovery_confirmed_calls; });

    auto payload = make_block_payload(6771846, 0xAA);
    auto result = solo->get_template_interface()->read_template(payload, "test_node", false);
    print_test_result("read_template() accepts first template", result.is_valid);

    bool fed = SameTipReconfirmationHarness::finalize(*solo, 6771846, 0, "test");
    print_test_result("finalize_and_feed_current_template() succeeds on first feed", fed);
    print_test_result("Block handler invoked exactly once", block_handler_calls == 1);
    print_test_result("Recovery-confirmed handler NOT invoked on first feed", recovery_confirmed_calls == 0);
}

void test_same_tip_reconfirmation_fires_recovery_confirmed_not_block_handler()
{
    std::cout << "\nTest 2: Same-tip re-confirmation within cooldown suppresses re-feed "
                 "but fires recovery-confirmed handler\n";
    auto solo = make_solo();

    int block_handler_calls = 0;
    int recovery_confirmed_calls = 0;
    solo->set_block_handler([&](::LLP::CBlock, std::uint32_t) { ++block_handler_calls; return true; });
    solo->set_recovery_confirmed_handler([&]() { ++recovery_confirmed_calls; });

    // Initial feed establishes m_last_fed_unified_height / m_last_fed_hash_prev_block.
    auto payload1 = make_block_payload(6771846, 0xAA);
    solo->get_template_interface()->read_template(payload1, "test_node", false);
    SameTipReconfirmationHarness::finalize(*solo, 6771846, 0, "test");
    print_test_result("Sanity: block handler called once after initial feed", block_handler_calls == 1);

    // Node re-serves BLOCK_DATA for the SAME (height, hashPrevBlock) — simulating the
    // reported trace where a recovery GET_BLOCK is answered with the still-current tip.
    auto payload2 = make_block_payload(6771846, 0xAA);
    auto result2 = solo->get_template_interface()->read_template(payload2, "test_node", false);
    print_test_result("read_template() accepts re-served identical template", result2.is_valid);

    bool fed2 = SameTipReconfirmationHarness::finalize(*solo, 6771846, 0, "test");
    print_test_result("finalize_and_feed_current_template() still reports success "
                      "(receive succeeded, feed intentionally suppressed)", fed2);
    print_test_result("Block handler NOT re-invoked for suppressed same-tip re-feed "
                      "(pre-existing guard behavior preserved)", block_handler_calls == 1);
    print_test_result("Recovery-confirmed handler IS invoked for the same-tip re-confirmation "
                      "(the fix under test)", recovery_confirmed_calls == 1);
}

void test_genuinely_new_tip_bypasses_guard()
{
    std::cout << "\nTest 3: A genuinely new tip (different hashPrevBlock) bypasses the guard "
                 "and does not spuriously fire recovery-confirmed\n";
    auto solo = make_solo();

    int block_handler_calls = 0;
    int recovery_confirmed_calls = 0;
    solo->set_block_handler([&](::LLP::CBlock, std::uint32_t) { ++block_handler_calls; return true; });
    solo->set_recovery_confirmed_handler([&]() { ++recovery_confirmed_calls; });

    auto payload1 = make_block_payload(6771846, 0xAA);
    solo->get_template_interface()->read_template(payload1, "test_node", false);
    SameTipReconfirmationHarness::finalize(*solo, 6771846, 0, "test");

    // A genuinely new tip arrives at the next height with a different hashPrevBlock.
    auto payload2 = make_block_payload(6771847, 0xBB);
    auto result2 = solo->get_template_interface()->read_template(payload2, "test_node", false);
    print_test_result("read_template() accepts new-tip template", result2.is_valid);

    bool fed2 = SameTipReconfirmationHarness::finalize(*solo, 6771847, 0, "test");
    print_test_result("finalize_and_feed_current_template() succeeds for new tip", fed2);
    print_test_result("Block handler invoked again for the genuinely new tip",
                      block_handler_calls == 2);
    print_test_result("Recovery-confirmed handler NOT invoked for a normal new-tip feed",
                      recovery_confirmed_calls == 0);
}

void test_same_tip_refed_after_workers_stop()
{
    std::cout << "\nTest 4: A replacement worker generation receives the same tip\n";
    auto solo = make_solo();
    int feeds = 0;
    int confirmations = 0;
    solo->set_block_handler([&](::LLP::CBlock, std::uint32_t) { ++feeds; return true; });
    solo->set_recovery_confirmed_handler([&]() { ++confirmations; });
    auto payload = make_block_payload(6771846, 0xAA);
    auto* tmpl = solo->get_template_interface();
    print_test_result("Initial template accepted",
                      tmpl->read_template(payload, "test_node", false).is_valid);
    print_test_result("Initial worker generation fed",
                      SameTipReconfirmationHarness::finalize(*solo, 6771846, 0, "test"));

    solo->reset_worker_feed_guards();
    print_test_result("Same-tip replacement accepted",
                      tmpl->read_template(payload, "test_node", false).is_valid);
    print_test_result("Same-tip template reaches replacement workers",
                      SameTipReconfirmationHarness::finalize(*solo, 6771846, 0, "test") &&
                      feeds == 2 && confirmations == 0);
}

} // namespace

int main()
{
    auto null_sink = std::make_shared<spdlog::sinks::null_sink_mt>();
    auto logger = std::make_shared<spdlog::logger>("same_tip_reconfirmation_test_logger", null_sink);
    spdlog::set_default_logger(logger);

    std::cout << "========================================\n";
    std::cout << "Same-Tip Reconfirmation Tests\n";
    std::cout << "========================================\n";

    test_first_feed_invokes_block_handler_only();
    test_same_tip_reconfirmation_fires_recovery_confirmed_not_block_handler();
    test_genuinely_new_tip_bypasses_guard();
    test_same_tip_refed_after_workers_stop();

    std::cout << "\n========================================\n";
    std::cout << "Test Summary\n";
    std::cout << "========================================\n";
    std::cout << "Tests run:    " << tests_run << '\n';
    std::cout << "Tests passed: " << tests_passed << '\n';
    std::cout << "Tests failed: " << tests_failed << '\n';
    std::cout << "========================================\n";

    return tests_failed == 0 ? 0 : 1;
}
