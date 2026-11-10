/**
 * @file stateless_metadata_divergence_test.cpp
 * @brief Unit tests for stateless BLOCK_DATA metadata divergence diagnostics
 *
 * Verifies Diagnostics 1-3 added to MiningTemplateInterface::read_stateless_payload()
 * and the three atomic counters exposed via TemplateStats / reset_stats().
 *
 * Tests:
 *  1. No divergence when metadata matches body → counters remain zero, no warn logged
 *  2. nBits divergence detected → m_stateless_nbits_divergence_count == 1, warn logged
 *  3. Height divergence detected → m_stateless_height_divergence_count == 1, warn logged
 *  4. channel_height > unified_height detected → m_stateless_channel_sanity_violations == 1
 *  5. Counters survive multiple divergent payloads → counter == 3
 *  6. reset_stats() zeroes the new counters
 *  7. Counters are independent — only the triggered counter increments
 */

#include "protocol/mining_template_interface.hpp"
#include <iostream>
#include <sstream>
#include <cassert>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "spdlog/spdlog.h"
#include "spdlog/sinks/ostream_sink.h"

using namespace nexusminer::protocol;
using Payload = std::vector<uint8_t>;

class OwningOstreamSink : public spdlog::sinks::ostream_sink_mt
{
public:
    explicit OwningOstreamSink(std::shared_ptr<std::ostringstream> stream)
        : spdlog::sinks::ostream_sink_mt(*stream), m_stream(std::move(stream))
    {
    }

private:
    std::shared_ptr<std::ostringstream> m_stream;
};

// ── Test infrastructure ──────────────────────────────────────────────────────

static int tests_run    = 0;
static int tests_passed = 0;
static int tests_failed = 0;

static void print_result(const char* name, bool passed) {
    tests_run++;
    if (passed) {
        tests_passed++;
        std::cout << "  [PASS] " << name << "\n";
    } else {
        tests_failed++;
        std::cout << "  [FAIL] " << name << "\n";
    }
}

// ── Payload construction helpers ─────────────────────────────────────────────

static uint8_t be_byte(uint32_t v, int i) {
    return static_cast<uint8_t>((v >> (24 - 8 * i)) & 0xFF);
}

/**
 * Build a 228-byte STATELESS_GET_BLOCK payload.
 *
 * Layout:
 *   [0-3]    unified_height  (BE)
 *   [4-7]    channel_height  (BE)
 *   [8-11]   meta_nbits      (BE) — the metadata echo field
 *   [12-227] 216-byte Tritium block body (all BE except nNonce LE)
 *     body[0-3]     nVersion
 *     body[4-131]   hashPrevBlock (128 bytes, all zeros)
 *     body[132-195] hashMerkleRoot (64 bytes, non-zero)
 *     body[196-199] nChannel
 *     body[200-203] nHeight (body_height)
 *     body[204-207] body_nbits
 *     body[208-215] nNonce (LE, zeros)
 */
static Payload make_divergence_test_payload(
    uint32_t unified_height,
    uint32_t channel_height,
    uint32_t meta_nbits,   // the value written into the 12-byte prefix's nBits field
    uint32_t body_height,  // the value written into the block body's nHeight field
    uint32_t body_nbits,   // the value written into the block body's nBits field
    uint32_t body_channel = 2)  // Hash channel by default
{
    Payload buf(228, 0x00);

    // Metadata prefix (big-endian)
    buf[0]  = be_byte(unified_height,  0); buf[1]  = be_byte(unified_height,  1);
    buf[2]  = be_byte(unified_height,  2); buf[3]  = be_byte(unified_height,  3);
    buf[4]  = be_byte(channel_height,  0); buf[5]  = be_byte(channel_height,  1);
    buf[6]  = be_byte(channel_height,  2); buf[7]  = be_byte(channel_height,  3);
    buf[8]  = be_byte(meta_nbits,      0); buf[9]  = be_byte(meta_nbits,      1);
    buf[10] = be_byte(meta_nbits,      2); buf[11] = be_byte(meta_nbits,      3);

    // Block body at offset 12
    size_t b = 12;
    // nVersion [0-3]
    const uint32_t nVersion = 8;
    buf[b]   = be_byte(nVersion, 0); buf[b+1] = be_byte(nVersion, 1);
    buf[b+2] = be_byte(nVersion, 2); buf[b+3] = be_byte(nVersion, 3);
    b += 4;
    // hashPrevBlock [4-131] -- zeros
    b += 128;
    // hashMerkleRoot [132-195] -- non-zero so validate_template() passes
    for (size_t i = 0; i < 64; ++i)
        buf[b + i] = static_cast<uint8_t>(0xA0 + (i & 0x0F));
    b += 64;
    // nChannel [196-199]
    buf[b]   = be_byte(body_channel, 0); buf[b+1] = be_byte(body_channel, 1);
    buf[b+2] = be_byte(body_channel, 2); buf[b+3] = be_byte(body_channel, 3);
    b += 4;
    // nHeight [200-203]
    buf[b]   = be_byte(body_height, 0); buf[b+1] = be_byte(body_height, 1);
    buf[b+2] = be_byte(body_height, 2); buf[b+3] = be_byte(body_height, 3);
    b += 4;
    // nBits [204-207]
    buf[b]   = be_byte(body_nbits, 0); buf[b+1] = be_byte(body_nbits, 1);
    buf[b+2] = be_byte(body_nbits, 2); buf[b+3] = be_byte(body_nbits, 3);
    b += 4;
    // nNonce [208-215] -- little-endian zeros

    return buf;
}

/**
 * Build a valid payload where all metadata fields are consistent with the body.
 * Default: unified=6000000, channel=2000000, nBits=0x041cd34b, body_height=6000001.
 */
static Payload make_consistent_divergence_payload(
    uint32_t unified  = 6000000,
    uint32_t channel  = 2000000,
    uint32_t nbits    = 0x041cd34bu,
    uint32_t body_h   = 6000001)
{
    return make_divergence_test_payload(unified, channel, nbits, body_h, nbits);
}

/**
 * Create a MiningTemplateInterface with a capturing logger.
 * Returns (mti, log_stream) so callers can inspect what was logged.
 */
static std::pair<std::unique_ptr<MiningTemplateInterface>,
                 std::shared_ptr<std::ostringstream>>
make_capturing_mti(uint8_t channel = 2)
{
    auto oss  = std::make_shared<std::ostringstream>();
    auto sink = std::make_shared<OwningOstreamSink>(oss);
    sink->set_level(spdlog::level::debug);
    auto logger = std::make_shared<spdlog::logger>("test_capture", sink);
    logger->set_level(spdlog::level::debug);
    spdlog::set_default_logger(logger);

    auto mti = std::make_unique<MiningTemplateInterface>(channel, 0);
    return {std::move(mti), oss};
}

// ── Tests ────────────────────────────────────────────────────────────────────

// Test 1: No divergence — all counters stay zero, no warn logged
static void test_no_divergence() {
    auto [mti, oss] = make_capturing_mti();

    auto payload = make_consistent_divergence_payload();
    auto result  = mti->read_stateless_payload(payload, "test");

    auto stats = mti->get_stats();
    std::string log_text = oss->str();

    bool result_valid  = result.is_valid;
    bool no_nbits_div  = (stats.stateless_nbits_divergence_count    == 0);
    bool no_height_div = (stats.stateless_height_divergence_count   == 0);
    bool no_chan_viol  = (stats.stateless_channel_sanity_violations == 0);
    bool no_warn       = (log_text.find("DIVERGENCE") == std::string::npos &&
                          log_text.find("SANITY")     == std::string::npos);

    print_result("Test 1: no divergence — result valid",         result_valid);
    print_result("Test 1: no divergence — nbits counter == 0",   no_nbits_div);
    print_result("Test 1: no divergence — height counter == 0",  no_height_div);
    print_result("Test 1: no divergence — channel counter == 0", no_chan_viol);
    print_result("Test 1: no divergence — no DIVERGENCE/SANITY warn in log", no_warn);
}

// Test 2: nBits divergence detected
static void test_nbits_divergence() {
    auto [mti, oss] = make_capturing_mti();

    // metadata.nBits = 0xCAFEBABE, body.nBits = 0x041cd34b
    auto payload = make_divergence_test_payload(6000000, 2000000,
                                /*meta_nbits=*/0xCAFEBABEu,
                                /*body_height=*/6000001,
                                /*body_nbits=*/0x041cd34bu);
    auto result  = mti->read_stateless_payload(payload, "test");

    auto stats    = mti->get_stats();
    std::string log_text = oss->str();

    print_result("Test 2: nBits divergence — payload accepted",
                 result.is_valid);
    print_result("Test 2: nBits divergence — nbits counter == 1",
                 stats.stateless_nbits_divergence_count == 1);
    print_result("Test 2: nBits divergence — height counter unchanged",
                 stats.stateless_height_divergence_count == 0);
    print_result("Test 2: nBits divergence — channel counter unchanged",
                 stats.stateless_channel_sanity_violations == 0);
    print_result("Test 2: nBits divergence — DIVERGENCE (nBits) warn logged",
                 log_text.find("DIVERGENCE (nBits)") != std::string::npos);
}

// Test 3: Height divergence detected
static void test_height_divergence() {
    auto [mti, oss] = make_capturing_mti();

    // metadata.unified_height = 100, body.nHeight = 200 (not 101)
    auto payload = make_divergence_test_payload(/*unified=*/100, /*channel=*/50,
                                /*meta_nbits=*/0x041cd34bu,
                                /*body_height=*/200,
                                /*body_nbits=*/0x041cd34bu);
    auto result  = mti->read_stateless_payload(payload, "test");

    auto stats    = mti->get_stats();
    std::string log_text = oss->str();

    print_result("Test 3: height divergence — payload accepted",
                 result.is_valid);
    print_result("Test 3: height divergence — height counter == 1",
                 stats.stateless_height_divergence_count == 1);
    print_result("Test 3: height divergence — nbits counter unchanged",
                 stats.stateless_nbits_divergence_count == 0);
    print_result("Test 3: height divergence — DIVERGENCE (height) warn logged",
                 log_text.find("DIVERGENCE (height)") != std::string::npos);
}

// Test 4: channel_height > unified_height detected
static void test_channel_gt_unified() {
    auto [mti, oss] = make_capturing_mti();

    // channel_height (9000000) > unified_height (6000000) — impossible state
    auto payload = make_divergence_test_payload(/*unified=*/6000000, /*channel=*/9000000,
                                /*meta_nbits=*/0x041cd34bu,
                                /*body_height=*/6000001,
                                /*body_nbits=*/0x041cd34bu);
    auto result  = mti->read_stateless_payload(payload, "test");

    auto stats    = mti->get_stats();
    std::string log_text = oss->str();

    print_result("Test 4: channel > unified — payload accepted",
                 result.is_valid);
    print_result("Test 4: channel > unified — channel counter == 1",
                 stats.stateless_channel_sanity_violations == 1);
    print_result("Test 4: channel > unified — nbits counter unchanged",
                 stats.stateless_nbits_divergence_count == 0);
    print_result("Test 4: channel > unified — height counter unchanged",
                 stats.stateless_height_divergence_count == 0);
    print_result("Test 4: channel > unified — SANITY (channel > unified) warn logged",
                 log_text.find("SANITY (channel > unified)") != std::string::npos);
}

// Test 5: Counter survives multiple divergent payloads — counter == 3
static void test_counter_accumulates() {
    auto [mti, oss] = make_capturing_mti();

    // Three payloads all with nBits divergence
    for (int i = 0; i < 3; ++i) {
        // Each payload must have a different body height (sequential block advance)
        // so that validate_template() doesn't reject it as a duplicate.
        uint32_t h = 6000001u + static_cast<uint32_t>(i);
        auto payload = make_divergence_test_payload(h - 1, 2000000,
                                    /*meta_nbits=*/0xDEADBEEFu,
                                    /*body_height=*/h,
                                    /*body_nbits=*/0x041cd34bu);
        mti->read_stateless_payload(payload, "test");
    }

    auto stats = mti->get_stats();
    print_result("Test 5: counters accumulate — nbits counter == 3",
                 stats.stateless_nbits_divergence_count == 3);
    print_result("Test 5: counters accumulate — height/channel counters unchanged",
                 stats.stateless_height_divergence_count   == 0 &&
                 stats.stateless_channel_sanity_violations == 0);
}

// Test 6: reset_stats() zeroes the new counters
static void test_reset_stats() {
    auto [mti, oss] = make_capturing_mti();

    // All three payloads must use sequential, validate_template()-compatible heights
    // so they pass the body validation and reach the diagnostic code.

    // nBits divergence at height 6000001 (unified=6000000, body=6000001, consistent height)
    auto p1 = make_divergence_test_payload(6000000, 2000000, 0xDEADBEEFu, 6000001, 0x041cd34bu);
    mti->read_stateless_payload(p1, "test");

    // Height divergence at body height 6000003 (unified=6000001 → expects 6000002 but gets 6000003)
    // validate_template() accepts: jump +2 from previous 6000001, within ±100.
    auto p2 = make_divergence_test_payload(6000001, 2000000, 0x041cd34bu, 6000003, 0x041cd34bu);
    mti->read_stateless_payload(p2, "test");

    // Channel > unified at body height 6000004 (sequential +1 from 6000003)
    auto p3 = make_divergence_test_payload(6000003, 9000000, 0x041cd34bu, 6000004, 0x041cd34bu);
    mti->read_stateless_payload(p3, "test");

    // Pre-reset: all three counters should be non-zero
    auto before = mti->get_stats();
    bool pre_ok = (before.stateless_nbits_divergence_count    >= 1 &&
                   before.stateless_height_divergence_count   >= 1 &&
                   before.stateless_channel_sanity_violations >= 1);
    print_result("Test 6: reset_stats — pre-reset counters non-zero", pre_ok);

    mti->reset_stats();

    auto after = mti->get_stats();
    print_result("Test 6: reset_stats — nbits counter zeroed",
                 after.stateless_nbits_divergence_count == 0);
    print_result("Test 6: reset_stats — height counter zeroed",
                 after.stateless_height_divergence_count == 0);
    print_result("Test 6: reset_stats — channel counter zeroed",
                 after.stateless_channel_sanity_violations == 0);
}

// Test 7: Counters are independent — only the triggered counter increments
static void test_counters_independent() {
    // Sub-test A: only nBits divergence
    {
        auto [mti, oss] = make_capturing_mti();
        auto payload = make_divergence_test_payload(6000000, 2000000, 0xCAFEBABEu, 6000001, 0x041cd34bu);
        mti->read_stateless_payload(payload, "test");
        auto s = mti->get_stats();
        print_result("Test 7a: only nbits counter fires (nbits==1)",
                     s.stateless_nbits_divergence_count    == 1 &&
                     s.stateless_height_divergence_count   == 0 &&
                     s.stateless_channel_sanity_violations == 0);
    }

    // Sub-test B: only height divergence
    {
        auto [mti, oss] = make_capturing_mti();
        auto payload = make_divergence_test_payload(100, 50, 0x041cd34bu, 200, 0x041cd34bu);
        mti->read_stateless_payload(payload, "test");
        auto s = mti->get_stats();
        print_result("Test 7b: only height counter fires (height==1)",
                     s.stateless_nbits_divergence_count    == 0 &&
                     s.stateless_height_divergence_count   == 1 &&
                     s.stateless_channel_sanity_violations == 0);
    }

    // Sub-test C: only channel > unified
    {
        auto [mti, oss] = make_capturing_mti();
        auto payload = make_divergence_test_payload(6000000, 9000000, 0x041cd34bu, 6000001, 0x041cd34bu);
        mti->read_stateless_payload(payload, "test");
        auto s = mti->get_stats();
        print_result("Test 7c: only channel counter fires (channel==1)",
                     s.stateless_nbits_divergence_count    == 0 &&
                     s.stateless_height_divergence_count   == 0 &&
                     s.stateless_channel_sanity_violations == 1);
    }
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "  stateless_metadata_divergence_test\n";
    std::cout << "========================================\n";

    test_no_divergence();
    test_nbits_divergence();
    test_height_divergence();
    test_channel_gt_unified();
    test_counter_accumulates();
    test_reset_stats();
    test_counters_independent();

    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "Test Summary\n";
    std::cout << "========================================\n";
    std::cout << "Tests run:    " << tests_run    << "\n";
    std::cout << "Tests passed: " << tests_passed << "\n";
    std::cout << "Tests failed: " << tests_failed << "\n";
    std::cout << "Success rate: "
              << (tests_run > 0 ? 100 * tests_passed / tests_run : 0)
              << "%\n";
    std::cout << "========================================\n";

    return (tests_failed == 0) ? 0 : 1;
}
