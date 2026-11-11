/**
 * @file prepare_block_submission_from_solved_test.cpp
 * @brief Unit tests for MiningTemplateInterface::prepare_block_submission_from_solved()
 *
 * Verifies that the new submission path serializes the worker's Block_data snapshot
 * (not m_current_template), fixing the race condition where a concurrent BLOCK_DATA
 * push could advance nHeight between "worker found" and "submit-prep".
 *
 * Tests:
 *  1. Worker nHeight is preserved in the serialized payload under template advance.
 *  2. Worker hashPrevBlock is preserved in the serialized payload under template advance.
 *  3. Drift warning is emitted when worker height != current template height.
 *  4. No drift warning when worker height == current template height.
 *  5. vOffsets overload appends offsets correctly for Prime channel (nChannel == 1).
 *  6. Hash channel ignores vOffsets (no extra bytes appended).
 */

#include "protocol/mining_template_interface.hpp"
#include "worker/worker.hpp"
#include "LLP/block.hpp"
#include <iostream>
#include <cassert>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

#include "spdlog/spdlog.h"
#include "spdlog/sinks/null_sink.h"
#include "spdlog/sinks/ostream_sink.h"

using namespace nexusminer::protocol;

// ── Test infrastructure ───────────────────────────────────────────────────────
static int g_tests_run    = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;

static void print_result(const char* name, bool passed) {
    g_tests_run++;
    if (passed) {
        g_tests_passed++;
        std::cout << "  [PASS] " << name << "\n";
    } else {
        g_tests_failed++;
        std::cout << "  [FAIL] " << name << "\n";
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

static uint8_t be_byte(uint32_t v, int i) {
    return static_cast<uint8_t>((v >> (24 - 8 * i)) & 0xFF);
}

/**
 * Build a minimal valid 228-byte STATELESS_GET_BLOCK payload for the given height.
 * Uses a distinct hashPrevBlock pattern so tests can detect which template was used.
 */
static nexusminer::network::Payload make_template_payload(
    uint32_t unified_h,
    uint32_t channel   = 2,
    uint32_t nHeight   = 0,  // block nHeight (inside the 216-byte body)
    uint8_t  prev_fill = 0x00)
{
    if (nHeight == 0) nHeight = unified_h + 1;
    const uint32_t nBits = 0x04308519;

    nexusminer::network::Payload buf(228, 0x00);

    // 12-byte metadata prefix (big-endian)
    buf[0] = be_byte(unified_h, 0); buf[1] = be_byte(unified_h, 1);
    buf[2] = be_byte(unified_h, 2); buf[3] = be_byte(unified_h, 3);
    // channel_height = 0 (bytes 4-7 already zero)
    buf[8] = be_byte(nBits, 0); buf[9] = be_byte(nBits, 1);
    buf[10] = be_byte(nBits, 2); buf[11] = be_byte(nBits, 3);

    // 216-byte Tritium block body starts at offset 12
    size_t b = 12;
    // nVersion (4 bytes BE)
    const uint32_t nVersion = 8;
    buf[b]   = be_byte(nVersion, 0); buf[b+1] = be_byte(nVersion, 1);
    buf[b+2] = be_byte(nVersion, 2); buf[b+3] = be_byte(nVersion, 3);
    b += 4;
    // hashPrevBlock (128 bytes) — fill with prev_fill to distinguish templates
    for (size_t i = 0; i < 128; ++i) buf[b + i] = prev_fill;
    b += 128;
    // hashMerkleRoot (64 bytes) — non-zero pattern to pass validation
    for (size_t i = 0; i < 64; ++i) buf[b + i] = static_cast<uint8_t>(0xA0 + (i & 0x0F));
    b += 64;
    // nChannel (4 bytes BE)
    buf[b] = be_byte(channel, 0); buf[b+1] = be_byte(channel, 1);
    buf[b+2] = be_byte(channel, 2); buf[b+3] = be_byte(channel, 3);
    b += 4;
    // nHeight (4 bytes BE)
    buf[b] = be_byte(nHeight, 0); buf[b+1] = be_byte(nHeight, 1);
    buf[b+2] = be_byte(nHeight, 2); buf[b+3] = be_byte(nHeight, 3);
    b += 4;
    // nBits (4 bytes BE)
    buf[b] = be_byte(nBits, 0); buf[b+1] = be_byte(nBits, 1);
    buf[b+2] = be_byte(nBits, 2); buf[b+3] = be_byte(nBits, 3);
    b += 4;
    // nNonce (8 bytes LE) — leave as zero (template nonce)
    (void)b;

    return buf;
}

/**
 * Return a Block_data snapshot that matches the given template payload's fields.
 */
static nexusminer::Block_data make_worker_snapshot(uint32_t nHeight,
                                                    uint32_t channel  = 2,
                                                    uint8_t  prev_fill = 0x00,
                                                    uint64_t nNonce   = 0xDEADBEEFCAFEBABEULL)
{
    nexusminer::Block_data bd;
    bd.nVersion  = 8;
    bd.nChannel  = channel;
    bd.nHeight   = nHeight;
    bd.nBits     = 0x04308519;
    bd.nNonce    = nNonce;
    // hashPrevBlock — fill with prev_fill to match the corresponding template
    auto prev_bytes = bd.previous_hash.GetBytes();
    std::fill(prev_bytes.begin(), prev_bytes.end(), prev_fill);
    bd.previous_hash.SetBytes(prev_bytes);
    // hashMerkleRoot — non-zero pattern
    auto mr_bytes = bd.merkle_root.GetBytes();
    for (size_t i = 0; i < mr_bytes.size(); ++i) mr_bytes[i] = static_cast<uint8_t>(0xA0 + (i & 0x0F));
    bd.merkle_root.SetBytes(mr_bytes);
    return bd;
}

/**
 * Decode the big-endian uint32 at payload[offset..offset+3].
 */
static uint32_t read_be_u32(const std::vector<uint8_t>& payload, size_t offset) {
    return (static_cast<uint32_t>(payload[offset])     << 24)
         | (static_cast<uint32_t>(payload[offset + 1]) << 16)
         | (static_cast<uint32_t>(payload[offset + 2]) <<  8)
         |  static_cast<uint32_t>(payload[offset + 3]);
}

// ── Tests ─────────────────────────────────────────────────────────────────────

// Cunningham-chain offset vector reused across vOffsets tests.
static const std::vector<uint8_t> TEST_VOFFSETS{12, 6, 12, 12, 6, 12, 30, 17, 225, 0};

/**
 * Test 1: Worker nHeight is preserved in the serialized payload even when the current
 *         template has been refreshed to height N+1 between "found" and "submit-prep".
 */
static void test_preserves_worker_nheight_under_template_advance() {
    auto null_sink = std::make_shared<spdlog::sinks::null_sink_mt>();
    auto logger = std::make_shared<spdlog::logger>("logger", null_sink);
    spdlog::set_default_logger(logger);

    const uint32_t WORKER_HEIGHT  = 6679414u;
    const uint32_t REFRESH_HEIGHT = 6679415u;

    MiningTemplateInterface mti(2, 0);

    // Load template at WORKER_HEIGHT (the one the worker mined against).
    auto tmpl_N = make_template_payload(WORKER_HEIGHT - 1, 2, WORKER_HEIGHT, 0xAA);
    mti.read_stateless_payload(tmpl_N, "test");

    // Build a worker snapshot from that template.
    auto worker_snap = make_worker_snapshot(WORKER_HEIGHT, 2, 0xAA);

    // Simulate a concurrent BLOCK_DATA push advancing the template to REFRESH_HEIGHT.
    auto tmpl_N1 = make_template_payload(REFRESH_HEIGHT - 1, 2, REFRESH_HEIGHT, 0xBB);
    mti.read_stateless_payload(tmpl_N1, "test");

    // The current template is now at REFRESH_HEIGHT.
    assert(mti.get_template_height() == REFRESH_HEIGHT);

    // Prepare submission from the worker's snapshot.
    auto payload = mti.prepare_block_submission_from_solved(worker_snap);

    bool not_empty = !payload.empty();
    // Tritium: nHeight is big-endian uint32 at bytes [200-203].
    bool height_ok = not_empty && (payload.size() >= 204) &&
                     (read_be_u32(payload, 200) == WORKER_HEIGHT);

    print_result("Test 1: worker nHeight preserved in payload under template advance", height_ok);
}

/**
 * Test 2: Worker hashPrevBlock is preserved in the serialized payload under template advance.
 */
static void test_preserves_worker_prev_block_under_template_advance() {
    auto null_sink = std::make_shared<spdlog::sinks::null_sink_mt>();
    auto logger = std::make_shared<spdlog::logger>("logger", null_sink);
    spdlog::set_default_logger(logger);

    const uint32_t WORKER_HEIGHT  = 6679414u;
    const uint32_t REFRESH_HEIGHT = 6679415u;
    const uint8_t  WORKER_PREV_FILL  = 0xAA;
    const uint8_t  REFRESH_PREV_FILL = 0xBB;

    MiningTemplateInterface mti(2, 0);

    auto tmpl_N = make_template_payload(WORKER_HEIGHT - 1, 2, WORKER_HEIGHT, WORKER_PREV_FILL);
    mti.read_stateless_payload(tmpl_N, "test");

    auto worker_snap = make_worker_snapshot(WORKER_HEIGHT, 2, WORKER_PREV_FILL);

    // Advance template — different hashPrevBlock (fill 0xBB).
    auto tmpl_N1 = make_template_payload(REFRESH_HEIGHT - 1, 2, REFRESH_HEIGHT, REFRESH_PREV_FILL);
    mti.read_stateless_payload(tmpl_N1, "test");

    auto payload = mti.prepare_block_submission_from_solved(worker_snap);
    if (payload.empty()) {
        print_result("Test 2: worker hashPrevBlock preserved in payload under template advance", false);
        return;
    }

    // Tritium: hashPrevBlock is 128 bytes at [4..131].
    bool prev_ok = true;
    for (size_t i = 0; i < 128; ++i) {
        if (payload[4 + i] != WORKER_PREV_FILL) { prev_ok = false; break; }
    }
    print_result("Test 2: worker hashPrevBlock preserved in payload under template advance", prev_ok);
}

/**
 * Test 3: Drift warning is logged when worker nHeight != current template nHeight.
 */
static void test_drift_warning_emitted_on_height_mismatch() {
    std::ostringstream log_stream;
    auto ostream_sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(log_stream);
    auto logger = std::make_shared<spdlog::logger>("logger", ostream_sink);
    logger->set_level(spdlog::level::trace);
    spdlog::set_default_logger(logger);

    const uint32_t WORKER_HEIGHT  = 6679414u;
    const uint32_t REFRESH_HEIGHT = 6679415u;

    MiningTemplateInterface mti(2, 0);

    auto tmpl_N = make_template_payload(WORKER_HEIGHT - 1, 2, WORKER_HEIGHT, 0xAA);
    mti.read_stateless_payload(tmpl_N, "test");

    auto worker_snap = make_worker_snapshot(WORKER_HEIGHT, 2, 0xAA);

    // Advance template so there is a drift.
    auto tmpl_N1 = make_template_payload(REFRESH_HEIGHT - 1, 2, REFRESH_HEIGHT, 0xBB);
    mti.read_stateless_payload(tmpl_N1, "test");

    mti.prepare_block_submission_from_solved(worker_snap);

    logger->flush();
    const std::string log_out = log_stream.str();
    bool drift_warned = log_out.find("Template advanced between worker-found and submit-prep") != std::string::npos;
    print_result("Test 3: drift warning emitted when worker height != template height", drift_warned);
}

/**
 * Test 4: No drift warning when worker height == current template height.
 */
static void test_no_drift_warning_when_heights_match() {
    std::ostringstream log_stream;
    auto ostream_sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(log_stream);
    auto logger = std::make_shared<spdlog::logger>("logger", ostream_sink);
    logger->set_level(spdlog::level::trace);
    spdlog::set_default_logger(logger);

    const uint32_t HEIGHT = 6679414u;

    MiningTemplateInterface mti(2, 0);
    auto tmpl = make_template_payload(HEIGHT - 1, 2, HEIGHT, 0xAA);
    mti.read_stateless_payload(tmpl, "test");

    auto worker_snap = make_worker_snapshot(HEIGHT, 2, 0xAA);

    // No template refresh — heights still match.
    mti.prepare_block_submission_from_solved(worker_snap);

    logger->flush();
    const std::string log_out = log_stream.str();
    bool no_drift_warn  = log_out.find("Template advanced between worker-found and submit-prep") == std::string::npos;
    bool match_logged   = log_out.find("Worker snapshot matches current template") != std::string::npos;
    print_result("Test 4: no drift warning and match-line logged when heights match", no_drift_warn && match_logged);
}

/**
 * Test 5: vOffsets overload appends bytes correctly for Prime channel (nChannel == 1).
 */
static void test_prime_channel_voffsets_appended() {
    auto null_sink = std::make_shared<spdlog::sinks::null_sink_mt>();
    auto logger = std::make_shared<spdlog::logger>("logger", null_sink);
    spdlog::set_default_logger(logger);

    const uint32_t HEIGHT = 6679414u;
    MiningTemplateInterface mti(1, 0);  // channel 1 = Prime

    auto tmpl = make_template_payload(HEIGHT - 1, 1, HEIGHT, 0xCC);
    mti.read_stateless_payload(tmpl, "test");

    auto worker_snap = make_worker_snapshot(HEIGHT, 1, 0xCC);

    auto without_offsets = mti.prepare_block_submission_from_solved(worker_snap);
    auto with_offsets    = mti.prepare_block_submission_from_solved(worker_snap, TEST_VOFFSETS);

    bool base_nonempty   = !without_offsets.empty();
    bool with_nonempty   = !with_offsets.empty();
    bool size_correct    = base_nonempty && with_nonempty &&
                           (with_offsets.size() == without_offsets.size() + TEST_VOFFSETS.size());

    // Verify the appended bytes are the exact offset values.
    bool offsets_correct = size_correct;
    if (offsets_correct) {
        size_t base = without_offsets.size();
        for (size_t i = 0; i < TEST_VOFFSETS.size(); ++i) {
            if (with_offsets[base + i] != TEST_VOFFSETS[i]) { offsets_correct = false; break; }
        }
    }
    print_result("Test 5: Prime channel vOffsets appended correctly", size_correct && offsets_correct);
}

/**
 * Test 6: Hash channel ignores vOffsets — payload size equals the no-offsets case.
 */
static void test_hash_channel_ignores_voffsets() {
    auto null_sink = std::make_shared<spdlog::sinks::null_sink_mt>();
    auto logger = std::make_shared<spdlog::logger>("logger", null_sink);
    spdlog::set_default_logger(logger);

    const uint32_t HEIGHT = 6679414u;
    MiningTemplateInterface mti(2, 0);  // channel 2 = Hash

    auto tmpl = make_template_payload(HEIGHT - 1, 2, HEIGHT, 0xDD);
    mti.read_stateless_payload(tmpl, "test");

    auto worker_snap = make_worker_snapshot(HEIGHT, 2, 0xDD);

    auto without_offsets = mti.prepare_block_submission_from_solved(worker_snap);
    auto with_offsets    = mti.prepare_block_submission_from_solved(worker_snap, TEST_VOFFSETS);

    bool same_size = !without_offsets.empty() && (without_offsets.size() == with_offsets.size());
    print_result("Test 6: Hash channel ignores vOffsets (same payload size)", same_size);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "  prepare_block_submission_from_solved_test\n";
    std::cout << "========================================\n";

    test_preserves_worker_nheight_under_template_advance();
    test_preserves_worker_prev_block_under_template_advance();
    test_drift_warning_emitted_on_height_mismatch();
    test_no_drift_warning_when_heights_match();
    test_prime_channel_voffsets_appended();
    test_hash_channel_ignores_voffsets();

    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "Test Summary\n";
    std::cout << "========================================\n";
    std::cout << "Tests run:    " << g_tests_run    << "\n";
    std::cout << "Tests passed: " << g_tests_passed << "\n";
    std::cout << "Tests failed: " << g_tests_failed << "\n";
    std::cout << "Success rate: "
              << (g_tests_run > 0 ? 100 * g_tests_passed / g_tests_run : 0) << "%\n";
    std::cout << "========================================\n\n";

    return (g_tests_failed == 0) ? 0 : 1;
}
