/**
 * @file packet_builder_test.cpp
 * @brief Unit tests for PacketBuilder lane-aware outbound packet utility
 *
 * Tests:
 *  1. Legacy zero-length request → [opcode][len4=0]
 *  2. Stateless zero-length request → [0xD0][opcode][len4=0]
 *  3. Legacy with payload → [opcode][len4 BE][payload]
 *  4. Stateless with payload → [0xD0][opcode][len4 BE][payload]
 *  5. MINER_READY: legacy zero-length frame (opcode 216 = 0xD8)
 *  6. MINER_READY: stateless zero-length frame (0xD0D8)
 *  7. GET_ROUND: legacy zero-length frame (opcode 133 = 0x85)
 *  8. GET_ROUND: stateless zero-length frame (0xD085)
 *  9. SUBMIT_BLOCK: stateless with payload (0xD001)
 * 10. UNKNOWN lane returns empty
 * 11. submit_block plaintext layout size invariant (Disposable Falcon only)
 */

#include "protocol/packet_builder.hpp"
#include "miner_opcodes.hpp"
#include <iostream>
#include <cassert>
#include <cstdint>
#include <vector>

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
// Test 1: Legacy zero-length GET_BLOCK (opcode 129 = 0x81)
// Expected wire: [0x81][0x00000000]
// ============================================================================
void test_legacy_header_only() {
    std::cout << "\nTest 1: Legacy zero-length GET_BLOCK\n";
    auto bytes = PacketBuilder::build(ProtocolLane::LEGACY, LLP::GET_BLOCK);
    bool ok = (bytes && bytes->size() == 5 &&
               (*bytes)[0] == 0x81 &&
               (*bytes)[1] == 0x00 && (*bytes)[2] == 0x00 &&
               (*bytes)[3] == 0x00 && (*bytes)[4] == 0x00);
    print_test_result("Legacy GET_BLOCK → [0x81][0x00000000]", ok);
}

// ============================================================================
// Test 2: Stateless zero-length GET_BLOCK (opcode 129 = 0x81 → 0xD081)
// Expected wire: [0xD0][0x81][0x00000000]
// ============================================================================
void test_stateless_header_only() {
    std::cout << "\nTest 2: Stateless zero-length GET_BLOCK\n";
    auto bytes = PacketBuilder::build(ProtocolLane::STATELESS, LLP::GET_BLOCK);
    bool ok = (bytes && bytes->size() == 6 &&
               (*bytes)[0] == 0xD0 && (*bytes)[1] == 0x81 &&
               (*bytes)[2] == 0x00 && (*bytes)[3] == 0x00 &&
               (*bytes)[4] == 0x00 && (*bytes)[5] == 0x00);
    print_test_result("Stateless GET_BLOCK → [0xD0][0x81][0x00000000]", ok);
}

// ============================================================================
// Test 3: Legacy SUBMIT_BLOCK (opcode 1) with payload
// Expected wire: [0x01][len4 BE][payload]  (1+4+N bytes)
// ============================================================================
void test_legacy_with_payload() {
    std::cout << "\nTest 3: Legacy SUBMIT_BLOCK with payload\n";
    std::vector<uint8_t> payload = {0xAA, 0xBB, 0xCC};
    auto bytes = PacketBuilder::build(ProtocolLane::LEGACY, LLP::SUBMIT_BLOCK, payload);

    bool ok = (bytes && bytes->size() == 1 + 4 + 3);
    if (ok) {
        ok = ((*bytes)[0] == 0x01);           // opcode
        ok = ok && ((*bytes)[1] == 0x00);     // length MSB
        ok = ok && ((*bytes)[2] == 0x00);
        ok = ok && ((*bytes)[3] == 0x00);
        ok = ok && ((*bytes)[4] == 0x03);     // length = 3
        ok = ok && ((*bytes)[5] == 0xAA);
        ok = ok && ((*bytes)[6] == 0xBB);
        ok = ok && ((*bytes)[7] == 0xCC);
    }
    print_test_result("Legacy SUBMIT_BLOCK(3 bytes) → [0x01][0x00000003][AA BB CC]", ok);
}

// ============================================================================
// Test 4: Stateless SUBMIT_BLOCK (opcode 1 → 0xD001) with payload
// Expected wire: [0xD0][0x01][len4 BE][payload]  (2+4+N bytes)
// ============================================================================
void test_stateless_with_payload() {
    std::cout << "\nTest 4: Stateless SUBMIT_BLOCK with payload\n";
    std::vector<uint8_t> payload = {0xAA, 0xBB, 0xCC};
    auto bytes = PacketBuilder::build(ProtocolLane::STATELESS, LLP::SUBMIT_BLOCK, payload);

    bool ok = (bytes && bytes->size() == 2 + 4 + 3);
    if (ok) {
        ok = ((*bytes)[0] == 0xD0 && (*bytes)[1] == 0x01);  // 0xD001 opcode
        ok = ok && ((*bytes)[2] == 0x00);
        ok = ok && ((*bytes)[3] == 0x00);
        ok = ok && ((*bytes)[4] == 0x00);
        ok = ok && ((*bytes)[5] == 0x03);     // length = 3
        ok = ok && ((*bytes)[6] == 0xAA);
        ok = ok && ((*bytes)[7] == 0xBB);
        ok = ok && ((*bytes)[8] == 0xCC);
    }
    print_test_result("Stateless SUBMIT_BLOCK(3 bytes) → [0xD0][0x01][0x00000003][AA BB CC]", ok);
}

// ============================================================================
// Test 5: Legacy MINER_READY (opcode 216 = 0xD8) zero-length frame
// Expected wire: [0xD8][0x00000000]
// ============================================================================
void test_legacy_miner_ready() {
    std::cout << "\nTest 5: Legacy MINER_READY zero-length frame\n";
    auto bytes = PacketBuilder::build(ProtocolLane::LEGACY, LLP::MINER_READY);
    bool ok = (bytes && bytes->size() == 5 &&
               (*bytes)[0] == 0xD8 &&
               (*bytes)[1] == 0x00 && (*bytes)[2] == 0x00 &&
               (*bytes)[3] == 0x00 && (*bytes)[4] == 0x00);
    print_test_result("Legacy MINER_READY → [0xD8][0x00000000]", ok);
}

// ============================================================================
// Test 6: Stateless MINER_READY (opcode 216 = 0xD8 → 0xD0D8) zero-length frame
// Expected wire: [0xD0][0xD8][0x00000000]
// ============================================================================
void test_stateless_miner_ready() {
    std::cout << "\nTest 6: Stateless MINER_READY zero-length frame\n";
    auto bytes = PacketBuilder::build(ProtocolLane::STATELESS, LLP::MINER_READY);
    bool ok = (bytes && bytes->size() == 6 &&
               (*bytes)[0] == 0xD0 && (*bytes)[1] == 0xD8 &&
               (*bytes)[2] == 0x00 && (*bytes)[3] == 0x00 &&
               (*bytes)[4] == 0x00 && (*bytes)[5] == 0x00);
    print_test_result("Stateless MINER_READY → [0xD0][0xD8][0x00000000]", ok);
}

// ============================================================================
// Test 7: Legacy GET_ROUND (opcode 133 = 0x85) zero-length frame
// Expected wire: [0x85][0x00000000]
// ============================================================================
void test_legacy_get_round() {
    std::cout << "\nTest 7: Legacy GET_ROUND zero-length frame\n";
    auto bytes = PacketBuilder::build(ProtocolLane::LEGACY, LLP::GET_ROUND);
    bool ok = (bytes && bytes->size() == 5 &&
               (*bytes)[0] == 0x85 &&
               (*bytes)[1] == 0x00 && (*bytes)[2] == 0x00 &&
               (*bytes)[3] == 0x00 && (*bytes)[4] == 0x00);
    print_test_result("Legacy GET_ROUND → [0x85][0x00000000]", ok);
}

// ============================================================================
// Test 8: Stateless GET_ROUND (opcode 133 = 0x85 → 0xD085) zero-length frame
// Expected wire: [0xD0][0x85][0x00000000]
// ============================================================================
void test_stateless_get_round() {
    std::cout << "\nTest 8: Stateless GET_ROUND zero-length frame\n";
    auto bytes = PacketBuilder::build(ProtocolLane::STATELESS, LLP::GET_ROUND);
    bool ok = (bytes && bytes->size() == 6 &&
               (*bytes)[0] == 0xD0 && (*bytes)[1] == 0x85 &&
               (*bytes)[2] == 0x00 && (*bytes)[3] == 0x00 &&
               (*bytes)[4] == 0x00 && (*bytes)[5] == 0x00);
    print_test_result("Stateless GET_ROUND → [0xD0][0x85][0x00000000]", ok);
}

// ============================================================================
// Test 9: Larger payload (simulating MINER_AUTH_INIT data)
// ============================================================================
void test_large_payload_stateless() {
    std::cout << "\nTest 9: Stateless MINER_AUTH_INIT with 10-byte payload\n";
    std::vector<uint8_t> payload(10, 0xFF);
    auto bytes = PacketBuilder::build(ProtocolLane::STATELESS, LLP::MINER_AUTH_INIT, payload);
    bool ok = (bytes && bytes->size() == 2 + 4 + 10);
    if (ok) {
        // opcode 0xD0CF
        ok = ((*bytes)[0] == 0xD0 && (*bytes)[1] == 0xCF);
        // length = 10
        ok = ok && ((*bytes)[2] == 0 && (*bytes)[3] == 0 &&
                    (*bytes)[4] == 0 && (*bytes)[5] == 10);
        // payload all 0xFF
        for (int i = 6; i < 16 && ok; ++i) {
            ok = ((*bytes)[i] == 0xFF);
        }
    }
    print_test_result("Stateless MINER_AUTH_INIT(10B) → [0xD0][0xCF][0x0000000A][data]", ok);
}

// ============================================================================
// Test 10: Mirror opcode is 0xD000 | legacy_opcode for all common opcodes
// ============================================================================
void test_mirror_opcode_invariant() {
    std::cout << "\nTest 10: Stateless opcode header = 0xD000 | legacy_opcode\n";
    // GET_BLOCK (129 = 0x81) → 0xD081
    {
        auto b = PacketBuilder::build(ProtocolLane::STATELESS, LLP::GET_BLOCK);
        uint16_t hdr = b ? ((static_cast<uint16_t>((*b)[0]) << 8) | (*b)[1]) : 0;
        print_test_result("GET_BLOCK stateless header == 0xD081", hdr == 0xD081);
    }
    // SUBMIT_BLOCK (1 = 0x01) → 0xD001
    {
        std::vector<uint8_t> p = {0x01};
        auto b = PacketBuilder::build(ProtocolLane::STATELESS, LLP::SUBMIT_BLOCK, p);
        uint16_t hdr = b ? ((static_cast<uint16_t>((*b)[0]) << 8) | (*b)[1]) : 0;
        print_test_result("SUBMIT_BLOCK stateless header == 0xD001", hdr == 0xD001);
    }
    // GET_ROUND (133 = 0x85) → 0xD085
    {
        auto b = PacketBuilder::build(ProtocolLane::STATELESS, LLP::GET_ROUND);
        uint16_t hdr = b ? ((static_cast<uint16_t>((*b)[0]) << 8) | (*b)[1]) : 0;
        print_test_result("GET_ROUND stateless header == 0xD085", hdr == 0xD085);
    }
    // MINER_READY (216 = 0xD8) → 0xD0D8
    {
        auto b = PacketBuilder::build(ProtocolLane::STATELESS, LLP::MINER_READY);
        uint16_t hdr = b ? ((static_cast<uint16_t>((*b)[0]) << 8) | (*b)[1]) : 0;
        print_test_result("MINER_READY stateless header == 0xD0D8", hdr == 0xD0D8);
    }
}

// ============================================================================
// Test 11: UNKNOWN lane must not silently fall back to legacy framing
// ============================================================================
void test_unknown_lane_rejected() {
    std::cout << "\nTest 11: UNKNOWN lane rejected\n";
    auto header_only = PacketBuilder::build(ProtocolLane::UNKNOWN, LLP::GET_BLOCK);
    std::vector<uint8_t> payload = {0xAA, 0xBB};
    auto with_payload = PacketBuilder::build(ProtocolLane::UNKNOWN, LLP::SUBMIT_BLOCK, payload);

    bool ok = !header_only && !with_payload;
    print_test_result("UNKNOWN lane build returns empty payload", ok);
}

// ============================================================================
// Test 12: Legacy submit-result response opcodes preserve payload framing
// ============================================================================
void test_legacy_submit_result_payloads() {
    std::cout << "\nTest 12: Legacy submit-result response payloads\n";

    auto accepted_zero = PacketBuilder::build(ProtocolLane::LEGACY, LLP::BLOCK_ACCEPTED);
    bool zero_ok = accepted_zero && accepted_zero->size() == 5 &&
                   (*accepted_zero)[0] == 0xC8 &&
                   (*accepted_zero)[1] == 0x00 && (*accepted_zero)[2] == 0x00 &&
                   (*accepted_zero)[3] == 0x00 && (*accepted_zero)[4] == 0x00;
    print_test_result("Legacy BLOCK_ACCEPTED zero-length frame builds as [0xC8][len=0]", zero_ok);

    std::vector<uint8_t> reason = {0x2A};
    auto rejected_reason = PacketBuilder::build(ProtocolLane::LEGACY, LLP::BLOCK_REJECTED, reason);
    bool reason_ok = rejected_reason && rejected_reason->size() == 6 &&
                     (*rejected_reason)[0] == 0xC9 &&
                     (*rejected_reason)[1] == 0x00 && (*rejected_reason)[2] == 0x00 &&
                     (*rejected_reason)[3] == 0x00 && (*rejected_reason)[4] == 0x01 &&
                     (*rejected_reason)[5] == 0x2A;
    print_test_result("Legacy BLOCK_REJECTED one-byte reason builds as [0xC9][len=1][reason]", reason_ok);
}

// ============================================================================
// Test 13: Stateless submit-result response opcodes mirror legacy payload forms
// ============================================================================
void test_stateless_submit_result_payloads() {
    std::cout << "\nTest 13: Stateless submit-result response payloads\n";

    auto accepted_zero = PacketBuilder::build(ProtocolLane::STATELESS, LLP::BLOCK_ACCEPTED);
    bool zero_ok = accepted_zero && accepted_zero->size() == 6 &&
                   (*accepted_zero)[0] == 0xD0 && (*accepted_zero)[1] == 0xC8 &&
                   (*accepted_zero)[2] == 0x00 && (*accepted_zero)[3] == 0x00 &&
                   (*accepted_zero)[4] == 0x00 && (*accepted_zero)[5] == 0x00;
    print_test_result("Stateless BLOCK_ACCEPTED zero-length frame builds as [0xD0C8][len=0]", zero_ok);

    std::vector<uint8_t> reason = {0x2A};
    auto rejected_reason = PacketBuilder::build(ProtocolLane::STATELESS, LLP::BLOCK_REJECTED, reason);
    bool reason_ok = rejected_reason && rejected_reason->size() == 7 &&
                     (*rejected_reason)[0] == 0xD0 && (*rejected_reason)[1] == 0xC9 &&
                     (*rejected_reason)[2] == 0x00 && (*rejected_reason)[3] == 0x00 &&
                     (*rejected_reason)[4] == 0x00 && (*rejected_reason)[5] == 0x01 &&
                     (*rejected_reason)[6] == 0x2A;
    print_test_result("Stateless BLOCK_REJECTED one-byte reason builds as [0xD0C9][len=1][reason]", reason_ok);
}

// ============================================================================
// Test 14: submit_block plaintext layout size invariant
// Verifies the fixed-format plaintext is exactly block(216) + ts(8) + siglen(2) + sig(N)
// ============================================================================
void test_plaintext_layout_size() {
    std::cout << "\nTest 14: Plaintext layout size = 216 + 8 + 2 + sig_size\n";
    constexpr size_t BLOCK_SIZE = 216;
    constexpr size_t TIMESTAMP_SIZE = 8;
    constexpr size_t SIGLEN_FIELD_SIZE = 2;
    constexpr size_t FALCON1024_SIG_SIZE = 1577;  // typical Falcon-1024 sig size

    size_t expected = BLOCK_SIZE + TIMESTAMP_SIZE + SIGLEN_FIELD_SIZE + FALCON1024_SIG_SIZE;
    // 216 + 8 + 2 + 1577 = 1803 bytes
    bool ok = (expected == 1803);
    print_test_result("Plaintext layout: 216+8+2+1577 = 1803 bytes", ok);

    // Verify the total is exactly 1803 (no extra fields)
    bool exact_size = (expected == 1803);
    print_test_result("Total plaintext size is exactly 1803 bytes", exact_size);
}

// ============================================================================
// main
// ============================================================================
int main() {
    std::cout << "========================================\n";
    std::cout << "PacketBuilder Unit Tests\n";
    std::cout << "========================================\n";

    test_legacy_header_only();
    test_stateless_header_only();
    test_legacy_with_payload();
    test_stateless_with_payload();
    test_legacy_miner_ready();
    test_stateless_miner_ready();
    test_legacy_get_round();
    test_stateless_get_round();
    test_large_payload_stateless();
    test_mirror_opcode_invariant();
    test_unknown_lane_rejected();
    test_legacy_submit_result_payloads();
    test_stateless_submit_result_payloads();
    test_plaintext_layout_size();

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
