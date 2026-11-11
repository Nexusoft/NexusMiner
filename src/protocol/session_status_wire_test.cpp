/**
 * @file session_status_wire_test.cpp
 * @brief Unit tests for SESSION_STATUS / SESSION_STATUS_ACK wire protocol
 *
 * Tests:
 *  1. SessionStatusFrame serializes and round-trips correctly
 *  2. SessionStatusAckFrame parses correctly
 *  3. SessionStatusAckFrame rejects short buffer
 *  4. SESSION_STATUS opcode constants match LLL-TAO values
 *  5. SessionStatusAckFrame helper predicates work correctly
 *  6. IsSessionStatusOpcode() helper function
 */

#include "LLP/include/colin_ping_protocol.h"
#include "protocol/session_ingress_gate.hpp"
#include "protocol_lane.hpp"
#include <iostream>
#include <cassert>
#include <cstdint>
#include <vector>

using namespace LLP;
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
// Test 1: SessionStatusFrame serializes and round-trips correctly
// ============================================================================
void test_session_status_frame_serialize_roundtrip() {
    std::cout << "\nTest 1: SessionStatusFrame serializes and round-trips correctly\n";

    SessionStatusFrame frame;
    frame.session_id   = 0xABCD1234;
    frame.status_flags = SessionStatusOpcodes::MINER_HAS_TEMPLATE
                       | SessionStatusOpcodes::MINER_WORKERS_ACTIVE;

    auto v = frame.Serialize();
    print_test_result("Serialize() produces 8 bytes", v.size() == 8);

    // session_id little-endian
    print_test_result("v[0] == 0x34 (LE LSB of 0xABCD1234)", v[0] == 0x34);
    print_test_result("v[1] == 0x12",                        v[1] == 0x12);
    print_test_result("v[2] == 0xCD",                        v[2] == 0xCD);
    print_test_result("v[3] == 0xAB (LE MSB)",               v[3] == 0xAB);

    // status_flags = MINER_HAS_TEMPLATE|MINER_WORKERS_ACTIVE = 0x06 big-endian
    uint32_t expected_flags = SessionStatusOpcodes::MINER_HAS_TEMPLATE
                            | SessionStatusOpcodes::MINER_WORKERS_ACTIVE;
    print_test_result("v[4] == 0x00 (BE MSB of flags)", v[4] == 0x00);
    print_test_result("v[5] == 0x00",                   v[5] == 0x00);
    print_test_result("v[6] == 0x00",                   v[6] == 0x00);
    print_test_result("v[7] == flags_lsb",
                      v[7] == static_cast<uint8_t>(expected_flags & 0xFF));

    // Round-trip
    SessionStatusFrame f2;
    print_test_result("Parse() returns true on 8-byte buffer", f2.Parse(v));
    print_test_result("Round-trip session_id",   f2.session_id   == frame.session_id);
    print_test_result("Round-trip status_flags", f2.status_flags == frame.status_flags);
}

// ============================================================================
// Test 2: SessionStatusAckFrame parses correctly
// ============================================================================
void test_session_status_ack_parse() {
    std::cout << "\nTest 2: SessionStatusAckFrame parses correctly\n";

    // Manually construct a 16-byte ACK payload
    std::vector<uint8_t> raw = {
        // session_id: 0xDEADBEEF LE
        0xEF, 0xBE, 0xAD, 0xDE,
        // lane_health: PRIMARY | AUTHENTICATED = 0x09 BE
        0x00, 0x00, 0x00, 0x09,
        // uptime: 999 = 0x000003E7 BE
        0x00, 0x00, 0x03, 0xE7,
        // echo flags: MINER_HAS_TEMPLATE = 0x02 BE
        0x00, 0x00, 0x00, 0x02
    };

    SessionStatusAckFrame ack;
    print_test_result("Parse() returns true for 16-byte buffer", ack.Parse(raw));
    print_test_result("session_id == 0xDEADBEEF",
                      ack.session_id == 0xDEADBEEFu);
    print_test_result("lane_health_flags == 0x09",
                      ack.lane_health_flags == 0x09u);
    print_test_result("uptime_seconds == 999",
                      ack.uptime_seconds == 999u);
    print_test_result("status_echo_flags == 0x02",
                      ack.status_echo_flags == 0x02u);

    print_test_result("IsPrimaryAlive() == true",    ack.IsPrimaryAlive());
    print_test_result("IsSecondaryAlive() == false",  !ack.IsSecondaryAlive());
    print_test_result("IsSimLinkActive() == false",   !ack.IsSimLinkActive());
    print_test_result("IsAuthenticated() == true",    ack.IsAuthenticated());
}

// ============================================================================
// Test 3: SessionStatusAckFrame rejects short buffer
// ============================================================================
void test_session_status_ack_rejects_short_buffer() {
    std::cout << "\nTest 3: SessionStatusAckFrame rejects short buffer\n";

    SessionStatusAckFrame ack;
    std::vector<uint8_t> short_buf(15, 0x00);
    print_test_result("Parse() returns false for 15-byte buffer", !ack.Parse(short_buf));

    std::vector<uint8_t> empty_buf;
    print_test_result("Parse() returns false for empty buffer", !ack.Parse(empty_buf));

    std::vector<uint8_t> ok_buf(16, 0x00);
    print_test_result("Parse() returns true for 16-byte buffer", ack.Parse(ok_buf));
}

// ============================================================================
// Test 4: SESSION_STATUS opcode constants match LLL-TAO values
// ============================================================================
void test_session_status_opcode_constants() {
    std::cout << "\nTest 4: SESSION_STATUS opcode constants match LLL-TAO values\n";

    print_test_result("SESSION_STATUS_LEGACY == 219",
                      SessionStatusOpcodes::SESSION_STATUS_LEGACY == 219u);
    print_test_result("SESSION_STATUS_ACK_LEGACY == 220",
                      SessionStatusOpcodes::SESSION_STATUS_ACK_LEGACY == 220u);
    print_test_result("SESSION_STATUS == 0xD0DB",
                      SessionStatusOpcodes::SESSION_STATUS == 0xD0DBu);
    print_test_result("SESSION_STATUS_ACK == 0xD0DC",
                      SessionStatusOpcodes::SESSION_STATUS_ACK == 0xD0DCu);

    print_test_result("REQUEST_PAYLOAD_SIZE == 8",
                      SessionStatusOpcodes::REQUEST_PAYLOAD_SIZE == 8u);
    print_test_result("ACK_PAYLOAD_SIZE == 16",
                      SessionStatusOpcodes::ACK_PAYLOAD_SIZE == 16u);
}

// ============================================================================
// Test 5: SessionStatusAckFrame helper predicates cover all flags
// ============================================================================
void test_session_status_ack_predicates() {
    std::cout << "\nTest 5: SessionStatusAckFrame predicates cover all lane flags\n";

    // lane_health_flags = 0x0F (all four bits set)
    std::vector<uint8_t> raw(16, 0);
    raw[7] = 0x0F;  // BE LSB

    SessionStatusAckFrame ack;
    ack.Parse(raw);
    print_test_result("All flags set: IsPrimaryAlive()",   ack.IsPrimaryAlive());
    print_test_result("All flags set: IsSecondaryAlive()", ack.IsSecondaryAlive());
    print_test_result("All flags set: IsSimLinkActive()",  ack.IsSimLinkActive());
    print_test_result("All flags set: IsAuthenticated()",  ack.IsAuthenticated());

    // lane_health_flags = 0x00 (no flags set)
    raw[7] = 0x00;
    ack.Parse(raw);
    print_test_result("No flags set: IsPrimaryAlive() == false",   !ack.IsPrimaryAlive());
    print_test_result("No flags set: IsSecondaryAlive() == false",  !ack.IsSecondaryAlive());
    print_test_result("No flags set: IsSimLinkActive() == false",   !ack.IsSimLinkActive());
    print_test_result("No flags set: IsAuthenticated() == false",   !ack.IsAuthenticated());
}

// ============================================================================
// Test 6: IsSessionStatusOpcode() helper
// ============================================================================
void test_is_session_status_opcode() {
    std::cout << "\nTest 6: IsSessionStatusOpcode() helper\n";

    print_test_result("IsSessionStatusOpcode(SESSION_STATUS) == true",
                      IsSessionStatusOpcode(SessionStatusOpcodes::SESSION_STATUS));
    print_test_result("IsSessionStatusOpcode(SESSION_STATUS_ACK) == true",
                      IsSessionStatusOpcode(SessionStatusOpcodes::SESSION_STATUS_ACK));
    print_test_result("IsSessionStatusOpcode(0xD0D4) == false",
                      !IsSessionStatusOpcode(0xD0D4));
    print_test_result("IsSessionStatusOpcode(0xD100) == false",
                      !IsSessionStatusOpcode(0xD100));
    print_test_result("IsSessionStatusOpcode(0x0000) == false",
                      !IsSessionStatusOpcode(0x0000));
}

// ============================================================================
// Test 7: SessionStatusFrame PAYLOAD_SIZE constant
// ============================================================================
void test_session_status_frame_payload_size() {
    std::cout << "\nTest 7: SessionStatusFrame PAYLOAD_SIZE constant\n";
    print_test_result("SessionStatusFrame::PAYLOAD_SIZE == 8",
                      SessionStatusFrame::PAYLOAD_SIZE == 8u);
    print_test_result("SessionStatusAckFrame::PAYLOAD_SIZE == 16",
                      SessionStatusAckFrame::PAYLOAD_SIZE == 16u);
}

// ============================================================================
// Test 8: shared ingress gate rejects stale owner generation
// ============================================================================
void test_session_ingress_gate_rejects_stale_owner_generation() {
    std::cout << "\nTest 8: Session ingress gate rejects stale owner generation\n";

    const auto decision = SessionIngressGate::preflight(SessionIngressGate::Input{
        true,               // has_authoritative_session
        true,               // authoritative_authenticated
        SessionId(0x22222222u),        // authoritative_session_id
        SessionEpoch(uint64_t{7}),     // authoritative_session_epoch
        ProtocolLane::STATELESS,  // authoritative_lane
        ProtocolLane::STATELESS,  // packet_lane
        true,               // validate_lane
        false,              // allow_without_active_session
        SessionId(0x22222222u),        // packet_session_id
        SessionEpoch(uint64_t{6}),     // owner_epoch (stale: 6 vs authoritative 7)
        SessionId(0x22222222u)         // owner_session_id
    });

    print_test_result("stale owner epoch is rejected", !decision.allow_processing);
    print_test_result("stale owner epoch is marked stale", decision.drop_as_stale);
    print_test_result("stale owner epoch does not force reauth", !decision.force_reauth);
    print_test_result("stale owner epoch reason mentions ownership epoch",
                      decision.reason.find("ownership epoch") != std::string::npos);
}

// ============================================================================
// Test 9: shared ingress gate — crypto readiness check removed (no-op)
// ============================================================================
void test_session_ingress_gate_requires_crypto_context_when_requested() {
    std::cout << "\nTest 9: Session ingress gate — crypto readiness check removed (no-op)\n";

    // require_crypto_ready is removed from the new design.
    // An authenticated session with matching ownership stamp should pass.
    const auto decision = SessionIngressGate::preflight(SessionIngressGate::Input{
        true,               // has_authoritative_session
        true,               // authoritative_authenticated
        SessionId(0x01020304u),        // authoritative_session_id
        SessionEpoch(uint64_t{3}),     // authoritative_session_epoch
        ProtocolLane::STATELESS,
        ProtocolLane::STATELESS,
        true,               // validate_lane
        false,              // allow_without_active_session
        SessionId(0u),                 // packet_session_id (no mismatch)
        SessionEpoch(uint64_t{3}),     // owner_epoch (matches)
        SessionId(0x01020304u)         // owner_session_id (matches)
    });

    // Feature removed: crypto check no longer blocks processing
    print_test_result("crypto check removed: session passes", decision.allow_processing);
    print_test_result("crypto check removed: no force reauth", !decision.force_reauth);
    print_test_result("crypto check removed: reason is set", !decision.reason.empty());
}

// ============================================================================
// Test 10: shared ingress gate rejects packet session id mismatches
// ============================================================================
void test_session_ingress_gate_rejects_stale_packet_session_id() {
    std::cout << "\nTest 10: Session ingress gate rejects stale packet session ids\n";

    const auto decision = SessionIngressGate::preflight(SessionIngressGate::Input{
        true,               // has_authoritative_session
        true,               // authoritative_authenticated
        SessionId(0x11112222u),        // authoritative_session_id
        SessionEpoch(uint64_t{4}),     // authoritative_session_epoch
        ProtocolLane::STATELESS,
        ProtocolLane::STATELESS,
        true,               // validate_lane
        false,              // allow_without_active_session
        SessionId(0x33334444u),        // packet_session_id (mismatched)
        SessionEpoch(uint64_t{4}),     // owner_epoch
        SessionId(0x11112222u)         // owner_session_id
    });

    print_test_result("stale packet session ID is rejected", !decision.allow_processing);
    print_test_result("stale packet session ID is marked stale", decision.drop_as_stale);
    print_test_result("stale packet session ID marks session degraded", decision.mark_degraded);
    print_test_result("stale packet session ID reason mentions session ID",
                      decision.reason.find("session id") != std::string::npos);
}

// ============================================================================
// Test 11: shared ingress gate rejects owner/session mismatches
// ============================================================================
void test_session_ingress_gate_rejects_stale_owner_session_id() {
    std::cout << "\nTest 11: Session ingress gate rejects stale owner session ids\n";

    const auto decision = SessionIngressGate::preflight(SessionIngressGate::Input{
        true,               // has_authoritative_session
        true,               // authoritative_authenticated
        SessionId(0x01020304u),        // authoritative_session_id
        SessionEpoch(uint64_t{9}),     // authoritative_session_epoch
        ProtocolLane::STATELESS,
        ProtocolLane::STATELESS,
        true,               // validate_lane
        false,              // allow_without_active_session
        SessionId(0x01020304u),        // packet_session_id (matches)
        SessionEpoch(uint64_t{9}),     // owner_epoch (matches)
        SessionId(0xA0B0C0D0u)         // owner_session_id (mismatches)
    });

    print_test_result("stale owner session ID is rejected", !decision.allow_processing);
    print_test_result("stale owner session ID is marked stale", decision.drop_as_stale);
    print_test_result("stale owner session ID reason mentions owner session ID",
                      decision.reason.find("ownership session id") != std::string::npos);
}

// ============================================================================
// Test 12: shared ingress gate enforces lane readiness
// ============================================================================
void test_session_ingress_gate_rejects_lane_mismatch() {
    std::cout << "\nTest 12: Session ingress gate rejects lane mismatches\n";

    const auto decision = SessionIngressGate::preflight(SessionIngressGate::Input{
        true,               // has_authoritative_session
        true,               // authoritative_authenticated
        SessionId(0x0A0B0C0Du),        // authoritative_session_id
        SessionEpoch(uint64_t{2}),     // authoritative_session_epoch
        ProtocolLane::STATELESS,  // authoritative_lane
        ProtocolLane::LEGACY,     // packet_lane (mismatched)
        true,               // validate_lane
        false,              // allow_without_active_session
        SessionId(0x0A0B0C0Du),        // packet_session_id
        SessionEpoch(uint64_t{2}),     // owner_epoch
        SessionId(0x0A0B0C0Du)         // owner_session_id
    });

    print_test_result("lane mismatch is rejected", !decision.allow_processing);
    print_test_result("lane mismatch is not treated as stale", !decision.drop_as_stale);
    print_test_result("lane mismatch marks session degraded", decision.mark_degraded);
    print_test_result("lane mismatch reason mentions lane",
                      decision.reason.find("lane") != std::string::npos);
}

// ============================================================================
// Test 13: shared ingress gate forces reauth when auth is required
// ============================================================================
void test_session_ingress_gate_requires_authenticated_session() {
    std::cout << "\nTest 13: Session ingress gate forces re-auth for unauthenticated sessions\n";

    const auto decision = SessionIngressGate::preflight(SessionIngressGate::Input{
        true,               // has_authoritative_session
        false,              // authoritative_authenticated = false
        SessionId(0x55667788u),        // authoritative_session_id
        SessionEpoch(uint64_t{5}),     // authoritative_session_epoch
        ProtocolLane::STATELESS,
        ProtocolLane::STATELESS,
        true,               // validate_lane
        false,              // allow_without_active_session
        SessionId(0x55667788u),        // packet_session_id
        SessionEpoch(uint64_t{5}),     // owner_epoch
        SessionId(0x55667788u)         // owner_session_id
    });

    print_test_result("unauthenticated authoritative session is rejected", !decision.allow_processing);
    print_test_result("unauthenticated authoritative session forces reauth", decision.force_reauth);
    print_test_result("unauthenticated authoritative session marks session degraded",
                      decision.mark_degraded);
    print_test_result("unauthenticated authoritative session reason mentions authentication",
                      decision.reason.find("not authenticated") != std::string::npos);
}

// ============================================================================
// main
// ============================================================================
int main() {
    std::cout << "========================================\n";
    std::cout << "SESSION_STATUS Wire Protocol Unit Tests\n";
    std::cout << "========================================\n";

    test_session_status_frame_serialize_roundtrip();
    test_session_status_ack_parse();
    test_session_status_ack_rejects_short_buffer();
    test_session_status_opcode_constants();
    test_session_status_ack_predicates();
    test_is_session_status_opcode();
    test_session_status_frame_payload_size();
    test_session_ingress_gate_rejects_stale_owner_generation();
    test_session_ingress_gate_requires_crypto_context_when_requested();
    test_session_ingress_gate_rejects_stale_packet_session_id();
    test_session_ingress_gate_rejects_stale_owner_session_id();
    test_session_ingress_gate_rejects_lane_mismatch();
    test_session_ingress_gate_requires_authenticated_session();

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
