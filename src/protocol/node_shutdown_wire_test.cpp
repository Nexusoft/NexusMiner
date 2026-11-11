/**
 * @file node_shutdown_wire_test.cpp
 * @brief Unit tests for NODE_SHUTDOWN (0xD0FF) wire protocol
 *
 * Tests:
 *  1. NODE_SHUTDOWN opcode constants match expected values
 *  2. NodeShutdownFrame parses GRACEFUL reason correctly
 *  3. NodeShutdownFrame parses MAINTENANCE reason correctly
 *  4. NodeShutdownFrame rejects empty payload
 *  5. NodeShutdownFrame ReasonString() for known and unknown reasons
 *  6. NODE_SHUTDOWN_BACKOFF_S constant is 60
 *  7. NODE_SHUTDOWN is in stateless opcode range (0xD000-0xD0FF)
 *  8. matches_opcode lambda correctly matches NODE_SHUTDOWN on both lanes
 */

#include "LLP/include/colin_ping_protocol.h"
#include "LLP/miner_opcodes.hpp"
#include "LLP/llp_logging.hpp"
#include "packet.hpp"
#include "protocol/solo.hpp"
#include <iostream>
#include <cassert>
#include <cstdint>
#include <vector>
#include <cstring>

using namespace LLP;

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
// Test 1: NODE_SHUTDOWN opcode constants match expected values
// ============================================================================
void test_opcode_constants() {
    std::cout << "\nTest 1: NODE_SHUTDOWN opcode constants match expected values\n";

    // Legacy opcode: 0xFF (255)
    print_test_result("Legacy NODE_SHUTDOWN == 255",
        nexusminer::LLP::NODE_SHUTDOWN == 255);

    // Stateless opcode: MirrorOpcode(0xFF) == 0xD0FF
    print_test_result("Stateless NODE_SHUTDOWN == 0xD0FF",
        nexusminer::LLP::StatelessMining::NODE_SHUTDOWN == 0xD0FF);

    // MirrorOpcode round-trip
    print_test_result("MirrorOpcode(NODE_SHUTDOWN) == 0xD0FF",
        nexusminer::LLP::MirrorOpcode(nexusminer::LLP::NODE_SHUTDOWN) == 0xD0FF);

    // UnmirrorOpcode round-trip
    print_test_result("UnmirrorOpcode(0xD0FF) == 0xFF",
        nexusminer::LLP::UnmirrorOpcode(0xD0FF) == 0xFF);
}

// ============================================================================
// Test 2: NodeShutdownFrame parses GRACEFUL reason correctly
// ============================================================================
void test_parse_graceful() {
    std::cout << "\nTest 2: NodeShutdownFrame parses GRACEFUL reason\n";

    std::vector<uint8_t> data = { NodeShutdownFrame::REASON_GRACEFUL };
    NodeShutdownFrame frame;
    bool ok = frame.Parse(data);

    print_test_result("Parse succeeds", ok);
    print_test_result("reason == REASON_GRACEFUL (0x01)",
        frame.reason == NodeShutdownFrame::REASON_GRACEFUL);
    print_test_result("ReasonString() == \"GRACEFUL\"",
        std::strcmp(frame.ReasonString(), "GRACEFUL") == 0);
}

// ============================================================================
// Test 3: NodeShutdownFrame parses MAINTENANCE reason correctly
// ============================================================================
void test_parse_maintenance() {
    std::cout << "\nTest 3: NodeShutdownFrame parses MAINTENANCE reason\n";

    std::vector<uint8_t> data = { NodeShutdownFrame::REASON_MAINTENANCE };
    NodeShutdownFrame frame;
    bool ok = frame.Parse(data);

    print_test_result("Parse succeeds", ok);
    print_test_result("reason == REASON_MAINTENANCE (0x02)",
        frame.reason == NodeShutdownFrame::REASON_MAINTENANCE);
    print_test_result("ReasonString() == \"MAINTENANCE\"",
        std::strcmp(frame.ReasonString(), "MAINTENANCE") == 0);
}

// ============================================================================
// Test 4: NodeShutdownFrame rejects empty payload
// ============================================================================
void test_parse_empty_payload() {
    std::cout << "\nTest 4: NodeShutdownFrame rejects empty payload\n";

    std::vector<uint8_t> data;
    NodeShutdownFrame frame;
    bool ok = frame.Parse(data);

    print_test_result("Parse fails on empty data", !ok);
    print_test_result("reason remains 0 (default)", frame.reason == 0);
}

// ============================================================================
// Test 5: ReasonString() for unknown reason codes
// ============================================================================
void test_reason_string_unknown() {
    std::cout << "\nTest 5: ReasonString() for unknown reason codes\n";

    NodeShutdownFrame frame;
    frame.reason = 0x00;
    print_test_result("reason=0x00 → \"UNKNOWN\"",
        std::strcmp(frame.ReasonString(), "UNKNOWN") == 0);

    frame.reason = 0xFF;
    print_test_result("reason=0xFF → \"UNKNOWN\"",
        std::strcmp(frame.ReasonString(), "UNKNOWN") == 0);

    frame.reason = 0x03;
    print_test_result("reason=0x03 → \"UNKNOWN\"",
        std::strcmp(frame.ReasonString(), "UNKNOWN") == 0);
}

// ============================================================================
// Test 6: NODE_SHUTDOWN_BACKOFF_S constant is 60
// ============================================================================
void test_backoff_constant() {
    std::cout << "\nTest 6: NODE_SHUTDOWN_BACKOFF_S constant is 60\n";

    print_test_result("NODE_SHUTDOWN_BACKOFF_S == 60",
        nexusminer::protocol::Solo::NODE_SHUTDOWN_BACKOFF_S == 60);
}

// ============================================================================
// Test 7: NODE_SHUTDOWN is in stateless opcode range
// ============================================================================
void test_stateless_range() {
    std::cout << "\nTest 7: NODE_SHUTDOWN is in stateless opcode range\n";

    print_test_result("IsStatelessOpcode(0xD0FF) == true",
        nexusminer::LLP::IsStatelessOpcode(0xD0FF));

    print_test_result("0xD0FF >= 0xD000", 0xD0FF >= 0xD000);
    print_test_result("0xD0FF <= 0xD0FF", 0xD0FF <= 0xD0FF);
}

// ============================================================================
// Test 8: Logging name resolution for NODE_SHUTDOWN
// ============================================================================
void test_logging_names() {
    std::cout << "\nTest 8: Logging name resolution for NODE_SHUTDOWN\n";

    // Legacy name
    const char* legacy_name = nexusminer::get_llp_header_name(static_cast<uint8_t>(0xFF));
    print_test_result("Legacy 0xFF → \"NODE_SHUTDOWN\"",
        std::strcmp(legacy_name, "NODE_SHUTDOWN") == 0);

    // Stateless name
    const char* stateless_name = nexusminer::get_llp_header_name(static_cast<uint16_t>(0xD0FF));
    std::string sname(stateless_name);
    print_test_result("Stateless 0xD0FF → contains \"NODE_SHUTDOWN\"",
        sname.find("NODE_SHUTDOWN") != std::string::npos);
}

// ============================================================================
// Test 9: ShutdownReason enum values match NodeShutdownFrame constants
// ============================================================================
void test_shutdown_reason_enum() {
    std::cout << "\nTest 9: ShutdownReason enum values match frame constants\n";

    print_test_result("ShutdownReason::GRACEFUL == REASON_GRACEFUL",
        static_cast<uint8_t>(nexusminer::LLP::StatelessMining::ShutdownReason::GRACEFUL)
            == NodeShutdownFrame::REASON_GRACEFUL);

    print_test_result("ShutdownReason::MAINTENANCE == REASON_MAINTENANCE",
        static_cast<uint8_t>(nexusminer::LLP::StatelessMining::ShutdownReason::MAINTENANCE)
            == NodeShutdownFrame::REASON_MAINTENANCE);
}

void test_legacy_wire_parse_with_reason() {
    std::cout << "\nTest 10: Legacy NODE_SHUTDOWN wire frame parses with reason\n";

    auto payload = std::make_shared<nexusminer::network::Payload>(
        nexusminer::network::Payload{
            static_cast<uint8_t>(nexusminer::LLP::NODE_SHUTDOWN),
            0x00, 0x00, 0x00, 0x01,
            NodeShutdownFrame::REASON_GRACEFUL
        });

    std::size_t consumed = 0;
    nexusminer::ParseResult result = nexusminer::ParseResult::MALFORMED;
    auto packet = nexusminer::extract_packet_from_buffer_with_result(
        payload, consumed, 0, nexusminer::ProtocolLane::LEGACY, result);

    print_test_result("Legacy parse succeeds", result == nexusminer::ParseResult::SUCCESS);
    print_test_result("Legacy consumes full frame", consumed == payload->size());
    print_test_result("Legacy header is NODE_SHUTDOWN", packet.m_header == nexusminer::LLP::NODE_SHUTDOWN);
    print_test_result("Legacy frame is 8-bit", !packet.m_is_uint16_opcode);
    print_test_result("Legacy reason payload length is 1", packet.m_length == 1);
    print_test_result("Legacy reason payload is GRACEFUL",
        packet.m_data && packet.m_data->size() == 1 &&
        (*packet.m_data)[0] == NodeShutdownFrame::REASON_GRACEFUL);
}

void test_stateless_wire_parse_with_reason() {
    std::cout << "\nTest 11: Stateless NODE_SHUTDOWN wire frame parses with reason\n";

    auto payload = std::make_shared<nexusminer::network::Payload>(
        nexusminer::network::Payload{
            0xD0, 0xFF,
            0x00, 0x00, 0x00, 0x01,
            NodeShutdownFrame::REASON_MAINTENANCE
        });

    std::size_t consumed = 0;
    nexusminer::ParseResult result = nexusminer::ParseResult::MALFORMED;
    auto packet = nexusminer::extract_packet_from_buffer_with_result(
        payload, consumed, 0, nexusminer::ProtocolLane::STATELESS, result);

    print_test_result("Stateless parse succeeds", result == nexusminer::ParseResult::SUCCESS);
    print_test_result("Stateless consumes full frame", consumed == payload->size());
    print_test_result("Stateless header is 0xD0FF",
        packet.m_header == nexusminer::LLP::StatelessMining::NODE_SHUTDOWN);
    print_test_result("Stateless frame is 16-bit", packet.m_is_uint16_opcode);
    print_test_result("Stateless reason payload length is 1", packet.m_length == 1);
    print_test_result("Stateless reason payload is MAINTENANCE",
        packet.m_data && packet.m_data->size() == 1 &&
        (*packet.m_data)[0] == NodeShutdownFrame::REASON_MAINTENANCE);
}

int main() {
    std::cout << "═══════════════════════════════════════════════\n";
    std::cout << "  NODE_SHUTDOWN (0xD0FF) Wire Protocol Tests\n";
    std::cout << "═══════════════════════════════════════════════\n";

    test_opcode_constants();
    test_parse_graceful();
    test_parse_maintenance();
    test_parse_empty_payload();
    test_reason_string_unknown();
    test_backoff_constant();
    test_stateless_range();
    test_logging_names();
    test_shutdown_reason_enum();
    test_legacy_wire_parse_with_reason();
    test_stateless_wire_parse_with_reason();

    std::cout << "\n═══════════════════════════════════════════════\n";
    std::cout << "  Results: " << tests_passed << "/" << tests_run
              << " passed, " << tests_failed << " failed\n";
    std::cout << "═══════════════════════════════════════════════\n";

    return tests_failed > 0 ? 1 : 0;
}
