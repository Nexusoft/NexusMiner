/**
 * @file packet_framing_test.cpp
 * @brief Unit test harness for LLP packet framing parser with TCP fragmentation
 * 
 * Tests the accumulator-based packet parser with various TCP fragmentation scenarios:
 * - Partial headers split across multiple receives
 * - Partial length fields split across receives
 * - Partial payload data split across receives
 * - Multiple packets arriving in single receive
 * - Malformed packet detection
 * 
 * Run this test to verify TCP stream handling robustness.
 */

#include "packet.hpp"
#include "protocol_lane.hpp"
#include <iostream>
#include <iomanip>
#include <cassert>
#include <vector>
#include <deque>
#include <sstream>

using namespace nexusminer;

// Test statistics
static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

void print_test_result(const char* name, bool passed) {
    tests_run++;
    if (passed) {
        tests_passed++;
        std::cout << "  [PASS] " << name << std::endl;
    } else {
        tests_failed++;
        std::cout << "  [FAIL] " << name << std::endl;
    }
}

void print_hex(const std::vector<uint8_t>& data, size_t max_bytes = 32) {
    std::ostringstream ss;
    ss << std::hex << std::setfill('0');
    for (size_t i = 0; i < std::min(data.size(), max_bytes); ++i) {
        if (i > 0) ss << " ";
        ss << std::setw(2) << static_cast<unsigned>(data[i]);
    }
    if (data.size() > max_bytes) {
        ss << "...";
    }
    std::cout << "    Data: " << ss.str() << " (" << data.size() << " bytes)" << std::endl;
}

/**
 * Simulates TCP receive accumulator behavior
 */
class TestAccumulator {
public:
    std::deque<uint8_t> buffer;
    
    void feed(const std::vector<uint8_t>& data) {
        buffer.insert(buffer.end(), data.begin(), data.end());
    }
    
    bool parse_one_packet(ProtocolLane lane, Packet& out_packet, ParseResult& out_result) {
        if (buffer.empty()) {
            return false;
        }
        
        // Create view for parsing
        std::vector<uint8_t> view(buffer.begin(), buffer.end());
        auto view_shared = std::make_shared<network::Payload>(std::move(view));
        
        std::size_t bytes_consumed = 0;
        out_packet = extract_packet_from_buffer_with_result(
            view_shared, bytes_consumed, 0, lane, out_result);
        
        if (out_result == ParseResult::SUCCESS) {
            // Remove consumed bytes
            buffer.erase(buffer.begin(), buffer.begin() + bytes_consumed);
            return true;
        } else if (out_result == ParseResult::MALFORMED) {
            buffer.clear(); // Simulate disconnect
            return false;
        }
        
        // NEED_MORE_DATA - keep buffer intact
        return false;
    }
    
    size_t size() const { return buffer.size(); }
    bool empty() const { return buffer.empty(); }
    void clear() { buffer.clear(); }
};

// ============================================================================
// Test Case 1: Complete packet in single receive
// ============================================================================
void test_complete_packet_single_receive() {
    std::cout << "\nTest 1: Complete packet in single receive" << std::endl;
    
    TestAccumulator acc;
    
    // Legacy packet: header=0x01, length=5, data="hello"
    std::vector<uint8_t> complete_packet = {
        0x01,                           // header
        0x00, 0x00, 0x00, 0x05,        // length = 5
        'h', 'e', 'l', 'l', 'o'        // data
    };
    
    acc.feed(complete_packet);
    
    Packet packet;
    ParseResult result;
    bool parsed = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    
    bool test_passed = parsed && 
                       (result == ParseResult::SUCCESS) &&
                       (packet.m_header == 0x01) &&
                       (packet.m_length == 5) &&
                       acc.empty();
    
    print_test_result("Complete packet parsed successfully", test_passed);
}

// ============================================================================
// Test Case 2: Header + length split across receives
// ============================================================================
void test_header_fragmented() {
    std::cout << "\nTest 2: Header + length split across multiple receives" << std::endl;
    
    TestAccumulator acc;
    
    // First receive: header + first 2 bytes of length field (incomplete)
    acc.feed({0x02, 0x00, 0x00});
    
    Packet packet;
    ParseResult result;
    
    bool parsed1 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test1 = !parsed1 && (result == ParseResult::NEED_MORE_DATA) && (acc.size() == 3);
    print_test_result("Incomplete length field triggers NEED_MORE_DATA", test1);
    
    // Second receive: rest of length + data
    acc.feed({0x00, 0x03, 'a', 'b', 'c'});
    
    bool parsed2 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test2 = parsed2 && 
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 0x02) &&
                 (packet.m_length == 3) &&
                 acc.empty();
    print_test_result("Complete packet after split length field", test2);
}

// ============================================================================
// Test Case 3: Payload split across receives
// ============================================================================
void test_payload_fragmented() {
    std::cout << "\nTest 3: Payload split across multiple receives" << std::endl;
    
    TestAccumulator acc;
    
    // First receive: header + length
    acc.feed({0x03, 0x00, 0x00, 0x00, 0x0A}); // length = 10 bytes
    
    Packet packet;
    ParseResult result;
    
    bool parsed1 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test1 = !parsed1 && (result == ParseResult::NEED_MORE_DATA);
    print_test_result("Header+length without payload triggers NEED_MORE_DATA", test1);
    
    // Second receive: first 5 bytes of payload
    acc.feed({'a', 'b', 'c', 'd', 'e'});
    
    bool parsed2 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test2 = !parsed2 && (result == ParseResult::NEED_MORE_DATA);
    print_test_result("Partial payload triggers NEED_MORE_DATA", test2);
    
    // Third receive: remaining 5 bytes
    acc.feed({'f', 'g', 'h', 'i', 'j'});
    
    bool parsed3 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test3 = parsed3 && 
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_length == 10) &&
                 acc.empty();
    print_test_result("Complete payload after fragmentation", test3);
}

// ============================================================================
// Test Case 4: Multiple packets in single receive
// ============================================================================
void test_multiple_packets_single_receive() {
    std::cout << "\nTest 4: Multiple packets in single receive" << std::endl;
    
    TestAccumulator acc;
    
    // Two complete packets back-to-back
    std::vector<uint8_t> two_packets = {
        // Packet 1: header=0x04, length=3, data="xyz"
        0x04, 0x00, 0x00, 0x00, 0x03, 'x', 'y', 'z',
        // Packet 2: header=0x05, length=2, data="ab"
        0x05, 0x00, 0x00, 0x00, 0x02, 'a', 'b'
    };
    
    acc.feed(two_packets);
    
    // Parse first packet
    Packet packet1;
    ParseResult result1;
    bool parsed1 = acc.parse_one_packet(ProtocolLane::LEGACY, packet1, result1);
    bool test1 = parsed1 && (packet1.m_header == 0x04) && (packet1.m_length == 3);
    print_test_result("First packet parsed from batch", test1);
    
    // Parse second packet
    Packet packet2;
    ParseResult result2;
    bool parsed2 = acc.parse_one_packet(ProtocolLane::LEGACY, packet2, result2);
    bool test2 = parsed2 && (packet2.m_header == 0x05) && (packet2.m_length == 2) && acc.empty();
    print_test_result("Second packet parsed from batch", test2);
}

// ============================================================================
// Test Case 5: Stateless lane with 16-bit header fragmentation
// ============================================================================
void test_stateless_header_fragmented() {
    std::cout << "\nTest 5: Stateless lane - 16-bit header fragmented" << std::endl;
    
    TestAccumulator acc;
    
    // First receive: only first byte of 16-bit header
    acc.feed({0xD0});
    
    Packet packet;
    ParseResult result;
    
    bool parsed1 = acc.parse_one_packet(ProtocolLane::STATELESS, packet, result);
    bool test1 = !parsed1 && (result == ParseResult::NEED_MORE_DATA) && (acc.size() == 1);
    print_test_result("Partial 16-bit header triggers NEED_MORE_DATA", test1);
    
    // Second receive: second byte of header + partial length
    acc.feed({0x01, 0x00, 0x00});
    
    bool parsed2 = acc.parse_one_packet(ProtocolLane::STATELESS, packet, result);
    bool test2 = !parsed2 && (result == ParseResult::NEED_MORE_DATA);
    print_test_result("Incomplete length field in stateless", test2);
    
    // Third receive: complete length + data
    acc.feed({0x00, 0x04, 't', 'e', 's', 't'});
    
    bool parsed3 = acc.parse_one_packet(ProtocolLane::STATELESS, packet, result);
    bool test3 = parsed3 && 
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 0xD001) &&
                 (packet.m_length == 4) &&
                 acc.empty();
    print_test_result("Stateless packet after multiple receives", test3);
}

// ============================================================================
// Test Case 6: Malformed packet - unreasonably large length
// ============================================================================
void test_malformed_huge_length() {
    std::cout << "\nTest 6: Malformed packet - unreasonably large length" << std::endl;
    
    TestAccumulator acc;
    
    // Packet with 100MB length (exceeds 10MB limit)
    std::vector<uint8_t> malformed = {
        0x10,                           // header
        0x06, 0x40, 0x00, 0x00         // length = 100MB
    };
    
    acc.feed(malformed);
    
    Packet packet;
    ParseResult result;
    bool parsed = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    
    bool test_passed = !parsed && 
                       (result == ParseResult::MALFORMED) &&
                       acc.empty(); // Buffer cleared on malformed
    
    print_test_result("Huge length detected as malformed", test_passed);
}

void test_packet_constructor_rejects_huge_length() {
    std::cout << "\nTest 6b: Packet constructor rejects unreasonably large length" << std::endl;

    auto malformed = std::make_shared<network::Payload>(network::Payload{
        0x10,                           // header
        0x06, 0x40, 0x00, 0x00         // length = 100MB
    });

    Packet packet(malformed);

    bool test_passed = !packet.m_is_valid &&
                       packet.m_length == 0 &&
                       !packet.m_data;

    print_test_result("Constructor huge length detected as malformed", test_passed);
}

// ============================================================================
// Test Case 7: Malformed - invalid stateless opcode
// ============================================================================
void test_malformed_invalid_stateless_opcode() {
    std::cout << "\nTest 7: Malformed - invalid stateless opcode" << std::endl;
    
    TestAccumulator acc;
    
    // Invalid opcode for stateless lane (not in 0xD000-0xD0FF range)
    std::vector<uint8_t> malformed = {
        0x00, 0x01,                    // header = 0x0001 (not stateless)
        0x00, 0x00, 0x00, 0x00         // length = 0
    };
    
    acc.feed(malformed);
    
    Packet packet;
    ParseResult result;
    bool parsed = acc.parse_one_packet(ProtocolLane::STATELESS, packet, result);
    
    bool test_passed = !parsed && 
                       (result == ParseResult::MALFORMED) &&
                       acc.empty();
    
    print_test_result("Invalid stateless opcode detected", test_passed);
}

// ============================================================================
// Test Case 8: Complex scenario - mixed fragmentation and batching
// ============================================================================
void test_complex_mixed_scenario() {
    std::cout << "\nTest 8: Complex scenario - mixed fragmentation and batching" << std::endl;
    
    TestAccumulator acc;
    
    // Simulate realistic TCP stream:
    // Receive 1: Partial packet 1
    acc.feed({0x20, 0x00, 0x00});
    
    Packet packet;
    ParseResult result;
    bool parsed1 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test1 = !parsed1 && (result == ParseResult::NEED_MORE_DATA);
    print_test_result("Partial first packet", test1);
    
    // Receive 2: Rest of packet 1 + complete packet 2 + partial packet 3
    acc.feed({
        0x00, 0x02, 'A', 'B',          // Complete packet 1
        0x21, 0x00, 0x00, 0x00, 0x01, 'X',  // Complete packet 2
        0x22, 0x00                     // Partial packet 3
    });
    
    // Parse packet 1
    bool parsed2 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test2 = parsed2 && (packet.m_header == 0x20) && (packet.m_length == 2);
    print_test_result("First packet completed", test2);
    
    // Parse packet 2
    bool parsed3 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test3 = parsed3 && (packet.m_header == 0x21) && (packet.m_length == 1);
    print_test_result("Second packet completed", test3);
    
    // Try to parse packet 3 (incomplete)
    bool parsed4 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test4 = !parsed4 && (result == ParseResult::NEED_MORE_DATA) && (acc.size() == 2);
    print_test_result("Third packet incomplete", test4);
    
    // Receive 3: Complete packet 3
    acc.feed({0x00, 0x00, 0x03, 'a', 'b', 'c'});
    
    bool parsed5 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test5 = parsed5 && (packet.m_header == 0x22) && (packet.m_length == 3) && acc.empty();
    print_test_result("Third packet completed", test5);
}

// ============================================================================
// Test Case 9: Byte-by-byte feeding (extreme fragmentation)
// ============================================================================
void test_byte_by_byte_feeding() {
    std::cout << "\nTest 9: Extreme fragmentation - byte-by-byte feeding" << std::endl;
    
    TestAccumulator acc;
    
    // Build a packet byte by byte
    // Start with at least 2 bytes to avoid header-only packet interpretation
    std::vector<uint8_t> complete_packet = {
        0x30, 0x00, 0x00, 0x00, 0x05, 'h', 'e', 'l', 'l', 'o'
    };
    
    Packet packet;
    ParseResult result;
    
    // Feed first two bytes together (header + start of length)
    acc.feed({complete_packet[0], complete_packet[1]});
    bool parsed0 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    if (parsed0 || result != ParseResult::NEED_MORE_DATA) {
        print_test_result("Byte-by-byte feeding - initial state", false);
        return;
    }
    
    // Feed remaining bytes one at a time
    for (size_t i = 2; i < complete_packet.size() - 1; ++i) {
        acc.feed({complete_packet[i]});
        bool parsed = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
        
        if (parsed || result != ParseResult::NEED_MORE_DATA) {
            print_test_result("Byte-by-byte feeding - intermediate", false);
            return;
        }
    }
    
    // Feed last byte
    acc.feed({complete_packet.back()});
    bool parsed = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    
    bool test_passed = parsed && 
                       (result == ParseResult::SUCCESS) &&
                       (packet.m_header == 0x30) &&
                       (packet.m_length == 5) &&
                       acc.empty();
    
    print_test_result("Byte-by-byte feeding completes successfully", test_passed);
}

// ============================================================================
// Test Case 10: Zero-length payload
// ============================================================================
void test_zero_length_payload() {
    std::cout << "\nTest 10: Zero-length payload packet" << std::endl;
    
    TestAccumulator acc;
    
    // Packet with zero-length payload
    std::vector<uint8_t> zero_packet = {
        0x40, 0x00, 0x00, 0x00, 0x00   // header + length=0
    };
    
    acc.feed(zero_packet);
    
    Packet packet;
    ParseResult result;
    bool parsed = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    
    bool test_passed = parsed && 
                       (result == ParseResult::SUCCESS) &&
                       (packet.m_header == 0x40) &&
                       (packet.m_length == 0) &&
                       acc.empty();
    
    print_test_result("Zero-length payload handled correctly", test_passed);
}

// ============================================================================
// Test Case 11: Push notification packets (PRIME_BLOCK_AVAILABLE / HASH_BLOCK_AVAILABLE)
// These carry 12-byte payloads on the legacy lane and must be parsed correctly
// ============================================================================
void test_push_notification_legacy_lane() {
    std::cout << "\nTest 11: Push notification packets on legacy lane (12-byte and 148-byte payloads)" << std::endl;
    
    TestAccumulator acc;
    Packet packet;
    ParseResult result;
    
    // PRIME_BLOCK_AVAILABLE (217 = 0xD9) with 12-byte payload
    // Payload: unified_height(4) + prime_height(4) + difficulty(4)
    std::vector<uint8_t> prime_notification = {
        217,                                // header = PRIME_BLOCK_AVAILABLE
        0x00, 0x00, 0x00, 0x0C,            // length = 12
        0x00, 0x01, 0x00, 0x00,            // unified_height = 65536
        0x00, 0x00, 0xFF, 0x00,            // prime_height = 65280
        0x00, 0x00, 0x00, 0x1E             // difficulty = 30
    };
    
    acc.feed(prime_notification);
    bool parsed1 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    
    bool test1 = parsed1 &&
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 217) &&
                 (packet.m_length == 12) &&
                 (packet.m_data && packet.m_data->size() == 12) &&
                 acc.empty();
    print_test_result("PRIME_BLOCK_AVAILABLE (217) with 12-byte payload", test1);
    
    // Verify is_auth_packet() returns true for PRIME_BLOCK_AVAILABLE
    Packet prime_pkt(static_cast<uint8_t>(217));
    bool test1b = prime_pkt.is_auth_packet();
    print_test_result("is_auth_packet() returns true for PRIME_BLOCK_AVAILABLE (217)", test1b);
    
    // HASH_BLOCK_AVAILABLE (218 = 0xDA) with 12-byte payload
    std::vector<uint8_t> hash_notification = {
        218,                                // header = HASH_BLOCK_AVAILABLE
        0x00, 0x00, 0x00, 0x0C,            // length = 12
        0x00, 0x02, 0x00, 0x00,            // unified_height = 131072
        0x00, 0x01, 0x00, 0x00,            // hash_height = 65536
        0x00, 0x00, 0x00, 0x20             // difficulty = 32
    };
    
    acc.feed(hash_notification);
    bool parsed2 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    
    bool test2 = parsed2 &&
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 218) &&
                 (packet.m_length == 12) &&
                 (packet.m_data && packet.m_data->size() == 12) &&
                 acc.empty();
    print_test_result("HASH_BLOCK_AVAILABLE (218) with 12-byte payload", test2);
    
    // Verify is_auth_packet() returns true for HASH_BLOCK_AVAILABLE
    Packet hash_pkt(static_cast<uint8_t>(218));
    bool test2b = hash_pkt.is_auth_packet();
    print_test_result("is_auth_packet() returns true for HASH_BLOCK_AVAILABLE (218)", test2b);

    // PRIME_BLOCK_AVAILABLE (217 = 0xD9) with 148-byte full-picture payload
    // Layout: unified(4) + prime(4) + difficulty(4) + hash_height(4) + stake_height(4) + hashBestChain(128)
    std::vector<uint8_t> prime_notification_148(5 + 148, 0x00);  // 1-byte header + 4-byte length + 148-byte payload
    prime_notification_148[0] = 217;                              // header = PRIME_BLOCK_AVAILABLE
    prime_notification_148[1] = 0x00;
    prime_notification_148[2] = 0x00;
    prime_notification_148[3] = 0x00;
    prime_notification_148[4] = 148;                             // length = 148
    // [0-3]  unified_height = 6500000
    prime_notification_148[5]  = 0x00; prime_notification_148[6]  = 0x63;
    prime_notification_148[7]  = 0x4E; prime_notification_148[8]  = 0xA0;
    // [4-7]  prime_height = 2300000
    prime_notification_148[9]  = 0x00; prime_notification_148[10] = 0x23;
    prime_notification_148[11] = 0x12; prime_notification_148[12] = 0x60;
    // [8-11] difficulty = 0x0422E6FC
    prime_notification_148[13] = 0x04; prime_notification_148[14] = 0x22;
    prime_notification_148[15] = 0xE6; prime_notification_148[16] = 0xFC;
    // [12-15] hash_height = 900000
    prime_notification_148[17] = 0x00; prime_notification_148[18] = 0x0D;
    prime_notification_148[19] = 0xBB; prime_notification_148[20] = 0xA0;
    // [16-19] stake_height = 400000
    prime_notification_148[21] = 0x00; prime_notification_148[22] = 0x06;
    prime_notification_148[23] = 0x1A; prime_notification_148[24] = 0x80;
    // [20-147] hashBestChain: fill with 0xAB pattern for recognizability
    for (int i = 25; i < 5 + 148; ++i) {
        prime_notification_148[i] = 0xAB;
    }

    acc.feed(prime_notification_148);
    bool parsed3 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);

    bool test3 = parsed3 &&
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 217) &&
                 (packet.m_length == 148) &&
                 (packet.m_data && packet.m_data->size() == 148) &&
                 acc.empty();
    print_test_result("PRIME_BLOCK_AVAILABLE (217) with 148-byte full-picture payload", test3);

    // HASH_BLOCK_AVAILABLE (218 = 0xDA) with 148-byte full-picture payload
    std::vector<uint8_t> hash_notification_148(5 + 148, 0x00);
    hash_notification_148[0] = 218;                              // header = HASH_BLOCK_AVAILABLE
    hash_notification_148[1] = 0x00;
    hash_notification_148[2] = 0x00;
    hash_notification_148[3] = 0x00;
    hash_notification_148[4] = 148;                             // length = 148
    // fill payload bytes with distinct sentinel values
    for (int i = 5; i < 5 + 148; ++i) {
        hash_notification_148[i] = static_cast<uint8_t>((i - 5) & 0xFF);
    }

    acc.feed(hash_notification_148);
    bool parsed4 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);

    bool test4 = parsed4 &&
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 218) &&
                 (packet.m_length == 148) &&
                 (packet.m_data && packet.m_data->size() == 148) &&
                 acc.empty();
    print_test_result("HASH_BLOCK_AVAILABLE (218) with 148-byte full-picture payload", test4);
}

// ============================================================================
// Test Case 12: is_auth_packet() covers full 206-218 range
// ============================================================================
void test_is_auth_packet_full_range() {
    std::cout << "\nTest 12: is_auth_packet() covers full 206-218 range" << std::endl;
    
    // All opcodes 206-218 should return true
    for (uint16_t opcode = 206; opcode <= 218; ++opcode) {
        Packet pkt(static_cast<uint8_t>(opcode));
        bool in_range = pkt.is_auth_packet();
        std::string name = "is_auth_packet() returns true for opcode " + std::to_string(opcode);
        print_test_result(name.c_str(), in_range);
    }
    
    // Opcodes just outside the range should return false
    Packet below(static_cast<uint8_t>(205));
    print_test_result("is_auth_packet() returns false for opcode 205", !below.is_auth_packet());
    
    Packet above(static_cast<uint8_t>(219));
    print_test_result("is_auth_packet() returns false for opcode 219", !above.is_auth_packet());
}

// ============================================================================
// Test Case 13: MINER_READY (216) is header-only and validates correctly
// ============================================================================
void test_miner_ready_header_only() {
    std::cout << "\nTest 13: MINER_READY (216) is header-only and validates correctly" << std::endl;
    
    // MINER_READY is header-only (no payload)
    Packet ready_pkt(static_cast<uint8_t>(216));
    
    // Should be in is_auth_packet() range
    bool test1 = ready_pkt.is_auth_packet();
    print_test_result("MINER_READY (216) is in is_auth_packet() range", test1);
    
    // Should still validate as valid (header-only request)
    bool test2 = ready_pkt.is_valid();
    print_test_result("MINER_READY (216) with m_length=0 validates as valid", test2);
}

// ============================================================================
// Test Case 14: Legacy data packet single-byte fragmentation
// When only 1 byte of a data packet (opcode < 128) arrives, it should NOT
// be treated as a header-only packet - NEED_MORE_DATA should be returned
// ============================================================================
void test_legacy_data_packet_single_byte() {
    std::cout << "\nTest 14: Legacy data packet single-byte fragmentation" << std::endl;
    
    TestAccumulator acc;
    Packet packet;
    ParseResult result;
    
    // Feed only 1 byte of a BLOCK_DATA (opcode 0) packet
    acc.feed({0x00});
    
    bool parsed1 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test1 = !parsed1 && (result == ParseResult::NEED_MORE_DATA) && (acc.size() == 1);
    print_test_result("Single byte of BLOCK_DATA triggers NEED_MORE_DATA", test1);
    
    // Feed remaining bytes (length + payload)
    acc.feed({0x00, 0x00, 0x00, 0x05, 'h', 'e', 'l', 'l', 'o'});
    
    bool parsed2 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test2 = parsed2 && 
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 0x00) &&
                 (packet.m_length == 5) &&
                 acc.empty();
    print_test_result("Complete BLOCK_DATA after fragmented single byte", test2);
    
    // Test with SUBMIT_BLOCK (opcode 1) - also a data packet
    acc.feed({0x01});
    
    bool parsed3 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test3 = !parsed3 && (result == ParseResult::NEED_MORE_DATA) && (acc.size() == 1);
    print_test_result("Single byte of SUBMIT_BLOCK triggers NEED_MORE_DATA", test3);
    
    acc.feed({0x00, 0x00, 0x00, 0x03, 'a', 'b', 'c'});
    
    bool parsed4 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test4 = parsed4 && 
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 0x01) &&
                 (packet.m_length == 3) &&
                 acc.empty();
    print_test_result("Complete SUBMIT_BLOCK after fragmented single byte", test4);
}

// ============================================================================
// Test Case 15: Legacy header-only opcodes require explicit zero-length frames.
// ============================================================================
void test_legacy_header_only_single_byte() {
    std::cout << "\nTest 15: Legacy header-only opcode framing" << std::endl;
    
    TestAccumulator acc;
    Packet packet;
    ParseResult result;
    
    // GET_BLOCK (129) now waits for the explicit zero-length field when still ambiguous.
    acc.feed({129});
    
    bool parsed1 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test1 = !parsed1 && (result == ParseResult::NEED_MORE_DATA) && (acc.size() == 1);
    print_test_result("GET_BLOCK (129) waits for zero-length framing bytes while ambiguous", test1);

    acc.feed({0x00, 0x00, 0x00, 0x00});
    bool parsed2 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test2 = parsed2 &&
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 129) &&
                 (packet.m_length == 0) &&
                 acc.empty();
    print_test_result("GET_BLOCK (129) explicit zero-length frame parsed correctly", test2);
    
    // Bare-header fallback is no longer allowed: a header-only opcode with a
    // non-zero explicit length is malformed.
    acc.feed({253, 0x00, 0x00, 0x00, 0x01});
    
    bool parsed3 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test3 = !parsed3 &&
                 (result == ParseResult::MALFORMED) &&
                 acc.empty();
    print_test_result("PING (253) bare-header form is rejected as malformed", test3);
    acc.clear();
    
    // MINER_READY (216) is still accepted as a framed zero-length header-only packet.
    acc.feed({216, 0x00, 0x00, 0x00, 0x00});
    
    bool parsed4 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test4 = parsed4 && 
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 216) &&
                 (packet.m_length == 0) &&
                 acc.empty();
    print_test_result("MINER_READY (216) explicit zero-length frame parsed correctly", test4);
}

// ============================================================================
// Test Case 16: Legacy auth packet single-byte fragmentation
// Auth packets (206-218, except MINER_READY) always have payload,
// so a single byte should trigger NEED_MORE_DATA
// ============================================================================
void test_legacy_auth_packet_single_byte() {
    std::cout << "\nTest 16: Legacy auth packet single-byte fragmentation" << std::endl;
    
    TestAccumulator acc;
    Packet packet;
    ParseResult result;
    
    // MINER_AUTH_CHALLENGE (208) - has payload
    acc.feed({208});
    
    bool parsed1 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test1 = !parsed1 && (result == ParseResult::NEED_MORE_DATA) && (acc.size() == 1);
    print_test_result("Single byte of MINER_AUTH_CHALLENGE triggers NEED_MORE_DATA", test1);
    
    // Complete the packet
    acc.feed({0x00, 0x00, 0x00, 0x04, 0xAA, 0xBB, 0xCC, 0xDD});
    
    bool parsed2 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test2 = parsed2 && 
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 208) &&
                 (packet.m_length == 4) &&
                 acc.empty();
    print_test_result("Complete MINER_AUTH_CHALLENGE after single byte", test2);
}

// ============================================================================
// Test Case 17: Stateless data packet two-byte fragmentation
// When only 2 bytes of a stateless data packet arrive, NEED_MORE_DATA
// should be returned (not treated as header-only)
// ============================================================================
void test_stateless_data_packet_two_byte() {
    std::cout << "\nTest 17: Stateless data packet two-byte fragmentation" << std::endl;
    
    TestAccumulator acc;
    Packet packet;
    ParseResult result;
    
    // STATELESS_SUBMIT_BLOCK (0xD001) - has payload, only 2 bytes arrive
    acc.feed({0xD0, 0x01});
    
    bool parsed1 = acc.parse_one_packet(ProtocolLane::STATELESS, packet, result);
    bool test1 = !parsed1 && (result == ParseResult::NEED_MORE_DATA) && (acc.size() == 2);
    print_test_result("Two bytes of STATELESS_SUBMIT_BLOCK triggers NEED_MORE_DATA", test1);
    
    // Complete the packet
    acc.feed({0x00, 0x00, 0x00, 0x03, 'x', 'y', 'z'});
    
    bool parsed2 = acc.parse_one_packet(ProtocolLane::STATELESS, packet, result);
    bool test2 = parsed2 && 
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 0xD001) &&
                 (packet.m_length == 3) &&
                 acc.empty();
    print_test_result("Complete STATELESS_SUBMIT_BLOCK after fragmentation", test2);
}

// ============================================================================
// Test Case 18: Stateless header-only opcodes require explicit zero-length frames.
// NOTE: GET_BLOCK (0xD081) is NOT header-only on stateless (template push).
// ============================================================================
void test_stateless_header_only_two_byte() {
    std::cout << "\nTest 18: Stateless header-only opcode framing" << std::endl;
    
    TestAccumulator acc;
    Packet packet;
    ParseResult result;
    
    // STATELESS_GET_BLOCK (0xD081) is NOT header-only on stateless lane
    // (node sends 228-byte template push via this opcode)
    acc.feed({0xD0, 0x81});
    
    bool parsed1 = acc.parse_one_packet(ProtocolLane::STATELESS, packet, result);
    bool test1 = !parsed1 && 
                 (result == ParseResult::NEED_MORE_DATA) &&
                 (acc.size() == 2);
    print_test_result("STATELESS_GET_BLOCK (0xD081) is NOT header-only (needs payload)", test1);
    acc.clear();
    
    // STATELESS MINER_READY (0xD0D8) waits until the zero-length field is disambiguated.
    acc.feed({0xD0, 0xD8});
    
    bool parsed2 = acc.parse_one_packet(ProtocolLane::STATELESS, packet, result);
    bool test2 = !parsed2 &&
                 (result == ParseResult::NEED_MORE_DATA) &&
                 (acc.size() == 2);
    print_test_result("STATELESS_MINER_READY (0xD0D8) waits for zero-length framing bytes while ambiguous", test2);

    acc.feed({0x00, 0x00, 0x00, 0x00});
    bool parsed3 = acc.parse_one_packet(ProtocolLane::STATELESS, packet, result);
    bool test3 = parsed3 &&
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 0xD0D8) &&
                 (packet.m_length == 0) &&
                 acc.empty();
    print_test_result("STATELESS_MINER_READY (0xD0D8) explicit zero-length frame parsed correctly", test3);

    // Bare-header fallback is no longer allowed on the stateless lane either.
    acc.feed({0xD0, 0xD8, 0xD0});
    bool parsed4 = acc.parse_one_packet(ProtocolLane::STATELESS, packet, result);
    bool test4 = !parsed4 &&
                 (result == ParseResult::NEED_MORE_DATA) &&
                 (acc.size() == 3);
    print_test_result("STATELESS_MINER_READY (0xD0D8) bare-header form waits/rejects instead of parsing", test4);
    acc.clear();
}

// ============================================================================
// Test Case 18b: Stateless submit-result opcodes require explicit zero/one-length frames.
// ============================================================================
void test_stateless_submit_result_compat_forms() {
    std::cout << "\nTest 18b: Stateless compat submit-result opcode wire forms" << std::endl;

    TestAccumulator acc;
    Packet packet;
    ParseResult result;

    // Header-only BLOCK_ACCEPTED_COMPAT (0xD002) is no longer accepted without len4.
    acc.feed({0xD0, 0x02});
    bool parsed1 = acc.parse_one_packet(ProtocolLane::STATELESS, packet, result);
    bool test1 = !parsed1 &&
                 (result == ParseResult::NEED_MORE_DATA) &&
                 (acc.size() == 2);
    print_test_result("BLOCK_ACCEPTED_COMPAT bare header waits for required len4", test1);
    acc.clear();

    // Explicit framed zero-length BLOCK_ACCEPTED_COMPAT
    acc.feed({0xD0, 0x02, 0x00, 0x00, 0x00, 0x00});
    bool parsed2 = acc.parse_one_packet(ProtocolLane::STATELESS, packet, result);
    bool test2 = parsed2 &&
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 0xD002) &&
                 (packet.m_length == 0) &&
                 acc.empty();
    print_test_result("BLOCK_ACCEPTED_COMPAT explicit zero-length frame parsed", test2);

    // A bare compat header followed immediately by another packet is malformed
    // now because the required zero-length field is missing.
    acc.feed({0xD0, 0x02, 0xD0, 0xD8, 0x00, 0x00});
    bool parsed3 = acc.parse_one_packet(ProtocolLane::STATELESS, packet, result);
    bool test3 = !parsed3 &&
                 (result == ParseResult::MALFORMED) &&
                 acc.empty();
    print_test_result("Bare compat header followed by another packet is malformed", test3);

    acc.feed({0xD0, 0x02, 0x00, 0x00, 0x00, 0x00,
              0xD0, 0xD8, 0x00, 0x00, 0x00, 0x00});
    bool parsed4 = acc.parse_one_packet(ProtocolLane::STATELESS, packet, result);
    bool parsed5 = acc.parse_one_packet(ProtocolLane::STATELESS, packet, result);
    bool test4 = parsed4 && parsed5 &&
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 0xD0D8) &&
                 (packet.m_length == 0) &&
                 acc.empty();
    print_test_result("Adjacent framed compat + MINER_READY zero-length packets parse correctly", test4);

    // BLOCK_REJECTED_COMPAT supports zero-length and 1-byte reason payload forms.
    acc.feed({0xD0, 0x03, 0x00, 0x00, 0x00, 0x00});
    bool parsed6 = acc.parse_one_packet(ProtocolLane::STATELESS, packet, result);
    bool test5 = parsed6 &&
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 0xD003) &&
                 (packet.m_length == 0) &&
                 acc.empty();
    print_test_result("BLOCK_REJECTED_COMPAT explicit zero-length frame parsed", test5);

    acc.feed({0xD0, 0x03, 0x00, 0x00, 0x00, 0x01, 0x04});
    bool parsed7 = acc.parse_one_packet(ProtocolLane::STATELESS, packet, result);
    bool test6 = parsed7 &&
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 0xD003) &&
                 (packet.m_length == 1) &&
                 (packet.m_data != nullptr) &&
                 ((*packet.m_data)[0] == 0x04) &&
                 acc.empty();
    print_test_result("BLOCK_REJECTED_COMPAT one-byte reason frame parsed", test6);
}

// ============================================================================
// Test Case 19: NEW_ROUND (204) with 12-byte payload
// NEW_ROUND carries payload: [unified_height][channel_height][difficulty]
// ============================================================================
void test_new_round_with_payload() {
    std::cout << "\nTest 19: NEW_ROUND (204) with 12-byte payload" << std::endl;
    
    TestAccumulator acc;
    Packet packet;
    ParseResult result;
    
    // NEW_ROUND (204) with 12-byte payload: height=6500000, channel=6499900, difficulty=0x1E00FFFF
    std::vector<uint8_t> new_round_packet = {
        204,                                // header (NEW_ROUND)
        0x00, 0x00, 0x00, 0x0C,           // length = 12
        0x00, 0x63, 0x32, 0x20,           // unified_height = 6500000 (big-endian)
        0x00, 0x63, 0x31, 0xBC,           // channel_height = 6499900 (big-endian)
        0x1E, 0x00, 0xFF, 0xFF            // difficulty = 0x1E00FFFF (big-endian)
    };
    
    acc.feed(new_round_packet);
    
    bool parsed = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test1 = parsed && 
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 204) &&
                 (packet.m_length == 12) &&
                 (packet.m_data != nullptr) &&
                 (packet.m_data->size() == 12) &&
                 acc.empty();
    print_test_result("NEW_ROUND 12-byte payload parsed correctly", test1);
    
    // NEW_ROUND single byte should trigger NEED_MORE_DATA (not treated as header-only)
    acc.feed({204});
    
    bool parsed2 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test2 = !parsed2 && (result == ParseResult::NEED_MORE_DATA) && (acc.size() == 1);
    print_test_result("NEW_ROUND single byte triggers NEED_MORE_DATA", test2);
    acc.clear();
}

// ============================================================================
// Test Case 20: OLD_ROUND (205) with 12-byte payload
// OLD_ROUND carries payload: [unified_height][channel_height][difficulty]
// ============================================================================
void test_old_round_with_payload() {
    std::cout << "\nTest 20: OLD_ROUND (205) with 12-byte payload" << std::endl;
    
    TestAccumulator acc;
    Packet packet;
    ParseResult result;
    
    // OLD_ROUND (205) with 12-byte payload
    std::vector<uint8_t> old_round_packet = {
        205,                                // header (OLD_ROUND)
        0x00, 0x00, 0x00, 0x0C,           // length = 12
        0x00, 0x63, 0x32, 0x20,           // unified_height = 6500000 (big-endian)
        0x00, 0x63, 0x31, 0xBC,           // channel_height = 6499900 (big-endian)
        0x1E, 0x00, 0xFF, 0xFF            // difficulty = 0x1E00FFFF (big-endian)
    };
    
    acc.feed(old_round_packet);
    
    bool parsed = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test1 = parsed && 
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 205) &&
                 (packet.m_length == 12) &&
                 (packet.m_data != nullptr) &&
                 (packet.m_data->size() == 12) &&
                 acc.empty();
    print_test_result("OLD_ROUND 12-byte payload parsed correctly", test1);
    
    // OLD_ROUND single byte should trigger NEED_MORE_DATA
    acc.feed({205});
    
    bool parsed2 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test2 = !parsed2 && (result == ParseResult::NEED_MORE_DATA) && (acc.size() == 1);
    print_test_result("OLD_ROUND single byte triggers NEED_MORE_DATA", test2);
    acc.clear();
}

// ============================================================================
// Test Case 21: NEW_ROUND/OLD_ROUND with legacy 16-byte payload
// Legacy lane accepts 16-byte format: [unified][prime][hash][stake]
// ============================================================================
void test_round_legacy_16byte_payload() {
    std::cout << "\nTest 21: NEW_ROUND/OLD_ROUND with legacy 16-byte payload" << std::endl;
    
    TestAccumulator acc;
    Packet packet;
    ParseResult result;
    
    // NEW_ROUND with 16-byte legacy payload
    std::vector<uint8_t> new_round_16 = {
        204,                                // header (NEW_ROUND)
        0x00, 0x00, 0x00, 0x10,           // length = 16
        0x00, 0x63, 0x32, 0x20,           // unified_height (big-endian)
        0x00, 0x63, 0x31, 0x00,           // prime_height (big-endian)
        0x00, 0x63, 0x31, 0x50,           // hash_height (big-endian)
        0x00, 0x63, 0x30, 0xA0            // stake_height (big-endian)
    };
    
    acc.feed(new_round_16);
    
    bool parsed = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test1 = parsed && 
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 204) &&
                 (packet.m_length == 16) &&
                 (packet.m_data != nullptr) &&
                 (packet.m_data->size() == 16) &&
                 acc.empty();
    print_test_result("NEW_ROUND 16-byte legacy payload parsed correctly", test1);
}

// ============================================================================
// Test Case 22: Legacy response opcodes (200-203) are data-bearing
//
// Before this fix, BLOCK_ACCEPTED (200), BLOCK_REJECTED (201), COINBASE_SET (202),
// COINBASE_FAIL (203) were classified as header-only (zero-length required).
// The LLL-TAO node sends BLOCK_REJECTED (0xC9) with a 1-byte reason payload on
// the legacy lane:
//   Wire prefix observed in logs: [c9 00 00 00 01 06]
//   → opcode=0xC9(201), length=1, reason=0x06
// With the old classification the parser returned MALFORMED → connection drop.
//
// Fix: opcodes 200-203 are now data-bearing. The data-bearing path handles
// length=0 correctly, so zero-length frames still parse.
// ============================================================================
void test_legacy_response_opcodes_are_data_bearing() {
    std::cout << "\nTest 22: Legacy response opcodes (200-203) are data-bearing" << std::endl;

    TestAccumulator acc;
    Packet packet;
    ParseResult result;

    // ── Zero-length forms (backward-compat: node may still send length=0) ──────

    // ACCEPT (200) zero-length: must still parse via data-bearing path
    acc.feed({200, 0x00, 0x00, 0x00, 0x00});
    bool parsed1 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test1 = parsed1 &&
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 200) &&
                 (packet.m_length == 0) &&
                 acc.empty();
    print_test_result("ACCEPT (200) zero-length frame parses via data-bearing path", test1);

    // REJECT (201) zero-length: must still parse via data-bearing path
    acc.feed({201, 0x00, 0x00, 0x00, 0x00});
    bool parsed2 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test2 = parsed2 &&
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 201) &&
                 (packet.m_length == 0) &&
                 acc.empty();
    print_test_result("REJECT (201) zero-length frame parses via data-bearing path", test2);

    // COINBASE_SET (202) zero-length
    acc.feed({202, 0x00, 0x00, 0x00, 0x00});
    bool parsed3 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test3 = parsed3 &&
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 202) &&
                 (packet.m_length == 0) &&
                 acc.empty();
    print_test_result("COINBASE_SET (202) zero-length frame parses", test3);

    // COINBASE_FAIL (203) zero-length
    acc.feed({203, 0x00, 0x00, 0x00, 0x00});
    bool parsed4 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test4 = parsed4 &&
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 203) &&
                 (packet.m_length == 0) &&
                 acc.empty();
    print_test_result("COINBASE_FAIL (203) zero-length frame parses", test4);

    // ── 1-byte payload forms ───────────────────────────────────────────────────

    // REJECT (201) with 1-byte reason 0x06 — the exact wire prefix from the live
    // miner log that broke the legacy lane:
    //   [c9 00 00 00 01 06] → opcode=201, length=1, reason=0x06
    // Was: MALFORMED (header-only check rejected length != 0)
    // Now: SUCCESS with 1-byte payload
    acc.feed({0xC9, 0x00, 0x00, 0x00, 0x01, 0x06});
    bool parsed5 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test5 = parsed5 &&
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 201) &&
                 (packet.m_length == 1) &&
                 (packet.m_data != nullptr) &&
                 (packet.m_data->size() == 1) &&
                 ((*packet.m_data)[0] == 0x06) &&
                 acc.empty();
    print_test_result("REJECT (201) with 1-byte reason 0x06 parses correctly (log regression)", test5);

    // ACCEPT (200) with 1-byte reason 0x01
    acc.feed({0xC8, 0x00, 0x00, 0x00, 0x01, 0x01});
    bool parsed6 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test6 = parsed6 &&
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 200) &&
                 (packet.m_length == 1) &&
                 (packet.m_data != nullptr) &&
                 ((*packet.m_data)[0] == 0x01) &&
                 acc.empty();
    print_test_result("ACCEPT (200) with 1-byte reason parses correctly", test6);

    // ── Fragmentation: partial REJECT+reason triggers NEED_MORE_DATA ──────────
    acc.feed({0xC9, 0x00, 0x00, 0x00, 0x01}); // 5 bytes — length field present, data missing
    bool parsed7 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test7 = !parsed7 &&
                 (result == ParseResult::NEED_MORE_DATA) &&
                 (acc.size() == 5);
    print_test_result("REJECT (201) + length but no reason byte → NEED_MORE_DATA", test7);
    acc.clear();
}

// ============================================================================
// Test Case 22b: Legacy submit-result opcodes mirror stateless framed behavior
// ============================================================================
void test_legacy_submit_result_compat_forms() {
    std::cout << "\nTest 22b: Legacy submit-result opcode wire forms" << std::endl;

    TestAccumulator acc;
    Packet packet;
    ParseResult result;

    acc.feed({0xC8});
    bool parsed1 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test1 = !parsed1 &&
                 (result == ParseResult::NEED_MORE_DATA) &&
                 (acc.size() == 1);
    print_test_result("Legacy BLOCK_ACCEPTED bare header waits for required len4", test1);
    acc.clear();

    acc.feed({0xC8, 0x00, 0x00, 0x00, 0x00});
    bool parsed2 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test2 = parsed2 &&
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 0xC8) &&
                 (packet.m_length == 0) &&
                 acc.empty();
    print_test_result("Legacy BLOCK_ACCEPTED explicit zero-length frame parsed", test2);

    acc.feed({0xC8, 0xD8, 0x00, 0x00, 0x00});
    bool parsed3 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test3 = !parsed3 &&
                 (result == ParseResult::MALFORMED) &&
                 acc.empty();
    print_test_result("Legacy bare BLOCK_ACCEPTED followed by another packet is malformed", test3);

    acc.feed({0xC8, 0x00, 0x00, 0x00, 0x00,
              0xD8, 0x00, 0x00, 0x00, 0x00});
    bool parsed4 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool parsed5 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test4 = parsed4 && parsed5 &&
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 0xD8) &&
                 (packet.m_length == 0) &&
                 acc.empty();
    print_test_result("Adjacent legacy framed BLOCK_ACCEPTED + MINER_READY parse correctly", test4);

    acc.feed({0xC9, 0x00, 0x00, 0x00, 0x00});
    bool parsed6 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test5 = parsed6 &&
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 0xC9) &&
                 (packet.m_length == 0) &&
                 acc.empty();
    print_test_result("Legacy BLOCK_REJECTED explicit zero-length frame parsed", test5);

    acc.feed({0xC9, 0x00, 0x00, 0x00, 0x01, 0x04});
    bool parsed7 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test6 = parsed7 &&
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 0xC9) &&
                 (packet.m_length == 1) &&
                 (packet.m_data != nullptr) &&
                 ((*packet.m_data)[0] == 0x04) &&
                 acc.empty();
    print_test_result("Legacy BLOCK_REJECTED one-byte reason frame parsed", test6);
}

// ============================================================================
// Test Case 23: Stateless GET_BLOCK (0xD081) with 228-byte payload
// On stateless lane, GET_BLOCK is a template push with 228-byte payload
// ============================================================================
void test_stateless_get_block_with_payload() {
    std::cout << "\nTest 23: Stateless GET_BLOCK (0xD081) with 228-byte payload" << std::endl;
    
    TestAccumulator acc;
    Packet packet;
    ParseResult result;
    
    // Build a complete STATELESS_GET_BLOCK (0xD081) with 228-byte payload
    // Wire format: [0xD0][0x81][00 00 00 E4][228 bytes]
    std::vector<uint8_t> get_block_packet;
    get_block_packet.push_back(0xD0);  // header MSB
    get_block_packet.push_back(0x81);  // header LSB
    get_block_packet.push_back(0x00);  // length MSB
    get_block_packet.push_back(0x00);
    get_block_packet.push_back(0x00);
    get_block_packet.push_back(0xE4);  // length LSB = 228
    // 228 bytes of payload (simulated template data)
    for (int i = 0; i < 228; ++i) {
        get_block_packet.push_back(static_cast<uint8_t>(i));
    }
    
    acc.feed(get_block_packet);
    
    bool parsed = acc.parse_one_packet(ProtocolLane::STATELESS, packet, result);
    bool test1 = parsed &&
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 0xD081) &&
                 (packet.m_length == 228) &&
                 (packet.m_data != nullptr) &&
                 (packet.m_data->size() == 228) &&
                 acc.empty();
    print_test_result("STATELESS_GET_BLOCK (0xD081) with 228-byte payload parsed correctly", test1);
    
    // Verify first few bytes of payload
    if (packet.m_data && packet.m_data->size() >= 4) {
        bool data_correct = ((*packet.m_data)[0] == 0x00) &&
                           ((*packet.m_data)[1] == 0x01) &&
                           ((*packet.m_data)[2] == 0x02) &&
                           ((*packet.m_data)[3] == 0x03);
        print_test_result("STATELESS_GET_BLOCK payload data verified", data_correct);
    } else {
        print_test_result("STATELESS_GET_BLOCK payload data verified", false);
    }
}

// ============================================================================
// Test Case 23b: Stateless GET_BLOCK (0xD081) with zero-length payload
// Node can send GET_BLOCK with length=0 (no template available yet)
// ============================================================================
void test_stateless_get_block_zero_length() {
    std::cout << "\nTest 23b: Stateless GET_BLOCK (0xD081) with zero-length payload" << std::endl;
    
    TestAccumulator acc;
    Packet packet;
    ParseResult result;
    
    // Build STATELESS_GET_BLOCK (0xD081) with length=0
    // Wire format: [0xD0][0x81][00 00 00 00]
    std::vector<uint8_t> get_block_packet = {
        0xD0, 0x81,                    // header = 0xD081
        0x00, 0x00, 0x00, 0x00         // length = 0
    };
    
    acc.feed(get_block_packet);
    
    bool parsed = acc.parse_one_packet(ProtocolLane::STATELESS, packet, result);
    bool test1 = parsed &&
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 0xD081) &&
                 (packet.m_length == 0) &&
                 acc.empty();
    print_test_result("STATELESS_GET_BLOCK (0xD081) with length=0 parsed correctly", test1);
}

// ============================================================================
// Test Case 23c: Legacy GET_BLOCK (0x81) accepts request and template response forms
// ============================================================================
void test_legacy_get_block_template_forms() {
    std::cout << "\nTest 23c: Legacy GET_BLOCK (0x81) request/template forms" << std::endl;

    TestAccumulator acc;
    Packet packet;
    ParseResult result;

    acc.feed({0x81, 0x00, 0x00, 0x00, 0x00});
    bool parsed1 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test1 = parsed1 &&
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 0x81) &&
                 (packet.m_length == 0) &&
                 acc.empty();
    print_test_result("Legacy GET_BLOCK zero-length request frame parsed", test1);

    std::vector<uint8_t> template_response = {0x81, 0x00, 0x00, 0x00, 0xE4};
    for (int i = 0; i < 228; ++i) {
        template_response.push_back(static_cast<uint8_t>(i & 0xFF));
    }
    acc.feed(template_response);
    bool parsed2 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test2 = parsed2 &&
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 0x81) &&
                 (packet.m_length == 228) &&
                 (packet.m_data != nullptr) &&
                 (packet.m_data->size() == 228) &&
                 ((*packet.m_data)[0] == 0x00) &&
                 ((*packet.m_data)[3] == 0x03) &&
                 acc.empty();
    print_test_result("Legacy GET_BLOCK 228-byte template response parsed", test2);

    std::vector<uint8_t> partial_template = {0x81, 0x00, 0x00, 0x00, 0xE4};
    partial_template.resize(partial_template.size() + 100, 0xAA);
    acc.feed(partial_template);
    bool parsed3 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test3 = !parsed3 &&
                 (result == ParseResult::NEED_MORE_DATA) &&
                 (acc.size() == partial_template.size());
    print_test_result("Legacy GET_BLOCK partial template waits for more data", test3);
    acc.clear();

    acc.feed({0x81, 0x00, 0x00, 0x00, 0x01, 0xAA});
    bool parsed4 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test4 = !parsed4 &&
                 (result == ParseResult::MALFORMED) &&
                 acc.empty();
    print_test_result("Legacy GET_BLOCK invalid nonzero length is malformed", test4);
}

// ============================================================================
// Test Case 23d: Stateless BLOCK_DATA (0xD000) with 216-byte payload
// Node responds to GET_BLOCK with BLOCK_DATA containing 216-byte block template
// ============================================================================
void test_stateless_block_data_with_payload() {
    std::cout << "\nTest 23c: Stateless BLOCK_DATA (0xD000) with 216-byte payload" << std::endl;
    
    TestAccumulator acc;
    Packet packet;
    ParseResult result;
    
    // Build a complete STATELESS_BLOCK_DATA (0xD000) with 216-byte payload
    // Wire format: [0xD0][0x00][00 00 00 D8][216 bytes]
    std::vector<uint8_t> block_data_packet;
    block_data_packet.push_back(0xD0);  // header MSB
    block_data_packet.push_back(0x00);  // header LSB
    block_data_packet.push_back(0x00);  // length MSB
    block_data_packet.push_back(0x00);
    block_data_packet.push_back(0x00);
    block_data_packet.push_back(0xD8);  // length LSB = 216
    // 216 bytes of payload (simulated block template)
    for (int i = 0; i < 216; ++i) {
        block_data_packet.push_back(static_cast<uint8_t>(i));
    }
    
    acc.feed(block_data_packet);
    
    bool parsed = acc.parse_one_packet(ProtocolLane::STATELESS, packet, result);
    bool test1 = parsed &&
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 0xD000) &&
                 (packet.m_is_uint16_opcode) &&
                 (packet.m_length == 216) &&
                 (packet.m_data != nullptr) &&
                 (packet.m_data->size() == 216) &&
                 acc.empty();
    print_test_result("STATELESS_BLOCK_DATA (0xD000) with 216-byte payload parsed correctly", test1);
    
    // Verify first few bytes of payload
    if (packet.m_data && packet.m_data->size() >= 4) {
        bool data_correct = ((*packet.m_data)[0] == 0x00) &&
                           ((*packet.m_data)[1] == 0x01) &&
                           ((*packet.m_data)[2] == 0x02) &&
                           ((*packet.m_data)[3] == 0x03);
        print_test_result("STATELESS_BLOCK_DATA payload data verified", data_correct);
    } else {
        print_test_result("STATELESS_BLOCK_DATA payload data verified", false);
    }
    
    // Verify opcode matches the StatelessMining::BLOCK_DATA constant
    bool opcode_test = (packet.m_header == nexusminer::LLP::StatelessMining::BLOCK_DATA);
    print_test_result("STATELESS_BLOCK_DATA header matches StatelessMining::BLOCK_DATA", opcode_test);
}

// ============================================================================
// Test Case 24: Stateless auth opcodes (mirror-mapped) with payload
// Auth opcodes 0xD0CE, 0xD0D0, 0xD0D2 should parse as 2-byte headers with payload
// ============================================================================
void test_stateless_auth_opcodes_with_payload() {
    std::cout << "\nTest 24: Stateless auth opcodes (mirror-mapped) with payload" << std::endl;
    
    TestAccumulator acc;
    Packet packet;
    ParseResult result;
    
    // CHANNEL_ACK (0xD0CE = mirror of 206) with 1-byte payload
    std::vector<uint8_t> channel_ack = {
        0xD0, 0xCE,                    // header = 0xD0CE
        0x00, 0x00, 0x00, 0x01,        // length = 1
        0x02                           // channel = 2 (Hash)
    };
    
    acc.feed(channel_ack);
    bool parsed1 = acc.parse_one_packet(ProtocolLane::STATELESS, packet, result);
    bool test1 = parsed1 &&
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 0xD0CE) &&
                 (packet.m_length == 1) &&
                 (packet.m_data != nullptr) &&
                 acc.empty();
    print_test_result("CHANNEL_ACK (0xD0CE) parsed with payload", test1);
    
    // MINER_AUTH_CHALLENGE (0xD0D0 = mirror of 208) with 34-byte nonce payload
    std::vector<uint8_t> auth_challenge;
    auth_challenge.push_back(0xD0);  // header MSB
    auth_challenge.push_back(0xD0);  // header LSB (mirror of 208)
    auth_challenge.push_back(0x00);  // length
    auth_challenge.push_back(0x00);
    auth_challenge.push_back(0x00);
    auth_challenge.push_back(0x22);  // length = 34
    // 34 bytes of nonce data
    for (int i = 0; i < 34; ++i) {
        auth_challenge.push_back(static_cast<uint8_t>(0xAA + i));
    }
    
    acc.feed(auth_challenge);
    bool parsed2 = acc.parse_one_packet(ProtocolLane::STATELESS, packet, result);
    bool test2 = parsed2 &&
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 0xD0D0) &&
                 (packet.m_length == 34) &&
                 (packet.m_data != nullptr) &&
                 acc.empty();
    print_test_result("MINER_AUTH_CHALLENGE (0xD0D0) parsed with payload", test2);
    
    // MINER_AUTH_RESULT (0xD0D2 = mirror of 210) with 5-byte payload
    std::vector<uint8_t> auth_result = {
        0xD0, 0xD2,                    // header = 0xD0D2
        0x00, 0x00, 0x00, 0x05,        // length = 5
        0x01,                          // status = success
        0x12, 0x34, 0x56, 0x78         // session_id (LE)
    };
    
    acc.feed(auth_result);
    bool parsed3 = acc.parse_one_packet(ProtocolLane::STATELESS, packet, result);
    bool test3 = parsed3 &&
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 0xD0D2) &&
                 (packet.m_length == 5) &&
                 (packet.m_data != nullptr) &&
                 acc.empty();
    print_test_result("MINER_AUTH_RESULT (0xD0D2) parsed with payload", test3);
}

// ============================================================================
// Test Case 25: Legacy auth opcode 208 (0xD0) parses correctly (NOT rejected)
// Byte 0xD0 on legacy lane is MINER_AUTH_CHALLENGE and must NOT be rejected
// ============================================================================
void test_legacy_auth_opcode_208_not_rejected() {
    std::cout << "\nTest 25: Legacy auth opcode 208 (0xD0) not rejected" << std::endl;
    
    TestAccumulator acc;
    Packet packet;
    ParseResult result;
    
    // MINER_AUTH_CHALLENGE (208 = 0xD0) with 34-byte nonce payload on legacy lane
    std::vector<uint8_t> auth_challenge;
    auth_challenge.push_back(208);     // header = 0xD0 (MINER_AUTH_CHALLENGE)
    auth_challenge.push_back(0x00);    // length
    auth_challenge.push_back(0x00);
    auth_challenge.push_back(0x00);
    auth_challenge.push_back(0x22);    // length = 34
    for (int i = 0; i < 34; ++i) {
        auth_challenge.push_back(static_cast<uint8_t>(0xBB + i));
    }
    
    acc.feed(auth_challenge);
    bool parsed = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test1 = parsed &&
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 208) &&
                 (packet.m_length == 34) &&
                 (packet.m_data != nullptr) &&
                 (packet.m_data->size() == 34) &&
                 acc.empty();
    print_test_result("Legacy MINER_AUTH_CHALLENGE (208/0xD0) parsed correctly (NOT rejected)", test1);
    
    // Also verify MINER_AUTH_INIT (207) on legacy lane
    std::vector<uint8_t> auth_init = {
        207,                           // header = MINER_AUTH_INIT
        0x00, 0x00, 0x00, 0x04,       // length = 4
        0x01, 0x02, 0x03, 0x04        // payload
    };
    
    acc.feed(auth_init);
    bool parsed2 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test2 = parsed2 &&
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 207) &&
                 (packet.m_length == 4) &&
                 acc.empty();
    print_test_result("Legacy MINER_AUTH_INIT (207) parsed correctly", test2);
    
    // Also verify MINER_AUTH_RESULT (210) on legacy lane
    std::vector<uint8_t> auth_result = {
        210,                           // header = MINER_AUTH_RESULT
        0x00, 0x00, 0x00, 0x05,       // length = 5
        0x01,                          // status = success
        0x12, 0x34, 0x56, 0x78        // session_id
    };
    
    acc.feed(auth_result);
    bool parsed3 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test3 = parsed3 &&
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 210) &&
                 (packet.m_length == 5) &&
                 acc.empty();
    print_test_result("Legacy MINER_AUTH_RESULT (210) parsed correctly", test3);
}

// ============================================================================
// Test Case 26: STATELESS_BLOCK_ACCEPTED (0xD0C8) framed form — Fix A regression
// LLL-TAO commit 206d1a2c emits the full 6-byte framed form D0 C8 00 00 00 00.
// Before Fix A the miner consumed only 2 bytes (header-only), leaving 4 orphan
// zero bytes in the accumulator and triggering a burst of MALFORMED errors.
// ============================================================================
void test_stateless_block_accepted_framed_form() {
    std::cout << "\nTest 26: STATELESS_BLOCK_ACCEPTED (0xD0C8) framed 6-byte form" << std::endl;

    TestAccumulator acc;
    Packet packet;
    ParseResult result;

    // Wire form emitted by LLL-TAO 206d1a2c: [D0 C8][00 00 00 00]
    acc.feed({0xD0, 0xC8, 0x00, 0x00, 0x00, 0x00});
    bool parsed = acc.parse_one_packet(ProtocolLane::STATELESS, packet, result);
    bool test1 = parsed &&
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 0xD0C8) &&
                 (packet.m_length == 0) &&
                 acc.empty();
    print_test_result("STATELESS_BLOCK_ACCEPTED framed form: SUCCESS, bytes_consumed=6, accumulator empty", test1);
}

// ============================================================================
// Test Case 27: STATELESS_BLOCK_REJECTED (0xD0C9) framed form
// ============================================================================
void test_stateless_block_rejected_framed_form() {
    std::cout << "\nTest 27: STATELESS_BLOCK_REJECTED (0xD0C9) framed 6-byte form" << std::endl;

    TestAccumulator acc;
    Packet packet;
    ParseResult result;

    acc.feed({0xD0, 0xC9, 0x00, 0x00, 0x00, 0x00});
    bool parsed = acc.parse_one_packet(ProtocolLane::STATELESS, packet, result);
    bool test1 = parsed &&
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 0xD0C9) &&
                 (packet.m_length == 0) &&
                 acc.empty();
    print_test_result("STATELESS_BLOCK_REJECTED framed form: SUCCESS, bytes_consumed=6, accumulator empty", test1);
}

// ============================================================================
// Test Case 28: STATELESS_BLOCK_REJECTED (0xD0C9) framed form with 1-byte reason
// ============================================================================
void test_stateless_block_rejected_framed_with_reason_byte() {
    std::cout << "\nTest 28: STATELESS_BLOCK_REJECTED (0xD0C9) framed form with 1-byte reason" << std::endl;

    TestAccumulator acc;
    Packet packet;
    ParseResult result;

    // [D0 C9][00 00 00 01][2A]  — length=1, reason=0x2A (STALE or similar)
    acc.feed({0xD0, 0xC9, 0x00, 0x00, 0x00, 0x01, 0x2A});
    bool parsed = acc.parse_one_packet(ProtocolLane::STATELESS, packet, result);
    bool test1 = parsed &&
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 0xD0C9) &&
                 (packet.m_length == 1) &&
                 (packet.m_data != nullptr) &&
                 ((*packet.m_data)[0] == 0x2A) &&
                 acc.empty();
    print_test_result("STATELESS_BLOCK_REJECTED framed+reason: SUCCESS, bytes_consumed=7, length=1", test1);
}

void test_get_block_control_frames() {
    std::cout << "\nGET_BLOCK control frame fragmentation and coalescing" << std::endl;
    for (auto lane : {ProtocolLane::STATELESS, ProtocolLane::LEGACY}) {
        TestAccumulator acc;
        Packet packet;
        ParseResult result;
        if (lane == ProtocolLane::STATELESS)
            acc.feed({0xD0, 0xC9, 0, 0, 0, 8, 1, 4});
        else
            acc.feed({0xC9, 0, 0, 0, 8, 1, 4});
        const bool incomplete = !acc.parse_one_packet(lane, packet, result) &&
                                result == ParseResult::NEED_MORE_DATA;
        acc.feed({0, 0, 0, 0, 3, 0xE8});
        if (lane == ProtocolLane::STATELESS)
            acc.feed({0xD0, 0xC8, 0, 0, 0, 0});
        else
            acc.feed({0xC8, 0, 0, 0, 0});
        const bool control = acc.parse_one_packet(lane, packet, result) &&
                             packet.m_length == 8 && packet.m_data &&
                             (*packet.m_data)[1] == 4;
        const bool next = acc.parse_one_packet(lane, packet, result) &&
                          packet.m_length == 0 && acc.empty();
        print_test_result(lane == ProtocolLane::STATELESS
                          ? "Stateless control is fragmented and followed by a submit result"
                          : "Legacy control is fragmented and followed by a submit result",
                          incomplete && control && next);
    }

    TestAccumulator acc;
    Packet packet;
    ParseResult result;
    acc.feed({0xD0, 0xC8, 0, 0, 0, 8});
    print_test_result("Eight-byte BLOCK_ACCEPTED remains malformed",
                      !acc.parse_one_packet(ProtocolLane::STATELESS, packet, result) &&
                      result == ParseResult::MALFORMED);
}

// ============================================================================
// Test Case 29: STATELESS_BLOCK_ACCEPTED (0xD0C8) bare 2-byte form is rejected.
// ============================================================================
void test_stateless_block_accepted_bare_form_is_rejected() {
    std::cout << "\nTest 29: STATELESS_BLOCK_ACCEPTED (0xD0C8) bare 2-byte form rejected" << std::endl;

    TestAccumulator acc;
    Packet packet;
    ParseResult result;

    // Only 2 bytes — no length field present.
    acc.feed({0xD0, 0xC8});
    bool parsed = acc.parse_one_packet(ProtocolLane::STATELESS, packet, result);
    bool test1 = !parsed &&
                 (result == ParseResult::NEED_MORE_DATA) &&
                 (acc.size() == 2);
    print_test_result("STATELESS_BLOCK_ACCEPTED bare form waits for required len4", test1);
}

// ============================================================================
// Test Case 30: Full production log pattern regression test
// Feed D0 C8 00 00 00 00 (BLOCK_ACCEPTED framed) followed immediately by
// D0 D9 00 00 00 94 <148B> (STATELESS_PRIME_BLOCK_AVAILABLE, 148-byte payload).
// Both packets must parse successfully, zero MALFORMED results, accumulator empty.
// This is the exact sequence that triggered the bug reported in the issue.
// ============================================================================
void test_stateless_block_accepted_followed_by_prime_available() {
    std::cout << "\nTest 30: BLOCK_ACCEPTED framed + PRIME_BLOCK_AVAILABLE regression" << std::endl;

    TestAccumulator acc;
    Packet packet;
    ParseResult result;

    // First packet: STATELESS_BLOCK_ACCEPTED [D0 C8][00 00 00 00]
    std::vector<uint8_t> stream;
    stream.insert(stream.end(), {0xD0, 0xC8, 0x00, 0x00, 0x00, 0x00});

    // Second packet: STATELESS_PRIME_BLOCK_AVAILABLE [D0 D9][00 00 00 94][148 bytes]
    stream.insert(stream.end(), {0xD0, 0xD9, 0x00, 0x00, 0x00, 0x94});
    for (int i = 0; i < 148; ++i) {
        stream.push_back(static_cast<uint8_t>(i & 0xFF));
    }

    acc.feed(stream);

    // Parse first packet — must be BLOCK_ACCEPTED, no MALFORMED
    bool parsed1 = acc.parse_one_packet(ProtocolLane::STATELESS, packet, result);
    bool test1 = parsed1 &&
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 0xD0C8) &&
                 (packet.m_length == 0);
    print_test_result("First packet: BLOCK_ACCEPTED SUCCESS (no MALFORMED)", test1);

    // Parse second packet — must be PRIME_BLOCK_AVAILABLE with 148-byte payload
    bool parsed2 = acc.parse_one_packet(ProtocolLane::STATELESS, packet, result);
    bool test2 = parsed2 &&
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 0xD0D9) &&
                 (packet.m_length == 148) &&
                 (packet.m_data != nullptr) &&
                 (packet.m_data->size() == 148) &&
                 acc.empty();
    print_test_result("Second packet: PRIME_BLOCK_AVAILABLE SUCCESS, 148B payload, accumulator empty", test2);
}

// ============================================================================
// Test Case 30b: Legacy production pattern matching stateless submit-result flow
// ============================================================================
void test_legacy_block_accepted_followed_by_prime_available() {
    std::cout << "\nTest 30b: Legacy BLOCK_ACCEPTED + PRIME_BLOCK_AVAILABLE regression" << std::endl;

    constexpr int PRIME_BLOCK_AVAILABLE_PAYLOAD_SIZE = 148;
    TestAccumulator acc;
    Packet packet;
    ParseResult result;

    std::vector<uint8_t> stream;
    stream.insert(stream.end(), {0xC8, 0x00, 0x00, 0x00, 0x00});

    stream.insert(stream.end(), {0xD9, 0x00, 0x00, 0x00, 0x94});
    for (int i = 0; i < PRIME_BLOCK_AVAILABLE_PAYLOAD_SIZE; ++i) {
        stream.push_back(static_cast<uint8_t>(i & 0xFF));
    }

    acc.feed(stream);

    bool parsed1 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test1 = parsed1 &&
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 0xC8) &&
                 (packet.m_length == 0);
    print_test_result("Legacy first packet: BLOCK_ACCEPTED SUCCESS (no MALFORMED)", test1);

    bool parsed2 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test2 = parsed2 &&
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 0xD9) &&
                 (packet.m_length == PRIME_BLOCK_AVAILABLE_PAYLOAD_SIZE) &&
                 (packet.m_data != nullptr) &&
                 (packet.m_data->size() == PRIME_BLOCK_AVAILABLE_PAYLOAD_SIZE) &&
                 acc.empty();
    print_test_result("Legacy second packet: PRIME_BLOCK_AVAILABLE SUCCESS, 148B payload, accumulator empty", test2);
}

// ============================================================================
// Test Case 30c: Legacy submit-result fragmented across TCP reads
// ============================================================================
void test_legacy_block_accepted_fragmented_before_push() {
    std::cout << "\nTest 30c: Legacy fragmented BLOCK_ACCEPTED + PRIME_BLOCK_AVAILABLE regression" << std::endl;

    constexpr int PRIME_BLOCK_AVAILABLE_PAYLOAD_SIZE = 148;
    TestAccumulator acc;
    Packet packet;
    ParseResult result;

    acc.feed({0xC8});
    bool parsed1 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test1 = !parsed1 &&
                 (result == ParseResult::NEED_MORE_DATA) &&
                 (acc.size() == 1);
    print_test_result("Legacy fragmented BLOCK_ACCEPTED keeps 1-byte header pending", test1);

    acc.feed({0x00, 0x00});
    bool parsed2 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test2 = !parsed2 &&
                 (result == ParseResult::NEED_MORE_DATA) &&
                 (acc.size() == 3);
    print_test_result("Legacy fragmented BLOCK_ACCEPTED waits for full len4", test2);

    std::vector<uint8_t> tail = {0x00, 0x00, 0xD9, 0x00, 0x00, 0x00, 0x94};
    // Distinct non-zero payload bytes prove the accumulator drains exactly the
    // accepted-block frame before parsing the following PUSH payload.
    for (int i = 0; i < PRIME_BLOCK_AVAILABLE_PAYLOAD_SIZE; ++i) {
        tail.push_back(static_cast<uint8_t>((0x80 + i) & 0xFF));
    }
    acc.feed(tail);

    bool parsed3 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test3 = parsed3 &&
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 0xC8) &&
                 (packet.m_length == 0);
    print_test_result("Legacy fragmented BLOCK_ACCEPTED completes when len4 arrives", test3);

    bool parsed4 = acc.parse_one_packet(ProtocolLane::LEGACY, packet, result);
    bool test4 = parsed4 &&
                 (result == ParseResult::SUCCESS) &&
                 (packet.m_header == 0xD9) &&
                 (packet.m_length == PRIME_BLOCK_AVAILABLE_PAYLOAD_SIZE) &&
                 (packet.m_data != nullptr) &&
                 (packet.m_data->size() == PRIME_BLOCK_AVAILABLE_PAYLOAD_SIZE) &&
                 acc.empty();
    print_test_result("Legacy PUSH following fragmented accept parses after accumulator drain", test4);
}

// ============================================================================
// Test Case 31: Cross-lane frame prefixes are recognizable before byte-drop resync
// ============================================================================
void test_cross_lane_prefix_detection() {
    std::cout << "\nTest 31: Cross-lane frame prefix detection" << std::endl;

    auto stateless_reject_on_legacy =
        std::make_shared<network::Payload>(network::Payload{0xD0, 0xC9, 0x00, 0x00, 0x00, 0x00});
    auto legacy_reject_on_stateless =
        std::make_shared<network::Payload>(network::Payload{0xC9, 0x00, 0x00, 0x00, 0x00});
    auto legacy_auth_challenge =
        std::make_shared<network::Payload>(network::Payload{0xD0, 0x00, 0x00, 0x00, 0x04,
                                                            0xAA, 0xBB, 0xCC, 0xDD});

    print_test_result("0xD0C9 framed stateless reject is cross-lane on LEGACY",
        looks_like_cross_lane_frame_prefix(stateless_reject_on_legacy, 0, ProtocolLane::LEGACY));
    print_test_result("0xC9 legacy reject is cross-lane on STATELESS",
        looks_like_cross_lane_frame_prefix(legacy_reject_on_stateless, 0, ProtocolLane::STATELESS));
    print_test_result("0xD0 legacy auth challenge with len4 is not cross-lane on LEGACY",
        !looks_like_cross_lane_frame_prefix(legacy_auth_challenge, 0, ProtocolLane::LEGACY));
}

// ============================================================================
// Main test runner
// ============================================================================
int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "LLP Packet Framing Test Suite" << std::endl;
    std::cout << "Testing TCP fragmentation handling" << std::endl;
    std::cout << "========================================" << std::endl;
    
    test_complete_packet_single_receive();
    test_header_fragmented();
    test_payload_fragmented();
    test_multiple_packets_single_receive();
    test_stateless_header_fragmented();
    test_malformed_huge_length();
    test_packet_constructor_rejects_huge_length();
    test_malformed_invalid_stateless_opcode();
    test_complex_mixed_scenario();
    test_byte_by_byte_feeding();
    test_zero_length_payload();
    test_push_notification_legacy_lane();
    test_is_auth_packet_full_range();
    test_miner_ready_header_only();
    test_legacy_data_packet_single_byte();
    test_legacy_header_only_single_byte();
    test_legacy_auth_packet_single_byte();
    test_stateless_data_packet_two_byte();
    test_stateless_header_only_two_byte();
    test_stateless_submit_result_compat_forms();
    test_new_round_with_payload();
    test_old_round_with_payload();
    test_round_legacy_16byte_payload();
    test_legacy_response_opcodes_are_data_bearing();
    test_legacy_submit_result_compat_forms();
    test_stateless_get_block_with_payload();
    test_stateless_get_block_zero_length();
    test_legacy_get_block_template_forms();
    test_stateless_block_data_with_payload();
    test_stateless_auth_opcodes_with_payload();
    test_legacy_auth_opcode_208_not_rejected();
    test_stateless_block_accepted_framed_form();
    test_stateless_block_rejected_framed_form();
    test_stateless_block_rejected_framed_with_reason_byte();
    test_get_block_control_frames();
    test_stateless_block_accepted_bare_form_is_rejected();
    test_stateless_block_accepted_followed_by_prime_available();
    test_legacy_block_accepted_followed_by_prime_available();
    test_legacy_block_accepted_fragmented_before_push();
    test_cross_lane_prefix_detection();
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Test Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Tests run:    " << tests_run << std::endl;
    std::cout << "Tests passed: " << tests_passed << std::endl;
    std::cout << "Tests failed: " << tests_failed << std::endl;
    std::cout << "Success rate: " << (tests_run > 0 ? (100 * tests_passed / tests_run) : 0) << "%" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return (tests_failed == 0) ? 0 : 1;
}
