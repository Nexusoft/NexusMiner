/**
 * @file protocol_lane_header_test.cpp
 * @brief Unit tests for Protocol Lane Determination from PORT and Header Size
 *
 * Validates the relationship between:
 *   - Connection port → Protocol lane (LEGACY vs STATELESS)
 *   - Protocol lane → Header byte size (1-byte vs 2-byte)
 *   - Packet wire format (1-byte legacy header vs 2-byte stateless header)
 *   - Header-only vs data-bearing classification per lane
 *
 * Port-Lane-Header mapping:
 *   Port 8323  → LEGACY    → 1-byte header (uint8_t,  0x00-0xFF)
 *   Port 9323+ → STATELESS → 2-byte header (uint16_t, 0xD000-0xD0FF)
 */

#include "miner_opcodes.hpp"
#include "protocol_lane.hpp"
#include "packet.hpp"
#include <iostream>
#include <cassert>
#include <cstdint>

using namespace nexusminer;

// Fully qualified aliases to avoid ambiguity with global ::LLP namespace from block.hpp
namespace MinerLLP = nexusminer::LLP;

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

int main()
{
    std::cout << "========================================" << std::endl;
    std::cout << "Protocol Lane & Header Size Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    // ====================================================================
    // Test 1: Port → Lane determination
    // ====================================================================
    std::cout << "\nTest 1: Port → Lane determination" << std::endl;
    {
        // Legacy port
        print_test_result("Port 8323 → LEGACY",
            determine_lane_from_port(ProtocolPorts::LEGACY_PORT) == ProtocolLane::LEGACY);

        // Stateless ports
        print_test_result("Port 9323 → STATELESS",
            determine_lane_from_port(ProtocolPorts::STATELESS_PORT) == ProtocolLane::STATELESS);
        print_test_result("Port 9324 → STATELESS",
            determine_lane_from_port(9324) == ProtocolLane::STATELESS);
        print_test_result("Port 10000 → STATELESS",
            determine_lane_from_port(10000) == ProtocolLane::STATELESS);
        
        // Edge cases
        print_test_result("Port 0 → STATELESS (not legacy)",
            determine_lane_from_port(0) == ProtocolLane::STATELESS);
        print_test_result("Port 8322 → STATELESS (off-by-one low)",
            determine_lane_from_port(8322) == ProtocolLane::STATELESS);
        print_test_result("Port 8324 → STATELESS (off-by-one high)",
            determine_lane_from_port(8324) == ProtocolLane::STATELESS);
    }

    // ====================================================================
    // Test 2: Lane → Header byte size (1-byte legacy vs 2-byte stateless)
    // ====================================================================
    std::cout << "\nTest 2: Lane → Header byte size via Packet construction" << std::endl;
    {
        // Legacy lane: 1-byte header packets
        Packet legacy_pkt(static_cast<uint8_t>(MinerLLP::GET_BLOCK));
        print_test_result("Legacy Packet(uint8_t GET_BLOCK) → 1-byte header",
            !legacy_pkt.m_is_uint16_opcode);
        print_test_result("Legacy Packet header value == 129",
            legacy_pkt.m_header == MinerLLP::GET_BLOCK);

        // Stateless lane: 2-byte header packets
        Packet stateless_pkt(static_cast<uint16_t>(MinerLLP::StatelessMining::GET_BLOCK));
        print_test_result("Stateless Packet(uint16_t 0xD081) → 2-byte header",
            stateless_pkt.m_is_uint16_opcode);
        print_test_result("Stateless Packet header value == 0xD081",
            stateless_pkt.m_header == 0xD081);
    }

    // ====================================================================
    // Test 3: Wire format verification (1-byte vs 2-byte on the wire)
    // ====================================================================
    std::cout << "\nTest 3: Wire format header bytes" << std::endl;
    {
        // Legacy GET_BLOCK (opcode 129) → 5-byte zero-length frame on wire
        Packet legacy_get_block(static_cast<uint8_t>(MinerLLP::GET_BLOCK));
        auto legacy_bytes = legacy_get_block.get_bytes();
        print_test_result("Legacy GET_BLOCK wire: 5 bytes total",
            legacy_bytes && legacy_bytes->size() == 5);
        print_test_result("Legacy GET_BLOCK wire: byte[0] == 0x81 (129)",
            legacy_bytes && (*legacy_bytes)[0] == 0x81);
        print_test_result("Legacy GET_BLOCK wire: zero length field",
            legacy_bytes && (*legacy_bytes)[1] == 0x00 && (*legacy_bytes)[2] == 0x00 &&
            (*legacy_bytes)[3] == 0x00 && (*legacy_bytes)[4] == 0x00);

        // Stateless MINER_READY (0xD0D8) → 6-byte zero-length frame on wire
        Packet stateless_ready(static_cast<uint16_t>(MinerLLP::StatelessMining::MINER_READY));
        auto stateless_bytes = stateless_ready.get_bytes();
        print_test_result("Stateless MINER_READY wire: 6 bytes total",
            stateless_bytes && stateless_bytes->size() == 6);
        print_test_result("Stateless MINER_READY wire: byte[0] == 0xD0",
            stateless_bytes && (*stateless_bytes)[0] == 0xD0);
        print_test_result("Stateless MINER_READY wire: byte[1] == 0xD8",
            stateless_bytes && (*stateless_bytes)[1] == 0xD8);
        print_test_result("Stateless MINER_READY wire: zero length field",
            stateless_bytes && (*stateless_bytes)[2] == 0x00 && (*stateless_bytes)[3] == 0x00 &&
            (*stateless_bytes)[4] == 0x00 && (*stateless_bytes)[5] == 0x00);
    }

    // ====================================================================
    // Test 4: Full port → lane → header size → wire format chain
    // ====================================================================
    std::cout << "\nTest 4: Port → Lane → Header → Wire format chain" << std::endl;
    {
        // Scenario A: Legacy miner on port 8323
        {
            uint16_t port = ProtocolPorts::LEGACY_PORT;
            ProtocolLane lane = determine_lane_from_port(port);
            bool use_stateless = (lane == ProtocolLane::STATELESS);
            
            // Build GET_ROUND packet based on lane
            Packet pkt = use_stateless
                ? Packet(static_cast<uint16_t>(MinerLLP::MirrorOpcode(static_cast<uint8_t>(MinerLLP::GET_ROUND))))
                : Packet(static_cast<uint8_t>(MinerLLP::GET_ROUND));
            
            auto wire = pkt.get_bytes();
            print_test_result("Port 8323: lane=LEGACY, 1-byte header",
                lane == ProtocolLane::LEGACY && !pkt.m_is_uint16_opcode);
            print_test_result("Port 8323: GET_ROUND wire is 5 bytes",
                wire && wire->size() == 5);
            print_test_result("Port 8323: GET_ROUND wire[0] == 0x85 (133)",
                wire && (*wire)[0] == 133);
            print_test_result("Port 8323: GET_ROUND wire length == 0",
                wire && (*wire)[1] == 0x00 && (*wire)[2] == 0x00 &&
                (*wire)[3] == 0x00 && (*wire)[4] == 0x00);
        }
        
        // Scenario B: Stateless miner on port 9323
        {
            uint16_t port = ProtocolPorts::STATELESS_PORT;
            ProtocolLane lane = determine_lane_from_port(port);
            bool use_stateless = (lane == ProtocolLane::STATELESS);
            
            // Build GET_ROUND packet based on lane
            Packet pkt = use_stateless
                ? Packet(static_cast<uint16_t>(MinerLLP::MirrorOpcode(static_cast<uint8_t>(MinerLLP::GET_ROUND))))
                : Packet(static_cast<uint8_t>(MinerLLP::GET_ROUND));
            
            auto wire = pkt.get_bytes();
            print_test_result("Port 9323: lane=STATELESS, 2-byte header",
                lane == ProtocolLane::STATELESS && pkt.m_is_uint16_opcode);
            print_test_result("Port 9323: GET_ROUND wire is 6 bytes",
                wire && wire->size() == 6);
            print_test_result("Port 9323: GET_ROUND wire[0:1] == 0xD085",
                wire && (*wire)[0] == 0xD0 && (*wire)[1] == 0x85);
            print_test_result("Port 9323: GET_ROUND wire length == 0",
                wire && (*wire)[2] == 0x00 && (*wire)[3] == 0x00 &&
                (*wire)[4] == 0x00 && (*wire)[5] == 0x00);
        }
    }

    // ====================================================================
    // Test 5: Push notification opcodes: port determines header size
    // ====================================================================
    std::cout << "\nTest 5: Push notification header size per lane" << std::endl;
    {
        // Legacy push: PRIME_BLOCK_AVAILABLE (217 = 0xD9) → 1-byte header
        Packet legacy_prime(static_cast<uint8_t>(MinerLLP::PRIME_BLOCK_AVAILABLE));
        print_test_result("Legacy PRIME_BLOCK_AVAILABLE: 1-byte header",
            !legacy_prime.m_is_uint16_opcode && legacy_prime.m_header == 217);
        
        // Stateless push: PRIME_BLOCK_AVAILABLE (0xD0D9) → 2-byte header
        Packet stateless_prime(static_cast<uint16_t>(MinerLLP::StatelessMining::PRIME_BLOCK_AVAILABLE));
        print_test_result("Stateless PRIME_BLOCK_AVAILABLE: 2-byte header",
            stateless_prime.m_is_uint16_opcode && stateless_prime.m_header == 0xD0D9);
        
        // Legacy push: HASH_BLOCK_AVAILABLE (218 = 0xDA) → 1-byte header
        Packet legacy_hash(static_cast<uint8_t>(MinerLLP::HASH_BLOCK_AVAILABLE));
        print_test_result("Legacy HASH_BLOCK_AVAILABLE: 1-byte header",
            !legacy_hash.m_is_uint16_opcode && legacy_hash.m_header == 218);
        
        // Stateless push: HASH_BLOCK_AVAILABLE (0xD0DA) → 2-byte header
        Packet stateless_hash(static_cast<uint16_t>(MinerLLP::StatelessMining::HASH_BLOCK_AVAILABLE));
        print_test_result("Stateless HASH_BLOCK_AVAILABLE: 2-byte header",
            stateless_hash.m_is_uint16_opcode && stateless_hash.m_header == 0xD0DA);
    }

    // ====================================================================
    // Test 6: Header-only classification per lane
    // ====================================================================
    std::cout << "\nTest 6: Header-only vs data-bearing per lane" << std::endl;
    {
        // Legacy header-only opcodes
        print_test_result("Legacy GET_BLOCK (129): header-only",
            PacketConstants::is_legacy_header_only_opcode(MinerLLP::GET_BLOCK));
        print_test_result("Legacy GET_ROUND (133): header-only",
            PacketConstants::is_legacy_header_only_opcode(MinerLLP::GET_ROUND));
        print_test_result("Legacy ACCEPT (200): has payload",
            !PacketConstants::is_legacy_header_only_opcode(MinerLLP::ACCEPT));
        print_test_result("Legacy REJECT (201): has payload",
            !PacketConstants::is_legacy_header_only_opcode(MinerLLP::REJECT));
        print_test_result("Legacy COINBASE_SET (202): has payload",
            !PacketConstants::is_legacy_header_only_opcode(MinerLLP::COINBASE_SET));
        print_test_result("Legacy COINBASE_FAIL (203): has payload",
            !PacketConstants::is_legacy_header_only_opcode(MinerLLP::COINBASE_FAIL));
        print_test_result("Legacy MINER_READY (216): header-only",
            PacketConstants::is_legacy_header_only_opcode(MinerLLP::MINER_READY));
        print_test_result("Legacy PING (253): header-only",
            PacketConstants::is_legacy_header_only_opcode(MinerLLP::PING));
        
        // Legacy data-bearing opcodes
        print_test_result("Legacy SUBMIT_BLOCK (1): has payload",
            !PacketConstants::is_legacy_header_only_opcode(MinerLLP::SUBMIT_BLOCK));
        print_test_result("Legacy NEW_ROUND (204): has payload",
            !PacketConstants::is_legacy_header_only_opcode(MinerLLP::NEW_ROUND));
        print_test_result("Legacy PRIME_BLOCK_AVAILABLE (217): has payload",
            !PacketConstants::is_legacy_header_only_opcode(MinerLLP::PRIME_BLOCK_AVAILABLE));
        
        // Stateless header-only opcodes (mirror-mapped, same classification)
        print_test_result("Stateless MINER_READY (0xD0D8): header-only",
            PacketConstants::is_stateless_header_only_opcode(MinerLLP::StatelessMining::MINER_READY));
        
        // Stateless data-bearing opcodes
        print_test_result("Stateless GET_BLOCK (0xD081): has payload (228-byte template push)",
            !PacketConstants::is_stateless_header_only_opcode(MinerLLP::StatelessMining::GET_BLOCK));
        print_test_result("Stateless SUBMIT_BLOCK (0xD001): has payload",
            !PacketConstants::is_stateless_header_only_opcode(MinerLLP::StatelessMining::SUBMIT_BLOCK));
        print_test_result("Stateless PRIME_BLOCK_AVAILABLE (0xD0D9): has payload",
            !PacketConstants::is_stateless_header_only_opcode(MinerLLP::StatelessMining::PRIME_BLOCK_AVAILABLE));
    }

    // ====================================================================
    // Test 7: Data packet wire format: legacy 1+4+N vs stateless 2+4+N
    // ====================================================================
    std::cout << "\nTest 7: Data packet wire format per lane" << std::endl;
    {
        // Create a 12-byte payload (push notification size)
        network::Payload payload_data(12, 0xAB);
        
        // Legacy data packet: [header(1)][length(4)][data(12)] = 17 bytes
        // Note: Legacy data packets with header < 128 include length field
        Packet legacy_data(static_cast<uint8_t>(MinerLLP::SUBMIT_BLOCK), payload_data);
        auto legacy_wire = legacy_data.get_bytes();
        print_test_result("Legacy SUBMIT_BLOCK+12b: wire = 17 bytes (1+4+12)",
            legacy_wire && legacy_wire->size() == 17);
        print_test_result("Legacy SUBMIT_BLOCK+12b: wire[0] = 0x01 (1-byte header)",
            legacy_wire && (*legacy_wire)[0] == 0x01);
        print_test_result("Legacy SUBMIT_BLOCK+12b: wire[1:4] = 0x0000000C (BE length)",
            legacy_wire &&
            (*legacy_wire)[1] == 0x00 &&
            (*legacy_wire)[2] == 0x00 &&
            (*legacy_wire)[3] == 0x00 &&
            (*legacy_wire)[4] == 0x0C);
        
        // Stateless data packet: [header(2)][length(4)][data(12)] = 18 bytes
        Packet stateless_data(static_cast<uint16_t>(MinerLLP::StatelessMining::SUBMIT_BLOCK), payload_data);
        auto stateless_wire = stateless_data.get_bytes();
        print_test_result("Stateless SUBMIT_BLOCK+12b: wire = 18 bytes (2+4+12)",
            stateless_wire && stateless_wire->size() == 18);
        print_test_result("Stateless SUBMIT_BLOCK+12b: wire[0:1] = 0xD001 (2-byte header)",
            stateless_wire && (*stateless_wire)[0] == 0xD0 && (*stateless_wire)[1] == 0x01);
        print_test_result("Stateless SUBMIT_BLOCK+12b: wire[2:5] = 0x0000000C (BE length)",
            stateless_wire &&
            (*stateless_wire)[2] == 0x00 &&
            (*stateless_wire)[3] == 0x00 &&
            (*stateless_wire)[4] == 0x00 &&
            (*stateless_wire)[5] == 0x0C);
    }

    // ====================================================================
    // Test 8: Lane name strings
    // ====================================================================
    std::cout << "\nTest 8: Lane name strings" << std::endl;
    {
        print_test_result("LEGACY → \"Legacy\"",
            std::string(get_lane_name(ProtocolLane::LEGACY)) == "Legacy");
        print_test_result("STATELESS → \"Stateless\"",
            std::string(get_lane_name(ProtocolLane::STATELESS)) == "Stateless");
        print_test_result("UNKNOWN → \"Unknown\"",
            std::string(get_lane_name(ProtocolLane::UNKNOWN)) == "Unknown");
    }

    // ====================================================================
    // Test 9: Mirror-mapping round-trip for all key opcodes
    // ====================================================================
    std::cout << "\nTest 9: Mirror-mapping round-trip per port" << std::endl;
    {
        uint8_t key_opcodes[] = {
            MinerLLP::SUBMIT_BLOCK, MinerLLP::SET_CHANNEL, MinerLLP::GET_BLOCK,
            MinerLLP::GET_ROUND, MinerLLP::MINER_READY,
            MinerLLP::PRIME_BLOCK_AVAILABLE, MinerLLP::HASH_BLOCK_AVAILABLE
        };
        const char* names[] = {
            "SUBMIT_BLOCK", "SET_CHANNEL", "GET_BLOCK",
            "GET_ROUND", "MINER_READY",
            "PRIME_BLOCK_AVAILABLE", "HASH_BLOCK_AVAILABLE"
        };
        
        bool all_roundtrip = true;
        for (size_t i = 0; i < sizeof(key_opcodes); ++i) {
            uint16_t mirrored = MinerLLP::MirrorOpcode(key_opcodes[i]);
            uint8_t unmirrored = MinerLLP::UnmirrorOpcode(mirrored);
            if (unmirrored != key_opcodes[i]) {
                std::cout << "  [FAIL] Round-trip failed for " << names[i] << std::endl;
                all_roundtrip = false;
            }
            if (!MinerLLP::IsStatelessOpcode(mirrored)) {
                std::cout << "  [FAIL] MirrorOpcode(" << names[i] << ") not in stateless range" << std::endl;
                all_roundtrip = false;
            }
        }
        print_test_result("All key opcodes survive mirror/unmirror round-trip", all_roundtrip);
    }

    // ====================================================================
    // Summary
    // ====================================================================
    std::cout << "\n========================================" << std::endl;
    std::cout << "Test Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Tests run:    " << tests_run << std::endl;
    std::cout << "Tests passed: " << tests_passed << std::endl;
    std::cout << "Tests failed: " << tests_failed << std::endl;
    std::cout << "Success rate: " << (tests_run > 0 ? (100 * tests_passed / tests_run) : 0) << "%" << std::endl;
    std::cout << "========================================" << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
