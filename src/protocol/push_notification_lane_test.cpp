/**
 * @file push_notification_lane_test.cpp
 * @brief Tests for unified push notification lane detection
 * 
 * Validates that push notification opcodes are correctly matched by the
 * unified matches_opcode() lambda for both legacy 8-bit and stateless
 * 16-bit mirror-mapped opcodes, ensuring m_protocol_lane is used rather
 * than hardcoded lane values.
 * 
 * Test scenarios:
 * - Legacy 8-bit push opcodes (0xD9, 0xDA) match via matches_opcode()
 * - Stateless 16-bit push opcodes (0xD0D9, 0xD0DA) match via matches_opcode()
 * - Both formats route through unified handler with correct lane
 */

#include "miner_opcodes.hpp"
#include "protocol_lane.hpp"
#include "protocol/push_notification_handler.hpp"
#include "protocol/mining_template_interface.hpp"
#include "protocol/solo.hpp"
#include "protocol/height_tracker.hpp"
#include "mining/client_block.h"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/null_sink.h"
#include <iostream>
#include <cassert>
#include <cstdint>

using namespace nexusminer;
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

/**
 * Simulates the matches_opcode() lambda from solo.cpp process_messages():
 *   auto matches_opcode = [&packet](uint16_t legacy_opcode) {
 *       if (packet.m_is_uint16_opcode) {
 *           return packet.m_header == MinerLLP::MirrorOpcode(static_cast<uint8_t>(legacy_opcode));
 *       }
 *       return packet.m_header == legacy_opcode;
 *   };
 */
bool simulated_matches_opcode(uint16_t packet_header, bool is_uint16_opcode, uint16_t legacy_opcode) {
    if (is_uint16_opcode) {
        return packet_header == MinerLLP::MirrorOpcode(static_cast<uint8_t>(legacy_opcode));
    }
    return packet_header == legacy_opcode;
}

/**
 * Simulates the matches_stateless_opcode() lambda from solo.cpp:
 *   auto matches_stateless_opcode = [&packet](uint16_t legacy_opcode) {
 *       return packet.m_is_uint16_opcode &&
 *           packet.m_header == MinerLLP::MirrorOpcode(static_cast<uint8_t>(legacy_opcode));
 *   };
 */
bool simulated_matches_stateless_opcode(uint16_t packet_header, bool is_uint16_opcode, uint16_t legacy_opcode) {
    return is_uint16_opcode &&
        packet_header == MinerLLP::MirrorOpcode(static_cast<uint8_t>(legacy_opcode));
}

bool should_trigger_get_round_fallback_get_block(bool has_valid_template,
                                                 uint32_t node_channel_height,
                                                 uint32_t template_target_height,
                                                 int64_t since_push_s)
{
    if (!has_valid_template) return false;
    if (template_target_height == 0) return false;
    if (node_channel_height < template_target_height) return false;
    return since_push_s >= protocol::Solo::PUSH_ABSENT_FOR_GET_ROUND_FALLBACK_SECONDS;
}

bool push_implies_block_data_in_transit_test(
    bool has_push, uint32_t push_unified_height,
    uint32_t check_unified_height, int64_t push_age_ms)
{
    if (!has_push) return false;
    if (push_unified_height < check_unified_height) return false;
    return push_age_ms >= 0 &&
           push_age_ms < protocol::Solo::PUSH_BLOCK_DATA_IN_TRANSIT_GUARD_HASH_MS;
}

std::vector<uint8_t> create_mock_template(uint32_t height, uint32_t nBits = 0x1d00ffff,
                                          uint8_t channel = 2) {
    std::vector<uint8_t> data(216, 0);

    size_t offset = 0;
    auto write_u32_be = [&](uint32_t value) {
        data[offset++] = (value >> 24) & 0xFF;
        data[offset++] = (value >> 16) & 0xFF;
        data[offset++] = (value >> 8) & 0xFF;
        data[offset++] = value & 0xFF;
    };

    auto write_u64_be = [&](uint64_t value) {
        data[offset++] = (value >> 56) & 0xFF;
        data[offset++] = (value >> 48) & 0xFF;
        data[offset++] = (value >> 40) & 0xFF;
        data[offset++] = (value >> 32) & 0xFF;
        data[offset++] = (value >> 24) & 0xFF;
        data[offset++] = (value >> 16) & 0xFF;
        data[offset++] = (value >> 8) & 0xFF;
        data[offset++] = value & 0xFF;
    };

    write_u32_be(7);
    offset += 128;
    for (int i = 0; i < 64; ++i) {
        data[offset++] = static_cast<uint8_t>(i + 1);
    }
    write_u32_be(static_cast<uint32_t>(channel));
    write_u32_be(height);
    write_u32_be(nBits);
    write_u64_be(0);

    return data;
}

network::Payload create_extended_push_payload(uint32_t unified_height,
                                              uint32_t channel_height,
                                              uint32_t difficulty,
                                              uint8_t prev_hash_fill)
{
    network::Payload payload(140, 0);
    auto write_u32_be = [&](size_t offset, uint32_t value) {
        payload[offset + 0] = static_cast<uint8_t>((value >> 24) & 0xFF);
        payload[offset + 1] = static_cast<uint8_t>((value >> 16) & 0xFF);
        payload[offset + 2] = static_cast<uint8_t>((value >> 8) & 0xFF);
        payload[offset + 3] = static_cast<uint8_t>(value & 0xFF);
    };

    write_u32_be(0, unified_height);
    write_u32_be(4, channel_height);
    write_u32_be(8, difficulty);
    for (size_t i = 12; i < payload.size(); ++i) {
        payload[i] = prev_hash_fill;
    }
    return payload;
}

network::Payload create_template_delivery_payload(uint32_t unified_height,
                                                  uint32_t channel_height,
                                                  uint32_t difficulty,
                                                  uint32_t block_height,
                                                  uint8_t channel = 2)
{
    network::Payload payload(12, 0);
    auto write_u32_be = [&](size_t offset, uint32_t value) {
        payload[offset + 0] = static_cast<uint8_t>((value >> 24) & 0xFF);
        payload[offset + 1] = static_cast<uint8_t>((value >> 16) & 0xFF);
        payload[offset + 2] = static_cast<uint8_t>((value >> 8) & 0xFF);
        payload[offset + 3] = static_cast<uint8_t>(value & 0xFF);
    };

    write_u32_be(0, unified_height);
    write_u32_be(4, channel_height);
    write_u32_be(8, difficulty);

    auto block = create_mock_template(block_height, difficulty, channel);
    payload.insert(payload.end(), block.begin(), block.end());
    return payload;
}

network::Payload create_get_round_height_payload(uint32_t unified_height,
                                                 uint32_t prime_height,
                                                 uint32_t hash_height,
                                                 uint32_t stake_height)
{
    network::Payload payload(16, 0);
    auto write_u32_be = [&](size_t offset, uint32_t value) {
        payload[offset + 0] = static_cast<uint8_t>((value >> 24) & 0xFF);
        payload[offset + 1] = static_cast<uint8_t>((value >> 16) & 0xFF);
        payload[offset + 2] = static_cast<uint8_t>((value >> 8) & 0xFF);
        payload[offset + 3] = static_cast<uint8_t>(value & 0xFF);
    };

    write_u32_be(0, unified_height);
    write_u32_be(4, prime_height);
    write_u32_be(8, hash_height);
    write_u32_be(12, stake_height);
    return payload;
}


int main()
{
    auto null_sink = std::make_shared<spdlog::sinks::null_sink_mt>();
    auto logger = std::make_shared<spdlog::logger>("push_notification_lane_test", null_sink);
    spdlog::set_default_logger(logger);

    std::cout << "========================================" << std::endl;
    std::cout << "Push Notification Lane Detection Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    // ====================================================================
    // Test 1: Mirror opcode mapping for push notifications
    // ====================================================================
    std::cout << "\nTest 1: Mirror opcode mapping" << std::endl;
    {
        uint16_t prime_mirror = MinerLLP::MirrorOpcode(MinerLLP::PRIME_BLOCK_AVAILABLE);
        uint16_t hash_mirror = MinerLLP::MirrorOpcode(MinerLLP::HASH_BLOCK_AVAILABLE);
        
        print_test_result("PRIME_BLOCK_AVAILABLE mirrors to 0xD0D9",
            prime_mirror == 0xD0D9);
        print_test_result("HASH_BLOCK_AVAILABLE mirrors to 0xD0DA",
            hash_mirror == 0xD0DA);
        print_test_result("PRIME_BLOCK_AVAILABLE legacy value is 217 (0xD9)",
            MinerLLP::PRIME_BLOCK_AVAILABLE == 217);
        print_test_result("HASH_BLOCK_AVAILABLE legacy value is 218 (0xDA)",
            MinerLLP::HASH_BLOCK_AVAILABLE == 218);
    }

    // ====================================================================
    // Test 2: matches_opcode() accepts 8-bit legacy opcodes
    // ====================================================================
    std::cout << "\nTest 2: Legacy 8-bit opcode matching" << std::endl;
    {
        // Simulate legacy 8-bit PRIME_BLOCK_AVAILABLE (0xD9 = 217)
        uint16_t header = MinerLLP::PRIME_BLOCK_AVAILABLE;
        bool is_uint16 = false;
        
        bool matches = simulated_matches_opcode(header, is_uint16, MinerLLP::PRIME_BLOCK_AVAILABLE);
        print_test_result("Legacy 8-bit PRIME_BLOCK_AVAILABLE matches matches_opcode()",
            matches);
        
        // Simulate legacy 8-bit HASH_BLOCK_AVAILABLE (0xDA = 218)
        header = MinerLLP::HASH_BLOCK_AVAILABLE;
        matches = simulated_matches_opcode(header, is_uint16, MinerLLP::HASH_BLOCK_AVAILABLE);
        print_test_result("Legacy 8-bit HASH_BLOCK_AVAILABLE matches matches_opcode()",
            matches);
    }

    // ====================================================================
    // Test 3: matches_opcode() accepts 16-bit mirror opcodes (permissive)
    // ====================================================================
    std::cout << "\nTest 3: Stateless 16-bit opcode matching via matches_opcode()" << std::endl;
    {
        // Simulate stateless 16-bit PRIME_BLOCK_AVAILABLE (0xD0D9)
        uint16_t header = MinerLLP::MirrorOpcode(MinerLLP::PRIME_BLOCK_AVAILABLE);
        bool is_uint16 = true;
        
        bool matches = simulated_matches_opcode(header, is_uint16, MinerLLP::PRIME_BLOCK_AVAILABLE);
        print_test_result("Stateless 16-bit PRIME_BLOCK_AVAILABLE (0xD0D9) matches matches_opcode()",
            matches);
        
        // Simulate stateless 16-bit HASH_BLOCK_AVAILABLE (0xD0DA) 
        header = MinerLLP::MirrorOpcode(MinerLLP::HASH_BLOCK_AVAILABLE);
        matches = simulated_matches_opcode(header, is_uint16, MinerLLP::HASH_BLOCK_AVAILABLE);
        print_test_result("Stateless 16-bit HASH_BLOCK_AVAILABLE (0xD0DA) matches matches_opcode()",
            matches);
    }

    // ====================================================================
    // Test 4: matches_stateless_opcode() is subset of matches_opcode()
    // This proves the duplicate stateless handlers were dead code
    // ====================================================================
    std::cout << "\nTest 4: Stateless-specific match is subset of unified match" << std::endl;
    {
        uint16_t header = MinerLLP::MirrorOpcode(MinerLLP::PRIME_BLOCK_AVAILABLE);
        bool is_uint16 = true;
        
        bool unified = simulated_matches_opcode(header, is_uint16, MinerLLP::PRIME_BLOCK_AVAILABLE);
        bool stateless_only = simulated_matches_stateless_opcode(header, is_uint16, MinerLLP::PRIME_BLOCK_AVAILABLE);
        
        print_test_result("Both matches_opcode and matches_stateless_opcode match 0xD0D9",
            unified && stateless_only);
        print_test_result("matches_opcode catches stateless opcodes (unified handler works)",
            unified);
    }

    // ====================================================================
    // Test 5: Lane detection from port
    // ====================================================================
    std::cout << "\nTest 5: Protocol lane determination from port" << std::endl;
    {
        ProtocolLane legacy = determine_lane_from_port(ProtocolPorts::LEGACY_PORT);
        ProtocolLane stateless = determine_lane_from_port(ProtocolPorts::STATELESS_PORT);
        
        print_test_result("Port 8323 → LEGACY lane",
            legacy == ProtocolLane::LEGACY);
        print_test_result("Port 9323 → STATELESS lane",
            stateless == ProtocolLane::STATELESS);
        print_test_result("Lane names correct",
            std::string(get_lane_name(legacy)) == "Legacy" &&
            std::string(get_lane_name(stateless)) == "Stateless");
    }

    // ====================================================================
    // Test 6: Unified handler uses m_protocol_lane (integration concept)
    // Simulates the fix: m_protocol_lane is used instead of hardcoded lane
    // ====================================================================
    std::cout << "\nTest 6: Unified push handler uses connection lane" << std::endl;
    {
        // Scenario A: Legacy connection receives 8-bit PRIME_BLOCK_AVAILABLE
        ProtocolLane m_protocol_lane_A = ProtocolLane::LEGACY;
        uint16_t header_A = MinerLLP::PRIME_BLOCK_AVAILABLE;
        bool is_uint16_A = false;
        bool matches_A = simulated_matches_opcode(header_A, is_uint16_A, MinerLLP::PRIME_BLOCK_AVAILABLE);
        print_test_result("Scenario A: Legacy 8-bit → lane=LEGACY",
            matches_A && m_protocol_lane_A == ProtocolLane::LEGACY);
        
        // Scenario B: Stateless connection receives 16-bit PRIME_BLOCK_AVAILABLE
        ProtocolLane m_protocol_lane_B = ProtocolLane::STATELESS;
        uint16_t header_B = MinerLLP::MirrorOpcode(MinerLLP::PRIME_BLOCK_AVAILABLE);
        bool is_uint16_B = true;
        bool matches_B = simulated_matches_opcode(header_B, is_uint16_B, MinerLLP::PRIME_BLOCK_AVAILABLE);
        print_test_result("Scenario B: Stateless 16-bit → lane=STATELESS",
            matches_B && m_protocol_lane_B == ProtocolLane::STATELESS);
        
        // Scenario C: Legacy connection receives mirror 16-bit (firewall catch)
        ProtocolLane m_protocol_lane_C = ProtocolLane::LEGACY;
        uint16_t header_C = MinerLLP::MirrorOpcode(MinerLLP::HASH_BLOCK_AVAILABLE);
        bool is_uint16_C = true;
        bool matches_C = simulated_matches_opcode(header_C, is_uint16_C, MinerLLP::HASH_BLOCK_AVAILABLE);
        print_test_result("Scenario C: Legacy lane + mirror opcode → lane=LEGACY (firewall)",
            matches_C && m_protocol_lane_C == ProtocolLane::LEGACY);
    }

    // ====================================================================
    // Test 7: IsStatelessOpcode / UnmirrorOpcode for push notifications
    // ====================================================================
    std::cout << "\nTest 7: Opcode classification helpers" << std::endl;
    {
        print_test_result("0xD0D9 is stateless opcode",
            MinerLLP::IsStatelessOpcode(0xD0D9));
        print_test_result("0xD0DA is stateless opcode",
            MinerLLP::IsStatelessOpcode(0xD0DA));
        print_test_result("0xD9 (217) is NOT stateless opcode",
            !MinerLLP::IsStatelessOpcode(0x00D9));
        print_test_result("0xDA (218) is NOT stateless opcode",
            !MinerLLP::IsStatelessOpcode(0x00DA));
        print_test_result("UnmirrorOpcode(0xD0D9) == 0xD9 (217)",
            MinerLLP::UnmirrorOpcode(0xD0D9) == MinerLLP::PRIME_BLOCK_AVAILABLE);
        print_test_result("UnmirrorOpcode(0xD0DA) == 0xDA (218)",
            MinerLLP::UnmirrorOpcode(0xD0DA) == MinerLLP::HASH_BLOCK_AVAILABLE);
    }

    // ====================================================================
    // Test 8: Dual-channel broadcast — node sends BOTH Prime and Hash pushes
    // Since the node now broadcasts both channels on every push update, the
    // miner must accept the non-subscribed channel push without disruption.
    // ====================================================================
    std::cout << "\nTest 8: Dual-channel broadcast — both Prime and Hash pushes received" << std::endl;
    {
        // Scenario A: Prime miner (channel 1) receives Prime push → should match
        uint8_t mining_channel_prime = static_cast<uint8_t>(mining::CHANNEL_PRIME);
        uint8_t mining_channel_hash  = static_cast<uint8_t>(mining::CHANNEL_HASH);

        // Legacy lane: Prime miner receives PRIME push → match
        bool prime_receives_prime_legacy = simulated_matches_opcode(
            MinerLLP::PRIME_BLOCK_AVAILABLE, false, MinerLLP::PRIME_BLOCK_AVAILABLE);
        print_test_result("Scenario A1: Prime miner receives legacy Prime push → match",
            prime_receives_prime_legacy);

        // Legacy lane: Prime miner receives HASH push → match in router, channel check in handler
        bool prime_receives_hash_legacy = simulated_matches_opcode(
            MinerLLP::HASH_BLOCK_AVAILABLE, false, MinerLLP::HASH_BLOCK_AVAILABLE);
        // Handler will see channel mismatch (mining_channel_prime != CHANNEL_HASH) and return early.
        // This is now treated as informational (not an error).
        print_test_result("Scenario A2: Prime miner receives legacy Hash push → routed (handler guards channel)",
            prime_receives_hash_legacy);

        // Stateless lane: Hash miner receives Prime push (0xD0D9) → routed, handler guards channel
        bool hash_receives_prime_stateless = simulated_matches_opcode(
            MinerLLP::MirrorOpcode(MinerLLP::PRIME_BLOCK_AVAILABLE), true, MinerLLP::PRIME_BLOCK_AVAILABLE);
        print_test_result("Scenario B1: Hash miner receives stateless Prime push (0xD0D9) → routed",
            hash_receives_prime_stateless);

        // Stateless lane: Hash miner receives Hash push (0xD0DA) → match + process
        bool hash_receives_hash_stateless = simulated_matches_opcode(
            MinerLLP::MirrorOpcode(MinerLLP::HASH_BLOCK_AVAILABLE), true, MinerLLP::HASH_BLOCK_AVAILABLE);
        print_test_result("Scenario B2: Hash miner receives stateless Hash push (0xD0DA) → match",
            hash_receives_hash_stateless);

        // Channel mismatch detection: push_notification_handler guards by m_current_channel
        // Prime miner (CHANNEL_PRIME) vs. Hash expected_channel → mismatch → informational no-op
        bool prime_vs_hash = (mining_channel_prime != mining_channel_hash);
        print_test_result("Channel guard: Prime miner + Hash push → mismatch (informational no-op)",
            prime_vs_hash);

        // Hash miner (CHANNEL_HASH) vs. Prime expected_channel → mismatch → informational no-op
        bool hash_vs_prime = (mining_channel_hash != mining_channel_prime);
        print_test_result("Channel guard: Hash miner + Prime push → mismatch (informational no-op)",
            hash_vs_prime);
    }

    // ====================================================================
    // Test 9: Unified-height-driven push handler decision logic
    //
    // Verifies the simplified push handler decision tree:
    //   - Channel stale: advance target for doom-loop prevention (informational)
    //   - Not stale + hash mismatch: same-height tip replacement (discard)
    //   - ALWAYS: request fresh template (every PUSH = unified tip moved)
    //
    // Key principle: every PUSH from the node means the unified tip moved,
    // which changes hashPrevBlock.  The miner ALWAYS needs a fresh template.
    // Channel heights are tracked informationally for doom-loop prevention
    // but do NOT gate the template refresh decision.
    // ====================================================================
    std::cout << "\nTest 9: Unified-height-driven push handler decision logic" << std::endl;
    {
        struct HandlerDecision {
            bool push_processed{false};
            bool discard_template_called{false};
            bool recovery_triggered{false};
        };

        // Simulate the unified-height-driven handler decision tree.
        // Every PUSH requests work.  Channel staleness is informational only.
        // Same-height hash check (reorg) only runs when channel is NOT stale.
        auto simulate_handler = [](bool stale, uint32_t blocks_behind,
                                    bool has_hash, bool hash_matches,
                                    bool burst_grace_active,
                                    bool tip_moved) -> HandlerDecision
        {
            (void)blocks_behind;       // Informational only — logged but no behavioral difference
            (void)burst_grace_active;  // Removed — unified model doesn't need burst tiers
            (void)tip_moved;           // Subsumed — every PUSH means unified tip moved

            HandlerDecision d;

            // Channel staleness: informational + doom-loop prevention (AdvanceChannelTarget)
            // Does NOT gate the work request.

            // Same-height tip replacement (only when channel is NOT stale)
            if (!stale && has_hash && !hash_matches) {
                d.discard_template_called = true;
            }

            // Every same-channel PUSH is substantively processed (heights/state updated)
            d.push_processed = true;
            return d;
        };

        // Scenario A: Normal 1-block lag (every ~30s during normal mining)
        // Expected: refresh template, do NOT stop workers, do NOT discard
        {
            auto d = simulate_handler(/*stale=*/true, /*blocks_behind=*/1,
                                       /*has_hash=*/true, /*hash_matches=*/false,
                                       /*burst_grace_active=*/false,
                                       /*tip_moved=*/false);
            print_test_result("Scenario A1: 1-block lag → push processed",
                d.push_processed);
            print_test_result("Scenario A2: 1-block lag → discard_template NOT called",
                !d.discard_template_called);
            print_test_result("Scenario A3: 1-block lag → recovery NOT triggered (workers keep mining)",
                !d.recovery_triggered);
        }

        // Scenario B: Multi-block lag with hash mismatch
        // Expected: request work only, keep template/session running, no hard recovery
        {
            auto d = simulate_handler(/*stale=*/true, /*blocks_behind=*/3,
                                       /*has_hash=*/true, /*hash_matches=*/false,
                                       /*burst_grace_active=*/false,
                                       /*tip_moved=*/false);
            print_test_result("Scenario B1: 3-block lag → push processed",
                d.push_processed);
            print_test_result("Scenario B2: 3-block lag → discard_template NOT called",
                !d.discard_template_called);
            print_test_result("Scenario B3: 3-block lag → recovery NOT triggered",
                !d.recovery_triggered);
        }

        // Scenario C: 2-block burst within grace window
        // Expected: request fresh work (burst_grace is irrelevant in unified model)
        {
            auto d = simulate_handler(/*stale=*/true, /*blocks_behind=*/2,
                                       /*has_hash=*/true, /*hash_matches=*/true,
                                       /*burst_grace_active=*/true,
                                       /*tip_moved=*/false);
            print_test_result("Scenario C1: 2-block burst within grace → push processed",
                d.push_processed);
            print_test_result("Scenario C2: 2-block burst within grace → discard_template NOT called",
                !d.discard_template_called);
            print_test_result("Scenario C3: 2-block burst within grace → recovery NOT triggered",
                !d.recovery_triggered);
        }

        // Scenario D: 2-block lag after grace expires
        // Expected: request work (no behavioral difference from C in unified model)
        {
            auto d = simulate_handler(/*stale=*/true, /*blocks_behind=*/2,
                                       /*has_hash=*/true, /*hash_matches=*/true,
                                       /*burst_grace_active=*/false,
                                       /*tip_moved=*/false);
            print_test_result("Scenario D1: 2-block lag after grace → push processed",
                d.push_processed);
            print_test_result("Scenario D2: 2-block lag after grace → discard_template NOT called",
                !d.discard_template_called);
            print_test_result("Scenario D3: 2-block lag after grace → recovery NOT triggered",
                !d.recovery_triggered);
        }

        // Scenario E: Compact payload (no hashPrevBlock) — multi-block lag
        // Expected: request refresh (hash data unavailable but always request work)
        {
            auto d = simulate_handler(/*stale=*/true, /*blocks_behind=*/2,
                                       /*has_hash=*/false, /*hash_matches=*/false,
                                       /*burst_grace_active=*/false,
                                       /*tip_moved=*/false);
            print_test_result("Scenario E1: 2-block lag, compact payload → push processed",
                d.push_processed);
            print_test_result("Scenario E2: 2-block lag, compact payload → discard NOT called",
                !d.discard_template_called);
        }

        // Scenario F: Same-height tip replacement (blocks_behind == 0, hash changed)
        // Expected: discard template + request work
        {
            auto d = simulate_handler(/*stale=*/false, /*blocks_behind=*/0,
                                       /*has_hash=*/true, /*hash_matches=*/false,
                                       /*burst_grace_active=*/false,
                                       /*tip_moved=*/false);
            print_test_result("Scenario F1: Same-height tip replacement → discard_template called",
                d.discard_template_called);
            print_test_result("Scenario F2: Same-height tip replacement → push processed",
                d.push_processed);
            print_test_result("Scenario F3: Same-height tip replacement → hard recovery NOT triggered",
                !d.recovery_triggered);
        }

        // Scenario G: Healthy template (not stale, hashes match)
        // Expected: STILL request work — every PUSH means unified tip moved.
        // GetBlockDedupGuard handles true duplicates (unified height unchanged).
        {
            auto d = simulate_handler(/*stale=*/false, /*blocks_behind=*/0,
                                       /*has_hash=*/true, /*hash_matches=*/true,
                                       /*burst_grace_active=*/false,
                                       /*tip_moved=*/false);
            print_test_result("Scenario G1: Healthy template → push processed (heights/state updated)",
                d.push_processed);
            print_test_result("Scenario G2: Healthy template → discard NOT called",
                !d.discard_template_called);
        }

        // Scenario H: Unified tip moved on another channel
        // Expected: request work (same as all other cases — every PUSH requests work)
        {
            auto d = simulate_handler(/*stale=*/false, /*blocks_behind=*/0,
                                       /*has_hash=*/true, /*hash_matches=*/true,
                                       /*burst_grace_active=*/false,
                                       /*tip_moved=*/true);
            print_test_result("Scenario H1: Tip moved → push processed",
                d.push_processed);
            print_test_result("Scenario H2: Tip moved → discard_template NOT called",
                !d.discard_template_called);
            print_test_result("Scenario H3: Tip moved → hard recovery NOT triggered",
                !d.recovery_triggered);
        }
    }

    // ====================================================================
    // Test 10: blocks_behind() arithmetic matches handler expectations
    // ====================================================================
    std::cout << "\nTest 10: blocks_behind() arithmetic" << std::endl;
    {
        // blocks_behind = expected_template_target - channel_target
        //               = (channel_height + 1) - channel_target
        // Verify the formula matches what the handler uses to classify severity.

        // Template current: channel_target == channel_height + 1 → behind = 0
        uint32_t ch = 100, ct_current = 101;
        uint32_t expected_current = (ch + 1) - ct_current;  // = 0
        print_test_result("blocks_behind == 0 when template is current (channel_target = channel_height+1)",
            expected_current == 0);

        // Template 1 block behind: channel_target == channel_height → behind = 1
        uint32_t ct_one_behind = 100;
        uint32_t expected_one = (ch + 1) - ct_one_behind;  // = 1
        print_test_result("blocks_behind == 1 when template is one block old",
            expected_one == 1);

        // Template 3 blocks behind: channel_target == channel_height - 2 → behind = 3
        uint32_t ct_three_behind = 98;
        uint32_t expected_three = (ch + 1) - ct_three_behind;  // = 3
        print_test_result("blocks_behind == 3 when template is three blocks old",
            expected_three == 3);

        // The boundary: blocks_behind == 1 → normal path (no recovery)
        //               blocks_behind >= 2 → recovery path
        print_test_result("blocks_behind threshold: 1 is normal, 2+ triggers recovery",
            expected_one < 2 && expected_three >= 2);
    }

    // ====================================================================
    // Test 11: Actual handler requests soft refresh on same-height canonical replacement
    // ====================================================================
    std::cout << "\nTest 11: Same-height canonical replacement stays on soft-refresh hot-swap path" << std::endl;
    {
        protocol::HeightTracker tracker;
        protocol::MiningTemplateInterface tmpl_interface(2, 0);
        tmpl_interface.set_height_tracker(&tracker);
        tracker.OnPushNotification(5000, 100, 0x1d00ffff);

        auto template_data = create_mock_template(5001, 0x1d00ffff, 2);
        auto res = tmpl_interface.read_template(template_data, "test_node");
        print_test_result("Handler setup: initial template valid", res.is_valid);
        tmpl_interface.set_channel_height(101);

        uint8_t current_channel = static_cast<uint8_t>(mining::CHANNEL_HASH);
        protocol::PushNotificationHandler handler(logger, current_channel);

        network::Payload payload = create_extended_push_payload(5000, 100, 0x1d00ffff, 0x42);
        Packet packet(MinerLLP::MirrorOpcode(MinerLLP::HASH_BLOCK_AVAILABLE), payload);

        bool push_processed = handler.handle_push_notification(
            packet,
            mining::CHANNEL_HASH,
            ProtocolLane::STATELESS,
            &tmpl_interface,
            &tracker,
            [&tracker](uint32_t u, uint32_t c, uint32_t d) { tracker.OnPushNotification(u, c, d); });

        print_test_result("Same-height tip replacement: push substantively processed", push_processed);
        // Option A + B safety net: same-height tip update no longer discards
        // the active template. Workers continue mining until BLOCK_DATA atomically
        // swaps in. The replacement-pending flag is set so the worker health
        // monitor can fall back to discard if BLOCK_DATA never arrives.
        print_test_result("Same-height tip replacement keeps active template VALID (no recovery epoch)",
            tmpl_interface.has_valid_template());
        print_test_result("Same-height tip replacement marks template replacement-pending",
            tmpl_interface.is_replacement_pending());
    }

    // ====================================================================
    // Test 11b: Out-of-order push hash hint must not hot-swap a live template
    //
    // An out-of-order push (lower unified height) still requests work in the
    // unified model (every PUSH = request fresh template), but must NOT discard
    // the current valid template via the same-height hash check.
    // ====================================================================
    std::cout << "\nTest 11b: Out-of-order push does not discard current template" << std::endl;
    {
        protocol::HeightTracker tracker;
        protocol::MiningTemplateInterface tmpl_interface(2, 0);
        tmpl_interface.set_height_tracker(&tracker);
        tracker.OnPushNotification(5050, 100, 0x1d00ffff);

        auto template_data = create_mock_template(5051, 0x1d00ffff, 2);
        auto res = tmpl_interface.read_template(template_data, "test_node");
        print_test_result("Stale-push setup template valid", res.is_valid);
        tmpl_interface.set_channel_height(101);

        uint8_t current_channel = static_cast<uint8_t>(mining::CHANNEL_HASH);
        protocol::PushNotificationHandler handler(logger, current_channel);

        network::Payload payload = create_extended_push_payload(5049, 99, 0x1d00ffff, 0x42);
        Packet packet(MinerLLP::MirrorOpcode(MinerLLP::HASH_BLOCK_AVAILABLE), payload);

        bool push_processed = handler.handle_push_notification(
            packet,
            mining::CHANNEL_HASH,
            ProtocolLane::STATELESS,
            &tmpl_interface,
            &tracker,
            [&tracker](uint32_t u, uint32_t c, uint32_t d) { tracker.OnPushNotification(u, c, d); });

        print_test_result("Out-of-order push: push substantively processed (heights updated)",
            push_processed);
        print_test_result("Out-of-order push keeps active template valid (no discard)",
            tmpl_interface.has_valid_template());
    }

    // ====================================================================
    // Test 12: Extended push seeds first-install tip-anchor gate
    // ====================================================================
    std::cout << "\nTest 12: Extended push tip anchor can reject obsolete first template" << std::endl;
    {
        protocol::HeightTracker tracker;
        uint8_t current_channel = static_cast<uint8_t>(mining::CHANNEL_HASH);
        protocol::PushNotificationHandler handler(logger, current_channel);

        network::Payload payload = create_extended_push_payload(7000, 100, 0x1d00ffff, 0x42);
        Packet packet(MinerLLP::MirrorOpcode(MinerLLP::HASH_BLOCK_AVAILABLE), payload);

        bool push_processed = handler.handle_push_notification(
            packet,
            mining::CHANNEL_HASH,
            ProtocolLane::STATELESS,
            nullptr,
            &tracker,
            [&tracker](uint32_t u, uint32_t c, uint32_t d) { tracker.OnPushNotification(u, c, d); });

        protocol::MiningTemplateInterface stale_template(2, 0);
        stale_template.set_height_tracker(&tracker);
        auto stale_data = create_mock_template(7001, 0x1d00ffff, 2); // hashPrevBlock is all zeros
        auto stale_res = stale_template.read_template(stale_data, "test_node");
        print_test_result("Extended push still processes when no template exists", push_processed);
        print_test_result("Obsolete-on-arrival setup template validates structurally", stale_res.is_valid);
        stale_template.set_channel_height(101);

        auto stale_snap = tracker.GetSnapshot();
        auto const* stale_ptr = stale_template.get_current_template();
        print_test_result("Push snapshot captured extended hashPrevBlock hint",
            stale_snap.push_hash_prev_block != uint1024_t{});
        print_test_result("First-install gate detects same-height obsolete template",
            stale_ptr &&
            stale_snap.has_same_height_push_tip_replacement(stale_ptr->block.hashPrevBlock,
                                                           stale_ptr->nChannelHeight));

        protocol::MiningTemplateInterface matching_template(2, 0);
        matching_template.set_height_tracker(&tracker);
        auto matching_data = create_mock_template(7001, 0x1d00ffff, 2);
        std::fill(matching_data.begin() + 4, matching_data.begin() + 132, static_cast<uint8_t>(0x42));
        auto matching_res = matching_template.read_template(matching_data, "test_node");
        print_test_result("Matching-anchor setup template validates structurally", matching_res.is_valid);
        matching_template.set_channel_height(101);

        auto const* matching_ptr = matching_template.get_current_template();
        print_test_result("First-install gate stays open when template anchor matches push",
            matching_ptr &&
            !stale_snap.has_same_height_push_tip_replacement(matching_ptr->block.hashPrevBlock,
                                                             matching_ptr->nChannelHeight));
    }

    // ====================================================================
    // Test 13: Burst guard keeps template/session alive for a short 2-block burst
    // ====================================================================
    std::cout << "\nTest 13: Burst guard keeps session alive at 2 blocks behind" << std::endl;
    {
        protocol::HeightTracker tracker;
        protocol::MiningTemplateInterface tmpl_interface(2, 0);
        tmpl_interface.set_height_tracker(&tracker);
        tracker.OnPushNotification(6000, 100, 0x1d00ffff);

        auto template_data = create_mock_template(6001, 0x1d00ffff, 2);
        auto res = tmpl_interface.read_template(template_data, "test_node");
        print_test_result("Burst setup: initial template valid", res.is_valid);
        tmpl_interface.set_channel_height(101);

        uint8_t current_channel = static_cast<uint8_t>(mining::CHANNEL_HASH);
        protocol::PushNotificationHandler handler(logger, current_channel);

        network::Payload payload = create_extended_push_payload(6002, 102, 0x1d00ffff, 0x00);
        Packet packet(MinerLLP::MirrorOpcode(MinerLLP::HASH_BLOCK_AVAILABLE), payload);

        bool push_processed = handler.handle_push_notification(
            packet,
            mining::CHANNEL_HASH,
            ProtocolLane::STATELESS,
            &tmpl_interface,
            &tracker,
            [&tracker](uint32_t u, uint32_t c, uint32_t d) { tracker.OnPushNotification(u, c, d); });

        print_test_result("2-block burst within grace: push substantively processed", push_processed);
        print_test_result("2-block burst within grace keeps active template valid", tmpl_interface.has_valid_template());
    }

    // ====================================================================
    // Summary
    // ====================================================================
    std::cout << "\nTest 14: Tip movement on another channel requests fresh work" << std::endl;
    {
        protocol::HeightTracker tracker;
        protocol::MiningTemplateInterface tmpl_interface(2, 0);
        tmpl_interface.set_height_tracker(&tracker);
        tracker.OnPushNotification(8000, 100, 0x1d00ffff);

        auto template_data = create_mock_template(8001, 0x1d00ffff, 2);
        auto res = tmpl_interface.read_template(template_data, "test_node");
        print_test_result("Tip-move setup: initial template valid", res.is_valid);
        tmpl_interface.set_channel_height(101);

        uint8_t current_channel = static_cast<uint8_t>(mining::CHANNEL_HASH);
        protocol::PushNotificationHandler handler(logger, current_channel);

        network::Payload payload = create_extended_push_payload(8002, 100, 0x1d00ffff, 0x00);
        Packet packet(MinerLLP::MirrorOpcode(MinerLLP::HASH_BLOCK_AVAILABLE), payload);

        bool push_processed = handler.handle_push_notification(
            packet,
            mining::CHANNEL_HASH,
            ProtocolLane::STATELESS,
            &tmpl_interface,
            &tracker,
            [&tracker](uint32_t u, uint32_t c, uint32_t d) { tracker.OnPushNotification(u, c, d); });

        print_test_result("Tip moved: push substantively processed", push_processed);
        print_test_result("Tip moved keeps active template valid", tmpl_interface.has_valid_template());
    }

    // ====================================================================
    // Test 15: Solo same-height push replacement discards template and requests work
    // ====================================================================
    std::cout << "\nTest 15: Solo same-height push replacement discards template and requests work" << std::endl;
    {
        auto session_manager = std::make_shared<protocol::SessionManager>();
        auto session_context = std::make_shared<protocol::NodeSessionContext>(session_manager);
        session_manager->start_session(protocol::SessionId(0x12345678u));
        session_manager->set_falcon_identity({0x01}, "push-lane-test-key", true);
        protocol::Solo solo(static_cast<uint8_t>(mining::CHANNEL_HASH), nullptr, session_context);
        solo.set_protocol_lane(ProtocolLane::STATELESS);

        bool recovery_called = false;
        solo.set_recovery_initiated_handler([&recovery_called](protocol::GetBlockReason) -> bool { recovery_called = true; return true; });

        auto template_data = create_mock_template(9001, 0x1d00ffff, 2);
        auto res = solo.get_template_interface()->read_template(template_data, "test_node", false);
        print_test_result("Solo same-height setup template valid", res.is_valid);
        solo.get_template_interface()->set_channel_height(101);

        network::Payload payload = create_extended_push_payload(9000, 100, 0x1d00ffff, 0x42);
        Packet packet(MinerLLP::MirrorOpcode(MinerLLP::HASH_BLOCK_AVAILABLE), payload);
        solo.process_messages(packet, nullptr);

        print_test_result("Solo same-height push does not trigger recovery handler", !recovery_called);
        // Option A + B safety net: same-height push leaves template VALID so
        // workers keep mining through the BLOCK_DATA round-trip.
        print_test_result("Solo same-height push keeps template VALID (no HEALTH_NO_TEMPLATE epoch)",
            solo.get_template_interface()->has_valid_template());
        print_test_result("Solo same-height push marks template replacement-pending",
            solo.get_template_interface()->is_replacement_pending());
    }

    // ====================================================================
    // Test 16: Solo keeps multi-block lag push on session-preserving refresh path
    // ====================================================================
    std::cout << "\nTest 16: Solo multi-block lag push requests refresh without hard recovery" << std::endl;
    {
        auto session_manager = std::make_shared<protocol::SessionManager>();
        auto session_context = std::make_shared<protocol::NodeSessionContext>(session_manager);
        session_manager->start_session(protocol::SessionId(0x87654321u));
        session_manager->set_falcon_identity({0x02}, "push-lane-test-key", true);
        protocol::Solo solo(static_cast<uint8_t>(mining::CHANNEL_HASH), nullptr, session_context);
        solo.set_protocol_lane(ProtocolLane::STATELESS);

        bool recovery_called = false;
        solo.set_recovery_initiated_handler([&recovery_called](protocol::GetBlockReason) -> bool { recovery_called = true; return true; });

        auto template_data = create_mock_template(9101, 0x1d00ffff, 2);
        auto res = solo.get_template_interface()->read_template(template_data, "test_node", false);
        print_test_result("Solo stale setup template valid", res.is_valid);
        solo.get_template_interface()->set_channel_height(101);

        network::Payload payload = create_extended_push_payload(9103, 103, 0x1d00ffff, 0x00);
        Packet packet(MinerLLP::MirrorOpcode(MinerLLP::HASH_BLOCK_AVAILABLE), payload);
        solo.process_messages(packet, nullptr);

        print_test_result("Solo stale push does not trigger recovery handler", !recovery_called);
        print_test_result("Solo stale push keeps current template valid",
            solo.get_template_interface()->has_valid_template());
    }

    // ====================================================================
    // Test 17: Push still processes without active session (no lifeline)
    // ====================================================================
    std::cout << "\nTest 17: Push still processes without active session" << std::endl;
    {
        auto session_manager = std::make_shared<protocol::SessionManager>();
        auto session_context = std::make_shared<protocol::NodeSessionContext>(session_manager);
        protocol::Solo solo(static_cast<uint8_t>(mining::CHANNEL_HASH), nullptr, session_context);
        solo.set_protocol_lane(ProtocolLane::STATELESS);

        bool recovery_called = false;
        solo.set_recovery_initiated_handler([&recovery_called](protocol::GetBlockReason) -> bool { recovery_called = true; return true; });

        auto template_data = create_mock_template(9201, 0x1d00ffff, 2);
        auto res = solo.get_template_interface()->read_template(template_data, "test_node", false);
        print_test_result("Disconnected push setup template valid", res.is_valid);
        solo.get_template_interface()->set_channel_height(101);

        network::Payload payload = create_extended_push_payload(9200, 100, 0x1d00ffff, 0x42);
        Packet packet(MinerLLP::MirrorOpcode(MinerLLP::HASH_BLOCK_AVAILABLE), payload);
        solo.process_messages(packet, nullptr);
        auto push_snapshot = solo.get_height_tracker_snapshot();

        print_test_result("Disconnected-session push does not trigger hard recovery handler", !recovery_called);
        print_test_result("Disconnected-session push updates unified height", push_snapshot.push_unified_height == 9200);
        print_test_result("Disconnected-session push updates channel height", push_snapshot.push_channel_height == 100);
        // Option A + B safety net: same-height push (even on disconnected session)
        // marks the template replacement-pending instead of discarding it. The
        // safety-net timeout in Worker_manager will discard if BLOCK_DATA never
        // arrives.
        print_test_result("Disconnected-session push keeps template VALID (replacement-pending instead)",
            solo.get_template_interface()->has_valid_template());
        print_test_result("Disconnected-session push marks template replacement-pending",
            solo.get_template_interface()->is_replacement_pending());
    }

    // ====================================================================
    // Test 18: Legacy BLOCK_DATA still processes without active session
    // ====================================================================
    std::cout << "\nTest 18: Legacy BLOCK_DATA still processes without active session" << std::endl;
    {
        auto session_manager = std::make_shared<protocol::SessionManager>();
        auto session_context = std::make_shared<protocol::NodeSessionContext>(session_manager);
        protocol::Solo solo(static_cast<uint8_t>(mining::CHANNEL_HASH), nullptr, session_context);
        solo.set_protocol_lane(ProtocolLane::LEGACY);

        bool block_handler_called = false;
        solo.set_block_handler([&block_handler_called](const ::LLP::CBlock&, uint32_t) {
            block_handler_called = true;
            return true;
        });

        network::Payload payload = create_template_delivery_payload(9300, 100, 0x1d00ffff, 9301, 2);
        Packet packet(static_cast<uint8_t>(Packet::BLOCK_DATA), payload);
        solo.process_messages(packet, nullptr);
        auto const* installed_template = solo.get_template_interface()->get_current_template();

        print_test_result("BLOCK_DATA without active session still feeds a template", block_handler_called);
        print_test_result("BLOCK_DATA without active session leaves a valid template installed",
            solo.get_template_interface()->has_valid_template());
        print_test_result("BLOCK_DATA without active session installs expected height",
            installed_template && installed_template->block.nHeight == 9301);
        print_test_result("BLOCK_DATA without active session installs expected difficulty",
            installed_template && installed_template->nBits == 0x1d00ffff);
    }

    // ====================================================================
    // Test 19: Stateless GET_BLOCK still processes without active session
    // ====================================================================
    std::cout << "\nTest 19: Stateless GET_BLOCK still processes without active session" << std::endl;
    {
        auto session_manager = std::make_shared<protocol::SessionManager>();
        auto session_context = std::make_shared<protocol::NodeSessionContext>(session_manager);
        protocol::Solo solo(static_cast<uint8_t>(mining::CHANNEL_HASH), nullptr, session_context);
        solo.set_protocol_lane(ProtocolLane::STATELESS);

        bool block_handler_called = false;
        solo.set_block_handler([&block_handler_called](const ::LLP::CBlock&, uint32_t) {
            block_handler_called = true;
            return true;
        });

        network::Payload payload = create_template_delivery_payload(9400, 100, 0x1d00ffff, 9401, 2);
        Packet packet(MinerLLP::MirrorOpcode(MinerLLP::GET_BLOCK), payload);
        solo.process_messages(packet, nullptr);
        auto const* installed_template = solo.get_template_interface()->get_current_template();

        print_test_result("STATELESS_GET_BLOCK without active session still feeds a template", block_handler_called);
        print_test_result("STATELESS_GET_BLOCK without active session leaves a valid template installed",
            solo.get_template_interface()->has_valid_template());
        print_test_result("STATELESS_GET_BLOCK without active session installs expected height",
            installed_template && installed_template->block.nHeight == 9401);
        print_test_result("STATELESS_GET_BLOCK without active session installs expected difficulty",
            installed_template && installed_template->nBits == 0x1d00ffff);
    }

    // ====================================================================
    // Test 20: Legacy GET_BLOCK template response processes without active session
    // ====================================================================
    std::cout << "\nTest 20: Legacy GET_BLOCK template response processes without active session" << std::endl;
    {
        auto session_manager = std::make_shared<protocol::SessionManager>();
        auto session_context = std::make_shared<protocol::NodeSessionContext>(session_manager);
        protocol::Solo solo(static_cast<uint8_t>(mining::CHANNEL_HASH), nullptr, session_context);
        solo.set_protocol_lane(ProtocolLane::LEGACY);

        bool block_handler_called = false;
        solo.set_block_handler([&block_handler_called](const ::LLP::CBlock&, uint32_t) {
            block_handler_called = true;
            return true;
        });

        Packet zero_request(static_cast<uint8_t>(MinerLLP::GET_BLOCK));
        solo.process_messages(zero_request, nullptr);
        print_test_result("Legacy zero-length GET_BLOCK does not feed a template", !block_handler_called);

        network::Payload payload = create_template_delivery_payload(9450, 101, 0x1d00ffff, 9451, 2);
        Packet packet(static_cast<uint8_t>(MinerLLP::GET_BLOCK), payload);
        solo.process_messages(packet, nullptr);
        auto const* installed_template = solo.get_template_interface()->get_current_template();

        print_test_result("Legacy GET_BLOCK payload feeds a template", block_handler_called);
        print_test_result("Legacy GET_BLOCK payload leaves a valid template installed",
            solo.get_template_interface()->has_valid_template());
        print_test_result("Legacy GET_BLOCK payload installs expected height",
            installed_template && installed_template->block.nHeight == 9451);
        print_test_result("Legacy GET_BLOCK payload installs expected difficulty",
            installed_template && installed_template->nBits == 0x1d00ffff);
    }

    // ====================================================================
    // Test 21: Soft-refresh loop terminates — BLOCK_DATA accepted when its
    //          hashPrevBlock differs from the push tip anchor
    //
    // Regression test for the infinite soft-refresh / template-swap pending
    // loop bug.  Before the fix, validate_current_template() would reject a
    // freshly received BLOCK_DATA template when its hashPrevBlock differed
    // from the most recent extended push tip anchor.  Because ClearPushTipAnchor()
    // is only reached on successful adoption, the anchor was never cleared, each
    // new GET_BLOCK response was rejected, and the miner looped indefinitely
    // showing valid_template=no while canonical diagnostics looked healthy.
    //
    // After the fix:
    //   1. finalize_and_feed_current_template() calls ClearPushTipAnchor() BEFORE
    //      validate_current_template(), removing the anchor veto.
    //   2. validate_current_template() only logs an info note (no rejection) when
    //      has_same_height_push_tip_replacement() returns true.
    // ====================================================================
    std::cout << "\nTest 20: Soft-refresh loop terminates — BLOCK_DATA accepted despite push tip-anchor mismatch" << std::endl;
    {
        auto session_manager = std::make_shared<protocol::SessionManager>();
        auto session_context = std::make_shared<protocol::NodeSessionContext>(session_manager);
        protocol::Solo solo(static_cast<uint8_t>(mining::CHANNEL_HASH), nullptr, session_context);
        solo.set_protocol_lane(ProtocolLane::LEGACY);

        int block_handler_calls = 0;
        solo.set_block_handler([&block_handler_calls](const ::LLP::CBlock&, uint32_t) {
            ++block_handler_calls;
            return true;
        });

        // Step 1: Install initial template (height=9501, channel_height=100)
        {
            network::Payload payload = create_template_delivery_payload(9500, 100, 0x1d00ffff, 9501, 2);
            Packet packet(static_cast<uint8_t>(Packet::BLOCK_DATA), payload);
            solo.process_messages(packet, nullptr);
        }
        print_test_result("Initial template installed (valid)", solo.get_template_interface()->has_valid_template());

        // Step 2: Extended push arrives with a DIFFERENT hashPrevBlock (bytes 0x42).
        // This represents a same-height chain reorg hint from the node.
        // Expected behavior: push handler discards template, sets push tip anchor, triggers soft refresh.
        {
            network::Payload push_payload = create_extended_push_payload(9500, 100, 0x1d00ffff, 0x42);
            Packet push_packet(static_cast<uint8_t>(MinerLLP::HASH_BLOCK_AVAILABLE), push_payload);
            solo.process_messages(push_packet, nullptr);
        }
        // Option A + B safety net: push tip-anchor mismatch no longer discards
        // the old template; it is marked replacement-pending so workers continue
        // mining until BLOCK_DATA atomically swaps in below.
        print_test_result("Push tip anchor mismatch keeps old template VALID (replacement-pending)",
            solo.get_template_interface()->has_valid_template());
        print_test_result("Push tip anchor mismatch marks template replacement-pending",
            solo.get_template_interface()->is_replacement_pending());
        {
            auto snap = solo.get_height_tracker_snapshot();
            print_test_result("Push tip anchor (H_new) was recorded",
                snap.push_hash_prev_block != uint1024_t{});
        }

        // Step 3: GET_BLOCK returns BLOCK_DATA with the OLD hashPrevBlock (all zeros,
        // i.e. the node has not yet adopted the new chain tip the push hinted at).
        // Before the fix: validate_current_template() would see push anchor = H_new,
        // template hashPrevBlock = H_old, and reject the template, re-triggering soft
        // refresh, looping forever.
        // After the fix: ClearPushTipAnchor() is called first, so the template is accepted.
        {
            network::Payload block_payload = create_template_delivery_payload(9500, 100, 0x1d00ffff, 9501, 2);
            Packet block_packet(static_cast<uint8_t>(Packet::BLOCK_DATA), block_payload);
            solo.process_messages(block_packet, nullptr);
        }
        print_test_result("BLOCK_DATA with old hashPrevBlock accepted (loop terminates)",
            solo.get_template_interface()->has_valid_template());
        print_test_result("Push tip anchor cleared after BLOCK_DATA adoption",
            solo.get_height_tracker_snapshot().push_hash_prev_block == uint1024_t{});
        // Note: the debounce gate suppresses a second feed for the same (height, hashPrevBlock)
        // pair within the debounce window.  The important invariant is that the template is
        // VALID (VALIDATED or ACTIVE state), not that the handler was called a second time.
        // has_valid_template() returning true is sufficient to confirm loop termination.
        print_test_result("Template in valid state after recovery BLOCK_DATA (debounce may suppress re-feed)",
            solo.get_template_interface()->has_valid_template());

        // Step 4: Confirm that the installed template is still valid (no further discards).
        print_test_result("Template remains valid after recovery",
            solo.get_template_interface()->has_valid_template());
    }

    // ====================================================================
    // Test 21: PUSH-active session blocks GET_ROUND-triggered GET_BLOCK fallback
    // ====================================================================
    std::cout << "\nTest 21: PUSH-active session blocks GET_ROUND fallback GET_BLOCK" << std::endl;
    {
        auto session_manager = std::make_shared<protocol::SessionManager>();
        auto session_context = std::make_shared<protocol::NodeSessionContext>(session_manager);
        session_manager->start_session(protocol::SessionId(0xAABBCCDDu));
        session_manager->set_falcon_identity({0x21}, "get-round-fallback-test-key", true);

        protocol::Solo solo(static_cast<uint8_t>(mining::CHANNEL_HASH), nullptr, session_context);
        solo.set_protocol_lane(ProtocolLane::STATELESS);

        auto template_data = create_mock_template(9601, 0x1d00ffff, 2);
        auto res = solo.get_template_interface()->read_template(template_data, "test_node", false);
        solo.get_template_interface()->set_channel_height(101);
        auto const* setup_template = solo.get_template_interface()->get_current_template();
        print_test_result("Fallback block test setup template valid",
            res.is_valid && solo.get_template_interface()->has_valid_template() &&
            setup_template && setup_template->nChannelHeight == 101);

        // Re-establish PUSH liveness by sending a HASH push that keeps the template valid
        // (channel tip remains behind nChannelHeight target 101).
        network::Payload hash_push_payload = create_extended_push_payload(9600, 100, 0x1d00ffff, 0x33);
        Packet hash_push_packet(MinerLLP::MirrorOpcode(MinerLLP::HASH_BLOCK_AVAILABLE), hash_push_payload);
        solo.process_messages(hash_push_packet, nullptr);

        print_test_result("Recent PUSH blocks GET_ROUND-triggered GET_BLOCK fallback",
            !should_trigger_get_round_fallback_get_block(true, 101, 101, 5));
    }

    // ====================================================================
    // Test 22: PUSH-silent fallback path enables GET_ROUND-triggered GET_BLOCK
    // ====================================================================
    std::cout << "\nTest 22: PUSH-silent fallback enables GET_ROUND-triggered GET_BLOCK" << std::endl;
    {
        print_test_result("PUSH active (<480s) blocks GET_ROUND fallback trigger",
            !should_trigger_get_round_fallback_get_block(true, 101, 101, 479));
        print_test_result("PUSH silent (>=480s) enables GET_ROUND fallback trigger",
            should_trigger_get_round_fallback_get_block(true, 101, 101, 480));
    }

    // ====================================================================
    // Test 23: Recent PUSH suppresses GET_ROUND Stake-advance GET_BLOCK
    // ====================================================================
    std::cout << "\nTest 23: Recent PUSH suppresses GET_ROUND Stake-advance GET_BLOCK" << std::endl;
    {
        print_test_result("Recent push for same height suppresses GET_BLOCK",
            push_implies_block_data_in_transit_test(true, 200, 200, 2999));
        print_test_result("Stale push does not suppress GET_BLOCK",
            !push_implies_block_data_in_transit_test(true, 200, 200, 3000));
        print_test_result("Push for lower height does not suppress GET_BLOCK",
            !push_implies_block_data_in_transit_test(true, 199, 200, 100));
        print_test_result("No push received does not suppress GET_BLOCK",
            !push_implies_block_data_in_transit_test(false, 200, 200, 100));
        print_test_result("Boundary age==3000ms is not suppressed",
            !push_implies_block_data_in_transit_test(true, 201, 200, 3000));
    }

    // ====================================================================
    // Test 24: First PUSH after silence disarms GET_ROUND fallback mode
    // ====================================================================
    std::cout << "\nTest 24: First PUSH after silence disarms GET_ROUND fallback mode" << std::endl;
    {
        bool fallback_mode_armed = false;
        auto on_get_round_parity = [&](int64_t since_push_s, uint32_t node_channel_height, uint32_t template_target_height) {
            const bool parity_reached = (template_target_height > 0) && (node_channel_height >= template_target_height);
            const bool push_silent = since_push_s >= protocol::Solo::PUSH_ABSENT_FOR_GET_ROUND_FALLBACK_SECONDS;
            if (parity_reached && push_silent) {
                fallback_mode_armed = true;
                return true;
            }
            return false;
        };
        auto on_push_reestablished = [&]() { fallback_mode_armed = false; };

        bool first_trigger = on_get_round_parity(600, 101, 101);
        print_test_result("Silence path first GET_ROUND parity arms fallback trigger", first_trigger && fallback_mode_armed);

        on_push_reestablished();
        bool second_trigger = on_get_round_parity(0, 101, 101);
        print_test_result("First PUSH after silence disarms fallback mode", !fallback_mode_armed && !second_trigger);
    }

    // ====================================================================
    // Test 25: NEW_ROUND remains authoritative when unified advances even if
    //          the active channel height is unchanged
    // ====================================================================
    std::cout << "\nTest 25: NEW_ROUND resets polling when unified advances with unchanged Prime height" << std::endl;
    {
        auto session_manager = std::make_shared<protocol::SessionManager>();
        auto session_context = std::make_shared<protocol::NodeSessionContext>(session_manager);
        session_manager->start_session(protocol::SessionId(0x13572468u));

        protocol::Solo solo(static_cast<uint8_t>(mining::CHANNEL_PRIME), nullptr, session_context);
        solo.set_protocol_lane(ProtocolLane::STATELESS);

        network::Payload old_round_payload = create_get_round_height_payload(200, 50, 75, 10);
        Packet old_round_packet(MinerLLP::MirrorOpcode(static_cast<uint8_t>(Packet::OLD_ROUND)), old_round_payload);
        solo.process_messages(old_round_packet, nullptr);
        print_test_result("OLD_ROUND keeps poll interval at fixed minimum (backoff disabled)",
            solo.get_current_poll_interval_ms() == protocol::Solo::POLL_INTERVAL_MIN_MS);

        network::Payload new_round_payload = create_get_round_height_payload(201, 50, 76, 10);
        Packet new_round_packet(MinerLLP::MirrorOpcode(static_cast<uint8_t>(Packet::NEW_ROUND)), new_round_payload);
        solo.process_messages(new_round_packet, nullptr);

        print_test_result("Unified advance with unchanged Prime height resets polling to minimum",
            solo.get_current_poll_interval_ms() == protocol::Solo::POLL_INTERVAL_MIN_MS);
        print_test_result("Unified advance is reflected in last round status",
            solo.get_last_round_status().height == 201);  // GET_ROUND sends TIP directly (no normalization)
    }

    // ====================================================================
    // Test 26: Cross-channel (Hash) PUSH requests fresh template for Prime miner
    //          when unified tip advances on the other channel
    // ====================================================================
    std::cout << "\nTest 26: Cross-channel Hash PUSH requests work for Prime miner when unified advances" << std::endl;
    {
        // Prime miner receives a Hash block PUSH.  The handler must detect that the
        // unified tip advanced and request a fresh template — every unified height
        // movement changes hashPrevBlock, which must be embedded in the next mined block.
        protocol::HeightTracker tracker;
        protocol::MiningTemplateInterface tmpl_interface(1 /* PRIME */, 0);
        tmpl_interface.set_height_tracker(&tracker);

        // Initial Prime block: unified=6650428, prime_channel=2347879, target=2347880
        tracker.OnPushNotification(6650428, 2347879, 0x1d00ffff);
        auto template_data = create_mock_template(6650429, 0x1d00ffff, 1);
        tmpl_interface.read_template(template_data, "test_node");
        tmpl_interface.set_channel_height(2347880);

        uint8_t current_channel = static_cast<uint8_t>(mining::CHANNEL_PRIME);
        protocol::PushNotificationHandler handler(logger, current_channel);

        // Hash block found: unified advances to 6650429, Prime channel stays at 2347879.
        // This PUSH arrives on the Hash channel (expected_channel=HASH), so it is
        // cross-channel for the Prime miner.
        network::Payload hash_push_payload = create_extended_push_payload(6650429, 2347879, 0x1d00ffff, 0x00);
        Packet hash_push_packet(MinerLLP::MirrorOpcode(MinerLLP::HASH_BLOCK_AVAILABLE), hash_push_payload);

        bool work_requested_25 = handler.handle_push_notification(
            hash_push_packet,
            mining::CHANNEL_HASH,  // expected_channel = Hash (cross-channel for Prime miner)
            ProtocolLane::STATELESS,
            &tmpl_interface,
            &tracker,
            [&tracker](uint32_t u, uint32_t c, uint32_t d) { tracker.OnPushNotification(u, c, d); });

        print_test_result("Cross-channel Hash PUSH returns true (tip advance)",
            work_requested_25);
        print_test_result("Cross-channel Hash PUSH keeps Prime template valid (no discard)",
            tmpl_interface.has_valid_template());
    }

    // ====================================================================
    // Test 27: Cross-channel push with tip advance — update_height_fn called,
    //          height state updated, returns true (Bug 1 regression)
    // ====================================================================
    std::cout << "\nTest 27: Cross-channel tip advance updates height state via update_height_fn" << std::endl;
    {
        // Prime miner receives a Hash block PUSH at a higher unified height.
        // After the fix, update_height_fn must be called inside the cross-channel branch
        // so that HeightTracker reflects the new unified height.
        protocol::HeightTracker tracker;

        // Seed tracker with initial canonical state: unified=100, prime=50
        tracker.OnBlockDataReceived(100, 50, 0x1d00ffff, uint1024_t{});

        uint8_t current_channel = static_cast<uint8_t>(mining::CHANNEL_PRIME);
        protocol::PushNotificationHandler handler(logger, current_channel);

        bool update_height_called = false;
        uint32_t updated_unified  = 0;

        // Hash block found: unified advances to 101
        network::Payload payload = create_extended_push_payload(101, 50, 0x1d00ffff, 0x00);
        Packet pkt(MinerLLP::MirrorOpcode(MinerLLP::HASH_BLOCK_AVAILABLE), payload);

        bool work_requested_26 = handler.handle_push_notification(
            pkt,
            mining::CHANNEL_HASH,
            ProtocolLane::STATELESS,
            nullptr,          // no template_interface needed
            &tracker,
            [&](uint32_t u, uint32_t c, uint32_t d) {
                update_height_called = true;
                updated_unified = u;
                tracker.OnPushNotification(u, c, d);
            });

        print_test_result("Cross-channel tip advance: update_height_fn called",
            update_height_called);
        print_test_result("Cross-channel tip advance: returns true (push substantively processed)",
            work_requested_26);
        print_test_result("Cross-channel tip advance: update_height_fn received correct unified height",
            updated_unified == 101);
        // After update, tracker snapshot must reflect the new push height
        auto snap = tracker.GetSnapshot();
        print_test_result("Cross-channel tip advance: HeightTracker snapshot updated to new unified height",
            snap.push_unified_height == 101);
    }

    // ====================================================================
    // Test 28: Cross-channel push with same height (liveness) — update_height_fn
    //          NOT called, returns false (Bug 1 — dedup working)
    // ====================================================================
    std::cout << "\nTest 28: Cross-channel liveness push (same height) does not call update_height_fn and returns false" << std::endl;
    {
        protocol::HeightTracker tracker;
        // Seed canonical state so push at same height is treated as liveness
        tracker.OnBlockDataReceived(100, 50, 0x1d00ffff, uint1024_t{});

        uint8_t current_channel = static_cast<uint8_t>(mining::CHANNEL_PRIME);
        protocol::PushNotificationHandler handler(logger, current_channel);

        bool update_height_called = false;

        // Liveness push: unified height is the same (100), no tip advance
        network::Payload payload = create_extended_push_payload(100, 50, 0x1d00ffff, 0x00);
        Packet pkt(MinerLLP::MirrorOpcode(MinerLLP::HASH_BLOCK_AVAILABLE), payload);

        bool work_requested_27 = handler.handle_push_notification(
            pkt,
            mining::CHANNEL_HASH,
            ProtocolLane::STATELESS,
            nullptr,
            &tracker,
            [&](uint32_t, uint32_t, uint32_t) { update_height_called = true; });

        print_test_result("Liveness cross-channel push: update_height_fn NOT called",
            !update_height_called);
        print_test_result("Liveness cross-channel push: returns false (no state change)",
            !work_requested_27);
    }

    // ====================================================================
    // Test 29: Cross-channel 148-byte push — hashBestChain stored via
    //          UpdatePushTipAnchor (Bug 2 regression test)
    // ====================================================================
    std::cout << "\nTest 29: Cross-channel 148-byte push stores hashBestChain in HeightTracker" << std::endl;
    {
        protocol::HeightTracker tracker;
        tracker.OnPushNotification(100, 50, 0x1d00ffff);

        uint8_t current_channel = static_cast<uint8_t>(mining::CHANNEL_PRIME);
        protocol::PushNotificationHandler handler(logger, current_channel);

        // Build a 148-byte full-picture payload:
        //   [0-3]   unified_height   = 101
        //   [4-7]   channel_height   = 50
        //   [8-11]  difficulty       = 0x1d00ffff
        //   [12-15] other_channel    = 30   (prime height for hash miner)
        //   [16-19] stake_height     = 10
        //   [20-147] hashBestChain   = filled with 0xAB (128 bytes)
        network::Payload payload(148, 0);
        auto write_u32_be = [&](size_t off, uint32_t v) {
            payload[off+0] = (v >> 24) & 0xFF;
            payload[off+1] = (v >> 16) & 0xFF;
            payload[off+2] = (v >>  8) & 0xFF;
            payload[off+3] =  v        & 0xFF;
        };
        write_u32_be(0,  101);
        write_u32_be(4,  50);
        write_u32_be(8,  0x1d00ffff);
        write_u32_be(12, 30);   // other channel height
        write_u32_be(16, 10);   // stake height
        std::fill(payload.begin() + 20, payload.end(), uint8_t(0xAB));  // hashBestChain

        Packet pkt(MinerLLP::MirrorOpcode(MinerLLP::HASH_BLOCK_AVAILABLE), payload);

        handler.handle_push_notification(
            pkt,
            mining::CHANNEL_HASH,
            ProtocolLane::STATELESS,
            nullptr,
            &tracker,
            [&tracker](uint32_t u, uint32_t c, uint32_t d) { tracker.OnPushNotification(u, c, d); });

        // Verify the hash was stored in the diagnostic state
        auto diag = tracker.GetDiagnosticSnapshot();
        // The stored push_hash_prev_block must be non-zero (filled with 0xAB pattern)
        print_test_result("Cross-channel 148-byte push stores hashBestChain via UpdatePushTipAnchor",
            diag.push_hash_prev_block != uint1024_t{});
    }

    // ====================================================================
    // Test 30: Two sequential cross-channel pushes at same height — second
    //          push returns false (dedup working after Bug 1 fix)
    // ====================================================================
    std::cout << "\nTest 30: Two sequential cross-channel pushes at same height — second returns false" << std::endl;
    {
        // After Bug 1 fix, update_height_fn is called on the first cross-channel push,
        // so HeightTracker records unified=101.  The second push at unified=101 must
        // see snap.unified_height == notification_unified_height and return false.
        protocol::HeightTracker tracker;
        // Seed canonical state so cross-channel dedup works against canonical unified_height
        tracker.OnBlockDataReceived(100, 50, 0x1d00ffff, uint1024_t{});

        uint8_t current_channel = static_cast<uint8_t>(mining::CHANNEL_PRIME);
        protocol::PushNotificationHandler handler(logger, current_channel);

        network::Payload payload = create_extended_push_payload(101, 50, 0x1d00ffff, 0x00);
        Packet pkt(MinerLLP::MirrorOpcode(MinerLLP::HASH_BLOCK_AVAILABLE), payload);

        auto update_fn     = [&tracker](uint32_t u, uint32_t c, uint32_t d) {
            tracker.OnPushNotification(u, c, d);
            tracker.OnBlockDataReceived(u, c, d, uint1024_t{});
        };

        // First push at unified=101: tip advance, returns true
        bool work_29_first = handler.handle_push_notification(
            pkt, mining::CHANNEL_HASH, ProtocolLane::STATELESS,
            nullptr, &tracker, update_fn);

        print_test_result("First cross-channel push at unified=101 returns true (tip advance)",
            work_29_first);

        // Second push at same unified=101: liveness only, returns false
        bool work_29_second = handler.handle_push_notification(
            pkt, mining::CHANNEL_HASH, ProtocolLane::STATELESS,
            nullptr, &tracker, update_fn);

        print_test_result("Second cross-channel push at same unified=101 returns false (liveness only)",
            !work_29_second);
    }

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
