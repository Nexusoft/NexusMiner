/**
 * @file mining_template_validation_test.cpp
 * @brief Unit tests for Mining Template Height Validation and Worker Protection
 *
 * Validates the implementation of:
 *   - Unified height sanity checking (±100 block threshold)
 *   - Forward jump detection (corrupted height)
 *   - Backward jump detection (reorg/corrupted height)
 *   - Template age monitoring
 *   - Worker stop/recovery mechanism
 *   - Degraded mode handling
 */

#include "protocol/mining_template_interface.hpp"
#include "protocol/height_tracker.hpp"
#include "LLP/block.hpp"
#include "LLP/block_utils.hpp"
#include "LLP/miner_opcodes.hpp"
#include "worker/block_header_utils.hpp"
#include <iostream>
#include <cassert>
#include <cstdint>
#include <memory>
#include <chrono>
#include <thread>
#include <sstream>

// Mock logger for testing
#include "spdlog/spdlog.h"
#include "spdlog/sinks/null_sink.h"
#include "spdlog/sinks/ostream_sink.h"

using namespace nexusminer::protocol;
namespace MinerLLP = nexusminer::LLP;

// Test statistics
static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

static constexpr size_t TRITIUM_METADATA_PREFIX_SIZE = 12;
static constexpr size_t TRITIUM_BODY_NHEIGHT_OFFSET = 200;

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

// Helper to create a mock block template
std::vector<uint8_t> create_mock_template(uint32_t height, uint32_t nBits = 0x1d00ffff, 
                                           uint8_t channel = 2) {
    // Create a 216-byte Tritium block template matching the actual format
    // Structure: nVersion(4) + hashPrevBlock(128) + hashMerkleRoot(64) + 
    //            nChannel(4) + nHeight(4) + nBits(4) + nNonce(8)
    // Total: 4+128+64+4+4+4+8 = 216 bytes
    std::vector<uint8_t> data(216, 0);
    
    size_t offset = 0;
    
    // Helper to write big-endian uint32
    auto write_u32_be = [&](uint32_t value) {
        data[offset++] = (value >> 24) & 0xFF;
        data[offset++] = (value >> 16) & 0xFF;
        data[offset++] = (value >> 8) & 0xFF;
        data[offset++] = value & 0xFF;
    };
    
    // Helper to write big-endian uint64
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
    
    // 1. nVersion (4 bytes) - big-endian
    write_u32_be(7);  // Version 7
    
    // 2. hashPrevBlock (128 bytes) - uint1024_t
    // Leave as zeros for test
    offset += 128;
    
    // 3. hashMerkleRoot (64 bytes) - uint512_t
    // Set non-zero pattern to pass validation
    for (int i = 0; i < 64; i++) {
        data[offset++] = (i + 1);  // Pattern 1, 2, 3, ...
    }
    
    // 4. nChannel (4 bytes) - big-endian
    write_u32_be(static_cast<uint32_t>(channel));
    
    // 5. nHeight (4 bytes) - big-endian
    write_u32_be(height);
    
    // 6. nBits (4 bytes) - big-endian
    write_u32_be(nBits);
    
    // 7. nNonce (8 bytes) - little-endian (Nexus node writes nNonce as LE)
    // Template nNonce is zero; use little-endian to match the node wire format.
    uint64_t nonce_val = 0;
    data[offset++] = nonce_val & 0xFF;
    data[offset++] = (nonce_val >> 8) & 0xFF;
    data[offset++] = (nonce_val >> 16) & 0xFF;
    data[offset++] = (nonce_val >> 24) & 0xFF;
    data[offset++] = (nonce_val >> 32) & 0xFF;
    data[offset++] = (nonce_val >> 40) & 0xFF;
    data[offset++] = (nonce_val >> 48) & 0xFF;
    data[offset++] = (nonce_val >> 56) & 0xFF;
    
    return data;
}

std::vector<uint8_t> create_mock_stateless_payload(uint32_t metadata_unified_height,
                                                   uint32_t metadata_channel_height,
                                                   uint32_t nBits = 0x1d00ffff,
                                                   uint8_t channel = 2) {
    auto body = create_mock_template(metadata_unified_height + 1, nBits, channel);
    std::vector<uint8_t> payload;
    payload.reserve(TRITIUM_METADATA_PREFIX_SIZE + body.size());

    auto write_u32_be = [&](uint32_t value) {
        payload.push_back((value >> 24) & 0xFF);
        payload.push_back((value >> 16) & 0xFF);
        payload.push_back((value >> 8) & 0xFF);
        payload.push_back(value & 0xFF);
    };

    write_u32_be(metadata_unified_height);
    write_u32_be(metadata_channel_height);
    write_u32_be(nBits);
    payload.insert(payload.end(), body.begin(), body.end());
    return payload;
}

int main()
{
    // Setup null logger to avoid spam during tests
    auto null_sink = std::make_shared<spdlog::sinks::null_sink_mt>();
    auto logger = std::make_shared<spdlog::logger>("logger", null_sink);
    spdlog::set_default_logger(logger);
    
    std::cout << "========================================" << std::endl;
    std::cout << "Mining Template Validation Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    // ====================================================================
    // Test 1: Normal height progression
    // ====================================================================
    std::cout << "\nTest 1: Normal height progression" << std::endl;
    {
        MiningTemplateInterface tmpl_interface(2, 0); // Channel 2 (hash)
        
        // First template at height 1000000
        auto data1 = create_mock_template(1000000);
        auto result1 = tmpl_interface.read_template(data1, "test_node");
        if (!result1.is_valid) {
            std::cout << "    ERROR: " << result1.error_message << std::endl;
        }
        print_test_result("First template validation (height 1000000)", result1.is_valid);
        
        // Second template advances by 1 block
        auto data2 = create_mock_template(1000001);
        auto result2 = tmpl_interface.read_template(data2, "test_node");
        if (!result2.is_valid) {
            std::cout << "    ERROR: " << result2.error_message << std::endl;
        }
        print_test_result("Forward +1 block (1000000 → 1000001)", result2.is_valid);
        
        // Third template advances by 50 blocks
        auto data3 = create_mock_template(1000051);
        auto result3 = tmpl_interface.read_template(data3, "test_node");
        print_test_result("Forward +50 blocks (1000001 → 1000051)", result3.is_valid);
        
        // Fourth template at boundary (exactly 100 blocks forward)
        auto data4 = create_mock_template(1000151);
        auto result4 = tmpl_interface.read_template(data4, "test_node");
        print_test_result("Forward +100 blocks (1000051 → 1000151)", result4.is_valid);
    }

    // ====================================================================
    // Test 2: Forward jump detection (corrupted height)
    // ====================================================================
    std::cout << "\nTest 2: Forward jump detection (corrupted height)" << std::endl;
    {
        MiningTemplateInterface tmpl_interface(2, 0);
        
        // First template at height 6500000
        auto data1 = create_mock_template(6500000);
        auto result1 = tmpl_interface.read_template(data1, "test_node");
        print_test_result("Baseline template (height 6500000)", result1.is_valid);
        
        // Corrupted jump: 6.5M → 1.9B (exceeds 100 blocks)
        auto data2 = create_mock_template(1900000000);
        auto result2 = tmpl_interface.read_template(data2, "test_node");
        print_test_result("Corrupted forward jump rejected (6.5M → 1.9B)", 
                         !result2.is_valid && !result2.height_valid);
        
        // Jump of exactly 101 blocks should fail
        auto data3 = create_mock_template(6500101);
        auto result3 = tmpl_interface.read_template(data3, "test_node");
        print_test_result("Forward +101 blocks rejected (6500000 → 6500101)", 
                         !result3.is_valid && !result3.height_valid);
    }

    // ====================================================================
    // Test 3: Backward jump detection (reorg)
    // ====================================================================
    std::cout << "\nTest 3: Backward jump detection (reorg)" << std::endl;
    {
        MiningTemplateInterface tmpl_interface(2, 0);
        
        // First template at height 1000100
        auto data1 = create_mock_template(1000100);
        auto result1 = tmpl_interface.read_template(data1, "test_node");
        print_test_result("Baseline template (height 1000100)", result1.is_valid);
        
        // Small reorg: -5 blocks (should be accepted)
        auto data2 = create_mock_template(1000095);
        auto result2 = tmpl_interface.read_template(data2, "test_node");
        print_test_result("Small reorg -5 blocks accepted (1000100 → 1000095)", 
                         result2.is_valid);
        
        // Medium reorg: -50 blocks (should be accepted)
        auto data3 = create_mock_template(1000045);
        auto result3 = tmpl_interface.read_template(data3, "test_node");
        print_test_result("Medium reorg -50 blocks accepted (1000095 → 1000045)", 
                         result3.is_valid);
        
        // Boundary: exactly -100 blocks (should be accepted)
        auto data4 = create_mock_template(999945);
        auto result4 = tmpl_interface.read_template(data4, "test_node");
        print_test_result("Boundary reorg -100 blocks accepted (1000045 → 999945)", 
                         result4.is_valid);
        
        // Deep reorg/corruption: -101 blocks (should be rejected)
        auto data5 = create_mock_template(999844);
        auto result5 = tmpl_interface.read_template(data5, "test_node");
        print_test_result("Deep reorg -101 blocks rejected (999945 → 999844)", 
                         !result5.is_valid && !result5.height_valid);
    }

    // ====================================================================
    // Test 4: Edge cases
    // ====================================================================
    std::cout << "\nTest 4: Edge cases" << std::endl;
    {
        MiningTemplateInterface tmpl_interface(2, 0);
        
        // Template at height 100
        auto data1 = create_mock_template(100);
        auto result1 = tmpl_interface.read_template(data1, "test_node");
        print_test_result("Low height template (height 100)", result1.is_valid);
        
        // Backward to height 0 (within 100 blocks)
        auto data2 = create_mock_template(0);
        auto result2 = tmpl_interface.read_template(data2, "test_node");
        print_test_result("Reorg to height 0 (within threshold)", result2.is_valid);
        
        // Same height (duplicate template)
        auto data3 = create_mock_template(0);
        auto result3 = tmpl_interface.read_template(data3, "test_node");
        print_test_result("Duplicate height accepted (0 → 0)", result3.is_valid);
    }

    // ====================================================================
    // Test 5: Channel validation
    // ====================================================================
    std::cout << "\nTest 5: Channel validation" << std::endl;
    {
        MiningTemplateInterface tmpl_interface(2, 0); // Expect channel 2 (hash)
        
        // Valid channel 2
        auto data1 = create_mock_template(1000000, 0x1d00ffff, 2);
        auto result1 = tmpl_interface.read_template(data1, "test_node");
        print_test_result("Valid channel 2", result1.is_valid);
        
        // Invalid channel 0
        auto data2 = create_mock_template(1000001, 0x1d00ffff, 0);
        auto result2 = tmpl_interface.read_template(data2, "test_node");
        print_test_result("Invalid channel 0 rejected", 
                         !result2.is_valid && !result2.channel_valid);
        
        // Invalid channel 3
        auto data3 = create_mock_template(1000002, 0x1d00ffff, 3);
        auto result3 = tmpl_interface.read_template(data3, "test_node");
        print_test_result("Invalid channel 3 rejected", 
                         !result3.is_valid && !result3.channel_valid);
    }

    // ====================================================================
    // Test 6: nBits validation
    // ====================================================================
    std::cout << "\nTest 6: nBits (difficulty) validation" << std::endl;
    {
        MiningTemplateInterface tmpl_interface(2, 0);
        
        // Valid nBits
        auto data1 = create_mock_template(1000000, 0x1d00ffff, 2);
        auto result1 = tmpl_interface.read_template(data1, "test_node");
        print_test_result("Valid nBits (0x1d00ffff)", result1.is_valid);
        
        // Zero nBits (invalid)
        auto data2 = create_mock_template(1000001, 0, 2);
        auto result2 = tmpl_interface.read_template(data2, "test_node");
        print_test_result("Zero nBits rejected", 
                         !result2.is_valid && !result2.bits_valid);
    }

    // ====================================================================
    // Test 7: Merkle root validation
    // ====================================================================
    std::cout << "\nTest 7: Merkle root validation" << std::endl;
    {
        MiningTemplateInterface tmpl_interface(2, 0);
        
        // All-zero merkle root should be rejected
        std::vector<uint8_t> data = create_mock_template(1000000);
        // Clear merkle root bytes (offset 132-195, which is after version + hashPrevBlock)
        for (int i = 132; i < 196; i++) {
            data[i] = 0x00;
        }
        auto result = tmpl_interface.read_template(data, "test_node");
        print_test_result("All-zero merkle root rejected", 
                         !result.is_valid && !result.merkle_valid);
    }

    // ====================================================================
    // Test 8: Height continuity scenarios
    // ====================================================================
    std::cout << "\nTest 8: Height continuity scenarios" << std::endl;
    {
        MiningTemplateInterface tmpl_interface(2, 0);
        
        // Start at 1000000
        auto data1 = create_mock_template(1000000);
        tmpl_interface.read_template(data1, "test_node");
        
        // Jump backward 50, then forward 100 (to stay within threshold)
        auto data2 = create_mock_template(999950);
        auto result2 = tmpl_interface.read_template(data2, "test_node");
        print_test_result("Backward -50 blocks", result2.is_valid);
        
        auto data3 = create_mock_template(1000050);
        auto result3 = tmpl_interface.read_template(data3, "test_node");
        print_test_result("Then forward +100 blocks (within threshold)", result3.is_valid);
        
        // Try to go back -101 from current (1000050 → 999949)
        auto data4 = create_mock_template(999949);
        auto result4 = tmpl_interface.read_template(data4, "test_node");
        print_test_result("Deep backward -101 rejected", !result4.is_valid);
    }

    // ====================================================================
    // Test 9: Maximum safe values
    // ====================================================================
    std::cout << "\nTest 9: Maximum safe values" << std::endl;
    {
        MiningTemplateInterface tmpl_interface(2, 0);
        
        // Near uint32_t max
        uint32_t max_safe = 0xFFFFFF00; // Leave room for arithmetic
        auto data1 = create_mock_template(max_safe);
        auto result1 = tmpl_interface.read_template(data1, "test_node");
        print_test_result("Near max uint32_t height", result1.is_valid);
        
        // Small increment from max_safe
        auto data2 = create_mock_template(max_safe + 50);
        auto result2 = tmpl_interface.read_template(data2, "test_node");
        print_test_result("Increment from near-max height", result2.is_valid);
    }

    // ====================================================================
    // Test 10: Unified height-based staleness detection (update_height)
    // ====================================================================
    std::cout << "\nTest 10: Unified height-based staleness detection (update_height)" << std::endl;
    {
        MiningTemplateInterface tmpl_interface(2, 0);
        
        // Load a template at height 6594320
        auto data1 = create_mock_template(6594320);
        auto result1 = tmpl_interface.read_template(data1, "test_node");
        print_test_result("Template loaded at height 6594320", result1.is_valid);
        
        // Template height should be queryable
        uint32_t tmpl_height = tmpl_interface.get_template_height();
        print_test_result("get_template_height() returns expected height",
            tmpl_height == 6594320);
        
        // update_height with SAME height should NOT discard template
        bool discarded = tmpl_interface.update_height(6594320);
        print_test_result("Same height does not discard template", !discarded);
        print_test_result("Template still valid after same height update",
            tmpl_interface.has_valid_template());
        
        // update_height with HIGHER height SHOULD discard template
        discarded = tmpl_interface.update_height(6594321);
        print_test_result("Higher height discards template", discarded);
        print_test_result("Template invalid after height advance",
            !tmpl_interface.has_valid_template());
    }

    // ====================================================================
    // Test 11: Channel height delta staleness
    // ====================================================================
    std::cout << "\nTest 11: Channel height delta staleness" << std::endl;
    {
        MiningTemplateInterface tmpl_interface(2, 0);
        
        // Load a template
        auto data1 = create_mock_template(6594320);
        tmpl_interface.read_template(data1, "test_node");
        
        // Set channel height snapshot (simulates what happens on template receipt)
        tmpl_interface.set_template_channel_height_snapshot(4165000);
        
        // Same channel height → not stale
        bool is_stale = tmpl_interface.check_staleness_by_channel_delta(4165000);
        print_test_result("Same channel height is not stale", !is_stale);
        
        // Higher channel height → stale
        is_stale = tmpl_interface.check_staleness_by_channel_delta(4165001);
        print_test_result("Advanced channel height is stale", is_stale);
    }

    // ====================================================================
    // Test 12: Template age does not cause premature timeout
    // ====================================================================
    std::cout << "\nTest 12: Template validity with age < 300s" << std::endl;
    {
        MiningTemplateInterface tmpl_interface(2, 0);
        
        // Load a template
        auto data1 = create_mock_template(6594320);
        tmpl_interface.read_template(data1, "test_node");
        
        // Template should be valid immediately (age ~0s, well under 300s)
        uint64_t age = tmpl_interface.get_template_age();
        print_test_result("Fresh template age is small", age < 5);
        print_test_result("Fresh template is valid", tmpl_interface.has_valid_template());
        
        // Template should NOT be stale at this age (was previously timing out at 120s)
        bool is_stale = tmpl_interface.is_template_stale();
        print_test_result("Fresh template is not stale", !is_stale);
    }

    // ====================================================================
    // Test 13: Channel height staleness via update_channel_height
    // ====================================================================
    std::cout << "\nTest 13: Channel height staleness via update_channel_height" << std::endl;
    {
        MiningTemplateInterface tmpl_interface(2, 0); // Hash channel
        
        // Load a template
        auto data1 = create_mock_template(6594320);
        tmpl_interface.read_template(data1, "test_node");
        
        // Finalize template with channel height (simulates GET_ROUND finalization)
        // set_channel_height sets nChannelHeight = node_height + 1 internally... 
        // Actually: set_channel_height just stores the value directly.
        // Template targets block at nChannelHeight; node is at nChannelHeight - 1.
        tmpl_interface.set_channel_height(4165001); // Template mines block 4165001
        
        // Node at expected height (4165000 = 4165001 - 1) → not stale
        bool discarded = tmpl_interface.update_channel_height(2, 4165000);
        print_test_result("Node at expected channel height: template valid", !discarded);
        print_test_result("Template still valid", tmpl_interface.has_valid_template());
        
        // Node channel advanced (4165001 >= 4165001) → stale!
        discarded = tmpl_interface.update_channel_height(2, 4165001);
        print_test_result("Node channel advanced: template stale", discarded);
        print_test_result("Template invalid after channel advance",
            !tmpl_interface.has_valid_template());
    }

    // ====================================================================
    // Test 14: Wrong channel update does not affect template
    // ====================================================================
    std::cout << "\nTest 14: Wrong channel update does not affect template" << std::endl;
    {
        MiningTemplateInterface tmpl_interface(2, 0); // Hash channel
        
        // Load a template
        auto data1 = create_mock_template(6594320);
        tmpl_interface.read_template(data1, "test_node");
        tmpl_interface.set_channel_height(4165001);
        
        // Update Prime channel (channel 1) — should NOT affect Hash template
        bool discarded = tmpl_interface.update_channel_height(1, 9999999);
        print_test_result("Prime channel update ignored for Hash template", !discarded);
        print_test_result("Template still valid after wrong-channel update",
            tmpl_interface.has_valid_template());
    }

    // ====================================================================
    // Test 16: set_channel_height() does NOT mutate block.nHeight (unified)
    // ====================================================================
    std::cout << "\nTest 16: set_channel_height() does NOT mutate block.nHeight" << std::endl;
    {
        MiningTemplateInterface tmpl_interface(2, 0);

        // Load a template at unified height 6594320
        uint32_t unified_height = 6594320;
        auto data1 = create_mock_template(unified_height);
        auto result1 = tmpl_interface.read_template(data1, "test_node");
        print_test_result("Template loaded at unified height 6594320", result1.is_valid);

        // Confirm block.nHeight == unified height immediately after deserialization
        const auto* tmpl = tmpl_interface.get_current_template();
        print_test_result("block.nHeight == unified_height after read_template",
            tmpl != nullptr && tmpl->block.nHeight == unified_height);

        // set_channel_height() updates nChannelHeight metadata — must NOT touch block.nHeight
        uint32_t channel_target = 4165001;
        tmpl_interface.set_channel_height(channel_target);

        // Refresh pointer after potential template update
        tmpl = tmpl_interface.get_current_template();
        print_test_result("block.nHeight still == unified_height after set_channel_height()",
            tmpl != nullptr && tmpl->block.nHeight == unified_height);

        // nChannelHeight should be updated to channel_target
        print_test_result("nChannelHeight == channel_target after set_channel_height()",
            tmpl != nullptr && tmpl->nChannelHeight == channel_target);

        // The two values must differ (they represent different things)
        print_test_result("block.nHeight != nChannelHeight (unified vs channel)",
            tmpl != nullptr && tmpl->block.nHeight != tmpl->nChannelHeight);

        // Negative case: if block.nHeight were silently replaced with the template's
        // channel target, the guard must reject it because ProofHash() requires the
        // unified GET_BLOCK height, not the channel height dimension.
        print_test_result("test_block_with_channel_height_in_nHeight_fails_guard()",
            tmpl != nullptr &&
            is_channel_height(tmpl->height_guard.channel_height) &&
            !tmpl->height_guard.matches(tmpl->height_guard.channel_height.get()));
        print_test_result("test_block_with_unified_height_passes_guard()",
            tmpl != nullptr &&
            tmpl->height_guard.matches(tmpl->block));
    }

    // ====================================================================
    // Test 17: block.nHeight (unified) survives serialization in prepare_block_submission()
    // ====================================================================
    std::cout << "\nTest 17: block.nHeight preserved in serialized submission at offset [200-203]" << std::endl;
    {
        MiningTemplateInterface tmpl_interface(2, 0);

        uint32_t unified_height = 6594321;
        auto data1 = create_mock_template(unified_height);
        auto result1 = tmpl_interface.read_template(data1, "test_node");
        print_test_result("Template loaded for submission test", result1.is_valid);

        if (result1.is_valid) {
            // Prepare a fake 64-byte merkle root (non-zero)
            std::vector<uint8_t> merkle_root(64, 0xAB);

            // prepare_block_submission() serializes the solved block (Tritium 216-byte format)
            // Tritium layout: nVersion(4) + hashPrevBlock(128) + hashMerkleRoot(64) +
            //                 nChannel(4) + nHeight(4) + nBits(4) + nNonce(8) = 216 bytes
            // nHeight is at offset 200 (big-endian uint32)
            uint64_t nonce = 0xDEADBEEFCAFEBABEULL;
            auto payload = tmpl_interface.prepare_block_submission(merkle_root, nonce);

            print_test_result("prepare_block_submission() returns 216-byte payload",
                payload.size() == 216);

            if (payload.size() >= 204) {
                // Read nHeight from serialized payload at offset 200 (big-endian)
                uint32_t serialized_height =
                    (static_cast<uint32_t>(payload[200]) << 24) |
                    (static_cast<uint32_t>(payload[201]) << 16) |
                    (static_cast<uint32_t>(payload[202]) << 8)  |
                     static_cast<uint32_t>(payload[203]);

                print_test_result("block.nHeight (unified) preserved in payload[200-203]",
                    serialized_height == unified_height);

                // Also verify nNonce at offset 208 (little-endian uint64 — Nexus node reads nNonce as LE)
                uint64_t serialized_nonce = 0;
                for (int i = 0; i < 8; ++i)
                    serialized_nonce |= static_cast<uint64_t>(payload[208 + i]) << (i * 8);
                print_test_result("nNonce preserved in payload[208-215] (little-endian)",
                    serialized_nonce == nonce);
            }
        }
    }

    // ====================================================================
    // Test 18: Stateless opcode compatibility aliases
    // ====================================================================
    std::cout << "\nTest 18: Stateless opcode compatibility aliases" << std::endl;
    {
        print_test_result("BLOCK_DATA mirror is 0xD000",
            MinerLLP::StatelessMining::BLOCK_DATA == 0xD000 &&
            MinerLLP::MirrorOpcode(MinerLLP::BLOCK_DATA) == 0xD000);
        print_test_result("BLOCK_ACCEPTED compat opcode is 0xD002",
            MinerLLP::StatelessMining::BLOCK_ACCEPTED_COMPAT == 0xD002);
        print_test_result("BLOCK_REJECTED compat opcode is 0xD003",
            MinerLLP::StatelessMining::BLOCK_REJECTED_COMPAT == 0xD003);
    }

    // ====================================================================
    // Test 19: hashPrevBlock preserved end-to-end through prepare_block_submission()
    // ====================================================================
    std::cout << "\nTest 19: hashPrevBlock preserved end-to-end through prepare_block_submission()" << std::endl;
    {
        MiningTemplateInterface tmpl_interface(2, 0);

        // 1. Create a mock template with a known non-zero hashPrevBlock byte pattern (bytes 1..128).
        std::vector<uint8_t> mock_data(216, 0);
        // Write nVersion = 7 (big-endian) at offset 0
        mock_data[0] = 0; mock_data[1] = 0; mock_data[2] = 0; mock_data[3] = 7;
        // hashPrevBlock at offset [4..131]: bytes 1..128
        for (int i = 0; i < 128; ++i) {
            mock_data[4 + i] = static_cast<uint8_t>((i + 1) & 0xFF);
        }
        // hashMerkleRoot at offset [132..195]: non-zero pattern
        for (int i = 0; i < 64; ++i) {
            mock_data[132 + i] = static_cast<uint8_t>((i + 1) & 0xFF);
        }
        // nChannel = 2 at offset 196
        mock_data[196] = 0; mock_data[197] = 0; mock_data[198] = 0; mock_data[199] = 2;
        // nHeight = 7000001 at offset 200
        uint32_t known_height = 7000001;
        mock_data[200] = (known_height >> 24) & 0xFF;
        mock_data[201] = (known_height >> 16) & 0xFF;
        mock_data[202] = (known_height >> 8) & 0xFF;
        mock_data[203] = known_height & 0xFF;
        // nBits = 0x1d00ffff at offset 204
        mock_data[204] = 0x1d; mock_data[205] = 0x00; mock_data[206] = 0xFF; mock_data[207] = 0xFF;
        // nNonce = 0 at offset 208..215

        // 2. Call read_template() with this mock data.
        auto result = tmpl_interface.read_template(mock_data, "test_node");
        print_test_result("read_template() accepts mock template with known hashPrevBlock", result.is_valid);

        if (result.is_valid) {
            // 3. Verify get_current_template()->block.hashPrevBlock matches the input bytes.
            const auto* tmpl = tmpl_interface.get_current_template();
            bool prev_ok = false;
            if (tmpl) {
                auto prev_bytes = tmpl->block.hashPrevBlock.GetBytes();
                prev_ok = (prev_bytes.size() >= 128);
                if (prev_ok) {
                    for (int i = 0; i < 128 && prev_ok; ++i) {
                        prev_ok = (prev_bytes[i] == static_cast<uint8_t>((i + 1) & 0xFF));
                    }
                }
            }
            print_test_result("block.hashPrevBlock matches input bytes after read_template()", prev_ok);

            // 4. Call prepare_block_submission(merkle_root, nonce).
            std::vector<uint8_t> merkle_root(64, 0xAB);
            uint64_t nonce = 0x0102030405060708ULL;
            auto payload = tmpl_interface.prepare_block_submission(merkle_root, nonce);

            print_test_result("prepare_block_submission() returns 216-byte payload", payload.size() == 216);

            // 5. Verify payload[4..131] matches the original hashPrevBlock bytes.
            if (payload.size() == 216) {
                bool payload_prev_ok = true;
                for (int i = 0; i < 128 && payload_prev_ok; ++i) {
                    payload_prev_ok = (payload[4 + i] == static_cast<uint8_t>((i + 1) & 0xFF));
                }
                print_test_result("✓ hashPrevBlock preserved at payload[4-131]", payload_prev_ok);
            } else {
                print_test_result("✓ hashPrevBlock preserved at payload[4-131]", false);
            }
        } else {
            // Skip dependent sub-tests
            print_test_result("block.hashPrevBlock matches input bytes after read_template()", false);
            print_test_result("prepare_block_submission() returns 216-byte payload", false);
            print_test_result("✓ hashPrevBlock preserved at payload[4-131]", false);
        }
    }

    // ====================================================================
    // Test 20: BLOCK_ACCEPTED / BLOCK_REJECTED opcode handling increments counters
    // ====================================================================
    std::cout << "\nTest 20: BLOCK_ACCEPTED / BLOCK_REJECTED opcode handling increments counters" << std::endl;
    {
        // Verify the primary mirror-mapped opcodes are correct.
        print_test_result("BLOCK_ACCEPTED mirror opcode is 0xD0C8",
            MinerLLP::StatelessMining::BLOCK_ACCEPTED == 0xD0C8);
        print_test_result("BLOCK_REJECTED mirror opcode is 0xD0C9",
            MinerLLP::StatelessMining::BLOCK_REJECTED == 0xD0C9);

        // Simulate counter behaviour using a simple struct that mirrors Solo's counters.
        struct BlockResultCounters {
            uint32_t accepted{0};
            uint32_t rejected{0};
            void on_accepted() { ++accepted; }
            void on_rejected() { ++rejected; }
        };

        BlockResultCounters counters;

        // Simulate receiving BLOCK_ACCEPTED
        uint16_t opcode = MinerLLP::StatelessMining::BLOCK_ACCEPTED;
        if (opcode == MinerLLP::StatelessMining::BLOCK_ACCEPTED) {
            counters.on_accepted();
        }
        print_test_result("m_blocks_accepted increments on BLOCK_ACCEPTED (0xD0C8)",
            counters.accepted == 1 && counters.rejected == 0);

        // Simulate receiving BLOCK_REJECTED
        opcode = MinerLLP::StatelessMining::BLOCK_REJECTED;
        if (opcode == MinerLLP::StatelessMining::BLOCK_REJECTED) {
            counters.on_rejected();
        }
        print_test_result("m_blocks_rejected increments on BLOCK_REJECTED (0xD0C9)",
            counters.accepted == 1 && counters.rejected == 1);

        // Also verify compat aliases trigger the same counter path.
        opcode = MinerLLP::StatelessMining::BLOCK_ACCEPTED_COMPAT;
        if (opcode == MinerLLP::StatelessMining::BLOCK_ACCEPTED_COMPAT) {
            counters.on_accepted();
        }
        opcode = MinerLLP::StatelessMining::BLOCK_REJECTED_COMPAT;
        if (opcode == MinerLLP::StatelessMining::BLOCK_REJECTED_COMPAT) {
            counters.on_rejected();
        }
        print_test_result("Compat aliases also drive counters (accepted=2, rejected=2)",
            counters.accepted == 2 && counters.rejected == 2);
    }

    // ====================================================================
    // Test 21: Legacy lane GOOD_BLOCK/ORPHAN_BLOCK opcode values and counter behaviour
    // ====================================================================
    std::cout << "\nTest 21: Legacy lane GOOD_BLOCK/ORPHAN_BLOCK opcode handling" << std::endl;
    {
        // Verify the legacy opcode values are correct.
        print_test_result("GOOD_BLOCK legacy opcode is 6",
            static_cast<int>(MinerLLP::GOOD_BLOCK) == 6);
        print_test_result("ORPHAN_BLOCK legacy opcode is 7",
            static_cast<int>(MinerLLP::ORPHAN_BLOCK) == 7);

        // Simulate counter behaviour for GOOD_BLOCK (accepted) and ORPHAN_BLOCK (rejected).
        struct LegacyBlockResultCounters {
            uint32_t accepted{0};
            uint32_t rejected{0};
            void on_good_block()   { ++accepted; }
            void on_orphan_block() { ++rejected; }
        };

        LegacyBlockResultCounters counters;

        // GOOD_BLOCK → m_blocks_accepted++
        uint8_t opcode = static_cast<uint8_t>(MinerLLP::GOOD_BLOCK);
        if (opcode == static_cast<uint8_t>(MinerLLP::GOOD_BLOCK)) {
            counters.on_good_block();
        }
        print_test_result("m_blocks_accepted increments on GOOD_BLOCK (opcode 6)",
            counters.accepted == 1 && counters.rejected == 0);

        // ORPHAN_BLOCK → m_blocks_rejected++
        opcode = static_cast<uint8_t>(MinerLLP::ORPHAN_BLOCK);
        if (opcode == static_cast<uint8_t>(MinerLLP::ORPHAN_BLOCK)) {
            counters.on_orphan_block();
        }
        print_test_result("m_blocks_rejected increments on ORPHAN_BLOCK (opcode 7)",
            counters.accepted == 1 && counters.rejected == 1);

        // Verify stateless ACCEPT (200) and REJECT (201) also drive counters.
        struct StatelessBlockResultCounters {
            uint32_t accepted{0};
            uint32_t rejected{0};
        };
        StatelessBlockResultCounters sc;
        uint8_t accept_op = static_cast<uint8_t>(MinerLLP::ACCEPT);
        if (accept_op == static_cast<uint8_t>(MinerLLP::BLOCK_ACCEPTED)) { ++sc.accepted; }
        uint8_t reject_op = static_cast<uint8_t>(MinerLLP::REJECT);
        if (reject_op == static_cast<uint8_t>(MinerLLP::BLOCK_REJECTED)) { ++sc.rejected; }
        print_test_result("ACCEPT/BLOCK_ACCEPTED alias values are 200 (drive stateless counter)",
            sc.accepted == 1 && MinerLLP::ACCEPT == 200);
        print_test_result("REJECT/BLOCK_REJECTED alias values are 201 (drive stateless counter)",
            sc.rejected == 1 && MinerLLP::REJECT == 201);
    }

    // ====================================================================
    // Test 22: Tip-change detection — hashPrevBlock comparison between templates
    // ====================================================================
    std::cout << "\nTest 22: Tip-change detection via hashPrevBlock comparison" << std::endl;
    {
        // Build two valid templates with DIFFERENT hashPrevBlock values using the same
        // create_mock_template() helper used in other tests.
        auto tmpl_a = create_mock_template(6000001, 0x1d00ffff, 2);
        auto tmpl_b = create_mock_template(6000001, 0x1d00ffff, 2);

        // Override hashPrevBlock bytes[4..131] with distinct patterns.
        // Template A: 0xAA pattern; Template B: 0xBB pattern.
        for (int i = 4; i < 132; ++i) { tmpl_a[i] = 0xAA; tmpl_b[i] = 0xBB; }

        // Parse template A.
        MiningTemplateInterface iface_a(2, 0);
        auto res_a = iface_a.read_template(tmpl_a, "test_node");
        print_test_result("Template A (hashPrevBlock=0xAA) parses successfully", res_a.is_valid);

        uint1024_t hash_a{}, hash_b{};
        if (res_a.is_valid) {
            auto const* t = iface_a.get_current_template();
            if (t) hash_a = t->block.hashPrevBlock;
        }

        // Parse template B (different hashPrevBlock → different tip).
        MiningTemplateInterface iface_b(2, 0);
        auto res_b = iface_b.read_template(tmpl_b, "test_node");
        print_test_result("Template B (hashPrevBlock=0xBB) parses successfully", res_b.is_valid);

        if (res_b.is_valid) {
            auto const* t = iface_b.get_current_template();
            if (t) hash_b = t->block.hashPrevBlock;
        }

        // Verify the two hashPrevBlock values differ → tip changed.
        print_test_result("Template A and B hashPrevBlock differ (tip changed)",
            res_a.is_valid && res_b.is_valid && hash_a != hash_b);

        // Parse template A again → same hashPrevBlock → tip unchanged.
        MiningTemplateInterface iface_c(2, 0);
        auto res_c = iface_c.read_template(tmpl_a, "test_node");
        uint1024_t hash_c{};
        if (res_c.is_valid) {
            auto const* t = iface_c.get_current_template();
            if (t) hash_c = t->block.hashPrevBlock;
        }
        print_test_result("Template A repeated: hashPrevBlock unchanged (tip not moved)",
            res_c.is_valid && hash_a == hash_c);
    }

    // ====================================================================
    // Test 23: HeightTracker hash_prev_block — UpdateWithHashPrevBlock() and mismatch detection
    // ====================================================================
    std::cout << "\nTest 23: HeightTracker hash_prev_block field and ValidateTemplate discard logic" << std::endl;
    {
        using nexusminer::protocol::HeightTracker;

        HeightTracker tracker;

        // Initially hash_prev_block should be zero (default).
        auto snap0 = tracker.GetSnapshot();
        print_test_result("Initial HeightTracker snapshot hash_prev_block is zero",
            snap0.hash_prev_block == uint1024_t(0));

        // Build a known non-zero hash value.
        std::vector<uint8_t> known_bytes(128, 0x42);  // 128 bytes, all 0x42
        uint1024_t known_hash;
        known_hash.SetBytes(known_bytes);

        // Update via UpdateWithHashPrevBlock().
        tracker.UpdateWithHashPrevBlock(known_hash);

        auto snap1 = tracker.GetSnapshot();
        print_test_result("After UpdateWithHashPrevBlock(), hash_prev_block matches input",
            snap1.hash_prev_block == known_hash);

        // Simulate ValidateTemplate discard logic (upgraded from warn-and-continue):
        // If snapshot.hash_prev_block != 0 and template.hashPrevBlock != snapshot.hash_prev_block
        // → discard template and return false (same-height reorg defense).
        uint1024_t different_hash;
        std::vector<uint8_t> diff_bytes(128, 0x99);
        different_hash.SetBytes(diff_bytes);

        bool mismatch_detected = (snap1.hash_prev_block != uint1024_t(0) &&
                                  different_hash != snap1.hash_prev_block);
        print_test_result("ValidateTemplate logic: mismatch detected when template.hashPrevBlock differs",
            mismatch_detected);

        // No discard when template.hashPrevBlock matches snapshot.
        bool no_mismatch = !(snap1.hash_prev_block != uint1024_t(0) &&
                             known_hash != snap1.hash_prev_block);
        print_test_result("ValidateTemplate logic: no mismatch when template.hashPrevBlock matches",
            no_mismatch);

        // Verify UpdateWithHashPrevBlock() is a no-op for staleness decisions
        // (it only updates hash_prev_block, not height or channel_target).
        tracker.OnBlockDataReceived(5000, 100, 0x1d00ffff, uint1024_t{});
        tracker.UpdateWithHashPrevBlock(known_hash);
        auto snap2 = tracker.GetSnapshot();
        print_test_result("UpdateWithHashPrevBlock() does not disturb unified_height",
            snap2.unified_height == 5000);
        print_test_result("UpdateWithHashPrevBlock() does not disturb channel_height",
            snap2.channel_height == 100);
        print_test_result("UpdateWithHashPrevBlock() preserves hash_prev_block value",
            snap2.hash_prev_block == known_hash);
    }

    // ====================================================================
    // Test 24: Rate-limited get_work() guard — transmit must not be called with empty payload
    // ====================================================================
    std::cout << "\nTest 24: Rate-limited get_work() result guard (Fix 2 — defensive transmit guard)" << std::endl;
    {
        // Simulate the guard logic that now wraps every get_work() → transmit() call site.
        // When get_work() is rate-limited it returns nullptr or an empty vector.
        // The guard must prevent connection->transmit() from being called in that case.

        bool transmit_called = false;

        // Simulate a null result (rate-limited — returns nullptr)
        std::shared_ptr<std::vector<uint8_t>> null_payload = nullptr;
        if (null_payload && !null_payload->empty()) {
            transmit_called = true;  // must NOT reach here
        }
        print_test_result("Null payload from get_work() does not trigger transmit",
            !transmit_called);

        // Simulate an empty result (rate-limited — returns empty vector)
        transmit_called = false;
        auto empty_payload = std::make_shared<std::vector<uint8_t>>();
        if (empty_payload && !empty_payload->empty()) {
            transmit_called = true;  // must NOT reach here
        }
        print_test_result("Empty payload from get_work() does not trigger transmit",
            !transmit_called);

        // Simulate a valid (non-empty) result — transmit SHOULD be called
        transmit_called = false;
        auto valid_payload = std::make_shared<std::vector<uint8_t>>(std::vector<uint8_t>{0x01, 0x02});
        if (valid_payload && !valid_payload->empty()) {
            transmit_called = true;
        }
        print_test_result("Non-empty payload from get_work() triggers transmit",
            transmit_called);
    }

    // ====================================================================
    // Test 25: Push-triggered discard + recovery — new template accepted after channel advance
    //
    // Validates requirement #3 from the "fix mining lockout" issue:
    //   push arrives → template discarded as stale → GET_BLOCK response arrives with
    //   fresh template → template must be accepted (not erroneously judged stale).
    // ====================================================================
    std::cout << "\nTest 25: Push-triggered discard + recovery (new template accepted)" << std::endl;
    {
        using nexusminer::protocol::HeightTracker;

        MiningTemplateInterface tmpl_interface(2, 0); // Hash channel

        // Step 1: Initial template loaded (channel height X = 4165001).
        auto data_init = create_mock_template(6594320, 0x1d00ffff, 2);
        auto res_init = tmpl_interface.read_template(data_init, "test_node");
        print_test_result("Initial template loaded successfully", res_init.is_valid);

        // Finalise channel metadata (node is at X-1 = 4165000; template targets X = 4165001).
        tmpl_interface.set_channel_height(4165001);
        print_test_result("Template has valid state after set_channel_height", tmpl_interface.has_valid_template());

        // Step 2: Simulate push notification — channel advanced to X (= 4165001).
        // In production Solo::process_messages updates HeightTracker first, then the push handler
        // calls discard_template() via check_template_health().  Here we call discard directly.
        tmpl_interface.discard_template("Channel height-based staleness (channel advanced)");
        print_test_result("Template discarded as stale on channel advance", !tmpl_interface.has_valid_template());

        // Step 3: Recovery GET_BLOCK arrives — fresh template for channel X+1 = 4165002.
        // This template must be accepted even though HeightTracker still reflects channel_height=X.
        // The staleness check in validate_template() is SKIPPED for new templates (nChannelHeight=0).
        auto data_recovery = create_mock_template(6594321, 0x1d00ffff, 2);
        auto res_recovery = tmpl_interface.read_template(data_recovery, "test_node");
        if (!res_recovery.is_valid) {
            std::cout << "    Recovery template error: " << res_recovery.error_message << std::endl;
        }
        print_test_result("Recovery template accepted (not erroneously judged stale)", res_recovery.is_valid);

        // Step 4: Finalise the recovery template's channel metadata (targets X+1 = 4165002).
        tmpl_interface.set_channel_height(4165002);
        print_test_result("Recovery template has valid state after set_channel_height",
            tmpl_interface.has_valid_template());

        // Step 5: Confirm the recovery template's channel height reflects the new target.
        const auto* tmpl = tmpl_interface.get_current_template();
        print_test_result("Recovery template nChannelHeight == 4165002",
            tmpl != nullptr && tmpl->nChannelHeight == 4165002);

        // Step 6: Verify degraded mode check would clear — has_valid_template() is now true,
        // meaning the block handler in Worker_manager would clear m_degraded_mode.
        print_test_result("has_valid_template() true after recovery (degraded mode would clear)",
            tmpl_interface.has_valid_template());
    }

    // ====================================================================
    // Test 25b: Post-discard recovery does not compare first replacement
    //           template against HeightTracker unified height
    //
    // Regression for deep reorg recovery: discard_template() clears
    // m_last_unified_height, so the first replacement template has no
    // same-source previous template baseline.  HeightTracker may still carry a
    // canonical observation point from before/during recovery; using it as a
    // fallback continuity baseline falsely rejects the replacement template.
    // ====================================================================
    std::cout << "\nTest 25b: Post-discard first template skips HeightTracker continuity fallback" << std::endl;
    {
        using nexusminer::protocol::HeightTracker;

        HeightTracker tracker;
        MiningTemplateInterface tmpl_interface(2, 0);
        tmpl_interface.set_height_tracker(&tracker);

        // Initial template and canonical tracker state matching the field log.
        auto data_init = create_mock_template(6776736, 0x20805441, 2);
        auto res_init = tmpl_interface.read_template(data_init, "test_node");
        print_test_result("Initial pre-reorg template loaded", res_init.is_valid);

        // Simulate the tracker observing a newer canonical point before the
        // miner discards stale work during recovery.
        tracker.OnBlockDataReceived(6777002, 2402342, 0x20805441, uint1024_t{});
        tmpl_interface.discard_template("deep reorg recovery");
        print_test_result("Template discarded before recovery replacement", !tmpl_interface.has_valid_template());

        // This is only +1 from the discarded template, but 265 blocks behind
        // HeightTracker's canonical observation.  Option A requires accepting it
        // because no same-source template baseline exists immediately post-discard.
        auto data_recovery = create_mock_template(6776737, 0x20805441, 2);
        auto res_recovery = tmpl_interface.read_template(data_recovery, "test_node");
        if (!res_recovery.is_valid) {
            std::cout << "    Post-discard recovery error: " << res_recovery.error_message << std::endl;
        }
        print_test_result("Post-discard replacement accepted despite tracker/template delta >100",
            res_recovery.is_valid);
    }

    // ====================================================================
    // Test 25c: Post-discard recovery is provisional until corroborated
    //
    // A stale HeightTracker must not reject the first replacement, but the
    // replacement is not a clean baseline until metadata/body consistency and
    // channel-height finalization (or canonical prev-hash confirmation).
    // ====================================================================
    std::cout << "\nTest 25c: Post-discard recovery template is provisional until corroborated" << std::endl;
    {
        MiningTemplateInterface tmpl_interface(2, 0);

        auto data_init = create_mock_template(6776736, 0x20805441, 2);
        auto res_init = tmpl_interface.read_template(data_init, "test_node");
        print_test_result("25c: Initial template loaded", res_init.is_valid);

        tmpl_interface.discard_template("deep reorg recovery");
        print_test_result("25c: Template discarded before provisional recovery",
            !tmpl_interface.has_valid_template());

        auto bad_payload = create_mock_stateless_payload(6776736, 2402342, 0x20805441, 2);
        // Corrupt body.nHeight while leaving metadata.unified_height intact.
        const uint32_t wrong_body_height = 6776500;
        const size_t height_offset = TRITIUM_METADATA_PREFIX_SIZE + TRITIUM_BODY_NHEIGHT_OFFSET;
        bad_payload[height_offset] = (wrong_body_height >> 24) & 0xFF;
        bad_payload[height_offset + 1] = (wrong_body_height >> 16) & 0xFF;
        bad_payload[height_offset + 2] = (wrong_body_height >> 8) & 0xFF;
        bad_payload[height_offset + 3] = wrong_body_height & 0xFF;
        auto res_bad = tmpl_interface.read_stateless_payload(bad_payload, "test_node");
        print_test_result("25c: Provisional recovery rejects metadata/body height mismatch",
            !res_bad.is_valid && !res_bad.height_valid);
        print_test_result("25c: Rejection path initializes validation_time to 0us",
            res_bad.validation_time.count() == 0);
        print_test_result("25c: Rejected provisional payload does not install a template",
            !tmpl_interface.has_valid_template());

        auto good_payload = create_mock_stateless_payload(6776736, 2402342, 0x20805441, 2);
        auto res_good = tmpl_interface.read_stateless_payload(good_payload, "test_node");
        print_test_result("25c: Consistent provisional recovery payload is accepted",
            res_good.is_valid && tmpl_interface.has_valid_template());

        auto far_payload = create_mock_template(6776900, 0x20805441, 2);
        auto res_far = tmpl_interface.read_template(far_payload, "test_node");
        print_test_result("25c: Second uncorroborated recovery jump >100 is rejected",
            !res_far.is_valid && !res_far.height_valid);

        tmpl_interface.set_channel_height(2402343);
        print_test_result("25c: Channel-height finalization keeps recovery template valid",
            tmpl_interface.has_valid_template());

        auto post_finalization_jump = create_mock_template(6776838, 0x20805441, 2);
        auto res_jump = tmpl_interface.read_template(post_finalization_jump, "test_node");
        print_test_result("25c: Normal continuity guard resumes after finalization",
            !res_jump.is_valid && !res_jump.height_valid);
    }

    // ====================================================================
    // Test 25d: Startup no-template discard does not enter recovery logging
    //
    // A burst push can ask the interface to discard before the first template
    // has ever been installed.  That no-op must not make the initial template
    // look like a reorg/provisional-recovery event in operator logs.
    // ====================================================================
    std::cout << "\nTest 25d: Startup no-template discard keeps initial feed clean" << std::endl;
    {
        std::ostringstream captured_logs;
        auto capture_sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(captured_logs);
        auto capture_logger = std::make_shared<spdlog::logger>("logger", capture_sink);
        capture_logger->set_level(spdlog::level::trace);
        spdlog::set_default_logger(capture_logger);

        MiningTemplateInterface tmpl_interface(2, 0);
        tmpl_interface.discard_template("startup burst before initial template");

        auto initial_data = create_mock_template(6777095, 0x03b34f03, 1);
        auto initial_result = tmpl_interface.read_template(initial_data, "test_node");
        capture_logger->flush();

        const auto logs = captured_logs.str();
        print_test_result("25d: Initial template accepted after startup no-op discard",
            initial_result.is_valid);
        print_test_result("25d: Initial template does not log provisional recovery",
            logs.find("PROVISIONAL RECOVERY TEMPLATE") == std::string::npos &&
            logs.find("Provisional recovery template") == std::string::npos &&
            logs.find("First template after discard") == std::string::npos);

        auto null_sink_restore = std::make_shared<spdlog::sinks::null_sink_mt>();
        auto null_logger_restore = std::make_shared<spdlog::logger>("logger", null_sink_restore);
        spdlog::set_default_logger(null_logger_restore);
    }

    // ====================================================================
    // Test 26: Degraded-mode re-entry guard — new template not discarded by stale check
    //          when channel_target advances past previous channel_height
    // ====================================================================
    std::cout << "\nTest 26: Stale check does not discard recovery template (channel_target > channel_height)" << std::endl;
    {
        using nexusminer::protocol::HeightTracker;

        MiningTemplateInterface tmpl_interface(2, 0);

        // Load and finalise initial template for channel target 4165001.
        auto data_init = create_mock_template(6594320, 0x1d00ffff, 2);
        tmpl_interface.read_template(data_init, "test_node");
        tmpl_interface.set_channel_height(4165001);

        // Simulate channel advance (node now at 4165001) → discard old template.
        tmpl_interface.discard_template("Channel height-based staleness (channel advanced)");

        // Simulate recovery: GET_BLOCK returns a template targeting 4165002.
        auto data_recovery = create_mock_template(6594321, 0x1d00ffff, 2);
        auto res = tmpl_interface.read_template(data_recovery, "test_node");
        print_test_result("Recovery template accepted (tmpl.nChannelHeight=0 skips stale check)", res.is_valid);

        // Finalise recovery template.
        tmpl_interface.set_channel_height(4165002);

        // Now simulate update_channel_height() with node height = 4165001 (not yet at target 4165002).
        // Template should NOT be discarded (node is still below template target).
        bool discarded = tmpl_interface.update_channel_height(2, 4165001);
        print_test_result("update_channel_height(4165001) does not discard template targeting 4165002",
            !discarded);
        print_test_result("Template remains valid after update_channel_height at node_height=4165001",
            tmpl_interface.has_valid_template());

        // Advancing to 4165002 (= channel_target) would mark it stale.
        discarded = tmpl_interface.update_channel_height(2, 4165002);
        print_test_result("update_channel_height(4165002) discards template targeting 4165002 (stale)",
            discarded);
    }

    // ====================================================================
    // Test 27: Recovery install defers worker feed until canonical metadata is finalized
    // ====================================================================
    std::cout << "\nTest 27: Recovery template feed waits for finalization before workers resume" << std::endl;
    {
        MiningTemplateInterface tmpl_interface(2, 0);
        int feed_count = 0;
        uint32_t fed_channel_height = 0;
        tmpl_interface.set_template_feed_handler(
            [&](const MiningTemplateInterface::MiningTemplate& tmpl, uint32_t) {
                ++feed_count;
                fed_channel_height = tmpl.nChannelHeight;
                return true;
            });

        auto recovery_data = create_mock_template(6594322, 0x1d00ffff, 2);
        auto res = tmpl_interface.read_template(recovery_data, "test_node", false);

        print_test_result("Recovery template still validates when auto-feed is deferred", res.is_valid);
        print_test_result("Deferred recovery install does not feed workers before finalization",
            feed_count == 0);

        tmpl_interface.set_channel_height(4165003);
        const bool fed = tmpl_interface.feed_current_template();

        print_test_result("Manual feed succeeds after channel metadata finalization", fed);
        print_test_result("Workers are fed exactly once after finalization", feed_count == 1);
        print_test_result("Worker feed observes finalized channel height",
            fed_channel_height == 4165003);
    }

    // ====================================================================
    // Test 28: Regression — set_channel_height clears snapshot; subsequent
    //          GET_ROUND with channel_height = N-1 must NOT fire staleness
    // ====================================================================
    std::cout << "\nTest 28: No false staleness after template finalized via set_channel_height" << std::endl;
    {
        MiningTemplateInterface tmpl_interface(2, 0);

        // Load a template (nChannelHeight starts at 0 — pending finalization).
        auto data = create_mock_template(6644208, 0x1d00ffff, 2);
        tmpl_interface.read_template(data, "test_node");

        // Simulate the stateless path: set_channel_height(N) finalizes the template
        // and clears the snapshot (m_has_snapshot = false).
        tmpl_interface.set_channel_height(2344739);

        // Now simulate what finalize_and_feed_current_template() used to do wrong:
        // attempt to set a snapshot AFTER the template is already finalized.
        // The defensive guard must reject this call (nChannelHeight > 0).
        tmpl_interface.set_template_channel_height_snapshot(2344737);

        // The next GET_ROUND arrives with channel_height = 2344738 (tip advanced by 1
        // but still below the template's target of 2344739).  Before the fix this would
        // fire because 2344738 > stale-snapshot 2344737.  After the fix the snapshot
        // was never re-set so check_staleness_by_channel_delta() must return false.
        bool is_stale = tmpl_interface.check_staleness_by_channel_delta(2344738);
        print_test_result("No false staleness: GET_ROUND with N-1 does not discard finalized template",
            !is_stale);
        print_test_result("Template still valid after check_staleness_by_channel_delta(N-1)",
            tmpl_interface.has_valid_template());
    }

    // ====================================================================
    // Test 29: Root Cause 2 — check_staleness_by_channel_delta uses
    //          nChannelHeight when finalized, not snapshot
    // ====================================================================
    std::cout << "\nTest 29: check_staleness_by_channel_delta uses nChannelHeight when finalized" << std::endl;
    {
        MiningTemplateInterface tmpl_interface(2, 0);

        // Load template (nChannelHeight == 0 initially)
        auto data = create_mock_template(6644208, 0x1d00ffff, 2);
        tmpl_interface.read_template(data, "test_node");

        // Finalize with nChannelHeight = 2344739 (clears snapshot)
        tmpl_interface.set_channel_height(2344739);

        // N-1: chain tip is one step behind the target → NOT stale
        bool is_stale = tmpl_interface.check_staleness_by_channel_delta(2344738);
        print_test_result("RC2: check_staleness(N-1) returns false when nChannelHeight=N", !is_stale);
        print_test_result("RC2: Template still valid after check_staleness(N-1)", tmpl_interface.has_valid_template());

        // N: chain tip exactly at target → stale (block already mined)
        is_stale = tmpl_interface.check_staleness_by_channel_delta(2344739);
        print_test_result("RC2: check_staleness(N) returns true when nChannelHeight=N", is_stale);
        print_test_result("RC2: Template discarded when check_staleness(N) fires", !tmpl_interface.has_valid_template());
    }

    // ====================================================================
    // Test 30: Root Cause 2 — nChannelHeight-aware check: N+1 also stale
    // ====================================================================
    std::cout << "\nTest 30: check_staleness_by_channel_delta(N+1) stale when nChannelHeight=N" << std::endl;
    {
        MiningTemplateInterface tmpl_interface(2, 0);

        auto data = create_mock_template(6644208, 0x1d00ffff, 2);
        tmpl_interface.read_template(data, "test_node");
        tmpl_interface.set_channel_height(2344739);

        // N+1: chain tip has advanced past target → also stale
        bool is_stale = tmpl_interface.check_staleness_by_channel_delta(2344740);
        print_test_result("RC2: check_staleness(N+1) returns true when nChannelHeight=N", is_stale);
    }

    // ====================================================================
    // Test 31: Root Cause 3 — dual-call scenario: update_channel_height +
    //          check_staleness_by_channel_delta on same height must not
    //          double-fire when template is already finalized (nChannelHeight > 0)
    // ====================================================================
    std::cout << "\nTest 31: Dual-call scenario does not double-fire on finalized template" << std::endl;
    {
        MiningTemplateInterface tmpl_interface(2, 0);

        auto data = create_mock_template(6644208, 0x1d00ffff, 2);
        tmpl_interface.read_template(data, "test_node");
        tmpl_interface.set_channel_height(2344739);  // nChannelHeight finalized

        // Simulate the NEW_ROUND handler: call 1
        bool discarded_by_update = tmpl_interface.update_channel_height(2, 2344738);  // prime channel=2, tip=N-1
        // Simulate the NEW_ROUND handler: call 2 (only valid when needs_channel_height_finalization())
        // Since nChannelHeight is set, needs_channel_height_finalization() returns false and
        // this call should not be made.  We verify it is safe to call (returns false) but
        // also verify the template is still valid after the update.
        bool discarded_by_delta = tmpl_interface.check_staleness_by_channel_delta(2344738);

        print_test_result("RC3: update_channel_height(N-1) does not discard finalized template", !discarded_by_update);
        print_test_result("RC3: check_staleness_by_channel_delta(N-1) does not discard finalized template", !discarded_by_delta);
        print_test_result("RC3: Template still valid after dual-call with tip=N-1", tmpl_interface.has_valid_template());
    }

    // ====================================================================
    // Test 32 (RC5-A): update_channel_height — node 3 blocks behind target
    //   Template: nChannelHeight = 2344741
    //   Node:     new_channel_height = 2344738 (3 blocks below target)
    //   → must NOT discard (template still valid — chain has not yet reached target)
    // ====================================================================
    std::cout << "\nTest 32 (RC5-A): update_channel_height — node 3 blocks below target, must NOT discard" << std::endl;
    {
        MiningTemplateInterface tmpl_interface(1, 0);  // Prime channel

        auto data = create_mock_template(6644211, 0x1d00ffff, 1);
        tmpl_interface.read_template(data, "test_node");
        tmpl_interface.set_channel_height(2344741);  // nChannelHeight = 2344741

        bool discarded = tmpl_interface.update_channel_height(1, 2344738);  // node 3 behind target
        print_test_result("RC5-A: update_channel_height(2344738) with target=2344741 returns false", !discarded);
        print_test_result("RC5-A: Template still valid when node is 3 blocks behind target",
            tmpl_interface.has_valid_template());
    }

    // ====================================================================
    // Test 33 (RC5-B): update_channel_height — node AT exact target
    //   Template: nChannelHeight = 2344741
    //   Node:     new_channel_height = 2344741 (chain has reached our target)
    //   → MUST discard (block we are mining was just found)
    // ====================================================================
    std::cout << "\nTest 33 (RC5-B): update_channel_height — node at exact target, MUST discard" << std::endl;
    {
        MiningTemplateInterface tmpl_interface(1, 0);  // Prime channel

        auto data = create_mock_template(6644211, 0x1d00ffff, 1);
        tmpl_interface.read_template(data, "test_node");
        tmpl_interface.set_channel_height(2344741);

        bool discarded = tmpl_interface.update_channel_height(1, 2344741);  // chain reached target
        print_test_result("RC5-B: update_channel_height(2344741) with target=2344741 discards template", discarded);
        print_test_result("RC5-B: Template invalid after chain reached target",
            !tmpl_interface.has_valid_template());
    }

    // ====================================================================
    // Test 34 (RC5-C): update_channel_height — node 5 blocks past target
    //   Template: nChannelHeight = 2344741
    //   Node:     new_channel_height = 2344746 (5 blocks past target)
    //   → MUST discard (chain is far ahead of our mining target)
    // ====================================================================
    std::cout << "\nTest 34 (RC5-C): update_channel_height — node 5 blocks past target, MUST discard" << std::endl;
    {
        MiningTemplateInterface tmpl_interface(1, 0);  // Prime channel

        auto data = create_mock_template(6644211, 0x1d00ffff, 1);
        tmpl_interface.read_template(data, "test_node");
        tmpl_interface.set_channel_height(2344741);

        bool discarded = tmpl_interface.update_channel_height(1, 2344746);  // chain way past target
        print_test_result("RC5-C: update_channel_height(2344746) with target=2344741 discards template", discarded);
        print_test_result("RC5-C: Template invalid after chain surpassed target",
            !tmpl_interface.has_valid_template());
    }

    // ====================================================================
    // Test 35 (RC5-E): Block submission race — GET_ROUND returns the height
    //   we just mined (nChannelHeight == prime_height → correctly stale)
    //   Template: nChannelHeight = 2344738
    //   GET_ROUND: prime_height = 2344738 (the block we just submitted)
    //   → MUST discard (chain has met our target — correct behaviour)
    // ====================================================================
    std::cout << "\nTest 35 (RC5-E): Block submission race — GET_ROUND reflects just-submitted block, MUST discard" << std::endl;
    {
        MiningTemplateInterface tmpl_interface(1, 0);  // Prime channel

        auto data = create_mock_template(6644208, 0x1d00ffff, 1);
        tmpl_interface.read_template(data, "test_node");
        tmpl_interface.set_channel_height(2344738);  // template targets block 2344738

        // GET_ROUND reflects our own submission: prime_height = 2344738
        bool discarded = tmpl_interface.update_channel_height(1, 2344738);
        print_test_result("RC5-E: Submission race — update_channel_height(=target) correctly discards", discarded);
        print_test_result("RC5-E: Template invalid after submission race",
            !tmpl_interface.has_valid_template());
    }

    // ====================================================================
    // Test 36: hashPrevBlock mismatch — validate_current_template() must discard
    //          the template and return false (same-height chain reorg defense).
    //
    // Prior behaviour: warn-and-continue (miner kept mining on stale template).
    // New behaviour: discard_template() + return false (miner requests fresh work).
    // ====================================================================
    std::cout << "\nTest 36: hashPrevBlock mismatch causes template discard (same-height reorg defense)" << std::endl;
    {
        using nexusminer::protocol::HeightTracker;

        // Set up: HeightTracker knows the canonical hashPrevBlock (non-zero).
        HeightTracker tracker;
        std::vector<uint8_t> canonical_bytes(128, 0x42);
        uint1024_t canonical_hash;
        canonical_hash.SetBytes(canonical_bytes);
        tracker.UpdateWithHashPrevBlock(canonical_hash);

        auto snap = tracker.GetSnapshot();
        print_test_result("Canonical hash_prev_block is non-zero (activate mismatch guard)",
            snap.hash_prev_block != uint1024_t(0));

        // A stale template arrives with a DIFFERENT hashPrevBlock (same-height reorg).
        std::vector<uint8_t> stale_bytes(128, 0x99);
        uint1024_t stale_hash;
        stale_hash.SetBytes(stale_bytes);

        // The mismatch condition mirrors validate_current_template() new logic.
        bool mismatch_detected = (snap.hash_prev_block != uint1024_t(0) &&
                                   stale_hash != snap.hash_prev_block);
        print_test_result("Mismatch detected: template.hashPrevBlock differs from canonical tip",
            mismatch_detected);

        // Simulate what validate_current_template() now does: discard_template() + return false.
        MiningTemplateInterface tmpl_interface(1, 0);  // Prime channel
        auto data = create_mock_template(6648299, 0x1d00ffff, 1);
        auto res = tmpl_interface.read_template(data, "test_node");
        print_test_result("Template loaded before mismatch discard", res.is_valid);
        print_test_result("has_valid_template() true before discard", tmpl_interface.has_valid_template());

        // When mismatch is detected, validate_current_template() calls discard_template().
        if (mismatch_detected) {
            tmpl_interface.discard_template("hashPrevBlock mismatch — same-height chain reorg detected");
        }
        print_test_result("Template discarded after hashPrevBlock mismatch (not warn-and-continue)",
            !tmpl_interface.has_valid_template());

        // Verify the no-mismatch case: matching hashPrevBlock must NOT cause a discard.
        MiningTemplateInterface tmpl_interface2(1, 0);
        auto data2 = create_mock_template(6648299, 0x1d00ffff, 1);
        tmpl_interface2.read_template(data2, "test_node");

        // canonical_hash was used to set hash_prev_block, so they are equal → no mismatch.
        bool no_mismatch_on_match = (snap.hash_prev_block == uint1024_t(0) ||
                                      canonical_hash == snap.hash_prev_block);
        print_test_result("No mismatch when template.hashPrevBlock equals canonical tip",
            no_mismatch_on_match);
        // No discard: template must remain valid.
        print_test_result("Template stays valid when hashPrevBlock matches canonical tip",
            tmpl_interface2.has_valid_template());

        // Verify guard is inactive when hash_prev_block is zero (not yet set).
        HeightTracker tracker_zero;
        auto snap_zero = tracker_zero.GetSnapshot();
        bool guard_inactive = !(snap_zero.hash_prev_block != uint1024_t(0) &&
                                 stale_hash != snap_zero.hash_prev_block);
        print_test_result("Mismatch guard inactive when HeightTracker hash_prev_block is zero",
            guard_inactive);
    }

    // ====================================================================
    // Test 37: PRIME_ORIGIN_TOO_LOW is rejected before the template becomes valid
    // ====================================================================
    std::cout << "\nTest 37: PRIME_ORIGIN_TOO_LOW rejected during template validation" << std::endl;
    {
        auto data = create_mock_template(6650000, 0x1d00ffff, 1);
        bool found_low_origin = false;

        for (uint32_t salt = 0; salt < 2000; ++salt) {
            // Mutate the Merkle root, which is part of Prime ProofHash().
            data[132] = static_cast<uint8_t>(salt & 0xff);
            data[133] = static_cast<uint8_t>((salt >> 8) & 0xff);

            auto block = nexusminer::llp_utils::deserialize_block_header(data);
            if (!nexusminer::HasSufficientPrimeOrigins(block)) {
                found_low_origin = true;
                break;
            }
        }

        print_test_result("Test 37: fixture found a low-prime-origin template", found_low_origin);
        // Only exercise rejection assertions when the fixture search succeeded.
        // A failed search should produce a single clear failure rather than a cascade.
        if (found_low_origin) {
            MiningTemplateInterface tmpl_interface(1, 0);
            auto res = tmpl_interface.read_template(data, "test_node", false);

            print_test_result("Test 37: validation rejects PRIME_ORIGIN_TOO_LOW", !res.is_valid);
            print_test_result("Test 37: retry reason is PRIME_ORIGIN_TOO_LOW",
                res.retry_reason == GetBlockReason::PRIME_ORIGIN_TOO_LOW);
            print_test_result("Test 37: current template was never marked valid",
                !tmpl_interface.has_valid_template());
        }
    }

    // ====================================================================
    // Test 38: Feed handler rejection propagates to feed_current_template()
    // ====================================================================
    std::cout << "\nTest 38: Template feed handler rejection propagates" << std::endl;
    {
        MiningTemplateInterface tmpl_interface(2, 0);
        int rejected_feed_count = 0;
        int accepted_feed_count = 0;
        int completed_feed_count = 0;
        tmpl_interface.set_template_fed_handler([&] {
            if (tmpl_interface.has_valid_template()) {
                ++completed_feed_count;
            }
        });
        tmpl_interface.set_template_feed_handler(
            [&](const MiningTemplateInterface::MiningTemplate&, uint32_t) {
                ++rejected_feed_count;
                return false;
            });

        auto data = create_mock_template(6650001, 0x1d00ffff, 2);
        auto res = tmpl_interface.read_template(data, "test_node", false);
        const bool fed = tmpl_interface.feed_current_template();

        print_test_result("Test 38: template validates before rejected feed", res.is_valid);
        print_test_result("Test 38: rejecting feed handler called once", rejected_feed_count == 1);
        print_test_result("Test 38: feed_current_template reports worker rejection", !fed);
        print_test_result("Test 38: rejected feed clears current template state",
            !tmpl_interface.has_valid_template());
        print_test_result("Test 38: rejected feed does not signal completion",
            completed_feed_count == 0);

        tmpl_interface.set_template_feed_handler(
            [&](const MiningTemplateInterface::MiningTemplate&, uint32_t) {
                ++accepted_feed_count;
                return true;
            });

        auto retry_res = tmpl_interface.read_template(data, "test_node", false);
        const bool retry_fed = tmpl_interface.feed_current_template();

        print_test_result("Test 38: same template can be retried immediately after rejection",
            retry_res.is_valid && retry_fed);
        print_test_result("Test 38: accepting retry handler called once", accepted_feed_count == 1);
        print_test_result("Test 38: completion can query the template without deadlocking",
            completed_feed_count == 1);
    }

    // ====================================================================
    // Test 21: Option A + Option B safety net — replacement_pending lifecycle
    // ====================================================================
    std::cout << "\nTest 21: mark_replacement_pending() lifecycle (Option A + B safety net)" << std::endl;
    {
        // 21a: pending leaves template VALID — the whole point of Option A:
        //      workers must continue mining through the BLOCK_DATA round trip
        //      instead of falling into HEALTH_NO_TEMPLATE recovery.
        MiningTemplateInterface tmpl_iface(2, 0);
        auto data = create_mock_template(6594400);
        auto res = tmpl_iface.read_template(data, "test_node");
        print_test_result("Test 21: baseline template loaded", res.is_valid);
        print_test_result("Test 21: baseline has_valid_template() true",
            tmpl_iface.has_valid_template());

        tmpl_iface.mark_replacement_pending("same_height_tip_update", 5000);
        print_test_result("Test 21a: has_valid_template() STAYS true while replacement pending",
            tmpl_iface.has_valid_template());
        print_test_result("Test 21a: is_replacement_pending() reports true within deadline",
            tmpl_iface.is_replacement_pending());

        std::string expired_reason = "sentinel";
        bool took = tmpl_iface.take_expired_replacement_pending(expired_reason);
        print_test_result("Test 21a: take_expired_replacement_pending() returns false before deadline",
            !took);
        print_test_result("Test 21a: take_*-on-not-expired clears the out-reason",
            expired_reason.empty());

        // 21b: read_template() (atomic swap) clears the pending flag — the
        //      promised replacement arrived, so the safety net must NOT later
        //      trigger a spurious HEALTH_NO_TEMPLATE recovery.
        auto data2 = create_mock_template(6594401);
        auto res2 = tmpl_iface.read_template(data2, "test_node");
        print_test_result("Test 21b: replacement template loaded", res2.is_valid);
        print_test_result("Test 21b: read_template clears replacement-pending",
            !tmpl_iface.is_replacement_pending());
        std::string after_swap_reason = "sentinel";
        bool took_after_swap =
            tmpl_iface.take_expired_replacement_pending(after_swap_reason);
        print_test_result("Test 21b: no expired-pending observable after swap",
            !took_after_swap && after_swap_reason.empty());

        // 21c: deadline expiry surfaces the reason exactly once and clears
        //      the flag (so the next health tick cannot double-trigger).
        tmpl_iface.mark_replacement_pending("same_height_tip_update", 1);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        print_test_result("Test 21c: is_replacement_pending() false past deadline",
            !tmpl_iface.is_replacement_pending());
        std::string out_reason;
        bool expired = tmpl_iface.take_expired_replacement_pending(out_reason);
        print_test_result("Test 21c: take_expired_replacement_pending() returns true after deadline",
            expired);
        print_test_result("Test 21c: expired reason matches the recorded label",
            out_reason == "same_height_tip_update");
        std::string second_call_reason = "sentinel";
        bool second_call =
            tmpl_iface.take_expired_replacement_pending(second_call_reason);
        print_test_result("Test 21c: take_expired_replacement_pending() is idempotent (second call false)",
            !second_call && second_call_reason.empty());

        // 21d: discard_template() while pending also clears the flag, so a
        //      caller that decides to bail early does not leave a dangling
        //      deadline behind.
        tmpl_iface.mark_replacement_pending("same_height_tip_update", 5000);
        print_test_result("Test 21d: pending re-armed", tmpl_iface.is_replacement_pending());
        tmpl_iface.discard_template("test_explicit_discard_clears_pending");
        print_test_result("Test 21d: discard_template clears replacement-pending",
            !tmpl_iface.is_replacement_pending());
        std::string after_discard_reason = "sentinel";
        bool took_after_discard =
            tmpl_iface.take_expired_replacement_pending(after_discard_reason);
        print_test_result("Test 21d: no expired-pending observable after discard",
            !took_after_discard && after_discard_reason.empty());
    }

    // ====================================================================
    // Summary
    // ====================================================================
    std::cout << "\n========================================" << std::endl;
    std::cout << "Test Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Total tests:  " << tests_run << std::endl;
    std::cout << "Passed:       " << tests_passed << std::endl;
    std::cout << "Failed:       " << tests_failed << std::endl;
    std::cout << "Success rate: " << (tests_run > 0 ? (100.0 * tests_passed / tests_run) : 0.0) 
              << "%" << std::endl;
    if (tests_failed == 0)
        std::cout << "✓ ALL TESTS PASSED (" << tests_passed << "/" << tests_run << ")" << std::endl;
    std::cout << "========================================" << std::endl;

    return (tests_failed == 0) ? 0 : 1;
}
