/*__________________________________________________________________________________________

            Hash(BEGIN(Satoshi[2010]), END(Sunny[2012])) == Videlicet[2014]++

            (c) Copyright The Nexus Developers 2014 - 2023

            Distributed under the MIT software license, see the accompanying
            file COPYING or http://www.opensource.org/licenses/mit-license.php.

            "ad vocem populi" - To the Voice of the People

____________________________________________________________________________________________*/

/**
 * Unit tests for genesis_utils.hpp and serialization_helpers.hpp
 * These utilities were extracted to eliminate duplication between solo.cpp and session_manager.cpp
 */

#include "protocol/genesis_utils.hpp"
#include "protocol/serialization_helpers.hpp"
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cassert>

// Simple test framework
int test_count = 0;
int pass_count = 0;

void test_assert(bool condition, const char* test_name) {
    test_count++;
    if (condition) {
        std::cout << "  [PASS] " << test_name << std::endl;
        pass_count++;
    } else {
        std::cout << "  [FAIL] " << test_name << std::endl;
    }
}

void test_genesis_validation() {
    std::cout << "\nTest 1: Genesis validation - empty vector" << std::endl;
    {
        std::vector<uint8_t> empty;
        test_assert(!genesis_utils::is_valid_genesis(empty), "Empty genesis is invalid");
    }

    std::cout << "\nTest 2: Genesis validation - all zeros" << std::endl;
    {
        std::vector<uint8_t> all_zeros(32, 0);
        test_assert(!genesis_utils::is_valid_genesis(all_zeros), "All-zero genesis is invalid");
    }

    std::cout << "\nTest 3: Genesis validation - single non-zero byte" << std::endl;
    {
        std::vector<uint8_t> genesis(32, 0);
        genesis[0] = 1;
        test_assert(genesis_utils::is_valid_genesis(genesis), "Genesis with first byte non-zero is valid");
    }

    std::cout << "\nTest 4: Genesis validation - last byte non-zero" << std::endl;
    {
        std::vector<uint8_t> genesis(32, 0);
        genesis[31] = 0xFF;
        test_assert(genesis_utils::is_valid_genesis(genesis), "Genesis with last byte non-zero is valid");
    }

    std::cout << "\nTest 5: Genesis validation - typical genesis hash" << std::endl;
    {
        std::vector<uint8_t> genesis = {
            0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
            0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10,
            0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88,
            0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00
        };
        test_assert(genesis_utils::is_valid_genesis(genesis), "Typical genesis hash is valid");
    }

    std::cout << "\nTest 6: has_mainnet_genesis_type - correct mainnet byte 0xa1" << std::endl;
    {
        std::vector<uint8_t> genesis(32, 0);
        genesis[0] = 0xa1;
        test_assert(genesis_utils::has_mainnet_genesis_type(genesis),
                    "Genesis with leading byte 0xa1 is mainnet type");
    }

    std::cout << "\nTest 7: has_mainnet_genesis_type - register address type byte 0x02 rejected" << std::endl;
    {
        std::vector<uint8_t> genesis(32, 0);
        genesis[0] = 0x02;  // OBJECT register type — not a valid coinbase genesis
        test_assert(!genesis_utils::has_mainnet_genesis_type(genesis),
                    "Genesis with leading byte 0x02 (register) is NOT mainnet type");
    }

    std::cout << "\nTest 8: has_mainnet_genesis_type - wrong size rejected" << std::endl;
    {
        std::vector<uint8_t> too_short(16, 0xa1);
        test_assert(!genesis_utils::has_mainnet_genesis_type(too_short),
                    "16-byte vector is not a valid mainnet genesis");
    }

    std::cout << "\nTest 9: has_testnet_genesis_type - correct testnet byte 0xb1" << std::endl;
    {
        std::vector<uint8_t> genesis(32, 0);
        genesis[0] = 0xb1;
        test_assert(genesis_utils::has_testnet_genesis_type(genesis),
                    "Genesis with leading byte 0xb1 is testnet type");
    }

    std::cout << "\nTest 10: Hex-decode of known 64-char genesis hex produces correct 32-byte vector" << std::endl;
    {
        // Known: mainnet genesis hash starting with 0xa1
        const std::string hex = "a1000000000000000000000000000000"
                                "00000000000000000000000000000001";
        auto decoded = genesis_utils::hex_decode_genesis_hash(hex);
        test_assert(decoded.size() == 32, "Hex decode produces 32 bytes");
        test_assert(!decoded.empty() && decoded[0] == 0xa1, "First byte is 0xa1 (mainnet type)");
        test_assert(!decoded.empty() && decoded[31] == 0x01, "Last byte is 0x01");
        test_assert(genesis_utils::has_mainnet_genesis_type(decoded),
                    "Decoded genesis passes has_mainnet_genesis_type");
    }

    std::cout << "\nTest 10b: hex_decode_genesis_hash rejects non-64-char strings" << std::endl;
    {
        test_assert(genesis_utils::hex_decode_genesis_hash("").empty(), "Empty string returns empty");
        test_assert(genesis_utils::hex_decode_genesis_hash("a1b2c3").empty(), "Short string returns empty");
        std::string too_long(66, 'a');
        test_assert(genesis_utils::hex_decode_genesis_hash(too_long).empty(), "66-char string returns empty");
    }

    std::cout << "\nTest 10c: hex_decode_genesis_hash rejects non-hex characters" << std::endl;
    {
        // 64 chars but with 'g' and 'z' (not valid hex)
        const std::string bad_hex = "g1000000000000000000000000000000"
                                    "0000000000000000000000000000000z";
        test_assert(genesis_utils::hex_decode_genesis_hash(bad_hex).empty(),
                    "Non-hex characters return empty");
    }

    std::cout << "\nTest 10d: looks_like_base58_address helper" << std::endl;
    {
        test_assert(genesis_utils::looks_like_base58_address("8BMeGoCm4TysEn4MseoEwme9yVwkWRqGPYC4m1iHsFzb16udDRg"),
                    "Typical Base58 NXS address is detected");
        test_assert(!genesis_utils::looks_like_base58_address("a1000000000000000000000000000000"
                                                               "00000000000000000000000000000001"),
                    "64-char hex genesis hash is NOT flagged as Base58");
        test_assert(!genesis_utils::looks_like_base58_address(""), "Empty string is not Base58");
    }
}

void test_swap_genesis_word_order() {
    // Historical context (LLL-TAO PR #578):
    //
    // Before LLL-TAO PR #578 the node emitted the session genesis via
    // base_uint::GetBytes() which serialises pn[0..7] LSW-first — opposite to the
    // GetHex() / display order used by 'system/get/info'.  NexusMiner was compensating
    // with swap_genesis_word_order() in the SESSION_START handler.
    //
    // LLL-TAO PR #578 fixed the node to emit genesis in display/MSW-first order.
    // The solo.cpp comparison now works directly (no swap).  These tests document the
    // word-order invariant and confirm the utility is still mathematically correct.
    //
    // Values from live miner logs (screenshot 2026-05-04):
    //   display order (GetHex / reward_address / 'system/get/info'):
    //     a174011c93ca1c80bca5388382b167cacd33d3154395ea8f45ac99a8308cd122
    //   old LSW-first wire order (GetBytes() / pre-PR#578 SESSION_START):
    //     308cd12245ac99a84395ea8fcd33d31582b167cabca5388393ca1c80a174011c
    // Both represent the same 256-bit integer with 8 uint32_t words in opposite order.

    const std::vector<uint8_t> display_bytes = {
        0xa1,0x74,0x01,0x1c, 0x93,0xca,0x1c,0x80,
        0xbc,0xa5,0x38,0x83, 0x82,0xb1,0x67,0xca,
        0xcd,0x33,0xd3,0x15, 0x43,0x95,0xea,0x8f,
        0x45,0xac,0x99,0xa8, 0x30,0x8c,0xd1,0x22
    };
    const std::vector<uint8_t> lsw_first_bytes = {
        0x30,0x8c,0xd1,0x22, 0x45,0xac,0x99,0xa8,
        0x43,0x95,0xea,0x8f, 0xcd,0x33,0xd3,0x15,
        0x82,0xb1,0x67,0xca, 0xbc,0xa5,0x38,0x83,
        0x93,0xca,0x1c,0x80, 0xa1,0x74,0x01,0x1c
    };

    std::cout << "\nTest 20: swap_genesis_word_order - known display-vs-LSW-first pair" << std::endl;
    {
        // swap(LSW-first) == display
        const auto swapped = genesis_utils::swap_genesis_word_order(lsw_first_bytes);
        test_assert(swapped == display_bytes,
                    "swap_genesis_word_order(LSW-first) == display bytes");

        // swap(display) == LSW-first  (function is its own inverse)
        const auto swapped_back = genesis_utils::swap_genesis_word_order(display_bytes);
        test_assert(swapped_back == lsw_first_bytes,
                    "swap_genesis_word_order(display) == LSW-first bytes (inverse property)");
    }

    std::cout << "\nTest 21: swap_genesis_word_order - round-trip identity" << std::endl;
    {
        std::vector<uint8_t> original(32);
        for (int i = 0; i < 32; ++i) original[i] = static_cast<uint8_t>(i * 7 + 3);
        const auto twice = genesis_utils::swap_genesis_word_order(
                               genesis_utils::swap_genesis_word_order(original));
        test_assert(twice == original, "Applying swap twice returns original bytes");
    }

    std::cout << "\nTest 22: swap_genesis_word_order - wrong size passthrough" << std::endl;
    {
        std::vector<uint8_t> short_vec = {0x01, 0x02, 0x03};
        test_assert(genesis_utils::swap_genesis_word_order(short_vec) == short_vec,
                    "Non-32-byte input returned unchanged");
        std::vector<uint8_t> empty;
        test_assert(genesis_utils::swap_genesis_word_order(empty) == empty,
                    "Empty input returned unchanged");
    }

    // ------------------------------------------------------------------
    // Test 23: Post-LLL-TAO-PR#578 wire-contract pin
    //
    // The node now sends genesis in display/MSW-first order (same as GetHex()).
    // solo.cpp compares the node-echoed bytes directly against hex_decode_genesis_hash()
    // of the configured reward_address — NO word-swap.  This test pins that contract:
    //   hex_decode(reward_address) == node_session_genesis_wire  (no swap needed)
    // ------------------------------------------------------------------
    std::cout << "\nTest 23: Post-PR#578 SESSION_START genesis - direct compare, no swap (LLL-TAO PR #578)" << std::endl;
    {
        const auto reward_bytes =
            genesis_utils::hex_decode_genesis_hash(
                "a174011c93ca1c80bca5388382b167cacd33d3154395ea8f45ac99a8308cd122");
        // After LLL-TAO PR #578: node sends display_bytes directly on the wire.
        const auto& node_session_wire = display_bytes;

        test_assert(reward_bytes == node_session_wire,
                    "Direct compare: hex_decode(reward_address) == post-PR#578 node wire bytes");
        test_assert(reward_bytes != lsw_first_bytes,
                    "Direct compare would FAIL against old pre-PR#578 LSW-first bytes (regression guard)");
    }
}

void test_serialization_helpers() {
    std::cout << "\nTest 11: append_uint32_le - basic serialization" << std::endl;
    {
        std::vector<uint8_t> data;
        serialization::append_uint32_le(data, 0x12345678);
        test_assert(data.size() == 4, "uint32 produces 4 bytes");
        test_assert(data[0] == 0x78, "Byte 0 is LSB (0x78)");
        test_assert(data[1] == 0x56, "Byte 1 is second byte (0x56)");
        test_assert(data[2] == 0x34, "Byte 2 is third byte (0x34)");
        test_assert(data[3] == 0x12, "Byte 3 is MSB (0x12)");
    }

    std::cout << "\nTest 12: append_uint32_le - zero value" << std::endl;
    {
        std::vector<uint8_t> data;
        serialization::append_uint32_le(data, 0);
        test_assert(data.size() == 4, "uint32 zero produces 4 bytes");
        test_assert(data[0] == 0 && data[1] == 0 && data[2] == 0 && data[3] == 0,
                   "All bytes are zero");
    }

    std::cout << "\nTest 13: append_uint32_le - max value" << std::endl;
    {
        std::vector<uint8_t> data;
        serialization::append_uint32_le(data, 0xFFFFFFFF);
        test_assert(data.size() == 4, "uint32 max produces 4 bytes");
        test_assert(data[0] == 0xFF && data[1] == 0xFF && data[2] == 0xFF && data[3] == 0xFF,
                   "All bytes are 0xFF");
    }

    std::cout << "\nTest 14: append_uint64_le - basic serialization" << std::endl;
    {
        std::vector<uint8_t> data;
        serialization::append_uint64_le(data, 0x0123456789ABCDEF);
        test_assert(data.size() == 8, "uint64 produces 8 bytes");
        test_assert(data[0] == 0xEF, "Byte 0 is LSB (0xEF)");
        test_assert(data[7] == 0x01, "Byte 7 is MSB (0x01)");
    }

    std::cout << "\nTest 15: read_uint32_le - basic deserialization" << std::endl;
    {
        std::vector<uint8_t> data = {0x78, 0x56, 0x34, 0x12};
        uint32_t value = serialization::read_uint32_le(data);
        test_assert(value == 0x12345678, "Reads correct uint32 value");
    }

    std::cout << "\nTest 16: read_uint32_le - with offset" << std::endl;
    {
        std::vector<uint8_t> data = {0xFF, 0xFF, 0x78, 0x56, 0x34, 0x12};
        uint32_t value = serialization::read_uint32_le(data, 2);
        test_assert(value == 0x12345678, "Reads correct uint32 value with offset");
    }

    std::cout << "\nTest 17: read_uint32_le - insufficient data" << std::endl;
    {
        std::vector<uint8_t> data = {0x78, 0x56};
        uint32_t value = serialization::read_uint32_le(data);
        test_assert(value == 0, "Returns 0 for insufficient data");
    }

    std::cout << "\nTest 18: round-trip append/read uint32" << std::endl;
    {
        std::vector<uint8_t> data;
        uint32_t original = 0xDEADBEEF;
        serialization::append_uint32_le(data, original);
        uint32_t decoded = serialization::read_uint32_le(data);
        test_assert(decoded == original, "Round-trip preserves value");
    }

    std::cout << "\nTest 19: multiple appends" << std::endl;
    {
        std::vector<uint8_t> data;
        serialization::append_uint32_le(data, 0x11111111);
        serialization::append_uint32_le(data, 0x22222222);
        test_assert(data.size() == 8, "Two uint32s produce 8 bytes");
        test_assert(serialization::read_uint32_le(data, 0) == 0x11111111, "First value correct");
        test_assert(serialization::read_uint32_le(data, 4) == 0x22222222, "Second value correct");
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Genesis & Serialization Utils Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    test_genesis_validation();
    test_swap_genesis_word_order();
    test_serialization_helpers();

    std::cout << "\n========================================" << std::endl;
    std::cout << "Test Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Tests run:    " << test_count << std::endl;
    std::cout << "Tests passed: " << pass_count << std::endl;
    std::cout << "Tests failed: " << (test_count - pass_count) << std::endl;
    std::cout << "Success rate: " << (100 * pass_count / test_count) << "%" << std::endl;
    std::cout << "========================================" << std::endl;

    return (pass_count == test_count) ? 0 : 1;
}
