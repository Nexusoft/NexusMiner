/**
 * Unit tests for MerkleRootFeedGuard
 *
 * Verifies that:
 *   1. First feed always proceeds
 *   2. Same hashMerkleRoot within 2s is suppressed
 *   3. Different hashMerkleRoot at different heights always proceeds
 *   4. Same hashMerkleRoot after window expires proceeds (implicit — no sleep test)
 *   5. Zero (empty) hashMerkleRoot is never subject to suppression dedup
 *   6. reset() clears all state
 *   7. Same merkle root at different heights is allowed (height keying)
 *   8. [NEW] Same height, different merkle root within 500ms is suppressed (height-only guard)
 *   9. [NEW] Same height, different hashPrevBlock (reorg) always allowed
 *  10. [NEW] HEIGHT_ONLY_SUPPRESSION_MS constant check
 */

#include "protocol/merkle_root_feed_guard.hpp"
#include <cassert>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <thread>

using nexusminer::protocol::MerkleRootFeedGuard;

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

int main()
{
    std::cout << "=== MerkleRootFeedGuard Tests ===" << std::endl;

    // --- Test 1: First call always allows feed ---
    {
        MerkleRootFeedGuard guard;
        uint512_t merkle;
        merkle.SetHex("abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
                       "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890");
        test_assert(guard.should_feed(merkle), "first feed always allowed");
        test_assert(guard.suppressed_count() == 0, "suppressed count is 0 after first feed");
    }

    // --- Test 2: Same merkle root within window is suppressed ---
    {
        MerkleRootFeedGuard guard;
        uint512_t merkle;
        merkle.SetHex("1111111111111111111111111111111111111111111111111111111111111111"
                       "1111111111111111111111111111111111111111111111111111111111111111");
        test_assert(guard.should_feed(merkle), "first feed allowed");
        test_assert(!guard.should_feed(merkle), "duplicate within window suppressed");
        test_assert(guard.suppressed_count() == 1, "suppressed count incremented");
        test_assert(!guard.should_feed(merkle), "third duplicate also suppressed");
        test_assert(guard.suppressed_count() == 2, "suppressed count incremented again");
    }

    // --- Test 3: Different merkle root at different heights always allowed ---
    {
        MerkleRootFeedGuard guard;
        uint512_t merkle_a, merkle_b;
        merkle_a.SetHex("aaaa111111111111111111111111111111111111111111111111111111111111"
                         "1111111111111111111111111111111111111111111111111111111111111111");
        merkle_b.SetHex("bbbb222222222222222222222222222222222222222222222222222222222222"
                         "2222222222222222222222222222222222222222222222222222222222222222");
        // No height specified (0) → height key ignored, different merkle root always passes
        test_assert(guard.should_feed(merkle_a), "first merkle allowed");
        test_assert(guard.should_feed(merkle_b), "different merkle allowed immediately (no height)");
        test_assert(guard.suppressed_count() == 0, "no suppression for different roots (no height)");
    }

    // --- Test 4: Zero (empty) merkle root is never deduplicated ---
    {
        MerkleRootFeedGuard guard;
        uint512_t zero{};
        test_assert(guard.should_feed(zero), "zero merkle first call allowed");
        test_assert(guard.should_feed(zero), "zero merkle second call also allowed (no dedup on zero)");
        test_assert(guard.suppressed_count() == 0, "zero merkle never suppressed");
    }

    // --- Test 5: reset() clears state ---
    {
        MerkleRootFeedGuard guard;
        uint512_t merkle;
        merkle.SetHex("cccc333333333333333333333333333333333333333333333333333333333333"
                       "3333333333333333333333333333333333333333333333333333333333333333");
        guard.should_feed(merkle);
        guard.should_feed(merkle);  // suppressed
        test_assert(guard.suppressed_count() == 1, "pre-reset suppressed count");
        guard.reset();
        test_assert(guard.suppressed_count() == 0, "post-reset suppressed count cleared");
        test_assert(guard.should_feed(merkle), "post-reset same merkle allowed again");
    }

    // --- Test 6: Suppression window constant ---
    {
        test_assert(MerkleRootFeedGuard::SUPPRESSION_WINDOW_SECONDS == 2,
                    "suppression window is 2 seconds");
        test_assert(MerkleRootFeedGuard::HEIGHT_ONLY_SUPPRESSION_MS == 500,
                    "height-only suppression window is 500ms");
    }

    // --- Test 7: Same merkle root at different heights is allowed ---
    {
        MerkleRootFeedGuard guard;
        uint512_t merkle;
        merkle.SetHex("dddd444444444444444444444444444444444444444444444444444444444444"
                       "4444444444444444444444444444444444444444444444444444444444444444");
        test_assert(guard.should_feed(merkle, 1000), "first feed at height 1000 allowed");
        test_assert(!guard.should_feed(merkle, 1000), "same merkle+height within window suppressed");
        test_assert(guard.should_feed(merkle, 1001), "same merkle at different height allowed");
        test_assert(guard.suppressed_count() == 1, "only one suppression");
    }

    // --- Test 8: Same height, different merkle root within 500ms is suppressed (height-only guard) ---
    {
        MerkleRootFeedGuard guard;
        uint512_t merkle_a, merkle_b;
        uint1024_t prev_hash;
        merkle_a.SetHex("eeee555555555555555555555555555555555555555555555555555555555555"
                         "5555555555555555555555555555555555555555555555555555555555555555");
        merkle_b.SetHex("ffff666666666666666666666666666666666666666666666666666666666666"
                         "6666666666666666666666666666666666666666666666666666666666666666");
        prev_hash.SetHex("1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
                          "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
                          "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
                          "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef");

        // First feed at height 2000 with merkle_a
        test_assert(guard.should_feed(merkle_a, 2000, prev_hash),
                    "first feed at height 2000 allowed");

        // Same height, different merkle root, same hashPrevBlock → suppressed by height-only guard
        test_assert(!guard.should_feed(merkle_b, 2000, prev_hash),
                    "same height, different merkle root within 500ms suppressed");
        test_assert(guard.suppressed_count() == 1, "height-only suppression incremented count");
    }

    // --- Test 9: Same height, different hashPrevBlock (reorg) always allowed ---
    {
        MerkleRootFeedGuard guard;
        uint512_t merkle_a, merkle_b;
        uint1024_t prev_hash_a, prev_hash_b;
        merkle_a.SetHex("aaaa777777777777777777777777777777777777777777777777777777777777"
                         "7777777777777777777777777777777777777777777777777777777777777777");
        merkle_b.SetHex("bbbb888888888888888888888888888888888888888888888888888888888888"
                         "8888888888888888888888888888888888888888888888888888888888888888");
        prev_hash_a.SetHex("aaaa000000000000000000000000000000000000000000000000000000000000"
                            "0000000000000000000000000000000000000000000000000000000000000000"
                            "0000000000000000000000000000000000000000000000000000000000000000"
                            "0000000000000000000000000000000000000000000000000000000000000000");
        prev_hash_b.SetHex("bbbb000000000000000000000000000000000000000000000000000000000000"
                            "0000000000000000000000000000000000000000000000000000000000000000"
                            "0000000000000000000000000000000000000000000000000000000000000000"
                            "0000000000000000000000000000000000000000000000000000000000000000");

        // First feed at height 3000 with prev_hash_a
        test_assert(guard.should_feed(merkle_a, 3000, prev_hash_a),
                    "first feed at height 3000 with prev_hash_a allowed");

        // Same height, different hashPrevBlock (reorg) → always allowed
        test_assert(guard.should_feed(merkle_b, 3000, prev_hash_b),
                    "same height, different hashPrevBlock (reorg) allowed");
        test_assert(guard.suppressed_count() == 0, "no suppression for reorg");
    }

    // --- Test 10: Different height with same merkle root is always allowed ---
    {
        MerkleRootFeedGuard guard;
        uint512_t merkle;
        uint1024_t prev_hash;
        merkle.SetHex("abcd999999999999999999999999999999999999999999999999999999999999"
                       "9999999999999999999999999999999999999999999999999999999999999999");
        prev_hash.SetHex("1111000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000000"
                          "0000000000000000000000000000000000000000000000000000000000000000");

        test_assert(guard.should_feed(merkle, 4000, prev_hash), "first feed at height 4000 allowed");
        test_assert(guard.should_feed(merkle, 4001, prev_hash), "height change always allowed");
        test_assert(guard.suppressed_count() == 0, "no suppression on height change");
    }

    // --- Summary ---
    std::cout << "\n=== Results: " << pass_count << "/" << test_count << " passed ===" << std::endl;
    return (pass_count == test_count) ? 0 : 1;
}
