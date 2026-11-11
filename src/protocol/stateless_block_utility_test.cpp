/**
 * @file stateless_block_utility_test.cpp
 * @brief Unit tests for StatelessBlockUtility encode/decode utility
 *
 * Tests (acceptance criteria):
 *  1.  decode_template() with a valid 228-byte buffer succeeds
 *  2.  decode_template() rejects payloads shorter than 228 bytes
 *  3.  decode_template() rejects payloads longer than 228 bytes
 *  4.  encode_submit() rejects zero nonce
 *  5.  encode_submit() rejects invalid channel (e.g. 0)
 *  6.  encode_submit() rejects zero height
 *  7.  encode_submit() with STATELESS lane produces opcode 0xD001
 *  8.  encode_submit() with LEGACY lane produces opcode 0x01
 *  9.  encode_submit() with falcon=nullptr produces unsigned submit
 *      (payload = block bytes + any vOffsets, no Falcon signature suffix)
 * 10.  decode_template() populates metadata fields from 12-byte prefix (BE)
 * 11.  decode_template() populates canonical block fields from 216-byte body
 * 12.  decode_template() sets channel_consistent correctly
 * 13.  encode_submit() informational staleness/tip-moved do not block submission
 * 14.  encode_submit() with Prime-channel vOffsets produces larger payload than
 *      Hash-channel (no vOffsets) submission
 */

#include "include/stateless_block_utility.hpp"
#include "LLP/block_utils.hpp"
#include "protocol/mining_template_interface.hpp"
#include "protocol/height_tracker.hpp"
#include "worker/block_header_utils.hpp"
#include "miner_opcodes.hpp"
#include <iostream>
#include <cassert>
#include <cstdint>
#include <vector>
#include <cstring>

using namespace nexusminer;
using namespace nexusminer::protocol;

// ── Test infrastructure ──────────────────────────────────────────────────────
static int tests_run    = 0;
static int tests_passed = 0;
static int tests_failed = 0;

static void print_result(const char* name, bool passed) {
    tests_run++;
    if (passed) {
        tests_passed++;
        std::cout << "  [PASS] " << name << "\n";
    } else {
        tests_failed++;
        std::cout << "  [FAIL] " << name << "\n";
    }
}

// ── Helper: extract uint32 big-endian ────────────────────────────────────────
static uint8_t be_byte(uint32_t v, int i) {
    return static_cast<uint8_t>((v >> (24 - 8 * i)) & 0xFF);
}

// Default difficulty / nBits used across test helper functions
static constexpr uint32_t DEFAULT_DIFFICULTY = 0x4308519;

/**
 * Build a valid 228-byte STATELESS_GET_BLOCK payload:
 *   [0-3]   unified_height (BE)
 *   [4-7]   channel_height (BE)
 *   [8-11]  difficulty     (BE)
 *   [12-227] 216-byte Tritium block
 *
 * Tritium block layout (all big-endian except nNonce which is LE):
 *   [0-3]     nVersion
 *   [4-131]   hashPrevBlock (128 bytes)
 *   [132-195] hashMerkleRoot (64 bytes)
 *   [196-199] nChannel
 *   [200-203] nHeight
 *   [204-207] nBits
 *   [208-215] nNonce (little-endian)
 */
static network::Payload make_template_payload(
    uint32_t unified_h = 6000000,
    uint32_t channel_h = 2000000,
    uint32_t difficulty = DEFAULT_DIFFICULTY,
    uint32_t nVersion   = 8,
    uint32_t nChannel   = 2,        // Hash
    uint32_t nHeight    = 6000001,  // unified_h + 1
    uint32_t nBits      = DEFAULT_DIFFICULTY,
    uint64_t nNonce     = 0,
    uint8_t prev_fill   = 0x00,
    uint8_t merkle_base = 0xA0)
{
    network::Payload buf(228, 0x00);

    // Metadata (big-endian)
    buf[0]  = be_byte(unified_h,  0);  buf[1]  = be_byte(unified_h,  1);
    buf[2]  = be_byte(unified_h,  2);  buf[3]  = be_byte(unified_h,  3);
    buf[4]  = be_byte(channel_h,  0);  buf[5]  = be_byte(channel_h,  1);
    buf[6]  = be_byte(channel_h,  2);  buf[7]  = be_byte(channel_h,  3);
    buf[8]  = be_byte(difficulty, 0);  buf[9]  = be_byte(difficulty, 1);
    buf[10] = be_byte(difficulty, 2);  buf[11] = be_byte(difficulty, 3);

    // Block body at offset 12 (big-endian for uint32 fields)
    size_t b = 12;
    // nVersion [0-3]
    buf[b]   = be_byte(nVersion, 0); buf[b+1] = be_byte(nVersion, 1);
    buf[b+2] = be_byte(nVersion, 2); buf[b+3] = be_byte(nVersion, 3);
    b += 4;
    // hashPrevBlock [4-131]
    for (size_t i = 0; i < 128; ++i)
        buf[b + i] = prev_fill;
    b += 128;
    // hashMerkleRoot [132-195] -- non-zero so MTI validation passes
    for (size_t i = 0; i < 64; ++i)
        buf[b + i] = static_cast<uint8_t>(merkle_base + (i & 0x0F));
    b += 64;
    // nChannel [196-199]
    buf[b]   = be_byte(nChannel, 0); buf[b+1] = be_byte(nChannel, 1);
    buf[b+2] = be_byte(nChannel, 2); buf[b+3] = be_byte(nChannel, 3);
    b += 4;
    // nHeight [200-203]
    buf[b]   = be_byte(nHeight, 0); buf[b+1] = be_byte(nHeight, 1);
    buf[b+2] = be_byte(nHeight, 2); buf[b+3] = be_byte(nHeight, 3);
    b += 4;
    // nBits [204-207]
    buf[b]   = be_byte(nBits, 0); buf[b+1] = be_byte(nBits, 1);
    buf[b+2] = be_byte(nBits, 2); buf[b+3] = be_byte(nBits, 3);
    b += 4;
    // nNonce [208-215] -- little-endian
    for (int i = 0; i < 8; ++i)
        buf[b + i] = static_cast<uint8_t>((nNonce >> (i * 8)) & 0xFF);

    return buf;
}

/** Build a CBlock suitable for encode_submit() pre-checks. */
static ::LLP::CBlock make_solved_block(uint32_t nChannel = 2,
                                       uint32_t nHeight  = 6000001,
                                       uint64_t nNonce   = 0xDEADBEEFCAFEBABEULL,
                                       uint32_t nBits    = DEFAULT_DIFFICULTY) {
    ::LLP::CBlock blk;
    blk.nVersion = 8;
    blk.nChannel = nChannel;
    blk.nHeight  = nHeight;
    blk.nBits    = nBits;
    blk.nNonce   = nNonce;
    return blk;
}

static ::LLP::CBlock make_patterned_block(uint64_t nonce = 0x0123456789ABCDEFULL) {
    auto blk = make_solved_block(1, 6000001, nonce, 0x1A2B3C4D);
    blk.nVersion = 0x01020304;

    std::vector<uint8_t> prev(128);
    for (size_t i = 0; i < prev.size(); ++i)
        prev[i] = static_cast<uint8_t>(i);
    blk.hashPrevBlock.SetBytes(prev);

    std::vector<uint8_t> merkle(64);
    for (size_t i = 0; i < merkle.size(); ++i)
        merkle[i] = static_cast<uint8_t>(0x80 + i);
    blk.hashMerkleRoot.SetBytes(merkle);

    return blk;
}

static std::vector<unsigned char> get_raw_block_header_bytes(const ::LLP::CBlock& block, bool exclude_nonce) {
    // Intentional independent oracle: this mirrors the raw BEGIN/END span directly
    // so the tests verify GetBlockHeaderBytes() instead of reusing its implementation.
    const auto* begin = reinterpret_cast<const unsigned char*>(BEGIN(block.nVersion));
    const auto* end = exclude_nonce
        ? reinterpret_cast<const unsigned char*>(END(block.nBits))
        : reinterpret_cast<const unsigned char*>(END(block.nNonce));
    return std::vector<unsigned char>(begin, end);
}

/** Return a default (zero) HeightTracker::Snapshot for tests that don't need it. */
static HeightTracker::Snapshot make_snapshot() {
    return HeightTracker::Snapshot{};
}

static std::vector<uint8_t> strip_wire_header(const std::vector<uint8_t>& wire_bytes,
                                              ProtocolLane lane) {
    const size_t header_size = (lane == ProtocolLane::STATELESS) ? 6u : 5u;
    if (wire_bytes.size() <= header_size)
        return {};
    return std::vector<uint8_t>(wire_bytes.begin() + header_size, wire_bytes.end());
}

static SubmitContext make_submit_context(uint32_t template_height = 6000001,
                                         uint32_t chain_height = 6000000) {
    SubmitContext context;
    context.session_id = SessionId(0x12345678u);
    context.session_epoch = SessionEpoch(7u);
    context.template_height = template_height;
    context.chain_height = chain_height;
    return context;
}

/** Load a 228-byte template into a MiningTemplateInterface and return it. */
static std::unique_ptr<MiningTemplateInterface> make_loaded_mti(uint32_t channel = 2) {
    auto mti = std::make_unique<MiningTemplateInterface>(static_cast<uint8_t>(channel), 0);
    auto payload = make_template_payload(6000000, 2000000, DEFAULT_DIFFICULTY,
                                         8, channel, 6000001, DEFAULT_DIFFICULTY, 0);
    mti->read_stateless_payload(payload, "test");
    return mti;
}

// ── Test functions ────────────────────────────────────────────────────────────

// Test 1 -- decode_template(): valid 228-byte buffer
static void test_decode_valid() {
    auto mti     = make_loaded_mti();
    auto payload = make_template_payload();
    auto result  = StatelessBlockUtility::decode_template(*mti, payload, 2, nullptr);
    print_result("decode_template(): valid 228-byte buffer succeeds", result.valid);
}

// Test 2 -- decode_template(): reject shorter payload
static void test_decode_too_short() {
    MiningTemplateInterface mti(2, 0);
    auto payload = make_template_payload();
    payload.resize(227);  // one byte short
    auto result = StatelessBlockUtility::decode_template(mti, payload, 2, nullptr);
    print_result("decode_template(): rejects payload < 228 bytes",
                 !result.valid && !result.error_message.empty());
}

// Test 3 -- decode_template(): reject longer payload
static void test_decode_too_long() {
    MiningTemplateInterface mti(2, 0);
    auto payload = make_template_payload();
    payload.push_back(0xFF);  // one extra byte
    auto result = StatelessBlockUtility::decode_template(mti, payload, 2, nullptr);
    print_result("decode_template(): rejects payload > 228 bytes",
                 !result.valid && !result.error_message.empty());
}

// Test 4 -- encode_submit(): zero nonce rejected
static void test_encode_zero_nonce() {
    auto mti  = make_loaded_mti();
    auto blk  = make_solved_block(2, 6000001, /*nNonce=*/0);
    auto snap = make_snapshot();
    auto result = StatelessBlockUtility::encode_submit(
        *mti, blk, {}, nullptr, ProtocolLane::STATELESS, snap, nullptr);
    print_result("encode_submit(): rejects zero nonce",
                 !result.valid && !result.rejection_reason.empty());
}

// Test 5 -- encode_submit(): invalid channel rejected
static void test_encode_invalid_channel() {
    auto mti  = make_loaded_mti();
    auto blk  = make_solved_block(/*channel=*/0, 6000001, 0xDEADBEEFULL);
    auto snap = make_snapshot();
    auto result = StatelessBlockUtility::encode_submit(
        *mti, blk, {}, nullptr, ProtocolLane::STATELESS, snap, nullptr);
    print_result("encode_submit(): rejects invalid channel (0)",
                 !result.valid && !result.rejection_reason.empty());
}

// Test 6 -- encode_submit(): zero height rejected
static void test_encode_zero_height() {
    auto mti  = make_loaded_mti();
    auto blk  = make_solved_block(2, /*height=*/0, 0xDEADBEEFULL);
    auto snap = make_snapshot();
    auto result = StatelessBlockUtility::encode_submit(
        *mti, blk, {}, nullptr, ProtocolLane::STATELESS, snap, nullptr);
    print_result("encode_submit(): rejects zero height",
                 !result.valid && !result.rejection_reason.empty());
}

// Test 7 -- encode_submit(): STATELESS lane -> opcode 0xD001
static void test_encode_stateless_opcode() {
    auto mti  = make_loaded_mti();
    auto blk  = make_solved_block();
    auto snap = make_snapshot();
    auto result = StatelessBlockUtility::encode_submit(
        *mti, blk, {}, nullptr, ProtocolLane::STATELESS, snap, nullptr);

    bool ok = false;
    if (result.valid && result.wire_bytes && result.wire_bytes->size() >= 2) {
        const auto& w = *result.wire_bytes;
        ok = (w[0] == 0xD0 && w[1] == 0x01);
    }
    print_result("encode_submit(): STATELESS lane produces opcode 0xD001", ok);
}

// Test 8 -- encode_submit(): LEGACY lane -> opcode 0x01
static void test_encode_legacy_opcode() {
    auto mti  = make_loaded_mti();
    auto blk  = make_solved_block();
    auto snap = make_snapshot();
    auto result = StatelessBlockUtility::encode_submit(
        *mti, blk, {}, nullptr, ProtocolLane::LEGACY, snap, nullptr);

    bool ok = false;
    if (result.valid && result.wire_bytes && result.wire_bytes->size() >= 5) {
        const auto& w = *result.wire_bytes;
        ok = (w[0] == 0x01) &&
             (w[1] == 0x00) &&
             (w[2] == 0x00) &&
             (w[3] == 0x00) &&
             (w[4] == StatelessBlockUtility::BLOCK_BODY_SIZE);
    }
    print_result("encode_submit(): LEGACY lane produces opcode+length header [0x01][0x000000D8]", ok);
}

// Test 9 -- encode_submit(): falcon=nullptr -> no signature suffix
// For Hash channel with no vOffsets, payload = 216-byte block only.
// Stateless header: 2-byte opcode + 4-byte length = 6 bytes overhead.
static void test_encode_unsigned_submit() {
    auto mti  = make_loaded_mti(2);  // Hash channel, no vOffsets
    auto blk  = make_solved_block();
    auto snap = make_snapshot();

    auto result = StatelessBlockUtility::encode_submit(
        *mti, blk, {}, nullptr, ProtocolLane::STATELESS, snap, nullptr);

    bool ok = false;
    if (result.valid && result.wire_bytes) {
        // Stateless header: 2-byte opcode + 4-byte length = 6 bytes
        // Payload for Hash channel (no vOffsets): BLOCK_BODY_SIZE (216) bytes
        size_t expected_total = 6 + StatelessBlockUtility::BLOCK_BODY_SIZE;
        ok = (result.wire_bytes->size() == expected_total);
    }
    print_result("encode_submit(): falcon=nullptr produces unsigned submit "
                 "(no signature suffix)", ok);
}

// Test 10 -- decode_template(): metadata fields from 12-byte prefix
static void test_decode_metadata_fields() {
    MiningTemplateInterface mti(2, 0);
    auto payload = make_template_payload(
        /*unified_h=*/7654321,
        /*channel_h=*/2500000,
        /*difficulty=*/0xABCD1234);
    auto result = StatelessBlockUtility::decode_template(mti, payload, 2, nullptr);
    bool ok = result.valid &&
              result.unified_height   == 7654321  &&
              result.channel_height   == 2500000;
    print_result("decode_template(): metadata fields parsed correctly from "
                 "12-byte prefix (BE)", ok);
}

// Test 11 -- decode_template(): canonical block fields from 216-byte body
static void test_decode_block_fields() {
    MiningTemplateInterface mti(2, 0);
    auto payload = make_template_payload(
        /*unified_h=*/6000000, /*channel_h=*/2000000, /*difficulty=*/DEFAULT_DIFFICULTY,
        /*nVersion=*/8, /*nChannel=*/2, /*nHeight=*/6000001,
        /*nBits=*/DEFAULT_DIFFICULTY, /*nNonce=*/0);
    auto result = StatelessBlockUtility::decode_template(mti, payload, 2, nullptr);
    bool ok = result.valid &&
              result.block.nVersion == 8    &&
              result.block.nChannel == 2    &&
              result.block.nHeight  == 6000001 &&
              result.block.nBits    == DEFAULT_DIFFICULTY;
    print_result("decode_template(): canonical block fields from 216-byte body",
                 ok);
}

// Test 12 -- decode_template(): channel_consistent flag
static void test_decode_channel_consistent() {
    // mining channel 1 (Prime) but block says 2 (Hash) -> inconsistent
    MiningTemplateInterface mti(1, 0); // configured for Prime
    auto payload = make_template_payload(6000000, 2000000, 0, 8, 2, 6000001);
    auto result = StatelessBlockUtility::decode_template(mti, payload, /*mining_channel=*/1, nullptr);
    bool ok = result.valid && !result.channel_consistent;
    print_result("decode_template(): channel_consistent=false when nChannel "
                 "doesn't match mining_channel", ok);
}

// Test 13 -- encode_submit(): stale/tip-moved are warnings, not blocks
static void test_encode_stale_does_not_block() {
    auto mti  = make_loaded_mti();
    auto blk  = make_solved_block();

    // Simulate a stale snapshot: channel_height >= channel_target
    HeightTracker ht;
    ht.OnBlockDataReceived(6000000, 2000001, DEFAULT_DIFFICULTY, uint1024_t{});
    ht.OnTemplateReceived(2, 2000002);
    ht.OnBlockDataReceived(6000001, 2000002, DEFAULT_DIFFICULTY, uint1024_t{});
    auto snap = ht.GetSnapshot();

    auto result = StatelessBlockUtility::encode_submit(
        *mti, blk, {}, nullptr, ProtocolLane::STATELESS, snap, nullptr);
    print_result("encode_submit(): stale template is a warning, not a hard "
                 "rejection", result.valid);
}

// Test 14 -- Prime channel vOffsets produce larger payload than Hash channel
static void test_encode_rejects_submit_height_mismatch() {
    auto mti = make_loaded_mti();
    auto blk = make_solved_block(2, /*nHeight=*/6000002, 0xDEADBEEFCAFEBABEULL);
    auto result = StatelessBlockUtility::encode_submit(
        *mti, blk, {}, nullptr, ProtocolLane::STATELESS, make_snapshot(), nullptr,
        make_submit_context(/*template_height=*/6000001, /*chain_height=*/6000000));
    print_result("encode_submit(): rejects authoritative submit-height mismatch",
                 !result.valid &&
                 result.rejection_reason.find("template_height=6000001") != std::string::npos &&
                 result.rejection_reason.find("submit_height=6000002") != std::string::npos);
}

// Test 15 -- Prime channel vOffsets produce larger payload than Hash channel
static void test_encode_prime_voffsets_appended() {
    // Load a Prime-channel template (nChannel=1)
    MiningTemplateInterface mti_prime(1, 0);
    auto payload_prime = make_template_payload(6000000, 2000000, DEFAULT_DIFFICULTY,
                                               8, 1, 6000001, DEFAULT_DIFFICULTY, 0);
    mti_prime.read_stateless_payload(payload_prime, "test");

    // Load a Hash-channel template (nChannel=2)
    MiningTemplateInterface mti_hash(2, 0);
    auto payload_hash = make_template_payload(6000000, 2000000, DEFAULT_DIFFICULTY,
                                              8, 2, 6000001, DEFAULT_DIFFICULTY, 0);
    mti_hash.read_stateless_payload(payload_hash, "test");

    auto blk_prime = make_solved_block(1, 6000001, 0xDEADBEEFCAFEBABEULL);
    auto blk_hash  = make_solved_block(2, 6000001, 0xDEADBEEFCAFEBABEULL);
    auto snap = make_snapshot();

    // Canonical Prime vOffsets shape: 6 single-byte offsets + 4-byte LE fraction.
    std::vector<uint8_t> vOffsets = {0x02, 0x04, 0x06, 0x02, 0x04, 0x06, 0x10, 0x20, 0x30, 0x40};

    auto prime_result = StatelessBlockUtility::encode_submit(
        mti_prime, blk_prime, vOffsets, nullptr, ProtocolLane::STATELESS, snap, nullptr);
    auto hash_result  = StatelessBlockUtility::encode_submit(
        mti_hash, blk_hash, {}, nullptr, ProtocolLane::STATELESS, snap, nullptr);

    bool ok = prime_result.valid && hash_result.valid &&
              prime_result.wire_bytes->size() ==
                  hash_result.wire_bytes->size() + vOffsets.size();
    print_result("encode_submit(): Prime vOffsets are appended to payload "
                 "(Prime payload > Hash payload by vOffsets.size())", ok);
}

// Test 16 -- encode_submit(): preserve solved snapshot fields under template advance
static void test_encode_preserves_solved_snapshot_under_template_advance() {
    MiningTemplateInterface mti(2, 0);

    auto tmpl_n = make_template_payload(6000000, 2000000, DEFAULT_DIFFICULTY,
                                        8, 2, 6000001, DEFAULT_DIFFICULTY, 0,
                                        /*prev_fill=*/0x11, /*merkle_base=*/0x70);
    mti.read_stateless_payload(tmpl_n, "test");

    network::Payload solved_payload(tmpl_n.begin() + StatelessBlockUtility::METADATA_PREFIX_SIZE,
                                    tmpl_n.begin() + StatelessBlockUtility::TEMPLATE_PAYLOAD_SIZE);
    auto solved = llp_utils::deserialize_block_header(solved_payload);
    solved.nNonce = 0x0123456789ABCDEFULL;

    auto tmpl_n1 = make_template_payload(6000001, 2000001, DEFAULT_DIFFICULTY,
                                         8, 2, 6000002, DEFAULT_DIFFICULTY, 0,
                                         /*prev_fill=*/0x22, /*merkle_base=*/0x90);
    mti.read_stateless_payload(tmpl_n1, "test");

    auto result = StatelessBlockUtility::encode_submit(
        mti, solved, {}, nullptr, ProtocolLane::STATELESS, make_snapshot(), nullptr,
        make_submit_context(/*template_height=*/6000001, /*chain_height=*/6000000));

    bool ok = false;
    if (result.valid && result.wire_bytes) {
        auto plaintext = strip_wire_header(*result.wire_bytes, ProtocolLane::STATELESS);
        if (plaintext.size() == StatelessBlockUtility::BLOCK_BODY_SIZE) {
            network::Payload encoded_payload(plaintext.begin(), plaintext.end());
            auto encoded = llp_utils::deserialize_block_header(encoded_payload);
            ok = encoded.nHeight == solved.nHeight &&
                 encoded.hashPrevBlock == solved.hashPrevBlock &&
                 encoded.hashMerkleRoot == solved.hashMerkleRoot &&
                 encoded.nNonce == solved.nNonce;
        }
    }

    print_result("encode_submit(): solved snapshot survives template advance", ok);
}

// Test 15 -- read_stateless_payload() + set_channel_height() does not trigger
// the corruption guard.  The guard fires when block.nHeight != m_last_unified_height.
// read_stateless_payload() must NOT override m_last_unified_height with the
// metadata's nUnifiedHeightMeta (which is one less than block.nHeight).
static void test_set_channel_height_no_corruption_guard() {
    MiningTemplateInterface mti(2, 0);
    // unified_height = 6000000 (metadata/chain tip)
    // block.nHeight  = 6000001 (NEXT block = unified_height + 1)
    // make_template_payload(unified_height, channel_height, difficulty,
    //                        nVersion, nChannel, nHeight,  nBits,             nNonce)
    auto payload = make_template_payload(6000000, 2000000, DEFAULT_DIFFICULTY,
                                         8,       2,       6000001, DEFAULT_DIFFICULTY, 0);
    auto result = mti.read_stateless_payload(payload, "test");
    bool template_valid = result.is_valid;

    // Now call set_channel_height() — should NOT discard the template via the
    // block.nHeight != m_last_unified_height corruption guard.
    mti.set_channel_height(2000001);  // channel_target = channel_height + 1

    // Template must still be valid after set_channel_height().
    bool still_valid = mti.has_valid_template();

    print_result("read_stateless_payload() + set_channel_height(): corruption guard "
                 "does NOT falsely fire (m_last_unified_height is NOT overridden)",
                 template_valid && still_valid);
}

// Test 16 -- worker header bytes with nonce match raw CBlock memory hashed by node
static void test_worker_header_bytes_match_raw_block_with_nonce() {
    auto blk = make_patterned_block();
    const auto expected = get_raw_block_header_bytes(blk, false);
    const auto actual = nexusminer::GetBlockHeaderBytes(blk, false);
    print_result("GetBlockHeaderBytes(false) matches raw CBlock nVersion..nNonce bytes",
                 actual == expected && actual.size() == 216);
}

// Test 17 -- worker prime header bytes exclude nonce and match raw ProofHash span
static void test_worker_prime_header_bytes_match_raw_block_without_nonce() {
    auto blk = make_patterned_block();
    const auto expected = get_raw_block_header_bytes(blk, true);
    const auto actual = nexusminer::GetBlockHeaderBytes(blk, true);
    print_result("GetBlockHeaderBytes(true) matches raw CBlock nVersion..nBits bytes",
                 actual == expected && actual.size() == 208);
}

// Test 18 -- prime base hash uses upstream LLC::SK1024 on raw ProofHash span
static void test_worker_prime_base_hash_matches_llc_sk1024() {
    auto blk = make_patterned_block();
    // Intentional: the vector overload is the independent oracle here. It should
    // hash the same raw nVersion..nBits bytes as the BEGIN/END pointer span used
    // by GetPrimeProofHash(), without calling the production helper itself.
    const auto expected = LLC::SK1024(get_raw_block_header_bytes(blk, true));
    const auto actual = nexusminer::GetPrimeProofHash(blk);
    print_result("GetPrimeProofHash() matches LLC::SK1024(raw nVersion..nBits bytes)",
                 actual == expected);
}

// Test 19 -- prime base hash is nonce-independent just like node ProofHash
static void test_worker_prime_base_hash_ignores_nonce() {
    auto blk_a = make_patterned_block(0x0123456789ABCDEFULL);
    auto blk_b = blk_a;
    blk_b.nNonce = 0xFEDCBA9876543210ULL;

    print_result("GetPrimeProofHash() ignores nonce changes",
                 nexusminer::GetPrimeProofHash(blk_a) == nexusminer::GetPrimeProofHash(blk_b));
}

// Test 20 -- full worker header bytes remain nonce-sensitive for hash-channel mining
static void test_worker_hash_header_bytes_include_nonce() {
    auto blk_a = make_patterned_block(0x0123456789ABCDEFULL);
    auto blk_b = blk_a;
    blk_b.nNonce = 0xFEDCBA9876543210ULL;

    print_result("GetBlockHeaderBytes(false) changes when nonce changes",
                 nexusminer::GetBlockHeaderBytes(blk_a, false) != nexusminer::GetBlockHeaderBytes(blk_b, false) &&
                 nexusminer::GetBlockHeaderBytes(blk_a, true) == nexusminer::GetBlockHeaderBytes(blk_b, true));
}



int main() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "  stateless_block_utility_test\n";
    std::cout << "========================================\n";

    test_decode_valid();
    test_decode_too_short();
    test_decode_too_long();
    test_encode_zero_nonce();
    test_encode_invalid_channel();
    test_encode_zero_height();
    test_encode_stateless_opcode();
    test_encode_legacy_opcode();
    test_encode_unsigned_submit();
    test_decode_metadata_fields();
    test_decode_block_fields();
    test_decode_channel_consistent();
    test_encode_stale_does_not_block();
    test_encode_rejects_submit_height_mismatch();
    test_encode_prime_voffsets_appended();
    test_encode_preserves_solved_snapshot_under_template_advance();
    test_set_channel_height_no_corruption_guard();
    test_worker_header_bytes_match_raw_block_with_nonce();
    test_worker_prime_header_bytes_match_raw_block_without_nonce();
    test_worker_prime_base_hash_matches_llc_sk1024();
    test_worker_prime_base_hash_ignores_nonce();
    test_worker_hash_header_bytes_include_nonce();

    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "Test Summary\n";
    std::cout << "========================================\n";
    std::cout << "Tests run:    " << tests_run    << "\n";
    std::cout << "Tests passed: " << tests_passed << "\n";
    std::cout << "Tests failed: " << tests_failed << "\n";
    std::cout << "Success rate: "
              << (tests_run > 0 ? 100 * tests_passed / tests_run : 0)
              << "%\n";
    std::cout << "========================================\n";

    return (tests_failed == 0) ? 0 : 1;
}
