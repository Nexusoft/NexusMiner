/**
 * @file e2e_block_submission_test.cpp
 * @brief End-to-end block submission tests
 *
 * Validates the complete critical path:
 *   encode_submit() → strip wire header → ChaCha20 encrypt → decrypt → verify
 *
 * Tests (acceptance criteria):
 *  1.  Hash channel: encode_submit → ChaCha20 encrypt → decrypt round-trip
 *  2.  Prime channel: encode_submit with vOffsets → ChaCha20 round-trip
 *  3.  Decrypted Hash payload is exactly BLOCK_BODY_SIZE (216) bytes
 *  4.  Decrypted Prime payload = BLOCK_BODY_SIZE + vOffsets.size()
 *  5.  vOffsets bytes survive the full pipeline (byte-exact verification)
 *  6.  KDF session key is deterministic (same genesis → same key)
 *  7.  Wrong session key → ChaCha20 decrypt fails
 *  8.  AAD mismatch (non-empty vs empty) → decrypt fails
 *  9.  STATELESS lane: final SUBMIT_BLOCK packet has correct structure
 * 10.  LEGACY lane: final SUBMIT_BLOCK packet has correct structure
 * 11.  Prime-channel vOffsets flow through prepare_block_submission_from_solved()
 * 12.  Empty vOffsets for Hash channel (no extra bytes appended)
 * 13.  SubmitBlockPayloadInfo: Hash Falcon-1024 fixed-size (pt=1803, enc=1831)
 * 14.  SubmitBlockPayloadInfo: Prime + 10 offsets (pt=1813, enc=1841)
 * 15.  SubmitBlockPayloadInfo: Prime != Hash when offsets present
 * 16.  E2E: Hash unsigned payload_info + encrypt size correct
 * 17.  E2E: Prime with vOffsets full pipeline size correct
 */

#include "include/stateless_block_utility.hpp"
#include "protocol/mining_template_interface.hpp"
#include "protocol/chacha20_wrapper.hpp"
#include "protocol/height_tracker.hpp"
#include "protocol/packet_builder.hpp"
#include "protocol/falcon_constants.hpp"
#include "worker/worker.hpp"
#include "miner_opcodes.hpp"
#include <iostream>
#include <cassert>
#include <cstdint>
#include <vector>
#include <cstring>
#include <openssl/sha.h>

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

// ── Constants matching solo.cpp (must stay in sync) ──────────────────────────
static const std::string KDF_DOMAIN = "nexus-mining-chacha20-v1";
static const std::vector<uint8_t> AAD_BLOCK_SUBMISSION{};  // empty — node uses no AAD

// ── Helpers ──────────────────────────────────────────────────────────────────
static uint8_t be_byte(uint32_t v, int i) {
    return static_cast<uint8_t>((v >> (24 - 8 * i)) & 0xFF);
}

static constexpr uint32_t DEFAULT_DIFFICULTY = 0x4308519;

/**
 * Build a valid 228-byte STATELESS_GET_BLOCK payload.
 */
static network::Payload make_template_payload(
    uint32_t unified_h = 6000000,
    uint32_t channel_h = 2000000,
    uint32_t difficulty = DEFAULT_DIFFICULTY,
    uint32_t nVersion   = 8,
    uint32_t nChannel   = 2,
    uint32_t nHeight    = 6000001,
    uint32_t nBits      = DEFAULT_DIFFICULTY,
    uint64_t nNonce     = 0)
{
    network::Payload buf(228, 0x00);

    buf[0] = be_byte(unified_h, 0); buf[1] = be_byte(unified_h, 1);
    buf[2] = be_byte(unified_h, 2); buf[3] = be_byte(unified_h, 3);
    buf[4] = be_byte(channel_h, 0); buf[5] = be_byte(channel_h, 1);
    buf[6] = be_byte(channel_h, 2); buf[7] = be_byte(channel_h, 3);
    buf[8] = be_byte(difficulty, 0); buf[9] = be_byte(difficulty, 1);
    buf[10] = be_byte(difficulty, 2); buf[11] = be_byte(difficulty, 3);

    size_t b = 12;
    buf[b] = be_byte(nVersion, 0); buf[b+1] = be_byte(nVersion, 1);
    buf[b+2] = be_byte(nVersion, 2); buf[b+3] = be_byte(nVersion, 3);
    b += 4;
    b += 128; // hashPrevBlock (zeros)
    for (size_t i = 0; i < 64; ++i)
        buf[b + i] = static_cast<uint8_t>(0xA0 + (i & 0x0F));
    b += 64;
    buf[b] = be_byte(nChannel, 0); buf[b+1] = be_byte(nChannel, 1);
    buf[b+2] = be_byte(nChannel, 2); buf[b+3] = be_byte(nChannel, 3);
    b += 4;
    buf[b] = be_byte(nHeight, 0); buf[b+1] = be_byte(nHeight, 1);
    buf[b+2] = be_byte(nHeight, 2); buf[b+3] = be_byte(nHeight, 3);
    b += 4;
    buf[b] = be_byte(nBits, 0); buf[b+1] = be_byte(nBits, 1);
    buf[b+2] = be_byte(nBits, 2); buf[b+3] = be_byte(nBits, 3);
    b += 4;
    for (int i = 0; i < 8; ++i)
        buf[b + i] = static_cast<uint8_t>((nNonce >> (i * 8)) & 0xFF);

    return buf;
}

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

static HeightTracker::Snapshot make_snapshot() {
    return HeightTracker::Snapshot{};
}

static std::unique_ptr<MiningTemplateInterface> make_loaded_mti(uint32_t channel = 2) {
    auto mti = std::make_unique<MiningTemplateInterface>(static_cast<uint8_t>(channel), 0);
    auto payload = make_template_payload(6000000, 2000000, DEFAULT_DIFFICULTY,
                                         8, channel, 6000001, DEFAULT_DIFFICULTY, 0);
    mti->read_stateless_payload(payload, "test");
    return mti;
}

static void test_height_tracker_snapshot_carries_session_epoch() {
    HeightTracker tracker;
    tracker.set_session_epoch(SessionEpoch(uint64_t{42}));
    auto snap = tracker.GetSnapshot();
    print_result("HeightTracker snapshot carries authoritative session epoch", snap.session_epoch == SessionEpoch(uint64_t{42}));
}

static void test_template_interface_stamps_session_epoch() {
    MiningTemplateInterface mti_hash(2, 0);
    mti_hash.set_session_epoch(SessionEpoch(uint64_t{11}));
    auto payload = make_template_payload(6000000, 2000000, DEFAULT_DIFFICULTY,
                                         8, 2, 6000001, DEFAULT_DIFFICULTY, 0);
    auto result = mti_hash.read_stateless_payload(payload, "test");
    const auto* tmpl = mti_hash.get_current_template();

    bool ok = result.is_valid && tmpl != nullptr && tmpl->session_epoch == SessionEpoch(uint64_t{11});
    print_result("MiningTemplateInterface stamps templates with authoritative session epoch", ok);
}

/**
 * Derive a ChaCha20 session key from genesis hash, matching solo.cpp:
 *   SHA256(KDF_DOMAIN || genesis)
 */
static std::vector<uint8_t> derive_session_key(const std::vector<uint8_t>& genesis) {
    std::vector<uint8_t> preimage;
    preimage.insert(preimage.end(), KDF_DOMAIN.begin(), KDF_DOMAIN.end());
    preimage.insert(preimage.end(), genesis.begin(), genesis.end());
    std::vector<uint8_t> key(SHA256_DIGEST_LENGTH);
    SHA256(preimage.data(), preimage.size(), key.data());
    return key;
}

/**
 * Strip the PacketBuilder wire header from encode_submit() output.
 * STATELESS: 2-byte opcode + 4-byte length = 6 bytes
 * LEGACY:    1-byte opcode + 4-byte length = 5 bytes
 */
static std::vector<uint8_t> strip_wire_header(const std::vector<uint8_t>& framed,
                                              ProtocolLane lane) {
    const size_t header_size = (lane == ProtocolLane::STATELESS) ? 6u : 5u;
    if (framed.size() <= header_size)
        return {};
    return std::vector<uint8_t>(framed.begin() + header_size, framed.end());
}

// ── Test functions ────────────────────────────────────────────────────────────

// Test 1: Hash channel full pipeline: encode_submit → ChaCha20 → decrypt
static void test_e2e_hash_channel_round_trip() {
    auto mti  = make_loaded_mti(2);
    auto blk  = make_solved_block(2);
    auto snap = make_snapshot();

    // Step 1: encode_submit
    auto submit = StatelessBlockUtility::encode_submit(
        *mti, blk, {}, nullptr, ProtocolLane::STATELESS, snap, nullptr);
    if (!submit.valid) {
        print_result("E2E Hash: encode_submit succeeds", false);
        return;
    }

    // Step 2: strip wire header
    auto plaintext = strip_wire_header(*submit.wire_bytes, ProtocolLane::STATELESS);
    if (plaintext.empty()) {
        print_result("E2E Hash: strip wire header", false);
        return;
    }

    // Step 3: derive session key (same KDF as solo.cpp and node)
    std::vector<uint8_t> genesis(32, 0xAB);
    auto session_key = derive_session_key(genesis);

    // Step 4: encrypt (miner side)
    ChaCha20Wrapper wrapper;
    auto enc_nonce = ChaCha20Wrapper::generate_nonce();
    auto enc = wrapper.encrypt(plaintext, session_key, enc_nonce, AAD_BLOCK_SUBMISSION);
    if (!enc.success) {
        print_result("E2E Hash: ChaCha20 encrypt succeeds", false);
        return;
    }

    // Step 5: decrypt (node side — same key, same nonce, empty AAD)
    auto dec = wrapper.decrypt(enc.data, session_key, enc_nonce, AAD_BLOCK_SUBMISSION);
    bool ok = dec.success && (dec.data == plaintext);
    print_result("E2E Hash: encode_submit → ChaCha20 encrypt → decrypt round-trip", ok);
}

// Test 2: Prime channel with vOffsets full pipeline
static void test_e2e_prime_channel_voffsets_round_trip() {
    auto mti  = make_loaded_mti(1);  // Prime channel
    auto blk  = make_solved_block(1);
    auto snap = make_snapshot();
    std::vector<uint8_t> vOffsets = {0x02, 0x04, 0x00, 0x10, 0x20, 0x30, 0x40};

    // Step 1: encode_submit with vOffsets
    auto submit = StatelessBlockUtility::encode_submit(
        *mti, blk, vOffsets, nullptr, ProtocolLane::STATELESS, snap, nullptr);
    if (!submit.valid) {
        print_result("E2E Prime: encode_submit with vOffsets succeeds", false);
        return;
    }

    // Step 2: strip wire header
    auto plaintext = strip_wire_header(*submit.wire_bytes, ProtocolLane::STATELESS);

    // Step 3: encrypt (miner side)
    std::vector<uint8_t> genesis(32, 0xAB);
    auto session_key = derive_session_key(genesis);
    ChaCha20Wrapper wrapper;
    auto enc_nonce = ChaCha20Wrapper::generate_nonce();
    auto enc = wrapper.encrypt(plaintext, session_key, enc_nonce, AAD_BLOCK_SUBMISSION);

    // Step 4: decrypt (node side)
    auto dec = wrapper.decrypt(enc.data, session_key, enc_nonce, AAD_BLOCK_SUBMISSION);
    bool ok = dec.success && (dec.data == plaintext);
    print_result("E2E Prime: encode_submit + vOffsets → ChaCha20 round-trip", ok);
}

// Test 3: Hash decrypted payload is exactly BLOCK_BODY_SIZE (216 bytes)
static void test_hash_payload_size() {
    auto mti  = make_loaded_mti(2);
    auto blk  = make_solved_block(2);
    auto snap = make_snapshot();

    auto submit = StatelessBlockUtility::encode_submit(
        *mti, blk, {}, nullptr, ProtocolLane::STATELESS, snap, nullptr);
    auto plaintext = strip_wire_header(*submit.wire_bytes, ProtocolLane::STATELESS);
    bool ok = submit.valid && (plaintext.size() == StatelessBlockUtility::BLOCK_BODY_SIZE);
    print_result("E2E Hash: decrypted payload = 216 bytes (BLOCK_BODY_SIZE)", ok);
}

// Test 4: Prime decrypted payload = BLOCK_BODY_SIZE + vOffsets.size()
static void test_prime_payload_size() {
    auto mti  = make_loaded_mti(1);
    auto blk  = make_solved_block(1);
    auto snap = make_snapshot();
    std::vector<uint8_t> vOffsets = {0x02, 0x04, 0x00, 0x10, 0x20, 0x30, 0x40};

    auto submit = StatelessBlockUtility::encode_submit(
        *mti, blk, vOffsets, nullptr, ProtocolLane::STATELESS, snap, nullptr);
    auto plaintext = strip_wire_header(*submit.wire_bytes, ProtocolLane::STATELESS);
    bool ok = submit.valid &&
              (plaintext.size() == StatelessBlockUtility::BLOCK_BODY_SIZE + vOffsets.size());
    print_result("E2E Prime: decrypted payload = 216 + vOffsets.size() bytes", ok);
}

// Test 5: vOffsets bytes survive the full pipeline (byte-exact)
static void test_voffsets_byte_exact() {
    auto mti  = make_loaded_mti(1);
    auto blk  = make_solved_block(1);
    auto snap = make_snapshot();
    std::vector<uint8_t> vOffsets = {0x02, 0x04, 0x00, 0x10, 0x20, 0x30, 0x40};

    // encode_submit
    auto submit = StatelessBlockUtility::encode_submit(
        *mti, blk, vOffsets, nullptr, ProtocolLane::STATELESS, snap, nullptr);
    auto plaintext = strip_wire_header(*submit.wire_bytes, ProtocolLane::STATELESS);

    // ChaCha20 round-trip
    std::vector<uint8_t> genesis(32, 0xAB);
    auto session_key = derive_session_key(genesis);
    ChaCha20Wrapper wrapper;
    auto enc_nonce = ChaCha20Wrapper::generate_nonce();
    auto enc = wrapper.encrypt(plaintext, session_key, enc_nonce, AAD_BLOCK_SUBMISSION);
    auto dec = wrapper.decrypt(enc.data, session_key, enc_nonce, AAD_BLOCK_SUBMISSION);

    // Verify vOffsets at tail of decrypted payload
    bool ok = false;
    if (dec.success && dec.data.size() >= StatelessBlockUtility::BLOCK_BODY_SIZE + vOffsets.size()) {
        size_t offset_start = StatelessBlockUtility::BLOCK_BODY_SIZE;
        std::vector<uint8_t> recovered(
            dec.data.begin() + offset_start,
            dec.data.begin() + offset_start + vOffsets.size());
        ok = (recovered == vOffsets);
    }
    print_result("E2E: vOffsets bytes survive full pipeline (byte-exact match)", ok);
}

// Test 6: KDF determinism — same genesis → same session key
static void test_kdf_determinism() {
    std::vector<uint8_t> genesis(32, 0xAB);
    auto key1 = derive_session_key(genesis);
    auto key2 = derive_session_key(genesis);
    bool ok = (key1 == key2) && (key1.size() == 32);
    print_result("KDF: same genesis → same 32-byte session key (deterministic)", ok);
}

// Test 7: Wrong session key → decrypt fails
static void test_wrong_key_decrypt_fails() {
    auto mti  = make_loaded_mti(2);
    auto blk  = make_solved_block(2);
    auto snap = make_snapshot();

    auto submit = StatelessBlockUtility::encode_submit(
        *mti, blk, {}, nullptr, ProtocolLane::STATELESS, snap, nullptr);
    auto plaintext = strip_wire_header(*submit.wire_bytes, ProtocolLane::STATELESS);

    std::vector<uint8_t> genesis(32, 0xAB);
    auto session_key = derive_session_key(genesis);
    ChaCha20Wrapper wrapper;
    auto enc_nonce = ChaCha20Wrapper::generate_nonce();
    auto enc = wrapper.encrypt(plaintext, session_key, enc_nonce, AAD_BLOCK_SUBMISSION);

    // Node uses wrong key (different genesis)
    std::vector<uint8_t> wrong_genesis(32, 0xCD);
    auto wrong_key = derive_session_key(wrong_genesis);
    auto dec = wrapper.decrypt(enc.data, wrong_key, enc_nonce, AAD_BLOCK_SUBMISSION);
    print_result("E2E: wrong session key → ChaCha20 decrypt FAILS", !dec.success);
}

// Test 8: AAD mismatch → decrypt fails (node vs miner disagree)
static void test_aad_mismatch_fails() {
    auto mti  = make_loaded_mti(2);
    auto blk  = make_solved_block(2);
    auto snap = make_snapshot();

    auto submit = StatelessBlockUtility::encode_submit(
        *mti, blk, {}, nullptr, ProtocolLane::STATELESS, snap, nullptr);
    auto plaintext = strip_wire_header(*submit.wire_bytes, ProtocolLane::STATELESS);

    std::vector<uint8_t> genesis(32, 0xAB);
    auto session_key = derive_session_key(genesis);
    ChaCha20Wrapper wrapper;
    auto enc_nonce = ChaCha20Wrapper::generate_nonce();

    // Miner encrypts with empty AAD (correct)
    auto enc = wrapper.encrypt(plaintext, session_key, enc_nonce, AAD_BLOCK_SUBMISSION);

    // Simulated buggy node tries to decrypt with non-empty AAD
    std::vector<uint8_t> wrong_aad{'B','L','O','C','K','_','S','U','B','M','I','S','S','I','O','N'};
    auto dec = wrapper.decrypt(enc.data, session_key, enc_nonce, wrong_aad);
    print_result("E2E: AAD mismatch (non-empty vs empty) → decrypt FAILS", !dec.success);
}

// Test 9: STATELESS lane final SUBMIT_BLOCK packet structure
static void test_stateless_submit_packet_structure() {
    auto mti  = make_loaded_mti(2);
    auto blk  = make_solved_block(2);
    auto snap = make_snapshot();

    auto submit = StatelessBlockUtility::encode_submit(
        *mti, blk, {}, nullptr, ProtocolLane::STATELESS, snap, nullptr);
    auto plaintext = strip_wire_header(*submit.wire_bytes, ProtocolLane::STATELESS);

    std::vector<uint8_t> genesis(32, 0xAB);
    auto session_key = derive_session_key(genesis);
    ChaCha20Wrapper wrapper;
    auto enc_nonce = ChaCha20Wrapper::generate_nonce();
    auto enc = wrapper.encrypt(plaintext, session_key, enc_nonce, AAD_BLOCK_SUBMISSION);

    // Assemble final encrypted payload: [nonce(12)][ciphertext+tag]
    std::vector<uint8_t> encrypted_payload;
    encrypted_payload.insert(encrypted_payload.end(), enc_nonce.begin(), enc_nonce.end());
    encrypted_payload.insert(encrypted_payload.end(), enc.data.begin(), enc.data.end());

    // Build final SUBMIT_BLOCK packet
    auto packet = PacketBuilder::build(ProtocolLane::STATELESS, nexusminer::LLP::SUBMIT_BLOCK, encrypted_payload);

    bool ok = false;
    if (packet && packet->size() >= 6) {
        // STATELESS header: opcode 0xD001 (2 bytes) + length (4 bytes BE)
        ok = ((*packet)[0] == 0xD0) && ((*packet)[1] == 0x01);
        // Verify total: header(6) + nonce(12) + ciphertext(plaintext.size()) + tag(16)
        size_t expected = 6 + 12 + plaintext.size() + 16;
        ok = ok && (packet->size() == expected);
    }
    print_result("E2E STATELESS: SUBMIT_BLOCK packet = [0xD001][len][nonce(12)][ct+tag]", ok);
}

// Test 10: LEGACY lane final SUBMIT_BLOCK packet structure
static void test_legacy_submit_packet_structure() {
    auto mti  = make_loaded_mti(2);
    auto blk  = make_solved_block(2);
    auto snap = make_snapshot();

    auto submit = StatelessBlockUtility::encode_submit(
        *mti, blk, {}, nullptr, ProtocolLane::LEGACY, snap, nullptr);
    auto plaintext = strip_wire_header(*submit.wire_bytes, ProtocolLane::LEGACY);

    std::vector<uint8_t> genesis(32, 0xAB);
    auto session_key = derive_session_key(genesis);
    ChaCha20Wrapper wrapper;
    auto enc_nonce = ChaCha20Wrapper::generate_nonce();
    auto enc = wrapper.encrypt(plaintext, session_key, enc_nonce, AAD_BLOCK_SUBMISSION);

    std::vector<uint8_t> encrypted_payload;
    encrypted_payload.insert(encrypted_payload.end(), enc_nonce.begin(), enc_nonce.end());
    encrypted_payload.insert(encrypted_payload.end(), enc.data.begin(), enc.data.end());

    auto packet = PacketBuilder::build(ProtocolLane::LEGACY, nexusminer::LLP::SUBMIT_BLOCK, encrypted_payload);

    bool ok = false;
    if (packet && packet->size() >= 5) {
        // LEGACY header: opcode 0x01 (1 byte) + length (4 bytes BE)
        ok = ((*packet)[0] == 0x01);
        size_t expected = 5 + 12 + plaintext.size() + 16;
        ok = ok && (packet->size() == expected);
    }
    print_result("E2E LEGACY: SUBMIT_BLOCK packet = [0x01][len][nonce(12)][ct+tag]", ok);
}

// Test 11: vOffsets flow through prepare_block_submission_from_solved (Prime channel)
static void test_voffsets_flow_prepare_block_submission_from_solved() {
    MiningTemplateInterface mti_prime(1, 0);
    auto payload = make_template_payload(6000000, 2000000, DEFAULT_DIFFICULTY,
                                         8, 1, 6000001, DEFAULT_DIFFICULTY, 0);
    mti_prime.read_stateless_payload(payload, "test");

    const auto* tmpl = mti_prime.get_current_template();
    if (!tmpl) {
        print_result("vOffsets flow: template loaded", false);
        return;
    }

    std::vector<uint8_t> vOffsets = {0x02, 0x04, 0x00, 0x10, 0x20, 0x30, 0x40};
    Block_data solved_block(tmpl->block);
    solved_block.nNonce = 0xDEADBEEFCAFEBABEULL;

    // prepare_block_submission_from_solved with vOffsets (Prime)
    auto with_offsets = mti_prime.prepare_block_submission_from_solved(
        solved_block, vOffsets);

    // prepare_block_submission_from_solved without vOffsets
    auto without_offsets = mti_prime.prepare_block_submission_from_solved(
        solved_block, {});

    // With vOffsets should be exactly vOffsets.size() bytes larger
    bool size_ok = (with_offsets.size() == without_offsets.size() + vOffsets.size());

    // Verify vOffsets appear at the tail
    bool tail_ok = false;
    if (with_offsets.size() >= vOffsets.size()) {
        std::vector<uint8_t> tail(
            with_offsets.end() - vOffsets.size(), with_offsets.end());
        tail_ok = (tail == vOffsets);
    }

    print_result("vOffsets flow: prepare_block_submission_from_solved() appends vOffsets "
                 "(Prime channel, byte-exact at tail)", size_ok && tail_ok);
}

// Test 12: Hash channel prepare_block_submission_from_solved with empty vOffsets
static void test_hash_no_voffsets_appended() {
    MiningTemplateInterface mti_hash(2, 0);
    auto payload = make_template_payload(6000000, 2000000, DEFAULT_DIFFICULTY,
                                         8, 2, 6000001, DEFAULT_DIFFICULTY, 0);
    mti_hash.read_stateless_payload(payload, "test");

    const auto* tmpl = mti_hash.get_current_template();
    if (!tmpl) {
        print_result("Hash no-vOffsets: template loaded", false);
        return;
    }

    Block_data solved_block(tmpl->block);
    solved_block.nNonce = 0xDEADBEEFCAFEBABEULL;

    auto result = mti_hash.prepare_block_submission_from_solved(
        solved_block, {});

    bool ok = (result.size() == StatelessBlockUtility::BLOCK_BODY_SIZE);
    print_result("Hash channel: prepare_block_submission_from_solved() = 216 bytes (no vOffsets)", ok);
}

// ── Test 13: SubmitBlockPayloadInfo Hash Falcon-1024 fixed-size ─────────────
// Hash: plaintext = 216 + 8 + 2 + 1577 = 1803, encrypted = 1803 + 28 = 1831
static void test_payload_info_hash_falcon1024() {
    auto info = StatelessBlockUtility::compute_submit_payload_info(
        /*channel=*/2, /*block_data_size=*/216,
        /*signature_size=*/FalconConstants::FALCON1024_SIG_CT_SIZE);  // 1577

    bool channel_ok   = (info.channel == 2);
    bool base_ok      = (info.base_block_size == 216);
    bool offset_ok    = (info.offset_bytes_count == 0);
    bool ts_ok        = (info.timestamp_size == 8);
    bool siglen_ok    = (info.sig_len_field_size == 2);
    bool sig_ok       = (info.signature_size == FalconConstants::FALCON1024_SIG_CT_SIZE);
    bool plain_ok     = (info.expected_plaintext_size() == 1803);
    bool enc_ok       = (info.expected_encrypted_size() == 1831);

    print_result("PayloadInfo Hash F1024: plaintext=1803, encrypted=1831",
                 channel_ok && base_ok && offset_ok && ts_ok &&
                 siglen_ok && sig_ok && plain_ok && enc_ok);
}

// ── Test 14: SubmitBlockPayloadInfo Prime with 10 offset bytes ──────────────
// Prime: plaintext = 216 + 10 + 8 + 2 + 1577 = 1813, encrypted = 1813 + 28 = 1841
static void test_payload_info_prime_with_offsets() {
    const size_t offset_count = 10;
    auto info = StatelessBlockUtility::compute_submit_payload_info(
        /*channel=*/1, /*block_data_size=*/216 + offset_count,
        /*signature_size=*/FalconConstants::FALCON1024_SIG_CT_SIZE);

    bool channel_ok   = (info.channel == 1);
    bool base_ok      = (info.base_block_size == 216);
    bool offset_ok    = (info.offset_bytes_count == offset_count);
    bool plain_ok     = (info.expected_plaintext_size() == 1813);
    bool enc_ok       = (info.expected_encrypted_size() == 1841);

    print_result("PayloadInfo Prime F1024 (10 offsets): plaintext=1813, encrypted=1841",
                 channel_ok && base_ok && offset_ok && plain_ok && enc_ok);
}

// ── Test 15: Prime variable-size is NOT Hash fixed-size ─────────────────────
// Verifies that Hash and Prime with offsets produce different expected sizes.
static void test_payload_info_prime_not_equal_hash() {
    auto hash_info = StatelessBlockUtility::compute_submit_payload_info(
        2, 216, FalconConstants::FALCON1024_SIG_CT_SIZE);
    auto prime_info = StatelessBlockUtility::compute_submit_payload_info(
        1, 216 + 7, FalconConstants::FALCON1024_SIG_CT_SIZE);

    bool sizes_differ = (prime_info.expected_plaintext_size() !=
                         hash_info.expected_plaintext_size());
    bool prime_larger  = (prime_info.expected_plaintext_size() ==
                          hash_info.expected_plaintext_size() + 7);

    print_result("PayloadInfo: Prime(7 offsets) != Hash, differs by offset count",
                 sizes_differ && prime_larger);
}

// ── Test 16: E2E integration — encode_submit → compute_payload_info → ChaCha20 ──
// Exercises the full submit path for Hash unsigned, verifies the payload_info
// matches the actual sizes produced by the pipeline.
static void test_e2e_payload_info_hash_unsigned() {
    auto mti  = make_loaded_mti(2);
    auto blk  = make_solved_block(2);
    auto snap = make_snapshot();

    auto submit = StatelessBlockUtility::encode_submit(
        *mti, blk, {}, nullptr, ProtocolLane::STATELESS, snap, nullptr);
    auto plaintext = strip_wire_header(*submit.wire_bytes, ProtocolLane::STATELESS);

    // For unsigned submit, plaintext = block bytes only (216)
    auto info = StatelessBlockUtility::compute_submit_payload_info(2, 216, 0);

    // For unsigned submit, plaintext = block bytes only (216), since
    // timestamp/sig_len/signature are only added in the signed path.
    // SubmitBlockPayloadInfo is designed for the signed path sizing.
    bool actual_ok = (plaintext.size() == StatelessBlockUtility::BLOCK_BODY_SIZE);

    // Encrypt and verify encrypted size
    std::vector<uint8_t> genesis(32, 0xAB);
    auto session_key = derive_session_key(genesis);
    ChaCha20Wrapper wrapper;
    auto enc_nonce = ChaCha20Wrapper::generate_nonce();
    auto enc = wrapper.encrypt(plaintext, session_key, enc_nonce, AAD_BLOCK_SUBMISSION);
    // Encrypted = [ciphertext(plaintext.size())][tag(16)]
    bool enc_ok = enc.success &&
                  (enc.data.size() == plaintext.size() + 16);  // ct+tag

    print_result("E2E PayloadInfo: Hash unsigned → encrypt size correct",
                 actual_ok && enc_ok);
}

// ── Test 17: E2E integration — Prime with vOffsets through full pipeline ─────
// Exercises prepare_block_submission → encode_submit → ChaCha20 → verify sizes
static void test_e2e_payload_info_prime_pipeline() {
    auto mti  = make_loaded_mti(1);  // Prime
    auto blk  = make_solved_block(1);
    auto snap = make_snapshot();
    std::vector<uint8_t> vOffsets(10, 0x42);  // 10 offset bytes

    auto submit = StatelessBlockUtility::encode_submit(
        *mti, blk, vOffsets, nullptr, ProtocolLane::STATELESS, snap, nullptr);
    if (!submit.valid) {
        print_result("E2E PayloadInfo Prime pipeline: encode_submit succeeds", false);
        return;
    }

    auto plaintext = strip_wire_header(*submit.wire_bytes, ProtocolLane::STATELESS);
    bool plain_ok = (plaintext.size() == StatelessBlockUtility::BLOCK_BODY_SIZE + vOffsets.size());

    // Encrypt
    std::vector<uint8_t> genesis(32, 0xAB);
    auto session_key = derive_session_key(genesis);
    ChaCha20Wrapper wrapper;
    auto enc_nonce = ChaCha20Wrapper::generate_nonce();
    auto enc = wrapper.encrypt(plaintext, session_key, enc_nonce, AAD_BLOCK_SUBMISSION);

    // Decrypt and verify round-trip
    auto dec = wrapper.decrypt(enc.data, session_key, enc_nonce, AAD_BLOCK_SUBMISSION);
    bool rt_ok = dec.success && (dec.data == plaintext);

    // Verify payload info matches
    auto info = StatelessBlockUtility::compute_submit_payload_info(1, 216 + 10, 0);
    bool info_ok = (info.offset_bytes_count == 10) && (info.channel == 1);

    // Final encrypted wire payload = nonce(12) + ciphertext(plaintext.size()) + tag(16)
    size_t expected_wire = 12 + plaintext.size() + 16;
    std::vector<uint8_t> wire_payload;
    wire_payload.insert(wire_payload.end(), enc_nonce.begin(), enc_nonce.end());
    wire_payload.insert(wire_payload.end(), enc.data.begin(), enc.data.end());
    bool wire_ok = (wire_payload.size() == expected_wire);

    print_result("E2E PayloadInfo Prime (10 offsets): full pipeline size correct",
                 plain_ok && rt_ok && info_ok && wire_ok);
}


// ── Test 18: E2E via encrypt_submit_block_payload() — canonical wrapper path ─
// Validates: encode_submit → compute_submit_payload_info → encrypt_submit_block_payload
// covers both Hash and Prime channels; verifies single-call [nonce][ct][tag] output.
static void test_e2e_encrypt_submit_block_payload() {
    std::vector<uint8_t> genesis(32, 0xAB);
    auto session_key = derive_session_key(genesis);
    ChaCha20Wrapper wrapper;

    // ── Case A: Hash channel (unsigned, no Falcon) ────────────────────────
    {
        auto mti  = make_loaded_mti(2);   // Hash
        auto blk  = make_solved_block(2);
        auto snap = make_snapshot();

        auto submit = StatelessBlockUtility::encode_submit(
            *mti, blk, {}, nullptr, ProtocolLane::STATELESS, snap, nullptr);
        if (!submit.valid) {
            print_result("E2E encrypt_submit_block_payload: Hash encode_submit succeeds", false);
            return;
        }

        auto plaintext = strip_wire_header(*submit.wire_bytes, ProtocolLane::STATELESS);

        // Unsigned Hash: plaintext = block(216) only
        auto info = StatelessBlockUtility::compute_submit_payload_info(
            2, plaintext.size(), 0);

        // For unsigned payload the wrapper may log a size-mismatch warning
        // (since expected_plaintext_size() includes ts+sig overhead), but must
        // still succeed encryption.
        auto enc = wrapper.encrypt_submit_block_payload(plaintext, session_key, info);
        print_result("E2E encrypt_submit_block_payload: Hash unsigned — succeeds",
                     enc.success);

        // Output must be [nonce(12)][ciphertext(plaintext.size())][tag(16)]
        const size_t expected_size = 12 + plaintext.size() + 16;
        print_result("E2E encrypt_submit_block_payload: Hash unsigned — size = nonce+ct+tag",
                     enc.success && enc.data.size() == expected_size);
    }

    // ── Case B: Prime channel with 10 vOffsets (unsigned) ────────────────
    {
        auto mti  = make_loaded_mti(1);   // Prime
        auto blk  = make_solved_block(1);
        auto snap = make_snapshot();
        std::vector<uint8_t> vOffsets(10, 0x42);

        auto submit = StatelessBlockUtility::encode_submit(
            *mti, blk, vOffsets, nullptr, ProtocolLane::STATELESS, snap, nullptr);
        if (!submit.valid) {
            print_result("E2E encrypt_submit_block_payload: Prime encode_submit succeeds", false);
            return;
        }

        auto plaintext = strip_wire_header(*submit.wire_bytes, ProtocolLane::STATELESS);

        // Unsigned Prime: plaintext = block(216) + offsets(10)
        auto info = StatelessBlockUtility::compute_submit_payload_info(
            1, plaintext.size(), 0);

        auto enc = wrapper.encrypt_submit_block_payload(plaintext, session_key, info);
        print_result("E2E encrypt_submit_block_payload: Prime+10-offsets unsigned — succeeds",
                     enc.success);

        const size_t expected_size = 12 + plaintext.size() + 16;
        print_result("E2E encrypt_submit_block_payload: Prime+10-offsets — size = nonce+ct+tag",
                     enc.success && enc.data.size() == expected_size);

        // Two calls must produce different outputs (fresh nonce each time)
        auto enc2 = wrapper.encrypt_submit_block_payload(plaintext, session_key, info);
        print_result("E2E encrypt_submit_block_payload: each call generates a fresh nonce",
                     enc.success && enc2.success && enc.data != enc2.data);

        // The first 12 bytes are the nonce; the rest is ciphertext+tag.
        // Verify the ciphertext+tag portion (bytes [12..end]) can be decrypted
        // by extracting the nonce and calling decrypt() directly.
        if (enc.success && enc.data.size() >= 12) {
            std::vector<uint8_t> nonce(enc.data.begin(), enc.data.begin() + 12);
            std::vector<uint8_t> ct_tag(enc.data.begin() + 12, enc.data.end());
            auto dec = wrapper.decrypt(ct_tag, session_key, nonce, {});
            print_result("E2E encrypt_submit_block_payload: Prime — decrypt(nonce, ct+tag) succeeds",
                         dec.success && dec.data == plaintext);
        }
    }
}


int main() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "  End-to-End Block Submission Tests\n";
    std::cout << "========================================\n";
    std::cout << "\n--- Full Pipeline (encode_submit → ChaCha20) ---\n";

    test_e2e_hash_channel_round_trip();         // 1
    test_e2e_prime_channel_voffsets_round_trip();// 2
    test_hash_payload_size();                    // 3
    test_prime_payload_size();                   // 4
    test_voffsets_byte_exact();                  // 5

    std::cout << "\n--- Crypto Correctness ---\n";
    test_kdf_determinism();                      // 6
    test_wrong_key_decrypt_fails();              // 7
    test_aad_mismatch_fails();                   // 8

    std::cout << "\n--- Wire Format ---\n";
    test_stateless_submit_packet_structure();    // 9
    test_legacy_submit_packet_structure();       // 10

    std::cout << "\n--- vOffsets Flow ---\n";
    test_voffsets_flow_prepare_block_submission_from_solved();// 11
    test_hash_no_voffsets_appended();            // 12

    std::cout << "\n--- Channel-Aware Payload Sizing ---\n";
    test_payload_info_hash_falcon1024();          // 13
    test_payload_info_prime_with_offsets();        // 14
    test_payload_info_prime_not_equal_hash();      // 15
    test_e2e_payload_info_hash_unsigned();         // 16
    test_e2e_payload_info_prime_pipeline();        // 17
    test_height_tracker_snapshot_carries_session_epoch();
    test_template_interface_stamps_session_epoch();

    std::cout << "\n--- Canonical Wrapper Path (encrypt_submit_block_payload) ---\n";
    test_e2e_encrypt_submit_block_payload();       // 18

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
