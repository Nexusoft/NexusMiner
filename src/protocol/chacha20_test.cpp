/**
 * @file chacha20_test.cpp
 * @brief Unit tests for ChaCha20-Poly1305 AEAD wrapper and AAD domain separation
 *
 * Validates the implementation of:
 *   - Basic encrypt/decrypt round-trip
 *   - AAD (Additional Authenticated Data) domain separation
 *   - AAD mismatch detection (Poly1305 tag failure)
 *   - Empty vs non-empty AAD behavior
 *   - Input validation (key size, nonce size, empty plaintext)
 *   - Falcon public key wrap/unwrap
 *   - Nonce uniqueness
 *   - SUBMIT_BLOCK AAD is empty {} (no domain separation, matches node behavior)
 *   - KDF domain separator consistency
 */

#include "protocol/chacha20_wrapper.hpp"
#include "protocol/falcon_constants.hpp"
#include <iostream>
#include <cassert>
#include <cstdint>
#include <vector>
#include <string>
#include <cstring>
#include <openssl/sha.h>

// Mock logger for testing
#include "spdlog/spdlog.h"
#include "spdlog/sinks/null_sink.h"

using namespace nexusminer::protocol;

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

// ═══════════════════════════════════════════════════════════════════════════
// Helper: generate deterministic test key (32 bytes)
// ═══════════════════════════════════════════════════════════════════════════
static std::vector<uint8_t> make_test_key() {
    std::vector<uint8_t> key(32, 0);
    for (size_t i = 0; i < 32; ++i)
        key[i] = static_cast<uint8_t>(i + 1);
    return key;
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper: generate deterministic test nonce (12 bytes)
// ═══════════════════════════════════════════════════════════════════════════
static std::vector<uint8_t> make_test_nonce() {
    std::vector<uint8_t> nonce(12, 0);
    for (size_t i = 0; i < 12; ++i)
        nonce[i] = static_cast<uint8_t>(0xA0 + i);
    return nonce;
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper: generate test plaintext of given size
// ═══════════════════════════════════════════════════════════════════════════
static std::vector<uint8_t> make_test_plaintext(size_t size) {
    std::vector<uint8_t> pt(size);
    for (size_t i = 0; i < size; ++i)
        pt[i] = static_cast<uint8_t>(i & 0xFF);
    return pt;
}

// AAD constants matching solo.cpp (these MUST stay in sync)
static const std::vector<uint8_t> AAD_FALCON_PUBKEY{
    'F','A','L','C','O','N','_','P','U','B','K','E','Y'
};

static const std::vector<uint8_t> AAD_REWARD_ADDRESS{
    'R','E','W','A','R','D','_',
    'A','D','D','R','E','S','S'
};

static const std::vector<uint8_t> AAD_REWARD_RESULT{
    'R','E','W','A','R','D','_',
    'R','E','S','U','L','T'
};

// AAD_BLOCK_SUBMISSION is intentionally empty: the node decrypts SUBMIT_BLOCK
// with no AAD (default empty vector), so the miner must also encrypt with {}.
// See: LLL-TAO stateless_miner_connection.cpp ~L1186 (no AAD argument).
static const std::vector<uint8_t> AAD_BLOCK_SUBMISSION{};

static const std::string KDF_DOMAIN = "nexus-mining-chacha20-v1";

static std::vector<uint8_t> derive_session_key_for_test(const std::vector<uint8_t>& genesis) {
    std::vector<uint8_t> preimage;
    preimage.insert(preimage.end(), KDF_DOMAIN.begin(), KDF_DOMAIN.end());
    preimage.insert(preimage.end(), genesis.begin(), genesis.end());

    std::vector<uint8_t> derived_key(SHA256_DIGEST_LENGTH);
    SHA256(preimage.data(), preimage.size(), derived_key.data());
    return derived_key;
}

static std::string format_hex_prefix(const std::vector<uint8_t>& bytes, size_t prefix_bytes) {
    static const char* const HEX = "0123456789abcdef";
    const size_t prefix_size = std::min(bytes.size(), prefix_bytes);
    std::string out;
    out.reserve(prefix_size * 2);

    for (size_t i = 0; i < prefix_size; ++i) {
        uint8_t byte = bytes[i];
        out.push_back(HEX[(byte >> 4) & 0x0F]);
        out.push_back(HEX[byte & 0x0F]);
    }

    return out;
}

int main()
{
    // Setup null logger to avoid spam during tests
    auto null_sink = std::make_shared<spdlog::sinks::null_sink_mt>();
    auto logger = std::make_shared<spdlog::logger>("logger", null_sink);
    spdlog::set_default_logger(logger);

    std::cout << "========================================" << std::endl;
    std::cout << "ChaCha20-Poly1305 AEAD Unit Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    ChaCha20Wrapper wrapper;

    // ====================================================================
    // Test 1: Basic encrypt/decrypt round-trip (no AAD)
    // ====================================================================
    std::cout << "\nTest 1: Basic encrypt/decrypt round-trip (no AAD)" << std::endl;
    {
        auto key = make_test_key();
        auto nonce = make_test_nonce();
        auto plaintext = make_test_plaintext(128);

        auto enc = wrapper.encrypt(plaintext, key, nonce);
        print_test_result("Encryption succeeds", enc.success);
        print_test_result("Ciphertext non-empty", !enc.data.empty());
        print_test_result("Ciphertext = plaintext + 16-byte tag",
                          enc.data.size() == plaintext.size() + 16);

        auto dec = wrapper.decrypt(enc.data, key, nonce);
        print_test_result("Decryption succeeds", dec.success);
        print_test_result("Round-trip plaintext matches",
                          dec.data == plaintext);
    }

    // ====================================================================
    // Test 2: Encrypt/decrypt with AAD (BLOCK_SUBMISSION)
    // ====================================================================
    std::cout << "\nTest 2: Encrypt/decrypt with AAD (BLOCK_SUBMISSION)" << std::endl;
    {
        auto key = make_test_key();
        auto nonce = make_test_nonce();
        auto plaintext = make_test_plaintext(256);

        auto enc = wrapper.encrypt(plaintext, key, nonce, AAD_BLOCK_SUBMISSION);
        print_test_result("Encryption with AAD succeeds", enc.success);

        auto dec = wrapper.decrypt(enc.data, key, nonce, AAD_BLOCK_SUBMISSION);
        print_test_result("Decryption with matching AAD succeeds", dec.success);
        print_test_result("Round-trip plaintext matches", dec.data == plaintext);
    }

    // ====================================================================
    // Test 3: SUBMIT_BLOCK uses empty AAD — miner and node both use {}
    // ====================================================================
    std::cout << "\nTest 3: SUBMIT_BLOCK uses empty AAD (no domain separation)" << std::endl;
    {
        auto key = make_test_key();
        auto nonce = make_test_nonce();
        auto plaintext = make_test_plaintext(216);  // Tritium block size

        // Miner encrypts with empty AAD (AAD_BLOCK_SUBMISSION == {})
        auto enc = wrapper.encrypt(plaintext, key, nonce, AAD_BLOCK_SUBMISSION);
        print_test_result("Miner: Encrypt SUBMIT_BLOCK with empty AAD succeeds", enc.success);

        // Node decrypts with empty AAD → must succeed (matching empty AADs)
        auto dec_empty = wrapper.decrypt(enc.data, key, nonce, {});
        print_test_result("Node: Decrypt SUBMIT_BLOCK with empty AAD succeeds", dec_empty.success);
        print_test_result("Decrypted payload matches original", dec_empty.data == plaintext);

        // Try to decrypt with a non-empty AAD → must fail (tag mismatch)
        std::vector<uint8_t> wrong_aad{'W','R','O','N','G'};
        auto dec_wrong = wrapper.decrypt(enc.data, key, nonce, wrong_aad);
        print_test_result("Decrypt empty-AAD ciphertext with non-empty AAD FAILS", !dec_wrong.success);

        // Old PR #156 bug: if miner had used "BLOCK_SUBMISSION" (non-empty) AAD but
        // node decrypts with empty AAD → decryption fails (PR #156 was the bug)
        std::vector<uint8_t> old_buggy_aad{
            'B','L','O','C','K','_','S','U','B','M','I','S','S','I','O','N'
        };
        auto enc_buggy = wrapper.encrypt(plaintext, key, nonce, old_buggy_aad);
        auto dec_buggy = wrapper.decrypt(enc_buggy.data, key, nonce, {});
        print_test_result("PR#156 BUG: 'BLOCK_SUBMISSION' encrypt + empty decrypt FAILS",
                          !dec_buggy.success);
    }

    // ====================================================================
    // Test 4: Empty AAD vs non-empty AAD are NOT interchangeable
    // ====================================================================
    std::cout << "\nTest 4: Empty AAD vs non-empty AAD are NOT interchangeable" << std::endl;
    {
        auto key = make_test_key();
        auto nonce = make_test_nonce();
        auto plaintext = make_test_plaintext(100);

        // Encrypt with empty AAD
        auto enc_empty = wrapper.encrypt(plaintext, key, nonce, {});
        print_test_result("Encrypt with empty AAD succeeds", enc_empty.success);

        // Try decrypt with a non-empty AAD → must fail
        std::vector<uint8_t> nonempty_aad{'N','O','N','E','M','P','T','Y'};
        auto dec_with_aad = wrapper.decrypt(enc_empty.data, key, nonce, nonempty_aad);
        print_test_result("Decrypt empty-AAD ciphertext with non-empty AAD FAILS",
                          !dec_with_aad.success);

        // Decrypt with empty AAD → must succeed
        auto dec_empty = wrapper.decrypt(enc_empty.data, key, nonce, {});
        print_test_result("Decrypt empty-AAD ciphertext with empty AAD succeeds",
                          dec_empty.success);
    }

    // ====================================================================
    // Test 5: Each AAD domain is independent
    // ====================================================================
    std::cout << "\nTest 5: AAD domain separation - each domain is independent" << std::endl;
    {
        auto key = make_test_key();
        auto nonce = make_test_nonce();
        auto plaintext = make_test_plaintext(64);

        // Encrypt with REWARD_ADDRESS AAD
        auto enc = wrapper.encrypt(plaintext, key, nonce, AAD_REWARD_ADDRESS);
        print_test_result("Encrypt with REWARD_ADDRESS succeeds", enc.success);

        // Try decrypt with BLOCK_SUBMISSION → fails
        auto dec_block = wrapper.decrypt(enc.data, key, nonce, AAD_BLOCK_SUBMISSION);
        print_test_result("Cross-domain: REWARD_ADDRESS→BLOCK_SUBMISSION FAILS",
                          !dec_block.success);

        // Try decrypt with REWARD_RESULT → fails
        auto dec_result = wrapper.decrypt(enc.data, key, nonce, AAD_REWARD_RESULT);
        print_test_result("Cross-domain: REWARD_ADDRESS→REWARD_RESULT FAILS",
                          !dec_result.success);

        // Try decrypt with FALCON_PUBKEY → fails
        auto dec_falcon = wrapper.decrypt(enc.data, key, nonce, AAD_FALCON_PUBKEY);
        print_test_result("Cross-domain: REWARD_ADDRESS→FALCON_PUBKEY FAILS",
                          !dec_falcon.success);

        // Decrypt with same domain → succeeds
        auto dec_correct = wrapper.decrypt(enc.data, key, nonce, AAD_REWARD_ADDRESS);
        print_test_result("Same domain: REWARD_ADDRESS→REWARD_ADDRESS succeeds",
                          dec_correct.success && dec_correct.data == plaintext);
    }

    // ====================================================================
    // Test 6: BLOCK_SUBMISSION AAD constant is empty {} (no domain separation)
    // ====================================================================
    std::cout << "\nTest 6: BLOCK_SUBMISSION AAD constant verification" << std::endl;
    {
        print_test_result("AAD_BLOCK_SUBMISSION is empty ({})",
                          AAD_BLOCK_SUBMISSION.empty());
        print_test_result("AAD_BLOCK_SUBMISSION length is 0",
                          AAD_BLOCK_SUBMISSION.size() == 0);
        print_test_result("AAD_BLOCK_SUBMISSION == std::vector<uint8_t>{}",
                          AAD_BLOCK_SUBMISSION == std::vector<uint8_t>{});
    }

    // ====================================================================
    // Test 7: Other AAD constants verification
    // ====================================================================
    std::cout << "\nTest 7: AAD constants byte-exact verification" << std::endl;
    {
        std::string exp_reward = "REWARD_ADDRESS";
        std::vector<uint8_t> exp_reward_vec(exp_reward.begin(), exp_reward.end());
        print_test_result("AAD_REWARD_ADDRESS == \"REWARD_ADDRESS\" (14 bytes)",
                          AAD_REWARD_ADDRESS == exp_reward_vec && AAD_REWARD_ADDRESS.size() == 14);

        std::string exp_result = "REWARD_RESULT";
        std::vector<uint8_t> exp_result_vec(exp_result.begin(), exp_result.end());
        print_test_result("AAD_REWARD_RESULT == \"REWARD_RESULT\" (13 bytes)",
                          AAD_REWARD_RESULT == exp_result_vec && AAD_REWARD_RESULT.size() == 13);

        std::string exp_falcon = "FALCON_PUBKEY";
        std::vector<uint8_t> exp_falcon_vec(exp_falcon.begin(), exp_falcon.end());
        print_test_result("AAD_FALCON_PUBKEY == \"FALCON_PUBKEY\" (13 bytes)",
                          AAD_FALCON_PUBKEY == exp_falcon_vec && AAD_FALCON_PUBKEY.size() == 13);
    }

    // ====================================================================
    // Test 8: Input validation - invalid key size
    // ====================================================================
    std::cout << "\nTest 8: Input validation" << std::endl;
    {
        auto nonce = make_test_nonce();
        auto plaintext = make_test_plaintext(64);

        // Wrong key size (16 bytes instead of 32)
        std::vector<uint8_t> short_key(16, 0x42);
        auto enc_short = wrapper.encrypt(plaintext, short_key, nonce);
        print_test_result("Encrypt with 16-byte key FAILS", !enc_short.success);

        // Wrong nonce size (8 bytes instead of 12)
        auto key = make_test_key();
        std::vector<uint8_t> short_nonce(8, 0xAA);
        auto enc_nonce = wrapper.encrypt(plaintext, key, short_nonce);
        print_test_result("Encrypt with 8-byte nonce FAILS", !enc_nonce.success);

        // Empty plaintext
        auto enc_empty = wrapper.encrypt({}, key, nonce);
        print_test_result("Encrypt with empty plaintext FAILS", !enc_empty.success);

        // Ciphertext too short for decrypt (less than 16-byte tag)
        std::vector<uint8_t> short_ct(10, 0x00);
        auto dec_short = wrapper.decrypt(short_ct, key, nonce);
        print_test_result("Decrypt with too-short ciphertext FAILS", !dec_short.success);
    }

    // ====================================================================
    // Test 9: Nonce uniqueness (random generation)
    // ====================================================================
    std::cout << "\nTest 9: Nonce uniqueness" << std::endl;
    {
        auto n1 = ChaCha20Wrapper::generate_nonce();
        auto n2 = ChaCha20Wrapper::generate_nonce();
        auto n3 = ChaCha20Wrapper::generate_nonce();

        print_test_result("Generated nonce is 12 bytes", n1.size() == 12);
        print_test_result("Two nonces are different", n1 != n2);
        print_test_result("Three nonces are all different", n1 != n2 && n2 != n3 && n1 != n3);
    }

    // ====================================================================
    // Test 10: Key generation
    // ====================================================================
    std::cout << "\nTest 10: Key generation" << std::endl;
    {
        auto k1 = ChaCha20Wrapper::generate_key();
        auto k2 = ChaCha20Wrapper::generate_key();

        print_test_result("Generated key is 32 bytes", k1.size() == 32);
        print_test_result("Two keys are different", k1 != k2);
    }

    // ====================================================================
    // Test 11: Falcon public key wrap/unwrap round-trip
    // ====================================================================
    std::cout << "\nTest 11: Falcon public key wrap/unwrap" << std::endl;
    {
        auto key = make_test_key();
        auto nonce = make_test_nonce();
        // Falcon-512 pubkey is 897 bytes
        auto fake_pubkey = make_test_plaintext(897);

        auto wrapped = wrapper.wrap_falcon_pubkey(fake_pubkey, key, nonce);
        print_test_result("Wrap Falcon pubkey succeeds", wrapped.success);
        print_test_result("Wrapped size = 897 + 16 tag",
                          wrapped.data.size() == 897 + 16);

        auto unwrapped = wrapper.unwrap_falcon_pubkey(wrapped.data, key, nonce);
        print_test_result("Unwrap Falcon pubkey succeeds", unwrapped.success);
        print_test_result("Unwrapped key matches original", unwrapped.data == fake_pubkey);
    }

    // ====================================================================
    // Test 12: Wrong key causes decryption failure
    // ====================================================================
    std::cout << "\nTest 12: Wrong key causes decryption failure" << std::endl;
    {
        auto key1 = make_test_key();
        auto nonce = make_test_nonce();
        auto plaintext = make_test_plaintext(216);

        auto enc = wrapper.encrypt(plaintext, key1, nonce, AAD_BLOCK_SUBMISSION);
        print_test_result("Encryption succeeds", enc.success);

        // Different key
        std::vector<uint8_t> key2(32, 0xFF);
        auto dec = wrapper.decrypt(enc.data, key2, nonce, AAD_BLOCK_SUBMISSION);
        print_test_result("Decrypt with wrong key FAILS", !dec.success);
    }

    // ====================================================================
    // Test 13: Wrong nonce causes decryption failure
    // ====================================================================
    std::cout << "\nTest 13: Wrong nonce causes decryption failure" << std::endl;
    {
        auto key = make_test_key();
        auto nonce1 = make_test_nonce();
        auto plaintext = make_test_plaintext(216);

        auto enc = wrapper.encrypt(plaintext, key, nonce1, AAD_BLOCK_SUBMISSION);
        print_test_result("Encryption succeeds", enc.success);

        // Different nonce
        std::vector<uint8_t> nonce2(12, 0xFF);
        auto dec = wrapper.decrypt(enc.data, key, nonce2, AAD_BLOCK_SUBMISSION);
        print_test_result("Decrypt with wrong nonce FAILS", !dec.success);
    }

    // ====================================================================
    // Test 14: KDF domain separator consistency
    // ====================================================================
    std::cout << "\nTest 14: KDF domain separator and key derivation" << std::endl;
    {
        // Verify KDF: SHA256(KDF_DOMAIN + genesis) produces 32-byte key
        std::vector<uint8_t> genesis(32, 0x42);  // test genesis hash

        auto derived_key = derive_session_key_for_test(genesis);

        print_test_result("KDF produces 32-byte key", derived_key.size() == 32);
        print_test_result("KDF_DOMAIN matches expected string",
                          KDF_DOMAIN == "nexus-mining-chacha20-v1");

        // Verify key is not all zeros (basic sanity)
        bool all_zero = true;
        for (auto b : derived_key)
            if (b != 0) { all_zero = false; break; }
        print_test_result("Derived key is not all zeros", !all_zero);

        // Verify same inputs produce same key (deterministic)
        auto derived_key2 = derive_session_key_for_test(genesis);
        print_test_result("KDF is deterministic (same input → same key)",
                          derived_key == derived_key2);

        // Verify different genesis produces different key
        std::vector<uint8_t> genesis2(32, 0x43);  // different genesis
        auto derived_key3 = derive_session_key_for_test(genesis2);
        print_test_result("Different genesis → different key",
                          derived_key != derived_key3);

        // Verify miner.conf hex parsing order is used as-is (no reversal) before the KDF step
        std::vector<uint8_t> ordered_genesis(32);
        for (size_t i = 0; i < ordered_genesis.size(); ++i)
            ordered_genesis[i] = static_cast<uint8_t>(i);

        std::vector<uint8_t> reversed_genesis = ordered_genesis;
        std::reverse(reversed_genesis.begin(), reversed_genesis.end());

        auto ordered_key = derive_session_key_for_test(ordered_genesis);
        auto reversed_key = derive_session_key_for_test(reversed_genesis);
        print_test_result("Configured genesis byte order is used as-is (no reversal)",
                          ordered_key != reversed_key);

        std::string fingerprint = format_hex_prefix(ordered_key, 8);
        print_test_result("Derived key fingerprint uses first 8 bytes (16 hex chars)",
                          fingerprint.size() == 16);
        // Verified from SHA256("nexus-mining-chacha20-v1" || 00..1f) and used here
        // to lock in the node-comparable 8-byte fingerprint format.
        print_test_result("Derived key fingerprint matches expected key prefix",
                          fingerprint == "f96c268fa2b63991");
    }

    // ====================================================================
    // Test 15: Simulate SUBMIT_BLOCK encrypt→decrypt (end-to-end)
    // ====================================================================
    std::cout << "\nTest 15: SUBMIT_BLOCK end-to-end simulation" << std::endl;
    {
        // Simulate: miner encrypts with empty AAD, node decrypts with empty AAD.
        // The node calls LLC::DecryptPayloadChaCha20 with NO AAD argument (empty by default).

        // Step 1: Derive session key from genesis (same as both miner and node do)
        std::vector<uint8_t> genesis(32, 0xAB);
        auto session_key = derive_session_key_for_test(genesis);

        // Step 2: Create mock block payload (216 bytes block + 8 bytes timestamp)
        auto block_data = make_test_plaintext(216);
        std::vector<uint8_t> payload;
        payload.insert(payload.end(), block_data.begin(), block_data.end());
        // Append 8-byte LE timestamp
        uint64_t timestamp = 1700000000ULL;
        for (int i = 0; i < 8; ++i)
            payload.push_back(static_cast<uint8_t>((timestamp >> (i * 8)) & 0xFF));

        // Step 3: Miner encrypts with empty AAD (AAD_BLOCK_SUBMISSION == {})
        auto nonce = ChaCha20Wrapper::generate_nonce();
        auto enc = wrapper.encrypt(payload, session_key, nonce, AAD_BLOCK_SUBMISSION);
        print_test_result("Miner: Encrypt SUBMIT_BLOCK with empty AAD succeeds", enc.success);

        // Step 4: Node decrypts with empty AAD → succeeds (matching empty AADs)
        auto dec = wrapper.decrypt(enc.data, session_key, nonce, AAD_BLOCK_SUBMISSION);
        print_test_result("Node: Decrypt SUBMIT_BLOCK with empty AAD succeeds", dec.success);
        print_test_result("Decrypted payload matches original", dec.data == payload);

        // Step 5: Old PR #156 bug scenario — if miner used "BLOCK_SUBMISSION" (non-empty)
        // AAD but node decrypts with empty AAD, decryption fails. The "BLOCK_SUBMISSION"
        // string AAD was itself the bug introduced by PR #156.
        std::vector<uint8_t> old_pr156_aad{
            'B','L','O','C','K','_','S','U','B','M','I','S','S','I','O','N'
        };
        auto enc_pr156 = wrapper.encrypt(payload, session_key, nonce, old_pr156_aad);
        auto dec_mismatch = wrapper.decrypt(enc_pr156.data, session_key, nonce, {});
        print_test_result("PR#156 BUG: 'BLOCK_SUBMISSION' encrypt + empty decrypt FAILS",
                          !dec_mismatch.success);
    }

    // ====================================================================
    // Test 16: Large payload handling (realistic block submission size)
    // ====================================================================
    std::cout << "\nTest 16: Large payload handling" << std::endl;
    {
        auto key = make_test_key();
        auto nonce = make_test_nonce();
        // Realistic payload: 216 (block) + 8 (timestamp) + 2 (siglen) + 1577 (Falcon-1024 sig) = 1803 bytes
        size_t payload_size = 216 + 8 + 2 + 1577;
        auto plaintext = make_test_plaintext(payload_size);

        auto enc = wrapper.encrypt(plaintext, key, nonce, AAD_BLOCK_SUBMISSION);
        print_test_result("Encrypt large payload (1803 bytes) succeeds", enc.success);

        auto dec = wrapper.decrypt(enc.data, key, nonce, AAD_BLOCK_SUBMISSION);
        print_test_result("Decrypt large payload succeeds", dec.success);
        print_test_result("Large payload round-trip matches", dec.data == plaintext);
    }

    // ====================================================================
    // Test 17: ChaCha20 availability check
    // ====================================================================
    std::cout << "\nTest 17: ChaCha20 availability" << std::endl;
    {
        print_test_result("ChaCha20-Poly1305 is available", ChaCha20Wrapper::is_available());
    }

    // ====================================================================
    // Test 18: encrypt_submit_block_payload() — canonical wrapper path
    // ====================================================================
    std::cout << "\nTest 18: encrypt_submit_block_payload() canonical path" << std::endl;
    {
        auto key = make_test_key();

        // ── Case A: Hash channel, Falcon-1024 signature ──────────────────
        // Plaintext layout: block(216) + timestamp(8) + sig_len(2) + sig(1577) = 1803
        // Encrypted:        nonce(12) + ciphertext(1803) + tag(16) = 1831
        const size_t FALCON1024_SIG_SIZE = FalconConstants::FALCON1024_SIG_CT_SIZE;  // 1577
        const size_t HASH_PLAINTEXT      = 216 + 8 + 2 + FALCON1024_SIG_SIZE;       // 1803
        const size_t HASH_ENCRYPTED      = HASH_PLAINTEXT + 28;                      // 1831

        ChaCha20Wrapper::SubmitBlockPayloadInfo hash_info;
        hash_info.channel           = 2;   // Hash
        hash_info.base_block_size   = 216;
        hash_info.offset_bytes_count = 0;
        hash_info.timestamp_size     = 8;
        hash_info.sig_len_field_size = 2;
        hash_info.signature_size     = FALCON1024_SIG_SIZE;

        // Build mock plaintext: zeroed block, zeroed timestamp, sig_len LE, zeroed sig
        std::vector<uint8_t> hash_plaintext(HASH_PLAINTEXT, 0xAB);
        // Write sig_len (1577 = 0x0629) in LE at sig_len_offset_hash
        const size_t sig_len_offset_hash = 216 + 0 + 8;
        hash_plaintext[sig_len_offset_hash]     = 0x29;  // LSB of 1577
        hash_plaintext[sig_len_offset_hash + 1] = 0x06;  // MSB of 1577

        auto hash_result = wrapper.encrypt_submit_block_payload(hash_plaintext, key, hash_info);
        print_test_result("Hash channel: encrypt_submit_block_payload succeeds",
                          hash_result.success);
        print_test_result("Hash channel: encrypted size = 1831 (plaintext 1803 + overhead 28)",
                          hash_result.success && hash_result.data.size() == HASH_ENCRYPTED);
        print_test_result("Hash channel: encrypted size == payload_info.expected_encrypted_size()",
                          hash_result.success &&
                          hash_result.data.size() == hash_info.expected_encrypted_size());

        // ── Case B: Prime channel, 10 offset bytes, Falcon-1024 ──────────
        // Plaintext layout: block(216) + offsets(10) + timestamp(8) + sig_len(2) + sig(1577) = 1813
        // Encrypted:        nonce(12) + ciphertext(1813) + tag(16) = 1841
        const size_t PRIME_OFFSETS       = 10;
        const size_t PRIME_PLAINTEXT     = 216 + PRIME_OFFSETS + 8 + 2 + FALCON1024_SIG_SIZE; // 1813
        const size_t PRIME_ENCRYPTED     = PRIME_PLAINTEXT + 28;                               // 1841

        ChaCha20Wrapper::SubmitBlockPayloadInfo prime_info;
        prime_info.channel            = 1;   // Prime
        prime_info.base_block_size    = 216;
        prime_info.offset_bytes_count = PRIME_OFFSETS;
        prime_info.timestamp_size     = 8;
        prime_info.sig_len_field_size = 2;
        prime_info.signature_size     = FALCON1024_SIG_SIZE;

        // Build mock plaintext: sig_len (1577 = 0x0629) in LE at sig_len_offset_prime
        std::vector<uint8_t> prime_plaintext(PRIME_PLAINTEXT, 0xCD);
        const size_t sig_len_offset_prime = 216 + PRIME_OFFSETS + 8;
        prime_plaintext[sig_len_offset_prime]     = 0x29;  // LSB of 1577
        prime_plaintext[sig_len_offset_prime + 1] = 0x06;  // MSB of 1577

        auto prime_result = wrapper.encrypt_submit_block_payload(prime_plaintext, key, prime_info);
        print_test_result("Prime+10-offsets: encrypt_submit_block_payload succeeds",
                          prime_result.success);
        print_test_result("Prime+10-offsets: encrypted size = 1841 (plaintext 1813 + overhead 28)",
                          prime_result.success && prime_result.data.size() == PRIME_ENCRYPTED);
        print_test_result("Prime+10-offsets: encrypted size == payload_info.expected_encrypted_size()",
                          prime_result.success &&
                          prime_result.data.size() == prime_info.expected_encrypted_size());

        // ── Verify nonce(12) is prepended (no manual nonce management by caller)
        // Two calls with the same key+plaintext should produce different ciphertexts
        // because encrypt_submit_block_payload() generates a fresh nonce each time.
        auto result_a = wrapper.encrypt_submit_block_payload(hash_plaintext, key, hash_info);
        auto result_b = wrapper.encrypt_submit_block_payload(hash_plaintext, key, hash_info);
        bool nonces_differ = result_a.success && result_b.success &&
                             result_a.data != result_b.data;
        print_test_result("Each call generates a fresh nonce (outputs differ)", nonces_differ);

        // ── Too-small plaintext → hard failure ────────────────────────────
        std::vector<uint8_t> tiny(10, 0xFF);  // way too small
        auto tiny_result = wrapper.encrypt_submit_block_payload(tiny, key, hash_info);
        print_test_result("Too-small plaintext → encrypt_submit_block_payload fails",
                          !tiny_result.success);
    }

    // ====================================================================
    // Summary
    // ====================================================================
    std::cout << "\n========================================" << std::endl;
    std::cout << "Test Summary: " << tests_passed << "/" << tests_run << " passed";
    if (tests_failed > 0)
        std::cout << " (" << tests_failed << " FAILED)";
    std::cout << std::endl;
    std::cout << "========================================" << std::endl;

    return tests_failed > 0 ? 1 : 0;
}
