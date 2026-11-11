#include "protocol/chacha20_wrapper.hpp"
#include "protocol/falcon_constants.hpp"
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <openssl/err.h>
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <string>
#include <memory>

namespace nexusminer {
namespace protocol {

// ChaCha20-Poly1305 constants
constexpr size_t CHACHA20_KEY_SIZE = 32;    // 256 bits
constexpr size_t CHACHA20_NONCE_SIZE = 12;  // 96 bits
constexpr size_t CHACHA20_TAG_SIZE = 16;    // 128 bits
const std::vector<uint8_t> FALCON_PUBKEY_AAD{'F', 'A', 'L', 'C', 'O', 'N', '_', 'P', 'U', 'B', 'K', 'E', 'Y'};

// RAII deleter for EVP_CIPHER_CTX — guarantees cleanup on all exit paths,
// including exceptions thrown by std::vector::resize().
struct EvpCipherCtxDeleter {
    void operator()(EVP_CIPHER_CTX* ctx) const noexcept {
        if (ctx) EVP_CIPHER_CTX_free(ctx);
    }
};
using EvpCipherCtxPtr = std::unique_ptr<EVP_CIPHER_CTX, EvpCipherCtxDeleter>;

ChaCha20Wrapper::ChaCha20Wrapper()
    : m_logger(spdlog::get("logger"))
{
    if (!m_logger) {
        m_logger = spdlog::default_logger();
    }
}

ChaCha20Wrapper::~ChaCha20Wrapper()
{
}

bool ChaCha20Wrapper::is_available()
{
#if defined(OPENSSL_3_0_OR_LATER)
    // OpenSSL 3.0+ always has ChaCha20-Poly1305
    return true;
#else
    // OpenSSL 1.1.1 also has ChaCha20-Poly1305
    EvpCipherCtxPtr ctx(EVP_CIPHER_CTX_new());
    if (!ctx) return false;
    
    const EVP_CIPHER* cipher = EVP_chacha20_poly1305();
    bool available = (cipher != nullptr);
    
    return available;
#endif
}

std::vector<uint8_t> ChaCha20Wrapper::generate_nonce()
{
    std::vector<uint8_t> nonce(CHACHA20_NONCE_SIZE);
    if (RAND_bytes(nonce.data(), CHACHA20_NONCE_SIZE) != 1) {
        throw std::runtime_error("Failed to generate random nonce: OpenSSL RAND_bytes() failed");
    }
    return nonce;
}

std::vector<uint8_t> ChaCha20Wrapper::generate_key()
{
    std::vector<uint8_t> key(CHACHA20_KEY_SIZE);
    if (RAND_bytes(key.data(), CHACHA20_KEY_SIZE) != 1) {
        throw std::runtime_error("Failed to generate random key: OpenSSL RAND_bytes() failed");
    }
    return key;
}

ChaCha20Wrapper::CryptoResult ChaCha20Wrapper::encrypt(
    const std::vector<uint8_t>& plaintext,
    const std::vector<uint8_t>& key,
    const std::vector<uint8_t>& nonce,
    const std::vector<uint8_t>& aad)
{
    CryptoResult result;
    result.success = false;
    
    // Validate inputs
    if (key.size() != CHACHA20_KEY_SIZE) {
        result.error_message = "Invalid key size (expected 32 bytes)";
        m_logger->error("[ChaCha20] {}", result.error_message);
        return result;
    }
    
    if (nonce.size() != CHACHA20_NONCE_SIZE) {
        result.error_message = "Invalid nonce size (expected 12 bytes)";
        m_logger->error("[ChaCha20] {}", result.error_message);
        return result;
    }
    
    if (plaintext.empty()) {
        result.error_message = "Empty plaintext";
        m_logger->error("[ChaCha20] {}", result.error_message);
        return result;
    }
    
    // Create and initialize context (RAII — freed automatically on any exit path)
    EvpCipherCtxPtr ctx(EVP_CIPHER_CTX_new());
    if (!ctx) {
        result.error_message = "Failed to create cipher context";
        m_logger->error("[ChaCha20] {}", result.error_message);
        return result;
    }
    
    // Get ChaCha20-Poly1305 cipher
    const EVP_CIPHER* cipher = EVP_chacha20_poly1305();
    if (!cipher) {
        result.error_message = "ChaCha20-Poly1305 not available";
        m_logger->error("[ChaCha20] {}", result.error_message);
        return result;
    }
    
    // Initialize encryption
    if (EVP_EncryptInit_ex(ctx.get(), cipher, nullptr, key.data(), nonce.data()) != 1) {
        result.error_message = "Failed to initialize encryption";
        m_logger->error("[ChaCha20] {}", result.error_message);
        return result;
    }
    
    // Set AAD if provided
    int len = 0;
    if (!aad.empty()) {
        if (EVP_EncryptUpdate(ctx.get(), nullptr, &len, aad.data(), aad.size()) != 1) {
            result.error_message = "Failed to set AAD";
            m_logger->error("[ChaCha20] {}", result.error_message);
            return result;
        }
    }
    
    // Allocate output buffer (plaintext size + tag)
    result.data.resize(plaintext.size() + CHACHA20_TAG_SIZE);
    
    // Encrypt plaintext
    if (EVP_EncryptUpdate(ctx.get(), result.data.data(), &len, plaintext.data(), plaintext.size()) != 1) {
        result.error_message = "Encryption failed";
        m_logger->error("[ChaCha20] {}", result.error_message);
        return result;
    }
    int ciphertext_len = len;
    
    // Finalize encryption
    if (EVP_EncryptFinal_ex(ctx.get(), result.data.data() + len, &len) != 1) {
        result.error_message = "Encryption finalization failed";
        m_logger->error("[ChaCha20] {}", result.error_message);
        return result;
    }
    ciphertext_len += len;
    
    // Get authentication tag
    if (EVP_CIPHER_CTX_ctrl(ctx.get(), EVP_CTRL_AEAD_GET_TAG, CHACHA20_TAG_SIZE, 
                            result.data.data() + ciphertext_len) != 1) {
        result.error_message = "Failed to get authentication tag";
        m_logger->error("[ChaCha20] {}", result.error_message);
        return result;
    }
    
    // Resize to actual size (ciphertext + tag)
    result.data.resize(ciphertext_len + CHACHA20_TAG_SIZE);
    
    result.success = true;
    
    m_logger->debug("[ChaCha20] Encrypted {} bytes -> {} bytes (including tag)", 
                   plaintext.size(), result.data.size());
    
    return result;
}

ChaCha20Wrapper::CryptoResult ChaCha20Wrapper::decrypt(
    const std::vector<uint8_t>& ciphertext,
    const std::vector<uint8_t>& key,
    const std::vector<uint8_t>& nonce,
    const std::vector<uint8_t>& aad)
{
    CryptoResult result;
    result.success = false;
    
    // Validate inputs
    if (key.size() != CHACHA20_KEY_SIZE) {
        result.error_message = "Invalid key size (expected 32 bytes)";
        m_logger->error("[ChaCha20] {}", result.error_message);
        return result;
    }
    
    if (nonce.size() != CHACHA20_NONCE_SIZE) {
        result.error_message = "Invalid nonce size (expected 12 bytes)";
        m_logger->error("[ChaCha20] {}", result.error_message);
        return result;
    }
    
    if (ciphertext.size() < CHACHA20_TAG_SIZE) {
        result.error_message = "Ciphertext too short (must include 16-byte tag)";
        m_logger->error("[ChaCha20] {}", result.error_message);
        return result;
    }
    
    // Split ciphertext and tag
    assert(ciphertext.size() >= CHACHA20_TAG_SIZE);
    size_t ciphertext_len = ciphertext.size() - CHACHA20_TAG_SIZE;
    const auto* tag = ciphertext.data() + ciphertext_len;
    
    // Create and initialize context (RAII — freed automatically on any exit path)
    EvpCipherCtxPtr ctx(EVP_CIPHER_CTX_new());
    if (!ctx) {
        result.error_message = "Failed to create cipher context";
        m_logger->error("[ChaCha20] {}", result.error_message);
        return result;
    }
    
    // Get ChaCha20-Poly1305 cipher
    const EVP_CIPHER* cipher = EVP_chacha20_poly1305();
    if (!cipher) {
        result.error_message = "ChaCha20-Poly1305 not available";
        m_logger->error("[ChaCha20] {}", result.error_message);
        return result;
    }
    
    // Initialize decryption
    if (EVP_DecryptInit_ex(ctx.get(), cipher, nullptr, key.data(), nonce.data()) != 1) {
        result.error_message = "Failed to initialize decryption";
        m_logger->error("[ChaCha20] {}", result.error_message);
        return result;
    }
    
    // Set AAD if provided
    int len = 0;
    if (!aad.empty()) {
        if (EVP_DecryptUpdate(ctx.get(), nullptr, &len, aad.data(), aad.size()) != 1) {
            result.error_message = "Failed to set AAD";
            m_logger->error("[ChaCha20] {}", result.error_message);
            return result;
        }
    }
    
    // Allocate output buffer
    result.data.resize(ciphertext_len);
    
    // Decrypt ciphertext
    if (EVP_DecryptUpdate(ctx.get(), result.data.data(), &len, ciphertext.data(), ciphertext_len) != 1) {
        result.error_message = "Decryption failed";
        m_logger->error("[ChaCha20] {}", result.error_message);
        return result;
    }
    int plaintext_len = len;
    
    // Set expected tag
    if (EVP_CIPHER_CTX_ctrl(ctx.get(), EVP_CTRL_AEAD_SET_TAG, CHACHA20_TAG_SIZE, 
                            const_cast<uint8_t*>(tag)) != 1) {
        result.error_message = "Failed to set authentication tag";
        m_logger->error("[ChaCha20] {}", result.error_message);
        return result;
    }
    
    // Finalize decryption (verifies tag)
    if (EVP_DecryptFinal_ex(ctx.get(), result.data.data() + len, &len) != 1) {
        result.error_message = "Decryption finalization failed (authentication tag mismatch)";
        m_logger->error("[ChaCha20] {}", result.error_message);
        return result;
    }
    plaintext_len += len;
    
    // Resize to actual plaintext size
    result.data.resize(plaintext_len);
    
    result.success = true;
    
    m_logger->debug("[ChaCha20] Decrypted {} bytes -> {} bytes", 
                   ciphertext.size(), result.data.size());
    
    return result;
}

ChaCha20Wrapper::CryptoResult ChaCha20Wrapper::wrap_falcon_pubkey(
    const std::vector<uint8_t>& falcon_pubkey,
    const std::vector<uint8_t>& session_key,
    const std::vector<uint8_t>& nonce)
{
    if (!FalconConstants::is_valid_pubkey_size(falcon_pubkey.size())) {
        CryptoResult result;
        result.success = false;
        result.error_message = "Invalid Falcon public key size (expected 1793 bytes for Falcon-1024 or 897 bytes for Falcon-512)";
        m_logger->error("[ChaCha20] {}", result.error_message);
        return result;
    }
    
    m_logger->info("[ChaCha20] Wrapping Falcon public key ({} bytes)", falcon_pubkey.size());
    
    // Use "FALCON_PUBKEY" as AAD to bind encryption to this specific use case
    auto result = encrypt(falcon_pubkey, session_key, nonce, FALCON_PUBKEY_AAD);
    
    if (result.success) {
        m_logger->info("[ChaCha20] Successfully wrapped Falcon public key: {} bytes -> {} bytes",
                      falcon_pubkey.size(), result.data.size());
    }
    
    return result;
}

ChaCha20Wrapper::CryptoResult ChaCha20Wrapper::unwrap_falcon_pubkey(
    const std::vector<uint8_t>& wrapped_pubkey,
    const std::vector<uint8_t>& session_key,
    const std::vector<uint8_t>& nonce)
{
    m_logger->info("[ChaCha20] Unwrapping Falcon public key ({} bytes)", wrapped_pubkey.size());
    
    // Use same AAD as wrapping
    auto result = decrypt(wrapped_pubkey, session_key, nonce, FALCON_PUBKEY_AAD);
    
    if (result.success) {
        // Validate unwrapped key size — accept Falcon-1024 (1793) and Falcon-512 (897)
        if (!FalconConstants::is_valid_pubkey_size(result.data.size())) {
            result.success = false;
            result.error_message = "Unwrapped key has invalid size (expected 1793 bytes for Falcon-1024 or 897 bytes for Falcon-512)";
            m_logger->error("[ChaCha20] {}", result.error_message);
            result.data.clear();
        } else {
            m_logger->info("[ChaCha20] Successfully unwrapped Falcon public key: {} bytes",
                          result.data.size());
        }
    }
    
    return result;
}

ChaCha20Wrapper::CryptoResult ChaCha20Wrapper::encrypt_submit_block_payload(
    const std::vector<uint8_t>& plaintext,
    const std::vector<uint8_t>& session_key,
    const SubmitBlockPayloadInfo& payload_info)
{
    CryptoResult result;
    result.success = false;

    // Hard fail only when the plaintext doesn't contain even the block body.
    const size_t hard_min = payload_info.base_block_size + payload_info.offset_bytes_count;
    if (plaintext.size() < hard_min) {
        result.error_message =
            "SUBMIT_BLOCK plaintext too small for base block + offsets ("
            + std::to_string(plaintext.size()) + " < " + std::to_string(hard_min) + ")";
        m_logger->error("[ChaCha20] {}", result.error_message);
        return result;
    }

    // Log a warning if the Hash channel carries unexpected offset bytes
    if (payload_info.channel == 2 && payload_info.offset_bytes_count != 0) {
        m_logger->warn("[ChaCha20] Hash channel SUBMIT_BLOCK has {} offset bytes (expected 0)",
                       payload_info.offset_bytes_count);
    }

    // Read sig_len (2-byte LE uint16_t) at the channel-aware offset, if present.
    // For unsigned submissions the plaintext may end before the sig_len field;
    // in that case treat sig_len as 0 and the size-mismatch path below will log
    // the discrepancy without aborting encryption.
    const size_t sig_len_offset = payload_info.base_block_size
                                + payload_info.offset_bytes_count
                                + payload_info.timestamp_size;
    uint16_t sig_len = 0;
    if (plaintext.size() >= sig_len_offset + payload_info.sig_len_field_size) {
        sig_len = static_cast<uint16_t>(plaintext[sig_len_offset])
                | (static_cast<uint16_t>(plaintext[sig_len_offset + 1]) << 8);
    }

    const size_t expected_plaintext_size = payload_info.base_block_size
                                         + payload_info.offset_bytes_count
                                         + payload_info.timestamp_size
                                         + payload_info.sig_len_field_size
                                         + sig_len;
    const size_t expected_encrypted_size = expected_plaintext_size
                                         + SubmitBlockPayloadInfo::CHACHA20_OVERHEAD;

    // Log size mismatch as an error, but do NOT abort — the node can correlate
    // a size-based rejection back to the original block submission.
    if (plaintext.size() != expected_plaintext_size) {
        m_logger->error("[ChaCha20] SUBMIT_BLOCK plaintext size mismatch: "
                        "actual={} expected={} (channel={}, sig_len={})",
                        plaintext.size(), expected_plaintext_size,
                        payload_info.channel, sig_len);
    }

    // Generate a fresh nonce — callers never manage nonces for SUBMIT_BLOCK
    auto nonce = generate_nonce();

    // Encrypt with empty AAD — matches node-side LLC::DecryptPayloadChaCha20()
    auto enc = encrypt(plaintext, session_key, nonce, {});
    if (!enc.success || enc.data.empty()) {
        result.error_message = enc.error_message;
        m_logger->error("[ChaCha20] SUBMIT_BLOCK encryption failed: {}", result.error_message);
        return result;
    }

    // Assemble [nonce(12)][ciphertext(plaintext.size())][tag(16)]
    result.data.reserve(nonce.size() + enc.data.size());
    result.data.insert(result.data.end(), nonce.begin(), nonce.end());
    result.data.insert(result.data.end(), enc.data.begin(), enc.data.end());
    result.success = true;

    // Validate final assembled size
    if (result.data.size() != expected_encrypted_size) {
        m_logger->warn("[ChaCha20] SUBMIT_BLOCK encrypted size mismatch: "
                       "actual={} expected={}", result.data.size(), expected_encrypted_size);
    }

    m_logger->debug("[ChaCha20] SUBMIT_BLOCK encrypted: plaintext={} encrypted={} "
                    "(channel={}, sig_len={})",
                    plaintext.size(), result.data.size(),
                    payload_info.channel, sig_len);

    return result;
}

} // namespace protocol
} // namespace nexusminer
