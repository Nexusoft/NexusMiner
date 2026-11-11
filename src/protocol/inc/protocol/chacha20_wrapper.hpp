#ifndef NEXUSMINER_PROTOCOL_CHACHA20_WRAPPER_HPP
#define NEXUSMINER_PROTOCOL_CHACHA20_WRAPPER_HPP

#include <vector>
#include <cstdint>
#include <cstddef>
#include <string>
#include <memory>
#include "spdlog/spdlog.h"

namespace nexusminer {
namespace protocol {

/**
 * @brief ChaCha20 Encryption Wrapper for Falcon Public Key Protection
 * 
 * This class provides ChaCha20 encryption for protecting Falcon Public Keys
 * during the handshake process with the LLL-TAO Node. ChaCha20 encryption is:
 * - OPTIONAL for localhost mining (direct IPC communication)
 * - MANDATORY for public/remote mining (HTTPS/TLS communication)
 * 
 * The wrapper uses OpenSSL's EVP interface for ChaCha20-Poly1305 AEAD encryption.
 */
class ChaCha20Wrapper {
public:

    /**
     * @brief Channel-aware payload metadata for SUBMIT_BLOCK sizing.
     *
     * Computes expected plaintext and encrypted sizes from real inputs rather
     * than assuming a universal fixed Tritium payload size.
     *
     * Hash:   plaintext = base_block_size + timestamp_size + sig_len_field_size + signature_size
     * Prime:  plaintext = base_block_size + offset_bytes_count + timestamp_size + sig_len_field_size + signature_size
     * encrypted = plaintext + CHACHA20_OVERHEAD (nonce 12 + tag 16 = 28)
     */
    struct SubmitBlockPayloadInfo {
        uint32_t channel{0};              ///< 1 = Prime, 2 = Hash
        size_t   base_block_size{0};      ///< 216 for Tritium (empty block body)
        size_t   offset_bytes_count{0};   ///< 0 for Hash, variable for Prime (vOffsets.size())
        size_t   timestamp_size{8};       ///< always 8 (uint64_t LE)
        size_t   sig_len_field_size{2};   ///< always 2 (uint16_t LE)
        size_t   signature_size{0};       ///< parsed/actual Falcon signature length

        /// ChaCha20-Poly1305 overhead: nonce(12) + auth_tag(16)
        static constexpr size_t CHACHA20_OVERHEAD = 28;

        /// Expected plaintext size: block + offsets + timestamp + sig_len + sig
        size_t expected_plaintext_size() const {
            return base_block_size + offset_bytes_count
                 + timestamp_size + sig_len_field_size + signature_size;
        }

        /// Expected encrypted size: plaintext + ChaCha20 overhead
        size_t expected_encrypted_size() const {
            return expected_plaintext_size() + CHACHA20_OVERHEAD;
        }
    };

    /**
     * @brief Encryption/Decryption result structure
     */
    struct CryptoResult {
        bool success;
        std::vector<uint8_t> data;
        std::string error_message;
    };
    
    /**
     * @brief Constructor
     */
    ChaCha20Wrapper();
    
    /**
     * @brief Destructor
     */
    ~ChaCha20Wrapper();
    
    /**
     * @brief Encrypt data using ChaCha20-Poly1305
     * 
     * @param plaintext Data to encrypt
     * @param key 256-bit (32-byte) encryption key
     * @param nonce 96-bit (12-byte) nonce/IV
     * @param aad Additional authenticated data (optional)
     * @return CryptoResult containing ciphertext + tag or error
     */
    CryptoResult encrypt(const std::vector<uint8_t>& plaintext,
                        const std::vector<uint8_t>& key,
                        const std::vector<uint8_t>& nonce,
                        const std::vector<uint8_t>& aad = {});
    
    /**
     * @brief Decrypt data using ChaCha20-Poly1305
     * 
     * @param ciphertext Data to decrypt (includes 16-byte auth tag at end)
     * @param key 256-bit (32-byte) encryption key
     * @param nonce 96-bit (12-byte) nonce/IV
     * @param aad Additional authenticated data (must match encryption)
     * @return CryptoResult containing plaintext or error
     */
    CryptoResult decrypt(const std::vector<uint8_t>& ciphertext,
                        const std::vector<uint8_t>& key,
                        const std::vector<uint8_t>& nonce,
                        const std::vector<uint8_t>& aad = {});
    
    /**
     * @brief Canonical encryption path for SUBMIT_BLOCK payloads.
     *
     * Generates a fresh 12-byte nonce internally, validates the plaintext size
     * against the channel-aware @p payload_info, then encrypts with
     * ChaCha20-Poly1305 (empty AAD — matches node-side behaviour).
     *
     * Returned data layout: [nonce(12)][ciphertext(plaintext.size())][tag(16)]
     * No manual nonce management is required by the caller.
     *
     * A size mismatch between @p plaintext and payload_info.expected_plaintext_size()
     * is logged as an error but does NOT abort encryption — the node can
     * correlate a size-wrong rejection back to the original block.
     *
     * @param plaintext       Raw SUBMIT_BLOCK payload (extracted from wire frame)
     * @param session_key     32-byte ChaCha20 session key from login()
     * @param payload_info    Channel-aware sizing metadata from compute_submit_payload_info()
     * @return CryptoResult whose data is [nonce(12)][ciphertext][tag(16)], or
     *         a failure result if inputs are invalid or encryption fails.
     */
    CryptoResult encrypt_submit_block_payload(
        const std::vector<uint8_t>& plaintext,
        const std::vector<uint8_t>& session_key,
        const SubmitBlockPayloadInfo& payload_info);

    /**
     * @brief Wrap Falcon Public Key for handshake transmission
     * 
     * This is the primary use case: encrypt Falcon Public Key before sending
     * to LLL-TAO Node during initial handshake.
     * 
     * @param falcon_pubkey Falcon-512 public key (897 bytes)
     * @param session_key Derived session key for encryption
     * @param nonce Random nonce for this encryption
     * @return CryptoResult containing wrapped key or error
     */
    CryptoResult wrap_falcon_pubkey(const std::vector<uint8_t>& falcon_pubkey,
                                    const std::vector<uint8_t>& session_key,
                                    const std::vector<uint8_t>& nonce);
    
    /**
     * @brief Unwrap received Falcon Public Key (for testing/validation)
     * 
     * @param wrapped_pubkey Encrypted public key data
     * @param session_key Decryption key
     * @param nonce Nonce used during encryption
     * @return CryptoResult containing unwrapped key or error
     */
    CryptoResult unwrap_falcon_pubkey(const std::vector<uint8_t>& wrapped_pubkey,
                                      const std::vector<uint8_t>& session_key,
                                      const std::vector<uint8_t>& nonce);
    
    /**
     * @brief Generate random nonce for ChaCha20
     * 
     * @return 12-byte random nonce
     */
    static std::vector<uint8_t> generate_nonce();
    
    /**
     * @brief Generate random encryption key
     * 
     * @return 32-byte random key
     */
    static std::vector<uint8_t> generate_key();
    
    /**
     * @brief Check if ChaCha20 is available in OpenSSL
     * 
     * @return true if ChaCha20-Poly1305 is supported
     */
    static bool is_available();

private:
    
    // Logger
    std::shared_ptr<spdlog::logger> m_logger;
};

} // namespace protocol
} // namespace nexusminer

#endif // NEXUSMINER_PROTOCOL_CHACHA20_WRAPPER_HPP
