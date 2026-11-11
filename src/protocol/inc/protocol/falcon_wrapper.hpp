#ifndef NEXUSMINER_PROTOCOL_FALCON_WRAPPER_HPP
#define NEXUSMINER_PROTOCOL_FALCON_WRAPPER_HPP

#include <vector>
#include <cstdint>
#include <memory>
#include <string>
#include <chrono>
#include <atomic>
#include "spdlog/spdlog.h"

namespace nexusminer {
namespace protocol {

/**
 * @brief Unified Hybrid Falcon Signature Protocol Wrapper
 * 
 * This class provides a centralized, optimized interface for all Falcon-512
 * signature operations in the mining protocol. It supports:
 * 
 * - Authentication signatures (MINER_AUTH_RESPONSE protocol)
 * - Block signatures for enhanced validation
 * - Payload signatures for work verification
 * - Signature caching and reuse for performance optimization
 * - Thread-safe operations for multi-worker environments
 * 
 * KEY CONSISTENCY: This wrapper uses a SINGLE Falcon key pair (the auth key)
 * for ALL signature operations throughout the mining session:
 * 1. During authentication: Signs with auth private key, sends auth public key to node
 * 2. During block submission: Signs with SAME auth private key
 * 3. Node verification: Node uses auth public key (from mapSessionKeys) to verify ALL signatures
 * 
 * This ensures signature verification succeeds because the signing key and
 * verifying key are from the SAME key pair (no ephemeral/session keys for blocks).
 * 
 * The wrapper builds upon Phase 2 MINER_AUTH_RESPONSE protocol and aligns
 * with updated LLL-TAO node-side protocols for seamless integration.
 */
class FalconSignatureWrapper {
public:
    
    /**
     * @brief Signature operation type
     */
    enum class SignatureType {
        AUTHENTICATION,  // For MINER_AUTH_RESPONSE (session authentication)
        BLOCK,          // For block submission validation
        PAYLOAD         // For generic payload signing
    };
    
    /**
     * @brief Signature result structure
     */
    struct SignatureResult {
        bool success;
        std::vector<uint8_t> signature;
        std::string error_message;
        std::chrono::microseconds generation_time;
    };
    
    /**
     * @brief Constructor
     * @param pubkey Falcon public key (897 bytes for Falcon-512, 1793 bytes for Falcon-1024)
     * @param privkey Falcon private key (1281 bytes for Falcon-512, 2305 bytes for Falcon-1024)
     */
    FalconSignatureWrapper(const std::vector<uint8_t>& pubkey,
                          const std::vector<uint8_t>& privkey);
    
    /**
     * @brief Destructor - securely clears private key from memory
     */
    ~FalconSignatureWrapper();
    
    // Disable copy to prevent key duplication
    FalconSignatureWrapper(const FalconSignatureWrapper&) = delete;
    FalconSignatureWrapper& operator=(const FalconSignatureWrapper&) = delete;
    
    /**
     * @brief Sign authentication message for MINER_AUTH_RESPONSE
     * 
     * This is the primary authentication signature used in Phase 2 protocol.
     * Message format: address + timestamp (8 bytes LE)
     * 
     * @param address Miner's network address (IP address)
     * @param timestamp Unix timestamp (8 bytes, little-endian)
     * @return SignatureResult containing signature or error
     */
    SignatureResult sign_authentication(const std::string& address, 
                                       uint64_t timestamp);
    
    /**
     * @brief Sign a block for submission validation
     * 
     * Signs the complete block payload (merkle_root + nonce) to provide 
     * cryptographic proof of block authorship. Uses the SAME auth private key
     * that was used during authentication (whose public key was sent to the node
     * and is stored in mapSessionKeys). This ensures the node can verify the
     * signature using the auth public key it received during authentication.
     * 
     * @param block_data Block merkle root (64 bytes)
     * @param nonce Block nonce (8 bytes)
     * @return SignatureResult containing signature or error
     */
    SignatureResult sign_block(const std::vector<uint8_t>& block_data,
                              uint64_t nonce);
    
    /**
     * @brief Sign arbitrary payload data
     * 
     * Generic signature function for protocol extensions or custom
     * payload validation requirements. Uses the auth private key
     * (the same key used for authentication and block signatures).
     * The node can verify this signature using the auth public key
     * from mapSessionKeys.
     * 
     * @param payload Data to sign
     * @param type Signature type for logging/metrics
     * @return SignatureResult containing signature or error
     */
    SignatureResult sign_payload(const std::vector<uint8_t>& payload,
                                SignatureType type = SignatureType::PAYLOAD);
    
    /**
     * @brief Get public key
     * @return Reference to public key
     */
    const std::vector<uint8_t>& get_public_key() const { return m_pubkey; }
    
    /**
     * @brief Check if wrapper is properly initialized
     * @return true if keys are valid and wrapper is ready
     */
    bool is_valid() const { return m_initialized; }
    
    /**
     * @brief Get expected signature size based on key version
     * @return Signature size in bytes (809 for Falcon-512, 1577 for Falcon-1024)
     */
    size_t get_signature_size() const;
    
    /**
     * @brief Check if using Falcon-1024 (vs Falcon-512)
     * @return true if Falcon-1024, false if Falcon-512
     */
    bool is_falcon1024() const;
    
    /**
     * @brief Get performance statistics
     */
    struct PerformanceStats {
        uint64_t total_signatures;
        uint64_t auth_signatures;
        uint64_t block_signatures;
        uint64_t payload_signatures;
        uint64_t total_time_microseconds;
        uint64_t average_time_microseconds;
    };
    
    PerformanceStats get_stats() const;
    
    /**
     * @brief Reset performance statistics
     */
    void reset_stats();

private:
    
    /**
     * @brief Internal signature implementation
     * @param data Data to sign
     * @param type Signature type for metrics
     * @return SignatureResult
     */
    SignatureResult sign_internal(const std::vector<uint8_t>& data,
                                  SignatureType type);
    
    /**
     * @brief Validate key sizes
     * @return true if keys are valid Falcon-512 format
     */
    bool validate_keys() const;
    
    /**
     * @brief Log public key fingerprint for debugging
     * @param prefix Log prefix (e.g., "[Auth]" or "[Submit]")
     * @param message Additional context message
     */
    void log_key_fingerprint(const std::string& prefix, const std::string& message) const;
    
    // Member variables
    std::vector<uint8_t> m_pubkey;
    std::vector<uint8_t> m_privkey;
    bool m_initialized;
    
    // Logger
    std::shared_ptr<spdlog::logger> m_logger;
    
    // Thread-safe performance tracking (atomic for multi-worker safety)
    std::atomic<uint64_t> m_total_signatures;
    std::atomic<uint64_t> m_auth_signatures;
    std::atomic<uint64_t> m_block_signatures;
    std::atomic<uint64_t> m_payload_signatures;
    std::atomic<uint64_t> m_total_time_us;
};

} // namespace protocol
} // namespace nexusminer

#endif // NEXUSMINER_PROTOCOL_FALCON_WRAPPER_HPP
