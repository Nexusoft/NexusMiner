#include "protocol/falcon_wrapper.hpp"
#include "protocol/falcon_constants.hpp"
#include "../miner_keys.hpp"
#include <chrono>
#include <cstring>

namespace nexusminer {
namespace protocol {

FalconSignatureWrapper::FalconSignatureWrapper(const std::vector<uint8_t>& pubkey,
                                              const std::vector<uint8_t>& privkey)
    : m_pubkey(pubkey)
    , m_privkey(privkey)
    , m_initialized(false)
    , m_logger(spdlog::get("logger"))
    , m_total_signatures(0)
    , m_auth_signatures(0)
    , m_block_signatures(0)
    , m_payload_signatures(0)
    , m_total_time_us(0)
{
    if (!m_logger) {
        m_logger = spdlog::default_logger();
    }
    
    // Validate keys on construction
    if (validate_keys()) {
        m_initialized = true;
        std::string version = is_falcon1024() ? "Falcon-1024" : "Falcon-512";
        m_logger->info("[FalconWrapper] Initialized successfully ({})", version);
        m_logger->debug("[FalconWrapper]   - Public key: {} bytes", m_pubkey.size());
        m_logger->debug("[FalconWrapper]   - Private key: {} bytes", m_privkey.size());
        m_logger->debug("[FalconWrapper]   - Signature size: {} bytes (CT)", get_signature_size());
    } else {
        m_logger->error("[FalconWrapper] Initialization failed - invalid key sizes");
        m_logger->error("[FalconWrapper]   - Got: pubkey {} bytes, privkey {} bytes", 
                       m_pubkey.size(), m_privkey.size());
        m_logger->error("[FalconWrapper]   - Expected Falcon-512: pubkey {} bytes, privkey {} bytes", 
                       FalconConstants::FALCON512_PUBKEY_SIZE, FalconConstants::FALCON512_PRIVKEY_SIZE);
        m_logger->error("[FalconWrapper]   - Expected Falcon-1024: pubkey {} bytes, privkey {} bytes", 
                       FalconConstants::FALCON1024_PUBKEY_SIZE, FalconConstants::FALCON1024_PRIVKEY_SIZE);
    }
}

FalconSignatureWrapper::~FalconSignatureWrapper()
{
    // Securely clear private key from memory
    // Use volatile pointer to prevent compiler optimization
    if (!m_privkey.empty()) {
        volatile uint8_t* vptr = m_privkey.data();
        for (size_t i = 0; i < m_privkey.size(); ++i) {
            vptr[i] = 0;
        }
        m_logger->debug("[FalconWrapper] Private key securely cleared from memory");
    }
}

bool FalconSignatureWrapper::validate_keys() const
{
    // Support both Falcon-512 and Falcon-1024
    bool is_falcon512 = (m_pubkey.size() == FalconConstants::FALCON512_PUBKEY_SIZE) && 
                        (m_privkey.size() == FalconConstants::FALCON512_PRIVKEY_SIZE);
    
    bool is_falcon1024 = (m_pubkey.size() == FalconConstants::FALCON1024_PUBKEY_SIZE) && 
                         (m_privkey.size() == FalconConstants::FALCON1024_PRIVKEY_SIZE);
    
    return is_falcon512 || is_falcon1024;
}

bool FalconSignatureWrapper::is_falcon1024() const
{
    return m_pubkey.size() == FalconConstants::FALCON1024_PUBKEY_SIZE;
}

size_t FalconSignatureWrapper::get_signature_size() const
{
    if (is_falcon1024()) {
        return FalconConstants::FALCON1024_SIG_CT_SIZE;  // 1577 bytes
    } else {
        return FalconConstants::FALCON512_SIG_CT_SIZE;   // 809 bytes
    }
}

FalconSignatureWrapper::SignatureResult 
FalconSignatureWrapper::sign_authentication(const std::string& address, 
                                           uint64_t timestamp)
{
    m_logger->debug("[FalconWrapper] Signing authentication message");
    m_logger->debug("[FalconWrapper]   - Address: '{}'", address);
    m_logger->debug("[FalconWrapper]   - Timestamp: 0x{:016x}", timestamp);
    
    if (!m_initialized) {
        m_logger->error("[FalconWrapper] Cannot sign - wrapper not initialized");
        m_logger->error("[FalconWrapper] SERIALIZATION_STATE: FAILED - Keys not properly loaded");
        return SignatureResult{false, {}, "Wrapper not initialized", {}};
    }
    
    // Log auth key being used (first 16 bytes for identification)
    log_key_fingerprint("[Auth]", "Using Falcon auth key for authentication");
    
    // Build authentication message: address + timestamp (8 bytes LE)
    std::vector<uint8_t> auth_message;
    auth_message.insert(auth_message.end(), address.begin(), address.end());
    
    // Append timestamp (8 bytes, little-endian)
    for (int i = 0; i < 8; ++i) {
        auth_message.push_back((timestamp >> (i * 8)) & 0xFF);
    }
    
    m_logger->debug("[FalconWrapper]   - Auth message size: {} bytes", auth_message.size());
    
    // Enhanced diagnostics: Log serialization details for debugging handshake issues
    m_logger->debug("[FalconWrapper] SERIALIZATION_CHECK: Auth message structure:");
    m_logger->debug("[FalconWrapper]   - Address bytes: {} (offset 0-{})", 
                   address.size(), address.size() > 0 ? address.size() - 1 : 0);
    m_logger->debug("[FalconWrapper]   - Timestamp bytes: 8 (offset {}-{})", 
                   address.size(), address.size() + 7);
    
    // Log first and last few bytes of the message for debugging
    if (auth_message.size() >= 8) {
        m_logger->debug("[FalconWrapper]   - First 8 message bytes: {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x}",
            auth_message[0], auth_message[1], auth_message[2], auth_message[3],
            auth_message[4], auth_message[5], auth_message[6], auth_message[7]);
        
        size_t end_offset = auth_message.size() - 8;
        m_logger->debug("[FalconWrapper]   - Timestamp bytes (LE): {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x}",
            auth_message[end_offset], auth_message[end_offset + 1], 
            auth_message[end_offset + 2], auth_message[end_offset + 3],
            auth_message[end_offset + 4], auth_message[end_offset + 5], 
            auth_message[end_offset + 6], auth_message[end_offset + 7]);
    }
    
    // Sign the message
    auto result = sign_internal(auth_message, SignatureType::AUTHENTICATION);
    
    if (result.success) {
        m_auth_signatures.fetch_add(1, std::memory_order_relaxed);
        m_logger->info("[FalconWrapper] Authentication signature generated successfully");
        m_logger->debug("[FalconWrapper]   - Signature size: {} bytes", result.signature.size());
        m_logger->debug("[FalconWrapper]   - Generation time: {} μs", result.generation_time.count());
        
        // Enhanced diagnostics: Verify signature is within Falcon-512 expected range
        // Using shared validation helper from falcon_constants.hpp for consistency
        if (!FalconConstants::is_valid_signature_size(result.signature.size())) {
            m_logger->warn("[FalconWrapper] SYNC_WARNING: Signature size {} outside valid Falcon-512 range ({}-{})",
                result.signature.size(), FalconConstants::FALCON512_SIG_MIN, FalconConstants::FALCON512_SIG_ABSOLUTE_MAX);
        } else {
            m_logger->debug("[FalconWrapper] SYNC_OK: Signature size within valid Falcon-512 range");
        }
    } else {
        m_logger->error("[FalconWrapper] Authentication signature failed: {}", result.error_message);
        m_logger->error("[FalconWrapper] SERIALIZATION_STATE: FAILED during signing");
    }
    
    return result;
}

FalconSignatureWrapper::SignatureResult 
FalconSignatureWrapper::sign_block(const std::vector<uint8_t>& block_data,
                                  uint64_t nonce)
{
    m_logger->debug("[FalconWrapper] Signing block submission");
    m_logger->debug("[FalconWrapper]   - Block data: {} bytes", block_data.size());
    m_logger->debug("[FalconWrapper]   - Nonce: 0x{:016x}", nonce);
    
    if (!m_initialized) {
        m_logger->error("[FalconWrapper] Cannot sign - wrapper not initialized");
        return SignatureResult{false, {}, "Wrapper not initialized", {}};
    }
    
    // Log that we're using the SAME auth key as used during authentication
    log_key_fingerprint("[Submit]", "Using Falcon auth key for block signature (same key as authentication)");
    
    // Build block signature payload: block_data + nonce (8 bytes LE)
    std::vector<uint8_t> block_payload;
    block_payload.reserve(block_data.size() + 8);
    block_payload.insert(block_payload.end(), block_data.begin(), block_data.end());
    
    // Append nonce (8 bytes, little-endian)
    for (int i = 0; i < 8; ++i) {
        block_payload.push_back((nonce >> (i * 8)) & 0xFF);
    }
    
    m_logger->debug("[FalconWrapper]   - Block payload size: {} bytes", block_payload.size());
    
    // Sign the block payload
    auto result = sign_internal(block_payload, SignatureType::BLOCK);
    
    if (result.success) {
        m_block_signatures.fetch_add(1, std::memory_order_relaxed);
        m_logger->info("[FalconWrapper] Block signature generated successfully");
        m_logger->debug("[FalconWrapper]   - Signature size: {} bytes", result.signature.size());
        m_logger->debug("[FalconWrapper]   - Generation time: {} μs", result.generation_time.count());
    } else {
        m_logger->error("[FalconWrapper] Block signature failed: {}", result.error_message);
    }
    
    return result;
}

FalconSignatureWrapper::SignatureResult 
FalconSignatureWrapper::sign_payload(const std::vector<uint8_t>& payload,
                                    SignatureType type)
{
    m_logger->debug("[FalconWrapper] Signing generic payload");
    m_logger->debug("[FalconWrapper]   - Payload size: {} bytes", payload.size());
    
    if (!m_initialized) {
        m_logger->error("[FalconWrapper] Cannot sign - wrapper not initialized");
        return SignatureResult{false, {}, "Wrapper not initialized", {}};
    }
    
    auto result = sign_internal(payload, type);
    
    if (result.success) {
        m_payload_signatures.fetch_add(1, std::memory_order_relaxed);
        m_logger->info("[FalconWrapper] Payload signature generated successfully");
        m_logger->debug("[FalconWrapper]   - Signature size: {} bytes", result.signature.size());
        m_logger->debug("[FalconWrapper]   - Generation time: {} μs", result.generation_time.count());
    } else {
        m_logger->error("[FalconWrapper] Payload signature failed: {}", result.error_message);
    }
    
    return result;
}

FalconSignatureWrapper::SignatureResult 
FalconSignatureWrapper::sign_internal(const std::vector<uint8_t>& data,
                                     SignatureType type)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    
    SignatureResult result;
    result.success = false;
    
    // Enhanced diagnostics: Log signing operation context
    const char* type_str = (type == SignatureType::AUTHENTICATION) ? "AUTHENTICATION" :
                           (type == SignatureType::BLOCK) ? "BLOCK" : "PAYLOAD";
    m_logger->debug("[FalconWrapper] sign_internal: type={}, data_size={}", type_str, data.size());
    m_logger->debug("[FalconWrapper] Using auth private key (same key for all signature types)");
    
    // Validate input
    if (data.empty()) {
        result.error_message = "Cannot sign empty data";
        m_logger->error("[FalconWrapper] SYNC_ERROR: sign_internal received empty data");
        return result;
    }
    
    // Validate private key is still available
    if (m_privkey.empty()) {
        result.error_message = "Private key is empty";
        m_logger->error("[FalconWrapper] SYNC_ERROR: Private key was cleared or never set");
        return result;
    }
    
    // Enhanced diagnostics: Log key status before signing
    m_logger->debug("[FalconWrapper] SYNC_STATE: Private key size: {} bytes", m_privkey.size());
    
    // Use the miner_keys module for actual signature generation
    // IMPORTANT: Always uses m_privkey (auth private key) for ALL signature types
    // This ensures the node can verify signatures using the auth public key from mapSessionKeys
    if (!keys::falcon_sign(m_privkey, data, result.signature)) {
        result.error_message = "Falcon signature generation failed";
        result.success = false;
        m_logger->error("[FalconWrapper] SYNC_ERROR: falcon_sign() returned false");
        m_logger->error("[FalconWrapper]   - Input data size: {} bytes", data.size());
        m_logger->error("[FalconWrapper]   - Private key size: {} bytes", m_privkey.size());
    } else {
        result.success = true;
        m_total_signatures.fetch_add(1, std::memory_order_relaxed);
        m_logger->debug("[FalconWrapper] SYNC_OK: Signature generated, size={}", result.signature.size());
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.generation_time = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time);
    
    m_total_time_us.fetch_add(result.generation_time.count(), std::memory_order_relaxed);
    
    return result;
}

void FalconSignatureWrapper::log_key_fingerprint(const std::string& prefix, 
                                                 const std::string& message) const
{
    if (m_pubkey.size() >= 16) {
        m_logger->info("{} {}", prefix, message);
        m_logger->info("{} Auth public key (first 16 bytes): {:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
                      prefix,
                      m_pubkey[0], m_pubkey[1], m_pubkey[2], m_pubkey[3],
                      m_pubkey[4], m_pubkey[5], m_pubkey[6], m_pubkey[7],
                      m_pubkey[8], m_pubkey[9], m_pubkey[10], m_pubkey[11],
                      m_pubkey[12], m_pubkey[13], m_pubkey[14], m_pubkey[15]);
    }
}

FalconSignatureWrapper::PerformanceStats FalconSignatureWrapper::get_stats() const
{
    PerformanceStats stats;
    stats.total_signatures = m_total_signatures.load(std::memory_order_relaxed);
    stats.auth_signatures = m_auth_signatures.load(std::memory_order_relaxed);
    stats.block_signatures = m_block_signatures.load(std::memory_order_relaxed);
    stats.payload_signatures = m_payload_signatures.load(std::memory_order_relaxed);
    stats.total_time_microseconds = m_total_time_us.load(std::memory_order_relaxed);
    stats.average_time_microseconds = stats.total_signatures > 0 
        ? stats.total_time_microseconds / stats.total_signatures 
        : 0;
    
    return stats;
}

void FalconSignatureWrapper::reset_stats()
{
    m_logger->debug("[FalconWrapper] Resetting performance statistics");
    m_total_signatures.store(0, std::memory_order_relaxed);
    m_auth_signatures.store(0, std::memory_order_relaxed);
    m_block_signatures.store(0, std::memory_order_relaxed);
    m_payload_signatures.store(0, std::memory_order_relaxed);
    m_total_time_us.store(0, std::memory_order_relaxed);
}

} // namespace protocol
} // namespace nexusminer
