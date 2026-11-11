#ifndef NEXUSMINER_PROTOCOL_FALCON_CONSTANTS_HPP
#define NEXUSMINER_PROTOCOL_FALCON_CONSTANTS_HPP

#include <cstddef>
#include <cstdint>

#include "mining/mining_constants.hpp"
#include "protocol/protocol_constants.hpp"

namespace nexusminer {
namespace protocol {

/**
 * @brief Falcon-512/1024 Protocol Constants for Stateless Mining
 *
 * These constants define size limits for the Falcon signature protocol
 * used in NexusMiner ↔ LLL-TAO Node communication.
 *
 * DEFAULT KEY MODE: Falcon-1024 (256-bit quantum security).
 * Falcon-512 constants are retained for backward-compatibility and opt-in testing only.
 *
 * IMPORTANT: All multi-byte integer fields use LITTLE-ENDIAN byte order
 * for consistency with the rest of the protocol (nonce, timestamp, etc.)
 *
 * STATELESS MINING RESPONSIBILITIES:
 * - Miner receives a compact block template (header metadata only) from the node.
 *   Templates are ~216 bytes (Tritium) or ~220 bytes (Legacy) — NOT 2 MB.
 * - Miner mines the template (finds a valid nonce/hash prevalidation payload).
 * - Miner submits the solved compact header + Falcon signature back to the node.
 * - Authoritative transaction assembly is performed exclusively by the node.
 *   The miner does NOT load, store, or transmit the transaction set.
 *
 * SIGNATURE ARCHITECTURE:
 * The miner uses a SINGLE Falcon key pair (the auth key) for ALL operations:
 * 1. Authentication - Signs challenge with auth private key during handshake.
 * 2. Block Submission - Signs block data with SAME auth private key.
 * 3. Node Verification - Node uses auth public key (from mapSessionKeys) to verify ALL signatures.
 *
 * This ensures signature verification succeeds because signing and verifying keys match.
 * The auth key is session-specific (generated fresh for each mining session) but is
 * used consistently throughout that session for both authentication and block signatures.
 *
 * Disposable Falcon signatures are ALWAYS ON and NOT stored on blockchain (0 bytes overhead).
 * Block signing does not store signatures on-chain.
 *
 * References:
 * - LLL-TAO: src/LLC/falcon/falcon.h (FALCON_SIG_VARTIME_MAXSIZE / FALCON_SIG_CT_SIZE)
 * - LLL-TAO: src/TAO/Ledger/include/constants.h (MAX_BLOCK_SIZE — node-side protocol limit)
 * - LLL-TAO: src/LLP/disposable_falcon.cpp (SignedWorkSubmission format)
 */
namespace FalconConstants {

    //==========================================================================
    // Falcon-512 Key Sizes (Fixed per Falcon Specification)
    //==========================================================================
    
    /** Falcon-512 public key size (fixed) */
    constexpr size_t FALCON512_PUBKEY_SIZE = 897;
    
    /** Falcon-512 private key size (fixed) */
    constexpr size_t FALCON512_PRIVKEY_SIZE = 1281;

    //==========================================================================
    // Falcon-1024 Key Sizes (Fixed per Falcon Specification)
    //==========================================================================
    
    /** Falcon-1024 public key size (fixed) */
    constexpr size_t FALCON1024_PUBKEY_SIZE = 1793;
    
    /** Falcon-1024 private key size (fixed) */
    constexpr size_t FALCON1024_PRIVKEY_SIZE = 2305;

    //==========================================================================
    // Falcon-512 Signature Sizes (Variable due to compression algorithm)
    //==========================================================================
    
    /** Minimum Falcon-512 signature size (typical lower bound) */
    constexpr size_t FALCON512_SIG_MIN = 600;
    
    /** Typical maximum for VARIABLE-TIME signatures (reference only)
     *  Per Falcon spec: FALCON_SIG_VARTIME_MAXSIZE(logn=9) = 752 bytes
     *  This is the maximum for variable-time signatures, but LLL-TAO uses constant-time mode. */
    constexpr size_t FALCON512_SIG_VARTIME_MAX = 752;
    
    /** Constant-Time Falcon-512 signature size (exact)
     *  Per Falcon spec: FALCON_SIG_CT_SIZE(logn=9) = 809 bytes
     *  LLL-TAO's FLKey::Sign() uses ct=1, producing exactly 809 bytes */
    constexpr size_t FALCON512_SIG_CT_SIZE = 809;
    
    /** Absolute maximum Falcon-512 signature size
     *  This is the CT size since LLL-TAO uses constant-time signing */
    constexpr size_t FALCON512_SIG_ABSOLUTE_MAX = 809;
    
    /** Legacy alias for backward compatibility */
    constexpr size_t FALCON512_SIG_MAX = FALCON512_SIG_ABSOLUTE_MAX;

    //==========================================================================
    // Falcon-1024 Signature Sizes (Variable due to compression algorithm)
    //==========================================================================
    
    /** Minimum Falcon-1024 signature size (typical lower bound) */
    constexpr size_t FALCON1024_SIG_MIN = 1100;
    
    /** Typical maximum for VARIABLE-TIME signatures (reference only)
     *  Per Falcon spec: FALCON_SIG_VARTIME_MAXSIZE(logn=10) = 1462 bytes
     *  This is the maximum for variable-time signatures, but LLL-TAO uses constant-time mode. */
    constexpr size_t FALCON1024_SIG_VARTIME_MAX = 1462;
    
    /** Constant-Time Falcon-1024 signature size (exact)
     *  Per Falcon spec: FALCON_SIG_CT_SIZE(logn=10) = 1577 bytes
     *  LLL-TAO's FLKey::Sign() uses ct=1, producing exactly 1577 bytes */
    constexpr size_t FALCON1024_SIG_CT_SIZE = 1577;
    
    /** Absolute maximum Falcon-1024 signature size
     *  This is the CT size since LLL-TAO uses constant-time signing */
    constexpr size_t FALCON1024_SIG_ABSOLUTE_MAX = 1577;
    
    /** Alias for consistency */
    constexpr size_t FALCON1024_SIG_MAX = FALCON1024_SIG_ABSOLUTE_MAX;

    //==========================================================================
    // Default Falcon Constants (Falcon-1024)
    // Use these in new code where both variants could theoretically be used
    // but Falcon-1024 is the required default deployment mode.
    //==========================================================================

    /** Default public key size — Falcon-1024 (1793 bytes) */
    constexpr size_t DEFAULT_PUBKEY_SIZE    = FALCON1024_PUBKEY_SIZE;

    /** Default private key size — Falcon-1024 (2305 bytes) */
    constexpr size_t DEFAULT_PRIVKEY_SIZE   = FALCON1024_PRIVKEY_SIZE;

    /** Default signature size (CT mode) — Falcon-1024 (1577 bytes) */
    constexpr size_t DEFAULT_SIG_CT_SIZE    = FALCON1024_SIG_CT_SIZE;

    /** Default maximum signature size — Falcon-1024 (1577 bytes) */
    constexpr size_t DEFAULT_SIG_MAX        = FALCON1024_SIG_ABSOLUTE_MAX;

    //==========================================================================
    // ChaCha20-Poly1305 AEAD Encryption Constants
    //==========================================================================
    
    /** ChaCha20-Poly1305 nonce size (prepended to ciphertext) */
    constexpr size_t CHACHA20_NONCE_SIZE = 12;
    
    /** ChaCha20-Poly1305 authentication tag size (appended to ciphertext) */
    constexpr size_t CHACHA20_AUTH_TAG_SIZE = 16;
    
    /** Total ChaCha20-Poly1305 overhead (nonce + auth tag) */
    constexpr size_t CHACHA20_OVERHEAD = CHACHA20_NONCE_SIZE + CHACHA20_AUTH_TAG_SIZE;  // 28 bytes

    //==========================================================================
    // Protocol Field Sizes
    //==========================================================================
    
    /** Merkle root size (uint512_t) */
    constexpr size_t MERKLE_ROOT_SIZE = 64;
    
    /** Nonce size (uint64_t, little-endian) */
    constexpr size_t NONCE_SIZE = 8;
    
    /** Timestamp size (uint64_t, little-endian) */
    constexpr size_t TIMESTAMP_SIZE = 8;
    
    /** Length field size for variable-length fields (2 bytes, little-endian) */
    constexpr size_t LENGTH_FIELD_SIZE = 2;
    
    /** Tritium GenesisHash size (uint256_t) */
    constexpr size_t GENESIS_HASH_SIZE = 32;

    //==========================================================================
    // Compact Block Header Sizes (Stateless Mining Protocol)
    //==========================================================================
    //
    // In stateless mining the miner receives a compact block template
    // (header metadata only) from the node and submits a solved compact
    // header back.  The node holds the full transaction set and assembles
    // the complete block after a successful submission.  The miner NEVER
    // handles 2 MB full blocks — the historical "2 MB max" values in this
    // section were an early beta assumption that has been corrected here.

    /** Compact Tritium block header size as seen by the miner (header only, no transactions).
     *  This is the payload the miner receives from the node as a mining template
     *  and what it includes when submitting a solved block.
     *  Value: 216 bytes (fixed Tritium header fields). */
    constexpr size_t FULL_BLOCK_TRITIUM_SIZE = 216;

    /** Compact Legacy block header size as seen by the miner (header only, no transactions).
     *  Value: 220 bytes (fixed Legacy header fields). */
    constexpr size_t FULL_BLOCK_LEGACY_SIZE = 220;

    /** Compact block header size (Phase-2 stateless mining protocol, legacy pool format) */
    constexpr size_t COMPACT_BLOCK_HEADER_SIZE = 92;

    /** Prime-channel vOffsets MAXIMUM size appended to the block payload.
     *  Format (per prime_validation.cpp GetOffsets()):
     *    (N - 1) × 1-byte chain offsets  (one per prime gap in a length-N chain)
     *    + 4 bytes fractional difficulty (uint32_t LE, always present)
     *  So a length-N chain serializes to (N - 1) + 4 bytes:
     *    chain length 2  -> 5 bytes  (minimum)
     *    chain length 7  -> 10 bytes
     *    chain length 8  -> 11 bytes
     *    chain length 19 -> 22 bytes (current upper bound, well past the
     *                                 Cunningham chain world record of 17)
     *  This MAX value drives the SUBMIT_BLOCK_WRAPPER_*_MAX upper bounds
     *  below.  At runtime the on-wire payload is variable length:
     *  ChaCha20Wrapper::SubmitBlockPayloadInfo::offset_bytes_count and
     *  StatelessBlockUtility::compute_submit_payload_info() derive the
     *  actual length from vOffsets.size() — no consumer of the wire format
     *  ever assumes this MAX value at runtime.
     *  Hash-channel submissions carry no vOffsets (empty vector — zero overhead).
     *
     *  Single source of truth lives in
     *  mining/mining_constants.hpp::mining::kMaxSerializedPrimeOffsets.
     *  This alias keeps the wire-format constant addressable inside the
     *  FalconConstants namespace without re-declaring the magic number. */
    constexpr size_t PRIME_VOFFSETS_MAX_SIZE = mining::kMaxSerializedPrimeOffsets;

    /** Deprecated alias retained for backward-compatibility with callers that
     *  still spell the old fixed-length name.  New code should use
     *  PRIME_VOFFSETS_MAX_SIZE.  Historically this was 10 (chain length 7),
     *  which silently dropped every length-8+ find. */
    constexpr size_t PRIME_VOFFSETS_SIZE = PRIME_VOFFSETS_MAX_SIZE;

    //==========================================================================
    // Submit Block Message (What Gets Signed)
    //==========================================================================
    
    /** OLD compact format size (merkle_root + nonce + timestamp)
     *  merkle_root(64) + nonce(8) + timestamp(8) = 80 bytes
     *  NOTE: Kept for reference, no longer used in SOLO mining after PR #65 */
    constexpr size_t SUBMIT_BLOCK_MESSAGE_COMPACT_SIZE = 
        MERKLE_ROOT_SIZE + NONCE_SIZE + TIMESTAMP_SIZE;  // 80 bytes
    
    /** Tritium compact-header format (compact_header + timestamp)
     *  compact_header(216) + timestamp(8) = 224 bytes
     *  Used for SOLO mining: miner submits compact solved header, node assembles full block. */
    constexpr size_t SUBMIT_BLOCK_MESSAGE_TRITIUM_SIZE = 
        FULL_BLOCK_TRITIUM_SIZE + TIMESTAMP_SIZE;  // 224 bytes
    
    /** Legacy compact-header format (compact_header + timestamp)
     *  compact_header(220) + timestamp(8) = 228 bytes
     *  Used for SOLO mining: miner submits compact solved header, node assembles full block. */
    constexpr size_t SUBMIT_BLOCK_MESSAGE_LEGACY_SIZE = 
        FULL_BLOCK_LEGACY_SIZE + TIMESTAMP_SIZE;  // 228 bytes
    
    /** Legacy alias for backward compatibility (deprecated, use SUBMIT_BLOCK_MESSAGE_TRITIUM_SIZE) */
    constexpr size_t SUBMIT_BLOCK_MESSAGE_SIZE = SUBMIT_BLOCK_MESSAGE_COMPACT_SIZE;  // 80 bytes (old)

    //==========================================================================
    // Submit Block Wrapper Sizes (Serialized Transmission)
    // Falcon-1024 signature size (1577 bytes CT) is used throughout because
    // Falcon-1024 is the default key mode.  Falcon-512 (809 bytes) is
    // supported for opt-in/testing only.
    //==========================================================================
    
    /** Submit Block wrapper - Tritium LOCALHOST (no encryption)
     *  Format: [compact_header(216)][vOffsets(<=22)][timestamp(8)][sig_len(2)][signature(1577 max)]
     *  vOffsets: up to PRIME_VOFFSETS_MAX_SIZE = 22 bytes for Prime channel
     *            (chain length up to 19 = 18 chain-gap bytes + 4 fraction bytes).
     *            Hash channel carries no vOffsets, so this MAX size also covers Hash
     *            (conservative upper bound).  Actual transmitted size is variable.
     *  Calculation: 216 + 22 + 8 + 2 + 1577 = 1825 bytes
     */
    constexpr size_t SUBMIT_BLOCK_WRAPPER_TRITIUM_MAX = 
        FULL_BLOCK_TRITIUM_SIZE + PRIME_VOFFSETS_MAX_SIZE + TIMESTAMP_SIZE + 
        LENGTH_FIELD_SIZE + FALCON1024_SIG_ABSOLUTE_MAX;  // 1825 bytes
    static_assert(SUBMIT_BLOCK_WRAPPER_TRITIUM_MAX == 1825, "SUBMIT_BLOCK_WRAPPER_TRITIUM_MAX size calculation mismatch");
    
    /** Submit Block wrapper - Tritium PUBLIC MINER (with ChaCha20 encryption)
     *  ChaCha20-Poly1305 overhead: nonce(12) + auth_tag(16) = 28 bytes
     *  Calculation: 1825 + 28 = 1853 bytes
     */
    constexpr size_t SUBMIT_BLOCK_WRAPPER_TRITIUM_ENCRYPTED_MAX = 
        SUBMIT_BLOCK_WRAPPER_TRITIUM_MAX + CHACHA20_OVERHEAD;  // 1853 bytes
    static_assert(SUBMIT_BLOCK_WRAPPER_TRITIUM_ENCRYPTED_MAX == 1853, "SUBMIT_BLOCK_WRAPPER_TRITIUM_ENCRYPTED_MAX size calculation mismatch");
    
    /** Submit Block wrapper - Legacy LOCALHOST (no encryption)
     *  Legacy (Hash channel) submissions carry no vOffsets.
     *  Format: [compact_header(220)][timestamp(8)][sig_len(2)][signature(1577 max)]
     *  Calculation: 220 + 8 + 2 + 1577 = 1807 bytes
     */
    constexpr size_t SUBMIT_BLOCK_WRAPPER_LEGACY_MAX = 
        FULL_BLOCK_LEGACY_SIZE + TIMESTAMP_SIZE + 
        LENGTH_FIELD_SIZE + FALCON1024_SIG_ABSOLUTE_MAX;  // 1807 bytes
    static_assert(SUBMIT_BLOCK_WRAPPER_LEGACY_MAX == 1807, "SUBMIT_BLOCK_WRAPPER_LEGACY_MAX size calculation mismatch");
    
    /** Submit Block wrapper - Legacy PUBLIC MINER (with ChaCha20 encryption)
     *  Calculation: 1807 + 28 = 1835 bytes
     */
    constexpr size_t SUBMIT_BLOCK_WRAPPER_LEGACY_ENCRYPTED_MAX = 
        SUBMIT_BLOCK_WRAPPER_LEGACY_MAX + CHACHA20_OVERHEAD;  // 1835 bytes
    static_assert(SUBMIT_BLOCK_WRAPPER_LEGACY_ENCRYPTED_MAX == 1835, "SUBMIT_BLOCK_WRAPPER_LEGACY_ENCRYPTED_MAX size calculation mismatch");
    
    /** Legacy aliases for backward compatibility (point to Tritium values) */
    constexpr size_t SUBMIT_BLOCK_WRAPPER_MAX = SUBMIT_BLOCK_WRAPPER_TRITIUM_MAX;
    constexpr size_t SUBMIT_BLOCK_WRAPPER_ENCRYPTED_MAX = SUBMIT_BLOCK_WRAPPER_TRITIUM_ENCRYPTED_MAX;

    //==========================================================================
    // Authentication Response Sizes
    // Falcon-1024 pubkey (1793 bytes) and signature (1577 bytes CT) used as
    // defaults.  Falcon-512 (897-byte pubkey, 809-byte sig) is supported for
    // compatibility but is not the expected deployment mode.
    //==========================================================================
    
    /** Auth response - LOCALHOST (no encryption on pubkey)
     *  pubkey_len(2) + pubkey(1793) + timestamp(8) + sig_len(2) + sig(1577) = 3382 bytes */
    constexpr size_t AUTH_RESPONSE_MAX = 
        LENGTH_FIELD_SIZE + FALCON1024_PUBKEY_SIZE + TIMESTAMP_SIZE + 
        LENGTH_FIELD_SIZE + FALCON1024_SIG_ABSOLUTE_MAX;  // 3382 bytes
    static_assert(AUTH_RESPONSE_MAX == 3382, "AUTH_RESPONSE_MAX size calculation mismatch");
    
    /** Auth response - PUBLIC MINER (ChaCha20 wrapped pubkey)
     *  pubkey_len(2) + wrapped_pubkey(1793+28) + timestamp(8) + sig_len(2) + sig(1577) = 3410 bytes */
    constexpr size_t AUTH_RESPONSE_ENCRYPTED_MAX = 
        LENGTH_FIELD_SIZE + FALCON1024_PUBKEY_SIZE + CHACHA20_OVERHEAD + 
        TIMESTAMP_SIZE + LENGTH_FIELD_SIZE + FALCON1024_SIG_ABSOLUTE_MAX;  // 3410 bytes
    static_assert(AUTH_RESPONSE_ENCRYPTED_MAX == 3410, "AUTH_RESPONSE_ENCRYPTED_MAX size calculation mismatch");
    
    /** Auth response with optional GenesisHash binding
     *  Add 32 bytes for Tritium genesis hash */
    constexpr size_t AUTH_RESPONSE_WITH_GENESIS_MAX = 
        AUTH_RESPONSE_ENCRYPTED_MAX + GENESIS_HASH_SIZE;  // 3442 bytes
    static_assert(AUTH_RESPONSE_WITH_GENESIS_MAX == 3442, "AUTH_RESPONSE_WITH_GENESIS_MAX size calculation mismatch");

    //==========================================================================
    // Block Size Limits (Reference — Node-Side Protocol Constants)
    // These values mirror LLL-TAO TAO::Ledger::constants.h and are provided
    // as reference only.  The stateless miner does NOT operate on 2 MB blocks;
    // it handles compact headers (see FULL_BLOCK_TRITIUM_SIZE / FULL_BLOCK_LEGACY_SIZE).
    //==========================================================================
    
    /** Node-side maximum block size in transit (2 MB) — reference only.
     *  The miner never sends or receives a full 2 MB block; this is the
     *  upper bound enforced by the node when assembling the complete block. */
    constexpr uint32_t MAX_BLOCK_SIZE = 1024 * 1024 * 2;  // 2,097,152 bytes
    
    /** Node-side maximum block size for generation (~1 MB) — reference only. */
    constexpr uint32_t MAX_BLOCK_SIZE_GEN = 1000000;  // 1,000,000 bytes
    
    /** Maximum contracts per transaction */
    constexpr uint32_t MAX_TRANSACTION_CONTRACTS = 100;

    //==========================================================================
    // Validation Helpers
    //==========================================================================
    
    /** Check if signature size is within valid Falcon-512 or Falcon-1024 range */
    constexpr bool is_valid_signature_size(size_t size) {
        return (size >= FALCON512_SIG_MIN && size <= FALCON512_SIG_ABSOLUTE_MAX) ||
               (size >= FALCON1024_SIG_MIN && size <= FALCON1024_SIG_ABSOLUTE_MAX);
    }
    
    /** Check if public key size matches Falcon-512 or Falcon-1024 */
    constexpr bool is_valid_pubkey_size(size_t size) {
        return size == FALCON512_PUBKEY_SIZE || size == FALCON1024_PUBKEY_SIZE;
    }
    
    /** Check if private key size matches Falcon-512 or Falcon-1024 */
    constexpr bool is_valid_privkey_size(size_t size) {
        return size == FALCON512_PRIVKEY_SIZE || size == FALCON1024_PRIVKEY_SIZE;
    }

} // namespace FalconConstants

} // namespace protocol
} // namespace nexusminer

#endif // NEXUSMINER_PROTOCOL_FALCON_CONSTANTS_HPP
