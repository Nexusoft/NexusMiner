#ifndef NEXUSMINER_LLP_MINER_OPCODES_HPP
#define NEXUSMINER_LLP_MINER_OPCODES_HPP

/**
 * @file miner_opcodes.hpp
 * @brief Centralized LLP packet header opcodes for NexusMiner
 * 
 * This file defines all LLP packet opcodes used by NexusMiner, synchronized
 * with LLL-TAO's Phase 2 stateless miner implementation (src/LLP/types/miner.h).
 * 
 * IMPORTANT: These values MUST match the node implementation exactly for protocol
 * compatibility. Any changes should be coordinated with LLL-TAO repository updates.
 * 
 * Protocol Documentation: See docs/mining-llp-protocol.md for detailed flow
 */

#include <cstdint>

namespace nexusminer
{
namespace LLP
{

/**
 * LLP Packet Header Opcodes (Legacy uint8_t opcodes)
 * 
 * These opcodes define the message types in the Lower Level Protocol (LLP)
 * used for communication between NexusMiner and LLL-TAO mining servers.
 * 
 * NOTE: These are legacy uint8_t opcodes. For new stateless mining protocol,
 * see StatelessMining namespace below with uint16_t opcodes (0xD000+ range).
 */
enum MinerOpcodes : std::uint8_t
{
    // ============================================================================
    // DATA PACKETS (0-127)
    // These packets carry payload data and include a 4-byte length field
    // ============================================================================
    
    /** Block template data from node to miner */
    BLOCK_DATA = 0,
    
    /** Submit solved block from miner to node */
    SUBMIT_BLOCK = 1,
    
    /** Current blockchain height notification */
    BLOCK_HEIGHT = 2,
    
    /** Set mining channel (1=Prime, 2=Hash) */
    SET_CHANNEL = 3,
    
    /** Block reward information */
    BLOCK_REWARD = 4,
    
    /** Set coinbase address for rewards */
    SET_COINBASE = 5,
    
    /** Good block notification (valid but not best) */
    GOOD_BLOCK = 6,
    
    /** Orphaned block notification */
    ORPHAN_BLOCK = 7,
    
    // ============================================================================
    // POOL-SPECIFIC PACKETS (8-13)
    // These are NexusMiner extensions for pool mining compatibility
    // NOT used in solo stateless mining to LLL-TAO nodes
    // ============================================================================
    
    /** Pool login request (pool-only) */
    LOGIN = 8,
    
    /** Hashrate reporting (pool-only) */
    HASHRATE = 9,
    
    /** Work assignment from pool (pool-only) */
    WORK = 10,
    
    /** Pool login success v2 (pool-only) */
    LOGIN_V2_SUCCESS = 11,
    
    /** Pool login failure v2 (pool-only) */
    LOGIN_V2_FAIL = 12,
    
    /** Pool notification message (pool-only) */
    POOL_NOTIFICATION = 13,
    
    // ============================================================================
    // DATA REQUESTS (64-127)
    // Request packets with data payload
    // ============================================================================
    
    /** Request block validation */
    CHECK_BLOCK = 64,
    
    /** Subscribe to block notifications */
    SUBSCRIBE = 65,
    
    // ============================================================================
    // REQUEST PACKETS (128-199)
    // These are command packets with no payload (length = 0)
    // ============================================================================
    
    /** Request new block template */
    GET_BLOCK = 129,
    
    /** Request current blockchain height */
    GET_HEIGHT = 130,
    
    /** Request current block reward */
    GET_REWARD = 131,
    
    // ============================================================================
    // SERVER COMMANDS (132-135)
    // Commands from node to miner, or legacy pool commands
    // ============================================================================
    
    /** Clear miner's block cache (from LLL-TAO) */
    CLEAR_MAP = 132,
    
    /** Get current mining round info (from LLL-TAO) */
    GET_ROUND = 133,
    
    // ============================================================================
    // POOL-ONLY LEGACY OPCODES (overlapping values - use with caution!)
    // These legacy pool opcodes reuse the same numeric values as CLEAR_MAP/GET_ROUND
    // They are only valid when communicating with pool servers, NOT LLL-TAO nodes.
    // The numeric overlap is intentional for backward compatibility with pools.
    // ============================================================================
    
    /** Get payout info (pool-only, overlaps CLEAR_MAP) */
    GET_PAYOUT = 132,
    
    /** Get hashrate info (pool-only, overlaps GET_ROUND) */
    GET_HASHRATE = 133,
    
    /** Pool login success (legacy pool-only) */
    LOGIN_SUCCESS = 134,
    
    /** Pool login failure (legacy pool-only) */
    LOGIN_FAIL = 135,
    
    // ============================================================================
    // RESPONSE PACKETS (200-206)
    // Acknowledgments and status responses from node
    // ============================================================================
    
    /** Block accepted by node */
    ACCEPT = 200,
    BLOCK_ACCEPTED = 200,  // Alias for clarity
    
    /** Block rejected by node */
    REJECT = 201,
    BLOCK_REJECTED = 201,  // Alias for clarity
    
    /** Coinbase address set successfully */
    COINBASE_SET = 202,
    
    /** Coinbase address setting failed */
    COINBASE_FAIL = 203,
    
    // Legacy pool aliases (overlap with COINBASE_SET/FAIL)
    /** Block submitted (legacy pool, overlaps COINBASE_SET) */
    BLOCK = 202,
    
    /** Stale block (legacy pool, overlaps COINBASE_FAIL) */
    STALE = 203,
    
    /** New mining round started */
    NEW_ROUND = 204,
    
    /** Old/stale round */
    OLD_ROUND = 205,
    
    /** Channel selection acknowledged */
    CHANNEL_ACK = 206,
    
    // ============================================================================
    // FALCON AUTHENTICATION PACKETS (207-210)
    // Phase 2 stateless miner authentication using Falcon-512 signatures
    // These implement the challenge-response auth handshake
    // ============================================================================
    
    /**
     * MINER_AUTH_INIT: Miner initiates authentication
     * Direction: Miner -> Node
     * Payload (big-endian):
     *   [pubkey_len(2)] [pubkey bytes] [miner_id_len(2)] [miner_id string]
     */
    MINER_AUTH_INIT = 207,
    
    /**
     * MINER_AUTH_CHALLENGE: Node sends authentication challenge
     * Direction: Node -> Miner
     * Payload (big-endian):
     *   [nonce_len(2)] [nonce bytes]
     */
    MINER_AUTH_CHALLENGE = 208,
    
    /**
     * MINER_AUTH_RESPONSE: Miner sends signed challenge response
     * Direction: Miner -> Node
     * Payload (big-endian):
     *   [sig_len(2)] [signature bytes]
     */
    MINER_AUTH_RESPONSE = 209,
    
    /**
     * MINER_AUTH_RESULT: Node sends authentication result
     * Direction: Node -> Miner
     * Payload:
     *   [status(1)] [session_id(4, optional, little-endian)]
     *   status: 0x01 = success, 0x00 = failure
     */
    MINER_AUTH_RESULT = 210,
    
    // Aliases for backward compatibility
    MINER_AUTH_OK = 210,
    MINER_AUTH_FAIL = 210,  // Same opcode, status byte differentiates
    
    // ============================================================================
    // SESSION MANAGEMENT (211-212)
    // Phase 2 session handling (future use)
    // ============================================================================
    
    /** Start new mining session */
    SESSION_START = 211,
    
    /** Keep session alive (heartbeat) */
    SESSION_KEEPALIVE = 212,

    /**
     * SESSION_EXPIRED: Node notifies miner that session has expired
     * Direction: Node -> Miner
     * Payload: 5 bytes
     *   [0-3] session_id (uint32_t, little-endian) - session that expired
     *   [4]   reason_code (uint8_t) - 0x01 = EXPIRED_INACTIVITY
     * Miner action:
     *   - Clear local session state
     *   - Stop workers
     *   - Exponential backoff then re-authenticate on same connection
     */
    SESSION_EXPIRED = 221,

    // ============================================================================
    // STATELESS MINING REWARD BINDING (213-214)
    // Phase 2 encrypted reward address binding after Falcon auth
    // ============================================================================
    
    /**
     * MINER_SET_REWARD: Miner sends encrypted reward address
     * Direction: Miner -> Node
     * Payload (ChaCha20 encrypted):
     *   [encrypted_address(32)] - Decrypted to 32-byte register address
     * NOTE: Must be sent AFTER successful MINER_AUTH_RESULT establishes ChaCha20
     */
    MINER_SET_REWARD = 213,
    
    /**
     * MINER_REWARD_RESULT: Node sends reward binding result
     * Direction: Node -> Miner
     * Payload (ChaCha20 encrypted):
     *   [status(1)] [msg_len(1, optional)] [message(variable, optional)]
     *   status: 0x01 = success, 0x00 = failure
     */
    MINER_REWARD_RESULT = 214,
    
    // ============================================================================
    // PUSH NOTIFICATIONS (216-218)
    // LLL-TAO PR #156: Event-driven block notifications (replaces polling)
    // ============================================================================
    
    /**
     * MINER_READY: Miner subscribes to push notifications
     * Direction: Miner -> Node
     * Payload: None (header-only)
     * Requirements:
     *   - Must be sent AFTER successful authentication
     *   - Must be sent AFTER SET_CHANNEL (1=Prime or 2=Hash)
     *   - Stake channel (0) is REJECTED
     * Response:
     *   - Immediate PRIME_BLOCK_AVAILABLE or HASH_BLOCK_AVAILABLE
     *   - Then pushed on every block validation
     */
    MINER_READY = 216,
    
    /**
     * PRIME_BLOCK_AVAILABLE: Node notifies Prime miners of new block
     * Direction: Node -> Miner (Prime channel only)
     * Payload: 148 bytes (big-endian) — new full-picture format:
     *   [0-3]    unified_height (uint32)
     *   [4-7]    prime_height (uint32)     — miner's channel
     *   [8-11]   difficulty (uint32)
     *   [12-15]  hash_height (uint32)      — other PoW channel (NEW)
     *   [16-19]  stake_height (uint32)     — Stake channel (NEW)
     *   [20-147] hashBestChain (uint1024, 128 bytes LE)
     * Backward-compat: 12-byte compact and 140-byte v1 extended still accepted.
     * Triggered:
     *   - Immediately after MINER_READY
     *   - On every Prime block validation
     */
    PRIME_BLOCK_AVAILABLE = 217,
    
    /**
     * HASH_BLOCK_AVAILABLE: Node notifies Hash miners of new block
     * Direction: Node -> Miner (Hash channel only)
     * Payload: 148 bytes (big-endian) — new full-picture format:
     *   [0-3]    unified_height (uint32)
     *   [4-7]    hash_height (uint32)      — miner's channel
     *   [8-11]   difficulty (uint32)
     *   [12-15]  prime_height (uint32)     — other PoW channel (NEW)
     *   [16-19]  stake_height (uint32)     — Stake channel (NEW)
     *   [20-147] hashBestChain (uint1024, 128 bytes LE)
     * Backward-compat: 12-byte compact and 140-byte v1 extended still accepted.
     * Triggered:
     *   - Immediately after MINER_READY
     *   - On every Hash block validation
     */
    HASH_BLOCK_AVAILABLE = 218,
    
    // ============================================================================
    // GENERIC PACKETS (253-254)
    // Protocol-level control messages
    // ============================================================================
    
    /** Keepalive ping */
    PING = 253,
    
    /** Close connection */
    CLOSE = 254,

    /**
     * NODE_SHUTDOWN: Node sends graceful shutdown notice to miner
     * Direction: Node -> Miner
     * Payload: 1 byte (shutdown reason)
     *   [0] uint8_t reason:
     *     0x01 = GRACEFUL   (normal shutdown)
     *     0x02 = MAINTENANCE (planned maintenance window)
     * Miner action:
     *   - Stop all workers cleanly
     *   - Wait NODE_SHUTDOWN_BACKOFF_S (60s) before reconnecting
     */
    NODE_SHUTDOWN = 255
};

// ============================================================================
// MIRROR-MAPPED STATELESS MINING PROTOCOL (LLL-TAO PR #198)
// Mirror-mapped uint16_t opcodes: 0xD000 | legacyOpcode
// ============================================================================

/**
 * @brief Helper function to convert legacy uint8_t opcode to mirror-mapped uint16_t stateless opcode
 * @param legacy_opcode The legacy uint8_t opcode (e.g., 216 for MINER_READY)
 * @return Mirror-mapped stateless opcode (e.g., 0xD0D8 for MINER_READY)
 * 
 * Mirror-mapping formula: statelessOpcode = 0xD000 | legacyOpcode
 * Examples:
 *   - MINER_READY (216 = 0xD8) → 0xD0D8
 *   - GET_BLOCK (129 = 0x81) → 0xD081
 *   - SUBMIT_BLOCK (1 = 0x01) → 0xD001
 */
constexpr uint16_t MirrorOpcode(uint8_t legacy_opcode) {
    return 0xD000 | static_cast<uint16_t>(legacy_opcode);
}

/**
 * @brief Check if a uint16_t opcode is a stateless mining opcode
 * @param opcode The opcode to check
 * @return true if opcode is in the stateless range (0xD000-0xD0FF)
 * 
 * Stateless opcodes are mirror-mapped from legacy uint8_t opcodes, so they
 * always fall in the range 0xD000-0xD0FF (since legacy opcodes are 0x00-0xFF).
 */
constexpr bool IsStatelessOpcode(uint16_t opcode) {
    return (opcode >= 0xD000) && (opcode <= 0xD0FF);
}

/**
 * @brief Extract the legacy opcode from a mirror-mapped stateless opcode
 * @param stateless_opcode The mirror-mapped stateless opcode (e.g., 0xD0D8)
 * @return The legacy uint8_t opcode (e.g., 216 for 0xD0D8)
 * 
 * This is the inverse of MirrorOpcode(): legacyOpcode = statelessOpcode & 0xFF
 */
constexpr uint8_t UnmirrorOpcode(uint16_t stateless_opcode) {
    return static_cast<uint8_t>(stateless_opcode & 0xFF);
}

/**
 * @namespace StatelessMining
 * @brief MIRROR-MAPPED stateless mining protocol opcodes (uint16_t, 0xD000-0xD0FF range)
 * 
 * This namespace contains the opcodes for the modern stateless mining protocol
 * implemented in LLL-TAO PR #198. These opcodes are MIRROR-MAPPED from legacy
 * uint8_t opcodes using the formula: statelessOpcode = 0xD000 | legacyOpcode
 * 
 * KEY FEATURES:
 * - Mirror-mapped from legacy opcodes (not sequential!)
 * - Push notifications (no polling!)
 * - 228-byte templates (12-byte metadata + 216-byte block)
 * - Immediate template delivery after MINER_READY
 * - NEW_BLOCK push updates when blockchain advances
 * 
 * WIRE FORMAT:
 * [header(2 bytes, big-endian)][length(4 bytes, big-endian)][payload]
 * 
 * IMPORTANT: Port-based lane separation
 * - Stateless protocol is used on the stateless mining port
 * - Legacy protocol is used on the legacy mining port
 * - NO fallback between modes - strict separation enforced
 */
namespace StatelessMining {
    
    // ========================================================================
    // MIRROR-MAPPED OPCODES (using 0xD000 | legacyOpcode)
    // ========================================================================
    // Legacy opcode → Mirror-mapped stateless opcode
    // SUBMIT_BLOCK (1) → 0xD001
    // SET_CHANNEL (3) → 0xD003
    // GET_BLOCK (129) → 0xD081
    // BLOCK_ACCEPTED (200) → 0xD0C8
    // BLOCK_REJECTED (201) → 0xD0C9
    // MINER_SET_REWARD (213) → 0xD0D5
    // MINER_REWARD_RESULT (214) → 0xD0D6
    // MINER_READY (216) → 0xD0D8
    // PRIME_BLOCK_AVAILABLE (217) → 0xD0D9
    // HASH_BLOCK_AVAILABLE (218) → 0xD0DA
    
    /**
     * BLOCK_DATA: Node sends block template to miner
     * Direction: Node → Miner
     * Payload: 216 bytes (serialized Tritium block template)
     * Mirror-mapped from legacy BLOCK_DATA (0) → 0xD000
     * 
     * Triggered:
     * - As response to GET_BLOCK (0xD081)
     * - After MINER_READY subscription (push notification)
     * - When blockchain advances (new block available)
     */
    constexpr uint16_t BLOCK_DATA = MirrorOpcode(LLP::BLOCK_DATA);  // 0xD000
    
    /**
     * SUBMIT_BLOCK: Miner submits solved block
     * Direction: Miner → Node
     * Payload: 216 bytes (solved block)
     * Mirror-mapped from legacy SUBMIT_BLOCK (1) → 0xD001
     */
    constexpr uint16_t SUBMIT_BLOCK = MirrorOpcode(LLP::SUBMIT_BLOCK);  // 0xD001
    
    /**
     * SET_CHANNEL: Miner sets mining channel
     * Direction: Miner → Node
     * Payload: [channel(1)]  // 1=Prime, 2=Hash
     * Mirror-mapped from legacy SET_CHANNEL (3) → 0xD003
     */
    constexpr uint16_t SET_CHANNEL = MirrorOpcode(LLP::SET_CHANNEL);  // 0xD003
    
    /**
     * GET_BLOCK: Node sends mining template (PUSH!)
     * Direction: Node → Miner
     * Payload: 228 bytes (12-byte metadata + 216-byte block)
     * 
     * Metadata (12 bytes, big-endian):
     *   [0-3]   nUnifiedHeight  - Overall blockchain height
     *   [4-7]   nChannelHeight  - Channel-specific height
     *   [8-11]  nDifficulty     - Mining target (nBits)
     * 
     * Block (216 bytes):
     *   Full serialized block template for mining
     * 
     * Mirror-mapped from legacy GET_BLOCK (129) → 0xD081
     * 
     * Triggered:
     * - Immediately after MINER_READY (no polling needed!)
     * - When blockchain advances (push notification)
     */
    constexpr uint16_t GET_BLOCK = MirrorOpcode(LLP::GET_BLOCK);  // 0xD081
    
    /**
     * BLOCK_ACCEPTED: Node accepts submitted block
     * Direction: Node → Miner
     * Payload: None or [height(4)][hash(32)] (optional)
     * Mirror-mapped from legacy BLOCK_ACCEPTED (200) → 0xD0C8
     */
    constexpr uint16_t BLOCK_ACCEPTED = MirrorOpcode(LLP::BLOCK_ACCEPTED);  // 0xD0C8
    // Legacy-compatible stateless acceptance opcode used by some node builds.
    constexpr uint16_t BLOCK_ACCEPTED_COMPAT = 0xD002;
    
    /**
     * BLOCK_REJECTED: Node rejects submitted block
     * Direction: Node → Miner
     * Payload: [reason(1)]
     * Mirror-mapped from legacy BLOCK_REJECTED (201) → 0xD0C9
     */
    constexpr uint16_t BLOCK_REJECTED = MirrorOpcode(LLP::BLOCK_REJECTED);  // 0xD0C9
    // Legacy-compatible stateless rejection opcode used by some node builds.
    constexpr uint16_t BLOCK_REJECTED_COMPAT = 0xD003;
    
    /**
     * MINER_SET_REWARD: Miner sends encrypted reward address
     * Direction: Miner → Node
     * Payload (ChaCha20 encrypted): [encrypted_address(32)]
     * Mirror-mapped from legacy MINER_SET_REWARD (213) → 0xD0D5
     */
    constexpr uint16_t MINER_SET_REWARD = MirrorOpcode(LLP::MINER_SET_REWARD);  // 0xD0D5
    
    /**
     * MINER_REWARD_RESULT: Node confirms reward binding
     * Direction: Node → Miner
     * Payload (ChaCha20 encrypted): [status(1)][msg_len(1)][message(optional)]
     * Mirror-mapped from legacy MINER_REWARD_RESULT (214) → 0xD0D6
     */
    constexpr uint16_t MINER_REWARD_RESULT = MirrorOpcode(LLP::MINER_REWARD_RESULT);  // 0xD0D6
    
    /**
     * MINER_READY: Miner subscribes to push notifications
     * Direction: Miner → Node
     * Payload: None (header-only)
     * 
     * Requirements:
     * - Must be sent AFTER successful authentication
     * - Must be sent AFTER SET_CHANNEL
     * 
     * Response:
     * - Node sends GET_BLOCK immediately (228-byte template)
     * - Then sends GET_BLOCK on every block validation (push)
     * 
     * Mirror-mapped from legacy MINER_READY (216) → 0xD0D8
     */
    constexpr uint16_t MINER_READY = MirrorOpcode(LLP::MINER_READY);  // 0xD0D8
    
    /**
     * PRIME_BLOCK_AVAILABLE: Node notifies Prime miners of new block
     * Direction: Node → Miner (Prime channel only)
     * Payload: 148 bytes (big-endian) — new full-picture format:
     *   [0-3]    unified_height (uint32)
     *   [4-7]    prime_height (uint32)     — miner's channel
     *   [8-11]   difficulty (uint32)
     *   [12-15]  hash_height (uint32)      — other PoW channel (NEW)
     *   [16-19]  stake_height (uint32)     — Stake channel (NEW)
     *   [20-147] hashBestChain (uint1024, 128 bytes LE)
     * Backward-compat: 12-byte compact and 140-byte v1 extended still accepted.
     * Mirror-mapped from legacy PRIME_BLOCK_AVAILABLE (217) → 0xD0D9
     */
    constexpr uint16_t PRIME_BLOCK_AVAILABLE = MirrorOpcode(LLP::PRIME_BLOCK_AVAILABLE);  // 0xD0D9
    
    /**
     * HASH_BLOCK_AVAILABLE: Node notifies Hash miners of new block
     * Direction: Node → Miner (Hash channel only)
     * Payload: 148 bytes (big-endian) — new full-picture format:
     *   [0-3]    unified_height (uint32)
     *   [4-7]    hash_height (uint32)      — miner's channel
     *   [8-11]   difficulty (uint32)
     *   [12-15]  prime_height (uint32)     — other PoW channel (NEW)
     *   [16-19]  stake_height (uint32)     — Stake channel (NEW)
     *   [20-147] hashBestChain (uint1024, 128 bytes LE)
     * Backward-compat: 12-byte compact and 140-byte v1 extended still accepted.
     * Mirror-mapped from legacy HASH_BLOCK_AVAILABLE (218) → 0xD0DA
     */
    constexpr uint16_t HASH_BLOCK_AVAILABLE = MirrorOpcode(LLP::HASH_BLOCK_AVAILABLE);  // 0xD0DA
    
    // ========================================================================
    // MIRROR-MAPPED AUTH/SESSION OPCODES (using 0xD000 | legacyOpcode)
    // ========================================================================
    // These opcodes avoid the 0xD0 (208) ambiguity on stateless port:
    // legacy single-byte 208 = MINER_AUTH_CHALLENGE, but 0xD0 is also the
    // stateless prefix byte.  Mirror-mapping gives each auth opcode a unique
    // two-byte value in the 0xD0CE-0xD0D6 range.
    
    /** CHANNEL_ACK: Mirror-mapped from legacy 206 → 0xD0CE */
    constexpr uint16_t CHANNEL_ACK = MirrorOpcode(LLP::CHANNEL_ACK);  // 0xD0CE
    
    /** MINER_AUTH_INIT: Mirror-mapped from legacy 207 → 0xD0CF */
    constexpr uint16_t MINER_AUTH_INIT = MirrorOpcode(LLP::MINER_AUTH_INIT);  // 0xD0CF
    
    /** MINER_AUTH_CHALLENGE: Mirror-mapped from legacy 208 → 0xD0D0 */
    constexpr uint16_t MINER_AUTH_CHALLENGE = MirrorOpcode(LLP::MINER_AUTH_CHALLENGE);  // 0xD0D0
    
    /** MINER_AUTH_RESPONSE: Mirror-mapped from legacy 209 → 0xD0D1 */
    constexpr uint16_t MINER_AUTH_RESPONSE = MirrorOpcode(LLP::MINER_AUTH_RESPONSE);  // 0xD0D1
    
    /** MINER_AUTH_RESULT: Mirror-mapped from legacy 210 → 0xD0D2 */
    constexpr uint16_t MINER_AUTH_RESULT = MirrorOpcode(LLP::MINER_AUTH_RESULT);  // 0xD0D2
    
    /** SESSION_START: Mirror-mapped from legacy 211 → 0xD0D3 */
    constexpr uint16_t SESSION_START = MirrorOpcode(LLP::SESSION_START);  // 0xD0D3
    
    /** SESSION_KEEPALIVE: Mirror-mapped from legacy 212 → 0xD0D4 */
    constexpr uint16_t SESSION_KEEPALIVE = MirrorOpcode(LLP::SESSION_KEEPALIVE);  // 0xD0D4

    /** SESSION_EXPIRED: Mirror-mapped from legacy 221 → 0xD0DD
     *  Node notifies miner that session has expired.
     *  Payload: 5 bytes ([session_id:4B LE][reason:1B]) */
    constexpr uint16_t SESSION_EXPIRED = MirrorOpcode(LLP::SESSION_EXPIRED);  // 0xD0DD

    /** NODE_SHUTDOWN: Mirror-mapped from legacy 255 → 0xD0FF
     *  Node sends graceful shutdown notice to miner.
     *  Payload: 1 byte (shutdown reason: GRACEFUL=0x01, MAINTENANCE=0x02) */
    constexpr uint16_t NODE_SHUTDOWN = MirrorOpcode(LLP::NODE_SHUTDOWN);  // 0xD0FF
    
    // NOTE: NEW_BLOCK has been removed - the node now reuses GET_BLOCK for both
    // initial template delivery and push notifications when blockchain advances.
    
    /**
     * @enum RejectionReason
     * @brief Reasons why a block might be rejected
     */
    enum RejectionReason : uint8_t {
        STALE       = 0x01,  // Block already found (height mismatch)
        INVALID_POW = 0x02,  // Proof-of-work doesn't meet difficulty
        INVALID_SIG = 0x03,  // Falcon signature verification failed
        DUPLICATE   = 0x04,  // Duplicate block submission
        FORK        = 0x05,  // Blockchain forked, block invalid
        SESSION_INVALID = 0x06,
        REWARD_NOT_BOUND = 0x07,
        RATE_LIMITED_OR_STALE_TIMESTAMP = 0x08,
        UNKNOWN_TEMPLATE = 0x09,
        LOCAL_TEMPLATE_REJECT = 0x0A,
        MALFORMED_PRODUCER_OR_DECRYPTION_FAILED = 0x0B,
        ENCRYPTION_REQUIRED_OR_SIGNATURE_FAILED = 0x0C,
        NO_SESSION_KEY = 0x0D,
        WORK_REVOKED = 0x0E,
        PACKET_TOO_SMALL = 0x0F,
        VALIDATION_FAILED = 0x11,
        ACCEPTANCE_FAILED = 0x12,
        INTERNAL_ERROR = 0xFF,
    };
    
    /**
     * @enum ShutdownReason
     * @brief Reasons why the node is shutting down (NODE_SHUTDOWN payload byte)
     */
    enum ShutdownReason : uint8_t {
        GRACEFUL    = 0x01,  // Normal shutdown
        MAINTENANCE = 0x02,  // Planned maintenance window
    };

    /**
     * @enum SessionExpiredReason
     * @brief Reasons why a session has expired (SESSION_EXPIRED payload byte)
     */
    enum SessionExpiredReason : uint8_t {
        EXPIRED_INACTIVITY = 0x01,  // Session expired due to inactivity (no keepalive)
    };

} // namespace StatelessMining

} // namespace LLP
} // namespace nexusminer

#endif // NEXUSMINER_LLP_MINER_OPCODES_HPP
