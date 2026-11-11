#ifndef NEXUSMINER_MINING_CLIENT_BLOCK_H
#define NEXUSMINER_MINING_CLIENT_BLOCK_H

#include "LLC/types/uint1024.h"
#include <vector>
#include <cstdint>
#include <string>
#include <ctime>

namespace nexusminer {
namespace mining {

// Mining channel constants
constexpr uint32_t CHANNEL_PRIME = 1;
constexpr uint32_t CHANNEL_HASH = 2;
constexpr uint32_t CHANNEL_STAKE = 3;

// Template age timeout (seconds) — push-driven era client-level validation gate.
// 300 s: safely above the observed 275 s no-block drought (all channels combined).
// The node pushes a fresh template within ~2 s of ANY tip advance (hash blocks ~18 s avg),
// so 300 s only fires as a last-resort dead-connection detector at the ClientChannelManager
// validation level.  Higher-level dead-connection emergency is at 600 s
// (TEMPLATE_AGE_EMERGENCY_TIMEOUT_SECONDS in protocol_constants.hpp).
constexpr uint64_t MAX_TEMPLATE_AGE_SECONDS = 300;

/**
 * @brief ClientBlock - CLIENT-SIDE equivalent of NODE's Block class
 * 
 * Represents a mining template received from the LLL-TAO node.
 * Contains only fields needed for mining (no consensus validation).
 * 
 * This mirrors NODE's TAO::Ledger::Block but is lightweight for client-side use.
 * Separates serializable template data from chain state (see ClientBlockState).
 * 
 * Architecture Alignment:
 *   NODE: TAO::Ledger::Block
 *   CLIENT: ClientBlock (this class)
 */
class ClientBlock
{
public:
    // Block header fields (serializable)
    //
    // NOTE ON NAMING CONVENTIONS:
    // These fields use Hungarian notation (nHeight, nBits, nChannel, etc.) to match
    // the LLL-TAO node's TAO::Ledger::Block struct for wire-compatibility.
    // Modern protocol-layer code uses snake_case aliases:
    //   nHeight         → "unified_height"    (blockchain tip + 1, used in ProofHash)
    //   nChannel        → "channel"           (1=Prime, 2=Hash, 3=Stake)
    //   nBits           → "difficulty_bits"    (compact difficulty representation)
    //   nVersion        → "block_version"     (protocol version for this block)
    //   nNonce          → "mining_nonce"      (worker-incremented solution nonce)
    //   nTime           → "block_timestamp"   (block creation time, epoch seconds)
    //   hashPrevBlock   → "hash_prev_block"   (best-chain tip hash at template time)
    //   hashMerkleRoot  → "merkle_root"       (64-byte transaction commitment)
    //
    // See also: ClientBlockState::nChannelHeight → "channel_height"
    //   (channel-specific height, metadata only — NOT part of 216-byte block bytes).
    //   This is the HEIGHT for "how many blocks in THIS channel", not the TIP.
    //   nHeight is always the UNIFIED TIP height across all channels.

    uint32_t nVersion;              // Block version (alias: block_version)
    uint1024_t hashPrevBlock;       // Hash of the best chain tip at template creation time.
                                    // (alias: hash_prev_block)
                                    // MUST equal ChainState::hashBestChain at block acceptance.
                                    // This is the primary staleness anchor (StakeMinter pattern).
                                    // On any tip_moved notification, request a fresh template;
                                    // the new template's hashPrevBlock will reflect the new tip.
    uint512_t hashMerkleRoot;       // Merkle root (64 bytes) (alias: merkle_root)
    uint32_t nChannel;              // Mining channel (1=Prime, 2=Hash, 3=Stake) (alias: channel)
    uint32_t nHeight;               // UNIFIED blockchain height (alias: unified_height).
                                    // This is tStateBest.nHeight + 1 — the next unified blockchain
                                    // height to be mined, not a per-channel count.
                                    // Block::ProofHash() hashes nVersion→nBits range including this field.
                                    // MUST NOT be overwritten with channel-specific height — that
                                    // would corrupt ProofHash(). Channel-specific height is tracked
                                    // separately in ClientBlockState::nChannelHeight (metadata only).
    uint32_t nBits;                 // Difficulty bits (alias: difficulty_bits)
    uint64_t nNonce;                // Mining nonce (alias: mining_nonce)
    uint32_t nTime;                 // Block timestamp (alias: block_timestamp)
    
    // Prime-specific fields
    std::vector<uint32_t> vOffsets; // Prime offsets (Prime channel only)
    
    // Block signature (for submission)
    std::vector<uint8_t> vchBlockSig;
    
    /**
     * @brief Default constructor - creates null block
     */
    ClientBlock()
        : nVersion(0)
        , hashPrevBlock(0)
        , hashMerkleRoot(0)
        , nChannel(0)
        , nHeight(0)
        , nBits(0)
        , nNonce(0)
        , nTime(0)
    {
    }
    
    /**
     * @brief Get mining channel
     * @return Channel number (CHANNEL_PRIME, CHANNEL_HASH, or CHANNEL_STAKE)
     */
    uint32_t GetChannel() const
    {
        return nChannel;
    }
    
    /**
     * @brief Check if block is null/empty
     * @return true if block is uninitialized
     */
    bool IsNull() const
    {
        return nVersion == 0 && nHeight == 0 && nChannel == 0;
    }
    
    /**
     * @brief Check if this is a Prime channel block
     * @return true if Prime channel
     */
    bool IsPrime() const
    {
        return nChannel == CHANNEL_PRIME;
    }
    
    /**
     * @brief Check if this is a Hash channel block
     * @return true if Hash channel
     */
    bool IsHash() const
    {
        return nChannel == CHANNEL_HASH;
    }
    
    /**
     * @brief Get channel name as string (for logging/debugging)
     * @return Channel name string ("PRIME", "HASH", "STAKE", or "UNKNOWN")
     * @note Does NOT compute actual block hash - use GetPlaceholderHash() for identifier
     */
    const char* GetChannelName() const
    {
        switch (nChannel) {
            case CHANNEL_PRIME: return "PRIME";
            case CHANNEL_HASH: return "HASH";
            case CHANNEL_STAKE: return "STAKE";
            default: return "UNKNOWN";
        }
    }
    
    /**
     * @brief Get placeholder hash for identification
     * @return Previous block hash as identifier
     * @note This is NOT a real hash computation. Real hash verification happens on NODE.
     *       CLIENT-SIDE doesn't need to compute actual hashes.
     */
    uint1024_t GetPlaceholderHash() const
    {
        return hashPrevBlock;
    }
    
    /**
     * @brief Get block hash (legacy method - calls GetPlaceholderHash)
     * @deprecated Use GetPlaceholderHash() to make intent clear
     * @return Placeholder hash (previous block hash)
     */
    uint1024_t GetHash() const
    {
        return GetPlaceholderHash();
    }
    
    /**
     * @brief Convert block to string for logging
     * @return Human-readable block description
     */
    std::string ToString() const
    {
        std::string str;
        str += "ClientBlock(";
        str += "channel=" + std::string(GetChannelName());
        str += ", height=" + std::to_string(nHeight);
        str += ", version=" + std::to_string(nVersion);
        str += ", bits=0x" + std::to_string(nBits);
        str += ", nonce=" + std::to_string(nNonce);
        str += ", time=" + std::to_string(nTime);
        str += ")";
        return str;
    }
};

} // namespace mining
} // namespace nexusminer

#endif // NEXUSMINER_MINING_CLIENT_BLOCK_H
