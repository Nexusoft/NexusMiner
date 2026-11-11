#ifndef NEXUSMINER_HASH_VALIDATION_HPP
#define NEXUSMINER_HASH_VALIDATION_HPP

#include <cstdint>
#include "LLC/types/uint1024.h"

namespace nexusminer {
namespace hash {

/** ValidateHashCandidate
 *
 *  Validates a hash mining candidate.
 *  Checks if hash meets proof-of-work target.
 *
 *  @param[in] hashProof The proof hash
 *  @param[in] nTarget The target from nBits
 *
 *  @return True if hash <= target, false otherwise
 *
 **/
inline bool ValidateHashCandidate(const uint1024_t& hashProof, const uint1024_t& nTarget)
{
    return (hashProof <= nTarget);
}

/** BitsToTarget
 *
 *  Convert compact nBits representation to full 1024-bit target.
 *  The nBits compact format is: 0xSSEEEEEE where:
 *  - SS is the size byte (number of bytes in target)
 *  - EEEEEE is the mantissa (most significant 3 bytes)
 *
 *  @param[in] nBits Compact difficulty bits
 *
 *  @return Full 1024-bit target value
 *
 **/
inline uint1024_t BitsToTarget(uint32_t nBits)
{
    // Extract size and mantissa from compact representation
    uint32_t nSize = nBits >> 24;
    uint32_t nWord = nBits & 0x00ffffff;
    
    // Build target from compact format
    uint1024_t nTarget = 0;
    if (nSize <= 3)
    {
        nWord >>= 8 * (3 - nSize);
        nTarget = nWord;
    }
    else
    {
        nTarget = nWord;
        nTarget <<= (8 * (nSize - 3));
    }
    
    return nTarget;
}

/** BitsToDifficulty
 *
 *  Convert nBits to human-readable difficulty.
 *  For mining channels, difficulty represents the relative
 *  difficulty compared to the baseline.
 *
 *  @param[in] nBits Compact difficulty bits
 *  @param[in] nChannel Mining channel (1=Prime, 2=Hash)
 *
 *  @return Difficulty value
 *
 **/
inline double BitsToDifficulty(uint32_t nBits, uint32_t nChannel)
{
    // Convert nBits to target
    uint1024_t nTarget = BitsToTarget(nBits);
    
    // For hash channel, difficulty is inversely proportional to target
    // Higher target = easier mining = lower difficulty
    // This is a simplified calculation; the node may use more complex formulas
    if (nChannel == 2) // Hash channel
    {
        // Use a baseline maximum target for difficulty 1.0
        // This is an approximation; actual difficulty calculation may vary
        uint1024_t nMaxTarget = BitsToTarget(0x7e7fffff); // Example maximum
        
        // Difficulty = max_target / current_target
        // Since we can't divide uint1024_t directly, use bit shifting as approximation
        // Count leading zeros to estimate relative difficulty
        int nTargetBits = 1024;
        uint1024_t temp = nTarget;
        while (temp > 0 && nTargetBits > 0)
        {
            temp >>= 1;
            nTargetBits--;
        }
        
        // Approximate difficulty based on bit position
        return static_cast<double>(1024 - nTargetBits) / 10.0;
    }
    else if (nChannel == 1) // Prime channel
    {
        // For prime channel, difficulty is encoded differently
        // Extract the difficulty value from nBits format
        // This is a simplified version; actual may need adjustment
        return static_cast<double>(nBits & 0x00ffffff) / 10000000.0;
    }
    
    return 0.0;
}

} // namespace hash
} // namespace nexusminer

#endif // NEXUSMINER_HASH_VALIDATION_HPP
