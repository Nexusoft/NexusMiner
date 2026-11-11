#include "cpu/prime_validation.hpp"
#include "LLC/types/bignum.h"
#include <openssl/bn.h>
#include <spdlog/spdlog.h>

namespace nexusminer {
namespace prime {

namespace {
    // Match LLL-TAO exactly: 11 small primes
    const uint16_t SMALL_PRIMES[11] = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31 };
}

/** SmallDivisors
 *  Match LLL-TAO implementation exactly.
 *  Returns false if divisible by any small prime (composite).
 *  Returns true if passes all small prime tests (might be prime).
 **/
bool SmallDivisors(const uint1024_t& hashTest)
{
    // Handle edge cases - these would never occur in real mining but are useful for testing
    if (hashTest <= 1)
        return false;
    
    // Use native uint1024_t modulo operations (no conversion needed)
    // Match LLL-TAO: check all 11 primes, return false if divisible
    for (int i = 0; i < 11; ++i)
    {
        if ((hashTest % SMALL_PRIMES[i]) == 0)
        {
            // If hashTest equals the small prime itself, it's prime
            // This only happens in test scenarios, not in real mining
            if (hashTest == SMALL_PRIMES[i])
                continue;
            return false;  // Divisible by this prime (composite)
        }
    }
    
    return true;
}

/** FermatTest
 *  Performs Fermat primality test using OpenSSL (matching LLL-TAO exactly).
 *  Computes 2^(n-1) mod n and returns the result.
 *  If result == 1, the number is probably prime.
 **/
uint1024_t FermatTest(const uint1024_t& hashTest)
{
    // Handle edge cases
    if (hashTest <= 1)
        return uint1024_t(0);
    if (hashTest == 2)
        return uint1024_t(1);  // 2 is prime
    
    try {
        LLC::CAutoBN_CTX ctx;
        LLC::CBigNum bnPrime(hashTest);
        LLC::CBigNum bnBase(2);
        LLC::CBigNum bnExp = bnPrime - 1;
        LLC::CBigNum bnResult;
        
        // Compute 2^(prime-1) mod prime using OpenSSL
        if (BN_mod_exp(bnResult.getBN(), bnBase.getBN(), bnExp.getBN(), bnPrime.getBN(), ctx) == 0)
            return uint1024_t(0);
        
        return bnResult.getuint1024();
    } catch (...) {
        return uint1024_t(0);
    }
}

/** Miller_Rabin
 *  Wrapper for OpenSSL BN_is_prime_ex — matches LLL-TAO exactly.
 *  Uses 1 round of Miller-Rabin as the node does.
 *  Returns false on both composite (-1 error or 0) and OpenSSL errors,
 *  treating errors conservatively as composite (safe for mining).
 **/
bool Miller_Rabin(const uint1024_t& hashTest)
{
    LLC::CBigNum bnPrime(hashTest);
    // BN_is_prime_ex returns 1 (probably prime), 0 (composite), or -1 (error).
    // Treating -1 as composite is correct: we conservatively reject the candidate.
    return (BN_is_prime_ex(bnPrime.getBN(), 1, nullptr, nullptr) == 1);
}

bool PrimeCheck(const uint1024_t& hashTest)
{
    // Step 1: Small divisor tests (fast rejection)
    if (!SmallDivisors(hashTest))
        return false;

    // Step 2: Miller-Rabin test (matches LLL-TAO Check B)
    if (!Miller_Rabin(hashTest))
        return false;

    // Step 3: Fermat test - check if result equals 1
    uint1024_t result = FermatTest(hashTest);
    if (result != 1)
        return false;

    return true;
}

/** GetFractionalDifficulty
 *  Match LLL-TAO formula: ((composite - fermatResult) << 24) / composite
 *  Uses OpenSSL for all operations to match node exactly.
 **/
static uint32_t GetFractionalDifficulty(const uint1024_t& hashComposite)
{
    try {
        LLC::CAutoBN_CTX ctx;
        
        // Get Fermat test result
        uint1024_t fermatResult = FermatTest(hashComposite);
        
        // Convert to CBigNum
        LLC::CBigNum bnA(hashComposite);
        LLC::CBigNum bnB(fermatResult);
        
        // numerator = (a - b) << 24
        LLC::CBigNum bnNumerator = bnA - bnB;
        bnNumerator <<= 24;
        
        // result = numerator / a
        LLC::CBigNum bnResult = bnNumerator / bnA;
        
        // Convert to uint32_t
        return bnResult.getuint32();
    } catch (...) {
        return 0;
    }
}

/** GetOffsets - Match LLL-TAO exactly
 **/
static bool GetOffsetsImpl(const uint1024_t& hashPrime, std::vector<uint8_t>& vOffsets, bool alreadyValidated = false)
{
    vOffsets.clear();
    
    // Check if base is prime
    if (!alreadyValidated && !PrimeCheck(hashPrime))
        return false;
    
    // Match LLL-TAO loop structure exactly
    uint8_t nOffset = 2;
    uint1024_t hashLast = hashPrime;
    
    for (uint1024_t hashNext = hashPrime + 2; nOffset <= 12; hashNext += 2, nOffset += 2)
    {
        if (PrimeCheck(hashNext))
        {
            hashLast = hashNext;
            vOffsets.push_back(nOffset);
            nOffset = 0;  // Reset after finding prime
        }
    }
    
    // Append fractional difficulty as 4 bytes (LITTLE-ENDIAN to match LLL-TAO)
    uint32_t nFraction = GetFractionalDifficulty(hashLast + nOffset);
    
    // LLL-TAO uses raw memory insert which is little-endian on x86/x64
    vOffsets.push_back(nFraction & 0xFF);
    vOffsets.push_back((nFraction >> 8) & 0xFF);
    vOffsets.push_back((nFraction >> 16) & 0xFF);
    vOffsets.push_back((nFraction >> 24) & 0xFF);
    
    return true;  // Always return true if base was prime (offsets may be empty but fractional is appended)
}

bool GetOffsets(const uint1024_t& hashPrime, std::vector<uint8_t>& vOffsets)
{
    return GetOffsetsImpl(hashPrime, vOffsets, false);
}

double GetPrimeDifficulty(const uint1024_t& hashPrime, const std::vector<uint8_t>& vOffsets)
{
    if (vOffsets.size() < 4)
        return 0.0;
    
    // Cluster size = 1 (base prime) + number of offsets found
    // Last 4 bytes are fractional, so offset count = size - 4
    size_t nOffsetCount = vOffsets.size() - 4;
    uint32_t nClusterSize = 1 + static_cast<uint32_t>(nOffsetCount);
    
    // Extract fractional difficulty from last 4 bytes (LITTLE-ENDIAN)
    uint32_t nFraction = 
        static_cast<uint32_t>(vOffsets[vOffsets.size() - 4]) |
        (static_cast<uint32_t>(vOffsets[vOffsets.size() - 3]) << 8) |
        (static_cast<uint32_t>(vOffsets[vOffsets.size() - 2]) << 16) |
        (static_cast<uint32_t>(vOffsets[vOffsets.size() - 1]) << 24);
    
    // Calculate fractional remainder: 1000000.0 / nFraction
    double nRemainder = 0.0;
    if (nFraction != 0)
    {
        nRemainder = 1000000.0 / static_cast<double>(nFraction);
        
        // Keep in bounds [0, 1] per LLL-TAO
        if (nRemainder > 1.0 || nRemainder < 0.0)
            nRemainder = 0.0;
    }
    
    return static_cast<double>(nClusterSize) + nRemainder;
}

bool ValidatePrimeCandidate(
    const uint1024_t& hashPrime,
    double nRequiredDifficulty,
    std::vector<uint8_t>& vOffsets,
    double& nDifficulty)
{
    // Step 1: Check if base is prime
    if (!PrimeCheck(hashPrime))
    {
        nDifficulty = 0.0;
        vOffsets.clear();
        return false;
    }
    
    // Step 2: Find Cunningham chain offsets (skip redundant PrimeCheck)
    if (!GetOffsetsImpl(hashPrime, vOffsets, true))
    {
        nDifficulty = 0.0;
        return false;
    }
    
    // Step 3: Calculate difficulty
    nDifficulty = GetPrimeDifficulty(hashPrime, vOffsets);
    
    // Step 4: Check if difficulty meets requirement
    if (nDifficulty < nRequiredDifficulty)
    {
        return false;
    }
    
    // All checks passed
    return true;
}

} // namespace prime
} // namespace nexusminer
