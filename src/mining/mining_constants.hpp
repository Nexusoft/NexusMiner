#ifndef NEXUSMINER_MINING_CONSTANTS_HPP
#define NEXUSMINER_MINING_CONSTANTS_HPP

/// @file mining_constants.hpp
/// @brief Named constants for mining parameters, replacing magic numbers.
///
/// These constants are referenced by CPU, GPU, and protocol modules.
/// Centralizing them here makes tuning visible and grep-friendly.

#include <cstddef>
#include <cstdint>

namespace nexusminer {
namespace mining {

// ============================================================================
//  Prime sieve constants
// ============================================================================

/// Maximum sieve length in integers (CPU).
/// Controls the size of the boolean sieve array; larger values trade
/// memory for better chain coverage.
static constexpr int CPU_SIEVE_LENGTH_MAX = 30'000'000;

/// Primorial end-prime limit for the CPU sieve.
/// Primes up to this value are included in the primorial calculation.
static constexpr int CPU_PRIMORIAL_END_PRIME = 3'000'000;

/// Minimum acceptable prime-chain length for both CPU and GPU.
/// Chains shorter than this are discarded before Fermat testing.
static constexpr int MIN_CHAIN_LENGTH = 8;

/// Maximum allowable prime gap (half-value, since sieve excludes evens).
/// Gaps larger than 2 × this value terminate chain search.
static constexpr int MAX_PRIME_GAP_HALF = 6;

/// Upper bound for sieving primes (CPU segmented sieve).
/// Primes beyond this are not sieved; chain_sieve.hpp uses 3 × 10^8.
static constexpr uint32_t CPU_SIEVING_PRIME_LIMIT = 300'000'000;

/// L1 cache size assumption for segmented sieve (bytes).
static constexpr int L1_CACHE_BYTES = 32'768;

/// L2 cache size assumption for segmented sieve (bytes).
static constexpr int L2_CACHE_BYTES = 262'144;

/// First prime at which sieving begins (after primorial {2,3,5}).
static constexpr int SIEVING_START_PRIME = 7;

// ============================================================================
//  GPU-specific prime sieve constants
// ============================================================================

/// Default Fermat-test batch size for GPU prime sieve.
static constexpr int GPU_FERMAT_BATCH_SIZE_DEFAULT = 200'000;

/// Maximum Fermat-test batch size for GPU prime sieve.
static constexpr int GPU_FERMAT_BATCH_SIZE_MAX = 1'000'000;

// ============================================================================
//  Hash mining constants
// ============================================================================

/// How often the hash worker logs performance (number of hashes).
static constexpr uint64_t HASH_LOG_INTERVAL = 1'000'000;

// ============================================================================
//  Difficulty encoding
// ============================================================================

/// Fractional difficulty base used in nBits encoding / decoding.
/// GetFractionalDifficulty multiplies the fraction remainder by this value.
static constexpr uint64_t DIFFICULTY_FRACTION_BASE = 10'000'000;

// ============================================================================
//  Block / network sanity limits
// ============================================================================

/// Minimum unified height for mainnet blocks (genesis guard).
static constexpr uint32_t MAINNET_MIN_HEIGHT = 1'000'000;

// ============================================================================
//  Prime channel vOffsets layout (single source of truth)
// ============================================================================
//
// `prime::GetOffsetsImpl` (cpu/) emits, for a Cunningham chain of length N
// (N >= 2):
//
//   (N - 1) gap bytes, each in {2, 4, 6, 8, 10, 12}
//   followed by 4 bytes of fractional difficulty (uint32_t, little-endian)
//
// Therefore a well-formed serialized offsets vector has size:
//
//   size == (N - 1) + kPrimeOffsetFractionBytes
//
// These constants are the single source of truth shared by:
//   * protocol/falcon_constants.hpp  -> derives PRIME_VOFFSETS_MAX_SIZE and the
//                                        SUBMIT_BLOCK_WRAPPER_*_MAX wire-format
//                                        upper bounds.
//   * cpu/inc/cpu/prime_validation.hpp -> re-exports under
//                                          `nexusminer::prime::` and implements
//                                          is_well_formed_prime_offsets().
//
// Living in this leaf header (only depends on <cstddef>/<cstdint>) means the
// cross-cutting drift checks below fire on every translation unit that touches
// either consumer header — they are NOT gated on WITH_PRIME=ON build flags.
//
// Bounds rationale:
//   * kMinSerializedPrimeOffsets = 5  -> chain length 2 (smallest meaningful
//     Cunningham cluster) + 4 fraction bytes.
//   * kMaxSerializedPrimeOffsets = 22 -> chain length 19 + 4 fraction bytes.
//     Cunningham chain world record is currently 17 primes; 19 gives headroom
//     for future records without ever dropping a real find.
//
// Historical bug: the miner used to hard-code `size == 10` (chain length 7),
// silently dropping every length-8+ find as "malformed". The check is now a
// range gate driven by these constants.

/// Fractional-difficulty tail size appended by GetOffsetsImpl (uint32_t LE).
static constexpr std::size_t kPrimeOffsetFractionBytes = 4;

/// Maximum supported chain length recognised by the miner.
/// Set generously above the Cunningham world record (17) so a record-breaking
/// find is never silently dropped by the miner-side sanity gate.
static constexpr std::size_t kMaxRecognisedChainLength = 19;

/// Minimum well-formed serialized vOffsets size (chain length 2).
static constexpr std::size_t kMinSerializedPrimeOffsets =
    /*chain length 2 -> 1 gap*/ 1 + kPrimeOffsetFractionBytes;  // 5

/// Maximum well-formed serialized vOffsets size (chain length kMaxRecognisedChainLength).
static constexpr std::size_t kMaxSerializedPrimeOffsets =
    (kMaxRecognisedChainLength - 1) + kPrimeOffsetFractionBytes;  // 22

// Compile-time guarantee: any chain at the configured minimum target length
// must serialize to a size the sanity gate accepts.  This catches any future
// drift where MIN_CHAIN_LENGTH is bumped past kMaxRecognisedChainLength
// without also bumping the layout constants.  Lives in this leaf header so the
// assert fires on every TU that includes mining_constants.hpp — including
// builds with WITH_PRIME=OFF.
static_assert(
    static_cast<std::size_t>(MIN_CHAIN_LENGTH) <= kMaxRecognisedChainLength,
    "MIN_CHAIN_LENGTH exceeds kMaxRecognisedChainLength — bump "
    "kMaxRecognisedChainLength in mining_constants.hpp");
static_assert(
    static_cast<std::size_t>(MIN_CHAIN_LENGTH) >= 2,
    "MIN_CHAIN_LENGTH must be >= 2 (single-prime chains are meaningless)");

} // namespace mining
} // namespace nexusminer

#endif // NEXUSMINER_MINING_CONSTANTS_HPP
