#ifndef NEXUSMINER_BLOCK_HEADER_UTILS_HPP
#define NEXUSMINER_BLOCK_HEADER_UTILS_HPP

#include <vector>
#include "LLC/hash/SK.h"
#include "block.hpp"

#ifdef ui_type
#undef ui_type
#endif
#ifdef dec_unit_type
#undef dec_unit_type
#endif
#ifdef dec_bufr_type
#undef dec_bufr_type
#endif
#ifdef ptr_cast
#undef ptr_cast
#endif

namespace nexusminer {

/**
 * Extract the raw in-memory CBlock header span used by the node hashing code.
 * When exclude_nonce is true this returns nVersion..nBits (prime ProofHash span);
 * otherwise it returns nVersion..nNonce for full header hashing workers.
 */
inline std::vector<unsigned char> GetBlockHeaderBytes(const ::LLP::CBlock& block, bool exclude_nonce = false)
{
	const auto* begin = reinterpret_cast<const unsigned char*>(BEGIN(block.nVersion));
	const auto* end = exclude_nonce
		? reinterpret_cast<const unsigned char*>(END(block.nBits))
		: reinterpret_cast<const unsigned char*>(END(block.nNonce));
	return std::vector<unsigned char>(begin, end);
}

/**
 * Compute the prime-channel ProofHash exactly as the upstream node does:
 * LLC::SK1024 over the raw header bytes from nVersion through nBits, excluding nonce.
 * This uses BEGIN/END directly to mirror the node's Block::ProofHash() call site
 * rather than routing through GetBlockHeaderBytes() first, avoiding an
 * intermediate vector allocation while keeping the exact same span semantics.
 */
inline uint1024_t GetPrimeProofHash(const ::LLP::CBlock& block)
{
	return LLC::SK1024(BEGIN(block.nVersion), END(block.nBits));
}

/**
 * Minimum prime-origin threshold, matching the upstream node's consensus
 * constant exactly:
 *     bnPrimeMinOrigins = ~uint1024_t(0) >> 8   (a 1016-bit all-ones value)
 * (see LLL-TAO TAO/Ledger/include/constants.h / Nexus core.h). The node's
 * Block::VerifyWork()/CheckWork() (nVersion >= 5) rejects any Prime-channel
 * (nChannel == 1) block whose ProofHash() is below this value with the
 * error "prime origins below 1016-bits".
 */
inline const uint1024_t& GetPrimeMinOrigins()
{
	static const uint1024_t bnPrimeMinOrigins = ~uint1024_t(0) >> 8;
	return bnPrimeMinOrigins;
}

/**
 * Mirrors the node's Prime-channel origins gate:
 *     if (nVersion >= 5 && ProofHash() < bnPrimeMinOrigins) reject
 *
 * Returns false when the template is DEAD ON ARRIVAL for prime mining --
 * no nonce/offset search can ever produce a block the node will accept,
 * because ProofHash() depends only on header fields fixed at template
 * creation (nVersion, hashPrevBlock, hashMerkleRoot, nChannel, nHeight,
 * nBits) and never changes for this template regardless of nNonce. This
 * happens for roughly 1 in 256 Prime-channel templates (whenever the top
 * 8 bits of the 1024-bit ProofHash are all zero), so it only needs to be
 * evaluated once per template rather than once per candidate.
 *
 * Returns true (no-op) for non-Prime channels and for nVersion < 5, since
 * the upstream rule only applies to versioned Prime-channel blocks.
 */
inline bool HasSufficientPrimeOrigins(const ::LLP::CBlock& block)
{
	if (block.nChannel != 1 || block.nVersion < 5)
		return true;

	return GetPrimeProofHash(block) >= GetPrimeMinOrigins();
}

} // namespace nexusminer

#endif
