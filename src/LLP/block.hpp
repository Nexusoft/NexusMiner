#ifndef NEXUS_LLP_BLOCK_H
#define NEXUS_LLP_BLOCK_H

#include "LLC/types/uint1024.h"
#include <memory>

#define BEGIN(a)            ((char*)&(a))
#define END(a)              ((char*)&((&(a))[1]))

namespace LLP { 

/** Block structure for Full Solo Mining (Tritium/Legacy blocks) **/
class CBlock
{
public:
	using Uptr = std::unique_ptr<CBlock>;
	using Sptr = std::shared_ptr<CBlock>;

	/** Begin of Header.   BEGIN(nVersion) **/
	unsigned int  nVersion;
	uint1024_t hashPrevBlock;    // 128 bytes (full block format for solo mining)
	uint512_t hashMerkleRoot;    // 64 bytes (full block format for solo mining)
	unsigned int  nChannel;
	unsigned int   nHeight;
	unsigned int     nBits;
	std::uint64_t      nNonce;
	unsigned int  nTime;
	/** End of Header.     END(nTime).
		Full block structure for solo mining with LLL-TAO nodes **/

	CBlock()
	{
		nVersion = 0;
		hashPrevBlock = 0;
		hashMerkleRoot = 0;
		nChannel = 0;
		nHeight = 0;
		nBits = 0;
		nNonce = 0;
		nTime = 0;
	}

	// Compatibility helpers for legacy code expecting uint256_t
	uint256_t GetHashPrevBlock256() const {
		// Extract first 32 bytes from uint1024_t
		uint256_t result;
		auto bytes = hashPrevBlock.GetBytes();
		if (bytes.size() >= 32) {
			std::vector<uint8_t> first32(bytes.begin(), bytes.begin() + 32);
			result.SetBytes(first32);
		}
		return result;
	}
	
	uint256_t GetHashMerkleRoot256() const {
		// Extract first 32 bytes from uint512_t
		uint256_t result;
		auto bytes = hashMerkleRoot.GetBytes();
		if (bytes.size() >= 32) {
			std::vector<uint8_t> first32(bytes.begin(), bytes.begin() + 32);
			result.SetBytes(first32);
		}
		return result;
	}

	//inline uint1024 GetHash() const { return SK1024(BEGIN(nVersion), END(nBits)); }
	//inline uint1024 GetPrime() const { return GetHash() + nNonce; }
};
}

#endif