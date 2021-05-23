#ifndef NEXUS_LLP_BLOCK_H
#define NEXUS_LLP_BLOCK_H

#include "../hash/uint1024.h"
//#include "../hash/templates.h"
#include <memory>

#define BEGIN(a)            ((char*)&(a))
#define END(a)              ((char*)&((&(a))[1]))

namespace LLP { 

/** Mock Class for Building Block Hash. **/
class CBlock
{
public:
	using Uptr = std::unique_ptr<CBlock>;
	using Sptr = std::shared_ptr<CBlock>;

	/** Begin of Header.   BEGIN(nVersion) **/
	unsigned int  nVersion;
	uint1024 hashPrevBlock;
	uint512 hashMerkleRoot;
	unsigned int  nChannel;
	unsigned int   nHeight;
	unsigned int     nBits;
	uint64          nNonce;
	/** End of Header.     END(nNonce).
		All the components to build an SK1024 Block Hash. **/


	CBlock()
	{
		nVersion = 0;
		hashPrevBlock = 0;
		hashMerkleRoot = 0;
		nChannel = 0;
		nHeight = 0;
		nBits = 0;
		nNonce = 0;
	}

	//inline uint1024 GetHash() const { return SK1024(BEGIN(nVersion), END(nBits)); }
	//inline uint1024 GetPrime() const { return GetHash() + nNonce; }
};
}

#endif