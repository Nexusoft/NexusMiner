#ifndef NEXUSMINER_WORKER_HPP
#define NEXUSMINER_WORKER_HPP

#include <memory>
#include <functional>
#include "LLC/types/uint1024.h"
#include "block.hpp"
#include "hash/byte_utils.hpp"

namespace nexusminer {
namespace stats { class Collector; }

class Block_data
{
public:

	Block_data(const LLP::CBlock& block)
		:merkle_root{block.hashMerkleRoot}
		, previous_hash{block.hashPrevBlock}
		, nHeight {block.nHeight}
		, nVersion {block.nVersion}
		, nChannel {block.nChannel}
		, nBits {block.nBits}
		, nNonce{ block.nNonce }{}

	Block_data() {}

	std::vector<unsigned char> GetHeaderBytes()
	{
		//convert header data to byte strings
		std::vector<unsigned char> blockHeightB = IntToBytes(nHeight, 4);
		std::vector<unsigned char> versionB = IntToBytes(nVersion, 4);
		std::vector<unsigned char> channelB = IntToBytes(nChannel, 4);
		std::vector<unsigned char> bitsB = IntToBytes(nBits, 4);
		std::vector<unsigned char> nonceB = IntToBytes(nNonce, 8);
		std::string merkleStr = merkle_root.GetHex();
		std::string hashPrevBlockStr = previous_hash.GetHex();
		std::vector<unsigned char> merkleB = HexStringToBytes(merkleStr);
		std::vector<unsigned char> prevHashB = HexStringToBytes(hashPrevBlockStr);
		std::reverse(merkleB.begin(), merkleB.end());
		std::reverse(prevHashB.begin(), prevHashB.end());

		//Concatenate the bytes
		std::vector<unsigned char> headerB = versionB;
		headerB.insert(headerB.end(), prevHashB.begin(), prevHashB.end());
		headerB.insert(headerB.end(), merkleB.begin(), merkleB.end());
		headerB.insert(headerB.end(), channelB.begin(), channelB.end());
		headerB.insert(headerB.end(), blockHeightB.begin(), blockHeightB.end());
		headerB.insert(headerB.end(), bitsB.begin(), bitsB.end());
		headerB.insert(headerB.end(), nonceB.begin(), nonceB.end());

		return headerB;

	}

	uint512_t merkle_root;
    uint1024_t previous_hash;
	uint32_t nHeight = 2023276;
	uint32_t nVersion = 4;
	uint32_t nChannel = 2;
	uint32_t nBits = 0x7b032ed8;
	uint64_t nNonce = 21155560019;

};

class Worker {
public:
	
	virtual ~Worker() = default;

    // A call to the BlockFoundHandler informs the user about a new found block.
    using Block_found_handler = std::function<void(std::uint32_t id, std::unique_ptr<Block_data>&& block)>;

    // Sets a new block (nexus data type) for the miner worker. The miner worker must reset the current work.
    // When  the worker finds a new block, the BlockFoundHandler has to be called with the found BlockData
    virtual void set_block(LLP::CBlock block, std::uint32_t nbits, Block_found_handler result) = 0;

    virtual void update_statistics(stats::Collector& stats_collector) = 0;
};

}


#endif