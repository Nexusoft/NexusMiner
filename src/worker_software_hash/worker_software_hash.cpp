#include "worker_software_hash.hpp"
#include "statistics.hpp"

namespace nexusminer
{

Worker_software_hash::Worker_software_hash(): stop(false)
{
	runThread = std::thread(&Worker_software_hash::run,this);
	mine = true;  //todo: set this to false when headers come in correctly
}

void Worker_software_hash::set_block(const LLP::CBlock& block, Worker::Block_found_handler result)
{
	std::unique_lock<std::mutex> lck(mtx);
	mine = false;
	std::cout << "New Block" << std::endl;
	foundNonceCallback = result;
	block_.merkle_root = block.hashMerkleRoot;
	block_.previous_hash = block.hashPrevBlock;


	uint64_t starting_nonce = 0;
	block_.nonce = starting_nonce;
	//convert header data to byte strings
	std::vector<unsigned char> blockHeightB = IntToBytes(block.nHeight, 4);
	std::vector<unsigned char> versionB = IntToBytes(block.nVersion, 4);
	std::vector<unsigned char> channelB = IntToBytes(block.nChannel, 4);
	std::vector<unsigned char> bitsB = IntToBytes(block.nBits, 4);
	std::vector<unsigned char> nonceB = IntToBytes(starting_nonce, 8);
	std::vector<unsigned char> merkleB = block_.merkle_root.GetBytes();
	std::vector<unsigned char> prevHashB = block_.previous_hash.GetBytes();

	//Concatenate the bytes
	std::vector<unsigned char> headerB = versionB;
	headerB.insert(headerB.end(), prevHashB.begin(), prevHashB.end());
	headerB.insert(headerB.end(), merkleB.begin(), merkleB.end());
	headerB.insert(headerB.end(), channelB.begin(), channelB.end());
	headerB.insert(headerB.end(), blockHeightB.begin(), blockHeightB.end());
	headerB.insert(headerB.end(), bitsB.begin(), bitsB.end());
	headerB.insert(headerB.end(), nonceB.begin(), nonceB.end());
	//The header length should be 216 bytes
	//std::cout << "Header length: " << headerB.size() << " bytes" << std::endl;
	//std::cout << "Header: " << BytesToHexString(headerB) << std::endl;

	//calculate midstate
	skein.setMessage(headerB);
	mine = true;
	cv.notify_one();
    
}

void Worker_software_hash::run()
{
	
	while (!stop)
	{
		//block until the header is ready
		std::unique_lock<std::mutex> lck(mtx);
		cv.wait(lck, [&] {return mine; });
		//calculate the remainder of the skein hash starting from the midstate.
		skein.calculateHash();
		//run keccak on the result from skein
		NexusKeccak keccak(skein.getHash());
		keccak.calculateHash();
		uint64_t keccakHash = keccak.getResult();
		uint64_t nonce = skein.getNonce();
		//check the result for leading zeros
		if ((keccakHash & leadingZeroMask) == 0)
		{
			std::cout << "found a nonce candidate " << nonce << "... ";
			//verify the difficulty
			if (difficultyCheck())
			{
				std::cout << "PASSES difficulty check. " << nonce << std::endl;
				//update the block with the nonce and call the callback function;
				block_.nonce = nonce;
				//m_io_context_->post(foundNonceCallback(std::make_shared<Block_data>(block_)));  //i don't think this is correct
			}
			else
			{
				std::cout << "FAILS difficulty check. " << nonce << std::endl;
			}
		}
		skein.setNonce(++nonce);
	}
}

void Worker_software_hash::print_statistics()
{
    m_statistics->print();
}



bool Worker_software_hash::difficultyCheck()
{
	//perform a more precise difficulty check prior to submitting the nonce 
	return true;
}

}