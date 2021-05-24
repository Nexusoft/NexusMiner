#include "worker_software_hash.hpp"
#include "statistics.hpp"
#include "../LLP/block.hpp"

namespace nexusminer
{

Worker_software_hash::Worker_software_hash(std::shared_ptr<asio::io_context> io_context) 
: stop{false}
, mine{false}  //set this to true to mine without waiting for a block header.
, m_io_context{std::move(io_context)}
, m_logger{spdlog::get("logger")}
{
	runThread = std::thread(&Worker_software_hash::run,this);
}

Worker_software_hash::~Worker_software_hash() 
{ 
	stop = true;  
	runThread.join(); 
}

void Worker_software_hash::set_block(const LLP::CBlock& block, Worker::Block_found_handler result)
{
	std::unique_lock<std::mutex> lck(mtx);
	mine = false;
	m_logger->debug("New Block");
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
			m_logger->debug("found a nonce candidate {}", nonce);
			//verify the difficulty
			if (difficultyCheck())
			{
				m_logger->debug("PASSES difficulty check. {}", nonce);
				//update the block with the nonce and call the callback function;
				block_.nonce = nonce;
				m_io_context->post([self = shared_from_this()]()
				{
					auto block_data = self->get_block_data(); 
					self->foundNonceCallback(std::make_unique<Block_data>(block_data));
				});					
			}
			else
			{
				m_logger->debug("FAILS difficulty check {}", nonce);
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

Block_data Worker_software_hash::get_block_data() const
{
	return block_;
}

}