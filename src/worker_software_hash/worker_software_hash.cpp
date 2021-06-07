#include "worker_software_hash.hpp"
#include "statistics.hpp"
#include "LLP/block.hpp"
#include "nexus_hash_utils.hpp"
#include <random>

namespace nexusminer
{

Worker_software_hash::Worker_software_hash(std::shared_ptr<asio::io_context> io_context, int workerID) 
: stop{false}
, leadingZerosRequired{20}  //set this lower to find more nonce candidates.
, m_io_context{std::move(io_context)}
, m_logger{spdlog::get("logger")}
, log_leader{"Software Worker " + std::to_string(workerID) + ": " }
{
	workerID_ = workerID;
	runThread = std::thread(&Worker_software_hash::run,this);
}

Worker_software_hash::~Worker_software_hash() 
{ 
	//make sure the run thread exits the loop
	stop = true;  
	runThread.join(); 
}

void Worker_software_hash::set_block(const LLP::CBlock& block, Worker::Block_found_handler result)
{

	std::scoped_lock<std::mutex> lck(mtx);
	//m_logger->debug(log_leader + "New Block");
	foundNonceCallback = result;
	block_.merkle_root = block.hashMerkleRoot;
	block_.previous_hash = block.hashPrevBlock;
	block_.nVersion = block.nVersion;
	block_.nBits = block.nBits;
	block_.nChannel = block.nChannel;
	block_.nHeight = block.nHeight;

	//startingNonce = 0x0FFFFFFFFFFFFFFF;
	//TODO: remove random starting nonce once the connection to the wallet is stable.  
	std::random_device rd;
	std::mt19937_64 gen(rd());
	std::uniform_int_distribution<uint64_t> dis;
	startingNonce = dis(gen);
	block_.nNonce = startingNonce;
	//convert header data to byte strings
	std::vector<unsigned char> blockHeightB = IntToBytes(block.nHeight, 4);
	std::vector<unsigned char> versionB = IntToBytes(block.nVersion, 4);
	std::vector<unsigned char> channelB = IntToBytes(block.nChannel, 4);
	std::vector<unsigned char> bitsB = IntToBytes(block.nBits, 4);
	std::vector<unsigned char> nonceB = IntToBytes(block_.nNonce, 8);
	std::string merkleStr = block_.merkle_root.GetHex();
	std::string hashPrevBlockStr = block_.previous_hash.GetHex();
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
	//The header length should be 216 bytes
	//std::cout << "Header length: " << headerB.size() << " bytes" << std::endl;
	//std::cout << "Header: " << BytesToHexString(headerB) << std::endl;

	//calculate midstate
	skein.setMessage(headerB);
    
}

void Worker_software_hash::run()
{
	while (!stop)
	{
		std::scoped_lock<std::mutex> lck(mtx);
		//calculate the remainder of the skein hash starting from the midstate.
		skein.calculateHash();
		//run keccak on the result from skein
		NexusKeccak keccak(skein.getHash());
		keccak.calculateHash();
		uint64_t keccakHash = keccak.getResult();
		uint64_t nonce = skein.getNonce();
		//check the result for leading zeros
		if ((keccakHash & leadingZeroMask()) == 0)
		{
			m_logger->info(log_leader + "Found a nonce candidate {}", nonce);
			skein.setNonce(nonce);
			//verify the difficulty
			if (difficultyCheck())
			{
				//m_logger->debug("PASSES difficulty check. {}", nonce);
				//update the block with the nonce and call the callback function;
				block_.nNonce = nonce;
				{
					if (foundNonceCallback)
					{
						m_io_context->post([self = shared_from_this()]()
						{
							auto block_data = self->get_block_data();
							// TODO: add real internal id
							self->foundNonceCallback(0, std::make_unique<Block_data>(block_data));
						});
					}
					else
					{
						m_logger->debug(log_leader + "Miner callback function not set.");
					}
				}

			}
		}
		skein.setNonce(++nonce);	
	}
}

void Worker_software_hash::print_statistics()
{
   // m_statistics->print();
}

bool Worker_software_hash::difficultyCheck()
{
	//perform additional difficulty filtering prior to submitting the nonce 

	//leading zeros in bits required of the hash for it to pass the current difficulty.
	int leadingZerosRequired;
	uint64_t difficultyTest64;
	decodeBits(block_.nBits, leadingZerosRequired, difficultyTest64);
	skein.calculateHash();
	//run keccak on the result from skein
	NexusKeccak keccak(skein.getHash());
	keccak.calculateHash();
	uint64_t keccakHash = keccak.getResult();
	int hashActualLeadingZeros = 63 - findMSB(keccakHash);
	m_logger->info(log_leader + "Leading Zeros Found/Required {}/{}", hashActualLeadingZeros, leadingZerosRequired);

	//check the hash result is less than the difficulty.  We truncate to just use the upper 64 bits for easier calculation.
	if (keccakHash <= difficultyTest64)
	{
		m_logger->info(log_leader + "Nonce passes difficulty check.");
		return true;
	}
	else
	{
		m_logger->warn(log_leader + "Nonce fails difficulty check.");
		return false;
	}
}

uint64_t Worker_software_hash::leadingZeroMask()
{
	return ((1ull << leadingZerosRequired) - 1) << (64 - leadingZerosRequired);
}


Block_data Worker_software_hash::get_block_data() const
{
	return block_;
}

}