#include "worker_fpga.hpp"
#include "statistics.hpp"
#include "LLP/block.hpp"
#include "nexus_hash_utils.hpp"
#include <random>

namespace nexusminer
{

Worker_fpga::Worker_fpga(std::shared_ptr<asio::io_context> io_context, std::string serialPort)
	: stop{ false }
	, m_io_context{ std::move(io_context) }
	, m_logger{ spdlog::get("logger") }
	, serialPortStr{serialPort}
	, serial{ *m_io_context }
{
	try {
		serial.open(serialPort);
		serial.set_option(asio::serial_port_base::baud_rate(baud));
		serial.set_option(asio::serial_port_base::character_size(8));
		serial.set_option(asio::serial_port_base::stop_bits(asio::serial_port_base::stop_bits::one));
		serial.set_option(asio::serial_port_base::parity(asio::serial_port_base::parity::none));
		serial.set_option(asio::serial_port_base::flow_control(asio::serial_port_base::flow_control::none));
	}
	catch (asio::system_error& e)
	{
		m_logger->debug(e.what());
	}
	if (serial.is_open())
		runThread = std::thread(&Worker_fpga::run, this);
}

Worker_fpga::~Worker_fpga()
{
	//make sure the run thread exits the loop
	stop = true;
	serial.close();
	if (runThread.joinable())
		runThread.join();
}

void Worker_fpga::set_block(const LLP::CBlock& block, Worker::Block_found_handler result)
{
	std::scoped_lock<std::mutex> lck(mtx);
	
	m_logger->debug("New Block");
	foundNonceCallback = result;
	block_.merkle_root = block.hashMerkleRoot;
	block_.previous_hash = block.hashPrevBlock;
	block_.nVersion = block.nVersion;
	block_.nBits = block.nBits;
	block_.nChannel = block.nChannel;
	block_.nHeight = block.nHeight;

	startingNonce = 0;// 0x0FFFFFFFFFFFFFFF;
	//TODO: Remove random starting nonce once connection to wallet is stable
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


	//TEST
	// use a sample Nexus block as a test vector
	//uint32_t nHeight = 2023276;
	//uint32_t nVersion = 4;
	//uint32_t nChannel = 2;
	//uint32_t nBits = 0x7b032ed8;
	//uint64_t nNonce = 21155560019;
	//std::string merkleStr = "31f5a458fc4207cd30fd1c4f43c26a3140193ed088f75004aa5b07beebf6be905fd49a412294c73850b422437c414429a6160df514b8ec919ea8a2346d3a3025";
	//std::string hashPrevBlockStr = "00000902546301d2a29b00cad593cf05c798469b0e3f39fe623e6762111d6f9eed3a6a18e0e5453e81da8d0db5e89808e68e96c8df13005b714b1e63d7fa44a5025d1370f6f255af2d5121c4f65624489f1b401f651b5bd505002d3a5efc098aa6fa762d270433a51697d7d8d3252d56bbbfbe62f487f258c757690d31e493a7";
	//blockHeightB = IntToBytes(nHeight, 4);
	//versionB = IntToBytes(nVersion, 4);
	//channelB = IntToBytes(nChannel, 4);
	//bitsB = IntToBytes(nBits, 4);
	//nonceB = IntToBytes(nNonce, 8);
	////convert byte strings to little endian byte order
	//std::reverse(merkleB.begin(), merkleB.end());
	//std::reverse(prevHashB.begin(), prevHashB.end());
	//END TEST

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


	//calculate midstate
	skein.setMessage(headerB);
	NexusSkein::stateType m2 = skein.getMessage2();
	//std::cout << "Message 2" << std::endl << m2.toHexString() << std::endl;
	NexusSkein::keyType key2 = skein.getKey2();
	//std::cout << "Key 2" << std::endl << key2.toHexString() << std::endl;

	std::string key2Str = key2.toHexString(true);
	std::string message2Str = m2.toHexString(true);
	message2Str.resize(88 * 2);  //crop to first 88 bytes
	std::string workPackageStr = key2Str + message2Str;

	std::vector<unsigned char> fpgaWorkPackage = HexStringToBytes(workPackageStr);

	//send new work package over the serial port
	if (serial.is_open())
	{
		asio::write(serial, asio::buffer(fpgaWorkPackage));
	}
	//wait for at least 10ms before sending another package
	using namespace std::chrono_literals;
	std::this_thread::sleep_for(10ms);
    
}

void Worker_fpga::run()
{
	m_logger->debug("FPGA Test");

	//TEST

	////convert header data to byte strings
	//std::vector<unsigned char> blockHeightB;
	//std::vector<unsigned char> versionB;
	//std::vector<unsigned char> channelB;
	//std::vector<unsigned char> bitsB;
	//std::vector<unsigned char> nonceB;
	//std::vector<unsigned char> merkleB;
	//std::vector<unsigned char> prevHashB;

	//// use a sample Nexus block as a test vector
	//uint32_t nHeight = 2023276;
	//uint32_t nVersion = 4;
	//uint32_t nChannel = 2;
	//uint32_t nBits = 0x7b032ed8;
	//uint64_t nNonce = 21155560019;
	//std::string merkleStr = "31f5a458fc4207cd30fd1c4f43c26a3140193ed088f75004aa5b07beebf6be905fd49a412294c73850b422437c414429a6160df514b8ec919ea8a2346d3a3025";
	//std::string hashPrevBlockStr = "00000902546301d2a29b00cad593cf05c798469b0e3f39fe623e6762111d6f9eed3a6a18e0e5453e81da8d0db5e89808e68e96c8df13005b714b1e63d7fa44a5025d1370f6f255af2d5121c4f65624489f1b401f651b5bd505002d3a5efc098aa6fa762d270433a51697d7d8d3252d56bbbfbe62f487f258c757690d31e493a7";
	//merkleB = HexStringToBytes(merkleStr);
	//prevHashB = HexStringToBytes(hashPrevBlockStr);
	//blockHeightB = IntToBytes(nHeight, 4);
	//versionB = IntToBytes(nVersion, 4);
	//channelB = IntToBytes(nChannel, 4);
	//bitsB = IntToBytes(nBits, 4);
	//nonceB = IntToBytes(nNonce, 8);
	////convert byte strings to little endian byte order
	//std::reverse(merkleB.begin(), merkleB.end());
	//std::reverse(prevHashB.begin(), prevHashB.end());

	////Concatenate the bytes
	//std::vector<unsigned char> headerB = versionB;
	//headerB.insert(headerB.end(), prevHashB.begin(), prevHashB.end());
	//headerB.insert(headerB.end(), merkleB.begin(), merkleB.end());
	//headerB.insert(headerB.end(), channelB.begin(), channelB.end());
	//headerB.insert(headerB.end(), blockHeightB.begin(), blockHeightB.end());
	//headerB.insert(headerB.end(), bitsB.begin(), bitsB.end());
	//headerB.insert(headerB.end(), nonceB.begin(), nonceB.end());
	////The header length should be 216 bytes
	////std::cout << "Header length: " << headerB.size() << " bytes" << std::endl;
	////std::cout << "Header: " << BytesToHexString(headerB) << std::endl;

	////calculate midstate
	//skein.setMessage(headerB);
	//NexusSkein::stateType m2 = skein.getMessage2();
	////std::cout << "Message 2" << std::endl << m2.toHexString() << std::endl;
	//NexusSkein::keyType key2 = skein.getKey2();
	////std::cout << "Key 2" << std::endl << key2.toHexString() << std::endl;

	//std::string key2Str = key2.toHexString(true);
	//std::string message2Str = m2.toHexString(true);
	//message2Str.resize(88 * 2);  //crop to first 88 bytes
	//std::string workPackageStr = key2Str + message2Str;

	//std::vector<unsigned char> fpgaWorkPackage = HexStringToBytes(workPackageStr);
	//startingNonce = nNonce;
	////send new work package over the serial port
	//if (serial.is_open())
	//{
	//	asio::write(serial, asio::buffer(fpgaWorkPackage));
	//}
	////wait for at least 10ms before sending another package
	//using namespace std::chrono_literals;
	//std::this_thread::sleep_for(10ms);

	//END TEST

	while (!stop)
	{

		//wait for response from serial port
		std::vector<unsigned char> receive_bytes(8);
		// read bytes from the serial port
		// asio::read will read bytes until the buffer is filled
		try {
			size_t nread = asio::read(
				serial, asio::buffer(receive_bytes)
			);
		}
		catch (asio::system_error& e)
		{
			m_logger->debug(e.what());
		}
		

		std::reverse(receive_bytes.begin(), receive_bytes.end());  //nonce byte order is sent big endian over the serial port
		uint64_t nonce = bytesToInt<uint64_t>(receive_bytes);
		if (startingNonce - nonce == 1)
		{
			//the fpga will respond with starting nonce - 1 to acknowledge receipt of the work package.
			m_logger->info("New block receipt acknowledged by FPGA.");
		}
		else
		{
			m_logger->info("Found a nonce candidate {}", nonce);
			skein.setNonce(nonce);
			//verify the difficulty
			if (difficultyCheck())
			{
				//update the block with the nonce and call the callback function;
				block_.nNonce = nonce;
				{
					std::scoped_lock<std::mutex> lck(mtx);
					if (foundNonceCallback)
					{
						m_io_context->post([self = shared_from_this()]()
						{
							auto block_data = self->get_block_data();
							// TODO add real internal id
							self->foundNonceCallback(0, std::make_unique<Block_data>(block_data));
						});
					}
					else
					{
						m_logger->debug("Miner callback function not set.");
					}
				}

			}
		}

	}
}

void Worker_fpga::print_statistics()
{
    //m_statistics->print();
}

bool Worker_fpga::difficultyCheck()
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
	m_logger->info("Leading Zeros Found/Required {}/{}", hashActualLeadingZeros, leadingZerosRequired);

	//check the hash result is less than the difficulty.  We truncate to just use the upper 64 bits for easier calculation.
	if (keccakHash <= difficultyTest64)
	{
		m_logger->info("Nonce passes difficulty check.");
		return true;
	}
	else
	{
		m_logger->warn("Nonce fails difficulty check.");
		return false;
	}
}

Block_data Worker_fpga::get_block_data() const
{
	return block_;
}

}