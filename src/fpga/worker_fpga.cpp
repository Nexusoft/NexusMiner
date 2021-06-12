#include "worker_fpga.hpp"
#include "stats/stats_collector.hpp"
#include "LLP/block.hpp"
#include "nexus_hash_utils.hpp"
#include "config/config.hpp"

namespace nexusminer
{

Worker_fpga::Worker_fpga(std::shared_ptr<asio::io_context> io_context, Worker_config& config)
	: m_io_context{ std::move(io_context) }
	, m_logger{ spdlog::get("logger") }
	, m_config{config}
	, m_serial{ *m_io_context }
	, m_nonce_candidates_recieved{ 0 }
	, m_best_leading_zeros{ 0 }
	, m_met_difficulty_count{ 0 }
{
	auto& worker_config_fpga = std::get<Worker_config_fpga>(m_config.m_worker_mode);
	m_receive_nonce_buffer.resize(responseLength);
	m_serial_port_path = worker_config_fpga.serial_port;
	m_log_leader = std::string{"FPGA Worker " + m_config.m_id + ": " };
	try {
		m_serial.open(m_serial_port_path);
		m_serial.set_option(asio::serial_port_base::baud_rate(baud));
		m_serial.set_option(asio::serial_port_base::character_size(8));
		m_serial.set_option(asio::serial_port_base::stop_bits(asio::serial_port_base::stop_bits::one));
		m_serial.set_option(asio::serial_port_base::parity(asio::serial_port_base::parity::none));
		m_serial.set_option(asio::serial_port_base::flow_control(asio::serial_port_base::flow_control::none));
	}
	catch (asio::system_error& e)
	{
		m_logger->debug(e.what());
	}
}

Worker_fpga::~Worker_fpga()
{
	m_serial.close();
}

void Worker_fpga::set_block(const LLP::CBlock& block, Worker::Block_found_handler result)
{
	//send new block info to the device
	std::scoped_lock<std::mutex> lck(m_mtx);
	//cancel any outstanding serial port async read
	m_serial.cancel();
	
	m_found_nonce_callback = result;
	m_block = Block_data{ block };

	m_starting_nonce = static_cast<uint64_t>(m_config.m_internal_id) << 48;
	m_block.nNonce = m_starting_nonce;

	std::vector<unsigned char> headerB = m_block.GetHeaderBytes();
	//calculate midstate
	m_skein.setMessage(headerB);
	//assemble the work package
	NexusSkein::stateType m2 = m_skein.getMessage2();
	NexusSkein::keyType key2 = m_skein.getKey2();
	//std::cout << "Key 2" << std::endl << key2.toHexString() << std::endl;
	std::string key2Str = key2.toHexString(true);
	std::string message2Str = m2.toHexString(true);
	message2Str.resize(88 * 2);  //crop to first 88 bytes
	std::string workPackageStr = key2Str + message2Str;

	std::vector<unsigned char> fpgaWorkPackage = HexStringToBytes(workPackageStr);

	//send new work package over the serial port
	if (m_serial.is_open())
	{
		asio::write(m_serial, asio::buffer(fpgaWorkPackage));
	}

	start_read();
    
}

void Worker_fpga::start_read()
{	
	// start the asynchronous read to wait for the next nonce to come across the serial port
	try {
		asio::async_read(m_serial, asio::buffer(m_receive_nonce_buffer), std::bind(&Worker_fpga::handle_read, this,
			std::placeholders::_1, std::placeholders::_2));
	}
	catch (asio::system_error& e)
	{
		m_logger->debug(e.what());
	}
}

void Worker_fpga::handle_read(const asio::error_code& error_code, std::size_t bytes_transferred)
{
	if (!error_code && bytes_transferred == m_receive_nonce_buffer.size())
	{
		std::reverse(m_receive_nonce_buffer.begin(), m_receive_nonce_buffer.end());  //nonce byte order is sent big endian over the serial port
		uint64_t nonce = bytesToInt<uint64_t>(m_receive_nonce_buffer);
		if (m_starting_nonce - nonce == 1)
		{
			//the fpga will respond with starting nonce - 1 to acknowledge receipt of the work package.
			m_logger->info(m_log_leader + "New block receipt acknowledged by FPGA.");
		}
		else
		{
			++m_nonce_candidates_recieved;
			m_logger->info(m_log_leader + "found a nonce candidate {}", nonce);
			m_skein.setNonce(nonce);
			//verify the difficulty
			if (difficulty_check())
			{
				++m_met_difficulty_count;
				//update the block with the nonce and call the callback function;
				m_block.nNonce = nonce;
				{
					std::scoped_lock<std::mutex> lck(m_mtx);
					if (m_found_nonce_callback)
					{
						m_io_context->post([self = shared_from_this()]()
						{
							self->m_found_nonce_callback(self->m_config.m_internal_id, std::make_unique<Block_data>(self->m_block));
						});
					}
					else
					{
						m_logger->debug(m_log_leader + "Miner callback function not set.");
					}
				}
			}
		}
		//wait for the next nonce
		start_read();
	}
	else
	{
		if (error_code != asio::error::operation_aborted)  //it's normal for the async_read to be canceled. 
		{
			if (error_code)
			{
				m_logger->error(m_log_leader + "ASIO Error {} " + error_code.message(), error_code.value());
			}
			else
			{
				m_logger->error(m_log_leader + "Received unexpected number of bytes on serial port.  Expected {} received {}.", m_receive_nonce_buffer.size(), bytes_transferred);
			}
		}
	}

}

void Worker_fpga::update_statistics(Stats_collector& stats_collector)
{
	std::scoped_lock<std::mutex> lck(m_mtx);

	stats_collector.update_worker_stats(m_config.m_internal_id, 
		Stats_hash{m_nonce_candidates_recieved * nonce_difficulty_filter, m_best_leading_zeros, m_met_difficulty_count});
}

bool Worker_fpga::difficulty_check()
{
	//perform additional difficulty filtering prior to submitting the nonce 
	
	//leading zeros in bits required of the hash for it to pass the current difficulty.
	int leadingZerosRequired;
	uint64_t difficultyTest64;
	decodeBits(m_block.nBits, leadingZerosRequired, difficultyTest64);
	m_skein.calculateHash();
	//run keccak on the result from skein
	NexusKeccak keccak(m_skein.getHash());
	keccak.calculateHash();
	uint64_t keccakHash = keccak.getResult();
	int hashActualLeadingZeros = 63 - findMSB(keccakHash);
	m_logger->info(m_log_leader + "Leading Zeros Found/Required {}/{}", hashActualLeadingZeros, leadingZerosRequired);
	if (hashActualLeadingZeros > m_best_leading_zeros)
	{
		m_best_leading_zeros = hashActualLeadingZeros;
	}
	//check the hash result is less than the difficulty.  We truncate to just use the upper 64 bits for easier calculation.
	if (keccakHash <= difficultyTest64)
	{
		m_logger->info(m_log_leader + "Nonce passes difficulty check.");
		return true;
	}
	else
	{
		//m_logger->warn(m_log_leader + "Nonce fails difficulty check.");
		return false;
	}
}

void Worker_fpga::set_test_block()
{
	// use a sample Nexus block as a test vector
	// hashing this block should result in a number with 49 leading zeros
	std::string merkleStr = "31f5a458fc4207cd30fd1c4f43c26a3140193ed088f75004aa5b07beebf6be905fd49a412294c73850b422437c414429a6160df514b8ec919ea8a2346d3a3025";
	std::string hashPrevBlockStr = "00000902546301d2a29b00cad593cf05c798469b0e3f39fe623e6762111d6f9eed3a6a18e0e5453e81da8d0db5e89808e68e96c8df13005b714b1e63d7fa44a5025d1370f6f255af2d5121c4f65624489f1b401f651b5bd505002d3a5efc098aa6fa762d270433a51697d7d8d3252d56bbbfbe62f487f258c757690d31e493a7";
	m_block.nBits = 0x7b032ed8;
	m_block.nChannel = 2;
	m_block.nHeight = 2023276;
	m_block.nVersion = 4;
	m_block.nNonce = 21155560019;
	m_block.merkle_root.SetHex(merkleStr);
	m_block.previous_hash.SetHex(hashPrevBlockStr);
	m_starting_nonce = m_block.nNonce;
	

}

void Worker_fpga::reset_statistics()
{
	m_nonce_candidates_recieved = 0;
	m_best_leading_zeros = 0;
	m_met_difficulty_count = 0;
}

}