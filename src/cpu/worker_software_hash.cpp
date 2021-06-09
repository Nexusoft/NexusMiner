#include "worker_software_hash.hpp"
#include "config.hpp"
#include "stats_collector.hpp"
#include "LLP/block.hpp"
#include "nexus_hash_utils.hpp"
#include <asio.hpp>
#include <sstream>

namespace nexusminer
{

Worker_software_hash::Worker_software_hash(std::shared_ptr<asio::io_context> io_context, Worker_config& config) 
: m_io_context{std::move(io_context)}
, m_logger{spdlog::get("logger")}
, m_config{config}
, m_stop{true}
, m_log_leader{"CPU Worker " + m_config.m_id + ": " }
, m_stats_start_time{std::chrono::steady_clock::now()}
, m_hash_count{0}
, m_best_leading_zeros{0}
, m_met_difficulty_count {0}
{
	
}

Worker_software_hash::~Worker_software_hash() 
{ 
	//make sure the run thread exits the loop
	m_stop = true;  
	if (m_run_thread.joinable())
		m_run_thread.join(); 
}

void Worker_software_hash::set_block(const LLP::CBlock& block, Worker::Block_found_handler result)
{
	//stop the existing mining loop if it is running
	m_stop = true;
	if (m_run_thread.joinable())
		m_run_thread.join();
	{
		std::scoped_lock<std::mutex> lck(m_mtx);
		m_found_nonce_callback = result;
		m_block = Block_data{ block };

		//set the starting nonce for each worker to something different that won't overlap with the others
		m_starting_nonce = static_cast<uint64_t>(m_config.m_internal_id) << 48;
		m_block.nNonce = m_starting_nonce;
		std::vector<unsigned char> headerB = m_block.GetHeaderBytes();
		//calculate midstate
		m_skein.setMessage(headerB);
	}
	//restart the mining loop
	m_stop = false;
	m_run_thread = std::thread(&Worker_software_hash::run, this);
	
    
}

void Worker_software_hash::run()
{
	while (!m_stop)
	{
		std::scoped_lock<std::mutex> lck(m_mtx);
		//calculate the remainder of the skein hash starting from the midstate.
		m_skein.calculateHash();
		//run keccak on the result from skein
		NexusKeccak keccak(m_skein.getHash());
		keccak.calculateHash();
		uint64_t keccakHash = keccak.getResult();
		uint64_t nonce = m_skein.getNonce();
		//check the result for leading zeros
		if ((keccakHash & leading_zero_mask()) == 0)
		{
			m_logger->info(m_log_leader + "Found a nonce candidate {}", nonce);
			m_skein.setNonce(nonce);
			//verify the difficulty
			if (difficulty_check())
			{
				++m_met_difficulty_count;
				//update the block with the nonce and call the callback function;
				m_block.nNonce = nonce;
				{
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
		m_skein.setNonce(++nonce);	
		++m_hash_count;
	}
}

void Worker_software_hash::update_statistics(Stats_collector& stats_collector)
{
	std::scoped_lock<std::mutex> lck(m_mtx);
	std::stringstream ss;
	ss << "Worker " << m_config.m_id << " stats: ";
	ss << std::setprecision(2) << std::fixed << get_hash_rate()/1.0e6 << "MH/s ";
	ss << m_met_difficulty_count << " blocks found.  Most difficult: " << m_best_leading_zeros;
	m_logger->info(ss.str());

	stats_collector.update_worker_stats(m_config.m_internal_id, 
		Stats_hash{m_hash_count, m_best_leading_zeros, m_met_difficulty_count});

}

bool Worker_software_hash::difficulty_check()
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

uint64_t Worker_software_hash::leading_zero_mask()
{
	return ((1ull << leading_zeros_required) - 1) << (64 - leading_zeros_required);
}

double Worker_software_hash::get_hash_rate()
{
	//returns the overall hashrate for this worker in hashes per second
	return m_hash_count / static_cast<double>(elapsed_seconds());
}

void Worker_software_hash::reset_statistics()
{
	m_stats_start_time = std::chrono::steady_clock::now();
	m_hash_count = 0;
	m_best_leading_zeros = 0;
	m_met_difficulty_count = 0;
}

int Worker_software_hash::elapsed_seconds()
{
	return static_cast<int>(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - m_stats_start_time).count());
}

}