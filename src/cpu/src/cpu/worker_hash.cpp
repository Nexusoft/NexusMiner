#include "cpu/worker_hash.hpp"
#include "config/config.hpp"
#include "stats/stats_collector.hpp"
#include "block.hpp"
#include "hash/nexus_hash_utils.hpp"
#include <asio.hpp>

namespace nexusminer
{
namespace cpu
{

Worker_hash::Worker_hash(std::shared_ptr<asio::io_context> io_context, Worker_config& config) 
: m_io_context{std::move(io_context)}
, m_logger{spdlog::get("logger")}
, m_config{config}
, m_stop{true}
, m_log_leader{"CPU Worker " + m_config.m_id + ": " }
, m_hash_count{0}
, m_best_leading_zeros{0}
, m_met_difficulty_count {0}
, m_pool_nbits{0}
{
	
}

Worker_hash::~Worker_hash() 
{ 
	//make sure the run thread exits the loop
	m_stop = true;  
	if (m_run_thread.joinable())
		m_run_thread.join(); 
}

void Worker_hash::set_block(LLP::CBlock block, std::uint32_t nbits, Worker::Block_found_handler result)
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
		if(nbits != 0)	// take nBits provided from pool
		{
			m_pool_nbits = nbits;
		}

		std::vector<unsigned char> headerB = m_block.GetHeaderBytes();
		//calculate midstate
		m_skein.setMessage(headerB);
	}
	//restart the mining loop
	m_stop = false;
	m_run_thread = std::thread(&Worker_hash::run, this);
}

void Worker_hash::run()
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

void Worker_hash::update_statistics(stats::Collector& stats_collector)
{
	std::scoped_lock<std::mutex> lck(m_mtx);

	auto hash_stats = std::get<stats::Hash>(stats_collector.get_worker_stats(m_config.m_internal_id));
	hash_stats.m_hash_count = m_hash_count;
	hash_stats.m_best_leading_zeros = m_best_leading_zeros;
	hash_stats.m_met_difficulty_count = m_met_difficulty_count;

	stats_collector.update_worker_stats(m_config.m_internal_id, hash_stats);

}


bool Worker_hash::difficulty_check()
{
	//perform additional difficulty filtering prior to submitting the nonce 

	//leading zeros in bits required of the hash for it to pass the current difficulty.
	int leadingZerosRequired;
	uint64_t difficultyTest64;
	decodeBits(m_pool_nbits != 0 ? m_pool_nbits : m_block.nBits, leadingZerosRequired, difficultyTest64);
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

uint64_t Worker_hash::leading_zero_mask()
{
	return ((1ull << leading_zeros_required) - 1) << (64 - leading_zeros_required);
}

void Worker_hash::reset_statistics()
{
	m_hash_count = 0;
	m_best_leading_zeros = 0;
	m_met_difficulty_count = 0;
}

}
}