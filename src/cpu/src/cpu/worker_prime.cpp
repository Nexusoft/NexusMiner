#include "..\..\inc\cpu\worker_prime.hpp"
#include "..\..\inc\cpu\worker_prime.hpp"
#include "..\..\inc\cpu\worker_prime.hpp"
#include "cpu/worker_prime.hpp"
#include "config/config.hpp"
#include "stats/stats_collector.hpp"
#include "prime/prime.hpp"
#include "block.hpp"
#include <asio.hpp>
#include <primesieve.hpp>
#include <sstream> 

namespace nexusminer
{
namespace cpu
{
Worker_prime::Worker_prime(std::shared_ptr<asio::io_context> io_context, config::Worker_config& config)
	: m_io_context{ std::move(io_context) }
	, m_logger{ spdlog::get("logger") }
	, m_config{ config }
	, m_prime_helper{std::make_unique<Prime>()}
	, m_stop{ true }
	, m_log_leader{ "CPU Worker " + m_config.m_id + ": " }
	, m_primes{ 0 }
	, m_chains{ 0 }
	, m_difficulty{ 0 }
	, m_pool_nbits{ 0 }
{
	//m_prime_helper->InitializePrimes();
	m_chain_histogram = std::vector<std::uint32_t>(10, 0);
}

Worker_prime::~Worker_prime() noexcept
{
	//make sure the run thread exits the loop
	m_stop = true;
	if (m_run_thread.joinable())
		m_run_thread.join();
}

void Worker_prime::set_block(LLP::CBlock block, std::uint32_t nbits, Worker::Block_found_handler result)
{
	//stop the existing mining loop if it is running
	m_stop = true;
	if (m_run_thread.joinable())
	{
		m_run_thread.join();
	}

	{
		std::scoped_lock<std::mutex> lck(m_mtx);
		m_found_nonce_callback = result;
		m_block = Block_data{ block };
		if (nbits != 0)	// take nBits provided from pool
		{
			m_pool_nbits = nbits;
		}

		m_difficulty = m_pool_nbits != 0 ? m_pool_nbits : m_block.nBits;
		bool excludeNonce = true;  //prime block hash excludes the nonce
		std::vector<unsigned char> headerB = m_block.GetHeaderBytes(excludeNonce);
		//calculate the block hash
		NexusSkein skein;
		skein.setMessage(headerB);
		skein.calculateHash();
		NexusSkein::stateType hash = skein.getHash();

		//keccak
		NexusKeccak keccak(hash);
		keccak.calculateHash();
		NexusKeccak::k_1024 keccakFullHash_i = keccak.getHashResult();
		keccakFullHash_i.isBigInt = true;
		uint1k keccakFullHash("0x" + keccakFullHash_i.toHexString(true));
		m_base_hash = keccakFullHash;
		//Now we have the hash of the block header.  We use this to feed the miner. 

		//set the starting nonce for each worker to something different that won't overlap with the others
		m_starting_nonce = static_cast<uint64_t>(m_config.m_internal_id) << 48;
		m_nonce = m_starting_nonce;
		
		//m_nonce = big_nonce.convert_to<uint64_t>();
		m_logger->debug("starting nonce: {}", m_nonce);
	}
	//restart the mining loop
	m_stop = false;
	m_run_thread = std::thread(&Worker_prime::run, this);
}

void Worker_prime::run()
{
	std::uint64_t i = 0;
	while (!m_stop)
	{
		uint1k startprime = m_base_hash + m_sieveRange * i;
		startprime = startprime + 1 - (startprime % 2);  //ensure odd start
		//std::cout << "Sieving range " << m_sieveRange * i << " to " << m_sieveRange * (i + 1) - 1 << std::endl;
		generate_seive(startprime);
		analyze_chains();
		mine_region(startprime);
		i++;
	}




	
}

double Worker_prime::getDifficulty(uint1k p)
{
	std::vector<unsigned int> offsets_to_test;
	LLC::CBigNum prime_to_test = boost_uint1024_t_to_CBignum(p);
	double difficulty = m_prime_helper->GetPrimeDifficulty(prime_to_test, 1, offsets_to_test);
	return difficulty;
}

double Worker_prime::getNetworkDifficulty()
{
	return m_difficulty / 10000000.0;
}

bool Worker_prime::difficulty_check(uint1k p)
{
	return getDifficulty(p) >= getNetworkDifficulty();
}

bool Worker_prime::isPrime(uint1k p)
{
	uint1k base = 2;
	uint1k result;
	result = boost::multiprecision::powm(base, p - 1, p);
	return (result == 1);
}

void Worker_prime::generate_seive(uint1k sieve_start)
{
	m_sieve = {};
	//sieve contains locations of possible primes.Only odd numbers are considered.
	//sieve6 is based on the primorial 2 * 3 = 6, and is 6 positions long(3 odd positions 1, 3, 5)
	std::vector<bool> sieve6{ true, false, true };
	//sieve30 is based on the primorial 2 * 3 * 5 = 30, is 30 positions long(15 odd positions).Generate it by replicating sieve6 five times and crossing out the multiples of 5.
	std::vector<bool> sieve30{ true, false, false, true, false, true, true, false, true, true, false, true, false, false, true };
	//the next sieve after sieve6 starts with the next prime, 5
	int startPrime = 5;
	//5 is the third prime
	int startPrimePosition = 3;
	int primorialFinalPrime = startPrime;  // last prime in the primorial chain
	//uint1k T("0x0000005ff320ec9f9599b9cb0156c793f61060c8a8c49185df9d25603e37259c2f0213d6d96745bbbbe7ea1e4e9da371aeeb5d20c204c22a038b10957b53c67d9eb3a00acfaeb6ccd4c231a8088d5a5745e19f70387a7d91463d9b318a1f0503819a32f5fa32cf3579c7d6a3546cbdceaa364cfa2e989defeb4f5fe29de687cc");
	uint64_t nonce = 4933493377870005061;
	double difficulty = 8.268849;
	//uint1k sieveStart = T + nonce - 2;
	//sieveStart = T
	//sieveStart += (sieveStart + 1) % 2;  //start on an odd number
	//sieveStart = T + 1
	//sieveStart = 1;
	uint1k sieveStart = sieve_start;
	//Initialize the sieve by replicating sieve6
	int replicate = m_maxSieveLength / 3 + (m_maxSieveLength % 3 != 0);

	//sieve might be a few elements longer than maxsievelength
	for (auto i = 0; i < replicate; i++)
	{
		m_sieve.insert(m_sieve.end(), sieve6.begin(), sieve6.end());
	}
	//adjust for sieve start location not on even multiple of 6
	int rotate_left_by = (((sieveStart - 1) % 6) >> 1).convert_to<int>();
	std::rotate(m_sieve.begin(), m_sieve.begin() + rotate_left_by, m_sieve.end());

	//create the sieve
	primesieve::iterator it(startPrime);
	uint64_t p = startPrime;
	uint64_t prime_index = 3;

	//iterate through the sequence of primes between the 3rd prime(5) and the desired primorial end prime
	while (prime_index <= m_primorialEndPrime)
	{
		//offset to the start of the sieve.  round up to the nearest odd integer multiple of the current prime.
		uint1k startIndex_big = (((sieveStart + p - 1) / p) * p);  //round up to nearest integer multiple
		//if it is even add another prime to get to the next odd multiple
		startIndex_big += ((startIndex_big + 1) % 2) * p;
		//subtract start offset
		int startIndex = (startIndex_big - sieveStart).convert_to<int>();
		//std::cout << "Prime: " << p << " Start Index: " << startIndex << std::endl;
		//cross out multiples of the current prime. We are incrementing by p * 2 to skip even numbers
		for (auto i = startIndex; i < m_sieveLength * 2; i = i + p * 2)
		{
			m_sieve[(i >> 1)] = false;
		}
		primorialFinalPrime = p;
		p = it.next_prime();
		prime_index++;
	}
	//std::cout << "Sieve Done" << std::endl;
	//std::cout << "Sieve Length: " << sieveLength << std::endl;
	//std::cout << "Primorial Final Prime: " << primorialFinalPrime << std::endl;
}

void Worker_prime::analyze_chains()
//search the seive for chains with minimum length
{

	m_chainStartPosArray = {};
	m_chainLengthArray = {};
	m_chainOffsets = {};
	m_chainCount = 0;
	m_candidateCount = 0;

	int chainLength = 0;     //measures chain length of current chain in process

	std::vector<int> lastPos{ 0, -1, -1, -1 }; //#last four posistions. lastpos[0] is the last position just to the left of the current position. - 1 means not valid
	//int lastGap = 0;
	std::vector<int> currentChainOffsets;



	//start position of chain in process
	//find the first true in the sieve
	int chainStartPos = 0;
	for (auto i = 0; i < m_sieveLength; i++)
	{
		if (m_sieve[i])
		{
			chainStartPos = i;
			break;
		}
	}


	//go through the sieve and cross out primes not part of a chain
	for (auto i = 0; i < m_sieveLength; i++)
	{
		if (m_sieve[i])
		{
			m_candidateCount += 1;
			if (i - lastPos[0] <= m_maxGap)  //chain is still good
			{
				currentChainOffsets.push_back(i - chainStartPos);
				chainLength += 1;
			}
			else // found a large gap.This is the end of the chain
			{
				if (m_minChainLength <= chainLength)
				{
					//save the chain
					m_chainStartPosArray.push_back(chainStartPos);
					m_chainLengthArray.push_back(chainLength);
					m_chainOffsets.push_back(currentChainOffsets);
					m_chainCount += 1;
				}
				else
				{
					//The chain is too short.

				}
				//start a new chain
				chainLength = 1;
				chainStartPos = i;
				currentChainOffsets = { 0 };
			}
			lastPos[0] = i;
		}
		else
		{
			//not a valid candidate
		}
	}

	//add the final chain if it is valid
	if (m_minChainLength <= chainLength)
	{
		//save the chain
		m_chainStartPosArray.push_back(chainStartPos);
		m_chainLengthArray.push_back(chainLength);
		m_chainOffsets.push_back(currentChainOffsets);
		m_chainCount += 1;
	}

	int sieveRange = m_sieveLength * 2;
	//std::cout << "Range: " << sieveLength * 2 << std::endl;
	//std::cout << "Candidates: " << candidateCount << std::endl;
	//std::cout << "Candidate Ratio: " << candidateCount / sieveRange << std::endl;
	int candidatesEliminated = sieveRange - m_candidateCount;
	//std::cout << "Candidates Eliminated: " << candidatesEliminated << std::endl;
	//std::cout << "Chains greater than or equal to " << m_minChainLength << " : " << m_chainCount << std::endl;
	//print("Million candidates eliminated per second:", candidatesEliminated / 1e6 / sieveCreationTime)
	int maxChainLength = *std::max_element(m_chainLengthArray.begin(), m_chainLengthArray.end());
	//std::cout << "Max chain length: " << maxChainLength << std::endl;

	//print a few chains
	/*int showChains = 0;
	for (auto i = 0; i < showChains; i++)
	{
		int chainLen = m_chainLengthArray[i];
		std::cout << "Length " << chainLen << " chain base offset " << (m_chainStartPosArray[i] * 2) + 1 << " offsets [";
		for (auto j = 0; j < chainLen; j++)
		{
			std::cout << m_chainOffsets[i][j] * 2 << ", ";
		}
		std::cout << "]" << std::endl;
	}*/
}

void Worker_prime::mine_region(uint1k start_here)
{
	int chain_index = 0;
	uint64_t offset_from_start = m_chainStartPosArray[chain_index] * 2;
	int offset_index = 0;
	uint1k first_prime_candidate = start_here + offset_from_start;
	uint1k prime_candidate = first_prime_candidate;
	int chain_length = 0;
	int target_length = 2;
	int chain_start_offset_index = 0;

	//int primorial_index = 0;
	int gap = 0;
	bool done_with_this_chain = false;
	while (chain_index < m_chainCount)
	{
		//m_logger->debug("testing offset {} gap {}", m_prime_candidate_offsets[offset_index], gap);
		if (isPrime(prime_candidate))
		{
			//std::cout << "found a prime" << std::endl;
			m_primes++;
			if (chain_length == 0)
			{
				chain_start_offset_index = offset_index;
			}
			chain_length += 1;
			gap = 0;

		}
		offset_index += 1;
		if (offset_index < m_chainLengthArray[chain_index])
		{
			int delta = (m_chainOffsets[chain_index][offset_index] - m_chainOffsets[chain_index][offset_index - 1]) * 2;
			gap += delta;
			prime_candidate += delta;

			if (gap > 12)
			{
				done_with_this_chain = true;
			}
		}
		else
		{
			done_with_this_chain = true;
		}
		if (done_with_this_chain)
		{
			if (chain_length >= 2)
			{
				m_chains++;
				offset_from_start = m_chainStartPosArray[chain_index] * 2;
				m_logger->debug("found a chain of length {} at offset {}", chain_length, offset_from_start);
				m_chain_histogram[chain_length]++;
				if (chain_length >= target_length)
				{
					uint1k chain_start = start_here + offset_from_start + m_chainOffsets[chain_index][chain_start_offset_index]*2;
					uint1k bigNonce = chain_start - m_base_hash;
					m_block.nNonce = bigNonce.convert_to<uint64_t>();
					m_logger->debug("actual difficulty {} required {}", getDifficulty(chain_start), getNetworkDifficulty());
					if (difficulty_check(chain_start))
					{
						//++m_met_difficulty_count;
						//update the block with the nonce and call the callback function;
						
						//std::cout << "nonce: " << m_block.nNonce << " block hash " << m_base_hash.str() << std::endl;
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
			}
			//m_logger->debug("done with chain.  found {} primes.", chain_length);
			chain_index++;
			prime_candidate = start_here + m_chainStartPosArray[chain_index] * 2;
			chain_length = 0;
			offset_index = 0;
			gap = 0;
			chain_start_offset_index = 0;
			done_with_this_chain = false;
			
		}
	}
}

LLC::CBigNum Worker_prime::boost_uint1024_t_to_CBignum(uint1k p)
{
	std::stringstream ss;
	ss << std::hex << p;
	std::string p_hex_str = ss.str();
	LLC::CBigNum p_CBignum;
	p_CBignum.SetHex(p_hex_str);
	return p_CBignum;
}

void Worker_prime::update_statistics(stats::Collector& stats_collector)
{
	auto prime_stats = std::get<stats::Prime>(stats_collector.get_worker_stats(m_config.m_internal_id));
	prime_stats.m_primes = m_primes;
	prime_stats.m_chains = m_chains;
	prime_stats.m_difficulty = m_difficulty;
	prime_stats.m_chain_histogram = m_chain_histogram;


	stats_collector.update_worker_stats(m_config.m_internal_id, prime_stats);

	m_primes = 0;
	m_chains = 0;
}


}
}