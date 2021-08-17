#ifndef CHAIN_SIEVE_HPP
#define CHAIN_SIEVE_HPP

#include <vector>
#include <atomic>
#include <spdlog/spdlog.h>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/gmp.hpp>
#include "sieve_utils.hpp"

namespace nexusminer {
namespace cpu
{

	class Chain
	{

	public:
		Chain();
		bool zero_length = true;
		uint64_t base_offset = 0;
		boost::multiprecision::uint1024_t base = 0;
		std::vector<uint8_t> offsets; //offsets excluding 0
		int length() { return zero_length ? 0 : offsets.size() + 1; }
	private:


	};



	class Sieve
	{
	public:
		Sieve();
		void generate_sieving_primes();
		void set_sieve_start(boost::multiprecision::uint1024_t);
		boost::multiprecision::uint1024_t get_sieve_start();
		void calculate_starting_multiples();
		void sieve_segment();
		std::uint32_t get_segment_size();
		void reset_sieve();
		std::vector<std::uint64_t> get_valid_chain_starting_offsets();
		void set_chain_length_threshold(int min_chain_length);
		void reset_stats();


	private:
		std::shared_ptr<spdlog::logger> m_logger;

		static constexpr uint8_t sieve30 = 0xFF;  //compressed sieve for primorial 2*3*5 = 30.  Each bit represents a possible prime location in the wheel {1,7,11,13,17,19,23,29} 
		static constexpr int sieve30_offsets[]{ 1,7,11,13,17,19,23,29 };  // each bit in the sieve30 represets an offset from the base mod 30
		static constexpr int sieve30_gaps[]{ 6,4,2,4,2,4,6,2 };
		static constexpr int sieve30_index[]{ -1,0,-1,-1,-1,-1,-1, 1, -1, -1, -1, 2, -1, 3, -1, -1, -1, 4, -1, 5, -1, -1, -1, 6, -1, -1, -1, -1, -1, 7 };  //reverse lookup table (offset mod 30 to index)
		static constexpr int L1_CACHE_SIZE = 32768;
		static constexpr int L2_CACHE_SIZE = 262144;
		//upper limit of the sieving range
		static constexpr uint64_t sieve_range = 3e10;//3e9;
		//upper limit of the sieving primes. 
		static constexpr uint32_t sieving_prime_limit = 3e8; //3e8;
		static constexpr uint32_t sieve_size = L2_CACHE_SIZE * 16;
		//each segment byte covers a range of 30 sieving primes 
		static constexpr uint32_t m_segment_size = sieve_size * 30;
		//number of segments needed to cover the sieving range
		static constexpr int segments = sieve_range / m_segment_size + (sieve_range % m_segment_size != 0);
		//we start sieving at 7
		static constexpr int sieving_start_prime = 7;

		/// Bitmasks used to unset bits
		static constexpr uint8_t unset_bit_mask[30] =
		{
			(uint8_t)~(1 << 0), (uint8_t)~(1 << 0),
			(uint8_t)~(1 << 1), (uint8_t)~(1 << 1), (uint8_t)~(1 << 1), (uint8_t)~(1 << 1), (uint8_t)~(1 << 1), (uint8_t)~(1 << 1),
			(uint8_t)~(1 << 2), (uint8_t)~(1 << 2), (uint8_t)~(1 << 2), (uint8_t)~(1 << 2),
			(uint8_t)~(1 << 3), (uint8_t)~(1 << 3),
			(uint8_t)~(1 << 4), (uint8_t)~(1 << 4), (uint8_t)~(1 << 4), (uint8_t)~(1 << 4),
			(uint8_t)~(1 << 5), (uint8_t)~(1 << 5),
			(uint8_t)~(1 << 6), (uint8_t)~(1 << 6), (uint8_t)~(1 << 6), (uint8_t)~(1 << 6),
			(uint8_t)~(1 << 7), (uint8_t)~(1 << 7), (uint8_t)~(1 << 7), (uint8_t)~(1 << 7), (uint8_t)~(1 << 7), (uint8_t)~(1 << 7)
		};


		//the sieve.  each bit that is set represents a possible prime.
		std::vector<uint8_t> m_sieve;
		std::vector<uint32_t> m_sieving_primes;
		std::vector<uint32_t> m_multiples;
		std::vector<int> m_wheel_indices;
		//
		//std::vector<uint64_t> m_multiples;

		static constexpr int minChainLength = 8;  //min chain length
		static constexpr int maxGap = 12;  //the largest allowable prime gap.

		std::vector<Chain> m_chain;
		std::vector<std::uint64_t> m_long_chain_starts;
		boost::multiprecision::uint1024_t m_sieve_start;  //starting integer for the sieve.  This must be a multiple of 30.
		bool m_chain_in_process = false;
		Chain m_current_chain;
		int m_gap_in_process = 0;
		
		int m_min_chain_length_threshold = 3;

		

		void close_chain();
		void open_chain(uint64_t base_offset);



	public:
		//void do_sieve();
		void find_chains(uint64_t sieve_size, uint64_t low);
		uint64_t count_fermat_primes(uint64_t sieve_size, uint64_t low);
		bool primality_test(boost::multiprecision::uint1024_t p);
		void test_chains();

		//stats
		std::vector<std::uint32_t> m_chain_histogram;
		uint64_t m_fermat_test_count = 0;
		uint64_t m_fermat_prime_count = 0;
		uint64_t m_chain_count = 0;
	};

}
}

#endif