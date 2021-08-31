#ifndef CHAIN_SIEVE_HPP
#define CHAIN_SIEVE_HPP

#include <vector>
#include <atomic>
#include <spdlog/spdlog.h>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/gmp.hpp>
#include "sieve_utils.hpp"

namespace nexusminer {
	namespace gpu
	{
		enum class Fermat_test_status {
			untested,
			fail,
			pass
		};

		static constexpr int maxGap = 12;  //the largest allowable prime gap.

		//a candidate for a dense prime cluster.  A chain consists of a base integer plus a list of offsets. 
		class Chain
		{

		public:
			enum class Chain_state {
				open, //immature chain empty or in process of being built
				closed, //chain is complete and available for fermat testing
				in_process, //fermat testing is in process. 
				complete  //fermat testing is complete. 
			};

			class Chain_offset {
			public:
				int m_offset = 0;  //offset from the base offset
				Fermat_test_status m_fermat_test_status = Fermat_test_status::untested;
				Chain_offset(int offset) :m_offset{ offset }, m_fermat_test_status{ Fermat_test_status::untested }{};
				friend bool operator < (const Chain_offset& c1, const Chain_offset& c2);
				friend bool operator == (const Chain_offset& c1, const Chain_offset& c2);
			};

			Chain();
			Chain(uint64_t base_offset);
			void open(uint64_t base_offset);
			void close();
			int length() { return m_offsets.size(); }
			void get_best_fermat_chain(uint64_t& base_offset, int& offset, int& length);
			bool is_there_still_hope();  //is it possible this chain can result in a valid fermat chain
			bool get_next_fermat_candidate(uint64_t& base_offset, int& offset);
			bool update_fermat_status(bool is_prime);
			void push_back(int offset);
			const std::string str();

			int m_min_chain_length = 8;
			int m_min_chain_report_length = 5;
			Chain_state m_chain_state = Chain_state::open;
			uint64_t m_base_offset = 0;
			std::vector<Chain_offset> m_offsets; //offsets including 0
			int m_gap_in_process = 0;
			int m_next_fermat_test_offset_index = 0;
			int m_prime_count = 0;
			int m_untested_count = 0;

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
			void clear_chains();
			std::vector<std::uint64_t> get_valid_chain_starting_offsets();
			void reset_stats();
			std::vector<std::uint64_t> m_long_chain_starts;

		private:
			class Fermat_test_candidate {
			public:
				uint64_t base_offset = 0;
				int offset = 0;
				Fermat_test_status fermat_test_status = Fermat_test_status::untested;
				uint64_t get_offset() { return base_offset + offset; }
			};

			std::shared_ptr<spdlog::logger> m_logger;

			static constexpr uint8_t sieve30 = 0xFF;  //compressed sieve for primorial 2*3*5 = 30.  Each bit represents a possible prime location in the wheel {1,7,11,13,17,19,23,29} 
			static constexpr int sieve30_offsets[]{ 1,7,11,13,17,19,23,29 };  // each bit in the sieve30 represets an offset from the base mod 30
			static constexpr int sieve30_gaps[]{ 6,4,2,4,2,4,6,2 };
			static constexpr int sieve30_index[]{ -1,0,-1,-1,-1,-1,-1, 1, -1, -1, -1, 2, -1, 3, -1, -1, -1, 4, -1, 5, -1, -1, -1, 6, -1, -1, -1, -1, -1, 7 };  //reverse lookup table (offset mod 30 to index)
			static constexpr int L1_CACHE_SIZE = 32768;
			static constexpr int L2_CACHE_SIZE = 262144;
			
			//upper limit of the sieving primes. 
			static constexpr uint32_t sieving_prime_limit = 3e7; //3e8;
			static constexpr uint32_t sieve_size = L2_CACHE_SIZE * 16;
			//each segment byte covers a range of 30 sieving primes 
			static constexpr uint32_t m_segment_size = sieve_size * 30;
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

			//static constexpr int minChainLength = 8;  //min chain length
			//static constexpr int maxGap = 12;  //the largest allowable prime gap.

			std::vector<Chain> m_chain;
			boost::multiprecision::uint1024_t m_sieve_start;  //starting integer for the sieve.  This must be a multiple of 30.
			bool m_chain_in_process = false;
			Chain m_current_chain;

			static constexpr int m_fermat_test_batch_size = 5000;
			//static constexpr int m_fermat_test_array_size = m_fermat_test_batch_size * 3/2;

			void close_chain();
			void open_chain(uint64_t base_offset);



		public:
			//void do_sieve();
			void find_chains(uint64_t sieve_size, uint64_t low);
			uint64_t count_fermat_primes(uint64_t sieve_size, uint64_t low);
			bool primality_test(boost::multiprecision::uint1024_t p);
			void test_chains();
			void primality_batch_test();
			void primality_batch_test_cpu();
			void clean_chains();
			uint64_t get_current_chain_list_length();
			int get_fermat_test_batch_size() { return m_fermat_test_batch_size; }

			//stats
			std::vector<std::uint32_t> m_chain_histogram;
			uint64_t m_fermat_test_count = 0;
			uint64_t m_fermat_prime_count = 0;
			uint64_t m_chain_count = 0;
			int m_chain_candidate_max_length = 0;
			uint64_t m_chain_candidate_total_length = 0;
		};

	}
}

#endif