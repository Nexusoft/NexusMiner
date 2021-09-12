#ifndef CHAIN_SIEVE_HPP
#define CHAIN_SIEVE_HPP

#include <vector>
#include <atomic>
#include <spdlog/spdlog.h>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/gmp.hpp>
#include "sieve_utils.hpp"
#include "../cuda_prime/fermat_test.cuh"
#include "../cuda_prime/sieve.cuh"

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
			void gpu_sieve_init(uint16_t device);
			void gpu_sieve_free();
			void sieve_segment();
			void sieve_batch(uint64_t low);
			void sieve_batch_cpu(uint64_t low);
			std::uint32_t get_segment_size();
			std::uint32_t get_segment_batch_size();
			void reset_sieve();
			void reset_sieve_batch(uint64_t low);
			void clear_chains();
			void reset_stats();
			void find_chains(uint64_t low, bool batch_sieve_mode);
			uint64_t count_fermat_primes(int sample_size);
			uint64_t count_fermat_primes_cpu(int sample_size);
			bool primality_test(boost::multiprecision::uint1024_t p);
			void test_chains();
			void primality_batch_test();
			void primality_batch_test_cpu();
			void clean_chains();
			uint64_t get_current_chain_list_length();
			int get_fermat_test_batch_size() { return m_fermat_test_batch_size; }
			double probability_is_prime_after_sieve();
			double sieve_pass_through_rate_expected();
			uint64_t count_prime_candidates();

		private:
			//static constexpr uint8_t sieve30 = 0xFF;  //compressed sieve for primorial 2*3*5 = 30.  Each bit represents a possible prime location in the wheel {1,7,11,13,17,19,23,29} 
			static constexpr int sieve30_offsets[]{ 1,7,11,13,17,19,23,29 };  // each bit in the sieve30 represets an offset from the base mod 30
			static constexpr int sieve30_gaps[]{ 6,4,2,4,2,4,6,2 };
			static constexpr int sieve30_index[]{ -1,0,-1,-1,-1,-1,-1, 1, -1, -1, -1, 2, -1, 3, -1, -1, -1, 4, -1, 5, -1, -1, -1, 6, -1, -1, -1, -1, -1, 7 };  //reverse lookup table (offset mod 30 to index)
			static constexpr int L1_CACHE_SIZE = 32768;
			static constexpr int L2_CACHE_SIZE = 262144;

		public:
			const uint32_t sieve_size = CudaSieve::m_kernel_sieve_size;  //size of the sieve in bytes

			std::vector<std::uint64_t> m_long_chain_starts;
			uint64_t m_sieve_batch_start_offset;
			uint32_t m_sieving_prime_limit = 3e6; //3e8;
			std::vector<uint8_t> m_sieve_results;  //accumulated results of sieving
			int m_fermat_test_batch_size = 10000;
			int m_segment_batch_size = CudaSieve::m_kernel_segments_per_block* CudaSieve::m_num_blocks; //number of segments to sieve in one batch
			uint32_t m_sieve_batch_buffer_size = sieve_size * m_segment_batch_size;

			//stats
			std::vector<std::uint32_t> m_chain_histogram;
			uint64_t m_fermat_test_count = 0;
			uint64_t m_fermat_prime_count = 0;
			uint64_t m_chain_count = 0;
			int m_chain_candidate_max_length = 0;
			uint64_t m_chain_candidate_total_length = 0;

		private:
			class Fermat_test_candidate {
			public:
				uint64_t base_offset = 0;
				int offset = 0;
				Fermat_test_status fermat_test_status = Fermat_test_status::untested;
				uint64_t get_offset() { return base_offset + offset; }
			};

			std::shared_ptr<spdlog::logger> m_logger;

			//each 8 bytes covers a range of 30 sieving primes 
			const uint32_t m_segment_size = sieve_size/8 * 30;
			//we start sieving at 7
			static constexpr int sieving_start_prime = 7;
			static constexpr int m_min_chain_length = 8;	

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


			//how many bits are set in a byte
			static constexpr int popcnt[256] =
			{
			  0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
			  1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
			  1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
			  2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
			  1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
			  2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
			  2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
			  3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
			  1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
			  2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
			  2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
			  3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
			  2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
			  3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
			  3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
			  4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8
			};

			//the sieve.  each bit that is set represents a possible prime.
			std::vector<uint8_t> m_sieve;
			std::vector<uint32_t> m_sieving_primes;
			std::vector<uint32_t> m_multiples;
			std::vector<uint32_t> m_prime_mod_inverses;
			std::vector<Chain> m_chain;

			boost::multiprecision::uint1024_t m_sieve_start;  //starting integer for the sieve.  This must be a multiple of 30.
			bool m_chain_in_process = false;
			Chain m_current_chain;
			
			void close_chain();
			void open_chain(uint64_t base_offset);
			uint64_t sieve_run_count = 0;

			CudaSieve m_cuda_sieve;

			//experimental
		};
	}
}

#endif