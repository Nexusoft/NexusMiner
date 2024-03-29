#ifndef NEXUSMINER_GPU_SIEVE_HPP
#define NEXUSMINER_GPU_SIEVE_HPP

#include <vector>
#include <atomic>
#include <spdlog/spdlog.h>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/gmp.hpp>
#include "sieve_utils.hpp"
#include "chain.hpp"
//#include "../cuda_prime/fermat_test.hpp"
#include "../cuda_prime/fermat_prime/fermat_prime.hpp"
#include "../cuda_prime/sieve.hpp"
#include "gpu/prime_common.hpp"

namespace nexusminer {
	namespace gpu
	{

		class Sieve
		{
		public:
			using sieve_word_t = Cuda_sieve::sieve_word_t;
			Sieve();
			void generate_sieving_primes();
			void generate_small_prime_tables();
			void set_sieve_start(boost::multiprecision::uint1024_t);
			boost::multiprecision::uint1024_t get_sieve_start();
			void calculate_starting_multiples();
			void gpu_sieve_load(uint16_t device);
			void gpu_sieve_init();
			void gpu_fermat_test_init(uint16_t device);
			void gpu_sieve_free();
			void gpu_fermat_free();
			void gpu_fermat_test_set_base_int(boost::multiprecision::uint1024_t base_big_int);
			uint64_t gpu_get_prime_candidate_count();
			void gpu_get_sieve();
			void gpu_sieve_small_primes(uint64_t sieve_start_offset);
			void gpu_sieve_large_primes(uint64_t sieve_start_offset);
			void gpu_sieve_medium_small_primes(uint64_t sieve_start_offset);
			void sieve_small_primes();
			void sieve_batch(uint64_t low);
			void sieve_batch_cpu(uint64_t low);
			//std::uint32_t get_segment_size();
			std::uint32_t get_segment_batch_size();
			void reset_sieve();
			void reset_sieve_batch(uint64_t low);
			void reset_batch_run_count();
			void clear_chains();
			void reset_stats();
			void find_chains();
			void get_chains();
			void sort_chains();
			void get_long_chains();
			void gpu_clean_chains();
			void gpu_run_fermat_chain_test();
			void gpu_run_trial_division_chain_test();
			void gpu_get_fermat_stats(uint64_t& tests, uint64_t& passes, uint64_t& trial_division_tests, uint64_t& trial_division_composites);
			void gpu_reset_fermat_stats();
			uint32_t get_chain_count();
			uint64_t count_fermat_primes(int sample_size, uint16_t device);
			uint64_t count_fermat_primes_cpu(int sample_size);
			bool primality_test(boost::multiprecision::uint1024_t p);
			void test_chains();
			void primality_batch_test(uint16_t device);
			void primality_batch_test_cpu();
			void clean_chains();
			uint64_t get_current_chain_list_length();
			uint64_t get_cuda_chain_list_length();
			int get_fermat_test_batch_size() { return m_fermat_test_batch_size; }
			double probability_is_prime_after_sieve(double bits = 1024);
			double sieve_pass_through_rate_expected();
			double expected_chain_density(int n, int bits);
			uint64_t count_prime_candidates();
			std::vector<sieve_word_t> get_sieve(); //return a copy of the raw sieve
			std::vector<uint64_t> get_prime_candidate_offsets();
			std::vector<uint32_t> get_sieving_primes();
			//bool chain_trial_division(Chain& chain);
			//void do_chain_trial_division_check();
			void gpu_get_stats();
			void gpu_sieve_synchronize();
			void gpu_fermat_synchronize();
			Cuda_sieve::Cuda_sieve_properties get_sieve_properties();
			void generate_trial_divisors();

		private:
			//mod 30 wheel using primorial 2*3*5 = 30.  Each bit represents a possible prime location in the wheel {1,7,11,13,17,19,23,29} 
			static constexpr int sieve30_offsets[]{ 1,7,11,13,17,19,23,29 };  // each bit in the sieve30 represets an offset from the base mod 30
			static constexpr int sieve30_gaps[]{ 6,4,2,4,2,4,6,2 };
			static constexpr int sieve30_index[]{ -1,0,-1,-1,-1,-1,-1, 1, -1, -1, -1, 2, -1, 3, -1, -1, -1, 4, -1, 5, -1, -1, -1, 6, -1, -1, -1, -1, -1, 7 };  //reverse lookup table (offset mod 30 to index)

		public:
			
			//const uint32_t sieve_size_bytes = Cuda_sieve::m_kernel_sieve_size_bytes;  //size of the sieve in bytes
			//const uint32_t sieve_size_words = Cuda_sieve::m_kernel_sieve_size_words;  //size of the sieve in words
			//const uint32_t sieve_size = Cuda_sieve::m_kernel_sieve_size_words;  
			static constexpr uint32_t m_sieve_range_per_word = Cuda_sieve::m_sieve_word_range;  
			static constexpr uint32_t m_sieve_range_per_byte = Cuda_sieve::m_sieve_byte_range;
			static constexpr uint32_t m_sieve_bytes_per_word = Cuda_sieve::m_sieve_word_byte_count;

			std::vector<std::uint64_t> m_long_chain_starts;
			uint64_t m_sieve_batch_start_offset;
			uint32_t m_small_prime_limit;
			uint32_t m_medium_small_prime_limit;
			uint32_t m_sieving_prime_limit;
			uint32_t m_large_prime_limit;
			uint32_t m_trial_division_prime_limit;
			std::vector<Cuda_sieve::sieve_word_t> m_sieve_results;  //accumulated results of sieving
			static constexpr int m_fermat_test_batch_size = 200000;
			static constexpr int m_fermat_test_batch_size_max = 1000000;
			static constexpr int m_segment_batch_size = Cuda_sieve::m_kernel_segments_per_block * Cuda_sieve::m_num_blocks; //number of segments to sieve in one batch
			uint64_t m_sieve_range;
			static constexpr int m_min_chain_length = 8;
			static constexpr uint32_t large_prime_count = 0;

			//stats
			std::vector<std::uint32_t> m_chain_histogram;
			uint64_t m_fermat_test_count = 0;
			uint64_t m_fermat_prime_count = 0;
			uint64_t m_chain_count = 0;
			int m_chain_candidate_max_length = 0;
			uint64_t m_chain_candidate_total_length = 0;
			uint64_t m_trial_division_chains_busted = 0;
			double m_best_chain = 0;
			

		private:
			class Fermat_test_candidate {
			public:
				uint64_t base_offset = 0;
				int offset = 0;
				Fermat_test_status fermat_test_status = Fermat_test_status::untested;
				uint64_t get_offset() { return base_offset + offset; }
			};

			std::shared_ptr<spdlog::logger> m_logger;

			//each byte covers a range of 30 sieving primes 
			//const uint32_t m_segment_size = sieve_size_bytes * Cuda_sieve::m_sieve_byte_range;

			//the sieve.  each bit that is set represents a possible prime.
			std::vector<Cuda_sieve::sieve_word_t> m_sieve;

			std::vector<uint8_t>m_small_primes;
			std::vector<uint16_t>m_small_prime_offsets;
			std::vector<uint32_t> m_sieving_primes;
			std::vector<uint32_t> m_multiples;
			std::vector<Prime_plus_multiple_32> m_medium_primes_plus_multiples;
			std::vector<uint32_t> m_medium_small_primes;
			std::vector<uint32_t> m_medium_small_multples;
			std::vector<uint32_t> m_large_sieving_primes;
			std::vector<uint32_t> m_large_multiples;
			std::vector<uint32_t> m_small_prime_lookup_table;
			std::vector<trial_divisors_uint32_t> m_trial_divisors;

			std::vector<Chain> m_chain;
			std::vector<CudaChain>m_cuda_chains;
			//std::vector<double>m_large_prime_mod_constants;

			boost::multiprecision::uint1024_t m_sieve_start;  //starting integer for the sieve.  This must be a multiple of 30.
			bool m_chain_in_process = false;
			Chain m_current_chain;
			
			uint64_t m_sieve_run_count = 0;

			Cuda_sieve m_cuda_sieve;
			//Cuda_fermat_test m_cuda_prime_test;
			Fermat_prime m_cuda_prime_test;

			bool m_cuda_sieve_allocated = false;

		};
	}
}

#endif