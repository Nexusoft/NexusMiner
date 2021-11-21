#ifndef NEXUSMINER_GPU_CUDA_SIEVE_HPP
#define NEXUSMINER_GPU_CUDA_SIEVE_HPP

#include "cuda_chain.cuh"
#include <stdint.h>
#include <memory>
#include <cmath>

namespace nexusminer {
	namespace gpu {

		struct Prime_plus_multiple_32
		{
			uint32_t prime;
			uint32_t multiple;
		};

		class Cuda_sieve_impl;
		class Cuda_sieve
		{
		public:
			//We choose to use a 32 bit word as the smallest unit of the sieve. Cuda works natively with 32 bit words. 
			//Using a 64 bit words is slower and byte operations are not natively supported by builtin cuda functions (atomics, popcount, etc).
			//One 32 bit sieve word represents a span of 30*4 = 120 integers.  
			using sieve_word_t = uint32_t;
			static const int m_sieve_word_byte_count = sizeof(sieve_word_t);
			static const int m_sieve_byte_range = 30;
			static const int m_sieve_word_range = m_sieve_byte_range * m_sieve_word_byte_count;
			//The wheel formed by primorial 2*3*5*7*11 = 2310 has two gaps of 14.  If we align the sieve start/stop to one of these gaps, 
			// we guarantee that no chains can cross through the segment boundary.
			static const int m_sieve_chain_search_boundary = 2310;
			static const int m_sieve_alignment = m_sieve_chain_search_boundary * m_sieve_word_byte_count;  //ensure the segment ends on a word boundary
			static const int m_sieve_alignment_offset = 120;  //offset from the wheel start to the first gap greater than 12.  Coincidentally its span 120 is a whole word.
			//The span of the primorial 30030 is represented by 30030/30 = 1001 bytes which conveniently is just below 1KB
			//We size the sieve segment to fill the block shared memory which is 48KB minimum.  Newer hardware supports larger shared memory.   
			//TODO: set the sieve size based on hardware capability.
			static const int m_kernel_sieve_size_bytes = 1001 * 48;  //this is the size of the sieve segment in bytes. It should be a multiple of 4 for a 32 bit word sieve.
			static const int m_kernel_sieve_size_words = m_kernel_sieve_size_bytes / m_sieve_word_byte_count;
			static const int m_segment_range = m_kernel_sieve_size_words * m_sieve_word_range;
			static const int m_kernel_segments_per_block = 32;  //number of times to repeat the sieve within a kernel call
			static const int m_kernel_sieve_size_words_per_block = m_kernel_sieve_size_words * m_kernel_segments_per_block;
			static const uint64_t m_block_range = m_segment_range * m_kernel_segments_per_block;
			static const int m_threads_per_block = 1024;
			static const int m_num_blocks = 360;  //each block sieves part of the range
			static const uint64_t m_sieve_total_size = m_kernel_sieve_size_words_per_block * m_num_blocks; //size of the sieve in words
			static const uint64_t m_sieve_range = m_sieve_total_size * m_sieve_word_range;
			static const int m_estimated_chains_per_million = 4;
			static const uint32_t m_max_chains = 2*m_estimated_chains_per_million*m_sieve_range/1e6;
			static const uint32_t m_max_long_chains = 32;
			static const int m_min_chain_length = 8;
			static const int m_start_prime = 7;
			static constexpr int m_small_prime_count = 14; //61 is the 15th prime starting at 7.  61 is first prime that hits each sieve word no more than 1 time.
			//If you change the small_prime_count, make sure you also change the hardcoded list of primes in the small prime sieve in sieve_impl.cu
//primes 7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103
//       1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22, 23,24
			static constexpr int m_medium_small_prime_count = 32 * 18;
			static constexpr int m_medium_prime_count = 32 * 2500;
			//static constexpr int m_medium_large_prime_count = 32 * 4500;
			static constexpr int m_large_prime_count = 32 * 200000;
			//static constexpr int m_large_prime_2_count = 32 * 140000;
			static const int chain_histogram_max = 10;  
			static const uint64_t m_bucket_ram_budget = 4.0e9;  //bytes avaialble for storing bucket data
			static const int m_large_prime_bucket_size = m_large_prime_count == 0 ? 1 : m_bucket_ram_budget/(m_num_blocks * m_kernel_segments_per_block)/4;
			

			Cuda_sieve();
			~Cuda_sieve();
			void load_sieve(uint32_t primes[], uint32_t prime_count, uint32_t large_primes[], uint32_t medium_small_primes[],
				uint32_t small_prime_masks[], uint32_t small_prime_mask_count, uint8_t small_primes[], uint32_t sieve_size, uint16_t device);
			void init_sieve(uint32_t starting_multiples[], uint16_t small_prime_offsets[], uint32_t large_prime_starting_multiples[],
				uint32_t medium_small_prime_starting_multiples[]);
			void reset_stats();
			void free_sieve();
			void run_small_prime_sieve(uint64_t sieve_start_offset);
			void run_large_prime_sieve(uint64_t sieve_start_offset);
			void run_sieve(uint64_t sieve_start_offset);
			void run_medium_small_prime_sieve(uint64_t sieve_start_offset);
			void find_chains();
			void clean_chains();
			void get_chains(CudaChain chains[], uint32_t& chain_count);
			void get_long_chains(CudaChain chains[], uint32_t& chain_count);
			void get_chain_count(uint32_t& chain_count);
			void get_chain_pointer(CudaChain*& chains_ptr, uint32_t*& chain_count_ptr);
			void get_sieve(sieve_word_t sieve[]);
			void get_prime_candidate_count(uint64_t& prime_candidate_count);
			void get_stats(uint32_t chain_histogram[], uint64_t& chain_count);
			void synchronize();

		private:
			std::unique_ptr<Cuda_sieve_impl> m_impl;
			

		};
	}
}

#endif
