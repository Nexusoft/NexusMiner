#ifndef NEXUSMINER_GPU_CUDA_SIEVE_HPP
#define NEXUSMINER_GPU_CUDA_SIEVE_HPP

//hardware dependent performance tweaks
#if defined(GPU_CUDA_ENABLED)
#define GPU_LARGE_PRIME_COUNT 50000
#else
//AMD gpu has slower fermat testing which can be offset somewhat with more sieving
#define GPU_LARGE_PRIME_COUNT 150000
#endif

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
			static constexpr int m_sieve_word_byte_count = sizeof(sieve_word_t);
			static constexpr int m_sieve_byte_range = 30;
			static constexpr int m_sieve_word_range = m_sieve_byte_range * m_sieve_word_byte_count;
			//The wheel formed by primorial 2*3*5*7*11 = 2310 has two gaps of 14.  If we align the sieve start/stop to one of these gaps, 
			// we guarantee that no chains can cross through the segment boundary.
			static constexpr int m_sieve_chain_search_boundary = 2310;
			static constexpr int m_sieve_alignment = m_sieve_chain_search_boundary * m_sieve_word_byte_count;  //ensure the segment ends on a word boundary
			static constexpr int m_sieve_alignment_offset = 120;  //offset from the wheel start to the first gap greater than 12.  Coincidentally its span 120 is a whole word.
			
			struct Cuda_sieve_properties
			{
				int m_shared_mem_size_kbytes;
				int m_shared_mem_size_bytes;
				unsigned int m_kernel_sieve_size_bytes;
				unsigned int m_kernel_sieve_size_words;
				unsigned int m_segment_range;
				unsigned int m_kernel_sieve_size_words_per_block;
				uint64_t m_block_range;
				uint64_t m_sieve_total_size; //size of the sieve in words
				uint64_t m_sieve_range;
				uint64_t m_bucket_ram_budget;
				int m_large_prime_bucket_size;
			};
			Cuda_sieve_properties m_sieve_properties;
			
			static constexpr int m_kernel_segments_per_block = 32;  //number of times to repeat the sieve within a kernel call
			static constexpr int m_num_blocks = 360;  //each block sieves part of the range
			
			static constexpr int m_threads_per_block = 1024;
			
			//the largest possible sieve based on max shared memory of 164K on A100 GPU
			static constexpr uint64_t m_sieve_max_range = 164ull * 1024 * m_sieve_byte_range * m_kernel_segments_per_block * m_num_blocks;

			static constexpr int m_estimated_chains_per_million = 4;
			static constexpr uint32_t m_max_chains = 2 * Cuda_sieve::m_estimated_chains_per_million * Cuda_sieve::m_sieve_max_range / 1e6;
			static constexpr uint32_t m_max_long_chains = 32;
			static constexpr int m_min_chain_length = 9;
			static constexpr int m_start_prime = 7;
			static constexpr int m_small_prime_count = 14; //61 is the 15th prime starting at 7.  61 is first prime that hits each sieve word no more than 1 time.
			//If you change the small_prime_count, make sure you also change the hardcoded list of primes in the small prime sieve in sieve_impl.cu
//primes 7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103
//       1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22, 23,24
			static constexpr int m_medium_small_prime_count = 32 * 18;
			static constexpr int m_medium_prime_count = 32 * 2500;			
			static constexpr int m_large_prime_count = 32 * GPU_LARGE_PRIME_COUNT;
			static constexpr int m_trial_division_prime_count = 0;

			static constexpr int chain_histogram_max = 10;  
			//static const uint64_t m_bucket_ram_budget = 4.5e9;  //bytes avaialble for storing bucket data
			//static const int m_large_prime_bucket_size = m_large_prime_count == 0 ? 1 : m_bucket_ram_budget/(m_num_blocks * m_kernel_segments_per_block)/4;
			

			Cuda_sieve();
			~Cuda_sieve();
			void load_sieve(uint32_t primes[], uint32_t prime_count, uint32_t large_primes[], uint32_t medium_small_primes[],
				uint32_t small_prime_masks[], uint32_t small_prime_mask_count, uint8_t small_primes[], uint16_t device);
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
