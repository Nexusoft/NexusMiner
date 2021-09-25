#ifndef NEXUSMINER_GPU_CUDA_SIEVE_HPP
#define NEXUSMINER_GPU_CUDA_SIEVE_HPP

#include "cuda_chain.cuh"
//#include "fermat_test.hpp"
#include <stdint.h>
#include <memory>

namespace nexusminer {
	namespace gpu {

		class Cuda_sieve_impl;
		class Cuda_sieve
		{
		public:
			using sieve_word_t = uint32_t;
			static const int m_sieve_word_byte_count = sizeof(sieve_word_t);
			static const int m_kernel_sieve_size_bytes = 4096 * 8;  //this is the size of the sieve in bytes.  it should be a multiple of 8. 
			static const int m_kernel_sieve_size_words = m_kernel_sieve_size_bytes / m_sieve_word_byte_count;
			static const int m_kernel_segments_per_block = 20;  //number of times to run the sieve within a kernel call
			static const int m_kernel_sieve_size_words_per_block = m_kernel_sieve_size_words * m_kernel_segments_per_block;
			static const int m_threads_per_block = 512;
			static const int m_num_blocks = 800;  //each block sieves part of the range
			static const uint64_t m_sieve_total_size = m_kernel_sieve_size_words_per_block * m_num_blocks; //size of the sieve in words
			static const int m_sieve_byte_range = 30;
			static const int m_sieve_word_range = m_sieve_byte_range * m_sieve_word_byte_count;
			static const uint64_t m_sieve_range = m_sieve_total_size * m_sieve_word_range;
			static const int m_estimated_chains_per_million = 12;
			static const uint32_t m_max_chains = 5*m_estimated_chains_per_million*m_sieve_range/1e6;
			static const int m_min_chain_length = 8;
			static const int m_small_prime_start = 7;
			static const int m_small_prime_end = 7;
			static const int m_small_prime_count = 1;
			static const int m_small_primes[]; //array is defined in sieve.cu
			//primes for reference 7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97
			//                     1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22   
			

			Cuda_sieve();
			~Cuda_sieve();
			void load_sieve(uint32_t primes[], uint32_t prime_count, uint32_t starting_multiples[],
				uint8_t prime_mod_inverses[], uint32_t small_prime_offsets[], uint32_t sieve_size, uint16_t device);
			void free_sieve();
			void run_small_prime_sieve(uint64_t sieve_start_offset);
			void run_sieve(uint64_t sieve_start_offset);
			void find_chains(CudaChain chains[], uint32_t& chain_count);
			void get_sieve(sieve_word_t sieve[]);
			void get_prime_candidate_count(uint64_t& prime_candidate_count);

		private:
			std::unique_ptr<Cuda_sieve_impl> m_impl;
			

		};
	}
}

#endif