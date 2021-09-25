#ifndef NEXUSMINER_GPU_CUDA_SIEVE_IMPL_CUH
#define NEXUSMINER_GPU_CUDA_SIEVE_IMPL_CUH

#include "sieve.hpp"
#include "cuda_chain.cuh"
#include <stdint.h>
namespace nexusminer {
	namespace gpu {

		class Cuda_sieve_impl
		{
		public:
			uint32_t m_sieving_prime_count;
			uint64_t m_sieve_start_offset;
			
			void load_sieve(uint32_t primes[], uint32_t prime_count, uint32_t starting_multiples[],
				uint8_t prime_mod_inverses[], uint32_t small_prime_offsets[], uint32_t sieve_size, uint16_t device);
			void free_sieve();

			void run_small_prime_sieve(uint64_t sieve_start_offset);
			void run_sieve(uint64_t sieve_start_offset);
			void get_sieve(Cuda_sieve::sieve_word_t sieve[]);
			void get_prime_candidate_count(uint64_t& prime_candidate_count);

			void find_chains(CudaChain chains[], uint32_t& chain_count);

		private:
			//device memory pointers
			uint32_t* d_sieving_primes;
			uint32_t* d_starting_multiples;
			uint8_t* d_prime_mod_inverses;
			uint32_t* d_small_prime_offsets;
			Cuda_sieve::sieve_word_t* d_sieve;
			uint32_t* d_multiples;
			uint8_t* d_wheel_indices;
			//array of chains
			CudaChain* d_chains;
			uint32_t* d_chain_index;
			unsigned long long* d_prime_candidate_count;
			

		};
	}
}

#endif