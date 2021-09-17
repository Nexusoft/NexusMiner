#ifndef NEXUSMINER_GPU_CUDA_SIEVE_IMPL_CUH
#define NEXUSMINER_GPU_CUDA_SIEVE_IMPL_CUH

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
				uint32_t prime_mod_inverses[], uint32_t sieve_size, uint16_t device);
			void free_sieve();

			void run_sieve(uint64_t sieve_start_offset, uint8_t sieve[]);
			void find_chains(CudaChain chains[], uint32_t& chain_count);

		private:
			//device memory pointers
			uint32_t* d_sieving_primes;
			uint32_t* d_starting_multiples;
			uint32_t* d_prime_mod_inverses;
			uint8_t* d_sieve;
			uint32_t* d_multiples;
			uint8_t* d_wheel_indices;
			//array of chains
			CudaChain* d_chains;
			uint32_t* d_chain_index;
			

		};
	}
}

#endif