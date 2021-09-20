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
			static const int m_kernel_sieve_size = 4096 * 8;  //this is the size of the sieve in bytes.  it should be a multiple of 8. 
			static const int m_kernel_segments_per_block = 20;  //number of times to run the sieve within a kernel call
			static const int m_kernel_sieve_size_per_block = m_kernel_sieve_size * m_kernel_segments_per_block;
			static const int m_threads_per_block = 1024;
			static const int m_num_blocks = 400;  //each block sieves part of the range
			static const uint64_t m_sieve_total_size = m_kernel_sieve_size_per_block * m_num_blocks; //size of the sieve in bytes
			static const uint64_t m_sieve_range = m_sieve_total_size / 8 * 30;
			static const int m_estimated_chains_per_million = 12;
			static const uint32_t m_max_chains = 10*m_estimated_chains_per_million*m_sieve_range/1e6;
			static const int m_min_chain_length = 8;

			Cuda_sieve();
			~Cuda_sieve();
			void load_sieve(uint32_t primes[], uint32_t prime_count, uint32_t starting_multiples[],
				uint32_t prime_mod_inverses[], uint32_t sieve_size, uint16_t device);
			void free_sieve();

			void run_sieve(uint64_t sieve_start_offset, uint8_t sieve[]);
			void find_chains(CudaChain chains[], uint32_t& chain_count);

		private:
			std::unique_ptr<Cuda_sieve_impl> m_impl;
			

		};
	}
}

#endif