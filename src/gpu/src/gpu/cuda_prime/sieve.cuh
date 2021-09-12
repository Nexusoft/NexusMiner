#ifndef CUDA_SIEVE_CUH
#define CUDA_SIEVE_CUH

#include <stdint.h>
namespace nexusminer {
	namespace gpu {

		class CudaSieve
		{
		public:
			static const int m_kernel_sieve_size = 4096 * 8;  //this is the size of the sieve in bytes.  it should be a multiple of 8. 
			static const int m_kernel_segments_per_block = 20;  //number of times to run the sieve within a kernel call
			static const int m_kernel_sieve_size_per_block = m_kernel_sieve_size * m_kernel_segments_per_block;
			static const int m_threads_per_block = 1024;
			static const int m_num_blocks = 200;  //each block sieves part of the range
			static const int m_sieve_total_size = m_kernel_sieve_size_per_block * m_num_blocks; //size of the sieve in bytes
			static const int m_sieve_range = m_sieve_total_size / 8 * 30;
			uint32_t m_sieving_prime_count;
			
			void load_sieve(uint32_t primes[], uint32_t prime_count, uint32_t starting_multiples[],
				uint32_t prime_mod_inverses[], uint32_t sieve_size, uint16_t device);
			void free_sieve();

			void run_sieve(uint64_t sieve_start_offset, uint8_t sieve[]);

		private:
			//device memory pointers
			uint32_t* d_sieving_primes;
			uint32_t* d_starting_multiples;
			uint32_t* d_prime_mod_inverses;
			uint8_t* d_sieve;
			uint32_t* d_multiples;
			uint8_t* d_wheel_indices;

		};
	}
}

#endif