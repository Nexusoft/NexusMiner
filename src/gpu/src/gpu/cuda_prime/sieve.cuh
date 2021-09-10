#ifndef SIEVE_CUH
#define SIEVE_CUH

#include <stdint.h>
namespace nexusminer {
	namespace gpu {

		const int kernel_sieve_size = 4096*8;  //this is the size of the sieve in bytes.  it should be a multiple of 8. 
		const int kernel_segments_per_block = 20;  //number of times to run the sieve within a kernel call
		const int kernel_sieve_size_per_block = kernel_sieve_size * kernel_segments_per_block;
		const int threads_per_block = 1024;
		const int num_blocks = 200;  //each block sieves part of the range
		const int sieve_total_size = kernel_sieve_size_per_block * num_blocks; //size of the sieve in bytes
		const int sieve_range = sieve_total_size / 8 * 30;
		

		void load_sieve(uint32_t primes[], uint32_t prime_count, uint32_t starting_multiples[],
			uint32_t prime_mod_inverses[], uint32_t sieve_size);
		void free_sieve();
		
		void run_sieve(uint64_t sieve_start_offset, uint8_t sieve[]);

	}
}

#endif