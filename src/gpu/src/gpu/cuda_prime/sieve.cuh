#ifndef SIEVE_CUH
#define SIEVE_CUH

#include <stdint.h>
namespace nexusminer {
	namespace gpu {

		const int kernel_sieve_size = 4096*8;  //this is the size of the sieve in bytes.  it should be a multiple of 8. 
		const int kernel_segments_per_block = 1;  //number of times to run the sieve within a kernel call
		const int kernel_sieve_size_per_block = kernel_sieve_size * kernel_segments_per_block;
	
		void load_sieve(uint32_t sieving_primes[], uint32_t sieving_prime_count);
		
		void run_sieve(uint32_t sieving_primes[], uint32_t sieving_prime_count,
			uint32_t starting_multiples[], int wheel_indices[],
			uint8_t sieve[], uint32_t& sieve_size, uint64_t sieve_start_offset);

	}
}

#endif