#ifndef SIEVE_CUH
#define SIEVE_CUH

#include <stdint.h>

void run_sieve(uint32_t sieving_primes[], uint32_t sieving_prime_count, 
			   uint32_t starting_multiples[], int wheel_indices[],
			   uint8_t sieve[], uint32_t& sieve_size);



#endif