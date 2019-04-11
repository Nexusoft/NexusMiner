/*******************************************************************************************

 Nexus Earth 2018

 [Scale Indefinitely] BlackJack. http://www.opensource.org/licenses/mit-license.php

*******************************************************************************************/
#ifndef NEXUS_CUDA_SIEVE_RESOURCES_H
#define NEXUS_CUDA_SIEVE_RESOURCES_H

#include <CUDA/include/macro.h>

extern uint4 *d_primesInverseInvk[GPU_MAX];
extern uint64_t *d_origins[GPU_MAX];
extern uint32_t *d_primes[GPU_MAX];
extern uint32_t *d_prime_remainders[GPU_MAX];
extern uint32_t *d_base_remainders[GPU_MAX];
extern uint16_t *d_blockoffset_mod_p[GPU_MAX];
extern uint32_t nOffsetsA;
extern uint32_t nOffsetsB;

#endif
