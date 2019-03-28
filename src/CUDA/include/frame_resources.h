/*******************************************************************************************

 Nexus Earth 2018

 [Scale Indefinitely] BlackJack. http://www.opensource.org/licenses/mit-license.php

*******************************************************************************************/
#ifndef NEXUS_CUDA_FRAME_RESOURCES_H
#define NEXUS_CUDA_FRAME_RESOURCES_H

#include <CUDA/include/macro.h>
#include <cstdint>

#define FRAME_COUNT 3
#define BUFFER_COUNT 2

struct FrameResource
{
    //testing
  uint64_t *d_result_offsets[FRAME_COUNT];
  uint64_t *h_result_offsets[FRAME_COUNT];

  uint64_t *d_result_meta[FRAME_COUNT];
  uint64_t *h_result_meta[FRAME_COUNT];

  uint32_t *d_result_count[FRAME_COUNT];
  uint32_t *h_result_count[FRAME_COUNT];

  uint32_t *d_primes_checked[FRAME_COUNT];
  uint32_t *h_primes_checked[FRAME_COUNT];

  uint32_t *d_primes_found[FRAME_COUNT];
  uint32_t *h_primes_found[FRAME_COUNT];

    //compacting
  uint64_t *d_nonce_offsets[FRAME_COUNT];
  uint64_t *d_nonce_meta[FRAME_COUNT];
  uint32_t *d_nonce_count[FRAME_COUNT];

  uint32_t *h_nonce_count[FRAME_COUNT];

    //sieving
  uint32_t  *d_bit_array_sieve[FRAME_COUNT];
  uint32_t  *d_prime_remainders[FRAME_COUNT]; //need to do away with this

    //bucket sieve
  uint32_t  *d_bucket_o[FRAME_COUNT]; //prime and offset
  uint16_t  *d_bucket_away[FRAME_COUNT]; //sieves away from next offset

};

/* Global instance. */
extern struct FrameResource frameResources[GPU_MAX];

#endif
