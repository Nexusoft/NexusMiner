/*******************************************************************************************

 Nexus Earth 2018

 (credits: cbuchner1 for sieving)

 [Scale Indefinitely] BlackJack. http://www.opensource.org/licenses/mit-license.php

*******************************************************************************************/

#include <CUDA/include/macro.h>
#include <CUDA/include/util.h>
#include <CUDA/include/frame_resources.h>
#include <CUDA/include/sieve_resources.h>
#include <CUDA/include/streams_events.h>
#include <CUDA/include/sieve.h>

#include <cstdint>

__global__ void bucketinit_kernel(uint64_t origin,
                                  uint32_t *bucket_o,
                                  uint16_t *bucket_away,
                                  uint4 *primes,
                                  uint32_t *base_remainders,
                                  uint32_t n_primorial_end,
                                  uint32_t n_prime_limit,
                                  uint8_t n_offsets,
                                  uint8_t bit_array_size_log2)
{
    uint32_t idx = n_primorial_end + blockDim.x * blockIdx.x + threadIdx.x;

    if(idx < n_prime_limit)
    {
        uint4 tmp = primes[idx];
        uint64_t recip = make_uint64_t(tmp.z, tmp.w);
        uint32_t bit_array_mask = (1 << bit_array_size_log2) - 1;

        //compute offsets
        for(uint32_t j = 0; j < n_offsets; ++j)
        {
            tmp.z = mod_p_small(origin + base_remainders[idx] + c_offsets[c_iA[j]], tmp.x, recip);
            tmp.w = mod_p_small((uint64_t)(tmp.x - tmp.z)*tmp.y, tmp.x, recip);

            bucket_o[(idx << 4) + j] =    tmp.w & bit_array_mask;
            bucket_away[(idx << 4) + j] = tmp.w >> bit_array_size_log2;
        }
    }
}

__global__ void bucketsieve_kernel(uint32_t *bit_array_sieve,
                                   uint32_t *primes,
                                   uint32_t *bucket_o,
                                   uint16_t *bucket_away,
                                   uint32_t n_prime_limit,
                                   uint16_t n_primorial_end,
                                   uint8_t n_offsets,
                                   uint8_t bit_array_size_log2)
{
    uint32_t position = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t i = n_primorial_end + (position >> 4);
    uint8_t j = position & 15;

    if(i < n_prime_limit && j < n_offsets)
    {
        uint32_t idx = (i << 4) + j;


        if(bucket_away[idx] == 0)
        {
            uint32_t p = primes[i];
            uint32_t o = bucket_o[idx];
            uint32_t bit_array_size = 1 << bit_array_size_log2;

            for(; o < bit_array_size; o += p)
                atomicOr(&bit_array_sieve[o >> 5], 1 << (o & 31));

            bucket_o[idx] = o & (bit_array_size - 1);          //computes next offset
            bucket_away[idx] = (o >> bit_array_size_log2) - 1; //computes sieves away
        }
        else
        --bucket_away[idx];
    }
}

void kernelC_launch(uint8_t thr_id,
                    uint8_t str_id,
                    uint8_t frame_index,
                    uint16_t nPrimorialEndPrime,
                    uint32_t nPrimeLimit,
                    uint32_t nBitArray_SizeLog2)
{
    uint32_t nThreads = nPrimeLimit - nPrimorialEndPrime;
    dim3 block(256);
    dim3 grid((nThreads + block.x - 1) / (block.x >> 4));

    bucketsieve_kernel<<<grid, block, 0, d_Streams[thr_id][str_id]>>>(
        frameResources[thr_id].d_bit_array_sieve[frame_index],
        d_primes[thr_id],
        frameResources[thr_id].d_bucket_o[frame_index],
        frameResources[thr_id].d_bucket_away[frame_index],
        nPrimeLimit,
        nPrimorialEndPrime,
        nOffsetsB,
        nBitArray_SizeLog2);
}

extern "C" void cuda_set_sieve(uint8_t thr_id,
                               uint64_t base_offset,
                               uint64_t primorial,
                               uint32_t n_primorial_end,
                               uint32_t n_prime_limit,
                               uint8_t bit_array_size_log2)
{
    dim3 block(256);
    dim3 grid(((n_prime_limit - n_primorial_end) + block.x - 1) / block.x);
    uint32_t curr_sieve = 0;
    for(; curr_sieve < FRAME_COUNT; ++curr_sieve)
    {
        bucketinit_kernel<<<grid, block, 0>>>(
            base_offset,
            frameResources[thr_id].d_bucket_o[curr_sieve],
            frameResources[thr_id].d_bucket_away[curr_sieve],
            d_primesInverseInvk[thr_id],
            d_base_remainders[thr_id],
            n_primorial_end,
            n_prime_limit,
            nOffsetsB,
            bit_array_size_log2);
    }
    CHECK(cudaDeviceSynchronize());
}
