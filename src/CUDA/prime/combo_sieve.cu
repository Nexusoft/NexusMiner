/*******************************************************************************************

 Nexus Earth 2018

 (credits: cbuchner1 for sieving)

 [Scale Indefinitely] BlackJack. http://www.opensource.org/licenses/mit-license.php

*******************************************************************************************/

#include <CUDA/include/macro.h>
#include <CUDA/include/combo_sieve.h>
#include <CUDA/include/frame_resources.h>
#include <CUDA/include/sieve_resources.h>
#include <CUDA/include/prime_helper.cuh>
#include <CUDA/include/constants.h>
#include <Util/include/prime_config.h>

template<uint8_t o>
__global__ void combosieve_kernelA_512(uint32_t *g_sieve_hierarchy,
                                       uint32_t *g_bit_array_sieve,
                                       uint32_t *prime_remainders,
                                       uint16_t *blockoffset_mod_p,
                                       uint32_t base_index,
                                       uint16_t nPrimorialEndPrime,
                                       uint16_t nPrimeLimitA)
{
    extern __shared__ uint32_t shared_array_sieve[];
    uint32_t nAdd;
    uint32_t index;
    uint32_t mask;
    uint32_t id;
    uint16_t i, j;
    uint16_t pr, pre2;

    #pragma unroll 16
    for (uint8_t i= 0; i <  16; ++i)
        shared_array_sieve[threadIdx.x + (i << 9)] = 0;

    __syncthreads();

    base_index = base_index << 9;

    for (i = nPrimorialEndPrime; i < nPrimeLimitA; ++i)
    {
        pr = c_primes[i];
        pre2 = blockoffset_mod_p[(blockIdx.x << 12) + i];

        // precompute
        nAdd = pr << 9;

        index = prime_remainders[((base_index + i) << 3) + o] + pre2;  // << 3 because we have space for 8 offsets
        if(index >= pr)
            index = index - pr;

        index = threadIdx.x * pr + index;

        for(; index < 262144; index += nAdd)
        {
            mask = 1 << (index & 31);
            id = index >> 5;

            if((g_sieve_hierarchy[id] & mask) == 0)
                atomicOr(&shared_array_sieve[id], mask);
        }

    }

    __syncthreads();
    g_bit_array_sieve += (blockIdx.x << 13);

    #pragma unroll 16
    for (uint8_t i = 0; i < 16; ++i) // fixed value
    {
        j = threadIdx.x + (i << 9);
        //atomicOr(&g_bit_array_sieve[j], shared_array_sieve[j]);
        g_bit_array_sieve[j] = shared_array_sieve[j];
    }
}

template<uint8_t o>
__global__ void combosieve_kernelA_256(uint32_t *g_sieve_hierarchy,
                                       uint32_t *g_bit_array_sieve,
                                       uint32_t *prime_remainders,
                                       uint16_t *blockoffset_mod_p,
                                       uint32_t base_index,
                                       uint16_t nPrimorialEndPrime,
                                       uint16_t nPrimeLimitA)
{
    extern __shared__ uint32_t shared_array_sieve[];

    #pragma unroll 32
    for (uint8_t i= 0; i <  32; ++i)
        shared_array_sieve[threadIdx.x + (i << 8)] = 0;

    __syncthreads();

    for (uint16_t i = nPrimorialEndPrime; i < nPrimeLimitA; ++i)
    {
        uint16_t pr = c_primes[i];
        uint16_t pre2 = blockoffset_mod_p[(blockIdx.x << 12) + i];

        // precompute
        uint32_t pIdx = threadIdx.x * pr;
        uint32_t nAdd = pr << 8;

        uint32_t index = prime_remainders[(((base_index << 9) + i) << 3) + o] + pre2;  // << 3 because we have space for 8 offsets
        if(index >= pr)
            index = index - pr;

        for(index = index + pIdx; index < 262144; index += nAdd)
        {
            if( (g_sieve_hierarchy[index >> 5] & (1 << (index & 31)) ) == 0)
                atomicOr(&shared_array_sieve[index >> 5], 1 << (index & 31));
        }

    }

    __syncthreads();
    g_bit_array_sieve += (blockIdx.x << 13);

    #pragma unroll 32
    for (uint8_t i = 0; i < 32; ++i) // fixed value
    {
        uint16_t j = threadIdx.x + (i << 8);
        //atomicOr(&g_bit_array_sieve[j], shared_array_sieve[j]);
        g_bit_array_sieve[j] = shared_array_sieve[j];
    }
}

template<uint8_t o>
__global__ void combosieve_kernelA_128(uint32_t *g_sieve_hierarchy,
                                       uint32_t *g_bit_array_sieve,
                                       uint32_t *prime_remainders,
                                       uint16_t *blockoffset_mod_p,
                                       uint32_t base_index,
                                       uint16_t nPrimorialEndPrime,
                                       uint16_t nPrimeLimitA)
{
    extern __shared__ uint32_t shared_array_sieve[];

    #pragma unroll 64
    for (uint8_t i= 0; i <  64; ++i)
        shared_array_sieve[threadIdx.x + (i << 7)] = 0;

    __syncthreads();

    for (uint16_t i = nPrimorialEndPrime; i < nPrimeLimitA; ++i)
    {
        uint16_t pr = c_primes[i];
        uint16_t pre2 = blockoffset_mod_p[(blockIdx.x << 12) + i];

        // precompute
        uint32_t pIdx = threadIdx.x * pr;
        uint32_t nAdd = pr << 7;

        uint32_t index = prime_remainders[(((base_index << 9) + i) << 3) + o] + pre2; // << 3 because we have space for 8 offsets

        if(index >= pr)
            index = index - pr;

        for(index = index + pIdx; index < 262144; index += nAdd)
        {
            if( (g_sieve_hierarchy[index >> 5] & (1 << (index & 31)) ) == 0)
                atomicOr(&shared_array_sieve[index >> 5], 1 << (index & 31));
        }
    }

    __syncthreads();
    g_bit_array_sieve += (blockIdx.x << 13);

    #pragma unroll 64
    for (uint8_t i = 0; i < 64; ++i) // fixed value
    {
        uint16_t j = threadIdx.x + (i << 7);
        //atomicOr(&g_bit_array_sieve[j], shared_array_sieve[j]);
        g_bit_array_sieve[j] = shared_array_sieve[j];
    }
}


template<uint8_t o>
__global__ void combosieve_kernelB(uint64_t *origins,
                                   uint32_t origin_index,
                                   uint32_t *g_sieve_hierarchy,
                                   uint32_t *bit_array_sieve,
                                   uint32_t bit_array_size,
                                   uint4 *primes,
                                   uint32_t *base_remainders,
                                   uint16_t nPrimorialEndPrime,
                                   uint32_t nPrimeLimit)
{
    uint32_t i = nPrimorialEndPrime + blockDim.x * blockIdx.x + threadIdx.x;

    if(i < nPrimeLimit)
    {
        uint4 tmp = primes[i];
        uint64_t recip = make_uint64_t(tmp.z, tmp.w);
        uint32_t index;
        uint32_t mask;

        tmp.z = mod_p_small(origins[origin_index] + base_remainders[i] + c_offsets[c_iB[o]], tmp.x, recip);
        tmp.w = mod_p_small((uint64_t)(tmp.x - tmp.z)*tmp.y, tmp.x, recip);

        for(; tmp.w < bit_array_size; tmp.w += tmp.x)
        {
            index = tmp.w >> 5;
            mask = c_mark_mask[tmp.w & 31];

            if((g_sieve_hierarchy[index] & mask) == 0)
                atomicOr(&bit_array_sieve[index], mask);
        }
    }
}

__global__ void compact_combo(uint64_t *d_origins,
                              uint64_t *d_nonce_offsets,
                              uint32_t *d_nonce_meta,
                              uint32_t *d_nonce_count,
                              uint32_t *d_bit_array_sieve_A,
                              uint32_t *d_bit_array_sieve_B,
                              uint32_t nBitArray_Size,
                              uint32_t origin_index,
                              uint8_t nThreshold,
                              uint8_t nOffsetsB,
                              uint8_t nOffsets)
{
    /* If the quit flag was set, early return to avoid wasting time. */
    //if(c_quit)
    //    return;


    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    //uint32_t sharedSizeBits = 32 * 1024 * 8;
    //uint32_t allocSize = ((nBitArray_Size + sharedSizeBits-1) / sharedSizeBits) * sharedSizeBits;
    //uint32_t nSieveWords = (allocSize + 31) >> 5;

    if(idx < nBitArray_Size)
    {
        uint64_t nonce_offset = c_primorial * (uint64_t)idx + d_origins[origin_index];
        uint32_t combo = 0;

        uint32_t index = idx >> 5;
        uint32_t mask = c_mark_mask[idx & 31];
        uint32_t nBitArray_Words = nBitArray_Size >> 5;

        /* Check the single sieve to see if offsets are valid. */
        if((d_bit_array_sieve_A[index] & mask) == 0)
        {
            combo = c_bitmaskA;

            /* Take the bit from each sieve and compact them into a single word. */
            for(uint8_t o = 0; o < nOffsetsB; ++o)
            {
                /* Use logical not operator to reduce result into inverted 0 or 1 bit. */
                uint32_t bit = !(d_bit_array_sieve_B[index] & mask);

                combo |= bit << c_iB[o];

                d_bit_array_sieve_B += nBitArray_Words;
            }

            /* Get the count of the remaining zero bits and compare to threshold. */
            uint32_t nRemaining = __popc(combo);

            /* Count the remaining bits for this combo. */
            if(nRemaining >= nThreshold)
            {
                //printf("%d: compact_sieve: combo=%08X, count=%d\n", idx, combo, nRemaining);

                /* Compute the end, tail, and next indices. */
                uint32_t e = 32 - __clz(combo);
                uint32_t t = 0;
                uint32_t n = __ffs(combo);

                /* Iterate through sieved bits and determine if there is
                 * a rule-breaking prime gap. */
                uint32_t nCount = 1;
                for(; nCount < nRemaining; ++nCount)
                {
                    t = n;
                    n = n + __ffs(combo >> (n+1) );

                    if( (c_offsets[n] - c_offsets[t]) > 12)
                        break;
                }

                /* Determine if combo will break prime gap or not. */
                if(nCount >= nThreshold)
                {
                    uint32_t i = atomicAdd(d_nonce_count, 1);

                    if(i < CANDIDATES_MAX)
                    {
                        /* Assign the global nonce offset and meta data. */
                        d_nonce_offsets[i] = nonce_offset;
                        d_nonce_meta[i] = ~combo;
                    }
                }
                //else
                //    printf("%d: combo=%08X, n=%d, t=%d, count=%d\n", idx, combo, n, t, nCount);
            }
        }
    }
}

#define COMBO_A_LAUNCH(X) combosieve_kernelA_512<X><<<grid, block, sharedSizeBits/8, d_Streams[thr_id][str_id]>>>(\
frameResources[thr_id].d_bit_array_sieve[frame_index], \
&frameResources[thr_id].d_bit_array_sieve[frame_index][X * nBitArray_Size >> 5], \
&d_prime_remainders[thr_id][nPrimeLimitA * nOrigins << 3], \
d_blockoffset_mod_p[thr_id], \
origin_index, \
nPrimorialEndPrime, \
nPrimeLimitA)

void comboA_launch(uint8_t thr_id,
                   uint8_t str_id,
                   uint32_t origin_index,
                   uint8_t frame_index,
                   uint16_t nPrimorialEndPrime,
                   uint16_t nPrimeLimitA,
                   uint32_t nBitArray_Size,
                   uint32_t nOrigins)
{
    uint32_t sharedSizeBits = 32 * 1024 * 8;
    uint32_t nBlocks = (nBitArray_Size + sharedSizeBits-1) / sharedSizeBits;

    //uint32_t allocSize = ((nBitArray_Size + sharedSizeBits-1) / sharedSizeBits) * sharedSizeBits;
    //uint32_t nSieveWords = (allocSize + 31) >> 5;

    dim3 block(512);
    dim3 grid(nBlocks);

    /* fall-through switch logic, zero-based indexing */
    switch(nOffsetsB)
    {
        case 8:  COMBO_A_LAUNCH(8);
        case 7:  COMBO_A_LAUNCH(7);
        case 6:  COMBO_A_LAUNCH(6);
        case 5:  COMBO_A_LAUNCH(5);
        case 4:  COMBO_A_LAUNCH(4);
        case 3:  COMBO_A_LAUNCH(3);
        case 2:  COMBO_A_LAUNCH(2);
        case 1:  COMBO_A_LAUNCH(1);
        break;
        default: debug::error("Unsupported Combo A Launch.");
        break;
    }

    debug::log(4, FUNCTION, (uint32_t)thr_id);
}

#define COMBO_B_LAUNCH(X)   combosieve_kernelB<X><<<grid, block, 0, d_Streams[thr_id][str_id]>>>( \
d_origins[thr_id], \
origin_index, \
frameResources[thr_id].d_bit_array_sieve[frame_index], \
&frameResources[thr_id].d_bit_array_sieve[frame_index][X * nBitArray_Size >> 5], \
nBitArray_Size, \
d_primesInverseInvk[thr_id], \
d_base_remainders[thr_id], \
nPrimorialEndPrime, \
nPrimeLimit)

void comboB_launch(uint8_t thr_id,
                    uint8_t str_id,
                    uint32_t origin_index,
                    uint8_t frame_index,
                    uint16_t nPrimorialEndPrime,
                    uint32_t nPrimeLimit,
                    uint32_t nBitArray_Size)
{
    uint32_t nThreads = nPrimeLimit - nPrimorialEndPrime;
    uint32_t nThreadsPerBlock = 512;
    uint32_t nBlocks = (nThreads + nThreadsPerBlock - 1) / nThreadsPerBlock;

    dim3 block(nThreadsPerBlock);
    dim3 grid(nBlocks);

    /* fall-through switch logic, zero-based indexing */
    switch(nOffsetsB)
    {
        case 8:  COMBO_B_LAUNCH(8);
        case 7:  COMBO_B_LAUNCH(7);
        case 6:  COMBO_B_LAUNCH(6);
        case 5:  COMBO_B_LAUNCH(5);
        case 4:  COMBO_B_LAUNCH(4);
        case 3:  COMBO_B_LAUNCH(3);
        case 2:  COMBO_B_LAUNCH(2);
        case 1:  COMBO_B_LAUNCH(1);
        break;
        default: debug::error("Unsupported Combo B Launch.");
        break;
    }

    debug::log(4, FUNCTION, (uint32_t)thr_id);
}

#define COMBO_COMPACT_LAUNCH(X) compact_combo<<<grid, block, 0, d_Streams[thr_id][str_id]>>>( \
    d_origins[thr_id], \
    frameResources[thr_id].d_nonce_offsets[curr_test], \
    frameResources[thr_id].d_nonce_meta[curr_test], \
    frameResources[thr_id].d_nonce_count[curr_test], \
    frameResources[thr_id].d_bit_array_sieve[curr_sieve], \
    &frameResources[thr_id].d_bit_array_sieve[curr_sieve][nBitArray_Size >> 5], \
    nBitArray_Size, \
    origin_index, \
    threshold, \
    X, \
    vOffsets.size())


void kernel_ccompact_launch(uint8_t thr_id,
                            uint8_t str_id,
                            uint32_t origin_index,
                            uint8_t curr_sieve,
                            uint8_t curr_test,
                            uint8_t next_test,
                            uint32_t nBitArray_Size,
                            uint8_t threshold)
{
    dim3 block(64);
    dim3 grid((nBitArray_Size + block.x - 1) / block.x);

    switch(nOffsetsB)
    {
        case 1:  COMBO_COMPACT_LAUNCH(1);  break;
        case 2:  COMBO_COMPACT_LAUNCH(2);  break;
        case 3:  COMBO_COMPACT_LAUNCH(3);  break;
        case 4:  COMBO_COMPACT_LAUNCH(4);  break;
        case 5:  COMBO_COMPACT_LAUNCH(5);  break;
        case 6:  COMBO_COMPACT_LAUNCH(6);  break;
        case 7:  COMBO_COMPACT_LAUNCH(7);  break;
        case 8:  COMBO_COMPACT_LAUNCH(8);  break;
        default: debug::error("Unsupported Combo Compact Launch."); break;
    }

    /* Copy the nonce count for this compaction. */
    CHECK(cudaMemcpyAsync(
            frameResources[thr_id].h_nonce_count[curr_test],
            frameResources[thr_id].d_nonce_count[curr_test],
            sizeof(uint32_t), cudaMemcpyDeviceToHost, d_Streams[thr_id][str_id]));

    /*Prepare empty initial count for next compaction buffer. */
    *frameResources[thr_id].h_nonce_count[next_test] = 0;

    CHECK(cudaMemcpyAsync(
            frameResources[thr_id].d_nonce_count[next_test],
            frameResources[thr_id].h_nonce_count[next_test],
            sizeof(uint32_t), cudaMemcpyHostToDevice, d_Streams[thr_id][str_id]));

    debug::log(4, FUNCTION, (uint32_t)thr_id);
}
