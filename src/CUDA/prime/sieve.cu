/*******************************************************************************************

 Nexus Earth 2018

 (credits: cbuchner1 for sieving)

 [Scale Indefinitely] BlackJack. http://www.opensource.org/licenses/mit-license.php

*******************************************************************************************/

#include <CUDA/include/macro.h>
#include <CUDA/include/sieve.h>
#include <CUDA/include/util.h>
#include <CUDA/include/frame_resources.h>
#include <CUDA/include/sieve_resources.h>
#include <CUDA/include/streams_events.h>

#include <CUDA/include/combo_sieve.h>

#include <CUDA/include/constants.h>
#include <CUDA/include/unroller.cuh>

#include <Util/include/debug.h>

#include <cuda.h>
#include <stdio.h>



struct FrameResource frameResources[GPU_MAX];
uint4 *d_primesInverseInvk[GPU_MAX];
uint64_t *d_origins[GPU_MAX];
uint32_t *d_primes[GPU_MAX];
uint32_t *d_base_remainders[GPU_MAX];
uint16_t *d_blockoffset_mod_p[GPU_MAX];
uint32_t nOffsetsA;
uint32_t nOffsetsB;


extern "C" void cuda_set_offset_patterns(uint8_t thr_id,
                                         const std::vector<uint32_t> &offsets,
                                         const std::vector<uint32_t> &indicesA,
                                         const std::vector<uint32_t> &indicesB,
                                         const std::vector<uint32_t> &indicesT)
{
    nOffsetsA = indicesA.size();
    nOffsetsB = indicesB.size();
    uint32_t nOffsetsT = indicesT.size();
    uint32_t nOffsets = offsets.size();
    uint32_t bitMaskA = 0;

    if(nOffsets > 16)
    {
        debug::error(FUNCTION, "Cannot have more than 16 total offsets.");
        return;
    }

    if(nOffsetsA > 8 || nOffsetsB > 8 || nOffsetsT > 8)
    {
        debug::error(FUNCTION, "Cannot have more than 8 offsets");
        return;
    }

    for(uint8_t i = 0; i < nOffsetsA; ++i)
        bitMaskA |= (1 << indicesA[i]);

    /* Invert the bits and mask off unwanted high bits. */
    //bitMaskA = (~bitMaskA) & (0xFFFFFFFF >> (32 - nOffsets));

    CHECK(cudaMemcpyToSymbol(c_bitmaskA, &bitMaskA,
         sizeof(uint32_t), 0, cudaMemcpyHostToDevice));

    CHECK(cudaMemcpyToSymbol(c_offsets, offsets.data(),
        nOffsets*sizeof(uint32_t), 0, cudaMemcpyHostToDevice));

    CHECK(cudaMemcpyToSymbol(c_iA, indicesA.data(),
        nOffsetsA*sizeof(uint32_t), 0, cudaMemcpyHostToDevice));

    CHECK(cudaMemcpyToSymbol(c_iB, indicesB.data(),
        nOffsetsB*sizeof(uint32_t), 0, cudaMemcpyHostToDevice));

    CHECK(cudaMemcpyToSymbol(c_iT, indicesT.data(),
        nOffsetsT*sizeof(uint32_t), 0, cudaMemcpyHostToDevice));

}


extern "C" void cuda_set_zTempVar(uint8_t thr_id, const uint64_t *limbs)
{
    CHECK(cudaMemcpyToSymbol(c_zTempVar, limbs, 17*sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
}


extern "C" void cuda_init_primes(uint8_t thr_id,
                                 uint32_t *primes,
                                 uint32_t *primesInverseInvk,
                                 uint32_t nPrimeLimit,
                                 uint32_t nBitArray_Size,
                                 uint32_t sharedSizeKB,
                                 uint32_t nPrimorialEndPrime,
                                 uint32_t nPrimeLimitA)
{
    uint32_t primeinverseinvk_size = sizeof(uint32_t) * 4 * nPrimeLimit;
    uint32_t nonce64_size = OFFSETS_MAX * sizeof(uint64_t);
    uint32_t nonce32_size = OFFSETS_MAX * sizeof(uint32_t);
    uint32_t sharedSizeBits = sharedSizeKB * 1024 * 8;
    uint32_t allocSize = ((nBitArray_Size * 16  + sharedSizeBits - 1) / sharedSizeBits) * sharedSizeBits;
    uint32_t bitarray_size = (allocSize+31)/32 * sizeof(uint32_t);
    uint32_t remainder_size = 2 * nPrimeLimitA * 8 * sizeof(uint32_t);

    /* Allocate memory for the primes, inverses, and reciprocals that are used
       as the basis for prime sieve computation */
    CHECK(cudaMalloc(&d_primesInverseInvk[thr_id],  primeinverseinvk_size));
    CHECK(cudaMemcpy(d_primesInverseInvk[thr_id], primesInverseInvk, primeinverseinvk_size, cudaMemcpyHostToDevice));

    /* Allocate base remainders that will be pre-computed once per block */
    CHECK(cudaMalloc(&d_base_remainders[thr_id],  nPrimeLimit * sizeof(uint32_t)));

    /* Create list of primes only */
    CHECK(cudaMalloc(&d_primes[thr_id], nPrimeLimit * sizeof(uint32_t)));
    CHECK(cudaMemcpy(d_primes[thr_id], primes, nPrimeLimit * sizeof(uint32_t), cudaMemcpyHostToDevice));

    /* Allocate multiple frame resources so we can keep multiple frames in flight
       to further improve CPU/GPU utilization */

    for(uint8_t i = 0; i < FRAME_COUNT; ++i)
    {
        /* test */
        CHECK(    cudaMalloc(&frameResources[thr_id].d_result_offsets[i], nonce64_size));
        CHECK(cudaMallocHost(&frameResources[thr_id].h_result_offsets[i], nonce64_size));
        CHECK(    cudaMalloc(&frameResources[thr_id].d_result_meta[i],    nonce32_size));
        CHECK(cudaMallocHost(&frameResources[thr_id].h_result_meta[i],    nonce32_size));
        CHECK(    cudaMalloc(&frameResources[thr_id].d_result_count[i],   sizeof(uint32_t)));
        CHECK(cudaMallocHost(&frameResources[thr_id].h_result_count[i],   sizeof(uint32_t)));

        /* test stats */
        CHECK(    cudaMalloc(&frameResources[thr_id].d_primes_checked[i], 16 * sizeof(uint32_t)));
        CHECK(cudaMallocHost(&frameResources[thr_id].h_primes_checked[i], 16 * sizeof(uint32_t)));
        CHECK(    cudaMalloc(&frameResources[thr_id].d_primes_found[i],   16 * sizeof(uint32_t)));
        CHECK(cudaMallocHost(&frameResources[thr_id].h_primes_found[i],   16 * sizeof(uint32_t)));

        /* compaction */
        CHECK(    cudaMalloc(&frameResources[thr_id].d_nonce_offsets[i], nonce64_size * BUFFER_COUNT));
        CHECK(    cudaMalloc(&frameResources[thr_id].d_nonce_meta[i],    nonce32_size * BUFFER_COUNT));
        CHECK(    cudaMalloc(&frameResources[thr_id].d_nonce_count[i],   sizeof(uint32_t) * 4 * BUFFER_COUNT));
        CHECK(cudaMallocHost(&frameResources[thr_id].h_nonce_count[i],   sizeof(uint32_t)));

        /* sieving */
        CHECK(    cudaMalloc(&frameResources[thr_id].d_prime_remainders[i], remainder_size));
        //CHECK(    cudaMalloc(&frameResources[thr_id].d_bit_array_sieve[i], bitarray_size));

        /* combo sieve */
        CHECK(    cudaMalloc(&frameResources[thr_id].d_bit_array_sieve[i], bitarray_size));


        /* bucket sieve (experimental) */
        // CHECK(cudaMalloc(&frameResources[thr_id].d_bucket_o[i], sizeof(uint32_t) * nPrimeLimit << 4));
        // CHECK(cudaMalloc(&frameResources[thr_id].d_bucket_away[i], sizeof(uint16_t) * nPrimeLimit << 4));
    }

    /* Have capacity for small primes up to 4096 */
    uint16_t p[4096];
    for(uint32_t i = 0; i < nPrimeLimitA; ++i)
        p[i] = primes[i];

    /* Copy small primes to GPU. */
    CHECK(cudaMemcpyToSymbol(c_primes, p,
        nPrimeLimitA * sizeof(uint16_t), 0, cudaMemcpyHostToDevice));

    /* Allocate and compute block offsets for a list of small prime mod offsets
       at each block offset in the gpu small sieve kernel */
    uint32_t nBlocks = (nBitArray_Size + sharedSizeBits-1) / sharedSizeBits;
    uint32_t blockoffset_size = nBlocks * 4096 * sizeof(uint16_t);

    CHECK(cudaMalloc(&d_blockoffset_mod_p[thr_id], blockoffset_size));

    uint16_t *offsets = (uint16_t *)malloc(blockoffset_size);

    for (uint32_t block = 0; block < nBlocks; ++block)
    {
        uint32_t blockOffset = sharedSizeBits * block;

        for (uint32_t i = 0; i < nPrimeLimitA; ++i)
            offsets[block*4096 + i] = primes[i] - (blockOffset % primes[i]);
    }
    CHECK(cudaMemcpy(d_blockoffset_mod_p[thr_id], offsets, blockoffset_size, cudaMemcpyHostToDevice));
    free(offsets);

    /* Create the CUDA streams and events used for sieve, compacting, and testing */
    streams_events_init(thr_id);
}

extern "C" void cuda_free_primes(uint8_t thr_id)
{
    CHECK(cudaFree(d_primesInverseInvk[thr_id]));
    CHECK(cudaFree(d_base_remainders[thr_id]));
    CHECK(cudaFree(d_primes[thr_id]));

    for(uint8_t i = 0; i < FRAME_COUNT; ++i)
    {
        CHECK(    cudaFree(frameResources[thr_id].d_result_offsets[i]));
        CHECK(cudaFreeHost(frameResources[thr_id].h_result_offsets[i]));

        CHECK(    cudaFree(frameResources[thr_id].d_result_meta[i]));
        CHECK(cudaFreeHost(frameResources[thr_id].h_result_meta[i]));

        CHECK(    cudaFree(frameResources[thr_id].d_result_count[i]));
        CHECK(cudaFreeHost(frameResources[thr_id].h_result_count[i]));

        CHECK(    cudaFree(frameResources[thr_id].d_primes_checked[i]));
        CHECK(cudaFreeHost(frameResources[thr_id].h_primes_checked[i]));

        CHECK(    cudaFree(frameResources[thr_id].d_primes_found[i]));
        CHECK(cudaFreeHost(frameResources[thr_id].h_primes_found[i]));

        CHECK(    cudaFree(frameResources[thr_id].d_nonce_offsets[i]));
        CHECK(    cudaFree(frameResources[thr_id].d_nonce_meta[i]));
        CHECK(    cudaFree(frameResources[thr_id].d_nonce_count[i]));
        CHECK(cudaFreeHost(frameResources[thr_id].h_nonce_count[i]));

        CHECK(    cudaFree(frameResources[thr_id].d_prime_remainders[i]));
        CHECK(    cudaFree(frameResources[thr_id].d_bit_array_sieve[i]));

        //CHECK(cudaFree(frameResources[thr_id].d_bucket_o[i]));
        //CHECK(cudaFree(frameResources[thr_id].d_bucket_away[i]));
    }

    CHECK(cudaFree(d_blockoffset_mod_p[thr_id]));

    streams_events_free(thr_id);
}

__global__ void base_remainders_kernel(uint4 *primes, uint32_t *base_remainders, uint32_t nPrimeLimit)
{
    uint32_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < nPrimeLimit)
    {
        uint4 tmp = primes[i];
        uint64_t rec = make_uint64_t(tmp.z, tmp.w);
        base_remainders[i] = mpi_mod_int(c_zTempVar, tmp.x, rec);
    }
}

extern "C" void cuda_base_remainders(uint8_t thr_id, uint32_t nPrimeLimit)
{
    int nThreads = nPrimeLimit;
    int nThreadsPerBlock = 256;
    int nBlocks = (nThreads + nThreadsPerBlock-1) / nThreadsPerBlock;

    dim3 block(nThreadsPerBlock);
    dim3 grid(nBlocks);

    base_remainders_kernel<<<grid, block, 0>>>(d_primesInverseInvk[thr_id],
                                               d_base_remainders[thr_id],
                                               nPrimeLimit);
}


__global__ void primesieve_kernelA0(uint64_t origin,
                                    uint4 *primes,
                                    uint32_t *prime_remainders,
                                    uint32_t *base_remainders,
                                    uint16_t nPrimeLimit,
                                    uint8_t nOffsets)
{
    uint16_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i < nPrimeLimit)
    {
        uint4 tmp = primes[i];
        uint64_t recip = make_uint64_t(tmp.z, tmp.w);

        for(uint8_t o = 0; o < nOffsets; ++o)
        {
            tmp.z = mod_p_small(origin + base_remainders[i] + c_offsets[c_iA[o]], tmp.x, recip);
            prime_remainders[(i << 3) + o] = mod_p_small((uint64_t)(tmp.x - tmp.z)*tmp.y, tmp.x, recip);
        }
    }
}

__global__ void primesieve_kernelB0(uint64_t origin,
                                    uint4 *primes,
                                    uint32_t *prime_remainders,
                                    uint32_t *base_remainders,
                                    uint16_t nPrimeLimit,
                                    uint8_t nOffsets)
{
    uint16_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i < nPrimeLimit)
    {
        uint4 tmp = primes[i];
        uint64_t recip = make_uint64_t(tmp.z, tmp.w);

        for(uint8_t o = 0; o < nOffsets; ++o)
        {
            tmp.z = mod_p_small(origin + base_remainders[i] + c_offsets[c_iB[o]], tmp.x, recip);
            prime_remainders[(i << 3) + o] = mod_p_small((uint64_t)(tmp.x - tmp.z)*tmp.y, tmp.x, recip);
        }
    }
}

template<uint8_t offsetsA>
__global__ void primesieve_kernelA_1024(uint32_t *g_bit_array_sieve,
                                        uint32_t *prime_remainders,
                                        uint16_t *blockoffset_mod_p,
                                        uint16_t nPrimorialEndPrime,
                                        uint16_t nPrimeLimitA)
{
    extern __shared__ uint32_t shared_array_sieve[];

    uint16_t i, j;

    #pragma unroll 8
    for (int i= 0; i <  8; ++i)
        shared_array_sieve[threadIdx.x + (i << 10)] = 0;

    __syncthreads();

    for (i = nPrimorialEndPrime; i < nPrimeLimitA; ++i)
    {
        uint16_t pr = c_primes[i];
        uint16_t pre2 = blockoffset_mod_p[(blockIdx.x << 12) + i];

        // precompute
        uint32_t pIdx = threadIdx.x * pr;
        uint32_t nAdd = pr << 10;

        uint32_t pre1[offsetsA];
        auto pre = [&pre1, &prime_remainders, &i](uint32_t o)
        {
            pre1[o] = prime_remainders[(i << 3) + o]; // << 4 because we have space for 16 offsets
        };

        Unroller<0, offsetsA>::step(pre);

        uint32_t index;
        auto loop = [&pIdx, &nAdd, &pre1, &pre2, &pr, &index](uint32_t o)
        {
            index = pre1[o] + pre2;
            if(index >= pr)
                index = index - pr;

            for(index = index + pIdx; index < 262144; index += nAdd)
            {
                atomicOr(&shared_array_sieve[index >> 5], 1 << (index & 31));
            }

        };

        Unroller<0, offsetsA>::step(loop);
    }

    __syncthreads();
    g_bit_array_sieve += (blockIdx.x << 13);

    #pragma unroll 8
    for (int i = 0; i < 8; ++i) // fixed value
    {
        j = threadIdx.x + (i << 10);
        //atomicOr(&g_bit_array_sieve[j], shared_array_sieve[j]);
        g_bit_array_sieve[j] = shared_array_sieve[j];
    }
}
template<uint8_t offsetsA>
__global__ void primesieve_kernelA_512(uint32_t *g_bit_array_sieve,
                                       uint32_t *prime_remainders,
                                       uint16_t *blockoffset_mod_p,
                                       uint16_t nPrimorialEndPrime,
                                       uint16_t nPrimeLimitA)
{
    extern __shared__ uint32_t shared_array_sieve[];

    uint16_t i, j;

    #pragma unroll 16
    for (int i= 0; i <  16; ++i)
        shared_array_sieve[threadIdx.x + (i << 9)] = 0;

    __syncthreads();

    for (i = nPrimorialEndPrime; i < nPrimeLimitA; ++i)
    {
        uint16_t pr = c_primes[i];
        uint16_t pre2 = blockoffset_mod_p[(blockIdx.x << 12) + i];

        // precompute
        uint32_t pIdx = threadIdx.x * pr;
        uint32_t nAdd = pr << 9;

        uint32_t pre1[offsetsA];
        auto pre = [&pre1, &prime_remainders, &i](uint32_t o)
        {
            pre1[o] = prime_remainders[(i << 3) + o]; // << 4 because we have space for 16 offsets
        };

        Unroller<0, offsetsA>::step(pre);

        uint32_t index;
        auto loop = [&pIdx, &nAdd, &pre1, &pre2, &pr, &index](uint32_t o)
        {
            index = pre1[o] + pre2;
            if(index >= pr)
                index = index - pr;

                for(index = index + pIdx; index < 262144; index += nAdd)
                    atomicOr(&shared_array_sieve[index >> 5], 1 << (index & 31));
        };

        Unroller<0, offsetsA>::step(loop);
    }

    __syncthreads();
    g_bit_array_sieve += (blockIdx.x << 13);

    #pragma unroll 16
    for (int i = 0; i < 16; ++i) // fixed value
    {
        j = threadIdx.x + (i << 9);
        atomicOr(&g_bit_array_sieve[j], shared_array_sieve[j]);
    }
}

template<uint8_t offsetsA>
__global__ void primesieve_kernelA_256(uint32_t *g_bit_array_sieve,
                                       uint32_t *prime_remainders,
                                       uint16_t *blockoffset_mod_p,
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

        uint32_t pre1[offsetsA];
        auto pre = [&pre1, &prime_remainders, &i](uint32_t o)
        {
            pre1[o] = prime_remainders[(i << 3) + o]; // << 4 because we have space for 16 offsets
        };

        Unroller<0, offsetsA>::step(pre);

        uint32_t index;
        auto loop = [&pIdx, &nAdd, &pre1, &pre2, &pr, &index](uint32_t o)
        {
            index = pre1[o] + pre2;
            if(index >= pr)
                index = index - pr;

            for(index = index + pIdx; index < 262144; index += nAdd)
                atomicOr(&shared_array_sieve[index >> 5], 1 << (index & 31));
        };

        Unroller<0, offsetsA>::step(loop);
    }

    __syncthreads();
    g_bit_array_sieve += (blockIdx.x << 13);

    #pragma unroll 32
    for (uint8_t i = 0; i < 32; ++i) // fixed value
    {
        uint16_t j = threadIdx.x + (i << 8);
        atomicOr(&g_bit_array_sieve[j], shared_array_sieve[j]);
    }
}

template<uint8_t offsetsA>
__global__ void primesieve_kernelA_128(uint32_t *g_bit_array_sieve,
                                       uint32_t *prime_remainders,
                                       uint16_t *blockoffset_mod_p,
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

        uint32_t pre1[offsetsA];
        auto pre = [&pre1, &prime_remainders, &i](uint32_t o)
        {
            pre1[o] = prime_remainders[(i << 3) + o]; // << 4 because we have space for 16 offsets
        };

        Unroller<0, offsetsA>::step(pre);

        uint32_t index;
        auto loop = [&pIdx, &nAdd, &pre1, &pre2, &pr, &index](uint32_t o)
        {
            index = pre1[o] + pre2;
            if(index >= pr)
                index = index - pr;

            for(index = index + pIdx; index < 262144; index += nAdd)
                atomicOr(&shared_array_sieve[index >> 5], 1 << (index & 31));
        };

        Unroller<0, offsetsA>::step(loop);
    }

    __syncthreads();
    g_bit_array_sieve += (blockIdx.x << 13);

    #pragma unroll 64
    for (uint8_t i = 0; i < 64; ++i) // fixed value
    {
        uint16_t j = threadIdx.x + (i << 7);
        atomicOr(&g_bit_array_sieve[j], shared_array_sieve[j]);
    }
}


__global__ void clearsieve_kernel(uint32_t *sieve, uint32_t n_words)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < n_words)
        sieve[i] = 0;
}

__global__ void primesieve_kernelB(uint64_t origin,
                                   uint32_t *bit_array_sieve,
                                   uint32_t bit_array_size,
                                   uint4 *primes,
                                   uint32_t *base_remainders,
                                   uint32_t nPrimorialEndPrime,
                                   uint32_t nPrimeLimit,
                                   uint32_t nOffsets)
{
    uint32_t position = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t i = nPrimorialEndPrime + (position >> 3);
    uint32_t o = position & 7;

    if(i < nPrimeLimit && o < nOffsets)
    {
        uint4 tmp = primes[i];
        uint64_t recip = make_uint64_t(tmp.z, tmp.w);

        tmp.z = mod_p_small(origin + base_remainders[i] + c_offsets[c_iA[o]], tmp.x, recip);
        tmp.w = mod_p_small((uint64_t)(tmp.x - tmp.z)*tmp.y, tmp.x, recip);

        for(; tmp.w < bit_array_size; tmp.w += tmp.x)
        {
            atomicOr(&bit_array_sieve[tmp.w >> 5], 1 << (tmp.w & 31));
        }
    }
}



__global__ void compact_offsets(uint64_t *d_nonce_offsets, uint32_t *d_nonce_meta, uint32_t *d_nonce_count,
                                uint32_t *d_bit_array_sieve, uint32_t nBitArray_Size, uint64_t sieve_start_index)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < nBitArray_Size)
    {
        uint64_t nonce_offset = sieve_start_index + idx;

        if((d_bit_array_sieve[idx >> 5] & (1 << (idx & 31))) == 0)
        {
            uint32_t i = atomicAdd(d_nonce_count, 1);

            if(i < OFFSETS_MAX)
            {
                d_nonce_offsets[i] = nonce_offset;

                uint32_t combo = 0xFFFF0000;
                d_nonce_meta[i] = combo;
            }
        }
    }
}


void kernelA0_launch(uint8_t thr_id,
                     uint8_t str_id,
                     uint8_t frame_index,
                     uint16_t nPrimeLimitA,
                     uint64_t origin)
{
    dim3 block(32);
    dim3 grid((nPrimeLimitA + block.x - 1) / block.x);

    primesieve_kernelA0<<<grid, block, 0, d_Streams[thr_id][str_id]>>>(
        origin,
        d_primesInverseInvk[thr_id],
        frameResources[thr_id].d_prime_remainders[frame_index],
        d_base_remainders[thr_id],
        nPrimeLimitA,
        nOffsetsA );
}

void kernelB0_launch(uint8_t thr_id,
                     uint8_t str_id,
                     uint8_t frame_index,
                     uint16_t nPrimeLimitA,
                     uint64_t origin)
{
    dim3 block(32);
    dim3 grid((nPrimeLimitA + block.x - 1) / block.x);

    primesieve_kernelB0<<<grid, block, 0, d_Streams[thr_id][str_id]>>>(
        origin,
        d_primesInverseInvk[thr_id],
        &frameResources[thr_id].d_prime_remainders[frame_index][nPrimeLimitA],
        d_base_remainders[thr_id],
        nPrimeLimitA,
        nOffsetsB );
}

#define KERNEL_A_LAUNCH(X) primesieve_kernelA_1024<X><<<grid, block, sharedSizeBits/8, d_Streams[thr_id][str_id]>>>(\
frameResources[thr_id].d_bit_array_sieve[frame_index], \
frameResources[thr_id].d_prime_remainders[frame_index], \
d_blockoffset_mod_p[thr_id], \
nPrimorialEndPrime, \
nPrimeLimitA)

void kernelA_launch(uint8_t thr_id,
                    uint8_t str_id,
                    uint8_t frame_index,
                    uint16_t nPrimorialEndPrime,
                    uint16_t nPrimeLimitA,
                    uint32_t nBitArray_Size)
{
    const int sharedSizeBits = 32 * 1024 * 8;
    int nBlocks = (nBitArray_Size + sharedSizeBits-1) / sharedSizeBits;

    dim3 block(1024);
    dim3 grid(nBlocks);

    switch(nOffsetsA)
    {
        case 1:  KERNEL_A_LAUNCH(1);  break;
        case 2:  KERNEL_A_LAUNCH(2);  break;
        case 3:  KERNEL_A_LAUNCH(3);  break;
        case 4:  KERNEL_A_LAUNCH(4);  break;
        case 5:  KERNEL_A_LAUNCH(5);  break;
        case 6:  KERNEL_A_LAUNCH(6);  break;
        case 7:  KERNEL_A_LAUNCH(7);  break;
        case 8:  KERNEL_A_LAUNCH(8);  break;
        case 9:  KERNEL_A_LAUNCH(9);  break;
        case 10: KERNEL_A_LAUNCH(10); break;
        case 11: KERNEL_A_LAUNCH(11); break;
        case 12: KERNEL_A_LAUNCH(12); break;
        case 13: KERNEL_A_LAUNCH(13); break;
        case 14: KERNEL_A_LAUNCH(14); break;
        case 15: KERNEL_A_LAUNCH(15); break;
    }
}


#define KERNEL_B_LAUNCH(X)   primesieve_kernelB<<<grid, block, 0, d_Streams[thr_id][str_id]>>>( \
origin, \
frameResources[thr_id].d_bit_array_sieve[frame_index], \
nBitArray_Size, \
d_primesInverseInvk[thr_id], \
d_base_remainders[thr_id], \
nPrimeLimitA, \
nPrimeLimitB, \
X )

void kernelB_launch(uint8_t thr_id,
                    uint8_t str_id,
                    uint64_t origin,
                    uint8_t frame_index,
                    uint32_t nPrimeLimitA,
                    uint32_t nPrimeLimitB,
                    uint32_t nBitArray_Size)
{
    uint32_t nThreads = nPrimeLimitB - nPrimeLimitA;
    uint32_t nThreadsPerBlock = 128;
    uint32_t nBlocks = (nThreads + (nThreadsPerBlock << 3) - 1) / (nThreadsPerBlock >> 3);

    dim3 block(nThreadsPerBlock);
    dim3 grid(nBlocks);

    switch(nOffsetsA)
    {
        case 1: KERNEL_B_LAUNCH(1); break;
        case 2: KERNEL_B_LAUNCH(2); break;
        case 3: KERNEL_B_LAUNCH(3); break;
        case 4: KERNEL_B_LAUNCH(4); break;
        case 5: KERNEL_B_LAUNCH(5); break;
        case 6: KERNEL_B_LAUNCH(6); break;
        case 7: KERNEL_B_LAUNCH(7); break;
        case 8: KERNEL_B_LAUNCH(8); break;
    }
}

void kernel_clear_launch(uint8_t thr_id, uint8_t str_id,
                         uint8_t curr_sieve, uint32_t nBitArray_Size)
{
    uint32_t sharedSizeBits = 32 * 1024 * 8;
    uint32_t allocSize = ((nBitArray_Size*16 + sharedSizeBits-1) / sharedSizeBits) * sharedSizeBits;

    uint32_t nSieveWords = (allocSize + 31) >> 5;

    dim3 block(64);
    dim3 grid((nSieveWords + block.x - 1) / block.x);

    clearsieve_kernel<<<grid, block, 0, d_Streams[thr_id][str_id]>>>(
    frameResources[thr_id].d_bit_array_sieve[curr_sieve], nSieveWords);
}

void kernel_compact_launch(uint8_t thr_id, uint8_t str_id, uint8_t curr_sieve, uint8_t curr_test,
                           uint8_t next_test, uint32_t nBitArray_Size, uint64_t primorial_start)
{
    dim3 block(128);
    dim3 grid((nBitArray_Size + block.x - 1) / block.x);

    compact_offsets<<<grid, block, 0, d_Streams[thr_id][str_id]>>>(
        frameResources[thr_id].d_nonce_offsets[curr_test],
        frameResources[thr_id].d_nonce_meta[curr_test],
        frameResources[thr_id].d_nonce_count[curr_test],
        frameResources[thr_id].d_bit_array_sieve[curr_sieve],
        nBitArray_Size,
        primorial_start);

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
}

extern "C" bool cuda_primesieve(uint8_t thr_id,
                                uint64_t base_offset,
                                uint64_t primorial,
                                uint16_t nPrimorialEndPrime,
                                uint16_t nPrimeLimitA,
                                uint32_t nPrimeLimitB,
                                uint32_t nBitArray_Size,
                                uint32_t nDifficulty,
                                uint32_t sieve_index,
                                uint32_t test_index)
{
    /* Get the current working sieve and test indices */
    uint8_t curr_sieve = sieve_index % FRAME_COUNT;
    uint8_t curr_test = test_index % FRAME_COUNT;
    uint32_t next_test = (test_index + 1) % FRAME_COUNT;
    uint32_t prev_test = (test_index - 1) % FRAME_COUNT;

    /* Make sure current working sieve is finished */
    //if(cudaEventQuery(d_Events[thr_id][curr_sieve][EVENT::COMPACT]) == cudaErrorNotReady
    //|| cudaEventQuery(d_Events[thr_id][prev_test ][EVENT::FERMAT ]) == cudaErrorNotReady)
    //    return false;
    CHECK(synchronize_event(thr_id, curr_sieve, EVENT::COMPACT));
    CHECK(synchronize_event(thr_id, prev_test, EVENT::FERMAT));



    /* Calculate bit array size, sieve start bit, and base offset */
    uint64_t primorial_start = (uint64_t)nBitArray_Size * (uint64_t)sieve_index;
    uint64_t base_offsetted = base_offset + primorial * primorial_start;
    uint8_t nComboThreshold = 8;


    {
        /* Wait for testing and compaction to finish before starting next round. */
        CHECK(stream_wait_event(thr_id, curr_sieve, STREAM::CLEAR, EVENT::COMPACT));
        CHECK(stream_wait_event(thr_id, prev_test, STREAM::CLEAR, EVENT::FERMAT));

        /* Clear the current working sieve and signal */
        kernel_clear_launch(thr_id, STREAM::CLEAR, curr_sieve, nBitArray_Size);


        /* Precompute prime remainders. */
        kernelA0_launch(thr_id, STREAM::SIEVE_A, curr_sieve,
          nPrimeLimitA, base_offsetted);

        kernelB0_launch(thr_id, STREAM::SIEVE_B, curr_sieve,
          nPrimeLimitA, base_offsetted);

        CHECK(stream_signal_event(thr_id, curr_sieve, STREAM::CLEAR, EVENT::CLEAR));
    }


    {
        CHECK(stream_wait_event(thr_id, curr_sieve, STREAM::SIEVE_A, EVENT::CLEAR));

        /* Single sieve (Launch small sieve, utilizing shared memory and signal) */
        kernelA_launch(thr_id, STREAM::SIEVE_A, curr_sieve,
                      nPrimorialEndPrime, nPrimeLimitA, nBitArray_Size);

        CHECK(stream_signal_event(thr_id, curr_sieve, STREAM::SIEVE_A, EVENT::SIEVE_A));
        CHECK(stream_wait_event(thr_id, curr_sieve, STREAM::SIEVE_B, EVENT::SIEVE_A));

        /* Single sieve (Launch large sieve, utilizing global memory and signal) */
        kernelB_launch(thr_id, STREAM::SIEVE_B, base_offsetted, curr_sieve,
                      nPrimeLimitA, nPrimeLimitB, nBitArray_Size);

        CHECK(stream_signal_event(thr_id, curr_sieve, STREAM::SIEVE_B, EVENT::SIEVE_B));
    }


    {
        CHECK(stream_wait_event(thr_id, curr_sieve, STREAM::SIEVE_A, EVENT::SIEVE_B));



        /* Combo sieve (Launch small sieve, utilizing shared memory and signal) */
        comboA_launch(thr_id, STREAM::SIEVE_A, curr_sieve,
                    nPrimorialEndPrime, nPrimeLimitA, nBitArray_Size);

        CHECK(stream_signal_event(thr_id, curr_sieve, STREAM::SIEVE_A, EVENT::SIEVE_A));
        CHECK(stream_wait_event(thr_id, curr_sieve, STREAM::SIEVE_B, EVENT::SIEVE_A));

        /* Combo sieve (Launch large sieve, utilizing global memory and signal) */
        comboB_launch(thr_id, STREAM::SIEVE_B, base_offsetted, curr_sieve,
                      nPrimeLimitA, nPrimeLimitB, nBitArray_Size);

        CHECK(stream_signal_event(thr_id, curr_sieve, STREAM::SIEVE_B, EVENT::SIEVE_B));
    }



    {   /* Launch compaction and signal */
        CHECK(stream_wait_events(thr_id, curr_sieve, STREAM::COMPACT, EVENT::SIEVE_A, EVENT::SIEVE_B));

        //kernel_compact_launch(thr_id, STREAM::COMPACT, curr_sieve, curr_test, next_test, nBitArray_Size, primorial_start);

        kernel_ccompact_launch(thr_id, STREAM::COMPACT, curr_sieve, curr_test, next_test, nBitArray_Size, primorial_start, nComboThreshold);

        CHECK(stream_signal_event(thr_id, curr_sieve, STREAM::COMPACT, EVENT::COMPACT));
    }


    debug::log(4, FUNCTION, (uint32_t)thr_id);

    return true;
}
