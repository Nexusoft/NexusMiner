/*******************************************************************************************

 Nexus Earth 2018

 (credits: cbuchner1 for sieving)

 [Scale Indefinitely] BlackJack. http://www.opensource.org/licenses/mit-license.php

*******************************************************************************************/

#include <CUDA/include/sieve.h>
#include <CUDA/include/util.h>
#include <CUDA/include/frame_resources.h>
#include <CUDA/include/streams_events.h>

#include <Util/include/debug.h>

#include <cuda.h>
#include <stdio.h>

/* create a 64-bit word from 32-bit lo and hi words */
__device__ __forceinline__
uint64_t make_uint64_t(uint32_t LO, uint32_t HI)
{
    #if __CUDA_ARCH__ >= 130
    return __double_as_longlong(__hiloint2double(HI, LO));
    #else
    return (((uint64_t)HI) << 32) | (uint64_t)LO;
    #endif
}

/* Given a 64-bit operand and reciprocal, and 32-bit modulo, return the 32-bit
modulus without using division  */
__device__ __forceinline__
uint32_t mod_p_small(uint64_t a, uint32_t p, uint64_t recip)
{
    uint64_t q = __umul64hi(a, recip);
    int64_t r = a - p*q;
    if (r >= p)
        r -= p;
    return (uint32_t)r;
}


__constant__ uint64_t c_zTempVar[17];
__constant__ uint32_t c_offsetsA[32];
__constant__ uint32_t c_offsetsB[32];
__constant__ uint16_t c_primes[4096];

uint4 *d_primesInverseInvk[GPU_MAX];
uint32_t *d_primes[GPU_MAX];
uint32_t *d_base_remainders[GPU_MAX];
uint16_t *d_blockoffset_mod_p[GPU_MAX];
uint8_t nOffsetsA;
uint8_t nOffsetsB;


struct FrameResource frameResources[GPU_MAX];


extern "C" void cuda_set_sieve_offsets(uint8_t thr_id,
                                       uint32_t *OffsetsA, uint8_t A_count,
                                       uint32_t *OffsetsB, uint8_t B_count)
{
    nOffsetsA = A_count;
    nOffsetsB = B_count;

    if(nOffsetsA > 16 || nOffsetsB > 16)
        exit(1);

    CHECK(cudaMemcpyToSymbol(c_offsetsA, OffsetsA,
        nOffsetsA*sizeof(uint32_t), 0, cudaMemcpyHostToDevice));

    CHECK(cudaMemcpyToSymbol(c_offsetsB, OffsetsB,
        nOffsetsB*sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
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
    //uint32_t nonce32_size = OFFSETS_MAX * sizeof(uint32_t);
    uint32_t sharedSizeBits = sharedSizeKB * 1024 * 8;
    uint32_t allocSize = ((nBitArray_Size * 16  + sharedSizeBits - 1) / sharedSizeBits) * sharedSizeBits;
    uint32_t bitarray_size = (allocSize+31)/32 * sizeof(uint32_t);
    uint32_t remainder_size = nPrimeLimit * 16 * sizeof(uint32_t);

    /* Allocate memory for the primes, inverses, and reciprocals that are used
       as the basis for prime sieve computation */
    CHECK(cudaMalloc(&d_primesInverseInvk[thr_id],  primeinverseinvk_size));
    CHECK(cudaMemcpy(d_primesInverseInvk[thr_id], primesInverseInvk, primeinverseinvk_size, cudaMemcpyHostToDevice));

    /* allocate base remainders that will be pre-computed once per block */
    CHECK(cudaMalloc(&d_base_remainders[thr_id],  nPrimeLimit * sizeof(uint32_t)));

    /* create list of primes only */
    CHECK(cudaMalloc(&d_primes[thr_id], nPrimeLimit * sizeof(uint32_t)));
    CHECK(cudaMemcpy(d_primes[thr_id], primes, nPrimeLimit * sizeof(uint32_t), cudaMemcpyHostToDevice));

    /* Allocate multiple frame resources so we can keep multiple frames in flight
       to further improve CPU/GPU utilization */
    for(uint8_t i = 0; i < FRAME_COUNT; ++i)
    {
        /* test */
        CHECK(    cudaMalloc(&frameResources[thr_id].d_result_offsets[i], nonce64_size));
        CHECK(cudaMallocHost(&frameResources[thr_id].h_result_offsets[i], nonce64_size));
        CHECK(    cudaMalloc(&frameResources[thr_id].d_result_meta[i],    nonce64_size));
        CHECK(cudaMallocHost(&frameResources[thr_id].h_result_meta[i],    nonce64_size));
        CHECK(    cudaMalloc(&frameResources[thr_id].d_result_count[i],   sizeof(uint32_t)));
        CHECK(cudaMallocHost(&frameResources[thr_id].h_result_count[i],   sizeof(uint32_t)));

        /* test stats */
        CHECK(    cudaMalloc(&frameResources[thr_id].d_primes_checked[i], sizeof(uint32_t)));
        CHECK(cudaMallocHost(&frameResources[thr_id].h_primes_checked[i], sizeof(uint32_t)));
        CHECK(    cudaMalloc(&frameResources[thr_id].d_primes_found[i],   sizeof(uint32_t)));
        CHECK(cudaMallocHost(&frameResources[thr_id].h_primes_found[i],   sizeof(uint32_t)));

        /* compaction */
        CHECK(    cudaMalloc(&frameResources[thr_id].d_nonce_offsets[i], nonce64_size * BUFFER_COUNT));
        CHECK(    cudaMalloc(&frameResources[thr_id].d_nonce_meta[i],    nonce64_size * BUFFER_COUNT));
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

    uint16_t p[4096];
    for(uint32_t i = 0; i < nPrimeLimitA; ++i)
        p[i] = primes[i];

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

__device__ uint32_t mpi_mod_int(uint64_t *A, uint32_t B, uint64_t recip)
{
    if (B == 1)
        return 0;
    else if (B == 2)
        return A[0]&1;

    uint8_t i;
    uint64_t x,y;

    #pragma unroll 16
    for( i = 16, y = 0; i > 0; --i)
    {
        x  = A[i-1];
        y = (y << 32) | (x >> 32);
        y = mod_p_small(y, B, recip);

        x <<= 32;
        y = (y << 32) | (x >> 32);
        y = mod_p_small(y, B, recip);
    }

    return (uint32_t)y;
}

__global__ void base_remainders_kernel(uint4 *primes, uint32_t *base_remainders,
uint32_t nPrimorialEndPrime, uint32_t nPrimeLimit)
{
    uint32_t i = nPrimorialEndPrime + blockDim.x * blockIdx.x + threadIdx.x;
    if (i < nPrimeLimit)
    {
        uint4 tmp = primes[i];
        uint64_t rec = make_uint64_t(tmp.z, tmp.w);
        base_remainders[i] = mpi_mod_int(c_zTempVar, tmp.x, rec);
    }
}

extern "C" void cuda_base_remainders(uint8_t thr_id,
                                     uint32_t nPrimorialEndPrime,
                                     uint32_t nPrimeLimit)
{
    int nThreads = nPrimeLimit - nPrimorialEndPrime;
    int nThreadsPerBlock = 256;
    int nBlocks = (nThreads + nThreadsPerBlock-1) / nThreadsPerBlock;

    dim3 block(nThreadsPerBlock);
    dim3 grid(nBlocks);

    base_remainders_kernel<<<grid, block, 0>>>(d_primesInverseInvk[thr_id],
                                               d_base_remainders[thr_id],
                                               nPrimorialEndPrime,
                                               nPrimeLimit);
}

template<int Begin, int End, int Step = 1>
struct Unroller
{
    template<typename Action>
    __device__ __forceinline__ static void step(Action& action)
    {
        action(Begin);
        Unroller<Begin+Step, End, Step>::step(action);
    }
};

template<int End, int Step>
struct Unroller<End, End, Step>
{
    template<typename Action>
    __device__ __forceinline__ static void step(Action& action)
    {
    }
};


template<uint8_t nOffsets>
__global__ void primesieve_kernelA0(uint64_t origin,
                                    uint4 *primes,
                                    uint32_t *prime_remainders,
                                    uint32_t *base_remainders,
                                    uint16_t nPrimorialEndPrime,
                                    uint16_t nPrimeLimit)
{
    uint16_t i = nPrimorialEndPrime + blockDim.x * blockIdx.x + threadIdx.x;

    if(i < nPrimeLimit)
    {
        uint4 tmp = primes[i];
        uint64_t recip = make_uint64_t(tmp.z, tmp.w);

        #pragma unroll nOffsets
        for(uint8_t o = 0; o < nOffsets; ++o)
        {
            tmp.z = mod_p_small(origin + base_remainders[i] + c_offsetsA[o], tmp.x, recip);
            prime_remainders[(i << 4) + o] = mod_p_small((uint64_t)(tmp.x - tmp.z)*tmp.y, tmp.x, recip);
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
            pre1[o] = prime_remainders[(i << 4) + o]; // << 4 because we have space for 16 offsets
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

    #pragma unroll 8
    for (int i = 0; i < 8; ++i) // fixed value
    {
        j = threadIdx.x + (i << 10);
        atomicOr(&g_bit_array_sieve[j], shared_array_sieve[j]);
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
            pre1[o] = prime_remainders[(i << 4) + o]; // << 4 because we have space for 16 offsets
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
            pre1[o] = prime_remainders[(i << 4) + o]; // << 4 because we have space for 16 offsets
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
            pre1[o] = prime_remainders[(i << 4) + o]; // << 4 because we have space for 16 offsets
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

template<uint8_t o>
__global__ void combosieve_kernelA_512(uint32_t *g_bit_array_sieve,
                                       uint32_t *prime_remainders,
                                       uint16_t *blockoffset_mod_p,
                                       uint16_t nPrimorialEndPrime,
                                       uint16_t nPrimeLimitA)
{
    extern __shared__ uint32_t shared_array_sieve[];

    #pragma unroll 16
    for (uint8_t i= 0; i <  16; ++i)
        shared_array_sieve[threadIdx.x + (i << 9)] = 0;

    __syncthreads();

    for (uint16_t i = nPrimorialEndPrime; i < nPrimeLimitA; ++i)
    {
        uint16_t pr = c_primes[i];
        uint16_t pre2 = blockoffset_mod_p[(blockIdx.x << 12) + i];

        // precompute
        uint32_t pIdx = threadIdx.x * pr;
        uint32_t nAdd = pr << 9;

        uint32_t index = prime_remainders[(i << 4) + o] + pre2; // << 4 because we have space for 16 offsets

        if(index >= pr)
            index = index - pr;

        for(index = index + pIdx; index < 262144; index += nAdd)
            atomicOr(&shared_array_sieve[index >> 5], 1 << (index & 31));
    }

    __syncthreads();
    g_bit_array_sieve += (blockIdx.x << 13);

    #pragma unroll 16
    for (uint8_t i = 0; i < 16; ++i) // fixed value
    {
        uint16_t j = threadIdx.x + (i << 9);
        atomicOr(&g_bit_array_sieve[j], shared_array_sieve[j]);
    }
}

template<uint8_t o>
__global__ void combosieve_kernelA_256(uint32_t *g_bit_array_sieve,
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

        uint32_t index = prime_remainders[(i << 4) + o] + pre2; // << 4 because we have space for 16 offsets

        if(index >= pr)
            index = index - pr;

        for(index = index + pIdx; index < 262144; index += nAdd)
            atomicOr(&shared_array_sieve[index >> 5], 1 << (index & 31));
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

template<uint8_t o>
__global__ void combosieve_kernelA_128(uint32_t *g_bit_array_sieve,
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

        uint32_t index = prime_remainders[(i << 4) + o] + pre2; // << 4 because we have space for 16 offsets

        if(index >= pr)
            index = index - pr;

        for(index = index + pIdx; index < 262144; index += nAdd)
            atomicOr(&shared_array_sieve[index >> 5], 1 << (index & 31));
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
            tmp.z = mod_p_small(origin + base_remainders[idx] + c_offsetsB[j], tmp.x, recip);
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

template<uint8_t offsetsB>
__global__ void primesieve_kernelB(uint64_t origin,
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

        #pragma unroll offsetsB
        for(uint8_t o = 0; o < offsetsB; ++o)
        {
            tmp.z = mod_p_small(origin + base_remainders[i] + c_offsetsB[o], tmp.x, recip);
            tmp.w = mod_p_small((uint64_t)(tmp.x - tmp.z)*tmp.y, tmp.x, recip);

            for(; tmp.w < bit_array_size; tmp.w += tmp.x)
            {
                atomicOr(&bit_array_sieve[tmp.w >> 5], 1 << (tmp.w & 31));
            }
        }
    }
}

template<uint8_t o>
__global__ void combosieve_kernelB(uint64_t origin,
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

        tmp.z = mod_p_small(origin + base_remainders[i] + c_offsetsB[o], tmp.x, recip);
        tmp.w = mod_p_small((uint64_t)(tmp.x - tmp.z)*tmp.y, tmp.x, recip);

        for(; tmp.w < bit_array_size; tmp.w += tmp.x)
        {
            atomicOr(&bit_array_sieve[tmp.w >> 5], 1 << (tmp.w & 31));
        }
    }
}

__global__ void compact_offsets(uint64_t *d_nonce_offsets, uint64_t *d_nonce_meta, uint32_t *d_nonce_count,
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
                d_nonce_meta[i] = 0;
            }
        }
    }
}

template<uint8_t nOffsets>
__global__ void compact_combo(uint64_t *d_nonce_offsets, uint64_t *d_nonce_meta, uint32_t *d_nonce_count,
                              uint32_t *d_bit_array_sieve, uint32_t nBitArray_Size, uint64_t sieve_start_index, uint8_t threshold)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    //uint32_t sharedSizeBits = 32 * 1024 * 8;
    //uint32_t allocSize = ((nBitArray_Size + sharedSizeBits-1) / sharedSizeBits) * sharedSizeBits;
    //uint32_t nSieveWords = (allocSize + 31) >> 5;

    if(idx < nBitArray_Size)
    {
        uint64_t nonce_offset = sieve_start_index + idx;
        uint32_t combo = 0;
        uint8_t count = 0;

        /* Take the bit from each sieve and compact them into a single word. */
        #pragma unroll nOffsets
        for(uint8_t o = 0; o < nOffsets; ++o)
        {
            /* Use logical not operator to reduce result into inverted 0 or 1 bit. */
            uint16_t bit = !((d_bit_array_sieve[idx >> 5]) & (1 << (idx & 31)));
            combo |= bit << o;

            d_bit_array_sieve += nBitArray_Size >> 5;
        }

        /* Count the bits for this combo. */
        count = __popc(combo);

        /* Make sure the sieved bit count is at least as large as the target
         * prime cluster threshold. */
        if(count >= threshold)
        {
            //printf("combo: %08X invert: %08X popc: %d\n", combo, ~combo, count);

            /* Shift the combo bits to the most significant bit. */
            uint32_t tmp = combo << (32 - nOffsets);

            /* Count the leading zero bits to compute next offset. */
            uint8_t next = __clz(tmp);
            uint8_t prev = next;

            /* Set the beginning index. */
            uint8_t beg = next;

            /* Clear the high order bit for next round offset calculation. */
            tmp ^= 0x80000000 >> next;
            combo = tmp;


            /* Count the first index */
            count = 1;

            /* Continue loop while the next bit is less than the number of offsets. */
            while(next < nOffsets)
            {
                /* Set previous and next indicies, clearing high order bit. */
                prev = next;
                next = __clz(tmp);
                tmp ^= 0x80000000 >> next;

                /* Make sure next does not overflow. */
                if(next >= nOffsets)
                    break;

                /* Check the prime gap, resetting count and begin index if violated. */
                if(c_offsetsA[next] - c_offsetsA[prev] > 12)
                {
                    beg = next;
                    count = 0;
                    combo = tmp;
                }

                /* Increment the count. */
                ++count;

                /* Stop search if threshold is met. */
                if(count >= threshold)
                    break;

            }


            /* We want to make sure that the combination follows prime gap rule
               as close as possible */
            if(count >= threshold)
            {
                uint32_t i = atomicAdd(d_nonce_count, 1);

                if(i < OFFSETS_MAX)
                {
                    /* Encode the beginning nonce meta data for testing. */
                    uint64_t nonce_meta = 0;
                    nonce_meta |= ((uint64_t)combo << 32);
                    nonce_meta |= ((uint64_t)beg << 24);
                    nonce_meta |= ((uint64_t)beg << 16);

                    /* Assign the global nonce offset and meta data. */
                    d_nonce_offsets[i] = nonce_offset;
                    d_nonce_meta[i] = nonce_meta;
                }
                else
                    printf("OFFSETS_MAX EXCEEDED\n");
            }
        }
    }
}


#define KERNEL_A0_LAUNCH(X) primesieve_kernelA0<X><<<grid, block, 0, d_Streams[thr_id][str_id]>>>( \
origin, \
d_primesInverseInvk[thr_id], \
frameResources[thr_id].d_prime_remainders[frame_index], \
d_base_remainders[thr_id], \
nPrimorialEndPrime, \
nPrimeLimitA); \

void kernelA0_launch(uint8_t thr_id,
                     uint8_t str_id,
                     uint8_t frame_index,
                     uint16_t nPrimorialEndPrime,
                     uint16_t nPrimeLimitA,
                     uint64_t origin)
{
    dim3 block(32);
    dim3 grid((nPrimeLimitA - nPrimorialEndPrime + block.x - 1) / block.x);

    switch(nOffsetsA)
    {
        case 1:  KERNEL_A0_LAUNCH(1);  break;
        case 2:  KERNEL_A0_LAUNCH(2);  break;
        case 3:  KERNEL_A0_LAUNCH(3);  break;
        case 4:  KERNEL_A0_LAUNCH(4);  break;
        case 5:  KERNEL_A0_LAUNCH(5);  break;
        case 6:  KERNEL_A0_LAUNCH(6);  break;
        case 7:  KERNEL_A0_LAUNCH(7);  break;
        case 8:  KERNEL_A0_LAUNCH(8);  break;
        case 9:  KERNEL_A0_LAUNCH(9);  break;
        case 10: KERNEL_A0_LAUNCH(10); break;
        case 11: KERNEL_A0_LAUNCH(11); break;
        case 12: KERNEL_A0_LAUNCH(12); break;
        case 13: KERNEL_A0_LAUNCH(13); break;
        case 14: KERNEL_A0_LAUNCH(14); break;
        case 15: KERNEL_A0_LAUNCH(15); break;
    }
}

#define KERNEL_A_LAUNCH(X) primesieve_kernelA_256<X><<<grid, block, sharedSizeBits/8, d_Streams[thr_id][str_id]>>>(\
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

    dim3 block(256);
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


#define COMBO_A_LAUNCH(X) combosieve_kernelA_256<X><<<grid, block, sharedSizeBits/8, d_Streams[thr_id][str_id]>>>(\
&frameResources[thr_id].d_bit_array_sieve[frame_index][X * nBitArray_Size >> 5], \
frameResources[thr_id].d_prime_remainders[frame_index], \
d_blockoffset_mod_p[thr_id], \
nPrimorialEndPrime, \
nPrimeLimitA)

void comboA_launch(uint8_t thr_id,
                    uint8_t str_id,
                    uint8_t frame_index,
                    uint16_t nPrimorialEndPrime,
                    uint16_t nPrimeLimitA,
                    uint32_t nBitArray_Size)
{
    uint32_t sharedSizeBits = 32 * 1024 * 8;
    uint32_t nBlocks = (nBitArray_Size + sharedSizeBits-1) / sharedSizeBits;

    //uint32_t allocSize = ((nBitArray_Size + sharedSizeBits-1) / sharedSizeBits) * sharedSizeBits;
    //uint32_t nSieveWords = (allocSize + 31) >> 5;

    dim3 block(256);
    dim3 grid(nBlocks);

    /* fall-through switch logic, zero-based indexing */
    switch(nOffsetsA)
    {
        case 15: COMBO_A_LAUNCH(14);
        case 14: COMBO_A_LAUNCH(13);
        case 13: COMBO_A_LAUNCH(12);
        case 12: COMBO_A_LAUNCH(11);
        case 11: COMBO_A_LAUNCH(10);
        case 10: COMBO_A_LAUNCH(9);
        case 9:  COMBO_A_LAUNCH(8);
        case 8:  COMBO_A_LAUNCH(7);
        case 7:  COMBO_A_LAUNCH(6);
        case 6:  COMBO_A_LAUNCH(5);
        case 5:  COMBO_A_LAUNCH(4);
        case 4:  COMBO_A_LAUNCH(3);
        case 3:  COMBO_A_LAUNCH(2);
        case 2:  COMBO_A_LAUNCH(1);
        case 1:  COMBO_A_LAUNCH(0);
        default:
        break;
    }

    debug::log(4, FUNCTION, (uint32_t)thr_id);
}


#define KERNEL_B_LAUNCH(X)   primesieve_kernelB<X><<<grid, block, 0, d_Streams[thr_id][str_id]>>>( \
origin, \
frameResources[thr_id].d_bit_array_sieve[frame_index], \
nBitArray_Size, \
d_primesInverseInvk[thr_id], \
d_base_remainders[thr_id], \
nPrimorialEndPrime, \
nPrimeLimit)

void kernelB_launch(uint8_t thr_id,
                    uint8_t str_id,
                    uint64_t origin,
                    uint8_t frame_index,
                    uint16_t nPrimorialEndPrime,
                    uint32_t nPrimeLimit,
                    uint32_t nBitArray_Size)
{
    uint32_t nThreads = nPrimeLimit - nPrimorialEndPrime;
    uint32_t nThreadsPerBlock = 128;
    uint32_t nBlocks = (nThreads + nThreadsPerBlock - 1) / nThreadsPerBlock;

    dim3 block(nThreadsPerBlock);
    dim3 grid(nBlocks);

    switch(nOffsetsB)
    {
        case 1: KERNEL_B_LAUNCH(1); break;
        case 2: KERNEL_B_LAUNCH(2); break;
        case 3: KERNEL_B_LAUNCH(3); break;
        case 4: KERNEL_B_LAUNCH(4); break;
        case 5: KERNEL_B_LAUNCH(5); break;
        case 6: KERNEL_B_LAUNCH(6); break;
        case 7: KERNEL_B_LAUNCH(7); break;
        case 8: KERNEL_B_LAUNCH(8); break;
        case 9: KERNEL_B_LAUNCH(9); break;
        case 10: KERNEL_B_LAUNCH(10); break;
        case 11: KERNEL_B_LAUNCH(11); break;
        case 12: KERNEL_B_LAUNCH(12); break;
        case 13: KERNEL_B_LAUNCH(13); break;
        case 14: KERNEL_B_LAUNCH(14); break;
        case 15: KERNEL_B_LAUNCH(15); break;
    }
}


#define COMBO_B_LAUNCH(X)   combosieve_kernelB<X><<<grid, block, 0, d_Streams[thr_id][str_id]>>>( \
origin, \
&frameResources[thr_id].d_bit_array_sieve[frame_index][X * nBitArray_Size >> 5], \
nBitArray_Size, \
d_primesInverseInvk[thr_id], \
d_base_remainders[thr_id], \
nPrimorialEndPrime, \
nPrimeLimit)

void comboB_launch(uint8_t thr_id,
                    uint8_t str_id,
                    uint64_t origin,
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
        case 15: COMBO_B_LAUNCH(14);
        case 14: COMBO_B_LAUNCH(13);
        case 13: COMBO_B_LAUNCH(12);
        case 12: COMBO_B_LAUNCH(11);
        case 11: COMBO_B_LAUNCH(10);
        case 10: COMBO_B_LAUNCH(9);
        case 9:  COMBO_B_LAUNCH(8);
        case 8:  COMBO_B_LAUNCH(7);
        case 7:  COMBO_B_LAUNCH(6);
        case 6:  COMBO_B_LAUNCH(5);
        case 5:  COMBO_B_LAUNCH(4);
        case 4:  COMBO_B_LAUNCH(3);
        case 3:  COMBO_B_LAUNCH(2);
        case 2:  COMBO_B_LAUNCH(1);
        case 1:  COMBO_B_LAUNCH(0);
        default:
        break;
    }

    debug::log(4, FUNCTION, (uint32_t)thr_id);
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
                           uint32_t nBitArray_Size, uint64_t primorial_start)
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

    CHECK(cudaMemcpyAsync(
            frameResources[thr_id].h_nonce_count[curr_test],
            frameResources[thr_id].d_nonce_count[curr_test],
            sizeof(uint32_t), cudaMemcpyDeviceToHost, d_Streams[thr_id][str_id]));
}

#define COMBO_COMPACT_LAUNCH(X) compact_combo<X><<<grid, block, 0, d_Streams[thr_id][str_id]>>>( \
    frameResources[thr_id].d_nonce_offsets[curr_test], \
    frameResources[thr_id].d_nonce_meta[curr_test], \
    frameResources[thr_id].d_nonce_count[curr_test], \
    frameResources[thr_id].d_bit_array_sieve[curr_sieve], \
    nBitArray_Size, \
    primorial_start, \
    threshold)

void kernel_ccompact_launch(uint8_t thr_id, uint8_t str_id, uint8_t curr_sieve, uint8_t curr_test,
                            uint8_t next_test, uint32_t nBitArray_Size, uint64_t primorial_start, uint8_t threshold)
{
    dim3 block(64);
    dim3 grid((nBitArray_Size + block.x - 1) / block.x);

    switch(nOffsetsA)
    {
        case 1:  COMBO_COMPACT_LAUNCH(1);  break;
        case 2:  COMBO_COMPACT_LAUNCH(2);  break;
        case 3:  COMBO_COMPACT_LAUNCH(3);  break;
        case 4:  COMBO_COMPACT_LAUNCH(4);  break;
        case 5:  COMBO_COMPACT_LAUNCH(5);  break;
        case 6:  COMBO_COMPACT_LAUNCH(6);  break;
        case 7:  COMBO_COMPACT_LAUNCH(7);  break;
        case 8:  COMBO_COMPACT_LAUNCH(8);  break;
        case 9:  COMBO_COMPACT_LAUNCH(9);  break;
        case 10: COMBO_COMPACT_LAUNCH(10); break;
        case 11: COMBO_COMPACT_LAUNCH(11); break;
        case 12: COMBO_COMPACT_LAUNCH(12); break;
        case 13: COMBO_COMPACT_LAUNCH(13); break;
        case 14: COMBO_COMPACT_LAUNCH(14); break;
        case 15: COMBO_COMPACT_LAUNCH(15); break;
        default: break;
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

        bucketinit_kernel<<<grid, block, 0>>>(base_offset,
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

static uint32_t sieve_index_prev = 0xFFFFFFFF;

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
    if(cudaEventQuery(d_Events[thr_id][curr_sieve][EVENT::COMPACT]) == cudaErrorNotReady
    || cudaEventQuery(d_Events[thr_id][prev_test ][EVENT::FERMAT ]) == cudaErrorNotReady)
        return false;

    /* Create a sieve index that starts in a different lane for each frame */
    //sieve_index = sieve_index / FRAME_COUNT;
    //if(sieve_index >= index_range)
    //{
    //    printf("range exhausted!\n");
    //    return false;
    //}
    //sieve_index += (uint32_t)curr_sieve * index_range;

    /* Calculate bit array size, sieve start bit, and base offset */

    if(sieve_index_prev == sieve_index)
        debug::error(FUNCTION, "duplicate sieve index");



    uint64_t primorial_start = (uint64_t)nBitArray_Size * (uint64_t)sieve_index;
    uint64_t base_offsetted = base_offset + primorial * primorial_start;

    uint8_t nComboThreshold = 8;

    //CHECK(cudaDeviceSynchronize());


    /* Clear the current working sieve and signal */

    CHECK(stream_wait_event(thr_id, curr_sieve, 0, EVENT::COMPACT));
    CHECK(stream_wait_event(thr_id, prev_test, 0, EVENT::FERMAT));

    kernel_clear_launch(thr_id, 0, curr_sieve, nBitArray_Size);
    CHECK(stream_signal_event(thr_id, curr_sieve, 0, EVENT::CLEAR));

    /* Precompute offsets for small sieve */
    kernelA0_launch(thr_id, 1, curr_sieve,
    nPrimorialEndPrime, nPrimeLimitA, base_offsetted);

    CHECK(stream_wait_event(thr_id, curr_sieve, 1, EVENT::CLEAR));

    /* Launch small sieve, utilizing shared memory and signal */
    comboA_launch(thr_id, 1, curr_sieve,
                  nPrimorialEndPrime, nPrimeLimitA, nBitArray_Size);

    CHECK(stream_wait_event(thr_id, curr_sieve, 2, EVENT::CLEAR));

    /* Launch large sieve, utilizing global memory and signal */
    comboB_launch(thr_id, 2, base_offsetted, curr_sieve,
                  nPrimeLimitA, nPrimeLimitB, nBitArray_Size);


    CHECK(streams_signal_events(thr_id, curr_sieve, 1, 2));

    /* Wait Stream 2, Events[0, 1] */
    CHECK(stream_wait_events(thr_id, curr_sieve, 3, EVENT::SIEVE_A, EVENT::SIEVE_B));

    /* Launch compaction and signal */
    kernel_ccompact_launch(thr_id, 3, curr_sieve, curr_test, next_test, nBitArray_Size, primorial_start, nComboThreshold);

    CHECK(stream_signal_event(thr_id, curr_sieve, 3, EVENT::COMPACT));

    debug::log(4, FUNCTION, (uint32_t)thr_id);

    sieve_index_prev = sieve_index;

    return true;
}

extern "C" void cuda_wait_sieve(uint8_t thr_id, uint32_t sieve_index)
{
    uint32_t curr_sieve = sieve_index % FRAME_COUNT;
    CHECK(cudaEventSynchronize(d_Events[thr_id][curr_sieve][EVENT::COMPACT]));
}
