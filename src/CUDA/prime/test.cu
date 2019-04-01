/*******************************************************************************************

 Nexus Earth 2018

 [Scale Indefinitely] BlackJack. http://www.opensource.org/licenses/mit-license.php

*******************************************************************************************/
#include <CUDA/include/test.h>
#include <CUDA/include/fermat.cuh>
#include <CUDA/include/util.h>
#include <CUDA/include/frame_resources.h>

#include <CUDA/include/streams_events.h>

#include <CUDA/include/constants.h>

#include <Util/include/debug.h>
#include <Util/include/prime_config.h>

#include <stdio.h>
#include <algorithm>


cudaError_t d_result_event_curr[GPU_MAX][FRAME_COUNT];
cudaError_t d_result_event_prev[GPU_MAX][FRAME_COUNT];

extern "C" void cuda_set_primorial(uint8_t thr_id, uint64_t nPrimorial)
{
    CHECK(cudaMemcpyToSymbol(c_primorial, &nPrimorial,
        sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
}

extern "C" void cuda_set_FirstSieveElement(uint32_t thr_id, uint32_t *limbs)
{
    debug::log(4, FUNCTION, thr_id);

    CHECK(cudaMemcpyToSymbol(c_zFirstSieveElement, limbs,
        WORD_MAX*sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
}


extern "C" void cuda_set_quit(uint32_t quit)
{
    debug::log(4, FUNCTION, quit ? "true" : "false");

    CHECK(cudaMemcpyToSymbol(c_quit, &quit,
                            sizeof(uint32_t), 0, cudaMemcpyHostToDevice));

}

__global__ void compact_test_offsets(uint64_t *in_nonce_offsets,
                                     uint32_t *in_nonce_meta,
                                     uint32_t *in_nonce_count,
                                     uint64_t *g_result_offsets,
                                     uint32_t *g_result_meta,
                                     uint32_t *g_result_count,
                                     uint32_t nThreshold,
                                     uint32_t nOffsets)
{
    /* Compute the global index for this nonce offset. */
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < *in_nonce_count)
    {
        uint32_t nonce_meta = in_nonce_meta[idx];

        /* Since extra 0-bits will invert to 1, mask them off the end.  */
        uint32_t count = __popc((~nonce_meta) & (0xFFFFFFFF >> (32 - nOffsets)));


        /* If the count meets the threshold, add to result buffer. */
        if(count >= nThreshold)
        {
            add_result(g_result_offsets, g_result_meta, g_result_count,
                       in_nonce_offsets[idx], nonce_meta, OFFSETS_MAX);
        }
    }
}

/* Fermat Test and sort offsets into resulting or working buffers. */
__global__ void fermat_kernel(uint64_t *nonce_offsets,
                              uint32_t *nonce_meta,
                              uint32_t *nonce_count,
                              uint32_t *g_primes_checked,
                              uint32_t *g_primes_found,
                              uint32_t nTestOffsets,
                              uint32_t nOffsets)
{

    /* Compute the global index for this nonce offset. */
    uint32_t position = blockIdx.x * blockDim.x + threadIdx.x;

    uint32_t idx = position >> 3;
    uint32_t o = position & 7;

    /* If the quit flag was set, early return to avoid wasting time. */
    if(c_quit)
        return;

    /* Make sure index is not out of bounds. */
    if(idx < *nonce_count && o < nTestOffsets)
    {
        uint32_t p[WORD_MAX];
        uint32_t test_index = c_iT[o];

        /* Compute the primorial offset from the primorial and
         * offset pattern (i.e 510510*n + [0,4,6,10] ) */
        uint64_t primorial_offset = c_primorial * nonce_offsets[idx] + (uint64_t)c_offsets[test_index];

        /* Add to the first sieving element to compute prime to test. */
        add_ui(p, c_zFirstSieveElement, primorial_offset);

        /* Check if prime passes fermat test base 2. */
        uint8_t prime = fermat_prime(p);

        /* Increment primes found. */
        atomicAdd(&g_primes_found[test_index], prime);

        /* Update the nonce combo. */
        atomicOr(&nonce_meta[idx], (!prime) << test_index);

        /* Increment primes checked. */
        atomicAdd(&g_primes_checked[test_index], 1);

    }
}


extern "C" __host__ void cuda_fermat(uint32_t thr_id,
                                     uint32_t sieve_index,
                                     uint32_t test_index,
                                     uint32_t nTestLevels)
{
    uint32_t curr_sieve = sieve_index % FRAME_COUNT;
    uint32_t curr_test = test_index % FRAME_COUNT;


    debug::log(4, FUNCTION, thr_id);

    /* Set the result event switch. */
    d_result_event_curr[thr_id][curr_test] = cudaErrorNotReady;
    d_result_event_prev[thr_id][curr_test] = cudaErrorNotReady;


    /*Make sure compaction event is finished before testing. */
    CHECK(stream_wait_event(thr_id, curr_sieve, STREAM::FERMAT, EVENT::COMPACT));


    /* Reset host-side counts to zero. */
    *frameResources[thr_id].h_result_count[curr_test] = 0;
    for(uint8_t i = 0; i < 16; ++i)
    {
        frameResources[thr_id].h_primes_checked[curr_test][i] = 0;
        frameResources[thr_id].h_primes_found[curr_test][i] = 0;
    }

    /* Copy zeroed-out primes checked. */
    CHECK(cudaMemcpyAsync(frameResources[thr_id].d_primes_checked[curr_test],
                          frameResources[thr_id].h_primes_checked[curr_test],
                          16 * sizeof(uint32_t), cudaMemcpyHostToDevice, d_Streams[thr_id][STREAM::FERMAT]));

    /* Copy zeroed-out primes found. */
    CHECK(cudaMemcpyAsync(frameResources[thr_id].d_primes_found[curr_test],
                          frameResources[thr_id].h_primes_found[curr_test],
                          16 * sizeof(uint32_t), cudaMemcpyHostToDevice, d_Streams[thr_id][STREAM::FERMAT]));

    /* Copy zeroed-out result count. */
    CHECK(cudaMemcpyAsync(frameResources[thr_id].d_result_count[curr_test],
                          frameResources[thr_id].h_result_count[curr_test],
                          sizeof(uint32_t), cudaMemcpyHostToDevice, d_Streams[thr_id][STREAM::FERMAT]));


    uint32_t nThreads = *frameResources[thr_id].h_nonce_count[curr_test];

    /* Make sure there are candidates. */
    if(nThreads == 0)
        return;

    if(nThreads >= OFFSETS_MAX)
        debug::error(FUNCTION, "WARNING: OFFSETS_MAX limit reached.");


    debug::log(3, FUNCTION, (uint32_t)thr_id,  ": nonce_count = ", nThreads);

    /* Loop unroll up to 8 offsets for testing. */
    dim3 block(32 << 3);
    dim3 grid((nThreads + block.x - 1) / (block.x >> 3));

    /* Launcth the fermat testing kernel. */
    fermat_kernel<<<grid, block, 0, d_Streams[thr_id][STREAM::FERMAT]>>>(
        frameResources[thr_id].d_nonce_offsets[curr_test],
        frameResources[thr_id].d_nonce_meta[curr_test],
        frameResources[thr_id].d_nonce_count[curr_test],
        frameResources[thr_id].d_primes_checked[curr_test],
        frameResources[thr_id].d_primes_found[curr_test],
        vOffsetsT.size(),
        vOffsets.size());

    dim3 block2(128);
    dim3 grid2((nThreads + block2.x - 1) / block2.x);

    /* Compact results down into result buffer. */
    compact_test_offsets<<<grid2, block2, 0, d_Streams[thr_id][STREAM::FERMAT]>>>(
        frameResources[thr_id].d_nonce_offsets[curr_test],
        frameResources[thr_id].d_nonce_meta[curr_test],
        frameResources[thr_id].d_nonce_count[curr_test],
        frameResources[thr_id].d_result_offsets[curr_test],
        frameResources[thr_id].d_result_meta[curr_test],
        frameResources[thr_id].d_result_count[curr_test],
        nTestLevels,
        vOffsets.size());

    /* Copy the result count. */
    CHECK(cudaMemcpyAsync(frameResources[thr_id].h_result_count[curr_test],
                          frameResources[thr_id].d_result_count[curr_test],
                          sizeof(uint32_t), cudaMemcpyDeviceToHost, d_Streams[thr_id][STREAM::FERMAT]));

    /* Copy the result offsets. */
    CHECK(cudaMemcpyAsync(frameResources[thr_id].h_result_offsets[curr_test],
                          frameResources[thr_id].d_result_offsets[curr_test],
                          OFFSETS_MAX * sizeof(uint64_t), cudaMemcpyDeviceToHost, d_Streams[thr_id][STREAM::FERMAT]));

    /* copy the result meta. */
    CHECK(cudaMemcpyAsync(frameResources[thr_id].h_result_meta[curr_test],
                          frameResources[thr_id].d_result_meta[curr_test],
                          OFFSETS_MAX * sizeof(uint32_t), cudaMemcpyDeviceToHost, d_Streams[thr_id][STREAM::FERMAT]));

    /* Copy the amount of primes checked. */
    CHECK(cudaMemcpyAsync(frameResources[thr_id].h_primes_checked[curr_test],
                          frameResources[thr_id].d_primes_checked[curr_test],
                          16 * sizeof(uint32_t), cudaMemcpyDeviceToHost, d_Streams[thr_id][STREAM::FERMAT]));

    /* Copy the amount of primes found. */
    CHECK(cudaMemcpyAsync(frameResources[thr_id].h_primes_found[curr_test],
                          frameResources[thr_id].d_primes_found[curr_test],
                          16 * sizeof(uint32_t), cudaMemcpyDeviceToHost, d_Streams[thr_id][STREAM::FERMAT]));

    /* Signal the Fermat event. */
    CHECK(stream_signal_event(thr_id, curr_test, STREAM::FERMAT, EVENT::FERMAT));
}


extern "C" void cuda_results(uint32_t thr_id,
                             uint32_t test_index,
                             uint64_t *result_offsets,
                             uint32_t *result_meta,
                             uint32_t *result_count,
                             uint32_t *primes_checked,
                             uint32_t *primes_found)
{
    /* Clear the stats. */
    *result_count = 0;
    for(uint16_t i = 0; i < 16; ++i)
    {
        primes_checked[i] = 0;
        primes_found[i] = 0;
    }

    uint32_t curr_test = test_index % FRAME_COUNT;


    d_result_event_prev[thr_id][curr_test] = d_result_event_curr[thr_id][curr_test];
    d_result_event_curr[thr_id][curr_test] = cudaEventQuery(d_Events[thr_id][curr_test][EVENT::FERMAT]);

    if(d_result_event_curr[thr_id][curr_test] == cudaSuccess &&
       d_result_event_prev[thr_id][curr_test] == cudaErrorNotReady)
    {
        /* Reset event to trigger one way switch. */
        d_result_event_prev[thr_id][curr_test] = cudaSuccess;

        *result_count   = *frameResources[thr_id].h_result_count[curr_test];


        /* Update the primes checked/found for each offset from GPU. */
        for(uint32_t i = 0; i < 16; ++i)
        {
            primes_checked[i] =  frameResources[thr_id].h_primes_checked[curr_test][i];
            primes_found[i] =  frameResources[thr_id].h_primes_found[curr_test][i];
        }

        if(*result_count == 0)
            return;

        uint64_t *pOffsets = frameResources[thr_id].h_result_offsets[curr_test];
        uint32_t *pMeta  =   frameResources[thr_id].h_result_meta[curr_test];

        std::copy(pOffsets, pOffsets + (*result_count), result_offsets);
        std::copy(pMeta,    pMeta    + (*result_count), result_meta);

        debug::log(4, FUNCTION, thr_id, "    ", *result_count, " results");
    }
}

extern "C" void cuda_init_counts(uint32_t thr_id)
{
    uint32_t zero[BUFFER_COUNT] = {0};

    debug::log(4, FUNCTION, thr_id);

    CHECK(cudaDeviceSynchronize());

    for(int i = 0; i < FRAME_COUNT; ++i)
    {
        *frameResources[thr_id].h_nonce_count[i] = 0;

        CHECK(cudaMemcpy(frameResources[thr_id].d_nonce_count[i],
                         zero,
                         sizeof(uint32_t) * BUFFER_COUNT,
                         cudaMemcpyHostToDevice));
    }
}
