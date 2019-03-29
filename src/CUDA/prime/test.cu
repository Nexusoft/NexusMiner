/*******************************************************************************************

 Nexus Earth 2018

 [Scale Indefinitely] BlackJack. http://www.opensource.org/licenses/mit-license.php

*******************************************************************************************/
#include <CUDA/include/test.h>
#include <CUDA/include/fermat.cuh>
#include <CUDA/include/util.h>
#include <CUDA/include/frame_resources.h>

#include <CUDA/include/streams_events.h>

#include <CUDA/include/constants.cuh>

#include <Util/include/debug.h>

#include <stdio.h>
#include <algorithm>


cudaError_t d_result_event_curr[GPU_MAX][FRAME_COUNT];
cudaError_t d_result_event_prev[GPU_MAX][FRAME_COUNT];


uint8_t nOffsetsT;

extern "C" void cuda_set_primorial(uint8_t thr_id, uint64_t nPrimorial)
{
    CHECK(cudaMemcpyToSymbol(c_primorial, &nPrimorial,
        sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
}

extern "C" void cuda_set_test_offsets(uint32_t thr_id,
                                       uint32_t *OffsetsT, uint32_t T_count)
{
    nOffsetsT = T_count;

    debug::log(4, FUNCTION, thr_id, "    ", nOffsetsT);

    if(nOffsetsT > 16)
        debug::error(FUNCTION, "test offsets cannot exceed 16");

    CHECK(cudaMemcpyToSymbol(c_offsetsT, OffsetsT,
        nOffsetsT*sizeof(uint32_t), 0, cudaMemcpyHostToDevice));

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

/* Fermat Test and sort offsets into resulting or working buffers. */
__global__ void fermat_kernel(uint64_t *in_nonce_offsets,
                              uint64_t *in_nonce_meta,
                              uint32_t in_nonce_count,
                              uint64_t *out_nonce_offsets,
                              uint64_t *out_nonce_meta,
                              uint32_t *out_nonce_count,
                              uint64_t *g_result_offsets,
                              uint64_t *g_result_meta,
                              uint32_t *g_result_count,
                              uint32_t *g_primes_checked,
                              uint32_t *g_primes_found,
                              uint8_t nTestOffsets,
                              uint8_t nTestLevels,
                              uint8_t o)
{

    /* Compute the global index for this nonce offset. */
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    /* If the quit flag was set, early return to avoid wasting time. */
    if(c_quit)
        return;

    /* Make sure index is not out of bounds. */
    if(idx < in_nonce_count)
    {
        /* Get the nonce offset and meta data. */
        uint64_t nonce_offset = in_nonce_offsets[idx];
        uint64_t nonce_meta =   in_nonce_meta[idx];
        uint32_t p[WORD_MAX];


        /* Decode the nonce meta data. */
        uint32_t combo           =  nonce_meta >> 32;
        uint8_t chain_offset_beg = (nonce_meta >> 24) & 0xFF;
        uint8_t chain_offset_end = (nonce_meta >> 16) & 0xFF;
        uint8_t prime_gap =        (nonce_meta >> 8 ) & 0xFF;
        uint8_t chain_length =      nonce_meta & 0xFF;

        /* Compute the primorial offset from the primorial and
         * offset pattern (i.e 510510*n + [0,4,6,10] ) */
        uint64_t primorial_offset = c_primorial * nonce_offset;
        primorial_offset += c_offsetsT[chain_offset_end];

        /* Add to the first sieving element to compute prime to test. */
        add_ui(p, c_zFirstSieveElement, primorial_offset);

        /* Check if prime passes fermat test base 2. */
        if(fermat_prime(p))
        {
            atomicAdd(g_primes_found, 1);
            ++chain_length;
            prime_gap = 0;

            /* If the chain length is satisfied, add it to result buffer. */
            if(chain_length == nTestLevels)
            {
                /* Encode the nonce meta data. */
                nonce_meta = 0;
                nonce_meta |= ((uint64_t)combo << 32);
                nonce_meta |= ((uint64_t)chain_offset_beg << 24);
                nonce_meta |= ((uint64_t)chain_offset_end << 16);
                nonce_meta |= ((uint64_t)prime_gap << 8);
                nonce_meta |= (uint64_t)chain_length;

                /* Add to result buffer. */
                add_result(g_result_offsets, g_result_meta, g_result_count,
                           nonce_offset, nonce_meta, OFFSETS_MAX);

            }
        }

        /* Otherwise, if chain length is not satisfied, keep testing. */
        if(chain_length < nTestLevels)
        {
            /* Make sure there are offsets to test in the combo bit mask. */
            if(combo)
            {
                /* Get the next offset index and clear it from the combo bits. */
                uint8_t chain_offset_next = __clz(combo);
                combo ^= 0x80000000 >> chain_offset_next;

                /* If the chain length is zero, shift to the next offset and start over. */
                if(chain_length == 0)
                {
                    chain_offset_beg = chain_offset_next;
                    chain_offset_end = chain_offset_next;
                    prime_gap = 0;
                }

                /* Make sure next offset and beginning are within bounds of testing
                 * for the next round. */
                if(chain_offset_next < nTestOffsets
                && chain_offset_beg <= nTestOffsets - nTestLevels)
                {
                    /* Calculate prime gap to next offset from last prime. */
                    prime_gap += c_offsetsT[chain_offset_next] - c_offsetsT[chain_offset_end];
                    chain_offset_end = chain_offset_next;

                    if(prime_gap <= 12)
                    {
                        /* Encode the nonce meta data. */
                        nonce_meta = 0;
                        nonce_meta |= ((uint64_t)combo << 32);
                        nonce_meta |= ((uint64_t)chain_offset_beg << 24);
                        nonce_meta |= ((uint64_t)chain_offset_end << 16);
                        nonce_meta |= ((uint64_t)prime_gap << 8);
                        nonce_meta |= (uint64_t)chain_length;

                        add_result(out_nonce_offsets, out_nonce_meta, out_nonce_count,
                                   nonce_offset, nonce_meta, OFFSETS_MAX);

                    }
                }
            }
        }
        atomicAdd(g_primes_checked, 1);
    }
}

__global__ void fermat_launcher(uint64_t *g_nonce_offsets,
                                uint64_t *g_nonce_meta,
                                uint32_t *g_nonce_count,
                                uint64_t *g_result_offsets,
                                uint64_t *g_result_meta,
                                uint32_t *g_result_count,
                                uint32_t *g_primes_checked,
                                uint32_t *g_primes_found,
                                uint8_t nTestOffsets,
                                uint8_t nTestLevels,
                                uint8_t o)
{
    uint8_t buffer_index = o & 1;

    uint64_t *in_nonce_offsets = g_nonce_offsets + buffer_index * OFFSETS_MAX;
    uint64_t *in_nonce_meta    = g_nonce_meta    + buffer_index * OFFSETS_MAX;
    uint32_t *in_nonce_count   = g_nonce_count   + buffer_index * 4;

    buffer_index ^= 1; //flip between two working buffers

    uint64_t *out_nonce_offsets = g_nonce_offsets + buffer_index * OFFSETS_MAX;
    uint64_t *out_nonce_meta    = g_nonce_meta    + buffer_index * OFFSETS_MAX;
    uint32_t *out_nonce_count   = g_nonce_count   + buffer_index * 4;

    uint32_t total_count = in_nonce_count[0];

    if(total_count >= OFFSETS_MAX && threadIdx.x == 0)
    {
        total_count = OFFSETS_MAX;
        printf("[WARNING] Candidates Max Reached. Use more Sieving Primes or Less Offsets.\n");
    }

    /* Clear the counts for each thread. */
    in_nonce_count[threadIdx.x] = 0;
    out_nonce_count[threadIdx.x] = 0;

    if(threadIdx.x == 0 && o == 0)
    {
        *g_result_count = 0;
        *g_primes_checked = 0;
        *g_primes_found = 0;
    }

    if(total_count > 0 && c_quit == false)
    {
        /* Split the workload into segments and seperate thread launches. */
        uint32_t segment_in_count = (total_count + 5) >> 2;
        uint32_t segment_offset = threadIdx.x * segment_in_count;
        int32_t diff = total_count - segment_offset;
        if(diff < 0)
            diff = 0;

        /* Compute the segment count for each thread launch. */
        segment_in_count = min(segment_in_count, diff);

        /* Launch child processes to test results */
        if(segment_in_count > 0)
        {
            /* Create a temporary stream for the kernel launch. */
            cudaStream_t stream;
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

            dim3 block(32);
            dim3 grid((segment_in_count+block.x-1)/block.x);

            /* Launch the fermat kernel for this segment. */
            fermat_kernel<<<grid, block, 0, stream>>>(
                &in_nonce_offsets[segment_offset],  &in_nonce_meta[segment_offset],  segment_in_count,
                &out_nonce_offsets[0], &out_nonce_meta[0],  &out_nonce_count[0],
                g_result_offsets, g_result_meta, g_result_count,
                g_primes_checked, g_primes_found, nTestOffsets, nTestLevels, threadIdx.x);

            /* Synchronize device across kernel launches. */
            cudaDeviceSynchronize();

            /* Destroy the temporary stream. */
            cudaStreamDestroy(stream);
        }
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


    for(uint8_t o = 0; o < 6; ++o)
    {
        fermat_launcher<<<1, 4, 0, d_Streams[thr_id][STREAM::FERMAT]>>>(
                 frameResources[thr_id].d_nonce_offsets[curr_test],
                 frameResources[thr_id].d_nonce_meta[curr_test],
                 frameResources[thr_id].d_nonce_count[curr_test],
                 frameResources[thr_id].d_result_offsets[curr_test],
                 frameResources[thr_id].d_result_meta[curr_test],
                 frameResources[thr_id].d_result_count[curr_test],
                 frameResources[thr_id].d_primes_checked[curr_test],
                 frameResources[thr_id].d_primes_found[curr_test],
                 nOffsetsT, nTestLevels, o);
    }

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
                          OFFSETS_MAX * sizeof(uint64_t), cudaMemcpyDeviceToHost, d_Streams[thr_id][STREAM::FERMAT]));

    /* Copy the amount of primes checked. */
    CHECK(cudaMemcpyAsync(frameResources[thr_id].h_primes_checked[curr_test],
                          frameResources[thr_id].d_primes_checked[curr_test],
                          sizeof(uint32_t), cudaMemcpyDeviceToHost, d_Streams[thr_id][STREAM::FERMAT]));

    /* Copy the amount of primes found. */
    CHECK(cudaMemcpyAsync(frameResources[thr_id].h_primes_found[curr_test],
                          frameResources[thr_id].d_primes_found[curr_test],
                          sizeof(uint32_t), cudaMemcpyDeviceToHost, d_Streams[thr_id][STREAM::FERMAT]));

    /* Signal the Fermat event. */
    CHECK(stream_signal_event(thr_id, curr_test, STREAM::FERMAT, EVENT::FERMAT));

}

extern "C" void cuda_results(uint32_t thr_id,
                             uint32_t test_index,
                             uint64_t *result_offsets,
                             uint64_t *result_meta,
                             uint32_t *result_count,
                             uint32_t *primes_checked,
                             uint32_t *primes_found)
{
    *result_count = 0;
    *primes_checked = 0;
    *primes_found = 0;

    uint32_t curr_test = test_index % FRAME_COUNT;

    d_result_event_prev[thr_id][curr_test] = d_result_event_curr[thr_id][curr_test];
    d_result_event_curr[thr_id][curr_test] = cudaEventQuery(d_Events[thr_id][curr_test][EVENT::FERMAT]);

    if(d_result_event_curr[thr_id][curr_test] == cudaSuccess &&
       d_result_event_prev[thr_id][curr_test] == cudaErrorNotReady)
    {
        d_result_event_prev[thr_id][curr_test] = cudaSuccess;

        *result_count   = *frameResources[thr_id].h_result_count[curr_test];
        *primes_checked = *frameResources[thr_id].h_primes_checked[curr_test];
        *primes_found   = *frameResources[thr_id].h_primes_found[curr_test];

        *frameResources[thr_id].h_result_count[curr_test] = 0;
        *frameResources[thr_id].h_primes_checked[curr_test] = 0;
        *frameResources[thr_id].h_primes_found[curr_test] = 0;

        if(*result_count == 0)
            return;

        uint64_t *pOffsets = frameResources[thr_id].h_result_offsets[curr_test];
        uint64_t *pMeta  =   frameResources[thr_id].h_result_meta[curr_test];

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
