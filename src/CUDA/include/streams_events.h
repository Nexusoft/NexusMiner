/*******************************************************************************************

 Nexus Earth 2018

 [Scale Indefinitely] BlackJack. http://www.opensource.org/licenses/mit-license.php

*******************************************************************************************/
#ifndef NEXUS_CUDA_STREAMS_H
#define NEXUS_CUDA_STREAMS_H

#include <CUDA/include/util.h>
#include <CUDA/include/frame_resources.h>

#include <cstdint>

#define CUDA_STREAM_MAX 5
#define CUDA_EVENT_MAX 5

namespace EVENT
{
    enum
    {
        CLEAR = 0,
        SIEVE_A,
        SIEVE_B,
        COMPACT,
        FERMAT,
        MAX
    };
}

namespace STREAM
{
    enum
    {
        CLEAR = 0,
        SIEVE_A,
        SIEVE_B,
        COMPACT,
        FERMAT,
        MAX
    };
}



extern cudaStream_t d_Streams[GPU_MAX][CUDA_STREAM_MAX];
extern cudaEvent_t d_Events[GPU_MAX][FRAME_COUNT][CUDA_EVENT_MAX];

void streams_events_init(uint8_t tid);

void streams_events_free(uint8_t tid);


cudaError_t stream_wait_events(uint8_t tid, uint8_t frame_index, uint8_t sid, uint8_t eid_b, uint8_t eid_e);

cudaError_t streams_wait_event(uint8_t tid, uint8_t frame_index, uint8_t sid_b, uint8_t sid_e, uint8_t eid);

cudaError_t streams_signal_events(uint8_t tid, uint8_t frame_index, uint8_t sid_b, uint8_t sid_e);

cudaError_t stream_wait_event(uint8_t tid, uint8_t frame_index, uint8_t sid, uint8_t eid);

cudaError_t stream_signal_event(uint8_t tid, uint8_t frame_index, uint8_t sid, uint8_t eid);

cudaError_t synchronize_event(uint8_t tid, uint8_t frame_index, uint8_t eid);


#endif
