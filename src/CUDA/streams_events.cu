#include <CUDA/include/streams_events.h>
#include <cuda.h>

cudaStream_t d_Streams[GPU_MAX][CUDA_STREAM_MAX];
cudaEvent_t d_Events[GPU_MAX][FRAME_COUNT][CUDA_EVENT_MAX];

/* Create the CUDA streams and events. */
void streams_events_init(uint8_t tid)
{

    for(uint8_t i = 0; i < CUDA_STREAM_MAX; ++i)
        CHECK(cudaStreamCreateWithFlags(&d_Streams[tid][i], cudaStreamNonBlocking));


    for(uint8_t i = 0; i < FRAME_COUNT; ++i)
    {
        for(uint8_t j = 0; j < CUDA_EVENT_MAX; ++j)
            CHECK(cudaEventCreateWithFlags(&d_Events[tid][i][j], cudaEventDisableTiming | cudaEventBlockingSync));
    }
}

void streams_events_free(uint8_t tid)
{
    for(uint8_t i = 0; i < CUDA_STREAM_MAX; ++i)
        CHECK(cudaStreamDestroy(d_Streams[tid][i]));

    for(uint8_t i = 0; i < FRAME_COUNT; ++i)
    {
        for(uint8_t j = 0; j < CUDA_EVENT_MAX; ++j)
            CHECK(cudaEventDestroy(d_Events[tid][i][j]));
    }
}


cudaError_t stream_wait_events(uint8_t tid, uint8_t frame_index, uint8_t sid, uint8_t eid_b, uint8_t eid_e)
{
    cudaError_t err;
    cudaError_t r = cudaSuccess;
    for(uint8_t e = eid_b; e <= eid_e; ++e)
    {
        err = cudaStreamWaitEvent(d_Streams[tid][sid], d_Events[tid][frame_index][e], 0);
        if(err != cudaSuccess)
        {
             r = err;
             break;
        }
    }
    return r;
}


cudaError_t streams_wait_event(uint8_t tid, uint8_t frame_index, uint8_t sid_b, uint8_t sid_e, uint8_t eid)
{
    cudaError_t e;
    cudaError_t r = cudaSuccess;
    for(uint8_t s = sid_b; s <= sid_e; ++s)
    {
        e = cudaStreamWaitEvent(d_Streams[tid][s], d_Events[tid][frame_index][eid], 0);
        if(e != cudaSuccess)
        {
             r = e;
             break;
        }
    }
    return r;
}


cudaError_t streams_signal_events(uint8_t tid, uint8_t frame_index, uint8_t sid_b, uint8_t sid_e)
{
    cudaError_t e;
    cudaError_t r = cudaSuccess;
    for(uint8_t s = sid_b; s <= sid_e; ++s)
    {
        e = cudaEventRecord(d_Events[tid][frame_index][s], d_Streams[tid][s]);
        if(e != cudaSuccess)
        {
             r = e;
             break;
        }
    }
    return r;
}


cudaError_t stream_wait_event(uint8_t tid, uint8_t frame_index, uint8_t sid, uint8_t eid)
{
    return cudaStreamWaitEvent(d_Streams[tid][sid], d_Events[tid][frame_index][eid],  0);
}


cudaError_t stream_signal_event(uint8_t tid, uint8_t frame_index, uint8_t sid, uint8_t eid)
{
    return cudaEventRecord(d_Events[tid][frame_index][eid], d_Streams[tid][sid]);
}


cudaError_t synchronize_event(uint8_t tid, uint8_t frame_index, uint8_t eid)
{
    return cudaEventSynchronize(d_Events[tid][frame_index][eid]);
}
