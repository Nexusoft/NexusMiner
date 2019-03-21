#ifndef NEXUS_CUDA_UTIL_H
#define NEXUS_CUDA_UTIL_H

#ifndef GPU_MAX
#define GPU_MAX 8
#endif

#include <Util/include/debug.h>
#include <cstdint>
#include <string>

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if(error != cudaSuccess)                                                   \
    {                                                                          \
        debug::error(__FILE__, __LINE__);                                      \
        debug::error("code: ", error, " reason: ", cudaGetErrorString(error)); \
        exit(1);                                                               \
    }                                                                          \
}

extern int device_map[GPU_MAX];

extern "C" void cuda_driver_version(int &major, int &minor);

extern "C" uint32_t cuda_device_multiprocessors(uint8_t index);

extern "C" int cuda_num_devices();

extern "C" std::string cuda_devicename(uint8_t index);

extern "C" void cuda_init(uint8_t thr_id);

extern "C" void cuda_free(uint8_t thr_id);

extern "C" void cuda_reset_device();

extern "C" void cuda_device_synchronize();

extern "C" void cuda_shutdown();

#endif
