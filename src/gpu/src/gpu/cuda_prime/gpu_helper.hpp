//GPU_CUDA_ENABLED and GPU_AMD_ENABLED are defined by cmake during configuration. 
//this file uses preprocessor macros as a switch to facilitate code reuse between hip (AMD) and Cuda
//This is needed for windows because at this time AMD does not have hip support on windows although they say it is planned. 
#ifndef NEXUSMINER_GPU_GPU_HELPER
#define NEXUSMINER_GPU_GPU_HELPER

#if defined(GPU_CUDA_ENABLED)
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#ifndef checkGPUErrors
#define checkGPUErrors(call)                                \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)
#endif
//here we make a macro only for the subset of the gpu functions that we use often.  
//Other functions get switched in directly in the file    
#define NEXUSMINER_GPU_Free cudaFree
#define NEXUSMINER_GPU_SetDevice cudaSetDevice
#define NEXUSMINER_GPU_Memset cudaMemset
#define NEXUSMINER_GPU_Memcpy cudaMemcpy
#define NEXUSMINER_GPU_MemcpyAsync cudaMemcpyAsync
#define NEXUSMINER_GPU_Malloc cudaMalloc
#define NEXUSMINER_GPU_DeviceSynchronize cudaDeviceSynchronize
#define NEXUSMINER_GPU_PeekAtLastError cudaPeekAtLastError
#define NEXUSMINER_GPU_MemGetInfo cudaMemGetInfo
#define NEXUSMINER_GPU_FuncSetAttribute cudaFuncSetAttribute
#define NEXUSMINER_GPU_FuncAttributeMaxDynamicSharedMemorySize cudaFuncAttributeMaxDynamicSharedMemorySize

#define NEXUSMINER_GPU_MemcpyHostToDevice cudaMemcpyHostToDevice
#define NEXUSMINER_GPU_MemcpyDeviceToHost cudaMemcpyDeviceToHost
#define NEXUSMINER_GPU_MemcpyDeviceToDevice cudaMemcpyDeviceToDevice

#elif defined(GPU_AMD_ENABLED)
#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#ifndef checkGPUErrors
#define checkGPUErrors(call)                                \
  do {                                                        \
    hipError_t err = call;                                   \
    if (err != hipSuccess) {                                 \
      printf("HIP GPU error at %s %d: %s\n", __FILE__, __LINE__, \
             hipGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)
#endif
#define NEXUSMINER_GPU_Free hipFree
#define NEXUSMINER_GPU_SetDevice hipSetDevice
#define NEXUSMINER_GPU_Memset hipMemset
#define NEXUSMINER_GPU_Memcpy hipMemcpy
#define NEXUSMINER_GPU_MemcpyAsync hipMemcpyAsync
#define NEXUSMINER_GPU_Malloc hipMalloc
#define NEXUSMINER_GPU_DeviceSynchronize hipDeviceSynchronize
#define NEXUSMINER_GPU_PeekAtLastError hipPeekAtLastError
#define NEXUSMINER_GPU_MemGetInfo hipMemGetInfo
#define NEXUSMINER_GPU_FuncSetAttribute hipFuncSetAttribute
#define NEXUSMINER_GPU_FuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize

#define NEXUSMINER_GPU_MemcpyHostToDevice hipMemcpyHostToDevice
#define NEXUSMINER_GPU_MemcpyDeviceToHost hipMemcpyDeviceToHost
#define NEXUSMINER_GPU_MemcpyDeviceToDevice hipMemcpyDeviceToDevice
#endif

#endif // !NEXUSMINER_GPU_GPU_HELPER

