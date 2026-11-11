#ifndef NEXUSMINER_GPU_CUDA_MEMORY_HPP
#define NEXUSMINER_GPU_CUDA_MEMORY_HPP

#include <memory>
#include <cuda_runtime.h>

namespace nexusminer {
namespace gpu {

/**
 * @brief Custom deleter for CUDA device memory allocated with cudaMalloc
 * Enhanced with error handling and logging for debugging segfaults
 */
struct CudaDeleter {
    void operator()(void* ptr) const;
};

/**
 * @brief Custom deleter for CUDA pinned host memory allocated with cudaMallocHost
 * Enhanced with error handling and logging for debugging segfaults
 */
struct CudaHostDeleter {
    void operator()(void* ptr) const;
};

/**
 * @brief Type alias for unique_ptr with CUDA device memory deleter
 * 
 * Usage:
 *   cuda_unique_ptr<uint32_t> ptr;
 *   uint32_t* raw_ptr = nullptr;
 *   checkGPUErrors(cudaMalloc(&raw_ptr, size * sizeof(uint32_t)));
 *   ptr.reset(raw_ptr);
 */
template<typename T>
using cuda_unique_ptr = std::unique_ptr<T, CudaDeleter>;

/**
 * @brief Type alias for unique_ptr with CUDA pinned host memory deleter
 * 
 * Usage:
 *   cuda_host_unique_ptr<uint64_t> ptr;
 *   uint64_t* raw_ptr = nullptr;
 *   checkGPUErrors(cudaMallocHost(&raw_ptr, size * sizeof(uint64_t)));
 *   ptr.reset(raw_ptr);
 */
template<typename T>
using cuda_host_unique_ptr = std::unique_ptr<T, CudaHostDeleter>;

/**
 * @brief Helper function to allocate CUDA device memory with RAII wrapper
 * 
 * @param count Number of elements to allocate
 * @return cuda_unique_ptr managing the allocated memory
 * @throws cudaError_t on allocation failure
 */
template<typename T>
cuda_unique_ptr<T> make_cuda_unique(size_t count) {
    T* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, count * sizeof(T));
    if (err != cudaSuccess) {
        throw err;
    }
    return cuda_unique_ptr<T>(ptr);
}

/**
 * @brief Helper function to allocate CUDA pinned host memory with RAII wrapper
 * 
 * @param count Number of elements to allocate
 * @return cuda_host_unique_ptr managing the allocated memory
 * @throws cudaError_t on allocation failure
 */
template<typename T>
cuda_host_unique_ptr<T> make_cuda_host_unique(size_t count) {
    T* ptr = nullptr;
    cudaError_t err = cudaMallocHost(&ptr, count * sizeof(T));
    if (err != cudaSuccess) {
        throw err;
    }
    return cuda_host_unique_ptr<T>(ptr);
}

} // namespace gpu
} // namespace nexusminer

#endif // NEXUSMINER_GPU_CUDA_MEMORY_HPP
