#include <gpu/cuda_memory.hpp>

#include <spdlog/spdlog.h>

#include <cstdio>
#include <exception>

namespace nexusminer::gpu {

void CudaDeleter::operator()(void* ptr) const
{
    if (!ptr) return;
    try {
        auto logger = spdlog::get("logger");
        if (logger) logger->debug("CudaDeleter: Freeing CUDA device memory at {}", ptr);
        const cudaError_t err = cudaFree(ptr);
        if (err != cudaSuccess) {
            if (logger) {
                logger->error("CudaDeleter: Failed to free CUDA memory at {}: {} ({})",
                              ptr, cudaGetErrorString(err), static_cast<int>(err));
            } else {
                std::fprintf(stderr, "CudaDeleter: Failed to free CUDA memory at %p: %s (%d)\n",
                             ptr, cudaGetErrorString(err), static_cast<int>(err));
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "CudaDeleter: Exception during CUDA memory cleanup: %s\n", e.what());
    } catch (...) {
        std::fprintf(stderr, "CudaDeleter: Unknown exception during CUDA memory cleanup\n");
    }
}

void CudaHostDeleter::operator()(void* ptr) const
{
    if (!ptr) return;
    try {
        auto logger = spdlog::get("logger");
        if (logger) logger->debug("CudaHostDeleter: Freeing CUDA pinned host memory at {}", ptr);
        const cudaError_t err = cudaFreeHost(ptr);
        if (err != cudaSuccess) {
            if (logger) {
                logger->error("CudaHostDeleter: Failed to free CUDA host memory at {}: {} ({})",
                              ptr, cudaGetErrorString(err), static_cast<int>(err));
            } else {
                std::fprintf(stderr, "CudaHostDeleter: Failed to free CUDA host memory at %p: %s (%d)\n",
                             ptr, cudaGetErrorString(err), static_cast<int>(err));
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "CudaHostDeleter: Exception during CUDA host memory cleanup: %s\n", e.what());
    } catch (...) {
        std::fprintf(stderr, "CudaHostDeleter: Unknown exception during CUDA host memory cleanup\n");
    }
}

} // namespace nexusminer::gpu
