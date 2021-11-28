#ifndef NEXUSMINER_GPU_CUDA_FIND_CHAIN_CUH
#define NEXUSMINER_GPU_CUDA_FIND_CHAIN_CUH

#include "cuda_chain.cuh"
#include <stdint.h>
namespace nexusminer {
	namespace gpu {


		__global__ void find_chain_kernel(Cuda_sieve::sieve_word_t* sieve, CudaChain* chains, uint32_t* chain_index, uint64_t sieve_start_offset,
			unsigned long long* chain_stat_count);

		__global__ void find_chain_kernel2(Cuda_sieve::sieve_word_t* sieve, CudaChain* chains, uint32_t* chain_index, uint64_t sieve_start_offset,
			unsigned long long* chain_stat_count);

		__global__ void filter_busted_chains(CudaChain* chains, uint32_t* chain_index, CudaChain* surviving_chains,
			uint32_t* surviving_chain_index, CudaChain* long_chains, uint32_t* long_chain_index, uint32_t* histogram);
	}
}

#endif