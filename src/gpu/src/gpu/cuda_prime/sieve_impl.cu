#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "sieve_impl.cuh"
#include "sieve.hpp"

#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <inttypes.h>


#define checkCudaErrors(call)                                \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)

namespace nexusminer {
    namespace gpu {

        __device__ void cuda_chain_push_back(CudaChain& chain, uint16_t offset);
        __device__ void cuda_chain_open(CudaChain& chain, uint64_t base_offset);

        __constant__ const int sieve30_offsets[]{ 1,7,11,13,17,19,23,29 };

        __constant__ const int sieve30_gaps[]{ 6,4,2,4,2,4,6,2 };

        __constant__ const int sieve30_index[]
        { -1,0,-1,-1,-1,-1,-1, 1, -1, -1, -1, 2, -1, 3, -1, -1, -1, 4, -1, 5, -1, -1, -1, 6, -1, -1, -1, -1, -1, 7 };  //reverse lookup table (offset mod 30 to index)

        __constant__ const uint8_t unset_bit_mask[] 
        {
            (uint8_t)~(1 << 0), (uint8_t)~(1 << 0),
            (uint8_t)~(1 << 1), (uint8_t)~(1 << 1), (uint8_t)~(1 << 1), (uint8_t)~(1 << 1), (uint8_t)~(1 << 1), (uint8_t)~(1 << 1),
            (uint8_t)~(1 << 2), (uint8_t)~(1 << 2), (uint8_t)~(1 << 2), (uint8_t)~(1 << 2),
            (uint8_t)~(1 << 3), (uint8_t)~(1 << 3),
            (uint8_t)~(1 << 4), (uint8_t)~(1 << 4), (uint8_t)~(1 << 4), (uint8_t)~(1 << 4),
            (uint8_t)~(1 << 5), (uint8_t)~(1 << 5),
            (uint8_t)~(1 << 6), (uint8_t)~(1 << 6), (uint8_t)~(1 << 6), (uint8_t)~(1 << 6),
            (uint8_t)~(1 << 7), (uint8_t)~(1 << 7), (uint8_t)~(1 << 7), (uint8_t)~(1 << 7), (uint8_t)~(1 << 7), (uint8_t)~(1 << 7)
        };


        // cross off small primes.  These primes hit the sieve often.  We iterate through the sieve words and cross them off using 
        // precalculated constants.  start is the sieve start offset mod 30
        __device__ void sieveSmallPrimes(Cuda_sieve::sieve_word_t* s_sieve, uint32_t sieveWords, uint64_t start)
        {
            //#pragma unroll 1
            for (uint32_t i = threadIdx.x; i < sieveWords; i += blockDim.x) 
            {
                //uint64_t j = i + start / 120; // 120 is 32 bits per uint32_t * 30/8 integers per bit
                //s_sieve[i] |= p7[j % 7];  
                //s_sieve[i] |= p11[j % 11]; 
                //s_sieve[i] |= p13[j % 13]; 
                //s_sieve[i] |= p17[j % 17]; 
                //s_sieve[i] |= p19[j % 19]; 
                //s_sieve[i] |= p23[j % 23];
                //s_sieve[i] |= p29[j % 29];
                //s_sieve[i] |= p31[j % 31];
                //s_sieve[i] |= p37[j % 37];
            }
        }
        
        //get the nth bit from the sieve.
        __device__ __forceinline__ bool get_bit(uint64_t bit_position, Cuda_sieve::sieve_word_t* sieve)
        {
            const uint32_t sieve_bits_per_word = Cuda_sieve::m_sieve_word_byte_count * 8;
            
            uint64_t word = bit_position / sieve_bits_per_word;
            unsigned bit_position_in_word = bit_position % sieve_bits_per_word;
            return ((sieve[word] >> bit_position_in_word) & 1) == 1;

        }

        //search the sieve for chains that meet the minimum length requirement.  
        __global__ void find_chain_kernel(Cuda_sieve::sieve_word_t* sieve, CudaChain* chains, uint32_t* chain_index, uint64_t sieve_start_offset)
        {
            
            //const uint64_t sieve_size = Cuda_sieve::m_sieve_total_size;
            const uint32_t sieve_bits_per_word = Cuda_sieve::m_sieve_word_byte_count * 8;
            const uint64_t sieve_total_bits = Cuda_sieve::m_sieve_total_size * sieve_bits_per_word;

            
            uint64_t num_blocks = gridDim.x;
            uint64_t num_threads = blockDim.x;
            uint64_t block_id = blockIdx.x;
            uint64_t index = block_id * num_threads + threadIdx.x;
            uint64_t stride = num_blocks * num_threads;
            int sieve_offset;
            int gap;
            uint64_t chain_start, prime_candidate_offset;
            
            
            if (index == 0)
                *chain_index = 0;
            __syncthreads();
           
            //search each sieve location for a possible chain
            for (uint64_t i = index; i < sieve_total_bits; i += stride)
            {
                if (i < sieve_total_bits)
                {
                    //chain must start with a prime
                    if (!get_bit(i, sieve))
                    {
                        return;
                    }
                    //search left for another prime less than max gap away
                    gap = 0;
                    int64_t j = i;
                    j--;
                    while (j >= 0 && gap <= maxGap)
                    {
                        gap += sieve30_gaps[j % 8];
                        if (gap <= maxGap && get_bit(j, sieve))
                        {
                            //there is a valid element to the left.  this is not the first element in a chain. abort.
                            return;
                        }
                        
                        j--;
                    }
                   
                    //this is the start of a possible chain.  search right
                    //where are we in the wheel
                    sieve_offset = sieve30_offsets[i % 8];
                    chain_start = sieve_start_offset + i / 8 * 30 + sieve_offset;
                    CudaChain current_chain;
                    cuda_chain_open(current_chain, chain_start);
                    gap = 0;
                    j = i;
                    j++;
                    while (j < sieve_total_bits && gap <= maxGap)
                    {
                        gap += sieve30_gaps[j % 8];
                        if (gap <= maxGap && get_bit(j, sieve))
                        {
                            //another possible candidate.  add it to the chain
                            gap = 0;
                            sieve_offset = sieve30_offsets[j % 8];
                            prime_candidate_offset = sieve_start_offset + j / 8 * 30 + sieve_offset;
                            cuda_chain_push_back(current_chain, static_cast<uint16_t>(prime_candidate_offset - chain_start));
                        }
                        j++;
                    }
                    //we reached the end of the chain.  check if it meets the length requirement
                    if (current_chain.m_offset_count >= Cuda_sieve::m_min_chain_length)
                    {
                        //increment the chain list index
                        uint32_t chain_idx = atomicInc(chain_index, Cuda_sieve::m_max_chains);
                        //copy the current chain to the global list
                        chains[chain_idx] = current_chain;
                    }
                }
            }
            

        }

       
        //return the offset from x to the next integer multiple of n greater than x that is not divisible by 2, 3, or 5.  
        //x must be a multiple of the primorial 30 and n must be a prime greater than 5.
        template <typename T1, typename T2>
        __device__ T2 get_offset_to_next_multiple(T1 x, T2 n)
        {
            T2 m = n - static_cast<T2>(x % n);
            if (m % 2 == 0)
            {
                m += n;
            }
            while (m % 3 == 0 || m % 5 == 0)
            {
                m += 2 * n;
            }
            return m;
        }

        //seive kernel
        __global__ void do_sieve(uint64_t sieve_start_offset, uint32_t* sieving_primes, uint32_t sieving_prime_count,
            uint32_t* starting_multiples, uint32_t* prime_mod_inverses, Cuda_sieve::sieve_word_t* sieve_results, uint32_t* multiples,
            uint8_t* wheel_indices, uint64_t* prime_candidate_count)
        {
            const uint32_t segment_size = Cuda_sieve::m_kernel_sieve_size_bytes * Cuda_sieve::m_sieve_byte_range;

            //local shared copy of the sieve
            __shared__ Cuda_sieve::sieve_word_t sieve[Cuda_sieve::m_kernel_sieve_size_words];

            uint64_t block_id = blockIdx.x;
            uint64_t index = threadIdx.x;
            uint64_t stride = blockDim.x;
            //uint64_t num_blocks = gridDim.x;
           
            const uint64_t segments = Cuda_sieve::m_kernel_segments_per_block;
            uint64_t sieve_results_index = block_id * Cuda_sieve::m_kernel_sieve_size_words_per_block;
            unsigned long long count = 0;

            //each block sieves a different region
            uint64_t start_offset = sieve_start_offset + block_id * Cuda_sieve::m_kernel_sieve_size_words_per_block * Cuda_sieve::m_sieve_word_range;
            
            int wheel_index;
            int next_wheel_gap;
            uint64_t j;
            uint64_t k;
            for (int s = 0; s < segments; s++)
            {
                //everyone in the block initialize part of the shared sieve
                for (int j1 = index; j1 < Cuda_sieve::m_kernel_sieve_size_words; j1 += stride)
                {
                    if (j1 < Cuda_sieve::m_kernel_sieve_size_words)
                        sieve[j1] = ~0;
                }

                __syncthreads();
                for (uint32_t i = index; i < sieving_prime_count; i += stride)
                {
                    if (i < sieving_prime_count)
                    {
                        
                        k = sieving_primes[i];
                        //get aligned to this region
                        if (s == 0)
                        {
                            j = starting_multiples[i];
                            //the first time through we need to calculate the starting offsets
                            if (start_offset >= j)
                                j = get_offset_to_next_multiple(start_offset - j, sieving_primes[i]);
                            else
                                j -= start_offset;
                            
                            //where are we in the wheel
                            wheel_index = sieve30_index[(prime_mod_inverses[i] * j) % 30];
                        }
                        else
                        {
                            j = multiples[block_id* sieving_prime_count +i];
                            wheel_index = wheel_indices[block_id * sieving_prime_count + i];
                        }
                        next_wheel_gap = sieve30_gaps[wheel_index];
                        
                        while (j < segment_size)
                        {
                            //cross off a multiple of the sieving prime
                            uint64_t sieve_index = j / Cuda_sieve::m_sieve_word_range;
                            Cuda_sieve::sieve_word_t bitmask = ~(static_cast<Cuda_sieve::sieve_word_t>(1) <<
                                (sieve30_index[j % 30] + (8 * (j/Cuda_sieve::m_sieve_byte_range % Cuda_sieve::m_sieve_word_byte_count))));

                            //printf("%" PRIu64 " %u\n", j, bitmask);
                            
                            atomicAnd(&sieve[sieve_index], bitmask);
                            //sieve[sieve_index] &= ~static_cast<unsigned int>(~unset_bit_mask[j % 30]) << (j % 4);
                            //increment the next multiple of the current prime (rotate the wheel).
                            j += k * next_wheel_gap;
                            wheel_index = (wheel_index + 1) % 8;
                            next_wheel_gap = sieve30_gaps[wheel_index];
                        }
                        //save the starting multiple and wheel index for the next segment
                        multiples[block_id * sieving_prime_count + i] = j - segment_size;
                        wheel_indices[block_id * sieving_prime_count + i] = wheel_index;
                    }
                }
                __syncthreads();
                

                //copy the sieve results to global memory
                
                for (uint32_t j2 = index; j2 < Cuda_sieve::m_kernel_sieve_size_words; j2 += stride)
                {
                    if (j2 < Cuda_sieve::m_kernel_sieve_size_words)
                    {
                        sieve_results[sieve_results_index + j2] = sieve[j2];
                        //count prime candidates
                        //__popcll is required if sieve type is uint64_t.  __popc is specified for uint32_t but seems to have no significant impact on performance
                        count += __popcll(sieve[j2]);  

                    }
                }
                
                sieve_results_index += Cuda_sieve::m_kernel_sieve_size_words;
                __syncthreads();
                
            }
            //update the global candidate count
             atomicAdd(static_cast<unsigned long long*>(prime_candidate_count), count);
            
        }


        void Cuda_sieve_impl::run_sieve(uint64_t sieve_start_offset)
        {
            m_sieve_start_offset = sieve_start_offset;
            
            do_sieve <<<Cuda_sieve::m_num_blocks, Cuda_sieve::m_threads_per_block >>> (sieve_start_offset, d_sieving_primes, m_sieving_prime_count,
                d_starting_multiples, d_prime_mod_inverses, d_sieve, d_multiples, d_wheel_indices, d_prime_candidate_count);

            checkCudaErrors(cudaDeviceSynchronize());
        }

        void Cuda_sieve_impl::get_sieve(Cuda_sieve::sieve_word_t sieve[])
        {
            checkCudaErrors(cudaMemcpy(sieve, d_sieve, Cuda_sieve::m_sieve_total_size * sizeof(*d_sieve), cudaMemcpyDeviceToHost));

        }

        void Cuda_sieve_impl::get_prime_candidate_count(uint64_t& prime_candidate_count)
        {
            checkCudaErrors(cudaMemcpy(&prime_candidate_count, d_prime_candidate_count, sizeof(*d_prime_candidate_count), cudaMemcpyDeviceToHost));

        }

        void Cuda_sieve_impl::find_chains(CudaChain chains[], uint32_t& chain_count)
        {
            const int sieve_threads = 256;
            const int checks_per_block = 1;
            const uint32_t sieve_bits_per_word = Cuda_sieve::m_sieve_word_byte_count * 8;
            const uint64_t sieve_total_bits = Cuda_sieve::m_sieve_total_size * sieve_bits_per_word;
            const int sieve_blocks = (sieve_total_bits /checks_per_block + sieve_threads - 1)/ sieve_threads;
            
            //run the kernel
            find_chain_kernel << <sieve_blocks, sieve_threads >> > (d_sieve, d_chains, d_chain_index, m_sieve_start_offset);

            checkCudaErrors(cudaDeviceSynchronize());
            checkCudaErrors(cudaMemcpy(&chain_count, d_chain_index, sizeof(*d_chain_index), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(chains, d_chains, chain_count * sizeof(*d_chains), cudaMemcpyDeviceToHost));
        }

        //allocate global memory and load values used by the sieve to the gpu 
        void Cuda_sieve_impl::load_sieve(uint32_t primes[], uint32_t prime_count, uint32_t starting_multiples_host[],
            uint32_t prime_mod_inverses_host[], uint32_t sieve_size, uint16_t device)
        {
          
            m_sieving_prime_count = prime_count;
            checkCudaErrors(cudaSetDevice(device));
            //allocate memory on the gpu
            checkCudaErrors(cudaMalloc(&d_sieving_primes, prime_count * sizeof(*d_sieving_primes)));
            checkCudaErrors(cudaMalloc(&d_starting_multiples, prime_count * sizeof(*d_starting_multiples)));
            checkCudaErrors(cudaMalloc(&d_prime_mod_inverses, prime_count * sizeof(*d_prime_mod_inverses)));
            checkCudaErrors(cudaMalloc(&d_sieve, sieve_size * sizeof(*d_sieve)));
            checkCudaErrors(cudaMalloc(&d_multiples, prime_count * Cuda_sieve::m_num_blocks * sizeof(*d_multiples)));
            checkCudaErrors(cudaMalloc(&d_wheel_indices, prime_count * Cuda_sieve::m_num_blocks * sizeof(*d_wheel_indices)));
            checkCudaErrors(cudaMalloc(&d_chains, Cuda_sieve::m_max_chains * sizeof(*d_chains)));
            checkCudaErrors(cudaMalloc(&d_chain_index, sizeof(*d_chain_index)));
            checkCudaErrors(cudaMalloc(&d_prime_candidate_count, sizeof(*d_prime_candidate_count)));


            //copy data to the gpu
            checkCudaErrors(cudaMemcpy(d_sieving_primes, primes, prime_count * sizeof(*d_sieving_primes), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_starting_multiples, starting_multiples_host, prime_count * sizeof(*d_starting_multiples), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_prime_mod_inverses, prime_mod_inverses_host, prime_count * sizeof(*d_prime_mod_inverses), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemset(d_chain_index, 0, sizeof(*d_chain_index)));
            checkCudaErrors(cudaMemset(d_prime_candidate_count, 0, sizeof(*d_prime_candidate_count)));


        }

        void Cuda_sieve_impl::free_sieve()
        {
            checkCudaErrors(cudaFree(d_sieving_primes));
            checkCudaErrors(cudaFree(d_starting_multiples));
            checkCudaErrors(cudaFree(d_wheel_indices));
            checkCudaErrors(cudaFree(d_multiples));
            checkCudaErrors(cudaFree(d_prime_mod_inverses));
            checkCudaErrors(cudaFree(d_sieve));
            checkCudaErrors(cudaFree(d_chains));
            checkCudaErrors(cudaFree(d_chain_index));
        }
    }
}