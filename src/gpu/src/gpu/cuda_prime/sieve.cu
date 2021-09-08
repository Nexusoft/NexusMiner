#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "sieve.cuh"
#include <cuda.h>
#include <stdio.h>
#include <math.h>


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

        //return the offset from x to the next integer multiple of n greater than x that is not divisible by 2, 3, or 5.  
//x must be a multiple of the primorial 30 and n must be a prime greater than 5.
        
        template <typename T1, typename T2>
        __device__ static T2 get_offset_to_next_multiple(T1 x, T2 n)
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

        __constant__ const int sieve30_gaps[]{ 6,4,2,4,2,4,6,2 };

        __constant__ const int sieve30_index[]
            { -1,0,-1,-1,-1,-1,-1, 1, -1, -1, -1, 2, -1, 3, -1, -1, -1, 4, -1, 5, -1, -1, -1, 6, -1, -1, -1, -1, -1, 7 };  //reverse lookup table (offset mod 30 to index)


        //seive kernel

        __global__ void do_sieve(uint32_t sieving_primes[], uint32_t sieving_prime_count, uint32_t starting_multiples[],
            int wheel_indices[], uint8_t sieve_results[], uint32_t sieve_results_size, uint64_t sieve_start_offset)
        {
            uint32_t segment_size = kernel_sieve_size / 8 * 30;
           

            //local shared copy of the sieve
            __shared__ uint8_t sieve[kernel_sieve_size];

            int block_id = blockIdx.x;
            int index = threadIdx.x;
            int stride = blockDim.x;
            int num_blocks = gridDim.x;
           
            int segments = kernel_segments_per_block;
            int sieve_results_index = block_id * kernel_sieve_size_per_block;

            int primes_per_block = (sieving_prime_count + stride - 1) / stride;
            //printf("sieving primes %i ppb %i ", sieving_prime_count, primes_per_block);
            //uint32_t* multiples;
            //multiples = new uint32_t[primes_per_block];
            //int index_count = 0;
            //each block sieves a different region
            uint64_t start_offset = sieve_start_offset + block_id * kernel_sieve_size_per_block / 8 * 30;
            //initialize the starting multiples for this block region
            //printf("a");
            //__syncthreads();
            //printf("1");
            //for (int i = index; i < sieving_prime_count; i += stride)
            //{
            //    multiples[index_count] = starting_multiples[i];
            //    //get aligned to this region
            //    if (start_offset >= starting_multiples[i])
            //        multiples[index_count] = get_offset_to_next_multiple(start_offset - starting_multiples[i], sieving_primes[i]);
            //    else
            //        multiples[index_count] -= start_offset;
            //    index_count++;
            //}
            //printf("b");

            for (int s = 0; s < segments; s++)
            {
                //everyone in the block initialize part of the shared sieve
                for (int j1 = index; j1 < kernel_sieve_size; j1 += stride)
                {
                    if (j1 < kernel_sieve_size)
                        sieve[j1] = 1;
                }

                __syncthreads();
                for (int i = index; i < sieving_prime_count; i += stride)
                {
                    if (i < sieving_prime_count)
                    {
                        uint64_t j = starting_multiples[i];
                        //get aligned to this region
                        if (start_offset >= j)
                            j = get_offset_to_next_multiple(start_offset - j, sieving_primes[i]);
                        else
                            j -= start_offset;
                        uint64_t k = sieving_primes[i];
                        
                        //where are we in the wheel
                        int wheel_index = sieve30_index[(wheel_indices[i] * j) % 30];
                        int next_wheel_gap = sieve30_gaps[wheel_index];
                        while (j < segment_size)
                        {
                            //cross off a multiple of the sieving prime
                            uint64_t sieve_index = (j / 30) * 8 + sieve30_index[j % 30];
                            sieve[sieve_index] = 0;
                            //increment the next multiple of the current prime (rotate the wheel).
                            j += k * next_wheel_gap;
                            wheel_index = (wheel_index + 1) % 8;
                            next_wheel_gap = sieve30_gaps[wheel_index];
                        }
                        //save the starting multiple and wheel index for the next segment

                        //multiples[index_count] = j - segment_size;
                        //wheel_indices[i] = wheel_index;
                    }
                }
                

                __syncthreads();
                //copy the sieve results to global memory
                
                for (int j2 = index; j2 < kernel_sieve_size; j2 += stride)
                {
                    if (j2 < kernel_sieve_size)
                    {
                        sieve_results[sieve_results_index + j2] = sieve[j2];
                    }
                }
                sieve_results_index += kernel_sieve_size;
                __syncthreads();
                
            }
            //delete[] multiples;
        }


        void run_sieve(uint32_t sieving_primes[], uint32_t sieving_prime_count,
            uint32_t starting_multiples[], int wheel_indices[], uint8_t sieve[], uint32_t& sieve_size, uint64_t sieve_start_offset)
        {
            //device memory pointers
            uint32_t* d_sieving_primes, * d_starting_multiples, * d_sieve_size, * d_sieving_prime_count;
            int* d_wheel_indices;
            uint8_t* d_sieve;

            //each block sieves its own range.  Within a block, each thread handles a subset of the sieving primes.
            int threads_per_block = 1024;
            uint32_t segments = sieve_size / kernel_sieve_size;
            int num_blocks = sieve_size / kernel_sieve_size_per_block;
            //dim3 grid(numBlocks,1);

            checkCudaErrors(cudaSetDevice(0));

            //allocate memory on the gpu
            checkCudaErrors(cudaMalloc(&d_sieving_primes, sieving_prime_count * sizeof(uint32_t)));
            checkCudaErrors(cudaMalloc(&d_starting_multiples, sieving_prime_count * sizeof(uint32_t)));
            //cudaMalloc(&d_sieve_size, sizeof(uint32_t));
            //cudaMalloc(&d_sieving_prime_count, sizeof(uint32_t));
            checkCudaErrors(cudaMalloc(&d_wheel_indices, sieving_prime_count * sizeof(int)));
            checkCudaErrors(cudaMalloc(&d_sieve, sieve_size * sizeof(uint8_t)));

            //copy data to the gpu
            checkCudaErrors(cudaMemcpy(d_sieving_primes, sieving_primes, sieving_prime_count * sizeof(uint32_t), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_starting_multiples, starting_multiples, sieving_prime_count * sizeof(uint32_t), cudaMemcpyHostToDevice));
            //cudaMemcpy(d_sieve_size, &sieve_size, sizeof(uint32_t), cudaMemcpyHostToDevice);
            //cudaMemcpy(d_sieving_prime_count, &sieving_prime_count, sizeof(uint32_t), cudaMemcpyHostToDevice);
            checkCudaErrors(cudaMemcpy(d_wheel_indices, wheel_indices, sieving_prime_count * sizeof(int), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_sieve, sieve, sieve_size * sizeof(uint8_t), cudaMemcpyHostToDevice));

            //run the kernel
            do_sieve <<<num_blocks, threads_per_block >>> (d_sieving_primes, sieving_prime_count, d_starting_multiples, d_wheel_indices, d_sieve, sieve_size, sieve_start_offset);

            checkCudaErrors(cudaDeviceSynchronize());

            //copy results from device to the host
            checkCudaErrors(cudaMemcpy(starting_multiples, d_starting_multiples, sieving_prime_count * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            //cudaMemcpy(&sieve_size, d_sieve_size, sizeof(uint32_t), cudaMemcpyDeviceToHost);
            checkCudaErrors(cudaMemcpy(wheel_indices, d_wheel_indices, sieving_prime_count * sizeof(int), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(sieve, d_sieve, sieve_size * sizeof(uint8_t), cudaMemcpyDeviceToHost));

            //clean up
            checkCudaErrors(cudaFree(d_sieving_primes));
            checkCudaErrors(cudaFree(d_starting_multiples));
            //cudaFree(d_sieve_size);
            checkCudaErrors(cudaFree(d_wheel_indices));
            checkCudaErrors(cudaFree(d_sieve));
            //cudaFree(d_sieving_prime_count);

        }

        void load_sieve(uint32_t sieving_primes[], uint32_t sieving_prime_count)
        {
            //device memory pointers
            uint32_t* d_sieving_primes, *d_sieving_prime_mod_inverses;

            



        }
    }
}