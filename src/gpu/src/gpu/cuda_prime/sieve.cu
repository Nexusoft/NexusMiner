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

        
        
        //global variables used by all kernels
        __device__ uint32_t* sieving_primes;
        __device__ uint32_t sieving_prime_count[1];
        __device__ uint32_t* multiples;
        __device__ uint32_t* starting_multiples;
        __device__ uint32_t* prime_mod_inverses;
        __device__ uint8_t* wheel_indices;
        __device__ uint8_t* sieve_global;  //the result of the sieve is stored here
        __device__ uint32_t sieve_global_size[1];  //the size of the sieve in bytes
        uint32_t sieving_prime_count_host;


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

        __constant__ const int sieve30_gaps[]{ 6,4,2,4,2,4,6,2 };

        __constant__ const int sieve30_index[]
            { -1,0,-1,-1,-1,-1,-1, 1, -1, -1, -1, 2, -1, 3, -1, -1, -1, 4, -1, 5, -1, -1, -1, 6, -1, -1, -1, -1, -1, 7 };  //reverse lookup table (offset mod 30 to index)


        //seive kernel

        __global__ void do_sieve(uint64_t sieve_start_offset)
        {
            uint32_t segment_size = kernel_sieve_size / 8 * 30;

            //local shared copy of the sieve
            __shared__ uint8_t sieve[kernel_sieve_size];

            uint64_t block_id = blockIdx.x;
            uint64_t index = threadIdx.x;
            uint64_t stride = blockDim.x;
            uint64_t num_blocks = gridDim.x;
           
            uint64_t segments = kernel_segments_per_block;
            uint64_t sieve_results_index = block_id * kernel_sieve_size_per_block;

            uint64_t primes_per_block = (sieving_prime_count[0] + stride - 1) / stride;
            
            //each block sieves a different region
            uint64_t start_offset = sieve_start_offset + block_id * kernel_sieve_size_per_block / 8 * 30;
            
            int wheel_index;
            int next_wheel_gap;
            uint64_t j;
            uint64_t k;
            for (int s = 0; s < segments; s++)
            {
                //everyone in the block initialize part of the shared sieve
                for (int j1 = index; j1 < kernel_sieve_size; j1 += stride)
                {
                    if (j1 < kernel_sieve_size)
                        sieve[j1] = 1;
                }

                __syncthreads();
                for (uint32_t i = index; i < sieving_prime_count[0]; i += stride)
                {
                    if (i < sieving_prime_count[0])
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
                            j = multiples[block_id* sieving_prime_count[0] +i];
                            wheel_index = wheel_indices[block_id * sieving_prime_count[0] + i];
                        }
                        next_wheel_gap = sieve30_gaps[wheel_index];
                        
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
                        multiples[block_id * sieving_prime_count[0] + i] = j - segment_size;
                        wheel_indices[block_id * sieving_prime_count[0] + i] = wheel_index;
                    }
                }
                __syncthreads();
                //copy the sieve results to global memory
                
                for (int j2 = index; j2 < kernel_sieve_size; j2 += stride)
                {
                    if (j2 < kernel_sieve_size)
                    {
                        sieve_global[sieve_results_index + j2] = sieve[j2];
                    }
                }
                
                sieve_results_index += kernel_sieve_size;
                __syncthreads();
                
            }
        }


        void run_sieve(uint64_t sieve_start_offset, uint8_t sieve[])
        {
            
            uint8_t* d_sieve;

            //run the kernel
            do_sieve <<<num_blocks, threads_per_block >>> (sieve_start_offset);

            checkCudaErrors(cudaDeviceSynchronize());

            checkCudaErrors(cudaMemcpyFromSymbol(&d_sieve, sieve_global, sizeof(uint8_t*)));
            checkCudaErrors(cudaMemcpy(sieve, d_sieve, sieve_total_size * sizeof(uint8_t), cudaMemcpyDeviceToHost));
        }

        //allocate global memory and load values used by the sieve to the gpu 
        void load_sieve(uint32_t primes[], uint32_t prime_count, uint32_t starting_multiples_host[],
            uint32_t prime_mod_inverses_host[], uint32_t sieve_size)
        {
            //device memory pointers
            uint32_t* d_sieving_primes;
            uint32_t* d_starting_multiples;
            uint32_t* d_prime_mod_inverses;
            uint8_t* d_sieve;
            uint32_t* d_multiples;
            uint8_t* d_wheel_indices;
            
            sieving_prime_count_host = prime_count;
            checkCudaErrors(cudaSetDevice(0));
            //allocate memory on the gpu
            checkCudaErrors(cudaMalloc(&d_sieving_primes, prime_count * sizeof(uint32_t)));
            checkCudaErrors(cudaMalloc(&d_starting_multiples, prime_count * sizeof(uint32_t)));
            checkCudaErrors(cudaMalloc(&d_prime_mod_inverses, prime_count * sizeof(uint32_t)));
            checkCudaErrors(cudaMalloc(&d_sieve, sieve_size * sizeof(uint8_t)));
            checkCudaErrors(cudaMalloc(&d_multiples, prime_count * num_blocks * sizeof(uint32_t)));
            checkCudaErrors(cudaMalloc(&d_wheel_indices, prime_count * num_blocks * sizeof(uint8_t)));


            //copy data to the gpu
            checkCudaErrors(cudaMemcpy(d_sieving_primes, primes, prime_count * sizeof(uint32_t), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_starting_multiples, starting_multiples_host, prime_count * sizeof(uint32_t), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_prime_mod_inverses, prime_mod_inverses_host, prime_count * sizeof(uint32_t), cudaMemcpyHostToDevice));

            //point the global device variable to the allocated memory
            checkCudaErrors(cudaMemcpyToSymbol(sieving_primes, &d_sieving_primes, sizeof(uint32_t*)));
            checkCudaErrors(cudaMemcpyToSymbol(starting_multiples, &d_starting_multiples, sizeof(uint32_t*)));
            checkCudaErrors(cudaMemcpyToSymbol(prime_mod_inverses, &d_prime_mod_inverses, sizeof(uint32_t*)));
            checkCudaErrors(cudaMemcpyToSymbol(sieve_global, &d_sieve, sizeof(uint8_t*)));
            checkCudaErrors(cudaMemcpyToSymbol(sieving_prime_count, &prime_count, sizeof(uint32_t)));
            checkCudaErrors(cudaMemcpyToSymbol(sieve_global_size, &sieve_size, sizeof(uint32_t)));
            checkCudaErrors(cudaMemcpyToSymbol(wheel_indices, &d_wheel_indices, sizeof(uint8_t*)));
            checkCudaErrors(cudaMemcpyToSymbol(multiples, &d_multiples, sizeof(uint32_t*)));

        
        }

        void free_sieve()
        {
            uint32_t* d_sieving_primes;
            uint32_t* d_starting_multiples;
            uint32_t* d_prime_mod_inverses;
            uint32_t* d_multiples;
            uint8_t* d_wheel_indices;
            uint8_t* d_sieve;

            checkCudaErrors(cudaMemcpyFromSymbol(&d_sieving_primes, sieving_primes, sizeof(uint32_t*)));
            checkCudaErrors(cudaMemcpyFromSymbol(&d_starting_multiples, starting_multiples, sizeof(uint32_t*)));
            checkCudaErrors(cudaMemcpyFromSymbol(&d_prime_mod_inverses, prime_mod_inverses, sizeof(uint32_t*)));
            checkCudaErrors(cudaMemcpyFromSymbol(&d_multiples, multiples, sizeof(uint32_t*)));
            checkCudaErrors(cudaMemcpyFromSymbol(&d_wheel_indices, wheel_indices, sizeof(uint8_t*)));
            checkCudaErrors(cudaMemcpyFromSymbol(&d_sieve, sieve_global, sizeof(uint8_t*)));

            checkCudaErrors(cudaFree(d_sieving_primes));
            checkCudaErrors(cudaFree(d_starting_multiples));
            checkCudaErrors(cudaFree(d_wheel_indices));
            checkCudaErrors(cudaFree(d_multiples));
            checkCudaErrors(cudaFree(d_prime_mod_inverses));
            checkCudaErrors(cudaFree(d_sieve));


        }
    }
}