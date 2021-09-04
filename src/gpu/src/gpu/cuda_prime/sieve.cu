#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "sieve.cuh"
#include <cuda.h>

__device__ constexpr uint8_t unset_bit_mask(int i)
{
    constexpr uint8_t unset_bit_mask[30] = {
    (uint8_t)~(1 << 0), (uint8_t)~(1 << 0),
    (uint8_t)~(1 << 1), (uint8_t)~(1 << 1), (uint8_t)~(1 << 1), (uint8_t)~(1 << 1), (uint8_t)~(1 << 1), (uint8_t)~(1 << 1),
    (uint8_t)~(1 << 2), (uint8_t)~(1 << 2), (uint8_t)~(1 << 2), (uint8_t)~(1 << 2),
    (uint8_t)~(1 << 3), (uint8_t)~(1 << 3),
    (uint8_t)~(1 << 4), (uint8_t)~(1 << 4), (uint8_t)~(1 << 4), (uint8_t)~(1 << 4),
    (uint8_t)~(1 << 5), (uint8_t)~(1 << 5),
    (uint8_t)~(1 << 6), (uint8_t)~(1 << 6), (uint8_t)~(1 << 6), (uint8_t)~(1 << 6),
    (uint8_t)~(1 << 7), (uint8_t)~(1 << 7), (uint8_t)~(1 << 7), (uint8_t)~(1 << 7), (uint8_t)~(1 << 7), (uint8_t)~(1 << 7)
    };
    return unset_bit_mask[i];
};

__device__ constexpr int sieve30_gaps(int i)
{
    constexpr int sieve30_gaps[]{ 6,4,2,4,2,4,6,2 };
    return sieve30_gaps[i];

}

//seive kernel

__global__ void do_sieve(uint32_t sieving_primes[], uint32_t sieving_prime_count, uint32_t starting_multiples[],
    int wheel_indices[], uint8_t sieve_results[], uint32_t sieve_results_size)
{
    
    const int sieve_size = 32768;  //this is the size of the sieve segment in bytes.  todo: find optimum sieve size
    uint32_t segment_size = sieve_size * 30;

    //local copy of the sieve
    uint8_t sieve[sieve_size];

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int segments = sieve_results_size / sieve_size;
    int sieve_results_index = 0;

    //initialize the sieve
    for (int j = 0; j < sieve_size; j++)
    {
        sieve[j] = 0xFF;
    }
    
    for (int s = 0; s < segments; s++)
    {
        for (int i = index; i < sieving_prime_count; i+=stride)
        {
            if (i < sieving_prime_count)
            {
                uint32_t j = starting_multiples[i];
                uint32_t k = sieving_primes[i];
                //where are we in the wheel
                int wheel_index = wheel_indices[i];
                int next_wheel_gap = sieve30_gaps(wheel_index);
                while (j < segment_size)
                {
                    //cross off a multiple of the sieving prime
                    sieve[j / 30] &= unset_bit_mask(j % 30);
                    //increment the next multiple of the current prime (rotate the wheel).
                    j += k * next_wheel_gap;
                    wheel_index = (wheel_index + 1) % 8;
                    next_wheel_gap = sieve30_gaps(wheel_index);
                    
                }
                //save the starting multiple and wheel index for the next segment
            
                starting_multiples[i] = j - segment_size;
                wheel_indices[i] = wheel_index;
            }
        }
        //bitwise and the sieve results to global memory and reset the sieve
        for (int j = 0; j < sieve_size; j++)
        {
            //uint32_t* sieve_global = reinterpret_cast<uint32_t*>(&sieve_results[sieve_results_index + j]);
            //uint32_t* sieve_local = reinterpret_cast<uint32_t*>(&sieve[j]);
            sieve_results[sieve_results_index + j] = sieve_results[sieve_results_index + j] & sieve[j];   
            sieve[j] = 0xFF;
        }
        
        sieve_results_index += sieve_size;
    }

}


void run_sieve(uint32_t sieving_primes[], uint32_t sieving_prime_count,
    uint32_t starting_multiples[], int wheel_indices[], uint8_t sieve[], uint32_t& sieve_size)
{
    //initialize the sieve
    for (int j = 0; j < sieve_size; j++)
    {
        sieve[j] = 0xFF;
    }

    //device memory pointers
    uint32_t *d_sieving_primes, *d_starting_multiples, *d_sieve_size, *d_sieving_prime_count;
    int *d_wheel_indices;
    uint8_t *d_sieve;

    cudaSetDevice(0);

    //allocate memory on the gpu
    cudaMalloc(&d_sieving_primes, sieving_prime_count * sizeof(uint32_t));
    cudaMalloc(&d_starting_multiples, sieving_prime_count * sizeof(uint32_t));
    //cudaMalloc(&d_sieve_size, sizeof(uint32_t));
    //cudaMalloc(&d_sieving_prime_count, sizeof(uint32_t));
    cudaMalloc(&d_wheel_indices, sieving_prime_count * sizeof(int));
    cudaMalloc(&d_sieve, sieve_size * sizeof(uint8_t));

    //copy data to the gpu
    cudaMemcpy(d_sieving_primes, sieving_primes, sieving_prime_count * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_starting_multiples, starting_multiples, sieving_prime_count * sizeof(uint32_t), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_sieve_size, &sieve_size, sizeof(uint32_t), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_sieving_prime_count, &sieving_prime_count, sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wheel_indices, wheel_indices, sieving_prime_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sieve, sieve, sieve_size * sizeof(uint8_t), cudaMemcpyHostToDevice);

    //run the kernel
    int blockSize = 256;
    int numBlocks = 1;// (sieving_prime_count + blockSize - 1) / blockSize;
    do_sieve<<<numBlocks, blockSize>>>(d_sieving_primes, sieving_prime_count, d_starting_multiples, d_wheel_indices, d_sieve, sieve_size);

    cudaDeviceSynchronize();

    //copy results from device to the host
    cudaMemcpy(starting_multiples, d_starting_multiples, sieving_prime_count * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    //cudaMemcpy(&sieve_size, d_sieve_size, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(wheel_indices, d_wheel_indices, sieving_prime_count * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(sieve, d_sieve, sieve_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    //clean up
    cudaFree(d_sieving_primes);
    cudaFree(d_starting_multiples);
    //cudaFree(d_sieve_size);
    cudaFree(d_wheel_indices);
    cudaFree(d_sieve);
    //cudaFree(d_sieving_prime_count);



}