#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "fermat_prime_impl.cuh"
#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <inttypes.h>
#include "../fermat_prime/fermat_utils.cuh"
#include "../cuda_chain.cuh"


#ifndef checkCudaErrors
#define checkCudaErrors(call)                                \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)
#endif

namespace nexusminer {
    namespace gpu {

        __device__  bool get_next_fermat_candidate(CudaChain& chain, uint64_t& base_offset, int& offset);
        __device__  bool update_fermat_status(CudaChain& chain, bool is_prime);

        __global__ void
        //__launch_bounds__(256, 1)

            kernel_fermat(uint64_t* offsets, uint64_t* offset_count,
                Cump<1024>* base_int, uint8_t* results, unsigned long long* test_count, unsigned long long* pass_count)
        {
            const unsigned int num_threads = blockDim.x;
            const unsigned int block_id = blockIdx.x;
            const unsigned int thread_index = threadIdx.x;
            const int threads_per_instance = 1;

            const uint32_t index = block_id * num_threads/threads_per_instance + thread_index/threads_per_instance;
            

            if (index < *offset_count)
            {
                const bool is_prime = powm_2(*base_int, offsets[index]) == 1;
                if (thread_index % threads_per_instance == 0)
                {
                    if (is_prime)
                    {
                        atomicAdd(pass_count, 1);
                    }
                    results[index] = is_prime ? 1 : 0;
                    atomicAdd(test_count, 1);

                }

            }

        }

        void Fermat_prime_impl::fermat_run()
        {
            //changing thread count seems to have negligible impact on the throughput
            const int32_t threads_per_block = 32*2;
            const int32_t threads_per_instance = 1;
            const int32_t instances_per_block = threads_per_block / threads_per_instance;

            int blocks = (m_offset_count + instances_per_block - 1) / instances_per_block;

           kernel_fermat << <blocks, threads_per_block >> > (d_offsets, d_offset_count, d_base_int,
                d_results, d_fermat_test_count, d_fermat_pass_count);

            checkCudaErrors(cudaPeekAtLastError());
            checkCudaErrors(cudaDeviceSynchronize());
        }


        __global__ void fermat_test_chains(CudaChain* chains, uint32_t* chain_count,
            Cump<1024>* base_int, uint8_t* results, unsigned long long* test_count, unsigned long long* pass_count) {
            
            const unsigned int num_threads = blockDim.x;
            const unsigned int block_id = blockIdx.x;
            const unsigned int thread_index = threadIdx.x;
            const int threads_per_instance = 1;
            const uint32_t index = block_id * num_threads / threads_per_instance + thread_index / threads_per_instance;


            if (index >= *chain_count)
                return;

            uint64_t offset64, base_offset;
            int relative_offset;
            get_next_fermat_candidate(chains[index], base_offset, relative_offset);
            offset64 = base_offset + relative_offset;
           
            const bool is_prime = powm_2(*base_int, offset64) == 1;
            update_fermat_status(chains[index], is_prime);
            if (thread_index % threads_per_instance == 0)
            {
                if (is_prime)
                {
                    atomicAdd(pass_count, 1);
                }
                results[index] = is_prime ? 1 : 0;
                atomicAdd(test_count, 1);

            }

        }


        void Fermat_prime_impl::fermat_chain_run()
        {
            const int32_t threads_per_block = 32 * 2;
            const int32_t threads_per_instance = 1;
            const int32_t instances_per_block = threads_per_block / threads_per_instance;

            uint32_t chain_count;
            checkCudaErrors(cudaMemcpy(&chain_count, d_chain_count, sizeof(*d_chain_count), cudaMemcpyDeviceToHost));
            
            int blocks = (chain_count + instances_per_block - 1) / instances_per_block;
            fermat_test_chains << <blocks, threads_per_block >> > (d_chains, d_chain_count, d_base_int,
                d_results, d_fermat_test_count, d_fermat_pass_count);

            checkCudaErrors(cudaPeekAtLastError());
            checkCudaErrors(cudaDeviceSynchronize());
        }

        //allocate device memory for gpu fermat testing.  we use a fixed maximum batch size and allocate device memory once at the beginning. 
        void Fermat_prime_impl::fermat_init(uint64_t batch_size, int device)
        {

            m_device = device;

            checkCudaErrors(cudaSetDevice(device));
            checkCudaErrors(cudaMalloc(&d_base_int, sizeof(*d_base_int)));
            checkCudaErrors(cudaMalloc(&d_offsets, sizeof(*d_offsets) * batch_size));
            checkCudaErrors(cudaMalloc(&d_results, sizeof(*d_results) * batch_size));
            checkCudaErrors(cudaMalloc(&d_offset_count, sizeof(*d_offset_count)));
            checkCudaErrors(cudaMalloc(&d_fermat_test_count, sizeof(*d_fermat_test_count)));
            checkCudaErrors(cudaMalloc(&d_fermat_pass_count, sizeof(*d_fermat_pass_count)));
            reset_stats();

        }

        void Fermat_prime_impl::fermat_free()
        {
            checkCudaErrors(cudaSetDevice(m_device));
            checkCudaErrors(cudaFree(d_base_int));
            checkCudaErrors(cudaFree(d_offsets));
            checkCudaErrors(cudaFree(d_results));
            checkCudaErrors(cudaFree(d_offset_count));
            checkCudaErrors(cudaFree(d_fermat_test_count));
            checkCudaErrors(cudaFree(d_fermat_pass_count));
        }

        void Fermat_prime_impl::set_base_int(mpz_t base_big_int)
        {
            checkCudaErrors(cudaSetDevice(m_device));
            Cump<1024> cuda_base_big_int;
            cuda_base_big_int.from_mpz(base_big_int);
            checkCudaErrors(cudaMemcpy(d_base_int, &cuda_base_big_int, sizeof(cuda_base_big_int), cudaMemcpyHostToDevice));
            mpz_set(m_base_int, base_big_int);
        }

        void Fermat_prime_impl::set_offsets(uint64_t offsets[], uint64_t offset_count)
        {
            checkCudaErrors(cudaMemcpy(d_offsets, offsets, sizeof(*offsets) * offset_count, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_offset_count, &offset_count, sizeof(offset_count), cudaMemcpyHostToDevice));
            m_offset_count = offset_count;
        }

        void Fermat_prime_impl::get_results(uint8_t results[])
        {
            checkCudaErrors(cudaMemcpy(results, d_results, sizeof(uint8_t) * m_offset_count, cudaMemcpyDeviceToHost));
        }

        void Fermat_prime_impl::get_stats(uint64_t& fermat_tests, uint64_t& fermat_passes)
        {
            checkCudaErrors(cudaMemcpy(&fermat_tests, d_fermat_test_count, sizeof(*d_fermat_test_count), cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaMemcpy(&fermat_passes, d_fermat_pass_count, sizeof(*d_fermat_pass_count), cudaMemcpyDeviceToHost));
        }

        void Fermat_prime_impl::reset_stats()
        {
            checkCudaErrors(cudaMemset(d_fermat_test_count, 0, sizeof(*d_fermat_test_count)));
            checkCudaErrors(cudaMemset(d_fermat_pass_count, 0, sizeof(*d_fermat_pass_count)));
        }

        void Fermat_prime_impl::set_chain_ptr(CudaChain* chains, uint32_t* chain_count)
        {
            d_chains = chains;
            d_chain_count = chain_count;
            uint32_t chain_count_test;
            checkCudaErrors(cudaMemcpy(&chain_count_test, d_chain_count, sizeof(*d_chain_count), cudaMemcpyDeviceToHost));
        }

        void Fermat_prime_impl::synchronize()
        {
            checkCudaErrors(cudaDeviceSynchronize());
        }

        void Fermat_prime_impl::test_init(uint64_t batch_size, int device)
        {
            m_device = device;
            checkCudaErrors(cudaSetDevice(device));
            checkCudaErrors(cudaMalloc(&d_test_a, sizeof(*d_test_a) * batch_size));
            checkCudaErrors(cudaMalloc(&d_test_b, sizeof(*d_test_b) * batch_size));
            checkCudaErrors(cudaMalloc(&d_test_results, sizeof(*d_test_results) * batch_size));
            checkCudaErrors(cudaMalloc(&d_test_vector_size, sizeof(*d_test_vector_size)));

        }

        void Fermat_prime_impl::test_free()
        {
            checkCudaErrors(cudaSetDevice(m_device));
            checkCudaErrors(cudaFree(d_test_a));
            checkCudaErrors(cudaFree(d_test_b));
            checkCudaErrors(cudaFree(d_test_results));
            checkCudaErrors(cudaFree(d_test_vector_size));

        }

        void Fermat_prime_impl::set_input_a(mpz_t* a, uint64_t count)
        {
            m_test_vector_a_size = count;
            Cump<1024>* vector_a = new Cump<1024>[count];
            for (auto i = 0; i < count; i++)
            {
                vector_a[i].from_mpz(a[i]);
            }
            checkCudaErrors(cudaMemcpy(d_test_a, vector_a, sizeof(*vector_a) * count, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_test_vector_size, &count, sizeof(count), cudaMemcpyHostToDevice));
            delete[] vector_a;
        }

        void Fermat_prime_impl::set_input_b(mpz_t* b, uint64_t count)
        {
            m_test_vector_b_size = count;
            Cump<1024>* vector_b = new Cump<1024>[count];
            for (auto i = 0; i < count; i++)
            {
                vector_b[i].from_mpz(b[i]);
            }
            checkCudaErrors(cudaMemcpy(d_test_b, vector_b, sizeof(*vector_b) * count, cudaMemcpyHostToDevice));
            delete[] vector_b;
        }

        

        void Fermat_prime_impl::get_test_results(mpz_t* test_results)
        {
            Cump<1024>* results = new Cump<1024>[m_test_vector_a_size];
            checkCudaErrors(cudaMemcpy(results, d_test_results, sizeof(*d_test_results) * m_test_vector_a_size, cudaMemcpyDeviceToHost));
            for (auto i = 0; i < m_test_vector_a_size; i++)
            {
                //mpz_init(test_results[i]);
                results[i].to_mpz(test_results[i]);
            }
            delete[] results;
        }

        
        
        __global__ void 
        //__launch_bounds__(128, 1)
        logic_test_kernel(Cump<1024>* a, Cump<1024>* b, Cump<1024>* results, uint64_t* test_vector_size)
        {
            unsigned int num_threads = blockDim.x;
            unsigned int block_id = blockIdx.x;
            unsigned int thread_index = threadIdx.x;

            uint32_t index = block_id * num_threads + thread_index;
            
            if (index < *test_vector_size)
            {
                //uint32_t m_primed = -mod_inverse_32(b[index].m_limbs[0]);
                //Cump<1024> Rmodm = b[index].R_mod_m();
                //results[index] = montgomery_square_2(Rmodm, b[index], m_primed);
                //results[index] = montgomery_square(Rmodm, b[index], m_primed);
                
                //results[index] = a[index].add_ptx(b[index]);
                //results[index] = powm_2(b[index]);

                //results[index] = results[index] - Rmodm;
                //results[index] += 1;

                

                

            }

        }

        void Fermat_prime_impl::logic_test()
        {
            const int32_t threads_per_block = 32 * 8;
            const int32_t threads_per_instance = 1;
            const int32_t instances_per_block = threads_per_block / threads_per_instance;

            int blocks = (m_test_vector_a_size + instances_per_block - 1) / instances_per_block;
            logic_test_kernel << <blocks, threads_per_block >> > (d_test_a, d_test_b, d_test_results, d_test_vector_size);
            checkCudaErrors(cudaPeekAtLastError());
            checkCudaErrors(cudaDeviceSynchronize());
        }

        
    }
}
