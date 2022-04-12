#include "../gpu_helper.hpp"
#include "fermat_prime_impl.cuh"
#include <stdio.h>
#include <math.h>
#include <inttypes.h>
#include "../fermat_prime/fermat_utils.cuh"
#include "../cuda_chain.cuh"

namespace nexusminer {
    namespace gpu {

        __device__  bool get_next_fermat_candidate(CudaChain& chain, uint64_t& base_offset, int& offset);
        __device__  bool update_fermat_status(CudaChain& chain, bool is_prime);

        __global__ void
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

             kernel_fermat <<<blocks, threads_per_block>>> (d_offsets, d_offset_count, d_base_int,
                 d_results, d_fermat_test_count, d_fermat_pass_count);

             checkGPUErrors(NEXUSMINER_GPU_PeekAtLastError());
             checkGPUErrors(NEXUSMINER_GPU_DeviceSynchronize());
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
            
            
            uint64_t offset64 = 0, base_offset = 0;
            int relative_offset = 0;
            get_next_fermat_candidate(chains[index], base_offset, relative_offset);
            offset64 = base_offset + relative_offset;
            const bool is_prime = powm_2(*base_int, offset64);
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
            checkGPUErrors(NEXUSMINER_GPU_Memcpy(&chain_count, d_chain_count, sizeof(*d_chain_count), NEXUSMINER_GPU_MemcpyDeviceToHost));
            //printf("chain count %i\n", chain_count);
            int blocks = (chain_count + instances_per_block - 1) / instances_per_block;
            
            fermat_test_chains <<<blocks, threads_per_block>>> (d_chains, d_chain_count, d_base_int,
                d_results, d_fermat_test_count, d_fermat_pass_count);

            checkGPUErrors(NEXUSMINER_GPU_PeekAtLastError());
            checkGPUErrors(NEXUSMINER_GPU_DeviceSynchronize());
        }

        //allocate device memory for gpu fermat testing.  we use a fixed maximum batch size and allocate device memory once at the beginning. 
        void Fermat_prime_impl::fermat_init(uint64_t batch_size, int device)
        {

            m_device = device;

            checkGPUErrors(NEXUSMINER_GPU_SetDevice(device));
            checkGPUErrors(NEXUSMINER_GPU_Malloc(&d_base_int, sizeof(*d_base_int)));
            checkGPUErrors(NEXUSMINER_GPU_Malloc(&d_offsets, sizeof(*d_offsets) * batch_size));
            checkGPUErrors(NEXUSMINER_GPU_Malloc(&d_results, sizeof(*d_results) * batch_size));
            checkGPUErrors(NEXUSMINER_GPU_Malloc(&d_offset_count, sizeof(*d_offset_count)));
            checkGPUErrors(NEXUSMINER_GPU_Malloc(&d_fermat_test_count, sizeof(*d_fermat_test_count)));
            checkGPUErrors(NEXUSMINER_GPU_Malloc(&d_fermat_pass_count, sizeof(*d_fermat_pass_count)));
            checkGPUErrors(NEXUSMINER_GPU_Memset(d_fermat_test_count, 0, sizeof(*d_fermat_test_count)));
            checkGPUErrors(NEXUSMINER_GPU_Memset(d_fermat_pass_count, 0, sizeof(*d_fermat_pass_count)));
            checkGPUErrors(NEXUSMINER_GPU_Malloc(&d_trial_division_test_count, sizeof(*d_trial_division_test_count)));
            checkGPUErrors(NEXUSMINER_GPU_Malloc(&d_trial_division_composite_count, sizeof(*d_trial_division_composite_count)));
            checkGPUErrors(NEXUSMINER_GPU_Memset(d_trial_division_test_count, 0, sizeof(*d_trial_division_test_count)));
            checkGPUErrors(NEXUSMINER_GPU_Memset(d_trial_division_composite_count, 0, sizeof(*d_trial_division_composite_count)));

        }

        void Fermat_prime_impl::fermat_free()
        {
            checkGPUErrors(NEXUSMINER_GPU_SetDevice(m_device));
            checkGPUErrors(NEXUSMINER_GPU_Free(d_base_int));
            checkGPUErrors(NEXUSMINER_GPU_Free(d_offsets));
            checkGPUErrors(NEXUSMINER_GPU_Free(d_results));
            checkGPUErrors(NEXUSMINER_GPU_Free(d_offset_count));
            checkGPUErrors(NEXUSMINER_GPU_Free(d_fermat_test_count));
            checkGPUErrors(NEXUSMINER_GPU_Free(d_fermat_pass_count));
            checkGPUErrors(NEXUSMINER_GPU_Free(d_trial_division_test_count));
            checkGPUErrors(NEXUSMINER_GPU_Free(d_trial_division_composite_count));
        }

        void Fermat_prime_impl::set_base_int(ump::uint1024_t base_big_int)
        {
            checkGPUErrors(NEXUSMINER_GPU_SetDevice(m_device));
            Cump<1024> cuda_base_big_int;
            cuda_base_big_int.from_ump(base_big_int);
            checkGPUErrors(NEXUSMINER_GPU_Memcpy(d_base_int, &cuda_base_big_int, sizeof(cuda_base_big_int), NEXUSMINER_GPU_MemcpyHostToDevice));
            m_base_int = base_big_int;
        }

        void Fermat_prime_impl::set_offsets(uint64_t offsets[], uint64_t offset_count)
        {
            checkGPUErrors(NEXUSMINER_GPU_Memcpy(d_offsets, offsets, sizeof(*offsets) * offset_count, NEXUSMINER_GPU_MemcpyHostToDevice));
            checkGPUErrors(NEXUSMINER_GPU_Memcpy(d_offset_count, &offset_count, sizeof(offset_count), NEXUSMINER_GPU_MemcpyHostToDevice));
            m_offset_count = offset_count;
        }

        void Fermat_prime_impl::get_results(uint8_t results[])
        {
            checkGPUErrors(NEXUSMINER_GPU_Memcpy(results, d_results, sizeof(uint8_t) * m_offset_count, NEXUSMINER_GPU_MemcpyDeviceToHost));
        }

        void Fermat_prime_impl::get_stats(uint64_t& fermat_tests, uint64_t& fermat_passes,
            uint64_t& trial_division_tests, uint64_t& trial_division_composites)
        {
            checkGPUErrors(NEXUSMINER_GPU_Memcpy(&fermat_tests, d_fermat_test_count, sizeof(*d_fermat_test_count), NEXUSMINER_GPU_MemcpyDeviceToHost));
            checkGPUErrors(NEXUSMINER_GPU_Memcpy(&fermat_passes, d_fermat_pass_count, sizeof(*d_fermat_pass_count), NEXUSMINER_GPU_MemcpyDeviceToHost));
            checkGPUErrors(NEXUSMINER_GPU_Memcpy(&trial_division_tests, d_trial_division_test_count, sizeof(*d_trial_division_test_count), NEXUSMINER_GPU_MemcpyDeviceToHost));
            checkGPUErrors(NEXUSMINER_GPU_Memcpy(&trial_division_composites, d_trial_division_composite_count, sizeof(*d_trial_division_composite_count), NEXUSMINER_GPU_MemcpyDeviceToHost));
        }

        void Fermat_prime_impl::reset_stats()
        {
            checkGPUErrors(NEXUSMINER_GPU_Memset(d_fermat_test_count, 0, sizeof(*d_fermat_test_count)));
            checkGPUErrors(NEXUSMINER_GPU_Memset(d_fermat_pass_count, 0, sizeof(*d_fermat_pass_count)));
            checkGPUErrors(NEXUSMINER_GPU_Memset(d_trial_division_test_count, 0, sizeof(*d_trial_division_test_count)));
            checkGPUErrors(NEXUSMINER_GPU_Memset(d_trial_division_composite_count, 0, sizeof(*d_trial_division_composite_count)));
        }

        void Fermat_prime_impl::set_chain_ptr(CudaChain* chains, uint32_t* chain_count)
        {
            d_chains = chains;
            d_chain_count = chain_count;
            //uint32_t chain_count_test;
            //checkGPUErrors(NEXUSMINER_GPU_Memcpy(&chain_count_test, d_chain_count, sizeof(*d_chain_count), NEXUSMINER_GPU_MemcpyDeviceToHost));
            //printf("chain count test %i\n", chain_count_test);
            //CudaChain cc;
            //checkGPUErrors(NEXUSMINER_GPU_Memcpy(&cc, d_chains, sizeof(*d_chains), NEXUSMINER_GPU_MemcpyDeviceToHost));
            //printf("first chain offset count %i\n",cc.m_offset_count);
        }

        void Fermat_prime_impl::synchronize()
        {
            checkGPUErrors(NEXUSMINER_GPU_DeviceSynchronize());
        }

        __global__ void trial_division_chains(CudaChain* chains, uint32_t* chain_count, trial_divisors_uint32_t* trial_divisors,
            uint32_t* trial_divisor_count, unsigned long long* test_count, unsigned long long* composite_count) {

            const unsigned int num_threads = blockDim.x;
            const unsigned int block_id = blockIdx.x;
            const unsigned int thread_index = threadIdx.x;
            const int threads_per_instance = 1;
            const uint32_t index = block_id * num_threads / threads_per_instance + thread_index / threads_per_instance;


            if (index >= *chain_count)
                return;

            uint64_t offset64, base_offset, prime_offset;
            int relative_offset;
            get_next_fermat_candidate(chains[index], base_offset, relative_offset);
            offset64 = base_offset + relative_offset;
            bool is_composite = false;
            for (int i = 0; i < *trial_divisor_count; i++)
            {
                prime_offset = trial_divisors[i].starting_multiple + offset64;
                if (prime_offset % trial_divisors[i].divisor == 0)
                {
                    is_composite = true;
                    break;
                }
            }

            if (is_composite)
                update_fermat_status(chains[index], false);
            if (thread_index % threads_per_instance == 0)
            {
                if (is_composite)
                {
                    atomicAdd(composite_count, 1);
                }
                atomicAdd(test_count, *trial_divisor_count);

            }

        }

        //Experimental.  This is too slow to be useful. 
        void Fermat_prime_impl::trial_division_chain_run()
        {
            const int32_t threads_per_block = 1024;
            const int32_t threads_per_instance = 1;
            const int32_t instances_per_block = threads_per_block / threads_per_instance;

            uint32_t chain_count;
            checkGPUErrors(NEXUSMINER_GPU_Memcpy(&chain_count, d_chain_count, sizeof(*d_chain_count), NEXUSMINER_GPU_MemcpyDeviceToHost));

            int blocks = (chain_count + instances_per_block - 1) / instances_per_block;
            trial_division_chains<<<blocks, threads_per_block>>> (d_chains, d_chain_count, d_trial_divisors, 
                d_trial_divisor_count, d_trial_division_test_count, d_trial_division_composite_count);

            checkGPUErrors(NEXUSMINER_GPU_PeekAtLastError());
            checkGPUErrors(NEXUSMINER_GPU_DeviceSynchronize());
        }

        void Fermat_prime_impl::trial_division_init(uint32_t trial_divisor_count, trial_divisors_uint32_t trial_divisors[],
            int device)
        {
            checkGPUErrors(NEXUSMINER_GPU_SetDevice(device));
            checkGPUErrors(NEXUSMINER_GPU_Malloc(&d_trial_divisor_count, sizeof(*d_trial_divisor_count)));
            checkGPUErrors(NEXUSMINER_GPU_Malloc(&d_trial_divisors, trial_divisor_count * sizeof(*d_trial_divisors)));
            
            checkGPUErrors(NEXUSMINER_GPU_Memcpy(d_trial_divisors, trial_divisors, trial_divisor_count * sizeof(*d_trial_divisors), NEXUSMINER_GPU_MemcpyHostToDevice));
            checkGPUErrors(NEXUSMINER_GPU_Memcpy(d_trial_divisor_count, &trial_divisor_count, sizeof(*d_trial_divisor_count), NEXUSMINER_GPU_MemcpyHostToDevice));

        }

        void Fermat_prime_impl::trial_division_free()
        {
            checkGPUErrors(NEXUSMINER_GPU_SetDevice(m_device));
            checkGPUErrors(NEXUSMINER_GPU_Free(d_trial_divisor_count));
            checkGPUErrors(NEXUSMINER_GPU_Free(d_trial_divisors));
            

        }

        void Fermat_prime_impl::test_init(uint64_t batch_size, int device)
        {
            m_device = device;
            checkGPUErrors(NEXUSMINER_GPU_SetDevice(device));
            checkGPUErrors(NEXUSMINER_GPU_Malloc(&d_test_a, sizeof(*d_test_a) * batch_size));
            checkGPUErrors(NEXUSMINER_GPU_Malloc(&d_test_b, sizeof(*d_test_b) * batch_size));
            checkGPUErrors(NEXUSMINER_GPU_Malloc(&d_test_results, sizeof(*d_test_results) * batch_size));
            checkGPUErrors(NEXUSMINER_GPU_Malloc(&d_test_vector_size, sizeof(*d_test_vector_size)));

        }

        void Fermat_prime_impl::test_free()
        {
            checkGPUErrors(NEXUSMINER_GPU_SetDevice(m_device));
            checkGPUErrors(NEXUSMINER_GPU_Free(d_test_a));
            checkGPUErrors(NEXUSMINER_GPU_Free(d_test_b));
            checkGPUErrors(NEXUSMINER_GPU_Free(d_test_results));
            checkGPUErrors(NEXUSMINER_GPU_Free(d_test_vector_size));

        }

        void Fermat_prime_impl::set_input_a(ump::uint1024_t* a, uint64_t count)
        {
            m_test_vector_a_size = count;
            Cump<1024>* vector_a = new Cump<1024>[count];
            for (auto i = 0; i < count; i++)
            {
                vector_a[i].from_ump(a[i]);
            }
            checkGPUErrors(NEXUSMINER_GPU_Memcpy(d_test_a, vector_a, sizeof(*vector_a) * count, NEXUSMINER_GPU_MemcpyHostToDevice));
            checkGPUErrors(NEXUSMINER_GPU_Memcpy(d_test_vector_size, &count, sizeof(count), NEXUSMINER_GPU_MemcpyHostToDevice));
            delete[] vector_a;
        }

        void Fermat_prime_impl::set_input_b(ump::uint1024_t* b, uint64_t count)
        {
            m_test_vector_b_size = count;
            Cump<1024>* vector_b = new Cump<1024>[count];
            for (auto i = 0; i < count; i++)
            {
                vector_b[i].from_ump(b[i]);
            }
            checkGPUErrors(NEXUSMINER_GPU_Memcpy(d_test_b, vector_b, sizeof(*vector_b) * count, NEXUSMINER_GPU_MemcpyHostToDevice));
            delete[] vector_b;
        }

        

        void Fermat_prime_impl::get_test_results(ump::uint1024_t* test_results)
        {
            Cump<1024>* results = new Cump<1024>[m_test_vector_a_size];
            checkGPUErrors(NEXUSMINER_GPU_Memcpy(results, d_test_results, sizeof(*d_test_results) * m_test_vector_a_size, NEXUSMINER_GPU_MemcpyDeviceToHost));
            for (auto i = 0; i < m_test_vector_a_size; i++)
            {
                //mpz_init(test_results[i]);
                results[i].to_ump(test_results[i]);
            }
            delete[] results;
        }

        
        //this is a generic test kernel for evaluating big int math functions
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
            logic_test_kernel <<<blocks, threads_per_block>>> (d_test_a, d_test_b, d_test_results, d_test_vector_size);
            checkGPUErrors(NEXUSMINER_GPU_PeekAtLastError());
            checkGPUErrors(NEXUSMINER_GPU_DeviceSynchronize());
        }

        
    }
}
