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

             kernel_fermat <<<blocks, threads_per_block>>> (d_offsets.get(), d_offset_count.get(), d_base_int.get(),
                 d_results.get(), d_fermat_test_count.get(), d_fermat_pass_count.get());

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
            
            fermat_test_chains <<<blocks, threads_per_block>>> (d_chains, d_chain_count, d_base_int.get(),
                d_results.get(), d_fermat_test_count.get(), d_fermat_pass_count.get());

            checkGPUErrors(NEXUSMINER_GPU_PeekAtLastError());
            checkGPUErrors(NEXUSMINER_GPU_DeviceSynchronize());
        }

        //allocate device memory for gpu fermat testing.  we use a fixed maximum batch size and allocate device memory once at the beginning. 
        void Fermat_prime_impl::fermat_init(uint64_t batch_size, int device)
        {

            m_device = device;

            checkGPUErrors(NEXUSMINER_GPU_SetDevice(device));
            
            // Allocate memory using RAII wrappers
            Cump<1024>* raw_ptr_cump = nullptr;
            checkGPUErrors(NEXUSMINER_GPU_Malloc(&raw_ptr_cump, sizeof(Cump<1024>)));
            d_base_int.reset(raw_ptr_cump);
            
            uint64_t* raw_ptr_u64 = nullptr;
            checkGPUErrors(NEXUSMINER_GPU_Malloc(&raw_ptr_u64, sizeof(uint64_t) * batch_size));
            d_offsets.reset(raw_ptr_u64);
            
            uint8_t* raw_ptr_u8 = nullptr;
            checkGPUErrors(NEXUSMINER_GPU_Malloc(&raw_ptr_u8, sizeof(uint8_t) * batch_size));
            d_results.reset(raw_ptr_u8);
            
            raw_ptr_u64 = nullptr;
            checkGPUErrors(NEXUSMINER_GPU_Malloc(&raw_ptr_u64, sizeof(uint64_t)));
            d_offset_count.reset(raw_ptr_u64);
            
            unsigned long long* raw_ptr_ull = nullptr;
            checkGPUErrors(NEXUSMINER_GPU_Malloc(&raw_ptr_ull, sizeof(unsigned long long)));
            d_fermat_test_count.reset(raw_ptr_ull);
            checkGPUErrors(NEXUSMINER_GPU_Memset(d_fermat_test_count.get(), 0, sizeof(unsigned long long)));
            
            raw_ptr_ull = nullptr;
            checkGPUErrors(NEXUSMINER_GPU_Malloc(&raw_ptr_ull, sizeof(unsigned long long)));
            d_fermat_pass_count.reset(raw_ptr_ull);
            checkGPUErrors(NEXUSMINER_GPU_Memset(d_fermat_pass_count.get(), 0, sizeof(unsigned long long)));
            
            raw_ptr_ull = nullptr;
            checkGPUErrors(NEXUSMINER_GPU_Malloc(&raw_ptr_ull, sizeof(unsigned long long)));
            d_trial_division_test_count.reset(raw_ptr_ull);
            checkGPUErrors(NEXUSMINER_GPU_Memset(d_trial_division_test_count.get(), 0, sizeof(unsigned long long)));
            
            raw_ptr_ull = nullptr;
            checkGPUErrors(NEXUSMINER_GPU_Malloc(&raw_ptr_ull, sizeof(unsigned long long)));
            d_trial_division_composite_count.reset(raw_ptr_ull);
            checkGPUErrors(NEXUSMINER_GPU_Memset(d_trial_division_composite_count.get(), 0, sizeof(unsigned long long)));

        }

        void Fermat_prime_impl::fermat_free()
        {
            checkGPUErrors(NEXUSMINER_GPU_SetDevice(m_device));
            // Smart pointers automatically free memory when reset
            d_base_int.reset();
            d_offsets.reset();
            d_results.reset();
            d_offset_count.reset();
            d_fermat_test_count.reset();
            d_fermat_pass_count.reset();
            d_trial_division_test_count.reset();
            d_trial_division_composite_count.reset();
        }

        void Fermat_prime_impl::set_base_int(mpz_t base_big_int)
        {
            checkGPUErrors(NEXUSMINER_GPU_SetDevice(m_device));
            Cump<1024> cuda_base_big_int;
            cuda_base_big_int.from_mpz(base_big_int);
            checkGPUErrors(NEXUSMINER_GPU_Memcpy(d_base_int.get(), &cuda_base_big_int, sizeof(cuda_base_big_int), NEXUSMINER_GPU_MemcpyHostToDevice));
            mpz_set(m_base_int, base_big_int);
        }

        void Fermat_prime_impl::set_offsets(uint64_t offsets[], uint64_t offset_count)
        {
            checkGPUErrors(NEXUSMINER_GPU_Memcpy(d_offsets.get(), offsets, sizeof(*offsets) * offset_count, NEXUSMINER_GPU_MemcpyHostToDevice));
            checkGPUErrors(NEXUSMINER_GPU_Memcpy(d_offset_count.get(), &offset_count, sizeof(offset_count), NEXUSMINER_GPU_MemcpyHostToDevice));
            m_offset_count = offset_count;
        }

        void Fermat_prime_impl::get_results(uint8_t results[])
        {
            checkGPUErrors(NEXUSMINER_GPU_Memcpy(results, d_results.get(), sizeof(uint8_t) * m_offset_count, NEXUSMINER_GPU_MemcpyDeviceToHost));
        }

        void Fermat_prime_impl::get_stats(uint64_t& fermat_tests, uint64_t& fermat_passes,
            uint64_t& trial_division_tests, uint64_t& trial_division_composites)
        {
            checkGPUErrors(NEXUSMINER_GPU_Memcpy(&fermat_tests, d_fermat_test_count.get(), sizeof(*d_fermat_test_count.get()), NEXUSMINER_GPU_MemcpyDeviceToHost));
            checkGPUErrors(NEXUSMINER_GPU_Memcpy(&fermat_passes, d_fermat_pass_count.get(), sizeof(*d_fermat_pass_count.get()), NEXUSMINER_GPU_MemcpyDeviceToHost));
            checkGPUErrors(NEXUSMINER_GPU_Memcpy(&trial_division_tests, d_trial_division_test_count.get(), sizeof(*d_trial_division_test_count.get()), NEXUSMINER_GPU_MemcpyDeviceToHost));
            checkGPUErrors(NEXUSMINER_GPU_Memcpy(&trial_division_composites, d_trial_division_composite_count.get(), sizeof(*d_trial_division_composite_count.get()), NEXUSMINER_GPU_MemcpyDeviceToHost));
        }

        void Fermat_prime_impl::reset_stats()
        {
            checkGPUErrors(NEXUSMINER_GPU_Memset(d_fermat_test_count.get(), 0, sizeof(*d_fermat_test_count.get())));
            checkGPUErrors(NEXUSMINER_GPU_Memset(d_fermat_pass_count.get(), 0, sizeof(*d_fermat_pass_count.get())));
            checkGPUErrors(NEXUSMINER_GPU_Memset(d_trial_division_test_count.get(), 0, sizeof(*d_trial_division_test_count.get())));
            checkGPUErrors(NEXUSMINER_GPU_Memset(d_trial_division_composite_count.get(), 0, sizeof(*d_trial_division_composite_count.get())));
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
            trial_division_chains<<<blocks, threads_per_block>>> (d_chains, d_chain_count, d_trial_divisors.get(), 
                d_trial_divisor_count.get(), d_trial_division_test_count.get(), d_trial_division_composite_count.get());

            checkGPUErrors(NEXUSMINER_GPU_PeekAtLastError());
            checkGPUErrors(NEXUSMINER_GPU_DeviceSynchronize());
        }

        void Fermat_prime_impl::trial_division_init(uint32_t trial_divisor_count, trial_divisors_uint32_t trial_divisors[],
            int device)
        {
            checkGPUErrors(NEXUSMINER_GPU_SetDevice(device));
            
            uint32_t* raw_ptr_u32 = nullptr;
            checkGPUErrors(NEXUSMINER_GPU_Malloc(&raw_ptr_u32, sizeof(uint32_t)));
            d_trial_divisor_count.reset(raw_ptr_u32);
            
            trial_divisors_uint32_t* raw_ptr_div = nullptr;
            checkGPUErrors(NEXUSMINER_GPU_Malloc(&raw_ptr_div, trial_divisor_count * sizeof(trial_divisors_uint32_t)));
            d_trial_divisors.reset(raw_ptr_div);
            
            checkGPUErrors(NEXUSMINER_GPU_Memcpy(d_trial_divisors.get(), trial_divisors, trial_divisor_count * sizeof(trial_divisors_uint32_t), NEXUSMINER_GPU_MemcpyHostToDevice));
            checkGPUErrors(NEXUSMINER_GPU_Memcpy(d_trial_divisor_count.get(), &trial_divisor_count, sizeof(uint32_t), NEXUSMINER_GPU_MemcpyHostToDevice));

        }

        void Fermat_prime_impl::trial_division_free()
        {
            checkGPUErrors(NEXUSMINER_GPU_SetDevice(m_device));
            // Smart pointers automatically free memory
            d_trial_divisor_count.reset();
            d_trial_divisors.reset();
            

        }

        void Fermat_prime_impl::test_init(uint64_t batch_size, int device)
        {
            m_device = device;
            checkGPUErrors(NEXUSMINER_GPU_SetDevice(device));
            
            Cump<1024>* raw_ptr_cump = nullptr;
            checkGPUErrors(NEXUSMINER_GPU_Malloc(&raw_ptr_cump, sizeof(Cump<1024>) * batch_size));
            d_test_a.reset(raw_ptr_cump);
            
            raw_ptr_cump = nullptr;
            checkGPUErrors(NEXUSMINER_GPU_Malloc(&raw_ptr_cump, sizeof(Cump<1024>) * batch_size));
            d_test_b.reset(raw_ptr_cump);
            
            raw_ptr_cump = nullptr;
            checkGPUErrors(NEXUSMINER_GPU_Malloc(&raw_ptr_cump, sizeof(Cump<1024>) * batch_size));
            d_test_results.reset(raw_ptr_cump);
            
            uint64_t* raw_ptr_u64 = nullptr;
            checkGPUErrors(NEXUSMINER_GPU_Malloc(&raw_ptr_u64, sizeof(uint64_t)));
            d_test_vector_size.reset(raw_ptr_u64);

        }

        void Fermat_prime_impl::test_free()
        {
            checkGPUErrors(NEXUSMINER_GPU_SetDevice(m_device));
            // Smart pointers automatically free memory
            d_test_a.reset();
            d_test_b.reset();
            d_test_results.reset();
            d_test_vector_size.reset();

        }

        void Fermat_prime_impl::set_input_a(mpz_t* a, uint64_t count)
        {
            m_test_vector_a_size = count;
            Cump<1024>* vector_a = new Cump<1024>[count];
            for (auto i = 0; i < count; i++)
            {
                vector_a[i].from_mpz(a[i]);
            }
            checkGPUErrors(NEXUSMINER_GPU_Memcpy(d_test_a.get(), vector_a, sizeof(*vector_a) * count, NEXUSMINER_GPU_MemcpyHostToDevice));
            checkGPUErrors(NEXUSMINER_GPU_Memcpy(d_test_vector_size.get(), &count, sizeof(count), NEXUSMINER_GPU_MemcpyHostToDevice));
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
            checkGPUErrors(NEXUSMINER_GPU_Memcpy(d_test_b.get(), vector_b, sizeof(*vector_b) * count, NEXUSMINER_GPU_MemcpyHostToDevice));
            delete[] vector_b;
        }

        

        void Fermat_prime_impl::get_test_results(mpz_t* test_results)
        {
            Cump<1024>* results = new Cump<1024>[m_test_vector_a_size];
            checkGPUErrors(NEXUSMINER_GPU_Memcpy(results, d_test_results.get(), sizeof(*d_test_results.get()) * m_test_vector_a_size, NEXUSMINER_GPU_MemcpyDeviceToHost));
            for (auto i = 0; i < m_test_vector_a_size; i++)
            {
                //mpz_init(test_results[i]);
                results[i].to_mpz(test_results[i]);
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
            logic_test_kernel <<<blocks, threads_per_block>>> (d_test_a.get(), d_test_b.get(), d_test_results.get(), d_test_vector_size.get());
            checkGPUErrors(NEXUSMINER_GPU_PeekAtLastError());
            checkGPUErrors(NEXUSMINER_GPU_DeviceSynchronize());
        }

        
    }
}
