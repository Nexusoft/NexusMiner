#ifndef NEXUSMINER_GPU_CUDA_FERMAT_PRIME_IMPL_CUH
#define NEXUSMINER_GPU_CUDA_FERMAT_PRIME_IMPL_CUH

#include "fermat_prime.hpp"
#include "cump.cuh"
#include "../cuda_chain.cuh"
#include <stdint.h>
#include <gmp.h>

namespace nexusminer {
	namespace gpu {

        

		class Fermat_prime_impl
		{
        public:

            void fermat_run();
            void fermat_chain_run();
            void fermat_init(uint64_t batch_size, int device);
            void fermat_free();
            void set_base_int(mpz_t base_big_int);
            void set_offsets(uint64_t offsets[], uint64_t offset_count);
            void get_results(uint8_t results[]);
            void get_stats(uint64_t& fermat_tests, uint64_t& fermat_passes, uint64_t& trial_division_tests,
                uint64_t& trial_division_composites);
            void reset_stats();
            void set_chain_ptr(CudaChain* chains, uint32_t* chain_count);
            void synchronize();

            void trial_division_chain_run();
            void trial_division_init(uint32_t trial_divisor_count, trial_divisors_uint32_t trial_divisors[], int device);
            void trial_division_free();


            //non-fermat related test functions
            void test_init(uint64_t batch_size, int device);
            void test_free();
            void set_input_a(mpz_t* a, uint64_t count);
            void set_input_b(mpz_t* b, uint64_t count);
            void get_test_results(mpz_t* test_results);
            void logic_test();


        private:
            //device memory
            
            //intputs to the fermat test
            uint64_t* d_offsets;  //64 bit offsets from the 1024 bit base integer
            uint64_t* d_offset_count; 
            Cump<1024>* d_base_int; //the 1024 bit base integer

            //results of the fermat test
            uint8_t* d_results;

            //trial division
            trial_divisors_uint32_t* d_trial_divisors;
            uint32_t* d_trial_divisor_count;
            unsigned long long* d_trial_division_test_count;
            unsigned long long* d_trial_division_composite_count;
            
            //test vectors
            Cump<1024>* d_test_a;
            Cump<1024>* d_test_b;
            Cump<1024>* d_test_results;
            uint64_t* d_test_vector_size;

            unsigned long long* d_fermat_test_count;
            unsigned long long* d_fermat_pass_count;

            //array of chain candidates
            CudaChain* d_chains;
            uint32_t* d_chain_count;

            //host memory
            int m_device = 0;
            mpz_t m_base_int;
            uint64_t m_offset_count;
            uint64_t m_test_vector_a_size;
            uint64_t m_test_vector_b_size;

			
			

		};
	}
}

#endif