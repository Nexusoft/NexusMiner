#ifndef NEXUSMINER_GPU_FERMAT_PRIME_HPP
#define NEXUSMINER_GPU_FERMAT_PRIME_HPP

//cuda library for primality testing 1024 bit unsigned integers.

#include <stdint.h>
#include <memory>
#include "../cuda_chain.cuh"
#include "ump.hpp"

namespace nexusminer {
	namespace gpu {

		struct trial_divisors_uint32_t
		{
			uint32_t divisor;
			uint32_t starting_multiple;
		};

		class Fermat_prime_impl;
		class Fermat_prime
		{
		public:

			Fermat_prime();
			~Fermat_prime();
			
			void fermat_run();
			void fermat_chain_run();
			void fermat_init(uint32_t batch_size, int device);
			void fermat_free();
			void set_base_int(ump::uint1024_t big_base_int);
			void set_chain_ptr(CudaChain* chains, uint32_t* chain_count);
			void set_offsets(uint64_t offsets[], uint64_t offset_count);
			void get_results(uint8_t results[]);
			void get_stats(uint64_t& fermat_tests, uint64_t& fermat_passes, uint64_t& trial_division_tests,
				uint64_t& trial_division_composites);
			void reset_stats();
			void synchronize();

			void trial_division_chain_run();
			void trial_division_init(uint32_t trial_divisor_count, trial_divisors_uint32_t trial_divisors[], int device);
			void trial_division_free();

			void test_init(uint64_t batch_size, int device);
			void test_free();
			void set_input_a(ump::uint1024_t* a, uint64_t count);
			void set_input_b(ump::uint1024_t* b, uint64_t count);
			void get_test_results(ump::uint1024_t* test_results);
			void logic_test();

		private:
			std::unique_ptr<Fermat_prime_impl> m_impl;
			

		};
	}
}

#endif
