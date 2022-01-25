#ifndef NEXUSMINER_GPU_BIG_INT_HPP
#define NEXUSMINER_GPU_BIG_INT_HPP

//big int library sufficient for primality testing 1024 bit unsigned integers.

#include <stdint.h>
#include <gmp.h>
#include <memory>
#include "../cuda_chain.cuh"

namespace nexusminer {
	namespace gpu {

		class Big_int_impl;
		class Big_int
		{
		public:

			Big_int();
			~Big_int();
			
			void fermat_run();
			void fermat_chain_run();
			void fermat_init(uint32_t batch_size, int device);
			void fermat_free();
			void set_base_int(mpz_t base_big_int);
			void set_chain_ptr(CudaChain* chains, uint32_t* chain_count);
			void set_offsets(uint64_t offsets[], uint64_t offset_count);
			void get_results(uint8_t results[]);
			void get_stats(uint64_t& fermat_tests, uint64_t& fermat_passes);
			void reset_stats();
			void synchronize();

			void test_init(uint64_t batch_size, int device);
			void test_free();
			void set_input_a(mpz_t* a, uint64_t count);
			void set_input_b(mpz_t* b, uint64_t count);
			void get_test_results(mpz_t* test_results);
			void add();
			void subtract();
			void logic_test();

		private:
			std::unique_ptr<Big_int_impl> m_impl;
			

		};
	}
}

#endif
