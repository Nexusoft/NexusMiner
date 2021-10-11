#ifndef NEXUSMINER_GPU_FERMAT_TEST_HPP
#define NEXUSMINER_GPU_FERMAT_TEST_HPP

#include <stdint.h>
#include <gmp.h>
#include <memory>
#include "cuda_chain.cuh"

namespace nexusminer {
	namespace gpu {

		class Cuda_fermat_test_impl;

		class Cuda_fermat_test
		{
		private:
			std::unique_ptr<Cuda_fermat_test_impl> m_impl;

		public:
			Cuda_fermat_test();
			~Cuda_fermat_test();
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

		};
	}
}
#endif