#ifndef NEXUSMINER_GPU_FERMAT_TEST_HPP
#define NEXUSMINER_GPU_FERMAT_TEST_HPP

#include <stdint.h>
#include <gmp.h>
#include <memory>

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
			void fermat_run(mpz_t base_big_int, uint64_t offsets[], uint32_t offset_count, uint8_t results[], int device);
            void fermat_init(uint32_t batch_size, int device);
            void fermat_free();
            void set_base_int(mpz_t base_big_int);

		};
	}
}
#endif