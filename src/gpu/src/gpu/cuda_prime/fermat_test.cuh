#ifndef FERMAT_TEST_CUH
#define FERMAT_TEST_CUH

#include <stdint.h>
#include <gmp.h>

namespace nexusminer {
	namespace gpu {

		class CudaPrimalityTest
		{
		public:


			void run_primality_test(mpz_t base_big_int, uint64_t offsets[], uint32_t offset_count, uint8_t results[], int device);



		};
	}
}
#endif