#ifndef NEXUSMINER_GPU_FERMAT_TEST_CUH
#define NEXUSMINER_GPU_FERMAT_TEST_CUH

#include <stdint.h>
#include <gmp.h>
#include "fermat.cuh"

namespace nexusminer {
	namespace gpu {

		class Cuda_fermat_test
		{
		public:
            static constexpr int threads_per_instance = 8;
            static constexpr int bignum_bits = 1024;
            static constexpr int window_size_bits = 5;

            using fermat_params = fermat_params_t<threads_per_instance, bignum_bits, window_size_bits>;
            using instance_t = fermat_t<fermat_params>::instance_t;
            
			void fermat_run(mpz_t base_big_int, uint64_t offsets[], uint32_t offset_count, uint8_t results[], int device);
            void fermat_init(uint32_t batch_size, int device);
            void fermat_free();
            void set_base_int(mpz_t base_big_int);

        private:
            instance_t* d_instances;
            cgbn_error_report_t* d_report;
            int m_device = 0;
            mpz_t m_base_int;

		};
	}
}
#endif