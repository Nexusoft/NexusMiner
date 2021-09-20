#ifndef NEXUSMINER_GPU_FERMAT_TEST_IMPL_CUH
#define NEXUSMINER_GPU_FERMAT_TEST_IMPL_CUH

#include <stdint.h>
#include <gmp.h>
#include "fermat.cuh"

namespace nexusminer {
	namespace gpu {

		class Cuda_fermat_test_impl
		{
		public:
           
			void fermat_run();
            void fermat_init(uint64_t batch_size, int device);
            void fermat_free();
            void set_base_int(mpz_t base_big_int);
            void set_offsets(uint64_t offsets[], uint64_t offset_count);
            void get_results(uint8_t results[]);
            void get_stats(uint64_t& fermat_tests, uint64_t& fermat_passes);
            void reset_stats();

        private:
            cgbn_mem_t<64>* d_offsets;
            uint64_t* d_offset_count;
            uint8_t* d_results;
            uint64_t* d_fermat_test_count;
            uint64_t* d_fermat_pass_count;
            fermat_t::instance_t* d_instances;
            cgbn_error_report_t* d_report;
            cgbn_mem_t<fermat_params_t::BITS>* d_base_int;
            int m_device = 0;
            mpz_t m_base_int;
            uint64_t m_offset_count;

		};
	}
}
#endif