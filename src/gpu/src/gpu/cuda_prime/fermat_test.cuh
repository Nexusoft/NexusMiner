#ifndef FERMAT_TEST_CUH
#define FERMAT_TEST_CUH

#include <stdint.h>
#include <gmp.h>
#include "fermat.cuh"

namespace nexusminer {
	namespace gpu {

        template<uint32_t tpi, uint32_t bits, uint32_t window_bits>
        class fermat_params_t {
        public:
            // parameters used by the CGBN context
            static const uint32_t TPB = 0;                     // get TPB from blockDim.x  
            static const uint32_t MAX_ROTATION = 4;            // good default value
            static const uint32_t SHM_LIMIT = 0;               // no shared mem available
            static const bool     CONSTANT_TIME = false;       // constant time implementations aren't available yet

            // parameters used locally in the application
            static const uint32_t TPI = tpi;                   // threads per instance
            static const uint32_t BITS = bits;                 // instance size
            static const uint32_t WINDOW_BITS = window_bits;   // window size
        };

		class CudaPrimalityTest
		{
		public:

            typedef fermat_params_t<8, 1024, 5> params;
            typedef typename fermat_t<params>::instance_t instance_t;
            
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