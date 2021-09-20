//A substatial amount of this template class was copied from the nvidia gbgn bignum library miller rabin example.
//It was modified to run a base 2 fermat test.  I'm including the Nvidia copyright notice that came with cgbn

/***

Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.

***/

#ifndef NEXUSMINER_GPU_FERMAT_CUH
#define NEXUSMINER_GPU_FERMAT_CUH

#include <stdint.h>
#include "cgbn/cgbn.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// The CGBN context uses the following three parameters:
//   TBP             - threads per block (zero means to use the blockDim.x)
//   MAX_ROTATION    - must be small power of 2, imperically, 4 works well
//   SHM_LIMIT       - number of bytes of dynamic shared memory available to the kernel
//   CONSTANT_TIME   - require constant time algorithms (currently, constant time algorithms are not available)

// Locally it will also be helpful to have several parameters:
//   TPI             - threads per instance
//   BITS            - number of bits per instance
//   WINDOW_BITS     - number of bits to use for the windowed exponentiation

namespace nexusminer {
	namespace gpu {

        class fermat_params_t {
        public:
            // parameters used by the CGBN context
            static const uint32_t TPB = 0;                     // get TPB from blockDim.x  
            static const uint32_t MAX_ROTATION = 8;            // must be a small power of 2.  4 or 8 work
            static const uint32_t SHM_LIMIT = 0;               // no shared mem available
            static const bool     CONSTANT_TIME = false;       // constant time implementations aren't available yet

            // parameters used locally in the application
            static const uint32_t TPI = 8;                   // threads per instance -- must be 8, 16, or 32 for 1024 bits.  8 is fastest.
            static const uint32_t BITS = 1024;                 // bit width of the big uint
            static const uint32_t WINDOW_BITS = 5;   // window size
        };


        class fermat_t {
        public:
            static const uint32_t window_bits = fermat_params_t::WINDOW_BITS;  // used a lot, give it an instance variable

            typedef struct {
                cgbn_mem_t<fermat_params_t::BITS> candidate;
                bool           passed;
            } instance_t;

            typedef cgbn_context_t<fermat_params_t::TPI, fermat_params_t>    context_t;
            typedef cgbn_env_t<context_t, fermat_params_t::BITS>    env_t;
            typedef typename env_t::cgbn_t                 bn_t;
            typedef typename env_t::cgbn_local_t           bn_local_t;
            typedef typename env_t::cgbn_wide_t            bn_wide_t;

            context_t _context;
            env_t     _env;
            int32_t   _instance;

            __device__ fermat_t(cgbn_monitor_t monitor, cgbn_error_report_t* report, int32_t instance);
            __device__ void powm(bn_t& x, const bn_t& power, const bn_t& modulus); 
            __device__ bool fermat(const bn_t& candidate); 
            __host__ static instance_t* generate_instances(mpz_t base_big_int, uint64_t offsets[], uint32_t count); 
            __host__ static void offsets_to_cgbn(uint64_t offsets[], uint32_t count, cgbn_mem_t<64> cgbn_offsets[]);
            __host__ static void verify_results(instance_t* instances, uint32_t instance_count); 
        };
	}
}
#endif