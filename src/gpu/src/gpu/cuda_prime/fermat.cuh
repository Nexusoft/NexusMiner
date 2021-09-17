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

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <gmp.h>
#include "cgbn/cgbn.h"
#include "cgbn/utility/support.h"
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

        template<class params>
        class fermat_t {
        public:
            static const uint32_t window_bits = params::WINDOW_BITS;  // used a lot, give it an instance variable

            // Definition of instance_t type.  Note, the size of instance_t is not a multiple of 128-bytes, so array loads and stores
            // will not be 128-byte aligned.  It's ok in this example, because there are so few loads and stores compared to compute,
            // but using non-128-byte aligned types could be a performance limiter for different load/store/compute balances.

            typedef struct {
                cgbn_mem_t<params::BITS> candidate;
                bool                 passed;
            } instance_t;

            typedef cgbn_context_t<params::TPI, params>    context_t;
            typedef cgbn_env_t<context_t, params::BITS>    env_t;
            typedef typename env_t::cgbn_t                 bn_t;
            typedef typename env_t::cgbn_local_t           bn_local_t;
            typedef typename env_t::cgbn_wide_t            bn_wide_t;

            context_t _context;
            env_t     _env;
            int32_t   _instance;

            __device__ __forceinline__ fermat_t(cgbn_monitor_t monitor, cgbn_error_report_t* report, int32_t instance) : _context(monitor, report, (uint32_t)instance), _env(_context), _instance(instance) {
            }

            __device__ __forceinline__ void powm(bn_t& x, const bn_t& power, const bn_t& modulus) {
                bn_t       t;
                bn_local_t window[1 << window_bits];
                int32_t    index, position, offset;
                uint32_t   np0;

                // conmpute x^power mod modulus, using the fixed window algorithm
                // requires:  x<modulus,  modulus is odd

                // compute x^0 (in Montgomery space, this is just 2^BITS - modulus)
                cgbn_negate(_env, t, modulus);
                cgbn_store(_env, window + 0, t);

                // convert x into Montgomery space, store into window table
                np0 = cgbn_bn2mont(_env, x, x, modulus);
                cgbn_store(_env, window + 1, x);
                cgbn_set(_env, t, x);

                // compute x^2, x^3, ... x^(2^window_bits-1), store into window table
#pragma nounroll
                for (index = 2; index < (1 << window_bits); index++) {
                    cgbn_mont_mul(_env, x, x, t, modulus, np0);
                    cgbn_store(_env, window + index, x);
                }

                // find leading high bit
                position = params::BITS - cgbn_clz(_env, power);

                // break the exponent into chunks, each window_bits in length
                // load the most significant non-zero exponent chunk
                offset = position % window_bits;
                if (offset == 0)
                    position = position - window_bits;
                else
                    position = position - offset;
                index = cgbn_extract_bits_ui32(_env, power, position, window_bits);
                cgbn_load(_env, x, window + index);

                // process the remaining exponent chunks
                while (position > 0) {
                    // square the result window_bits times
#pragma nounroll
                    for (int sqr_count = 0; sqr_count < window_bits; sqr_count++)
                        cgbn_mont_sqr(_env, x, x, modulus, np0);

                    // multiply by next exponent chunk
                    position = position - window_bits;
                    index = cgbn_extract_bits_ui32(_env, power, position, window_bits);
                    cgbn_load(_env, t, window + index);
                    cgbn_mont_mul(_env, x, x, t, modulus, np0);
                }

                // we've processed the exponent now, convert back to normal space
                cgbn_mont2bn(_env, x, x, modulus, np0);
            }

            __device__ __forceinline__ bool fermat(const bn_t& candidate) {

                bn_t      x, power;

                cgbn_sub_ui32(_env, power, candidate, 1);
                cgbn_set_ui32(_env, x, 2);
                powm(x, power, candidate);
                return cgbn_equals_ui32(_env, x, 1);
            }

            __host__ static instance_t* generate_instances(mpz_t base_big_int, uint64_t offsets[], uint32_t count) {
                instance_t* instances = (instance_t*)malloc(sizeof(instance_t) * count);
                int         index;

                mpz_t p, o;
                mpz_init(p);
                mpz_init(o);


                for (index = 0; index < count; index++) {
                    //mpz doesn't deal with 64 bit ints in windows.  need to use import function.
                    uint64_t off = offsets[index];
                    mpz_import(o, 1, 1, sizeof(off), 0, 0, &off);
                    mpz_add(p, base_big_int, o);
                    from_mpz(p, instances[index].candidate._limbs, params::BITS / 32);
                    instances[index].passed = 0;
                }
                mpz_clear(p);
                mpz_clear(o);

                return instances;
            }

            __host__ static void verify_results(instance_t* instances, uint32_t instance_count) {
                int   index, total = 0;
                mpz_t candidate;
                bool  gmp_prime, xmp_prime, match = true;

                mpz_init(candidate);

                for (index = 0; index < instance_count; index++) {
                    to_mpz(candidate, instances[index].candidate._limbs, params::BITS / 32);
                    gmp_prime = (mpz_probab_prime_p(candidate, 1) != 0);

                    xmp_prime = instances[index].passed;

                    if (gmp_prime != xmp_prime) {
                        printf("MISMATCH AT INDEX: %d\n", index);
                        printf("prime count=%d\n", instances[index].passed);
                        match = false;
                    }
                    if (xmp_prime)
                        total++;
                }
                if (match)
                    printf("All results matched\n");
                printf("%d probable primes found in %d random numbers\n", total, instance_count);
                printf("Based on an approximation of the prime gap, we would expect %0.1f primes\n", ((float)instance_count) * 2 / (0.69315f * params::BITS));
            }
        };
	}
}
#endif