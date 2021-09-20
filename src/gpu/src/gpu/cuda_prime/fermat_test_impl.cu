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

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <gmp.h>
#include "fermat_test_impl.cuh"
#include "cgbn/cgbn.h"
#include "cgbn/utility/support.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "nppdefs.h"

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


            __device__ __forceinline__ fermat_t::fermat_t(cgbn_monitor_t monitor, cgbn_error_report_t* report, int32_t instance) : _context(monitor, report, (uint32_t)instance), _env(_context), _instance(instance) {
            }

            __device__ __forceinline__ void fermat_t::powm(bn_t& x, const bn_t& power, const bn_t& modulus) {
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
                position = fermat_params_t::BITS - cgbn_clz(_env, power);

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

            __device__ __forceinline__ bool fermat_t::fermat(const bn_t& candidate) {

                bn_t      x, power;

                cgbn_sub_ui32(_env, power, candidate, 1);
                cgbn_set_ui32(_env, x, 2);
                powm(x, power, candidate);
                return cgbn_equals_ui32(_env, x, 1);
            }

            __host__ fermat_t::instance_t* fermat_t::generate_instances(mpz_t base_big_int, uint64_t offsets[], uint32_t count) {
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
                    from_mpz(p, instances[index].candidate._limbs, fermat_params_t::BITS / 32);
                    instances[index].passed = 0;
                }
                mpz_clear(p);
                mpz_clear(o);

                return instances;
            }

            //convert 64 bit offsets to equivalent cgbn mem types
            __host__ void fermat_t::offsets_to_cgbn(uint64_t offsets[], uint32_t count, cgbn_mem_t<64> cgbn_offsets[]) {
                //cgbn_mem_t<64>* instances = (cgbn_mem_t<64>*)malloc(sizeof(cgbn_mem_t<64>) * count);
                mpz_t o;
                //mpz_init(p);
                mpz_init(o);

                for (int index = 0; index < count; index++) {
                    //mpz doesn't natively handle 64 bit ints.  need to use import function.
                    uint64_t off = offsets[index];
                    mpz_import(o, 1, 1, sizeof(off), 0, 0, &off);
                    //mpz_add(p, base_big_int, o);
                    from_mpz(o, cgbn_offsets[index]._limbs, 64/32);
                    //instances[index].passed = 0;
                }
                //mpz_clear(p);
                mpz_clear(o);

            }

            __host__ void fermat_t::verify_results(instance_t* instances, uint32_t instance_count) {
                int   index, total = 0;
                mpz_t candidate;
                bool  gmp_prime, xmp_prime, match = true;

                mpz_init(candidate);

                for (index = 0; index < instance_count; index++) {
                    to_mpz(candidate, instances[index].candidate._limbs, fermat_params_t::BITS / 32);
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
                printf("Based on an approximation of the prime gap, we would expect %0.1f primes\n", ((float)instance_count) * 2 / (0.69315f * fermat_params_t::BITS));
            }


        __global__ void kernel_fermat(cgbn_error_report_t* report, cgbn_mem_t<64>* offsets, uint64_t* offset_count, 
            cgbn_mem_t<fermat_params_t::BITS>* base_int, uint8_t* results, uint64_t* test_count, uint64_t* pass_count) {
            int64_t instance = (blockIdx.x * blockDim.x + threadIdx.x)/ fermat_params_t::TPI;

            if (instance >= *offset_count)
                return;

            fermat_t            fermat_test(cgbn_report_monitor, report, instance);
            fermat_t::bn_t      bn_candidate, bn_base_int, bn_offset;
            bool                passed;

            //convert 64 bit bignum to 1024 bit bignum by padding with zeros.
            cgbn_mem_t<64> offset64 = offsets[instance];
            cgbn_mem_t<1024> offset;
            offset._limbs[0] = offset64._limbs[0];
            offset._limbs[1] = offset64._limbs[1];
            for (int i = 2; i<1024/32 - 2; i++)
                offset._limbs[i] = 0;
           
            //add the offset to the base int
            cgbn_load(fermat_test._env, bn_offset, &offset);
            cgbn_load(fermat_test._env, bn_base_int, base_int);
            cgbn_add(fermat_test._env, bn_candidate, bn_offset, bn_base_int);
            passed = fermat_test.fermat(bn_candidate);

            if (threadIdx.x % fermat_params_t::TPI == 0)
            {
                if (passed)
                {
                    atomicAdd(pass_count, 1);
                }
                results[instance] = passed ? 1 : 0;
                atomicAdd(test_count, 1);
            }
            

        }

        void Cuda_fermat_test_impl::fermat_run()
        {

            int32_t threads_per_block = (fermat_params_t::TPB == 0) ? 64 : fermat_params_t::TPB;
            int32_t threads_per_instance = fermat_params_t::TPI;
            int32_t instances_per_block = threads_per_block / threads_per_instance;

            int blocks = (m_offset_count + instances_per_block - 1) / instances_per_block;

            kernel_fermat << <blocks, threads_per_block>> > (d_report, d_offsets, d_offset_count, d_base_int,
                d_results, d_fermat_test_count, d_fermat_pass_count);
            CUDA_CHECK(cudaDeviceSynchronize());


        }

        //allocate device memory for gpu fermat testing.  we use a fixed maximum batch size and allocate device memory once at the beginning. 
        void Cuda_fermat_test_impl::fermat_init(uint64_t batch_size, int device)
        {
            
            m_device = device;

            CUDA_CHECK(cudaSetDevice(device));
            CUDA_CHECK(cudaMalloc(&d_instances, sizeof(*d_instances) * batch_size));
            CUDA_CHECK(cudaMalloc(&d_base_int, sizeof(*d_base_int)));
            CUDA_CHECK(cudaMalloc(&d_offsets, sizeof(*d_offsets) * batch_size));
            CUDA_CHECK(cudaMalloc(&d_results, sizeof(*d_results) * batch_size));
            CUDA_CHECK(cudaMalloc(&d_offset_count, sizeof(*d_offset_count)));
            CUDA_CHECK(cudaMalloc(&d_fermat_test_count, sizeof(*d_fermat_test_count)));
            CUDA_CHECK(cudaMalloc(&d_fermat_pass_count, sizeof(*d_fermat_pass_count)));
            reset_stats();

            // allocate memory for a cgbn_error_report for CGBN to report back errors. 
            CUDA_CHECK(cgbn_error_report_alloc(&d_report)); 
        }

        void Cuda_fermat_test_impl::fermat_free()
        {
            CUDA_CHECK(cudaFree(d_instances));
            CUDA_CHECK(cudaFree(d_base_int));
            CUDA_CHECK(cudaFree(d_offsets));
            CUDA_CHECK(cudaFree(d_results));
            CUDA_CHECK(cudaFree(d_offset_count));
            CUDA_CHECK(cudaFree(d_fermat_test_count));
            CUDA_CHECK(cudaFree(d_fermat_pass_count));

            CUDA_CHECK(cgbn_error_report_free(d_report));
        }

        void Cuda_fermat_test_impl::set_base_int(mpz_t base_big_int)
        {
            cgbn_mem_t<fermat_params_t::BITS> cgbn_base_int;
            from_mpz(base_big_int, cgbn_base_int._limbs, fermat_params_t::BITS / 32);
            CUDA_CHECK(cudaMemcpy(d_base_int, &cgbn_base_int, sizeof(cgbn_base_int), cudaMemcpyHostToDevice));
        }

        void Cuda_fermat_test_impl::set_offsets(uint64_t offsets[], uint64_t offset_count)
        {
            cgbn_mem_t<64>* cgbn_offsets = new cgbn_mem_t<64>[offset_count];
            fermat_t::offsets_to_cgbn(offsets, offset_count, cgbn_offsets);
            CUDA_CHECK(cudaMemcpy(d_offsets, cgbn_offsets, sizeof(cgbn_offsets) *  offset_count, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_offset_count, &offset_count, sizeof(offset_count), cudaMemcpyHostToDevice));
            m_offset_count = offset_count;
            delete[] cgbn_offsets;
        }

        void Cuda_fermat_test_impl::get_results(uint8_t results[])
        {
            CUDA_CHECK(cudaMemcpy(results, d_results, sizeof(uint8_t) * m_offset_count, cudaMemcpyDeviceToHost));
        }

        void Cuda_fermat_test_impl::get_stats(uint64_t& fermat_tests, uint64_t& fermat_passes)
        {
            CUDA_CHECK(cudaMemcpy(&fermat_tests, d_fermat_test_count, sizeof(*d_fermat_test_count), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&fermat_passes, d_fermat_pass_count, sizeof(*d_fermat_pass_count), cudaMemcpyDeviceToHost));
        }

        void Cuda_fermat_test_impl::reset_stats()
        {
            CUDA_CHECK(cudaMemset(d_fermat_test_count, 0, sizeof(*d_fermat_test_count)));
            CUDA_CHECK(cudaMemset(d_fermat_pass_count, 0, sizeof(*d_fermat_pass_count)));
        }

    }
}
