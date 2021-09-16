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
#include "fermat_test.cuh"
#include "fermat.cuh"
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

        
        template<class params>
        __global__ void kernel_fermat(cgbn_error_report_t* report, typename fermat_t<params>::instance_t* instances, uint32_t instance_count) {
            int32_t instance = (blockIdx.x * blockDim.x + threadIdx.x) / params::TPI;

            if (instance >= instance_count)
                return;

            typedef fermat_t<params> local_fermat_t;

            local_fermat_t                     fermat_test(cgbn_report_monitor, report, instance);
            typename local_fermat_t::bn_t      candidate;
            bool                               passed;

            cgbn_load(fermat_test._env, candidate, &(instances[instance].candidate));

            passed = fermat_test.fermat(candidate);

            instances[instance].passed = passed;
        }


        template<class params>
        void run_test(mpz_t base_big_int, uint64_t offsets[], uint32_t instance_count, uint8_t results[], int device) {
            typedef typename fermat_t<params>::instance_t instance_t;

            instance_t* instances, * gpuInstances;
            cgbn_error_report_t* report;
            int32_t              TPB = (params::TPB == 0) ? 128 : params::TPB;
            int32_t              TPI = params::TPI, IPB = TPB / TPI;


            //printf("Genereating instances ...\n");
            instances = fermat_t<params>::generate_instances(base_big_int, offsets, instance_count);
            //printf("Copying instances to the GPU ...\n");
            CUDA_CHECK(cudaSetDevice(device));
            CUDA_CHECK(cudaMalloc((void**)&gpuInstances, sizeof(instance_t) * instance_count));
            CUDA_CHECK(cudaMemcpy(gpuInstances, instances, sizeof(instance_t) * instance_count, cudaMemcpyHostToDevice));

            // create a cgbn_error_report for CGBN to report back errors
            //CUDA_CHECK(cgbn_error_report_alloc(&report)); 

            //printf("Running GPU kernel ...\n");
            // launch kernel with blocks=ceil(instance_count/IPB) and threads=TPB
            kernel_fermat<params> << <(instance_count + IPB - 1) / IPB, TPB >> > (report, gpuInstances, instance_count);

            // error report uses managed memory, so we sync the device (or stream) and check for cgbn errors
            CUDA_CHECK(cudaDeviceSynchronize());
            //CGBN_CHECK(report);

            // copy the instances back from gpuMemory
            //printf("Copying results back to CPU ...\n");
            CUDA_CHECK(cudaMemcpy(instances, gpuInstances, sizeof(instance_t) * instance_count, cudaMemcpyDeviceToHost));

            //printf("Verifying the results ...\n");
            //fermat_t<params>::verify_results(instances, instance_count);

            for (auto i = 0; i < instance_count; i++)
            {
                results[i] = instances[i].passed ? 1 : 0;
            }

            // clean up

            free(instances);
            CUDA_CHECK(cudaFree(gpuInstances));
            //CUDA_CHECK(cgbn_error_report_free(report));
        }


        void CudaPrimalityTest::fermat_run(mpz_t base_big_int, uint64_t offsets[], uint32_t offset_count, uint8_t results[], int device)
        {
            //printf("Testing %i prime candidates\n", offset_count);
            using params = fermat_params_t<8, 1024, 5>;
            
            
            run_test<params>(base_big_int, offsets, offset_count, results, device);
        }

        //allocate device memory for gpu fermat testing.  we used a fixed maximum batch size and allocate device memory once at the beginning. 
        //void CudaPrimalityTest::fermat_init(uint32_t batch_size, int device)
        //{
        //    
        //    instance_t* instances;

        //    m_device = device;

        //    CUDA_CHECK(cudaSetDevice(device));
        //    CUDA_CHECK(cudaMalloc((void**)&d_instances, sizeof(instance_t) * batch_size));

        //    // create a cgbn_error_report for CGBN to report back errors
        //    CUDA_CHECK(cgbn_error_report_alloc(&d_report)); 

        //    
        //}

        //void CudaPrimalityTest::fermat_free()
        //{
        //    CUDA_CHECK(cudaFree(d_instances));
        //    CUDA_CHECK(cgbn_error_report_free(d_report));
        //}

        void CudaPrimalityTest::set_base_int(mpz_t base_big_int)
        {
           
        }

    }
}
