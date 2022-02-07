#ifndef NEXUSMINER_GPU_FERMAT_UTILS_CUH
#define NEXUSMINER_GPU_FERMAT_UTILS_CUH
//math functions used with fermat testing

#include <stdint.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace nexusminer {
	namespace gpu {

		
		__host__ __device__ uint32_t mod_inverse_32(uint32_t d);
		//for debug printing
		__host__ __device__ void reverse(char str[], int length);
		__host__ __device__ char* itoa(unsigned int num, char* str);
		
	}
}

#endif