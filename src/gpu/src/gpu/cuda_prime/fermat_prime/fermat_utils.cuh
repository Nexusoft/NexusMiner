#ifndef NEXUSMINER_GPU_FERMAT_UTILS_CUH
#define NEXUSMINER_GPU_FERMAT_UTILS_CUH
//math functions used with fermat testing

#include <stdint.h>
#include "hip/hip_runtime.h"

namespace nexusminer {
	namespace gpu {

		
		__device__ uint32_t mod_inverse_32(uint32_t d);
		//for debug printing
		__device__ void reverse(char str[], int length);
		__device__ char* itoa(unsigned int num, char* str);
		
	}
}

#endif
