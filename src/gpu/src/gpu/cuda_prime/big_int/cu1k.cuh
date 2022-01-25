#ifndef NEXUSMINER_GPU_CU1K_CUH
#define NEXUSMINER_GPU_CU1K_CUH

//1024 bit cuda unsigned int
//The class stores the 1024 bit unsigned int in 32 x 32-bit unsigned ints

#include <stdint.h>
#include <gmp.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace nexusminer {
	namespace gpu {

		class Cu1k
		{
		public:
			static const int LIMBS = 33; //we allocate one extra word to handle overflow and normalization in division algorithm
			static const int BITS_PER_WORD = 32;
			__host__ __device__ Cu1k(uint32_t);
			__host__ __device__ Cu1k();
			__host__ __device__ Cu1k add(const Cu1k&) const;
			__host__ __device__ Cu1k sub(const Cu1k&) const;
			__host__ __device__ void increment(const Cu1k&);
			__host__ __device__ void operator += (const Cu1k&);
			__host__ __device__ void decrement(const Cu1k&);
			__host__ __device__ void operator -= (const Cu1k&);
			__device__ void divide(const Cu1k& divisor, Cu1k& quotient, Cu1k& remainder) const;
			__host__ __device__ Cu1k operator << (int) const;
			__host__ __device__ Cu1k operator >> (int) const;


			__host__ __device__ int compare(const Cu1k&) const;

			__host__ __device__ void to_cstr(char* s);
			__host__ void to_mpz(mpz_t r);
			__host__ void from_mpz(mpz_t s);

			//the least significant word is stored in array element 0
			uint32_t m_limbs[LIMBS];  
			//for negative nunmbers sign is -1.  Positive numbers the sign could be 0 or 1. 
			//int32_t m_sign;
			//The final carry after an add or subtraction. not valid for all operations. 
			//uint32_t m_carry;
		};

		__host__ __device__ Cu1k operator + (const Cu1k& lhs, const Cu1k& rhs);
		__host__ __device__ Cu1k operator - (const Cu1k& lhs, const Cu1k& rhs);
		__device__ Cu1k operator / (const Cu1k& lhs, const Cu1k& rhs);
		__device__ Cu1k operator % (const Cu1k& lhs, const Cu1k& rhs);
		__host__ __device__ bool operator > (const Cu1k& lhs, const Cu1k& rhs);
		__host__ __device__ bool operator < (const Cu1k& lhs, const Cu1k& rhs);
		__host__ __device__ bool operator == (const Cu1k& lhs, const Cu1k& rhs);
		__host__ __device__ bool operator >= (const Cu1k& lhs, const Cu1k& rhs);
		__host__ __device__ bool operator <= (const Cu1k& lhs, const Cu1k& rhs);
		__host__ __device__ bool operator != (const Cu1k& lhs, const Cu1k& rhs);






		//for debug
		__host__ __device__ void reverse(char str[], int length);
		__host__ __device__ char* itoa(unsigned int num, char* str);
	}
}

#endif
