#ifndef NEXUSMINER_GPU_CUMP_CUH
#define NEXUSMINER_GPU_CUMP_CUH

//cuda unsigned big integer class
//The size of the integer in bits is selectable via template

#include <stdint.h>
#include <gmp.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "fermat_utils.cuh"

namespace nexusminer {
	namespace gpu {

		template<int BITS> class Cump
		{
			static_assert(BITS > 0, "The big int must have at least one bit.");
		public:
			static const int BITS_PER_WORD = 32;
			//LIMBS is the number of machine words used to store the big int
			//we allocate one extra word to handle overflow and normalization in division algorithm
			static const int LIMBS = 1 + (BITS + BITS_PER_WORD - 1)/BITS_PER_WORD; //round up and add 1
			
			__host__ __device__ Cump(uint32_t);
			__host__ __device__ Cump(int);
			__host__ __device__ Cump();
			__host__ __device__ Cump add(const Cump&) const;
			__host__ __device__ Cump sub(const Cump&) const;
			__host__ __device__ void increment(const Cump&);
			__host__ __device__ void operator += (const Cump&);
			__host__ __device__ void decrement(const Cump&);
			__host__ __device__ void operator -= (const Cump&);
			__device__ void divide(const Cump& divisor, Cump& quotient, Cump& remainder) const;
			__host__ __device__ Cump operator << (int) const;
			__host__ __device__ void operator <<= (int);
			__host__ __device__ Cump operator >> (int) const;
			__host__ __device__ void operator >>= (int);
			__host__ __device__ Cump operator ~ () const;
			__host__ __device__ Cump modinv(const Cump&) const;
			__device__ void R_mod_m(Cump& R) const;
			__device__ void remainder(const Cump& divisor, Cump& remainder) const;



			__host__ __device__ int compare(const Cump&) const;

			__host__ __device__ void to_cstr(char* s);
			__host__ void to_mpz(mpz_t r);
			__host__ void from_mpz(mpz_t s);

			//the least significant word is stored in array element 0
			uint32_t m_limbs[LIMBS];  
			
		};

		
		template<int BITS> __host__ __device__ Cump<BITS> operator + (const Cump<BITS>& lhs, const Cump<BITS>& rhs);
		template<int BITS> __host__ __device__ Cump<BITS> operator - (const Cump<BITS>& lhs, const Cump<BITS>& rhs);
		template<int BITS> __device__ Cump<BITS> operator / (const Cump<BITS>& lhs, const Cump<BITS>& rhs);
		template<int BITS> __device__ Cump<BITS> operator % (const Cump<BITS>& lhs, const Cump<BITS>& rhs);
		template<int BITS> __host__ __device__ bool operator > (const Cump<BITS>& lhs, const Cump<BITS>& rhs);
		template<int BITS> __host__ __device__ bool operator > (const Cump<BITS>& lhs, int rhs);
		template<int BITS> __host__ __device__ bool operator < (const Cump<BITS>& lhs, const Cump<BITS>& rhs);
		template<int BITS> __host__ __device__ bool operator < (const Cump<BITS>& lhs, int rhs);
		template<int BITS> __host__ __device__ bool operator == (const Cump<BITS>& lhs, const Cump<BITS>& rhs);
		template<int BITS> __host__ __device__ bool operator == (const Cump<BITS>& lhs, int rhs);
		template<int BITS> __host__ __device__ bool operator >= (const Cump<BITS>& lhs, const Cump<BITS>& rhs);
		template<int BITS> __host__ __device__ bool operator >= (const Cump<BITS>& lhs, int rhs);
		template<int BITS> __host__ __device__ bool operator <= (const Cump<BITS>& lhs, const Cump<BITS>& rhs);
		template<int BITS> __host__ __device__ bool operator <= (const Cump<BITS>& lhs, int rhs);
		template<int BITS> __host__ __device__ bool operator != (const Cump<BITS>& lhs, const Cump<BITS>& rhs);
		template<int BITS> __host__ __device__ bool operator != (const Cump<BITS>& lhs, int rhs);



		

	}
}
#include "cump.cu"
#endif
