#ifndef NEXUSMINER_GPU_CUMP_CUH
#define NEXUSMINER_GPU_CUMP_CUH

//cuda unsigned big integer class
//The size of the integer in bits is selectable via template

#include "../gpu_helper.hpp"
#include <stdint.h>
#include <gmp.h>
#include "fermat_utils.cuh"

namespace nexusminer {
	namespace gpu {

		template<int BITS> class Cump
		{
			static_assert(BITS > 0, "The big int must have at least one bit.");
		public:
			static constexpr int BITS_PER_WORD = 32;
			//LIMBS is the number of machine words used to store the big int
			//we allocate one extra word to handle overflow and normalization in division algorithm
			static constexpr int EXTRA_WORDS = 1;
			static constexpr int HIGH_WORD = (BITS + BITS_PER_WORD - 1) / BITS_PER_WORD - 1; //round up 
			static constexpr int LIMBS = HIGH_WORD + 1 + EXTRA_WORDS;  //extra word(s) for overflow
			
			
			__host__ __device__ Cump(uint32_t);
			__host__ __device__ Cump(int);
			__host__ __device__ Cump();
			__device__ Cump add(const Cump&) const;
			__device__ Cump add(int) const;
			__device__ Cump add(uint32_t) const;
			__device__ Cump add(uint64_t) const;
#if defined(GPU_CUDA_ENABLED)
			__device__ Cump add_ptx(const Cump&) const;
			__device__ void increment_ptx(const Cump&);
#endif
			__host__ __device__ Cump sub(const Cump&) const;
			__host__ __device__ Cump sub(int) const;
			__host__ __device__ Cump sub(uint32_t) const;
			__device__ void increment(const Cump&);
			__device__ void increment(int);
			__device__ void increment(uint32_t);
			__device__ void operator += (const Cump&);
			__device__ void operator += (int);
			__device__ void operator += (uint32_t);
			__device__ void operator += (uint64_t);
			__host__ __device__ void decrement(const Cump&);
			__host__ __device__ void decrement(int);
			__host__ __device__ void decrement(uint32_t);
			__host__ __device__ void operator -= (const Cump&);
			__host__ __device__ void operator -= (int);
			__host__ __device__ void operator -= (uint32_t);
			__host__ __device__ Cump operator << (int) const;
			__host__ __device__ void operator <<= (int);
			__host__ __device__ Cump operator >> (int) const;
			__host__ __device__ void operator >>= (int);
			__host__ __device__ Cump operator ~ () const;
			__host__ __device__ Cump multiply(uint32_t) const;
			__host__ __device__ Cump multiply(const Cump&) const;
			//__device__ Cump multiply_ptx(uint32_t x) const;


			__host__ __device__ void operator *= (uint32_t);
			__host__ __device__ void operator *= (const Cump&);



			__host__ __device__ Cump modinv(const Cump&) const;
			//__device__ void divide(const Cump& divisor, Cump& quotient, Cump& remainder) const;
			__device__ Cump R_mod_m() const;
			//__device__ void remainder(const Cump& divisor, Cump& remainder) const;

			__host__ __device__ int compare(const Cump&) const;

			__host__ __device__ void to_cstr(char* s) const;
			__host__ void to_mpz(mpz_t r);
			__host__ void from_mpz(mpz_t s);

			//the least significant word is stored in array element 0
			uint32_t m_limbs[LIMBS];  
			
		};

		
		template<int BITS> __device__ Cump<BITS> operator + (const Cump<BITS>& lhs, const Cump<BITS>& rhs);
		template<int BITS> __device__ Cump<BITS> operator + (const Cump<BITS>& lhs, int rhs);
		template<int BITS> __device__ Cump<BITS> operator + (const Cump<BITS>& lhs, uint32_t rhs);
		template<int BITS> __device__ Cump<BITS> operator + (const Cump<BITS>& lhs, uint64_t rhs);
		template<int BITS> __host__ __device__ Cump<BITS> operator - (const Cump<BITS>& lhs, const Cump<BITS>& rhs);
		template<int BITS> __host__ __device__ Cump<BITS> operator - (const Cump<BITS>& lhs, int rhs);
		template<int BITS> __host__ __device__ Cump<BITS> operator - (const Cump<BITS>& lhs, uint32_t rhs);
		template<int BITS> __host__ __device__ Cump<BITS> operator * (const Cump<BITS>& lhs, uint32_t);
		template<int BITS> __host__ __device__ Cump<BITS> operator * (const Cump<BITS>& lhs, const Cump<BITS>& rhs);
		//template<int BITS> __device__ Cump<BITS> operator / (const Cump<BITS>& lhs, const Cump<BITS>& rhs);
		//template<int BITS> __device__ Cump<BITS> operator % (const Cump<BITS>& lhs, const Cump<BITS>& rhs);
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

		template<int BITS> __device__ Cump<BITS> montgomery_multiply(const Cump<BITS>& x, const Cump<BITS>& y, const Cump<BITS>& m, uint32_t m_primed);
		template<int BITS> __device__ void montgomery_square(Cump<BITS>& x, const Cump<BITS>& m, uint32_t m_primed);
		template<int BITS> __device__ void montgomery_square_2(Cump<BITS>& x, const Cump<BITS>& m, uint32_t m_primed);
		template<int BITS> __device__ void montgomery_square_3(Cump<BITS>& x, const Cump<BITS>& m, uint32_t m_primed);
		template<int BITS> __device__ void montgomery_reduce(Cump<BITS>& x, const Cump<BITS>& m, uint32_t m_primed);
		//template<int BITS> __device__ void montgomery_reduce(Cump<BITS>& A, uint32_t x, const Cump<BITS>& m, uint32_t m_primed);
		template<int BITS> __device__ bool powm_2(const Cump<BITS>& m, uint64_t offset);
		template<int BITS> __device__ Cump<BITS> double_and_reduce(const Cump<BITS>& x, const Cump<BITS>& m, int shift);

	}
}
#include "cump.cu"
#endif
