/*******************************************************************************************

 Nexus Earth 2018

 [Scale Indefinitely] BlackJack. http://www.opensource.org/licenses/mit-license.php

*******************************************************************************************/
#pragma once
#ifndef NEXUS_CUDA_HASH_HELPER_CUH
#define NEXUS_CUDA_HASH_HELPER_CUH

#ifdef __INTELLISENSE__
#define __launch_bounds__(x)
#endif

/* Ugly but needed to deal with the hash packing */
#define v32(x,y) ((uint32_t*)(x))[((y) & 1)+2*(((y)/2)*threads+thread)]


/* Make a 64-bit integer from lo and hi 32-bit words. */
__device__ uint64_t MAKE_ULONGLONG(uint32_t LO, uint32_t HI)
{
	return __double_as_longlong(__hiloint2double(HI, LO));
}


/* Extract the Lo Word from a 64-bit Type. */
__device__ uint32_t LOWORD(uint64_t x)
{
	return (uint32_t)__double2hiint(__longlong_as_double(x));
}


/* Extract the Hi Word from a 64-bit Type. */
__device__ uint32_t HIWORD(uint64_t x)
{
	return (uint32_t)__double2hiint(__longlong_as_double(x));
}


__device__ void LOHI(uint32_t &lo, uint32_t &hi, uint64_t x)
{
	asm("{\n\t"
		"mov.b64 {%0,%1},%2; \n\t"
		"}"
		: "=r"(lo), "=r"(hi) : "l"(x));
}



#if __CUDA_ARCH__ < 350

/* Kepler (Compute 3.0) */
#define SPH_ROTL32(x, n) SPH_T32(((x) << (n)) | ((x) >> (32 - (n))))
#define SPH_ROTR32(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#else
/* Kepler (Compute 3.5) */
#define SPH_ROTL32(x, n) __funnelshift_l( (x), (x), (n) )
#define SPH_ROTR32(x, n) __funnelshift_r( (x), (x), (n) )
#endif

/* Replace the Hi Word in a 64-bit Type */
__device__ uint64_t oREPLACE_HIWORD(const uint64_t &x, const uint32_t &y)
{
	return (x & 0xFFFFFFFFULL) | (((uint64_t)y) << 32ULL);
}


__host__ uint2 lohi_host(const uint64_t x)
{
	uint2 res;
	res.x = (uint32_t)(x & 0xFFFFFFFFULL);
	res.y = (uint32_t)(x >> 32);
	return res;
}


__host__ uint64_t make64_host(const uint2 x)
{
	return (uint64_t)x.x | (((uint64_t)x.y) << 32);
}


__device__ uint64_t REPLACE_HIWORD(uint64_t x, uint32_t y)
{
	return (x & 0xFFFFFFFFULL) | (((uint64_t)y) << 32U);
}


__device__ uint64_t REPLACE_LOWORD(uint64_t x, uint32_t y)
{
	return (x & 0xFFFFFFFF00000000ULL) | ((uint64_t)y);
}


__forceinline__ __device__ uint64_t sph_t64(uint64_t x)
{
	uint64_t result;
	asm("{\n\t"
		"and.b64 %0,%1,0xFFFFFFFFFFFFFFFF;\n\t"
		"}\n\t"
		: "=l"(result) : "l"(x));
	return result;
}


__forceinline__ __device__ uint32_t sph_t32(uint32_t x)
{
	uint32_t result;
	asm("{\n\t"
		"and.b32 %0,%1,0xFFFFFFFF;\n\t"
		"}\n\t"
		: "=r"(result) : "r"(x));
	return result;
}


__forceinline__ __device__ uint64_t shr_t64(uint64_t x, uint32_t n)
{
	uint64_t result;
	asm("{\n\t"
		"shr.b64 %0,%1,%2;\n\t"
		"}\n\t"
		: "=l"(result) : "l"(x), "r"(n));
	return result;
}


__forceinline__ __device__ uint64_t shl_t64(uint64_t x, uint32_t n)
{
	uint64_t result;
	asm("{\n\t"
		"shl.b64 %0,%1,%2;\n\t"
		"}\n\t"
		: "=l"(result) : "l"(x), "r"(n));
	return result;
}


__forceinline__ __device__ uint32_t shr_t32(uint32_t x, uint32_t n)
{
	uint32_t result;
	asm("{\n\t"
		"shr.b32 %0,%1,%2;\n\t"
		"}\n\t"
		: "=r"(result) : "r"(x), "r"(n));
	return result;
}


__forceinline__ __device__ uint32_t shl_t32(uint32_t x, uint32_t n)
{
	uint32_t result;
	asm("{\n\t"
		"shl.b32 %0,%1,%2;\n\t"
		"}\n\t"
		: "=r"(result) : "r"(x), "r"(n));
	return result;
}


__forceinline__ __device__ uint64_t mul(uint64_t a, uint64_t b)
{
	uint64_t result;
	asm("{\n\t"
		"mul.lo.u64 %0,%1,%2; \n\t"
		"}\n\t"
		: "=l"(result) : "l"(a), "l"(b));
	return result;
}

///uint2 method

#if  __CUDA_ARCH__ >= 350
__inline__ __device__ uint2 ROR2(const uint2 a, const int offset)
{
	uint2 result;
	if (offset < 32)
	{
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.x), "r"(a.y), "r"(offset));
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.y), "r"(a.x), "r"(offset));
	}
	else
	{
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.y), "r"(a.x), "r"(offset));
		asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(offset));
	}
	return result;
}
#else

__inline__ __device__ uint2 ROR2(const uint2 v, const int n)
{
	uint2 result;
	result.x = (((v.x) >> (n)) | ((v.x) << (64 - (n))));
	result.y = (((v.y) >> (n)) | ((v.y) << (64 - (n))));
	return result;
}
#endif


#if  __CUDA_ARCH__ >= 350
__inline__ __device__ uint2 ROL2(const uint2 a, const int offset)
{
	uint2 result;
	if (offset >= 32)
	{
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.x), "r"(a.y), "r"(offset));
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.y), "r"(a.x), "r"(offset));
	}
	else
	{
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.y), "r"(a.x), "r"(offset));
		asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(offset));
	}
	return result;
}
#else
__inline__ __device__ uint2 ROL2(const uint2 v, const int n)
{
	uint2 result;
	result.x = (((v.x) << (n)) | ((v.x) >> (64 - (n))));
	result.y = (((v.y) << (n)) | ((v.y) >> (64 - (n))));
	return result;
}
#endif

__forceinline__ __device__ uint64_t devectorize(uint2 v)
{
	return MAKE_ULONGLONG(v.x, v.y);
}

__forceinline__ __device__ uint2 vectorize(uint64_t v)
{
	uint2 result;
	LOHI(result.x, result.y, v);
	return result;
}

__forceinline__ __device__ uint2 xor2(uint2 a, uint2 b)
{
	uint2 result;
	asm("xor.b32 %0, %1, %2;" : "=r"(result.x) : "r"(a.x), "r"(b.x));
	asm("xor.b32 %0, %1, %2;" : "=r"(result.y) : "r"(a.y), "r"(b.y));
	return result;
}

__forceinline__ __device__ uint2 operator^(uint2 a, uint2 b)
{
	return make_uint2(a.x ^ b.x, a.y ^ b.y);
}

__forceinline__ __device__ uint2 operator&(uint2 a, uint2 b)
{
	return make_uint2(a.x & b.x, a.y & b.y);
}

__forceinline__ __device__ uint2 operator|(uint2 a, uint2 b)
{
	return make_uint2(a.x | b.x, a.y | b.y);
}

__forceinline__ __device__ uint2 operator~(uint2 a)
{
	return make_uint2(~a.x, ~a.y);
}

__forceinline__ __device__  void operator^= (uint2 &a, uint2 b)
{
	a = a ^ b;
}

__device__ __forceinline__ uint2 operator+(const uint2 a, const uint2 b)
{
#if defined(__CUDA_ARCH__) && CUDA_VERSION < 7000
	uint2 result;
	asm("{\n\t"
		"add.cc.u32 %0,%2,%4; \n\t"
		"addc.u32 %1,%3,%5;   \n\t"
		"}\n\t"
		: "=r"(result.x), "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(b.x), "r"(b.y));
	return result;
#else
	return vectorize(devectorize(a) + devectorize(b));
#endif
}

__forceinline__ __device__ void operator+=(uint2 &a, uint2 b)
{
	a = a + b;
}

__forceinline__ __device__ uint2 operator*(uint2 a, uint2 b)
{ /* basic multiplication between 64bit no carry outside that range (ie mul.lo.b64(a*b))
	(it's what does uint64 "*" operator) */
	uint2 result;
	asm("{\n\t"
		"mul.lo.u32        %0,%2,%4;  \n\t"
		"mul.hi.u32        %1,%2,%4;  \n\t"
		"mad.lo.cc.u32    %1,%3,%4,%1; \n\t"
		"madc.lo.u32      %1,%3,%5,%1; \n\t"
		"}\n\t"
		: "=r"(result.x), "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(b.x), "r"(b.y));
	return result;
}
#if  __CUDA_ARCH__ >= 350

__forceinline__ __device__ uint2 shiftl2(uint2 a, int offset)
{
	uint2 result;
	if (offset<32)
	{
		asm("{\n\t"
			"shf.l.clamp.b32 %1,%2,%3,%4; \n\t"
			"shl.b32 %0,%2,%4; \n\t"
			"}\n\t"
			: "=r"(result.x), "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(offset));
	}
	else
	{
		asm("{\n\t"
			"shf.l.clamp.b32 %1,%2,%3,%4; \n\t"
			"shl.b32 %0,%2,%4; \n\t"
			"}\n\t"
			: "=r"(result.x), "=r"(result.y) : "r"(a.y), "r"(a.x), "r"(offset));
	}
	return result;
}

__forceinline__ __device__ uint2 shiftr2(uint2 a, int offset)
{
	uint2 result;
	if (offset<32)
	{
		asm("{\n\t"
			"shf.r.clamp.b32 %0,%2,%3,%4; \n\t"
			"shr.b32 %1,%3,%4; \n\t"
			"}\n\t"
			: "=r"(result.x), "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(offset));
	}
	else
	{
		asm("{\n\t"
			"shf.l.clamp.b32 %0,%2,%3,%4; \n\t"
			"shl.b32 %1,%3,%4; \n\t"
			"}\n\t"
			: "=r"(result.x), "=r"(result.y) : "r"(a.y), "r"(a.x), "r"(offset));
	}
	return result;
}
#else

__forceinline__ __device__ uint2 shiftl2(uint2 a, int offset)
{
	uint2 result;
	asm("{\n\t"
		".reg .b64 u,v; \n\t"
		"mov.b64 v,{%2,%3}; \n\t"
		"shl.b64 u,v,%4; \n\t"
		"mov.b64 {%0,%1},v;  \n\t"
		"}\n\t"
		: "=r"(result.x), "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(offset));
	return result;
}

__forceinline__ __device__ uint2 shiftr2(uint2 a, int offset)
{
	uint2 result;
	asm("{\n\t"
		".reg .b64 u,v; \n\t"
		"mov.b64 v,{%2,%3}; \n\t"
		"shr.b64 u,v,%4; \n\t"
		"mov.b64 {%0,%1},v;  \n\t"
		"}\n\t"
		: "=r"(result.x), "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(offset));
	return result;
}
#endif


#endif
