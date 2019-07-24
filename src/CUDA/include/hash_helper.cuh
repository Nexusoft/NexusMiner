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


/* Make a 64-bit integer from lo and hi 32-bit words. */
__device__ uint64_t MAKE_ULONGLONG(uint32_t LO, uint32_t HI)
{
	return __double_as_longlong(__hiloint2double(HI, LO));
}



__device__ void LOHI(uint32_t &lo, uint32_t &hi, uint64_t x)
{
	asm("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "l"(x));
}


__host__ uint2 lohi_host(const uint64_t x)
{
	uint2 res;
	res.x = (uint32_t)(x & 0xFFFFFFFFULL);
	res.y = (uint32_t)(x >> 32);
	return res;
}


///uint2 method

__inline__ __device__
uint2 ROR2_0(const uint2 a, const int offset)
{
	uint2 result;

	asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.x), "r"(a.y), "r"(offset));
	asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.y), "r"(a.x), "r"(offset));

	return result;
}

__inline__ __device__
uint2 ROR2_1(const uint2 a, const int offset)
{
	uint2 result;
	asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.y), "r"(a.x), "r"(offset));
	asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(offset));
	return result;
}

__inline__ __device__
uint2 ROL2_0(const uint2 a, const int offset)
{
	uint2 result;
    asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.y), "r"(a.x), "r"(offset));
	asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.x), "r"(a.y), "r"(offset));
	return result;
}

__inline__ __device__
uint2 ROL2_1(const uint2 a, const int offset)
{
	uint2 result;
    asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.x) : "r"(a.x), "r"(a.y), "r"(offset));
	asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(result.y) : "r"(a.y), "r"(a.x), "r"(offset));
	return result;
}

#if  __CUDA_ARCH__ >= 350
__inline__ __device__
uint2 ROR2(const uint2 a, const int offset)
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

__inline__ __device__
uint2 ROR2(const uint2 v, const int n)
{
	uint2 result;
	result.x = (((v.x) >> (n)) | ((v.x) << (64 - (n))));
	result.y = (((v.y) >> (n)) | ((v.y) << (64 - (n))));
	return result;
}
#endif


#if  __CUDA_ARCH__ >= 350
__inline__ __device__
uint2 ROL2(const uint2 a, const int offset)
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
__inline__ __device__
uint2 ROL2(const uint2 v, const int n)
{
	uint2 result;
	result.x = (((v.x) << (n)) | ((v.x) >> (64 - (n))));
	result.y = (((v.y) << (n)) | ((v.y) >> (64 - (n))));
	return result;
}
#endif

__forceinline__ __device__
uint64_t devectorize(uint2 v)
{
	return MAKE_ULONGLONG(v.x, v.y);
}

__forceinline__ __device__
uint2 vectorize(uint64_t v)
{
	uint2 result;
	LOHI(result.x, result.y, v);
	return result;
}

__forceinline__ __device__
uint2 xor2(uint2 a, uint2 b)
{
	uint2 result;
	asm("xor.b32 %0, %1, %2;" : "=r"(result.x) : "r"(a.x), "r"(b.x));
	asm("xor.b32 %0, %1, %2;" : "=r"(result.y) : "r"(a.y), "r"(b.y));
	return result;
}

__forceinline__ __device__
uint2 operator^(uint2 a, uint2 b)
{
	return make_uint2(a.x ^ b.x, a.y ^ b.y);
}

__forceinline__ __device__
uint2 operator&(uint2 a, uint2 b)
{
	return make_uint2(a.x & b.x, a.y & b.y);
}

__forceinline__ __device__
uint2 operator|(uint2 a, uint2 b)
{
	return make_uint2(a.x | b.x, a.y | b.y);
}

__forceinline__ __device__
uint2 operator~(uint2 a)
{
	return make_uint2(~a.x, ~a.y);
}

__forceinline__ __device__
void operator^= (uint2 &a, uint2 b)
{
	a = a ^ b;
}

__device__ __forceinline__
uint2 operator+(const uint2 a, const uint2 b)
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

__forceinline__ __device__
void operator+=(uint2 &a, uint2 b)
{
	a = a + b;
}

__forceinline__ __device__
uint2 operator*(uint2 a, uint2 b)
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

#endif
