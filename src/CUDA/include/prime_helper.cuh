/*******************************************************************************************

 Nexus Earth 2018

 [Scale Indefinitely] BlackJack. http://www.opensource.org/licenses/mit-license.php

*******************************************************************************************/
#pragma once
#ifndef NEXUS_CUDA_PRIME_HELPER_CUH
#define NEXUS_CUDA_PRIME_HELPER_CUH

/* Create a 64-bit word from 32-bit lo and hi words. */
__device__ __forceinline__
uint64_t make_uint64_t(uint32_t LO, uint32_t HI)
{
    //return __double_as_longlong(__hiloint2double(HI, LO));

    uint64_t result;
    asm("mov.b64 %0, {%1, %2};" : "=l"(result) : "r"(LO), "r"(HI));
    return result;
}

/* Given a 64-bit operand and reciprocal, and 32-bit modulo, return the 32-bit
modulus without using division.  */
__device__ __forceinline__
uint32_t mod_p_small(uint64_t a, uint32_t p, uint64_t recip)
{
    uint64_t q=0;
    int64_t r;

    asm("{\n\t"
        " .reg .pred %p;\n\t"
        "mul.hi.u64 %1, %3, %2\n\t;"
        "mul.lo.u64 %2, %4, %1\n\t;"
        "sub.u64 %0, %3, %2\n\t;"
        " setp.ge.s64 %p, %0, %4\n\t;"
        " @%p sub.s64 %0, %0, %4\n\t;"
        "}\n\t"
        : "=l"(r) "+l"(q) "+l"(recip) : "l"(a) , "l"((uint64_t)p));

    //uint64_t q = __umul64hi(a, recip);
    //recip = p*q;
    //r = a - recip;
    //if (r >= p)
    //    r -= p;

    return (uint32_t)r;
}

/* Take uint1024_t A and mod it by uint32_t B. */
__device__ __forceinline__
uint32_t mpi_mod_int(uint64_t *A, uint32_t B, uint64_t recip)
{
    if (B == 1)
        return 0;
    else if (B == 2)
        return A[0]&1;

    uint64_t x;
    uint64_t y;

    #pragma unroll 16
    for(uint8_t i = 16; i > 0; --i)
    {
        x  = A[i-1];
        y = (y << 32) | (x >> 32);
        y = mod_p_small(y, B, recip);

        x <<= 32;
        y = (y << 32) | (x >> 32);
        y = mod_p_small(y, B, recip);
    }

    return (uint32_t)y;
}

#endif
