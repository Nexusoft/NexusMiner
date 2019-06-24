/*******************************************************************************************

 Nexus Earth 2018

 [Scale Indefinitely] BlackJack. http://www.opensource.org/licenses/mit-license.php

*******************************************************************************************/
#include <CUDA/include/constants.h>
#include <CUDA/include/prime_helper.cuh>
#include <CUDA/include/macro.h>

#include <cstdint>

__device__ __forceinline__
void assign(uint32_t *l, uint32_t *r)
{
    #pragma unroll
    for(uint8_t i = 0; i < WORD_MAX; ++i)
        l[i] = r[i];
}

__device__ __forceinline__
void assign_zero(uint32_t *l)
{
    #pragma unroll
    for(uint8_t i = 0; i < WORD_MAX; ++i)
        l[i] ^= l[i];
}


__device__ __forceinline__
int inv2adic(uint32_t x)
{
    uint32_t a;
    a = x;
    x = (((x+2)&4)<<1)+x;
    x *= 2 - a*x;
    x *= 2 - a*x;
    x *= 2 - a*x;
    return -x;
}


__device__ __forceinline__
uint32_t cmp_ge_n(uint32_t *x, uint32_t *y)
{
    #pragma unroll
    for(int8_t i = WORD_MAX-1; i >= 0; --i)
    {
        if(x[i] > y[i])
            return 1;

        if(x[i] < y[i])
            return 0;
    }
    return 1;
}


__device__ __forceinline__
void sub_n(uint32_t *z, uint32_t *x, uint32_t *y)
{
    asm("sub.cc.u32 %0, %1, %2;" : "=r"(z[0]) : "r"(x[0]), "r"(y[0]));

    #pragma unroll
    for(uint8_t i = 1; i < WORD_MAX - 1; ++i)
        asm("subc.cc.u32 %0, %1, %2;" : "=r"(z[i]) : "r"(x[i]), "r"(y[i]));

    asm("subc.u32 %0, %1, %2;" : "=r"(z[WORD_MAX - 1]) : "r"(x[WORD_MAX - 1]), "r"(y[WORD_MAX - 1]));
}


__device__ __forceinline__
void sub_ui(uint32_t *z, uint32_t *x, const uint32_t &ui)
{
    asm("sub.cc.u32 %0, %1, %2;" : "=r"(z[0]) : "r"(x[0]), "r"(ui));

    #pragma unroll
    for(uint8_t i = 1; i < WORD_MAX - 1; ++i)
        asm("subc.cc.u32 %0, %1, 0;" : "=r"(z[i]) : "r"(x[i]));

    asm("subc.u32 %0, %1, 0;" : "=r"(z[WORD_MAX - 1]) : "r"(x[WORD_MAX - 1]));
}


__device__ __forceinline__
void add_ui(uint32_t *z, uint32_t *x, const uint64_t &ui)
{
    uint32_t temp = x[0] + static_cast<uint32_t>(ui & 0xFFFFFFFF);
    uint8_t c = temp < x[0];
    z[0] = temp;

    temp = x[1] + static_cast<uint32_t>(ui >> 32) + c;
    c = temp < x[1];
    z[1] = temp;

    #pragma unroll
    for(uint8_t i = 2; i < WORD_MAX; ++i)
    {
        temp = x[i] + c;
        c = (temp < x[i]);
        z[i] = temp;
    }
}

/* Calculate result = A * B + C */
__device__ __forceinline__
uint64_t mul_add(uint64_t a, uint64_t b, uint64_t c)
{
	uint64_t result;
	asm("mad.lo.u64 %0, %1, %2, %3;" : "=l"(result) : "l"(a), "l"(b), "l"(c));

	return result;
}

__device__ __forceinline__
uint64_t mad32(uint32_t a, uint32_t b, uint64_t c)
{
	uint64_t result;
	asm("mad.wide.u32 %0, %1, %2, %3;" : "=l"(result) : "r"(a), "r"(b), "l"(c));

	return result;
}

__device__ __forceinline__
uint64_t mul32(uint32_t a, uint32_t b)
{
    uint64_t result;
    asm("mul.wide.u32 %0, %1, %2;" : "=l"(result) : "r"(a), "r"(b));
    return result;
}


__device__ __forceinline__
void addmul_1(uint32_t *z, uint32_t *x, const uint32_t y)
{
    uint64_t prod = 0;

    #pragma unroll
    for(uint8_t i = 0; i < WORD_MAX; ++i)
    {
        //prod >>= 32;
        //prod += z[i];
        //prod += static_cast<uint64_t>(x[i]) * static_cast<uint64_t>(y);
        prod = mad32(x[i], y, z[i] + (prod >> 32));// + z[i] + (prod >> 32);
        //prod = static_cast<uint64_t>(x[i]) * static_cast<uint64_t>(y) + z[i] + (prod >> 32);
        //prod = mul_add(x[i], y, static_cast<uint64_t>(z[i]) + (prod >> 32));
        z[i] = prod; //set the low word

        //asm("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r"(z[i]) : "r"(x[i]), "r"(y), "r"(z[i]));
        //asm("madc.hi.cc.u32 %0, %1, %2, %3;" : "=r"(z[i+1]) : "r"(x[i]), "r"(y), "r"(z[i+1]));
        //asm("addc.u32 %0, %1, 0;" : "=r"(z[i+2]) : "r"(z[i+2]));
    }

    //asm("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r"(z[WORD_MAX-1]) : "r"(x[WORD_MAX-1]), "r"(y), "r"(z[WORD_MAX-1]));
    //asm("madc.hi.u32 %0, %1, %2, %3;" : "=r"(z[WORD_MAX]) : "r"(x[WORD_MAX-1]), "r"(y), "r"(z[WORD_MAX]));


    z[WORD_MAX] += prod >> 32;

}

__device__ __forceinline__
void addmul_2(uint32_t *z, uint32_t *x, const uint32_t y)
{
    uint64_t prod = mad32(x[0], y, z[0]);

    #pragma unroll
    for(uint8_t i = 1; i < WORD_MAX; ++i)
    {
        prod = mad32(x[i], y, z[i] + (prod >> 32));// + z[i] + (prod >> 32);
        z[i-1] = prod; //set the low word

    }

    prod = z[WORD_MAX] + (prod >> 32);
    z[WORD_MAX - 1] = prod;
    z[WORD_MAX] = z[WORD_MAX+1] + (prod >> 32);

}



__device__ __forceinline__
void mulredc(uint32_t *z, uint32_t *x, uint32_t *y, uint32_t *n, const uint32_t d, uint32_t *t)
{
    //uint32_t m;//, c;
    //uint64_t temp;

    #pragma unroll
    for(uint8_t i = 0; i < WORD_MAX + 2; ++i)
        t[i] ^= t[i];

    for(uint8_t i = 0; i < WORD_MAX; ++i)
    {
        //c = addmul_1(t, x, y[i]);
        addmul_1(t, x, y[i]);
        //temp = static_cast<uint64_t>(t[WORD_MAX]) + c;
        //t[WORD_MAX] = temp;
        //t[WORD_MAX] += c;
        //t[WORD_MAX + 1] = temp >> 32;

        //m = t[0]*d;

        //c = addmul_1(t, n, m);
        //t[WORD_MAX] += addmul_1(t, n, m);
        addmul_2(t, n, t[0]*d);
        //temp = static_cast<uint64_t>(t[WORD_MAX]) + c;
        //t[WORD_MAX] = temp;
        //t[WORD_MAX] += c;
        //t[WORD_MAX + 1] = temp >> 32;

        //#pragma unroll
        //for(uint8_t j = 0; j <= WORD_MAX; ++j)
        //    t[j] = t[j+1];
    }

    if(cmp_ge_n(t, n))
        sub_n(t, t, n);

    assign(z, t);
}


__device__ __forceinline__
void redc(uint32_t *z, uint32_t *x, uint32_t *n, const uint32_t d, uint32_t *t)
{
    //uint32_t m;

    assign(t, x);

    t[WORD_MAX] ^= t[WORD_MAX];

    for(uint8_t i = 0; i < WORD_MAX; ++i)
    {
        //m = t[0]*d;
        //t[WORD_MAX] = addmul_1(t, n, m);
        addmul_2(t, n, t[0]*d);

        //#pragma unroll
        //for(uint8_t j = 0; j < WORD_MAX; ++j)
        //    t[j] = t[j+1];

        t[WORD_MAX] ^= t[WORD_MAX];
    }

    if(cmp_ge_n(t, n))
        sub_n(t, t, n);

    assign(z, t);
}


__device__ __forceinline__
uint16_t bit_count(uint32_t *x)
{
    #pragma unroll
    for(int8_t i = WORD_MAX - 1; i >= 0; --i)
    {
        if(x[i])
            return ((i+1) << 5) - __clz(x[i]);
    }

    return 1; //any number will have at least 1-bit
}


__device__ __forceinline__
void lshift(uint32_t *r, uint32_t *a, uint16_t shift)
{
    assign_zero(r);

    uint8_t ik;
    uint8_t ik1;
    uint8_t k = shift >> 5;
    shift = shift & 31;

    #pragma unroll
    for(int8_t i = 0; i < WORD_MAX; ++i)
    {
        ik = i + k;
        ik1 = ik + 1;

        if(ik1 < WORD_MAX && shift != 0)
            r[ik1] |= (a[i] >> (32-shift));
        if(ik < WORD_MAX)
            r[ik] |= (a[i] << shift);
    }
}


__device__ __forceinline__
void rshift(uint32_t *r, uint32_t *a, uint16_t shift)
{
    assign_zero(r);

    int8_t ik;
    int8_t ik1;
    uint8_t k = shift >> 5;
    shift = shift & 31;

    #pragma unroll
    for(int8_t i = 0; i < WORD_MAX; ++i)
    {
        ik = i - k;
        ik1 = ik - 1;

        if(ik1 >= 0 && shift != 0)
            r[ik1] |= (a[i] << (32-shift));
        if(ik >= 0)
            r[ik] |= (a[i] >> shift);
    }
}


__device__ __forceinline__
void lshift1(uint32_t *r, uint32_t *a)
{
    #pragma unroll
    for(uint8_t i = WORD_MAX - 1; i > 0; --i)
        asm("shf.l.wrap.b32 %0, %1, %2, 1;" : "=r"(r[i]) : "r"(a[i-1]), "r"(a[i]));

    asm("shl.b32 %0, %1, 1;" : "=r"(r[0]) : "r"(a[0]));
}


__device__ __forceinline__
void rshift1(uint32_t *r, uint32_t *a)
{
    #pragma unroll
    for(uint8_t i = 0; i < WORD_MAX - 1; ++i)
        asm("shf.r.wrap.b32 %0, %1, %2, 1;" : "=r"(r[i]) : "r"(a[i]), "r"(a[i+1]));

    asm("shr.b32 %0, %1, 1;" : "=r"(r[WORD_MAX - 1]) : "r"(a[WORD_MAX - 1]));
}


__device__ __forceinline__
void sqrredc(uint32_t *z, uint32_t *x, uint32_t *n, const uint32_t d, uint32_t *t)
{

    #pragma unroll
    for(uint8_t i = 0; i < WORD_MAX + 2; ++i)
        t[i] ^= t[i];

    for(uint8_t i = 0; i < WORD_MAX; ++i)
    {
        uint64_t prod = 0;
        #pragma unroll
        for(uint8_t j = i; j < WORD_MAX; ++j)
        {
            prod = mad32(x[j], x[i], t[j] + (prod >> 32));

            t[j] = prod; //set the low word
        }
        t[WORD_MAX] += (prod >> 32);

        lshift1(t, t);


        addmul_2(t, n, t[0]*d);
    }

    if(cmp_ge_n(t, n))
        sub_n(t, t, n);

    assign(z, t);
}


/* Calculate ABar and BBar for Montgomery Modular Multiplication. */
__device__ __forceinline__
void calcBar(uint32_t *a, uint32_t *b, uint32_t *n, uint32_t *t)
{
    assign_zero(a);

    lshift(t, n, (WORD_MAX<<5) - bit_count(n));
    sub_n(a, a, t);

    while(cmp_ge_n(a, n))  //calculate R mod N;
    {
        rshift1(t, t);
        if(cmp_ge_n(a, t))
            sub_n(a, a, t);
    }

    lshift1(b, a);     //calculate 2R mod N;
    if(cmp_ge_n(b, n))
        sub_n(b, b, n);
}

__device__ __forceinline__
void calcBar(uint32_t *a, uint32_t *n, uint32_t *t)
{
    assign_zero(a);

    lshift(t, n, (WORD_MAX<<5) - bit_count(n));
    sub_n(a, a, t);

    while(cmp_ge_n(a, n))  //calculate R mod N;
    {
        rshift1(t, t);
        if(cmp_ge_n(a, t))
            sub_n(a, a, t);
    }

}

/* Calculate The window table used for fixed window exponentiation. */
__device__ __forceinline__
void calcWindowTable(uint32_t *a, uint32_t *n, uint32_t *t, uint32_t *table)
{
    lshift1(t, a);     //calculate 2R mod N;
    if(cmp_ge_n(t, n))
        sub_n(t, t, n);

    assign(&table[WORD_MAX], t);


    for(uint16_t i = 2; i < WINDOW_SIZE; ++i) //calculate 2^i R mod N
    {
        lshift1(t, t);
        if(cmp_ge_n(t, n))
            sub_n(t, t, n);

        assign(&table[i * WORD_MAX], t);
    }
}


/* Calculate X = 2^Exp Mod N (Fermat test) */
__device__ __forceinline__
void pow2m(uint32_t *X, uint32_t *Exp, uint32_t *N, uint32_t *table)
{
    uint32_t t[WORD_MAX + 2];
    uint32_t wval = 1;
    uint32_t d = inv2adic(N[0]);

    calcBar(X, N, t);

    calcWindowTable(X, N, t, table);

    int32_t i = bit_count(Exp) - 1;

    if(Exp[i>>5] & (1 << (i & 31)))
        wval |= 1;

    if(((i % WINDOW_BITS) == 0) && wval)
    {
        mulredc(X, X, &table[wval * WORD_MAX], N, d, t);
        wval ^= wval;
    }

    for(--i; i >= 0; --i)
    {
        mulredc(X, X, X, N, d, t);
        //sqrredc(X, X, N, d, t);

        wval <<= 1;

        if(Exp[i>>5] & (1 << (i & 31)))
            wval |= 1;

        if(((i % WINDOW_BITS) == 0) && wval)
        {
            mulredc(X, X, &table[wval * WORD_MAX], N, d, t);
            wval ^= wval;
        }
    }

    redc(X, X, N, d, t);
}


/* Calculate X = 2^(N-1) Mod N (Fermat test, assume no overflow) */
__device__ __forceinline__
void pow2m(uint32_t *X, uint32_t *N)
{
    uint32_t A[WORD_MAX];
    uint32_t t[WORD_MAX + 2];

    uint32_t d = inv2adic(N[0]);

    uint64_t N0 = make_uint64_t(N[0], N[1]) - 1;

    calcBar(X, A, N, t);

    for(int16_t i = bit_count(N)-1; i >= 64; --i)
    {
        mulredc(X, X, X, N, d, t);

        if(N[i>>5] & (1 << (i & 31)))
            mulredc(X, X, A, N, d, t);
    }

    for(int16_t i = 63; i >= 0; --i)
    {
        mulredc(X, X, X, N, d, t);

        if(N0 & ((uint64_t)1 << (i & 63)))
            mulredc(X, X, A, N, d, t);
    }

    redc(X, X, N, d, t);
}



/* Test if number p passes Fermat Primality Test base 2. */
__device__ __forceinline__
bool fermat_prime(uint32_t *p, uint32_t *table)
{
    uint32_t e[WORD_MAX];
    uint32_t r[WORD_MAX];

    sub_ui(e, p, 1);
    //pow2m(r, e, p);
    pow2m(r, e, p, table);

    uint32_t result = r[0] - 1;

    #pragma unroll
    for(uint8_t i = 1; i < WORD_MAX; ++i)
        result |= r[i];

    return (result == 0);
}

/* Add a Result to the buffer. */
__device__ __forceinline__
void add_result(uint64_t *nonce_offsets, uint32_t *nonce_meta, uint32_t *nonce_count,
                           uint64_t &offset, uint32_t &meta, uint32_t max)
{
    uint32_t i = atomicAdd(nonce_count, 1);

    if(i < max)
    {
        nonce_offsets[i] = offset;
        nonce_meta[i] = meta;
    }
}
