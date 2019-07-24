#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


#include <CUDA/include/util.h>
#include <CUDA/include/constants.h>
#include <CUDA/include/hash_helper.cuh>

#include <LLC/hash/SK.h>

#include <Util/include/debug.h>

#include <cstdint>
#include <iomanip>

#define ROL64(x, n) (((x) << (n)) | ((x) >> (64 - (n))))

uint64_t *h_sknonce[GPU_MAX];
uint64_t *d_SKNonce[GPU_MAX];
uint64_t *d_SKLowerHash[GPU_MAX];
uint32_t nBestHeight = 0;

const uint64_t cpu_SKEIN1024_IV_1024[16] =
{
	0x5A4352BE62092156,
	0x5F6E8B1A72F001CA,
	0xFFCBFE9CA1A2CE26,
	0x6C23C39667038BCA,
	0x583A8BFCCE34EB6C,
	0x3FDBFB11D4A46A3E,
	0x3304ACFCA8300998,
	0xB2F6675FA17F0FD2,
	0x9D2599730EF7AB6B,
	0x0914A20D3DFEA9E4,
	0xCC1A9CAFA494DBD3,
	0x9828030DA0A6388C,
	0x0D339D5DAADEE3DC,
	0xFC46DE35C4E2A086,
	0x53D6E4F52E19A6D1,
	0x5663952F715D1DDD,
};

const int cpu_ROT1024[8][8] =
{
	{ 55, 43, 37, 40, 16, 22, 38, 12 },
	{ 25, 25, 46, 13, 14, 13, 52, 57 },
	{ 33, 8, 18, 57, 21, 12, 32, 54 },
	{ 34, 43, 25, 60, 44, 9, 59, 34 },
	{ 28, 7, 47, 48, 51, 9, 35, 41 },
	{ 17, 6, 18, 25, 43, 42, 40, 15 },
	{ 58, 7, 32, 45, 19, 18, 2, 56 },
	{ 47, 49, 27, 58, 37, 48, 53, 56 }
};

__forceinline__ __host__
void Round1024_host(uint64_t &p0, uint64_t &p1, uint64_t &p2, uint64_t &p3, uint64_t &p4, uint64_t &p5, uint64_t &p6, uint64_t &p7,
	                uint64_t &p8, uint64_t &p9, uint64_t &pA, uint64_t &pB, uint64_t &pC, uint64_t &pD, uint64_t &pE, uint64_t &pF,
	                int ROT)
{
	p0 += p1;
	p1 = ROL64(p1, cpu_ROT1024[ROT][0]);
	p1 ^= p0;
	p2 += p3;
	p3 = ROL64(p3, cpu_ROT1024[ROT][1]);
	p3 ^= p2;
	p4 += p5;
	p5 = ROL64(p5, cpu_ROT1024[ROT][2]);
	p5 ^= p4;
	p6 += p7;
	p7 = ROL64(p7, cpu_ROT1024[ROT][3]);
	p7 ^= p6;
	p8 += p9;
	p9 = ROL64(p9, cpu_ROT1024[ROT][4]);
	p9 ^= p8;
	pA += pB;
	pB = ROL64(pB, cpu_ROT1024[ROT][5]);
	pB ^= pA;
	pC += pD;
	pD = ROL64(pD, cpu_ROT1024[ROT][6]);
	pD ^= pC;
	pE += pF;
	pF = ROL64(pF, cpu_ROT1024[ROT][7]);
	pF ^= pE;
}


__forceinline__ __device__
void Round1024_0(uint2 &p0, uint2 &p1, uint2 &p2, uint2 &p3, uint2 &p4, uint2 &p5, uint2 &p6, uint2 &p7, uint2 &p8, uint2 &p9,
	             uint2 &pA, uint2 &pB, uint2 &pC, uint2 &pD, uint2 &pE, uint2 &pF)
{
	p0 += p1;
	p2 += p3;
	p4 += p5;
	p6 += p7;
	p8 += p9;
	pA += pB;
	pC += pD;
	pE += pF;

	p1 = ROL2_1(p1, 55) ^ p0;
	p3 = ROL2_1(p3, 43) ^ p2;
	p5 = ROL2_1(p5, 37) ^ p4;
	p7 = ROL2_1(p7, 40) ^ p6;
	p9 = ROL2_0(p9, 16) ^ p8;
	pB = ROL2_0(pB, 22) ^ pA;
	pD = ROL2_1(pD, 38) ^ pC;
	pF = ROL2_0(pF, 12) ^ pE;
}

__forceinline__ __device__
void Round1024_1(uint2 &p0, uint2 &p1, uint2 &p2, uint2 &p3, uint2 &p4, uint2 &p5, uint2 &p6, uint2 &p7, uint2 &p8, uint2 &p9,
	             uint2 &pA, uint2 &pB, uint2 &pC, uint2 &pD, uint2 &pE, uint2 &pF)
{
	p0 += p1;
	p2 += p3;
	p4 += p5;
	p6 += p7;
	p8 += p9;
	pA += pB;
	pC += pD;
	pE += pF;

	p1 = ROL2_0(p1, 25) ^ p0;
	p3 = ROL2_0(p3, 25) ^ p2;
	p5 = ROL2_1(p5, 46) ^ p4;
	p7 = ROL2_0(p7, 13) ^ p6;
	p9 = ROL2_0(p9, 14) ^ p8;
	pB = ROL2_0(pB, 13) ^ pA;
	pD = ROL2_1(pD, 52) ^ pC;
	pF = ROL2_1(pF, 57) ^ pE;
}

__forceinline__ __device__
void Round1024_2(uint2 &p0, uint2 &p1, uint2 &p2, uint2 &p3, uint2 &p4, uint2 &p5, uint2 &p6, uint2 &p7, uint2 &p8, uint2 &p9,
	             uint2 &pA, uint2 &pB, uint2 &pC, uint2 &pD, uint2 &pE, uint2 &pF)
{
	p0 += p1;
	p2 += p3;
	p4 += p5;
	p6 += p7;
	p8 += p9;
	pA += pB;
	pC += pD;
	pE += pF;

	p1 = ROL2_1(p1, 33) ^ p0;
	p3 = ROL2_0(p3, 8) ^ p2;
	p5 = ROL2_0(p5, 18) ^ p4;
	p7 = ROL2_1(p7, 57) ^ p6;
	p9 = ROL2_0(p9, 21) ^ p8;
	pB = ROL2_0(pB, 12) ^ pA;
	pD = ROL2_1(pD, 32) ^ pC;
	pF = ROL2_1(pF, 54) ^ pE;
}

__forceinline__ __device__
void Round1024_3(uint2 &p0, uint2 &p1, uint2 &p2, uint2 &p3, uint2 &p4, uint2 &p5, uint2 &p6, uint2 &p7, uint2 &p8, uint2 &p9,
	             uint2 &pA, uint2 &pB, uint2 &pC, uint2 &pD, uint2 &pE, uint2 &pF)
{
	p0 += p1;
	p2 += p3;
	p4 += p5;
	p6 += p7;
	p8 += p9;
	pA += pB;
	pC += pD;
	pE += pF;

	p1 = ROL2_1(p1, 34) ^ p0;
	p3 = ROL2_1(p3, 43) ^ p2;
	p5 = ROL2_0(p5, 25) ^ p4;
	p7 = ROL2_1(p7, 60) ^ p6;
	p9 = ROL2_1(p9, 44) ^ p8;
	pB = ROL2_0(pB,  9) ^ pA;
	pD = ROL2_1(pD, 59) ^ pC;
	pF = ROL2_1(pF, 34) ^ pE;
}

__forceinline__ __device__
void Round1024_4(uint2 &p0, uint2 &p1, uint2 &p2, uint2 &p3, uint2 &p4, uint2 &p5, uint2 &p6, uint2 &p7, uint2 &p8, uint2 &p9,
	             uint2 &pA, uint2 &pB, uint2 &pC, uint2 &pD, uint2 &pE, uint2 &pF)
{
	p0 += p1;
	p2 += p3;
	p4 += p5;
	p6 += p7;
	p8 += p9;
	pA += pB;
	pC += pD;
	pE += pF;

	p1 = ROL2_0(p1, 28) ^ p0;
	p3 = ROL2_0(p3,  7) ^ p2;
	p5 = ROL2_1(p5, 47) ^ p4;
	p7 = ROL2_1(p7, 48) ^ p6;
	p9 = ROL2_1(p9, 51) ^ p8;
	pB = ROL2_0(pB,  9) ^ pA;
	pD = ROL2_1(pD, 35) ^ pC;
	pF = ROL2_1(pF, 41) ^ pE;
}

__forceinline__ __device__
void Round1024_5(uint2 &p0, uint2 &p1, uint2 &p2, uint2 &p3, uint2 &p4, uint2 &p5, uint2 &p6, uint2 &p7, uint2 &p8, uint2 &p9,
	             uint2 &pA, uint2 &pB, uint2 &pC, uint2 &pD, uint2 &pE, uint2 &pF)
{
	p0 += p1;
	p2 += p3;
	p4 += p5;
	p6 += p7;
	p8 += p9;
	pA += pB;
	pC += pD;
	pE += pF;

	p1 = ROL2_0(p1, 17) ^ p0;
	p3 = ROL2_0(p3,  6) ^ p2;
	p5 = ROL2_0(p5, 18) ^ p4;
	p7 = ROL2_0(p7, 25) ^ p6;
	p9 = ROL2_1(p9, 43) ^ p8;
	pB = ROL2_1(pB, 42) ^ pA;
	pD = ROL2_1(pD, 40) ^ pC;
	pF = ROL2_0(pF, 15) ^ pE;
}

__forceinline__ __device__
void Round1024_6(uint2 &p0, uint2 &p1, uint2 &p2, uint2 &p3, uint2 &p4, uint2 &p5, uint2 &p6, uint2 &p7, uint2 &p8, uint2 &p9,
	             uint2 &pA, uint2 &pB, uint2 &pC, uint2 &pD, uint2 &pE, uint2 &pF)
{
	p0 += p1;
	p2 += p3;
	p4 += p5;
	p6 += p7;
	p8 += p9;
	pA += pB;
	pC += pD;
	pE += pF;

	p1 = ROL2_1(p1, 58) ^ p0;
	p3 = ROL2_0(p3,  7) ^ p2;
	p5 = ROL2_1(p5, 32) ^ p4;
	p7 = ROL2_1(p7, 45) ^ p6;
	p9 = ROL2_0(p9, 19) ^ p8;
	pB = ROL2_0(pB, 18) ^ pA;
	pD = ROL2_0(pD,  2) ^ pC;
	pF = ROL2_1(pF, 56) ^ pE;
}

__forceinline__ __device__
void Round1024_7(uint2 &p0, uint2 &p1, uint2 &p2, uint2 &p3, uint2 &p4, uint2 &p5, uint2 &p6, uint2 &p7, uint2 &p8, uint2 &p9,
	             uint2 &pA, uint2 &pB, uint2 &pC, uint2 &pD, uint2 &pE, uint2 &pF)
{
	p0 += p1;
	p2 += p3;
	p4 += p5;
	p6 += p7;
	p8 += p9;
	pA += pB;
	pC += pD;
	pE += pF;

	p1 = ROL2_1(p1, 47) ^ p0;
	p3 = ROL2_1(p3, 49) ^ p2;
	p5 = ROL2_0(p5, 27) ^ p4;
	p7 = ROL2_1(p7, 58) ^ p6;
	p9 = ROL2_1(p9, 37) ^ p8;
	pB = ROL2_1(pB, 48) ^ pA;
	pD = ROL2_1(pD, 53) ^ pC;
	pF = ROL2_1(pF, 56) ^ pE;
}

__device__ __forceinline__
uint2 xor3(const uint2& a, const uint2& b, const uint2& c)
{
	uint2 result;
	asm("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(result.x) : "r"(a.x), "r"(b.x), "r"(c.x)); //0x96 = 0xF0 ^ 0xCC ^ 0xAA
	asm("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(result.y) : "r"(a.y), "r"(b.y), "r"(c.y)); //0x96 = 0xF0 ^ 0xCC ^ 0xAA
	return result;
}

__device__ __forceinline__
uint2 chi(const uint2& a, const uint2& b, const uint2& c)
{
	uint2 result;
	asm("lop3.b32 %0, %1, %2, %3, 0xD2;" : "=r"(result.x) : "r"(a.x), "r"(b.x), "r"(c.x)); //0xD2 = 0xF0 ^ ((~0xCC) & 0xAA)
	asm("lop3.b32 %0, %1, %2, %3, 0xD2;" : "=r"(result.y) : "r"(a.y), "r"(b.y), "r"(c.y)); //0xD2 = 0xF0 ^ ((~0xCC) & 0xAA)
	return result;
}

__device__ __forceinline__
uint2 xor5(const uint2& a, const uint2& b, const uint2& c, const uint2& d, const uint2& e)
{
	return xor3(xor3(a, b, c), d, e);
}


__device__ __forceinline__
void Keccak1600(uint2 *s)
{
	uint2 t[5], u, v;

	#pragma unroll 2
	for (int i = 0; i < 24; ++i)
	{
		t[0] = xor5(s[0], s[5], s[10], s[15], s[20]);
		t[1] = xor5(s[1], s[6], s[11], s[16], s[21]);
		t[2] = xor5(s[2], s[7], s[12], s[17], s[22]);
		t[3] = xor5(s[3], s[8], s[13], s[18], s[23]);
		t[4] = xor5(s[4], s[9], s[14], s[19], s[24]);

		u = ROL2_0(t[1], 1);
		s[0] = xor3(s[0], t[4], u);
		s[5] = xor3(s[5], t[4], u);
		s[10] = xor3(s[10], t[4], u);
		s[15] = xor3(s[15], t[4], u);
		s[20] = xor3(s[20], t[4], u);

		u = ROL2_0(t[2], 1);
		s[1] = xor3(s[1], t[0], u);
		s[6] = xor3(s[6], t[0], u);
		s[11] = xor3(s[11], t[0], u);
		s[16] = xor3(s[16], t[0], u);
		s[21] = xor3(s[21], t[0], u);

		u = ROL2_0(t[3], 1);
		s[2] = xor3(s[2], t[1], u);
		s[7] = xor3(s[7], t[1], u);
		s[12] = xor3(s[12], t[1], u);
		s[17] = xor3(s[17], t[1], u);
		s[22] = xor3(s[22], t[1], u);

		u = ROL2_0(t[4], 1);
		s[3] = xor3(s[3], t[2], u);
		s[8] = xor3(s[8], t[2], u);
		s[13] = xor3(s[13], t[2], u);
		s[18] = xor3(s[18], t[2], u);
		s[23] = xor3(s[23], t[2], u);


		u = ROL2_0(t[0], 1);
		s[4] = xor3(s[4], t[3], u);
		s[9] = xor3(s[9], t[3], u);
		s[14] = xor3(s[14], t[3], u);
		s[19] = xor3(s[19], t[3], u);
		s[24] = xor3(s[24], t[3], u);

		u = s[1];

		s[1] = ROL2_1(s[6], 44);
		s[6] = ROL2_0(s[9], 20);
		s[9] = ROL2_1(s[22], 61);
		s[22] = ROL2_1(s[14], 39);
		s[14] = ROL2_0(s[20], 18);
		s[20] = ROL2_1(s[2], 62);
		s[2] = ROL2_1(s[12], 43);
		s[12] = ROL2_0(s[13], 25);
		s[13] = ROL2_0(s[19], 8);
		s[19] = ROL2_1(s[23], 56);
		s[23] = ROL2_1(s[15], 41);
		s[15] = ROL2_0(s[4], 27);
		s[4] = ROL2_0(s[24], 14);
		s[24] = ROL2_0(s[21], 2);
		s[21] = ROL2_1(s[8], 55);
		s[8] = ROL2_1(s[16], 45);
		s[16] = ROL2_1(s[5], 36);
		s[5] = ROL2_0(s[3], 28);
		s[3] = ROL2_0(s[18], 21);
		s[18] = ROL2_0(s[17], 15);
		s[17] = ROL2_0(s[11], 10);
		s[11] = ROL2_0(s[7], 6);
		s[7] = ROL2_0(s[10], 3);
		s[10] = ROL2_0(u, 1);

		u = s[0];
        v = s[1];
		s[0] = chi(s[0], s[1], s[2]);
		s[1] = chi(s[1], s[2], s[3]);
		s[2] = chi(s[2], s[3], s[4]);
		s[3] = chi(s[3], s[4], u);
		s[4] = chi(s[4], u, v);

		u = s[5];
        v = s[6];
		s[5] = chi(s[5], s[6], s[7]);
		s[6] = chi(s[6], s[7], s[8]);
		s[7] = chi(s[7], s[8], s[9]);
		s[8] = chi(s[8], s[9], u);
		s[9] = chi(s[9], u, v);

		u = s[10];
        v = s[11];
		s[10] = chi(s[10], s[11], s[12]);
		s[11] = chi(s[11], s[12], s[13]);
		s[12] = chi(s[12], s[13], s[14]);
		s[13] = chi(s[13], s[14], u);
		s[14] = chi(s[14], u, v);

		u = s[15];
        v = s[16];
		s[15] = chi(s[15], s[16], s[17]);
		s[16] = chi(s[16], s[17], s[18]);
		s[17] = chi(s[17], s[18], s[19]);
		s[18] = chi(s[18], s[19], u);
		s[19] = chi(s[19], u, v);

		u = s[20];
        v = s[21];
		s[20] = chi(s[20], s[21], s[22]);
		s[21] = chi(s[21], s[22], s[23]);
		s[22] = chi(s[22], s[23], s[24]);
		s[23] = chi(s[23], s[24], u);
		s[24] = chi(s[24], u, v);

		s[0] ^= keccak_round_constants[i];
	}
}

__forceinline__ __device__
void Skein1024(uint2 *p, uint2 *t, uint2 *b)
{

	#pragma unroll
	for (int i = 1; i < 21; i += 2)
	{
		Round1024_0(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13], p[14], p[15]);
		Round1024_1(p[0], p[9], p[2], p[13], p[6], p[11], p[4], p[15], p[10], p[7], p[12], p[3], p[14], p[5], p[8], p[1]);
		Round1024_2(p[0], p[7], p[2], p[5], p[4], p[3], p[6], p[1], p[12], p[15], p[14], p[13], p[8], p[11], p[10], p[9]);
		Round1024_3(p[0], p[15], p[2], p[11], p[6], p[13], p[4], p[9], p[14], p[1], p[8], p[5], p[10], p[3], p[12], p[7]);

		#pragma unroll
		for (int j = 0; j < 13; ++j)
			p[j] += b[(i + j) % 17];

		p[13] += b[(i + 13) % 17] + t[(i + 0) % 3];
		p[14] += b[(i + 14) % 17] + t[(i + 1) % 3];
		p[15] += b[(i + 15) % 17] + make_uint2(i, 0);

		Round1024_4(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12], p[13], p[14], p[15]);
		Round1024_5(p[0], p[9], p[2], p[13], p[6], p[11], p[4], p[15], p[10], p[7], p[12], p[3], p[14], p[5], p[8], p[1]);
		Round1024_6(p[0], p[7], p[2], p[5], p[4], p[3], p[6], p[1], p[12], p[15], p[14], p[13], p[8], p[11], p[10], p[9]);
		Round1024_7(p[0], p[15], p[2], p[11], p[6], p[13], p[4], p[9], p[14], p[1], p[8], p[5], p[10], p[3], p[12], p[7]);

		#pragma unroll
		for (int j = 0; j < 13; ++j)
			p[j] += b[(i + j + 1) % 17];

		p[13] += b[(i + 14) % 17] + t[(i + 1) % 3];
		p[14] += b[(i + 15) % 17] + t[(i + 2) % 3];
		p[15] += b[(i + 16) % 17] + make_uint2(i + 1, 0);
	}
}

/** CUDA 8
 *  GTX 1060 SC 6GB / 66% Power Limit, +100 Core, -502 Mem
 *  Skein (2 rounds) = 229 MH/s
 *  Keccak (3 rounds) = 146 MH/s
 **/

 __launch_bounds__(896, 1)
__global__ void  sk1024_gpu_hash(int threads, uint64_t startNonce, uint64_t *resNonce, uint64_t *resLowerHash)
{
	int thread = blockDim.x * blockIdx.x + threadIdx.x;

	if (thread < threads)
	{
		uint2 h[17];
		uint2 t[3];
		uint2 p[16];

		uint64_t nonce = startNonce + (uint64_t)thread;
		uint64_t LowerHash = 0xFFFFFFFFFFFFFFFF;

        #pragma unroll
		for(int i = 0; i < 10; ++i)
			p[i] = uMessage[i + 16] + c_hv[i];

		uint2 tempnonce = vectorize(nonce);
		p[10] = tempnonce + c_hv[10];

		t[0] = t12[3];
		t[1] = t12[4];
		t[2] = t12[5];

		p[11] = c_hv[11];
		p[12] = c_hv[12];
		p[13] = c_hv[13] + t[0];
		p[14] = c_hv[14] + t[1];
		p[15] = c_hv[15];

		Skein1024(p, t, c_hv);

        #pragma unroll
		for(int i = 0; i < 10; ++i)
			p[i] ^= uMessage[i + 16];

		p[10] ^= tempnonce;
		h[16] = skein_ks_parity;

		#pragma unroll
		for(int i = 0; i < 16; i += 2)
		{
			h[i] = p[i];
			h[i + 1] = p[i + 1];
			h[16] = xor3(h[16], h[i], h[i + 1]);
		}

		t[0] = t12[6];
		t[1] = t12[7];
		t[2] = t12[8];

		p[13] += t[0];
		p[14] += t[1];

		Skein1024(p, t, h);

		__align__(16) uint2 state[25];

		state[0] = p[0];
		state[1] = p[1];
		state[2] = p[2];
		state[3] = p[3];
		state[4] = p[4];
		state[5] = p[5];
		state[6] = p[6];
		state[7] = p[7];
		state[8] = p[8];

		#pragma unroll
		for(int i = 9; i < 25; ++i)
			state[i] = make_uint2(0, 0);

		Keccak1600(state);

		state[0] ^= p[9];
		state[1] ^= p[10];
		state[2] ^= p[11];
		state[3] ^= p[12];
		state[4] ^= p[13];
		state[5] ^= p[14];
		state[6] ^= p[15];
		state[7] ^= vectorize(0x05);
		state[8] ^= vectorize(1ULL << 63);

		Keccak1600(state);
		Keccak1600(state);

		//if (devectorize(state[6]) <= 0x3FFFC000) {
		if(devectorize(state[6]) <= pTarget[15])
		{
			LowerHash = *resLowerHash;
			if(devectorize(state[6]) < LowerHash)
			{
				*resLowerHash = devectorize(state[6]);
				*resNonce = nonce;
			}
		}
	}
}

__host__ uint64_t sk1024_cpu_hash(uint32_t thr_id, uint32_t threads, uint64_t startNonce, uint32_t threadsperblock)
{
	uint64_t result = 0xFFFFFFFFFFFFFFFF;
	CHECK(cudaMemset(d_SKNonce[thr_id], 0xFF, sizeof(uint64_t)));
	CHECK(cudaMemset(d_SKLowerHash[thr_id], result, sizeof(uint64_t)));

	//dim3 grid(threads / threadsperblock);
	dim3 block(threadsperblock);
    dim3 grid((threads + block.x - 1) / block.x);


	sk1024_gpu_hash<<<grid, block >>>(threads, startNonce, d_SKNonce[thr_id], d_SKLowerHash[thr_id]);
	CHECK(cudaMemcpy(h_sknonce[thr_id], d_SKNonce[thr_id], sizeof(uint64_t), cudaMemcpyDeviceToHost));

	result = *h_sknonce[thr_id];

	return result;
}

__host__ void cuda_sk1024_init(uint32_t thr_id)
{
	CHECK(cudaMalloc(&d_SKNonce[thr_id], sizeof(uint64_t)));
	CHECK(cudaMalloc(&d_SKLowerHash[thr_id], sizeof(uint64_t)));
	CHECK(cudaMallocHost(&h_sknonce[thr_id], 1 * sizeof(uint64_t)));
}

__host__ void cuda_sk1024_free(uint32_t thr_id)
{
	CHECK(cudaFree(d_SKNonce[thr_id]));
	CHECK(cudaFree(d_SKLowerHash[thr_id]));
	CHECK(cudaFreeHost(h_sknonce[thr_id]));
}


__host__ void cuda_sk1024_set_Target(const void *ptarget)
{
	CHECK(cudaMemcpyToSymbol(pTarget, ptarget, 16 * sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
}


__host__ void cuda_sk1024_setBlock(void *pdata, uint32_t nHeight)
{
	uint2 hv[17];
	uint64_t t[3];
	uint64_t h[17];
	uint64_t p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15;

	uint64_t cpu_skein_ks_parity = 0x5555555555555555;
	h[16] = cpu_skein_ks_parity;
	uint2 cpu_Message[27];


	for (int i = 0; i < 16; ++i)
	{
		h[i] = cpu_SKEIN1024_IV_1024[i];
		h[16] ^= h[i];
	}
	uint64_t* alt_data = (uint64_t*)pdata;

	/////////////////////// round 1 //////////////////////////// should be on cpu => constant on gpu
	p0 = alt_data[0];
	p1 = alt_data[1];
	p2 = alt_data[2];
	p3 = alt_data[3];
	p4 = alt_data[4];
	p5 = alt_data[5];
	p6 = alt_data[6];
	p7 = alt_data[7];
	p8 = alt_data[8];
	p9 = alt_data[9];
	p10 = alt_data[10];
	p11 = alt_data[11];
	p12 = alt_data[12];
	p13 = alt_data[13];
	p14 = alt_data[14];
	p15 = alt_data[15];
	t[0] = 0x80; // ptr
	t[1] = 0x7000000000000000; // etype
	t[2] = 0x7000000000000080;

	p0 += h[0];
	p1 += h[1];
	p2 += h[2];
	p3 += h[3];
	p4 += h[4];
	p5 += h[5];
	p6 += h[6];
	p7 += h[7];
	p8 += h[8];
	p9 += h[9];
	p10 += h[10];
	p11 += h[11];
	p12 += h[12];
	p13 += h[13] + t[0];
	p14 += h[14] + t[1];
	p15 += h[15];

	for (int i = 1; i < 21; i += 2)
	{
		Round1024_host(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, 0);
		Round1024_host(p0, p9, p2, p13, p6, p11, p4, p15, p10, p7, p12, p3, p14, p5, p8, p1, 1);
		Round1024_host(p0, p7, p2, p5, p4, p3, p6, p1, p12, p15, p14, p13, p8, p11, p10, p9, 2);
		Round1024_host(p0, p15, p2, p11, p6, p13, p4, p9, p14, p1, p8, p5, p10, p3, p12, p7, 3);

		p0 += h[(i + 0) % 17];
		p1 += h[(i + 1) % 17];
		p2 += h[(i + 2) % 17];
		p3 += h[(i + 3) % 17];
		p4 += h[(i + 4) % 17];
		p5 += h[(i + 5) % 17];
		p6 += h[(i + 6) % 17];
		p7 += h[(i + 7) % 17];
		p8 += h[(i + 8) % 17];
		p9 += h[(i + 9) % 17];
		p10 += h[(i + 10) % 17];
		p11 += h[(i + 11) % 17];
		p12 += h[(i + 12) % 17];
		p13 += h[(i + 13) % 17] + t[(i + 0) % 3];
		p14 += h[(i + 14) % 17] + t[(i + 1) % 3];
		p15 += h[(i + 15) % 17] + (uint64_t)i;

		Round1024_host(p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, 4);
		Round1024_host(p0, p9, p2, p13, p6, p11, p4, p15, p10, p7, p12, p3, p14, p5, p8, p1, 5);
		Round1024_host(p0, p7, p2, p5, p4, p3, p6, p1, p12, p15, p14, p13, p8, p11, p10, p9, 6);
		Round1024_host(p0, p15, p2, p11, p6, p13, p4, p9, p14, p1, p8, p5, p10, p3, p12, p7, 7);

		p0 += h[(i + 1) % 17];
		p1 += h[(i + 2) % 17];
		p2 += h[(i + 3) % 17];
		p3 += h[(i + 4) % 17];
		p4 += h[(i + 5) % 17];
		p5 += h[(i + 6) % 17];
		p6 += h[(i + 7) % 17];
		p7 += h[(i + 8) % 17];
		p8 += h[(i + 9) % 17];
		p9 += h[(i + 10) % 17];
		p10 += h[(i + 11) % 17];
		p11 += h[(i + 12) % 17];
		p12 += h[(i + 13) % 17];
		p13 += h[(i + 14) % 17] + t[(i + 1) % 3];
		p14 += h[(i + 15) % 17] + t[(i + 2) % 3];
		p15 += h[(i + 16) % 17] + (uint64_t)(i + 1);

	}

	h[0] = p0^alt_data[0];
	h[1] = p1^alt_data[1];
	h[2] = p2^alt_data[2];
	h[3] = p3^alt_data[3];
	h[4] = p4^alt_data[4];
	h[5] = p5^alt_data[5];
	h[6] = p6^alt_data[6];
	h[7] = p7^alt_data[7];
	h[8] = p8^alt_data[8];
	h[9] = p9^alt_data[9];
	h[10] = p10^alt_data[10];
	h[11] = p11^alt_data[11];
	h[12] = p12^alt_data[12];
	h[13] = p13^alt_data[13];
	h[14] = p14^alt_data[14];
	h[15] = p15^alt_data[15];
	h[16] = cpu_skein_ks_parity;

	for (int i = 0; i < 16; ++i)
	{
		h[16] ^= h[i];
	}

	/* will slow down things */
	for (int i = 0; i < 17; ++i)
	{
		hv[i] = lohi_host(h[i]);
	}

	/* might slow down things */
	for (int i = 0; i < 27; ++i)
	{
		cpu_Message[i] = lohi_host(alt_data[i]);
	}

	nBestHeight = nHeight;

	CHECK(cudaMemcpyToSymbol(c_hv, hv, sizeof(hv), 0, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpyToSymbol(uMessage, cpu_Message, sizeof(cpu_Message), 0, cudaMemcpyHostToDevice));
}


extern bool cuda_sk1024_hash(
	uint32_t thr_id,
	uint32_t* TheData,
	uint1024_t TheTarget,
	uint64_t &TheNonce,
	uint64_t *hashes_done,
	uint32_t throughput,
	uint32_t threadsPerBlock,
	uint32_t nHeight)
{
	uint64_t *ptarget = (uint64_t*)&TheTarget;

	const uint64_t first_nonce = TheNonce;
	const uint64_t Htarg = ptarget[15];

	uint64_t foundNonce = sk1024_cpu_hash(thr_id,
										  throughput,
										  ((uint64_t*)TheData)[26],
										  threadsPerBlock);

	if (foundNonce != 0xffffffffffffffff)
	{
		((uint64_t*)TheData)[26] = foundNonce;
		uint1024_t skein;
		Skein1024_Ctxt_t ctx;
		Skein1024_Init(&ctx, 1024);
		Skein1024_Update(&ctx, (uint8_t *)TheData, 216);
		Skein1024_Final(&ctx, (uint8_t *)&skein);

		uint64_t keccak[16];
		Keccak_HashInstance ctx_keccak;
		Keccak_HashInitialize(&ctx_keccak, 576, 1024, 1024, 0x05);
		Keccak_HashUpdate(&ctx_keccak, (uint8_t *)&skein, 1024);
		Keccak_HashFinal(&ctx_keccak, (uint8_t *)&keccak);

		if (keccak[15] <= Htarg)
		{
			TheNonce = foundNonce; //return the nonce
			*hashes_done = foundNonce - first_nonce + 1;
			return true;
		}
		else
		{
			debug::error("GPU #", thr_id, ": result for nonce ", foundNonce, " does not validate on CPU!");

			debug::log(0, std::hex, std::setw(16), std::setfill('0'), keccak[15], " > ", std::setw(16), std::setfill('0'), Htarg);
		}


	}

	((uint64_t*)TheData)[26] += throughput;

	uint64_t doneNonce = ((uint64_t*)TheData)[26];

	if (doneNonce < 18446744072149270489lu)
		*hashes_done = doneNonce - first_nonce + 1;

	return false;
}
