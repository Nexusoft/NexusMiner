/*******************************************************************************************

 Nexus Earth 2018

 (credits: cbuchner1 for sieving)

 [Scale Indefinitely] BlackJack. http://www.opensource.org/licenses/mit-license.php

*******************************************************************************************/
#include <CUDA/include/constants.h>

//PRIME/////////////////////////////////////////////////////////////////////////
__constant__ uint64_t c_zTempVar[17];
__constant__ uint64_t c_primorial;
__constant__ uint32_t c_zBaseOrigin[WORD_MAX];

__constant__ uint32_t c_offsets[OFFSETS_MAX];
__constant__ uint32_t c_iA[16];
__constant__ uint32_t c_iB[16];
__constant__ uint32_t c_iT[16];

__constant__ uint32_t c_iBeg;
__constant__ uint32_t c_iEnd;
__constant__ uint32_t c_bitmaskA;
__constant__ uint32_t c_bitmaskT;


__constant__ uint32_t c_quit;
__constant__ uint16_t c_primes[4096];

__constant__ uint32_t c_mark_mask[32] =
{
  0x00000001,0x00000002,0x00000004,0x00000008,
  0x00000010,0x00000020,0x00000040,0x00000080,
  0x00000100,0x00000200,0x00000400,0x00000800,
  0x00001000,0x00002000,0x00004000,0x00008000,
  0x00010000,0x00020000,0x00040000,0x00080000,
  0x00100000,0x00200000,0x00400000,0x00800000,
  0x01000000,0x02000000,0x04000000,0x08000000,
  0x10000000,0x20000000,0x40000000,0x80000000
};





//HASH//////////////////////////////////////////////////////////////////////////
__constant__ uint64_t pTarget[16];
__constant__ uint2 uMessage[27];
__constant__ uint2 c_hv[17];
__constant__ uint2 skein_ks_parity = { 0x55555555, 0x55555555 };

__constant__ uint2 t12[9] =
{
	{ 0x80, 0 },
	{ 0, 0x70000000 },
	{ 0x80, 0x70000000 },
	{ 0xd8, 0 },
	{ 0, 0xb0000000 },
	{ 0xd8, 0xb0000000 },
	{ 0x08, 0 },
	{ 0, 0xff000000 },
	{ 0x08, 0xff000000 }
};

__constant__ uint2 keccak_round_constants[24] =
{
	{ 0x00000001, 0x00000000 },
	{ 0x00008082, 0x00000000 },
	{ 0x0000808a, 0x80000000 },
	{ 0x80008000, 0x80000000 },
	{ 0x0000808b, 0x00000000 },
	{ 0x80000001, 0x00000000 },
	{ 0x80008081, 0x80000000 },
	{ 0x00008009, 0x80000000 },
	{ 0x0000008a, 0x00000000 },
	{ 0x00000088, 0x00000000 },
	{ 0x80008009, 0x00000000 },
	{ 0x8000000a, 0x00000000 },
	{ 0x8000808b, 0x00000000 },
	{ 0x0000008b, 0x80000000 },
	{ 0x00008089, 0x80000000 },
	{ 0x00008003, 0x80000000 },
	{ 0x00008002, 0x80000000 },
	{ 0x00000080, 0x80000000 },
	{ 0x0000800a, 0x00000000 },
	{ 0x8000000a, 0x80000000 },
	{ 0x80008081, 0x80000000 },
	{ 0x00008080, 0x80000000 },
	{ 0x80000001, 0x00000000 },
	{ 0x80008008, 0x80000000 }
};
