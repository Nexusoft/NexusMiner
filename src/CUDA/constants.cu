/*******************************************************************************************

 Nexus Earth 2018

 (credits: cbuchner1 for sieving)

 [Scale Indefinitely] BlackJack. http://www.opensource.org/licenses/mit-license.php

*******************************************************************************************/
#include <CUDA/include/constants.h>

//PRIME/////////////////////////////////////////////////////////////////////////
__constant__ uint64_t c_zTempVar[17];
__constant__ uint32_t c_offsetsA[16];
__constant__ uint32_t c_offsetsB[16];
__constant__ uint32_t c_offsetsT[16];
__constant__ uint16_t c_primes[4096];

__constant__ uint64_t c_primorial;
__constant__ uint32_t c_zFirstSieveElement[WORD_MAX];
__constant__ uint32_t c_quit;



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
