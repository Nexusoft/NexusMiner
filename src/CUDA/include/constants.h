/*******************************************************************************************

 Nexus Earth 2018

 (credits: cbuchner1 for sieving)

 [Scale Indefinitely] BlackJack. http://www.opensource.org/licenses/mit-license.php

*******************************************************************************************/
#pragma once

#include <CUDA/include/macro.h>
#include <cstdint>

//PRIME/////////////////////////////////////////////////////////////////////////
extern __constant__ uint64_t c_primorial;
extern __constant__ uint64_t c_zTempVar[17];
extern __constant__ uint32_t c_zBaseOrigin[WORD_MAX];
extern __constant__ uint32_t c_offsets[OFFSETS_MAX];
extern __constant__ uint32_t c_iA[16];
extern __constant__ uint32_t c_iB[16];
extern __constant__ uint32_t c_iT[16];

extern __constant__ uint32_t c_mark_mask[32];

extern __constant__ uint32_t c_iBeg;
extern __constant__ uint32_t c_iEnd;

extern __constant__ uint32_t c_bitmaskA;
extern __constant__ uint32_t c_bitmaskT;

extern __constant__ uint16_t c_primes[4096];



extern __constant__ uint32_t c_quit;



//HASH//////////////////////////////////////////////////////////////////////////
extern __constant__ uint64_t pTarget[16];
extern __constant__ uint2 uMessage[27];
extern __constant__ uint2 c_hv[17];
extern __constant__ uint2 skein_ks_parity;
extern __constant__ uint2 t12[9];
extern __constant__ uint2 keccak_round_constants[24];
