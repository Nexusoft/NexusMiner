/*******************************************************************************************

 Nexus Earth 2018

 (credits: cbuchner1 for sieving)

 [Scale Indefinitely] BlackJack. http://www.opensource.org/licenses/mit-license.php

*******************************************************************************************/
#pragma once
#ifndef NEXUS_CUDA_COMBO_SIEVE_H
#define NEXUS_CUDA_COMBO_SIEVE_H

#include <CUDA/include/macro.h>
#include <CUDA/include/frame_resources.h>
#include <CUDA/include/sieve_resources.h>
#include <CUDA/include/streams_events.h>
#include <CUDA/include/prime_helper.cuh>
#include <CUDA/include/constants.h>

void comboA_launch(uint8_t thr_id,
                   uint8_t str_id,
                   uint32_t origin_index,
                   uint8_t frame_index,
                   uint16_t nPrimorialEndPrime,
                   uint16_t nPrimeLimitA,
                   uint32_t nBitArray_Size,
                   uint32_t nOrigins);

void comboB_launch(uint8_t thr_id,
                   uint8_t str_id,
                   uint32_t origin_index,
                   uint8_t frame_index,
                   uint16_t nPrimorialEndPrime,
                   uint32_t nPrimeLimit,
                   uint32_t nBitArray_Size);

void kernel_ccompact_launch(uint8_t thr_id,
                            uint8_t str_id,
                            uint32_t origin_index,
                            uint32_t nMaxCandidates,
                            uint8_t curr_sieve,
                            uint8_t curr_test,
                            uint8_t next_test,
                            uint32_t nBitArray_Size,
                            uint8_t threshold);

#endif
