/*******************************************************************************************

 Nexus Earth 2018

 [Scale Indefinitely] BlackJack. http://www.opensource.org/licenses/mit-license.php

*******************************************************************************************/
#ifndef NEXUS_CUDA_FERMAT_H
#define NEXUS_CUDA_FERMAT_H

#include <cstdint>
#include <CUDA/include/macro.h>

extern "C" void cuda_set_primorial(uint8_t thr_id, uint64_t nPrimorial);

extern "C" void cuda_set_BaseOrigin(uint32_t thr_id, uint32_t *limbs);

extern "C" void cuda_init_counts(uint32_t thr_id);

extern "C" void cuda_set_quit(uint32_t quit);

extern "C" void cuda_fermat(uint32_t thr_id,
                            uint32_t sieve_index,
                            uint32_t test_index,
                            uint32_t nTestLevels,
                            uint32_t nMaxCandidates);

extern "C" void cuda_results(uint32_t thr_id,
                             uint32_t test_index,
                             uint64_t *result_offsets,
                             uint32_t *result_meta,
                             uint32_t *result_count,
                             uint32_t *primes_checked,
                             uint32_t *primes_found,
                             bool fSynchronize = false);

#endif
