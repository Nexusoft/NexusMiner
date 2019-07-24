/*******************************************************************************************

 Nexus Earth 2018

 [Scale Indefinitely] BlackJack. http://www.opensource.org/licenses/mit-license.php

*******************************************************************************************/
#ifndef NEXUS_CUDA_SIEVE_H
#define NEXUS_CUDA_SIEVE_H

#include <cstdint>
#include <vector>

extern "C" void cuda_set_zTempVar(uint8_t thr_id, const uint64_t *limbs);

extern "C" void cuda_init_primes(uint8_t thr_id,
                                 uint64_t *origins,
                                 uint32_t *primes,
                                 uint32_t *primesInverseInvk,
                                 uint32_t nPrimeLimit,
                                 uint32_t nBitArray_Size,
                                 uint32_t sharedSizeKB,
                                 uint32_t nPrimorialEndPrime,
                                 uint32_t nPrimeLimitA,
                                 uint32_t nOrigins,
                                 uint32_t nMaxCandidates);

extern "C" void cuda_free_primes(uint8_t thr_id);

extern "C" void cuda_base_remainders(uint8_t thr_id, uint32_t nPrimeLimit);

extern "C" void cuda_set_origins(uint8_t thr_id, uint32_t nPrimeLimitA, uint64_t *origins, uint32_t nOrigins);

extern "C" bool cuda_primesieve(uint8_t thr_id,
                                uint64_t primorial,
                                uint16_t nPrimorialEndPrime,
                                uint16_t nPrimeLimitA,
                                uint32_t nPrimeLimitB,
                                uint32_t nPrimeLimit,
                                uint32_t nBitArray_Size,
                                uint32_t nDifficulty,
                                uint32_t sieve_index,
                                uint32_t test_index,
                                uint32_t nOrigins,
                                uint32_t nMaxCandidates);

extern "C" void cuda_wait_sieve(uint8_t thr_id, uint32_t sieve_index);

extern "C" void cuda_set_sieve(uint8_t thr_id,
                               uint64_t base_offset,
                               uint64_t primorial,
                               uint32_t n_primorial_end,
                               uint32_t n_prime_limit,
                               uint8_t bit_array_size_log2);

extern "C" void cuda_set_offset_patterns(uint8_t thr_id,
                                         const std::vector<uint32_t> &offsets,
                                         const std::vector<uint32_t> &indicesA,
                                         const std::vector<uint32_t> &indicesB,
                                         const std::vector<uint32_t> &indicesT);

#endif
