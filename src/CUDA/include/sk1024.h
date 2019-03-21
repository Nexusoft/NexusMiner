
#pragma once
#ifndef NEXUS_CUDA_SK1024_H
#define NEXUS_CUDA_SK1024_H



//#ifdef __cplusplus
//extern "C" {
//#endif

#include <LLC/types/uint1024.h>
#include <cstdint>

void cuda_sk1024_init(uint32_t thr_id);

void cuda_sk1024_free(uint32_t thr_id);

void cuda_sk1024_set_Target(const void *ptarget);

void cuda_sk1024_setBlock(void *pdata, uint32_t nHeight);

extern bool cuda_sk1024_hash(uint32_t thr_id,
                             uint32_t* TheData,
                             uint1024_t TheTarget,
                             uint64_t &TheNonce,
                             uint64_t *hashes_done,
                             uint32_t throughput,
                             uint32_t threadsPerBlockSkein = 256,
                             uint32_t nHeight = 0);

//#ifdef __cplusplus
//}
//#endif

#endif
