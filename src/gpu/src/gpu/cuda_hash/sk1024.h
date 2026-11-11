#ifndef NEXUS_CUDA_SK1024_H
#define NEXUS_CUDA_SK1024_H



//#ifdef __cplusplus
//extern "C" {
//#endif

#include <LLC/types/uint1024.h>
#include <cstdint>

void cuda_sk1024_init(uint32_t thr_id);

void cuda_sk1024_free(uint32_t thr_id);

void cuda_sk1024_set_Target(const void* ptarget);

void cuda_sk1024_setBlock(void* pdata, uint32_t nHeight);

extern bool cuda_sk1024_hash(uint32_t thr_id,
    uint32_t* TheData,
    uint1024_t TheTarget,
    uint64_t& TheNonce,
    uint64_t* hashes_done,
    uint32_t throughput,
    uint32_t threadsPerBlockSkein = 256,
    uint32_t nHeight = 0,
    // Stone — bug #8 fix: out-counter for keccak (CPU-revalidation)
    // mismatches.  When the GPU returns a winning nonce that fails the
    // host-side Skein/Keccak revalidation, the kernel previously printed
    // a std::cout line and kept mining (silently inflating m_hashes for a
    // hardware-faulty window).  When non-null, sk1024 increments this
    // counter by 1 per mismatch and refuses to credit the suspect window
    // to *hashes_done.  Worker_hash routes the count through spdlog and
    // treats repeated mismatches as a hardware fault.  Default nullptr
    // preserves ABI for any other consumer.
    uint32_t* keccak_mismatches = nullptr);

//#ifdef __cplusplus
//}
//#endif

#endif