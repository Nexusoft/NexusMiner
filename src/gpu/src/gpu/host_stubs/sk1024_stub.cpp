// Host-stubs build — stub implementations of the CUDA sk1024 hash functions
// declared in cuda_hash/sk1024.h.  Compiled only when WITH_GPU_HOST_STUBS=ON.
//
// cuda_sk1024_hash contract (mirrors sk1024.cu:758-837):
//   - first_nonce = TheNonce (= ((uint64_t*)TheData)[26] by the aliasing
//     invariant enforced in worker_hash.cpp's static_assert).
//   - On miss: TheData[26] += throughput; *hashes_done = new_nonce - first_nonce + 1.
//   - On credited winner: TheNonce = foundNonce; *hashes_done = foundNonce - first_nonce + 1.
//   - On keccak mismatch (winner rejected by host): *hashes_done = 0;
//     *keccak_mismatches += 1; return false.
//
// Behaviour is controlled by two globals that the integration test sets
// directly before each call (much simpler than env-var parsing in a unit-
// test context; the globals are intentionally non-atomic because the test
// is single-threaded at the point it manipulates them):
//
//   nexusminer::gpu::stub_sk1024_win_on_call   (-1 = never win; >=0 = win
//       on exactly that call number, 0-based).
//   nexusminer::gpu::stub_sk1024_keccak_mismatch (if true AND call is the
//       winning call, simulate a keccak mismatch instead of a real win).
//
// Both variables are reset to "no winner" / "no mismatch" by
//   nexusminer::gpu::stub_sk1024_reset()
// which the test calls between subtests.

#include "../cuda_hash/sk1024.h"

#include <atomic>
#include <cstdint>
#include <cstring>

namespace nexusminer {
namespace gpu {

// Behaviour-control variables — written by the integration test.
int  stub_sk1024_win_on_call    = -1;   // -1 = never win
bool stub_sk1024_keccak_mismatch = false;

// Monotonic call counter — incremented at the start of each
// cuda_sk1024_hash invocation.  Reset by stub_sk1024_reset().
static int s_call_count = 0;

void stub_sk1024_reset()
{
    stub_sk1024_win_on_call     = -1;
    stub_sk1024_keccak_mismatch = false;
    s_call_count                = 0;
}

int stub_sk1024_call_count() { return s_call_count; }

}  // namespace gpu
}  // namespace nexusminer

// No-op lifecycle stubs.

void cuda_sk1024_init(uint32_t /*thr_id*/)
{
    // no-op
}

void cuda_sk1024_free(uint32_t /*thr_id*/)
{
    // no-op
}

void cuda_sk1024_set_Target(const void* /*ptarget*/)
{
    // no-op
}

void cuda_sk1024_setBlock(void* /*pdata*/, uint32_t /*nHeight*/)
{
    // no-op
}

// The heart of the stub — mirrors sk1024.cu:758-837 in plain C++.
bool cuda_sk1024_hash(uint32_t /*thr_id*/,
                      uint32_t* TheData,
                      uint1024_t /*TheTarget*/,
                      uint64_t& TheNonce,
                      uint64_t* hashes_done,
                      uint32_t throughput,
                      uint32_t /*threadsPerBlockSkein*/,
                      uint32_t /*nHeight*/,
                      uint32_t* keccak_mismatches)
{
    using namespace nexusminer::gpu;

    const int this_call = s_call_count++;

    // Alias: TheNonce must == ((uint64_t*)TheData)[26] on entry
    // (the worker_hash.cpp static_assert enforces this layout).
    uint64_t* nonce_slot   = reinterpret_cast<uint64_t*>(TheData) + 26;
    const uint64_t first_nonce = TheNonce;  // == *nonce_slot on entry

    const bool is_win  = (stub_sk1024_win_on_call >= 0 &&
                          this_call == stub_sk1024_win_on_call);
    const bool is_miss = (stub_sk1024_keccak_mismatch && is_win);

    if (is_win)
    {
        // Winning nonce lands just before the end of the current window.
        const uint64_t found_nonce = first_nonce + throughput - 1u;
        *nonce_slot = found_nonce;

        if (!is_miss)
        {
            // Credited winner — re-anchor TheNonce (mirrors sk1024.cu:796).
            TheNonce      = found_nonce;
            *hashes_done  = found_nonce - first_nonce + 1u;
            return true;
        }
        else
        {
            // Keccak mismatch — do NOT credit hashes, do NOT re-anchor
            // TheNonce (mirrors sk1024.cu:810-816).
            if (keccak_mismatches != nullptr)
                *keccak_mismatches += 1u;
            *hashes_done = 0u;
            return false;
        }
    }

    // Non-winner — advance nonce slot by throughput (mirrors sk1024.cu:822).
    *nonce_slot += throughput;
    const uint64_t done_nonce = *nonce_slot;
    if (done_nonce >= first_nonce)
        *hashes_done = done_nonce - first_nonce + 1u;
    else
        *hashes_done = 0u;  // overflow-safe guard (mirrors sk1024.cu:834)
    return false;
}
