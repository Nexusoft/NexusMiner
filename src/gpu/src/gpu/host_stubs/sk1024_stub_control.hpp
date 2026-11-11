#ifndef NEXUSMINER_GPU_HOST_STUBS_SK1024_STUB_CONTROL_HPP
#define NEXUSMINER_GPU_HOST_STUBS_SK1024_STUB_CONTROL_HPP

// Integration-test control interface for the cuda_sk1024_hash stub.
// Only include this header in test code built with WITH_GPU_HOST_STUBS=ON.

namespace nexusminer {
namespace gpu {

// Set to the 0-based call number that should return a winner.
// -1 (default) means the stub never returns a credited win.
extern int  stub_sk1024_win_on_call;

// If true AND the current call == stub_sk1024_win_on_call, the stub
// simulates a keccak (CPU-revalidation) mismatch instead of crediting
// a real winner.
extern bool stub_sk1024_keccak_mismatch;

// Reset all control variables and the call counter to their default
// (never-win / no-mismatch) state.  Call between subtests.
void stub_sk1024_reset();

// Returns the number of cuda_sk1024_hash calls made since the last reset.
int stub_sk1024_call_count();

}  // namespace gpu
}  // namespace nexusminer

#endif  // NEXUSMINER_GPU_HOST_STUBS_SK1024_STUB_CONTROL_HPP
