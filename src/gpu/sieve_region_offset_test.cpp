// Host-only regression for GPU sieve region alignment.
//
// The medium GPU sieve kernels used to compute `start_offset - starting_multiple`
// whenever the region was non-zero.  If the first wheel multiple was still ahead
// of the region, that unsigned subtract wrapped and the kernel crossed off the
// wrong residues.  CPU prime, which is the path verified on a live prime testnet,
// does not have this bug.  region_wheel_multiple_offset() is the shared host/device
// formula now used by the medium and large kernels.

#include "gpu/prime/sieve_utils.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

int g_failures = 0;

void check(bool ok, const char* label)
{
    if (!ok)
    {
        ++g_failures;
        std::fprintf(stderr, "FAIL: %s\n", label);
    }
    else
    {
        std::fprintf(stderr, "ok:   %s\n", label);
    }
}

}  // namespace

int main()
{
    std::fprintf(stderr, "=== gpu_sieve_region_offset_test ===\n");

    check(region_wheel_multiple_offset(0, 250, 101) == 250,
          "region start 0 returns the starting multiple");

    // Old medium kernels did `uint64_t x = 100 - 250`, which wraps, then
    // crossed off a residue 150 away from the real wheel hit.
    check(region_wheel_multiple_offset(100, 250, 101) == 150,
          "region before the first multiple returns the forward distance");

    check(region_wheel_multiple_offset(1000, 30, 7) == 17,
          "region past the first multiple recomputes the wheel offset");

    // Equality must take the recompute path.  Subtracting would return 0, and
    // get_offset_to_next_multiple never returns 0 for a prime greater than 5.
    const auto equal_offset = region_wheel_multiple_offset(100, 100, 7);
    check(equal_offset == 7,
          "equal region start recomputes instead of returning 0");

    check(region_wheel_multiple_offset(50, 10, 0) == 0,
          "a zero prime is rejected");

    if (g_failures != 0)
    {
        std::fprintf(stderr, "FAILED (%d failures)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::fprintf(stderr, "PASS\n");
    return EXIT_SUCCESS;
}
