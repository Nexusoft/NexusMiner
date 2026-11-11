// Host-stubs regression test for the prime-channel vOffsets length-8 path.
//
// The gpu-host-stubs preset is intentionally CUDA-free, so this test drives the
// shared prime validation and worker submit-gate logic directly with a synthetic
// known length-8 Cunningham cluster instead of waiting for a live network event.

#include "cpu/prime_validation.hpp"

#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

int tests_run = 0;
int tests_failed = 0;

void check(bool condition, const char* label)
{
    ++tests_run;
    if (condition) {
        std::cout << "  [PASS] " << label << '\n';
    } else {
        ++tests_failed;
        std::cout << "  [FAIL] " << label << '\n';
    }
}

void test_known_length8_candidate_serializes_11_byte_offsets()
{
    // 83 starts this exact dense cluster:
    //   83, 89, 97, 101, 103, 107, 109, 113
    // with gap offsets:
    //   6, 8, 4, 2, 4, 2, 4
    // The next prime is 127, which is beyond the maximum gap accepted by
    // GetOffsetsImpl's nOffset <= 12 loop, so the serialized chain length is
    // exactly 8 and the vector must be 7 gap bytes + 4 fraction bytes.
    const uint1024_t hash_prime{83};
    std::vector<std::uint8_t> offsets;
    double difficulty = 0.0;

    const bool valid = nexusminer::prime::ValidatePrimeCandidate(
        hash_prime,
        8.0,
        offsets,
        difficulty);

    const std::vector<std::uint8_t> expected_gaps{6, 8, 4, 2, 4, 2, 4};

    check(valid, "known synthetic length-8 candidate validates at required difficulty 8.0");
    check(difficulty >= 8.0 && difficulty < 9.0,
          "difficulty is in the length-8 bucket");
    check(offsets.size() == expected_gaps.size() + nexusminer::prime::kPrimeOffsetFractionBytes,
          "length-8 candidate serializes to 11-byte vOffsets");

    bool gaps_match = offsets.size() >= expected_gaps.size();
    for (std::size_t i = 0; gaps_match && i < expected_gaps.size(); ++i) {
        gaps_match = offsets[i] == expected_gaps[i];
    }
    check(gaps_match, "first 7 vOffsets bytes are the expected Cunningham gaps");

    check(nexusminer::prime::is_well_formed_prime_offsets(offsets),
          "worker submit-path vOffsets sanity gate accepts the length-8 serialization");
}

} // namespace

int main()
{
    std::cout << "prime_voffsets_length8_host_stubs_test\n";

    test_known_length8_candidate_serializes_11_byte_offsets();

    if (tests_failed != 0) {
        std::cout << "\nprime_voffsets_length8_host_stubs_test: " << tests_failed
                  << " of " << tests_run << " test(s) failed.\n";
        return EXIT_FAILURE;
    }

    std::cout << "\nprime_voffsets_length8_host_stubs_test: all " << tests_run
              << " test(s) passed.\n";
    return EXIT_SUCCESS;
}
