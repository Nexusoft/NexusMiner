#include "cpu/prime_validation.hpp"
#include "LLC/types/uint1024.h"

#include <boost/multiprecision/cpp_int.hpp>

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <vector>

// Cross-check between protocol/falcon_constants.hpp::PRIME_VOFFSETS_MAX_SIZE
// and prime/kMaxSerializedPrimeOffsets used to live here as a test-TU
// static_assert.  After the layout-constants consolidation, both names are
// derived from the same definition in protocol/protocol_constants.hpp, so
// the assertion is tautological and has been removed.

using boost_uint1024_t = boost::multiprecision::uint1024_t;

namespace
{
constexpr std::size_t kBoostUint1kLimbBytes = sizeof(boost::multiprecision::limb_type);

int tests_run = 0;
int tests_failed = 0;

void print_result(const char* name, bool passed)
{
    ++tests_run;
    std::cout << (passed ? "  [PASS] " : "  [FAIL] ") << name << '\n';
    if (!passed)
        ++tests_failed;
}

uint1024_t hex_roundtrip_reference(const boost_uint1024_t& value)
{
    std::stringstream stream;
    stream << std::hex << value;

    uint1024_t result;
    result.SetHex(stream.str());
    return result;
}

uint1024_t optimized_conversion(const boost_uint1024_t& value)
{
    uint1024_t result{};
    const auto limb_bytes = value.backend().size() * kBoostUint1kLimbBytes;
    if (limb_bytes > sizeof(result))
        throw std::runtime_error("Boost uint1024 limb storage exceeds LLC uint1024_t size");

    std::memcpy(&result, value.backend().limbs(), limb_bytes);
    return result;
}

void test_optimized_conversion_matches_hex_roundtrip()
{
    const std::vector<boost_uint1024_t> samples = {
        0,
        1,
        2,
        0xFFFFFFFFu,
        boost_uint1024_t{0x1122334455667788ULL},
        (boost_uint1024_t{0x1122334455667788ULL} << 64) | boost_uint1024_t{0x99AABBCCDDEEFF00ULL},
        boost_uint1024_t{1} << 511,
        boost_uint1024_t{1} << 1023,
        (boost_uint1024_t{1} << 1000) | (boost_uint1024_t{1} << 512) | boost_uint1024_t{12345}
    };

    bool matches = true;
    for (const auto& sample : samples)
    {
        if (optimized_conversion(sample) != hex_roundtrip_reference(sample))
        {
            matches = false;
            break;
        }
    }

    print_result("Optimized uint1024 conversion matches legacy hex round-trip", matches);
}

void test_is_well_formed_prime_offsets()
{
    using nexusminer::prime::is_well_formed_prime_offsets;
    using nexusminer::prime::kMinSerializedPrimeOffsets;
    using nexusminer::prime::kMaxSerializedPrimeOffsets;
    using nexusminer::prime::kPrimeOffsetFractionBytes;
    using nexusminer::prime::kMaxRecognisedChainLength;

    // The historical bug was rejecting size != 10 (chain length 7).  The new
    // gate must accept every chain length from 2 (kMin) up through
    // kMaxRecognisedChainLength (kMax) and reject obvious garbage.

    bool all_valid_lengths_accepted = true;
    for (std::size_t chain_length = 2;
         chain_length <= kMaxRecognisedChainLength;
         ++chain_length)
    {
        const std::size_t expected_size =
            (chain_length - 1) + kPrimeOffsetFractionBytes;
        std::vector<uint8_t> offsets(expected_size, 0x02);
        if (!is_well_formed_prime_offsets(offsets))
        {
            all_valid_lengths_accepted = false;
            std::cout << "    chain_length=" << chain_length
                      << " size=" << expected_size << " unexpectedly rejected\n";
        }
    }
    print_result("is_well_formed_prime_offsets accepts all chain lengths 2..kMax",
                 all_valid_lengths_accepted);

    // Boundary: the legacy "10 bytes" length (chain length 7) must still pass.
    print_result("is_well_formed_prime_offsets accepts legacy 10-byte (length 7)",
                 is_well_formed_prime_offsets(std::vector<uint8_t>(10, 0x02)));

    // The motivating bug: 11 bytes (chain length 8) must now pass.
    print_result("is_well_formed_prime_offsets accepts 11-byte (length 8) — the bug fix",
                 is_well_formed_prime_offsets(std::vector<uint8_t>(11, 0x02)));

    // Rejection cases.
    print_result("is_well_formed_prime_offsets rejects empty",
                 !is_well_formed_prime_offsets({}));
    print_result("is_well_formed_prime_offsets rejects size 1",
                 !is_well_formed_prime_offsets(std::vector<uint8_t>(1, 0)));
    print_result("is_well_formed_prime_offsets rejects size kMin-1 (just below minimum)",
                 !is_well_formed_prime_offsets(
                     std::vector<uint8_t>(kMinSerializedPrimeOffsets - 1, 0)));
    print_result("is_well_formed_prime_offsets rejects size kMax+1 (just above maximum)",
                 !is_well_formed_prime_offsets(
                     std::vector<uint8_t>(kMaxSerializedPrimeOffsets + 1, 0)));
    print_result("is_well_formed_prime_offsets rejects garbage size 10000",
                 !is_well_formed_prime_offsets(std::vector<uint8_t>(10000, 0)));

    // Sanity: the constants compose the way the wire format expects.
    const bool composition_ok =
        kMinSerializedPrimeOffsets == 1 + kPrimeOffsetFractionBytes &&
        kMaxSerializedPrimeOffsets ==
            (kMaxRecognisedChainLength - 1) + kPrimeOffsetFractionBytes;
    print_result("kMin/kMax compose from chain length + fraction bytes",
                 composition_ok);
}

}

int main()
{
    test_optimized_conversion_matches_hex_roundtrip();
    test_is_well_formed_prime_offsets();

    if (tests_failed != 0)
    {
        std::cout << "\nprime_validation_test: " << tests_failed << " of " << tests_run
                  << " test(s) failed.\n";
        return 1;
    }

    std::cout << "\nprime_validation_test: all " << tests_run << " test(s) passed.\n";
    return 0;
}
