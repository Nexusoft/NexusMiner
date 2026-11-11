#ifndef NEXUSMINER_CPU_PRIME_SIEVING_PRIME_TABLE_HPP
#define NEXUSMINER_CPU_PRIME_SIEVING_PRIME_TABLE_HPP

#include <cstdint>
#include <vector>

namespace nexusminer {
namespace cpu {

// Process-singleton, immutable list of CPU sieving primes.
//
// Stone 1 of the Option-3 plan: the prime list is identical for every CPU
// Worker_prime instance, so generating it once eliminates 18× redundant
// primesieve passes at startup and lets per-worker Sieve instances hold only
// their mutable wheel state (multiple, wheel_index) parallel to this shared
// table.
class Sieving_prime_table
{
public:
    // Returns the singleton.  First call performs the primesieve generation
    // (lazy, thread-safe).  Subsequent calls are O(1).
    static Sieving_prime_table& instance();

    // Immutable view of the sieving primes, sorted large-prime-first so the
    // sieve_segment() hot loop can iterate this order directly while keeping
    // its mutable per-prime state (multiple, wheel_index) in a parallel array
    // indexed against this table.
    const std::vector<std::uint32_t>& primes() const noexcept { return m_primes; }

    std::size_t size() const noexcept { return m_primes.size(); }

    Sieving_prime_table(const Sieving_prime_table&) = delete;
    Sieving_prime_table& operator=(const Sieving_prime_table&) = delete;

private:
    Sieving_prime_table();

    std::vector<std::uint32_t> m_primes;
};

} // namespace cpu
} // namespace nexusminer

#endif
