#include "sieving_prime_table.hpp"

#include "mining/mining_constants.hpp"

#include <algorithm>
#include <chrono>
#include <primesieve.hpp>
#include <spdlog/spdlog.h>
#include <sstream>

namespace nexusminer {
namespace cpu {

namespace {
constexpr std::uint32_t kSievingStartPrime = mining::SIEVING_START_PRIME;
constexpr std::uint32_t kSievingPrimeLimit = mining::CPU_SIEVING_PRIME_LIMIT;
} // namespace

Sieving_prime_table& Sieving_prime_table::instance()
{
    // C++11 guarantees thread-safe initialisation of function-local statics.
    static Sieving_prime_table table;
    return table;
}

Sieving_prime_table::Sieving_prime_table()
{
    auto logger = spdlog::get("logger");
    if (logger)
    {
        logger->info("Generating shared CPU sieving primes up to {}...", kSievingPrimeLimit);
    }

    const auto start = std::chrono::steady_clock::now();
    primesieve::generate_primes(kSievingStartPrime, kSievingPrimeLimit, &m_primes);
    // Canonical iteration order for sieve_segment(): large primes first.  The
    // sort is done once here so every per-worker Sieve gets the same ordering
    // and can keep its mutable wheel state (multiple, wheel_index) in a plain
    // parallel array indexed against this table.
    std::sort(m_primes.begin(), m_primes.end(), std::greater<std::uint32_t>{});
    const auto end = std::chrono::steady_clock::now();

    if (logger)
    {
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::stringstream ss;
        ss << "Done. Shared table holds " << m_primes.size() << " primes (generated in "
           << std::fixed << elapsed.count() / 1000.0 << "s, reused by all CPU prime workers).";
        logger->info(ss.str());
    }
}

} // namespace cpu
} // namespace nexusminer
