// Tests for cpu::Sieving_prime_table — process-singleton holding the
// immutable CPU sieving prime list (Stone 1 of the Option-3 plan).

#include "cpu/prime/sieving_prime_table.hpp"

#include <spdlog/sinks/null_sink.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdint>
#include <iostream>

namespace
{
int tests_run = 0;
int tests_failed = 0;

void print_result(const char* name, bool passed)
{
    ++tests_run;
    std::cout << (passed ? "  [PASS] " : "  [FAIL] ") << name << '\n';
    if (!passed)
        ++tests_failed;
}

void test_instance_returns_same_reference()
{
    auto& a = nexusminer::cpu::Sieving_prime_table::instance();
    auto& b = nexusminer::cpu::Sieving_prime_table::instance();
    print_result("instance() returns same reference across calls", &a == &b);
}

void test_primes_non_empty()
{
    const auto& primes = nexusminer::cpu::Sieving_prime_table::instance().primes();
    print_result("primes() non-empty", !primes.empty());
}

void test_primes_sorted_descending()
{
    const auto& primes = nexusminer::cpu::Sieving_prime_table::instance().primes();
    // Canonical iteration order for sieve_segment(): large primes first.
    const bool sorted_desc =
        std::is_sorted(primes.begin(), primes.end(), std::greater<std::uint32_t>{});
    print_result("primes() sorted strictly large-prime-first", sorted_desc);

    // Cheap independent witness: front() must dominate back().
    print_result("primes().front() > primes().back()",
                 !primes.empty() && primes.front() > primes.back());
}

void test_size_matches_primes_size()
{
    auto& table = nexusminer::cpu::Sieving_prime_table::instance();
    print_result("size() matches primes().size()", table.size() == table.primes().size());
}
} // namespace

int main()
{
    // chain_sieve and sieving_prime_table both call spdlog::get("logger");
    // ensure it exists so any incidental logging is a no-op rather than UB.
    if (!spdlog::get("logger"))
    {
        spdlog::create<spdlog::sinks::null_sink_mt>("logger");
    }

    std::cout << "Running cpu::Sieving_prime_table tests...\n";
    test_instance_returns_same_reference();
    test_primes_non_empty();
    test_primes_sorted_descending();
    test_size_matches_primes_size();

    std::cout << "\nResult: " << (tests_run - tests_failed) << "/" << tests_run << " passed\n";
    return tests_failed == 0 ? 0 : 1;
}
