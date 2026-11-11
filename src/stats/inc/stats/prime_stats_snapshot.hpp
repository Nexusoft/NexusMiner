#ifndef NEXUSMINER_PRIME_STATS_SNAPSHOT_HPP
#define NEXUSMINER_PRIME_STATS_SNAPSHOT_HPP

#include <array>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <utility>

namespace nexusminer {
namespace stats
{

// Bucket k (0..kPrimeHistogramBuckets-1) holds the count of chains whose
// best Fermat run had length k.  Sized to 12 so post-#675 chain lengths
// 10-11 are observable; bumped from 11 in lockstep with the CPU Sieve's
// m_chain_histogram size (see chain_sieve.cpp reset_stats).
inline constexpr std::size_t kPrimeHistogramBuckets = 12;
using Prime_histogram = std::array<std::uint32_t, kPrimeHistogramBuckets>;

// Snapshot of per-worker sieve diagnostic counters. Published from the worker
// thread alongside the rest of the Prime snapshot so the stats path never has
// to read live (mutable) sieve atomics directly.
struct Prime_sieve_diag
{
    std::uint64_t m_sieve_calls{0};
    std::uint64_t m_inner_hits{0};
    std::uint64_t m_starting_multiples_us{0};
    std::uint32_t m_prime_count{0};
};

template <typename Histogram>
Prime_histogram copy_prime_histogram(const Histogram& histogram)
{
    Prime_histogram snapshot{};
    const auto count = std::min(snapshot.size(), histogram.size());
    for (std::size_t i = 0; i < count; ++i)
    {
        snapshot[i] = histogram[i];
    }
    return snapshot;
}

template <typename Count>
std::uint32_t saturating_prime_stat(Count count)
{
    constexpr auto max_count = std::numeric_limits<std::uint32_t>::max();
    if (count > max_count)
    {
        return max_count;
    }
    return static_cast<std::uint32_t>(count);
}

template <typename Stats>
class Atomic_snapshot {
public:
    Atomic_snapshot()
        : m_snapshot(std::make_shared<const Stats>())
    {
    }

    void store(Stats snapshot)
    {
#if defined(__cpp_lib_atomic_shared_ptr) && __cpp_lib_atomic_shared_ptr >= 201711L
        m_snapshot.store(
            std::make_shared<const Stats>(std::move(snapshot)),
            std::memory_order_release);
#else
        std::scoped_lock lock(m_snapshot_mutex);
        m_snapshot = std::make_shared<const Stats>(std::move(snapshot));
#endif
    }

    std::shared_ptr<const Stats> load() const
    {
#if defined(__cpp_lib_atomic_shared_ptr) && __cpp_lib_atomic_shared_ptr >= 201711L
        return m_snapshot.load(std::memory_order_acquire);
#else
        std::scoped_lock lock(m_snapshot_mutex);
        return m_snapshot;
#endif
    }

private:
#if defined(__cpp_lib_atomic_shared_ptr) && __cpp_lib_atomic_shared_ptr >= 201711L
    std::atomic<std::shared_ptr<const Stats>> m_snapshot;
#else
    std::shared_ptr<const Stats> m_snapshot;
    mutable std::mutex m_snapshot_mutex;
#endif
};

}
}

#endif
