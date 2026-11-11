#include "stats/stats_collector.hpp"
#include "config/config.hpp"
#include "config/types.hpp"
#include <spdlog/spdlog.h>
#include <cassert>

namespace nexusminer
{
namespace stats
{

// ---------------- Collector base ----------------

Collector::Collector(config::Config& config)
    : m_config{config}
    , m_mining_mode{config.get_mining_mode()}
    , m_worker_count{config.get_worker_config().size()}
    , m_start_time{std::chrono::steady_clock::now()}
{
}

void Collector::update_global_stats(Global_delta const& delta)
{
    std::scoped_lock lock(m_global_mutex);
    m_global_counters += delta;
}

Global Collector::get_global_stats() const
{
    std::scoped_lock lock(m_global_mutex);
    return Global{
        m_global_counters.m_accepted_blocks,
        m_global_counters.m_rejected_blocks,
        m_global_counters.m_accepted_shares,
        m_global_counters.m_rejected_shares,
        m_global_counters.m_connection_retries,
        m_global_counters.m_discarded_blocks,
        m_global_state.m_degraded_mode
    };
}

void Collector::set_degraded_mode(bool degraded)
{
    std::scoped_lock lock(m_global_mutex);
    m_global_state.m_degraded_mode = degraded;
}

void Collector::reset_global_counters()
{
    std::scoped_lock lock(m_global_mutex);
    m_global_counters = Global_delta{};
}

std::chrono::duration<double> Collector::get_elapsed_time_seconds() const
{
    std::chrono::steady_clock::time_point start;
    {
        std::scoped_lock lock(m_global_mutex);
        start = m_start_time;
    }
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start);
}

void Collector::reset_start_time()
{
    std::scoped_lock lock(m_global_mutex);
    m_start_time = std::chrono::steady_clock::now();
}

// ---------------- Worker_stats_collector<T> ----------------

template<typename T>
Worker_stats_collector<T>::Worker_stats_collector(config::Config& config)
    : Collector{config}
    , m_workers(config.get_worker_config().size())
{
}

// Out-of-range worker ids are programming errors (the worker id space is
// fixed at construction from the worker config). We use a consistent
// "assert in debug, safe-fallback in release" stance on both the write and
// read paths: debug builds catch the bug at the call site, release builds
// degrade gracefully (silent drop on write, default-constructed T on read)
// instead of dereferencing past the end of m_workers.
template<typename T>
void Worker_stats_collector<T>::update_worker_stats(std::uint16_t internal_worker_id,
                                                    T const& stats)
{
    std::scoped_lock lock(m_worker_mutex);
    assert(internal_worker_id < m_workers.size());
    if (internal_worker_id >= m_workers.size()) {
        return;
    }
    m_workers[internal_worker_id] = stats;
}

template<typename T>
T Worker_stats_collector<T>::get_worker_stats(std::uint16_t internal_worker_id) const
{
    std::scoped_lock lock(m_worker_mutex);
    assert(internal_worker_id < m_workers.size());
    if (internal_worker_id >= m_workers.size()) {
        return T{};
    }
    return m_workers[internal_worker_id];
}

template<typename T>
std::vector<T> Worker_stats_collector<T>::get_workers_stats() const
{
    std::scoped_lock lock(m_worker_mutex);
    return m_workers;
}

// log_summary: per-T specializations below.
template<typename T>
void Worker_stats_collector<T>::log_summary()
{
    // Default no-op; specialized below for Hash and Prime.
}

template<>
void Worker_stats_collector<Prime>::log_summary()
{
    auto const workers = get_workers_stats();  // takes m_worker_mutex
    auto worker_configs = m_config.get_worker_config();
    for (std::size_t i = 0; i < workers.size() && i < worker_configs.size(); ++i)
    {
        const auto& wc = worker_configs[i];
        const auto& p  = workers[i];
        spdlog::info(
            "Stats: worker={}, primes={}, chains={}, difficulty={:.2f}, cpu_load={:.2f}",
            wc.m_id, p.m_primes, p.m_chains, p.m_difficulty / 10000000.0, p.m_cpu_load);
    }
}

template<>
void Worker_stats_collector<Hash>::log_summary()
{
    auto const workers = get_workers_stats();
    auto worker_configs = m_config.get_worker_config();
    for (std::size_t i = 0; i < workers.size() && i < worker_configs.size(); ++i)
    {
        const auto& wc = worker_configs[i];
        const auto& h  = workers[i];
        spdlog::info(
            "Stats: worker={}, hashes={}, best_zeros={}, difficulty_met={}",
            wc.m_id, h.m_hash_count, h.m_best_leading_zeros, h.m_met_difficulty_count);
    }
}

// Explicit instantiations so the template definitions remain in this TU.
template class Worker_stats_collector<Hash>;
template class Worker_stats_collector<Prime>;

// ---------------- Factory ----------------

std::shared_ptr<Collector> make_collector(config::Config& config)
{
    if (config.get_mining_mode() == config::Mining_mode::HASH) {
        return std::make_shared<Hash_collector>(config);
    }
    return std::make_shared<Prime_collector>(config);
}

}
}
