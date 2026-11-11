#ifndef NEXUSMINER_STATS_COLLECTOR_HPP
#define NEXUSMINER_STATS_COLLECTOR_HPP

#include "stats/types.hpp"
#include "config/types.hpp"
#include <cassert>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <type_traits>
#include <vector>

namespace nexusminer {
namespace config { class Config; }
namespace stats
{

// ------------------------------------------------------------------
// Abstract base. Owns global stats + elapsed time + mining mode.
// Worker-typed storage lives in Worker_stats_collector<T> below so
// the variant runtime tag check is gone and each type-specific code
// path uses static dispatch.
//
// Issue 6A: the global mutex below is independent from the per-worker
// mutex on the derived class, so worker writes (high frequency) and
// printer global-stat reads (every 15s) no longer contend.
// ------------------------------------------------------------------
class Collector {
public:
    virtual ~Collector() = default;

    // ---- Mode / configuration --------------------------------------------
    config::Mining_mode get_mining_mode() const noexcept { return m_mining_mode; }
    std::size_t worker_count() const noexcept { return m_worker_count; }

    // ---- Global stats ----------------------------------------------------
    void update_global_stats(Global_delta const& delta);
    Global get_global_stats() const;
    void set_degraded_mode(bool degraded);
    void reset_global_counters();

    // ---- Elapsed time ----------------------------------------------------
    std::chrono::duration<double> get_elapsed_time_seconds() const;
    void reset_start_time();

    // ---- Per-mode summary log -------------------------------------------
    virtual void log_summary() = 0;

protected:
    explicit Collector(config::Config& config);

    config::Config& m_config;

private:
    config::Mining_mode m_mining_mode;
    std::size_t m_worker_count;

    Global_delta m_global_counters;
    Global_state m_global_state;
    std::chrono::steady_clock::time_point m_start_time;

    // Single mutex for global-stats and start-time; both are tiny and the
    // contended path is per-worker writes (which now live on a separate
    // mutex in the derived class).
    mutable std::mutex m_global_mutex;
};

// ------------------------------------------------------------------
// Typed worker stats container. Replaces the prior
// std::vector<std::variant<Hash, Prime>> and removes runtime tag
// dispatch from the Collector.
// ------------------------------------------------------------------
template<typename T>
class Worker_stats_collector : public Collector {
public:
    explicit Worker_stats_collector(config::Config& config);

    void update_worker_stats(std::uint16_t internal_worker_id, T const& stats);

    // Returns a copy of a single worker's most-recent stats snapshot.
    T get_worker_stats(std::uint16_t internal_worker_id) const;

    // Returns a copy of every worker's most-recent stats snapshot.
    std::vector<T> get_workers_stats() const;

    void log_summary() override;

private:
    std::vector<T> m_workers;

    // Independent of Collector::m_global_mutex (Issue 6A) so worker writes
    // and global-stats reads/writes never contend on a single lock.
    mutable std::mutex m_worker_mutex;
};

using Hash_collector  = Worker_stats_collector<Hash>;
using Prime_collector = Worker_stats_collector<Prime>;

// Safe downcast helper for callers that know (by construction) which typed
// collector they were paired with. In debug builds this asserts the mode
// matches the requested template type, catching wiring mistakes early. In
// release builds it is a zero-cost static_cast.
template<typename T>
inline Worker_stats_collector<T>& as_typed(Collector& c) noexcept
{
    static_assert(std::is_same_v<T, Hash> || std::is_same_v<T, Prime>,
        "as_typed<T>: T must be one of the registered worker stats types "
        "(stats::Hash or stats::Prime).");
#ifndef NDEBUG
    if constexpr (std::is_same_v<T, Hash>) {
        assert(c.get_mining_mode() == config::Mining_mode::HASH);
    } else if constexpr (std::is_same_v<T, Prime>) {
        assert(c.get_mining_mode() == config::Mining_mode::PRIME);
    }
#endif
    return static_cast<Worker_stats_collector<T>&>(c);
}

// Factory: constructs the right typed collector based on the config's
// mining mode. All callers that just hold a base-class pointer remain
// mode-agnostic; only the workers and printers need the typed view.
std::shared_ptr<Collector> make_collector(config::Config& config);

}
}
#endif
