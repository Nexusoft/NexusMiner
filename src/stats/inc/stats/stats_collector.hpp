#ifndef NEXUSMINER_STATS_COLLECTOR_HPP
#define NEXUSMINER_STATS_COLLECTOR_HPP

#include "stats/types.hpp"
#include <memory>
#include <vector>
#include <variant>
#include <chrono>
#include <mutex>

namespace nexusminer {
namespace config { class Config; }
namespace stats
{

class Collector {
public:

	Collector(config::Config& config);

    void update_global_stats(Global const& stats);
    void update_worker_stats(std::uint16_t internal_worker_id, Hash const& stats);
    void update_worker_stats(std::uint16_t internal_worker_id, Prime const& stats);
    // copy of workers stats
    std::vector<std::variant<Hash, Prime>> get_workers_stats() const { return m_workers; }
    std::chrono::duration<double> get_elapsed_time_seconds() const { return 
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - m_start_time); }

    Global get_global_stats() const { return m_global_stats; }


private:

    config::Config& m_config;
    std::vector<std::variant<Hash, Prime>> m_workers;

    Global m_global_stats;
    std::chrono::steady_clock::time_point m_start_time;

    // worker stats are updated in seperate worker threads
    // the access to the worker data (form stats_printer) has to be protected
    std::mutex m_worker_mutex;


};

}
}
#endif