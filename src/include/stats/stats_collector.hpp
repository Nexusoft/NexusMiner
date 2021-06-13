#ifndef NEXUSMINER_STATS_COLLECTOR_HPP
#define NEXUSMINER_STATS_COLLECTOR_HPP

#include "types.hpp"
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

    using Config = config::Config;

	Collector(Config& config);

    void update_worker_stats(std::uint16_t internal_worker_id, Hash const& stats);
    void update_worker_stats(std::uint16_t internal_worker_id, Prime const& stats);
    // copy of workers stats
    std::vector<std::variant<Hash, Prime>> get_workers_stats() const { return m_workers; }
    std::chrono::duration<double> get_elapsed_time_seconds() const { return 
        std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - m_start_time); }

    void block_accepted() { m_accepted_blocks++; }
    void block_rejected() { m_rejected_blocks++; }
    void connection_retry_attempt() { m_connection_retries++; }

    std::uint32_t get_blocks_accepted() const { return m_accepted_blocks; }
    std::uint32_t get_blocks_rejected() const { return m_rejected_blocks; }
    std::uint32_t get_connection_retry_attempts() const { return m_connection_retries; }

private:

    Config& m_config;
    std::vector<std::variant<Hash, Prime>> m_workers;

    // global stats
    std::chrono::steady_clock::time_point m_start_time;
    std::uint32_t m_accepted_blocks;
    std::uint32_t m_rejected_blocks;
    std::uint32_t m_connection_retries;

    // worker stats are updated in seperate worker threads
    // the access to the worker data (form stats_printer) has to be protected
    std::mutex m_worker_mutex;


};

}
}
#endif