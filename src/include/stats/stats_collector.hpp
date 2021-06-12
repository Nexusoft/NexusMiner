#ifndef NEXUSMINER_STATS_COLLECTOR_HPP
#define NEXUSMINER_STATS_COLLECTOR_HPP

#include <memory>
#include <vector>
#include <variant>
#include <chrono>
#include <mutex>

namespace nexusminer {
namespace config { class Config; }

struct Stats_hash
{
    std::uint64_t m_hash_count{0};
    int m_best_leading_zeros{0};
    int m_met_difficulty_count{0};
    int m_nonce_candidates_recieved{0};

    Stats_hash& operator+=(Stats_hash const& other)
    {
        m_hash_count += other.m_hash_count;
        m_best_leading_zeros += other.m_best_leading_zeros;
        m_met_difficulty_count += other.m_met_difficulty_count;

        return *this;
    }
};

struct Stats_prime
{
    Stats_prime& operator+=(Stats_prime const& other)
    {
        return *this;
    }
};

class Stats_collector {
public:

    using Config = config::Config;

	Stats_collector(Config& config);

    void update_worker_stats(std::uint16_t internal_worker_id, Stats_hash const& stats);
    void update_worker_stats(std::uint16_t internal_worker_id, Stats_prime const& stats);
    // copy of workers stats
    std::vector<std::variant<Stats_hash, Stats_prime>> get_workers_stats() const { return m_workers; }
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
    std::vector<std::variant<Stats_hash, Stats_prime>> m_workers;

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
#endif