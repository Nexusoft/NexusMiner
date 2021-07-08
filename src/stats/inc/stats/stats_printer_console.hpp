#ifndef NEXUSMINER_STATS_PRINTER_CONSOLE_HPP
#define NEXUSMINER_STATS_PRINTER_CONSOLE_HPP

#include "stats/stats_printer.hpp"
#include "stats/stats_collector.hpp"
#include "config/worker_config.hpp"
#include "spdlog/sinks/stdout_color_sinks.h"
#include <sstream>
#include <iostream>
#include <variant>
#include <iomanip>

namespace nexusminer {
namespace stats
{

template<typename PrinterType>
class Printer_console : public Printer {
public:

    Printer_console(config::Mining_mode mining_mode,
        std::vector<config::Worker_config> const& worker_config, Collector& stats_collector);

    void print() override;

private:

    config::Mining_mode m_mining_mode;
    std::vector<config::Worker_config> const& m_worker_config;
    Collector& m_stats_collector;
    std::shared_ptr<spdlog::logger> m_logger;
    
};

template<typename PrinterType>
inline Printer_console<PrinterType>::Printer_console(config::Mining_mode mining_mode,
    std::vector<config::Worker_config> const& worker_config, Collector& stats_collector)
    : m_mining_mode{ mining_mode }
    , m_worker_config{ worker_config }
    , m_stats_collector{ stats_collector }
    , m_logger{ spdlog::stdout_color_mt("statistics") }
{
    m_logger->set_pattern("[%D %H:%M:%S.%e][%^%n%$] %v");
}

template<typename PrinterType>
inline void Printer_console<PrinterType>::print()
{
    // Log global stats
    auto const globals_string = PrinterType::print_global(m_stats_collector);

    auto const workers = m_stats_collector.get_workers_stats();
    std::stringstream ss;
    ss << globals_string;

    auto worker_config_index = 0U;
    for (auto const& worker : workers)
    {

        ss << "Worker " << m_worker_config[worker_config_index].m_id << " stats: ";
        if (m_mining_mode == config::Mining_mode::HASH)
        {
            auto& hash_stats = std::get<Hash>(worker);
            ss << std::setprecision(2) << std::fixed << (hash_stats.m_hash_count / static_cast<double>(m_stats_collector.get_elapsed_time_seconds().count())) / 1.0e6 << "MH/s. ";
            ss << (m_worker_config[worker_config_index].m_mode == config::Worker_mode::FPGA ? hash_stats.m_nonce_candidates_recieved : hash_stats.m_met_difficulty_count)
                << " candidates found. Most difficult: " << hash_stats.m_best_leading_zeros;

        }
        else
        {
            auto& prime_stats = std::get<Prime>(worker);

            ss << (prime_stats.m_primes / static_cast<double>(m_stats_collector.get_elapsed_time_seconds().count())) << " PPS ";
            ss << (prime_stats.m_chains / static_cast<double>(m_stats_collector.get_elapsed_time_seconds().count())) << " CPS ";
            ss << " Difficulty " << prime_stats.m_difficulty / 10000000.0;
        }
        worker_config_index++;
        ss << std::endl;
    }

    m_logger->info(ss.str());
}

}
}
#endif