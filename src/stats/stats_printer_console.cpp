#include "stats/stats_printer_console.hpp"
#include "config/worker_config.hpp"
#include <sstream>
#include <iostream>
#include <variant>
#include <iomanip>

namespace nexusminer
{

Stats_printer_console::Stats_printer_console(config::Mining_mode mining_mode, 
        std::vector<config::Worker_config> const& worker_config, Stats_collector& stats_collector)
: m_mining_mode{mining_mode}
, m_worker_config{worker_config}
, m_stats_collector{stats_collector}
, m_logger{spdlog::get("logger")}
{
}

void Stats_printer_console::print()
{
    // Log global stats
    std::stringstream ss;
    ss << "Blocks accepted: " << m_stats_collector.get_blocks_accepted() 
        << " rejected: " << m_stats_collector.get_blocks_rejected() << std::endl;
    ss << "Connection retry attempts: " << m_stats_collector.get_connection_retry_attempts();
    ss << std::endl;

    auto const workers = m_stats_collector.get_workers_stats();

    auto worker_config_index = 0U;
    for(auto const& worker : workers)
    {
        
        ss << "Worker " << m_worker_config[worker_config_index].m_id << " stats: ";
        if(m_mining_mode == config::Mining_mode::HASH)
        {
            auto& hash_stats = std::get<Stats_hash>(worker);
            ss << std::setprecision(2) << std::fixed << (hash_stats.m_hash_count / static_cast<double>(m_stats_collector.get_elapsed_time_seconds().count())) / 1.0e6 << "MH/s. ";
            ss << (m_worker_config[worker_config_index].m_mode == config::Worker_mode::FPGA ? hash_stats.m_nonce_candidates_recieved : hash_stats.m_met_difficulty_count) 
                << " candidates found. Most difficult: " << hash_stats.m_best_leading_zeros;

        }
        else
        {

        }
        worker_config_index++;
        ss << std::endl;
    }

    m_logger->info(ss.str());
}


}