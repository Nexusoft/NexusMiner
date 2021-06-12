#ifndef NEXUSMINER_STATS_PRINTER_CONSOLE_HPP
#define NEXUSMINER_STATS_PRINTER_CONSOLE_HPP

#include "stats_printer.hpp"
#include "stats_collector.hpp"
#include "config/types.hpp"
#include <spdlog/spdlog.h>

namespace nexusminer {
namespace config { class Worker_config; }

class Stats_printer_console : public Stats_printer {
public:

    Stats_printer_console(config::Mining_mode mining_mode,
        std::vector<config::Worker_config> const& worker_config, Stats_collector& stats_collector);

    void print() override;

private:

    config::Mining_mode m_mining_mode;
    std::vector<config::Worker_config> const& m_worker_config;
    Stats_collector& m_stats_collector;
    std::shared_ptr<spdlog::logger> m_logger;
    
};

}
#endif