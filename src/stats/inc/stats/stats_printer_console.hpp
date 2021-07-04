#ifndef NEXUSMINER_STATS_PRINTER_CONSOLE_HPP
#define NEXUSMINER_STATS_PRINTER_CONSOLE_HPP

#include "stats/stats_printer.hpp"
#include "stats/stats_collector.hpp"
#include "config/types.hpp"
#include <spdlog/spdlog.h>

namespace nexusminer {
namespace config { class Worker_config; }
namespace stats
{

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

}
}
#endif