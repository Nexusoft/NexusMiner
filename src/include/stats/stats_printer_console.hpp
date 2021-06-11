#ifndef NEXUSMINER_STATS_PRINTER_CONSOLE_HPP
#define NEXUSMINER_STATS_PRINTER_CONSOLE_HPP

#include "stats_printer.hpp"
#include "stats_collector.hpp"
#include <spdlog/spdlog.h>

namespace nexusminer {

class Config;

class Stats_printer_console : public Stats_printer {
public:

    Stats_printer_console(Config& config, Stats_collector& stats_collector);

    void print() override;

private:

    Config& m_config;
    Stats_collector& m_stats_collector;
    std::shared_ptr<spdlog::logger> m_logger;
    
};

}
#endif