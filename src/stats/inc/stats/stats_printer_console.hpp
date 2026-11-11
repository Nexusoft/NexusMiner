#ifndef NEXUSMINER_STATS_PRINTER_CONSOLE_HPP
#define NEXUSMINER_STATS_PRINTER_CONSOLE_HPP

#include "stats/stats_printer_base.hpp"
#include "spdlog/sinks/stdout_color_sinks.h"

namespace nexusminer {
namespace stats
{

template<typename PrinterType>
class Printer_console : public Printer_base<PrinterType> {
public:
    Printer_console(config::Mining_mode mining_mode,
                    std::vector<config::Worker_config> const& worker_config,
                    Collector& stats_collector)
        : Printer_base<PrinterType>{ mining_mode, worker_config, stats_collector,
                                     spdlog::stdout_color_mt("statistics"),
                                     /*flush_after_emit=*/false }
    {}
};

}
}
#endif
