#ifndef NEXUSMINER_STATS_PRINTER_FILE_HPP
#define NEXUSMINER_STATS_PRINTER_FILE_HPP

#include "stats/stats_printer_base.hpp"
#include "spdlog/sinks/basic_file_sink.h"

namespace nexusminer {
namespace stats
{

template<typename PrinterType>
class Printer_file : public Printer_base<PrinterType> {
public:
    Printer_file(std::string const& filename, config::Mining_mode mining_mode,
                 std::vector<config::Worker_config> const& worker_config,
                 Collector& stats_collector)
        : Printer_base<PrinterType>{ mining_mode, worker_config, stats_collector,
                                     spdlog::basic_logger_mt(
                                         "statistics_file",
                                         filename.empty() ? "stats.log" : filename,
                                         true),
                                     /*flush_after_emit=*/true }
    {}
};

}
}
#endif
