#include "stats/stats_printer_console.hpp"
#include <sstream>
#include <iostream>

namespace nexusminer
{

Stats_printer_console::Stats_printer_console(Stats_collector& stats_collector)
: m_stats_collector{stats_collector}
, m_logger{spdlog::get("logger")}
{
}

void Stats_printer_console::print()
{
    // TODO print the statistics from stats_collector for each worker
}


}