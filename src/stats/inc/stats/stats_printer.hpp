#ifndef NEXUSMINER_STATS_PRINTER_HPP
#define NEXUSMINER_STATS_PRINTER_HPP

#include "stats_collector.hpp"
#include "stats/types.hpp"
#include <string>
#include <sstream>

namespace nexusminer {
namespace stats
{
class Printer {
public:

    virtual ~Printer() = default;

    virtual void print() = 0;
};

class Printer_solo
{
public:

    static std::string print_global(Collector const& stats_collector)
    {
        auto const global_stats = stats_collector.get_global_stats();
        std::stringstream ss;
        ss << "Blocks accepted: " << global_stats.m_accepted_blocks
            << " rejected: " << global_stats.m_rejected_blocks;
        ss << " Connection retry attempts: " << global_stats.m_connection_retries << std::endl;

        return ss.str();
    }
};

class Printer_pool
{
public:

    static std::string print_global(Collector const& stats_collector)
    {
        auto const global_stats = stats_collector.get_global_stats();
        std::stringstream ss;
        ss << "Shares accepted: " << global_stats.m_accepted_shares
            << " rejected: " << global_stats.m_rejected_shares;
        ss << " Connection retry attempts: " << global_stats.m_connection_retries << std::endl;

        return ss.str();
    }
};

}
}
#endif