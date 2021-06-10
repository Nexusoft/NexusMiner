#ifndef NEXUSMINER_STATS_PRINTER_HPP
#define NEXUSMINER_STATS_PRINTER_HPP

namespace nexusminer {

class Stats_printer {
public:

    virtual ~Stats_printer() = default;

    virtual void print() = 0;
};

}
#endif