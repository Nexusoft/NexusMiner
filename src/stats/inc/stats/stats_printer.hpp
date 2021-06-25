#ifndef NEXUSMINER_STATS_PRINTER_HPP
#define NEXUSMINER_STATS_PRINTER_HPP

namespace nexusminer {
namespace stats
{

class Printer {
public:

    virtual ~Printer() = default;

    virtual void print() = 0;
};

}
}
#endif