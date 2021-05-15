#ifndef NEXUSMINER_STATISTICS_HPP
#define NEXUSMINER_STATISTICS_HPP

#include <memory>
#include <functional>

namespace nexusminer {

class Statistics {
public:

	virtual ~Statistics() = default;

    virtual void print() = 0;

};

}


#endif