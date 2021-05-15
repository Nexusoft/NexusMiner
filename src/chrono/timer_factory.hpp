#ifndef NEXUSMINER_CHRONO_TIMER_FACTORY_HPP
#define NEXUSMINER_CHRONO_TIMER_FACTORY_HPP

#include "timer.hpp"

#include "asio/io_service.hpp"

#include <memory>

namespace nexusminer {
namespace chrono {

class Timer_factory
{
public:
	using Sptr = std::shared_ptr<Timer_factory>;

    explicit Timer_factory(std::shared_ptr<asio::io_context> io_context)
        : m_io_context{std::move(io_context)}
    {
    }

    Timer::Uptr create_timer()  { return std::make_unique<Timer>(m_io_context); }

private:
    std::shared_ptr<asio::io_context> m_io_context;
};


}
}

#endif 
