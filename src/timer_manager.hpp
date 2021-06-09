#ifndef NEXUSMINER_TIMER_MANAGER_HPP
#define NEXUSMINER_TIMER_MANAGER_HPP

#include "chrono/timer_factory.hpp"
#include "chrono/timer.hpp"

#include <memory>

namespace nexusminer 
{
namespace network
{
    class Endpoint;
    class Connection;
}

class Config;
class Worker_manager;

class Timer_manager
{
public:

    Timer_manager(Config& config, chrono::Timer_factory::Sptr timer_factory);

    void start_connection_retry_timer(std::weak_ptr<Worker_manager> worker_manager, 
        network::Endpoint const& wallet_endpoint);
    void start_get_height_timer(std::weak_ptr<network::Connection> connection);

    void stop();

private:

    chrono::Timer::Handler connection_retry_handler(std::weak_ptr<Worker_manager> worker_manager, 
        network::Endpoint const& wallet_endpoint);
    chrono::Timer::Handler get_height_handler(std::weak_ptr<network::Connection> connection, 
        std::uint16_t get_height_interval);

    Config& m_config;
    chrono::Timer_factory::Sptr m_timer_factory;
    chrono::Timer::Uptr m_connection_retry_timer;
    chrono::Timer::Uptr m_get_height_timer;
};
}

#endif
