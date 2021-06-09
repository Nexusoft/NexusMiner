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
class Stats_collector;
class Worker_manager;
class Worker;

class Timer_manager
{
public:

    Timer_manager(Config& config, Stats_collector& stats_collector, chrono::Timer_factory::Sptr timer_factory);

    void start_connection_retry_timer(std::weak_ptr<Worker_manager> worker_manager, 
        network::Endpoint const& wallet_endpoint);
    // collect also data from the workers
    void start_get_height_timer(std::weak_ptr<network::Connection> connection, std::vector<std::shared_ptr<Worker>> m_workers);

    void stop();

private:

    chrono::Timer::Handler connection_retry_handler(std::weak_ptr<Worker_manager> worker_manager, 
        network::Endpoint const& wallet_endpoint);
    chrono::Timer::Handler get_height_handler(std::weak_ptr<network::Connection> connection, 
        std::vector<std::shared_ptr<Worker>> m_workers, std::uint16_t get_height_interval);

    Config& m_config;
    Stats_collector& m_stats_collector;
    chrono::Timer_factory::Sptr m_timer_factory;
    chrono::Timer::Uptr m_connection_retry_timer;
    chrono::Timer::Uptr m_get_height_timer;
};
}

#endif
