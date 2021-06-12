#include "timer_manager.hpp"
#include "network/endpoint.hpp"
#include "network/connection.hpp"
#include "packet.hpp"
#include "worker_manager.hpp"
#include "config/config.hpp"
#include "stats/stats_collector.hpp"
#include "stats/stats_printer_console.hpp"
#include "worker.hpp"

namespace nexusminer
{
Timer_manager::Timer_manager(Config& config, Stats_collector& stats_collector, chrono::Timer_factory::Sptr timer_factory)
: m_config{config}
, m_stats_collector{stats_collector}
, m_stats_printer{std::make_unique<Stats_printer_console>(config, stats_collector)} // TODO create printer from config
, m_timer_factory{std::move(timer_factory)}
{
    m_connection_retry_timer = m_timer_factory->create_timer();
    m_get_height_timer = m_timer_factory->create_timer();
    m_stats_printer_timer = m_timer_factory->create_timer();
}

void Timer_manager::start_connection_retry_timer(std::weak_ptr<Worker_manager> worker_manager, 
    network::Endpoint const& wallet_endpoint)
{
    auto const connection_retry_interval = m_config.get_connection_retry_interval();
    m_connection_retry_timer->start(chrono::Seconds(connection_retry_interval), 
        connection_retry_handler(std::move(worker_manager), wallet_endpoint));
}

void Timer_manager::start_get_height_timer(std::weak_ptr<network::Connection> connection,std::vector<std::shared_ptr<Worker>> m_workers)
{
    auto const get_height_interval = m_config.get_height_interval();
    m_get_height_timer->start(chrono::Seconds(get_height_interval), get_height_handler(std::move(connection), m_workers, get_height_interval));
}

void Timer_manager::start_stats_printer_timer()
{
    auto const stats_printer_interval = m_config.get_print_statistics_interval();
    m_stats_printer_timer->start(chrono::Seconds(stats_printer_interval), stats_printer_handler(stats_printer_interval));
}

void Timer_manager::stop()
{
    m_connection_retry_timer->cancel();
    m_get_height_timer->cancel();
    m_stats_printer_timer->cancel();
}

chrono::Timer::Handler Timer_manager::connection_retry_handler(std::weak_ptr<Worker_manager> worker_manager,
    network::Endpoint const& wallet_endpoint)
{
    return[worker_manager, wallet_endpoint](bool canceled)
    {
        if (canceled)	// don't do anything if the timer has been canceled
        {
            return;
        }

        auto worker_manager_shared = worker_manager.lock();
        if(worker_manager_shared)
        {
            worker_manager_shared->connect(wallet_endpoint);
        }
    }; 
}

chrono::Timer::Handler Timer_manager::get_height_handler(std::weak_ptr<network::Connection> connection, 
    std::vector<std::shared_ptr<Worker>> m_workers, std::uint16_t get_height_interval)
{
    return[this, connection, m_workers, get_height_interval](bool canceled)
    {
        if (canceled)	// don't do anything if the timer has been canceled
        {
            return;
        }

        auto connection_shared = connection.lock();
        if(connection_shared)
        {
            Packet packet_get_height;
            packet_get_height.m_header = Packet::NEW_BLOCK;
            connection_shared->transmit(packet_get_height.get_bytes());

            for(auto& worker : m_workers)
            {
                worker->update_statistics(m_stats_collector);
            }
            // restart timer
            m_get_height_timer->start(chrono::Seconds(get_height_interval), 
                get_height_handler(std::move(connection_shared), m_workers, get_height_interval));
        }
    }; 
}

chrono::Timer::Handler Timer_manager::stats_printer_handler(std::uint16_t stats_printer_interval)
{
    return[this, stats_printer_interval](bool canceled)
    {
        if (canceled)	// don't do anything if the timer has been canceled
        {
            return;
        }

        m_stats_printer->print();

        // restart timer
         m_stats_printer_timer->start(chrono::Seconds(stats_printer_interval), stats_printer_handler(stats_printer_interval));
    }; 
}

}