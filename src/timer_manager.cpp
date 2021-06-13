#include "timer_manager.hpp"
#include "network/endpoint.hpp"
#include "network/connection.hpp"
#include "packet.hpp"
#include "worker_manager.hpp"
#include "stats/stats_collector.hpp"
#include "stats/stats_printer.hpp"
#include "worker.hpp"

namespace nexusminer
{
Timer_manager::Timer_manager(chrono::Timer_factory::Sptr timer_factory)
: m_timer_factory{std::move(timer_factory)}
{
    m_connection_retry_timer = m_timer_factory->create_timer();
    m_get_height_timer = m_timer_factory->create_timer();
    m_stats_printer_timer = m_timer_factory->create_timer();
}

void Timer_manager::start_connection_retry_timer(std::uint16_t timer_interval, std::weak_ptr<Worker_manager> worker_manager, 
    network::Endpoint const& wallet_endpoint)
{
    m_connection_retry_timer->start(chrono::Seconds(timer_interval), 
        connection_retry_handler(std::move(worker_manager), wallet_endpoint));
}

void Timer_manager::start_get_height_timer(std::uint16_t timer_interval,  std::weak_ptr<network::Connection> connection,
    std::vector<std::shared_ptr<Worker>> m_workers, std::shared_ptr<stats::Collector> stats_collector)
{
    m_get_height_timer->start(chrono::Seconds(timer_interval), get_height_handler(std::move(connection), m_workers,
        timer_interval, std::move(stats_collector)));
}

void Timer_manager::start_stats_printer_timer(std::uint16_t timer_interval, std::vector<std::shared_ptr<stats::Printer>> stats_printers)
{
    m_stats_printer_timer->start(chrono::Seconds(timer_interval), stats_printer_handler(timer_interval, std::move(stats_printers)));
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
    std::vector<std::shared_ptr<Worker>> m_workers, std::uint16_t get_height_interval, std::shared_ptr<stats::Collector> stats_collector)
{
    return[this, connection, m_workers, get_height_interval, stats_collector = std::move(stats_collector)](bool canceled)
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
                worker->update_statistics(*stats_collector);
            }
            // restart timer
            m_get_height_timer->start(chrono::Seconds(get_height_interval), 
                get_height_handler(std::move(connection_shared), m_workers, get_height_interval, std::move(stats_collector)));
        }
    }; 
}

chrono::Timer::Handler Timer_manager::stats_printer_handler(std::uint16_t stats_printer_interval, 
    std::vector<std::shared_ptr<stats::Printer>> stats_printers)
{
    return[this, stats_printer_interval, stats_printers = std::move(stats_printers)](bool canceled)
    {
        if (canceled)	// don't do anything if the timer has been canceled
        {
            return;
        }

        for(auto& stats_printer : stats_printers)
        {
            stats_printer->print();
        }

        // restart timer
         m_stats_printer_timer->start(chrono::Seconds(stats_printer_interval), stats_printer_handler(stats_printer_interval, 
            std::move(stats_printers)));
    }; 
}

}