#ifndef NEXUSMINER_TIMER_MANAGER_HPP
#define NEXUSMINER_TIMER_MANAGER_HPP

#include "chrono/timer_factory.hpp"
#include "chrono/timer.hpp"

#include <chrono>
#include <memory>
#include <vector>

namespace nexusminer 
{
namespace network
{
    class Endpoint;
}
namespace stats
{
    class Printer;
    class Collector;
}
class Worker_manager;
class Worker;


// NOTE: Heartbeat / keepalive is NOT managed by Timer_manager.
// SessionManager::start_keepalive_timer() owns the 170-second SESSION_KEEPALIVE
// heartbeat for the active session lane. The former Packet::PING timer (bare
// header-only ping) was removed because it carried no payload and conveyed no
// height data to the node.
class Timer_manager
{
public:

    Timer_manager(chrono::Timer_factory::Sptr timer_factory);

    void start_connection_retry_timer(std::uint16_t timer_interval, std::weak_ptr<Worker_manager> worker_manager, 
        network::Endpoint const& wallet_endpoint);
    void start_stats_collector_timer(std::uint16_t timer_interval, std::weak_ptr<Worker_manager> worker_manager,
        std::shared_ptr<stats::Collector> stats_collector);
    void start_stats_printer_timer(std::uint16_t timer_interval, std::vector<std::shared_ptr<stats::Printer>> stats_printers);
    
    // Template Staleness Prevention (LLL-TAO PR #131 Client-Side Integration)
    // Uses weak_ptr<Worker_manager> (same pattern as other timers) so the timer
    // survives connection replacements.  Worker_manager::poll_get_round() fetches
    // the current connection and protocol on every tick, avoiding stale weak_ptr
    // captures that silently kill the timer after reconnection.
    void start_get_round_timer(std::uint16_t timer_interval, std::weak_ptr<Worker_manager> worker_manager);
    
    // Template Health Monitoring (Template Validation & Worker Protection)
    void start_template_health_timer(std::uint16_t timer_interval, std::weak_ptr<Worker_manager> worker_manager);

    // Periodic lane health-check / diagnostics timer.
    void start_lane_health_check_timer(std::uint16_t timer_interval,
        std::weak_ptr<Worker_manager> worker_manager);

    // Replacement-pending deadline timer (one-shot).
    // Armed by Worker_manager whenever a PUSH notification leaves the current
    // template marked replacement-pending, with the delay set to the exact
    // remaining time until the deadline. Fires check_template_health() once
    // at (or slightly after) the deadline so an expired promise is caught
    // immediately rather than waiting for the next 30s general-purpose poll
    // tick. Re-arming (start()) before the previous deadline fires simply
    // cancels and reschedules — no proliferation of timer objects.
    void start_replacement_pending_timer(std::chrono::milliseconds delay,
        std::weak_ptr<Worker_manager> worker_manager);

    void stop();

private:

    chrono::Timer::Handler connection_retry_handler(std::weak_ptr<Worker_manager> worker_manager,
        network::Endpoint const& wallet_endpoint);
    chrono::Timer::Handler stats_collector_handler(std::uint16_t stats_collector_interval,
        std::weak_ptr<Worker_manager> worker_manager,
        std::shared_ptr<stats::Collector> stats_collector);
    chrono::Timer::Handler stats_printer_handler(std::uint16_t stats_printer_interval, std::vector<std::shared_ptr<stats::Printer>> stats_printers);
    chrono::Timer::Handler get_round_handler(std::uint16_t get_round_interval,
        std::weak_ptr<Worker_manager> worker_manager);
    chrono::Timer::Handler template_health_handler(std::uint16_t health_check_interval, 
        std::weak_ptr<Worker_manager> worker_manager);

    chrono::Timer::Handler lane_health_check_handler(std::uint16_t health_check_interval,
        std::weak_ptr<Worker_manager> worker_manager);

    chrono::Timer::Handler replacement_pending_handler(std::weak_ptr<Worker_manager> worker_manager);

    chrono::Timer_factory::Sptr m_timer_factory;
    chrono::Timer::Uptr m_connection_retry_timer;
    chrono::Timer::Uptr m_stats_collector_timer;
    chrono::Timer::Uptr m_stats_printer_timer;
    chrono::Timer::Uptr m_get_round_timer;  // Template Staleness Prevention
    chrono::Timer::Uptr m_template_health_timer;  // Template Health Monitoring
    chrono::Timer::Uptr m_lane_health_check_timer;  // Periodic lane health log
    chrono::Timer::Uptr m_replacement_pending_timer;  // One-shot replacement-pending deadline check
};
}

#endif
