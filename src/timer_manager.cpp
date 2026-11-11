#include "timer_manager.hpp"
#include "network/endpoint.hpp"
#include "worker_manager.hpp"
#include "stats/stats_collector.hpp"
#include "stats/stats_printer.hpp"
#include "worker.hpp"

#include <spdlog/spdlog.h>

#include <exception>
#include <optional>

namespace nexusminer
{
Timer_manager::Timer_manager(chrono::Timer_factory::Sptr timer_factory)
: m_timer_factory{std::move(timer_factory)}
{
    m_connection_retry_timer = m_timer_factory->create_timer();
    m_stats_collector_timer = m_timer_factory->create_timer();
    m_stats_printer_timer = m_timer_factory->create_timer();
    m_get_round_timer = m_timer_factory->create_timer();  // Template Staleness Prevention
    m_template_health_timer = m_timer_factory->create_timer();  // Template Health Monitoring
    m_lane_health_check_timer = m_timer_factory->create_timer();  // SIM Link lane health log
    m_replacement_pending_timer = m_timer_factory->create_timer();  // Replacement-pending deadline check
}

void Timer_manager::start_connection_retry_timer(std::uint16_t timer_interval, std::weak_ptr<Worker_manager> worker_manager, 
    network::Endpoint const& wallet_endpoint)
{
    m_connection_retry_timer->start(chrono::Seconds(timer_interval), 
        connection_retry_handler(std::move(worker_manager), wallet_endpoint));
}

void Timer_manager::start_stats_collector_timer(std::uint16_t timer_interval, std::weak_ptr<Worker_manager> worker_manager,
    std::shared_ptr<stats::Collector> stats_collector)
{
    m_stats_collector_timer->start(chrono::Seconds(timer_interval), stats_collector_handler(timer_interval, worker_manager,
        std::move(stats_collector)));
}

void Timer_manager::start_stats_printer_timer(std::uint16_t timer_interval, std::vector<std::shared_ptr<stats::Printer>> stats_printers)
{
    m_stats_printer_timer->start(chrono::Seconds(timer_interval), stats_printer_handler(timer_interval, std::move(stats_printers)));
}

void Timer_manager::start_get_round_timer(std::uint16_t timer_interval, std::weak_ptr<Worker_manager> worker_manager)
{
    m_get_round_timer->start(chrono::Seconds(timer_interval), get_round_handler(timer_interval, std::move(worker_manager)));
}

void Timer_manager::stop()
{
    m_connection_retry_timer->cancel();
    m_stats_collector_timer->cancel();
    m_stats_printer_timer->cancel();
    m_get_round_timer->cancel();  // Template Staleness Prevention
    m_template_health_timer->cancel();  // Template Health Monitoring
    m_lane_health_check_timer->cancel();  // Lane health log
    m_replacement_pending_timer->cancel();  // Replacement-pending deadline check
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
            try {
                worker_manager_shared->connect(wallet_endpoint);
            } catch (const std::exception& e) {
                spdlog::error("[Timer_manager] connection_retry_handler threw: {}", e.what());
            } catch (...) {
                spdlog::error("[Timer_manager] connection_retry_handler threw unknown exception");
            }
        }
    }; 
}

chrono::Timer::Handler Timer_manager::stats_collector_handler(std::uint16_t stats_collector_interval,
    std::weak_ptr<Worker_manager> worker_manager,
    std::shared_ptr<stats::Collector> stats_collector)
{
    return[this, worker_manager, stats_collector_interval, stats_collector = std::move(stats_collector)](bool canceled)
    {
        if (canceled)	// don't do anything if the timer has been canceled
        {
            return;
        }

        try {
            auto worker_manager_shared = worker_manager.lock();
            if (!worker_manager_shared)
            {
                return;  // Worker_manager destroyed (shutdown), don't restart
            }
            worker_manager_shared->collect_worker_statistics();
        } catch (const std::exception& e) {
            spdlog::error("[Timer_manager] stats_collector_handler threw: {} — timer will restart", e.what());
        } catch (...) {
            spdlog::error("[Timer_manager] stats_collector_handler threw unknown exception — timer will restart");
        }

        // restart timer (always, even after exception)
        m_stats_collector_timer->start(chrono::Seconds(stats_collector_interval),
            stats_collector_handler(stats_collector_interval, worker_manager, std::move(stats_collector)));
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

        try {
            for(auto& stats_printer : stats_printers)
            {
                stats_printer->print();
            }
        } catch (const std::exception& e) {
            spdlog::error("[Timer_manager] stats_printer_handler threw: {} — timer will restart", e.what());
        } catch (...) {
            spdlog::error("[Timer_manager] stats_printer_handler threw unknown exception — timer will restart");
        }

        // restart timer (always, even after exception)
         m_stats_printer_timer->start(chrono::Seconds(stats_printer_interval), stats_printer_handler(stats_printer_interval, 
            std::move(stats_printers)));
    }; 
}

chrono::Timer::Handler Timer_manager::get_round_handler(std::uint16_t get_round_interval,
    std::weak_ptr<Worker_manager> worker_manager)
{
    // Capture session generation at timer-schedule time as a cancellation token.
    // When the handler fires, if the generation changed the session died and
    // reconnected — discard the stale callback to avoid sending GET_ROUND on a
    // dead session.  Uses std::optional to distinguish "not captured" (lock
    // failed) from a legitimate generation of 0.
    std::optional<uint64_t> gen_at_schedule;
    if (auto wm = worker_manager.lock()) {
        gen_at_schedule = wm->session_generation();
    }

    return [this, worker_manager, get_round_interval, gen_at_schedule](bool canceled)
    {
        if (canceled)	// don't do anything if the timer has been canceled
        {
            return;
        }

        auto wm = worker_manager.lock();
        if (!wm)
        {
            return;  // Worker_manager destroyed (shutdown), don't restart
        }

        // Session generation guard: if the session changed since this timer tick
        // was scheduled (or was never captured), skip the work.
        const auto gen_now = wm->session_generation();
        if (!gen_at_schedule.has_value() || gen_now != *gen_at_schedule) {
            spdlog::debug("[Timer_manager] get_round_handler: stale session (gen {} → {}), skipping",
                          gen_at_schedule.value_or(0), gen_now);
            // Still restart timer with current generation
            m_get_round_timer->start(chrono::Seconds(get_round_interval),
                get_round_handler(get_round_interval, worker_manager));
            return;
        }

        try {
            wm->poll_get_round();
        } catch (const std::exception& e) {
            spdlog::error("[Timer_manager] get_round_handler threw: {} — timer will restart", e.what());
        } catch (...) {
            spdlog::error("[Timer_manager] get_round_handler threw unknown exception — timer will restart");
        }

        // Always restart timer after callback completes (even if callback threw).
        // This prevents silent timer death that kills GET_ROUND polling permanently.
        m_get_round_timer->start(chrono::Seconds(get_round_interval),
            get_round_handler(get_round_interval, worker_manager));
    }; 
}

void Timer_manager::start_template_health_timer(std::uint16_t timer_interval, std::weak_ptr<Worker_manager> worker_manager)
{
    m_template_health_timer->start(chrono::Seconds(timer_interval), 
        template_health_handler(timer_interval, std::move(worker_manager)));
}

chrono::Timer::Handler Timer_manager::template_health_handler(std::uint16_t health_check_interval, 
    std::weak_ptr<Worker_manager> worker_manager)
{
    // Capture session generation at timer-schedule time as a cancellation token.
    // When the handler fires, if the generation changed the session died and
    // reconnected — discard the stale callback to avoid health-checking a dead
    // session.  Uses std::optional to distinguish "not captured" (lock failed)
    // from a legitimate generation of 0.
    std::optional<uint64_t> gen_at_schedule;
    if (auto wm = worker_manager.lock()) {
        gen_at_schedule = wm->session_generation();
    }

    return [this, health_check_interval, worker_manager, gen_at_schedule](bool canceled)
    {
        if (canceled)  // don't do anything if the timer has been canceled
        {
            return;
        }

        auto worker_manager_shared = worker_manager.lock();
        if (!worker_manager_shared)
        {
            return;  // Worker_manager destroyed (shutdown), don't restart
        }

        // Session generation guard: if the session changed since this timer tick
        // was scheduled (or was never captured), skip the work.
        const auto gen_now = worker_manager_shared->session_generation();
        if (!gen_at_schedule.has_value() || gen_now != *gen_at_schedule) {
            spdlog::debug("[Timer_manager] template_health_handler: stale session (gen {} → {}), skipping",
                          gen_at_schedule.value_or(0), gen_now);
            // Still restart timer with current generation
            m_template_health_timer->start(chrono::Seconds(health_check_interval), 
                template_health_handler(health_check_interval, worker_manager));
            return;
        }

        try {
            worker_manager_shared->check_template_health();
        } catch (const std::exception& e) {
            spdlog::error("[Timer_manager] template_health_handler threw: {} — timer will restart", e.what());
        } catch (...) {
            spdlog::error("[Timer_manager] template_health_handler threw unknown exception — timer will restart");
        }

        // Always restart timer after callback completes (even if callback threw).
        m_template_health_timer->start(chrono::Seconds(health_check_interval), 
            template_health_handler(health_check_interval, worker_manager));
    }; 
}

void Timer_manager::start_lane_health_check_timer(std::uint16_t timer_interval,
    std::weak_ptr<Worker_manager> worker_manager)
{
    m_lane_health_check_timer->start(chrono::Seconds(timer_interval),
        lane_health_check_handler(timer_interval, std::move(worker_manager)));
}

chrono::Timer::Handler Timer_manager::lane_health_check_handler(std::uint16_t health_check_interval,
    std::weak_ptr<Worker_manager> worker_manager)
{
    // Capture session generation at timer-schedule time as a cancellation token.
    // When the handler fires, if the generation changed the session died and
    // reconnected — discard the stale callback to avoid lane health-checking a
    // dead session.  Uses std::optional to distinguish "not captured" (lock
    // failed) from a legitimate generation of 0.
    std::optional<uint64_t> gen_at_schedule;
    if (auto wm = worker_manager.lock()) {
        gen_at_schedule = wm->session_generation();
    }

    return [this, health_check_interval, worker_manager, gen_at_schedule](bool canceled)
    {
        if (canceled)
        {
            return;
        }

        auto wm = worker_manager.lock();
        if (!wm)
        {
            return;  // Worker_manager destroyed (shutdown), don't restart
        }

        // Session generation guard: if the session changed since this timer tick
        // was scheduled (or was never captured), skip the work.
        const auto gen_now = wm->session_generation();
        if (!gen_at_schedule.has_value() || gen_now != *gen_at_schedule) {
            spdlog::debug("[Timer_manager] lane_health_check_handler: stale session (gen {} → {}), skipping",
                          gen_at_schedule.value_or(0), gen_now);
            // Still restart timer with current generation
            m_lane_health_check_timer->start(chrono::Seconds(health_check_interval),
                lane_health_check_handler(health_check_interval, worker_manager));
            return;
        }

        try {
            wm->log_lane_health();
        } catch (const std::exception& e) {
            spdlog::error("[Timer_manager] lane_health_check_handler threw: {} — timer will restart", e.what());
        } catch (...) {
            spdlog::error("[Timer_manager] lane_health_check_handler threw unknown exception — timer will restart");
        }

        // Always restart timer after callback completes (even if callback threw).
        m_lane_health_check_timer->start(chrono::Seconds(health_check_interval),
            lane_health_check_handler(health_check_interval, worker_manager));
    };
}

void Timer_manager::start_replacement_pending_timer(std::chrono::milliseconds delay,
    std::weak_ptr<Worker_manager> worker_manager)
{
    // Timer::start() cancels any prior outstanding wait before rescheduling, so
    // repeated calls (e.g. one per PUSH that refreshes the pending deadline)
    // simply re-arm to the new deadline rather than accumulating timers.
    m_replacement_pending_timer->start(delay, replacement_pending_handler(std::move(worker_manager)));
}

chrono::Timer::Handler Timer_manager::replacement_pending_handler(std::weak_ptr<Worker_manager> worker_manager)
{
    // One-shot: do NOT reschedule itself on completion. Each arming call comes
    // from Worker_manager reacting to a fresh replacement-pending mark with its
    // own deadline, so there is no fixed cadence to restart here.
    return [worker_manager](bool canceled)
    {
        if (canceled)
        {
            // Superseded by a newer deadline (or shutdown) — nothing to do.
            return;
        }

        auto wm = worker_manager.lock();
        if (!wm)
        {
            return;  // Worker_manager destroyed (shutdown)
        }

        try {
            wm->check_template_health();
        } catch (const std::exception& e) {
            spdlog::error("[Timer_manager] replacement_pending_handler threw: {}", e.what());
        } catch (...) {
            spdlog::error("[Timer_manager] replacement_pending_handler threw unknown exception");
        }
    };
}

}
