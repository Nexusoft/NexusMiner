#include "liveness_watchdog.hpp"

#include <algorithm>
#include <cstdlib>

namespace nexusminer
{

LivenessWatchdog::LivenessWatchdog(chrono::Timer_factory::Sptr timer_factory,
                                    std::shared_ptr<spdlog::logger> logger,
                                    std::chrono::seconds heartbeat_interval,
                                    std::chrono::seconds stall_timeout)
    : m_timer_factory{std::move(timer_factory)}
    , m_logger{std::move(logger)}
    , m_heartbeat_interval{heartbeat_interval}
    , m_stall_timeout{stall_timeout}
    , m_heartbeat_timer{m_timer_factory->create_timer()}
{
}

LivenessWatchdog::~LivenessWatchdog()
{
    stop();
}

void LivenessWatchdog::start()
{
    if (m_started.exchange(true)) {
        return;  // already started
    }
    m_shutdown.store(false, std::memory_order_release);
    m_heartbeat_counter.store(0, std::memory_order_release);
    schedule_heartbeat();
    m_watchdog_thread = std::thread(&LivenessWatchdog::watchdog_loop, this);
    if (m_logger) {
        m_logger->info("[LivenessWatchdog] started (heartbeat={}s, stall_timeout={}s) — "
                       "independent io_context liveness monitor armed",
                       m_heartbeat_interval.count(), m_stall_timeout.count());
    }
}

void LivenessWatchdog::stop()
{
    if (!m_started.exchange(false)) {
        return;  // already stopped / never started
    }
    m_shutdown.store(true, std::memory_order_release);
    m_heartbeat_timer->cancel();
    m_cv.notify_all();
    if (m_watchdog_thread.joinable()) {
        m_watchdog_thread.join();
    }
    if (m_logger) {
        m_logger->info("[LivenessWatchdog] stopped");
    }
}

void LivenessWatchdog::schedule_heartbeat()
{
    // m_heartbeat_interval is already a std::chrono::seconds (== chrono::Seconds),
    // so pass it through directly rather than round-tripping via int (which
    // would silently truncate for values beyond INT_MAX).
    m_heartbeat_timer->start(m_heartbeat_interval,
        [this](bool canceled) {
            if (canceled) {
                // stop() canceled this timer (graceful shutdown/restart) —
                // do not reschedule.
                return;
            }
            m_heartbeat_counter.fetch_add(1, std::memory_order_release);
            if (!m_shutdown.load(std::memory_order_acquire)) {
                schedule_heartbeat();
            }
        });
}

void LivenessWatchdog::watchdog_loop()
{
    std::uint64_t last_seen = m_heartbeat_counter.load(std::memory_order_acquire);
    auto last_change = std::chrono::steady_clock::now();

    // Poll at heartbeat-interval granularity (at least once a second) so a
    // stall is detected promptly without busy-spinning.
    auto poll_interval = std::max(std::chrono::seconds{1}, m_heartbeat_interval);

    std::unique_lock<std::mutex> lock(m_cv_mtx);
    while (!m_shutdown.load(std::memory_order_acquire))
    {
        bool shutdown_requested = m_cv.wait_for(lock, poll_interval,
            [this] { return m_shutdown.load(std::memory_order_acquire); });
        if (shutdown_requested) {
            break;
        }

        auto now = std::chrono::steady_clock::now();
        auto current = m_heartbeat_counter.load(std::memory_order_acquire);
        if (current != last_seen) {
            last_seen = current;
            last_change = now;
            continue;
        }

        auto stalled_for = std::chrono::duration_cast<std::chrono::seconds>(now - last_change);
        if (stalled_for >= m_stall_timeout)
        {
            if (m_logger) {
                m_logger->critical(
                    "[LivenessWatchdog] io_context heartbeat stalled for {}s (>= {}s threshold) — "
                    "the network/timer thread appears wedged (no timer, GET_BLOCK, GET_ROUND, or "
                    "stats activity can run while it is stuck). Forcing a clean process exit so the "
                    "service supervisor performs a fresh Full-Stop + Re-Auth restart, rather than "
                    "sitting silently dead.", stalled_for.count(), m_stall_timeout.count());
                m_logger->flush();
            }
            spdlog::shutdown();  // flush and drop all sinks before hard exit

            // Deliberately skip destructors — any of them could themselves be
            // blocked on the very deadlock being escaped — and exit with a
            // distinct non-zero code so operators/supervisors can identify
            // this specific failure mode from the process exit status.
            std::_Exit(42);
        }
    }
}

} // namespace nexusminer
