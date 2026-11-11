#ifndef NEXUSMINER_LIVENESS_WATCHDOG_HPP
#define NEXUSMINER_LIVENESS_WATCHDOG_HPP

#include "chrono/timer_factory.hpp"

#include <spdlog/spdlog.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>

namespace nexusminer
{

// ─────────────────────────────────────────────────────────────────────────────
// LivenessWatchdog — independent, dumb-on-purpose liveness monitor for the
// single shared asio::io_context thread.
//
// Rationale: every timer-driven subsystem (stats printer/collector, GET_ROUND
// polling, template health monitor, lane health check, and all network I/O)
// runs on the one thread pumped by Miner::run()'s io_context->run(). If that
// thread ever wedges (e.g. a lock-ordering issue or blocking wait inside the
// synchronous handler chain invoked from Connection_impl::receive()), every
// one of those subsystems goes silent simultaneously while CPU-mining worker
// threads (which do not depend on the io_context) keep running obliviously.
// No code running ON the wedged thread can rescue itself from that state.
//
// This class is deliberately simple and independent of RecoveryPhase /
// EpochCoordinator so a bug in that machinery can never defeat it:
//   1. A cheap heartbeat timer is posted onto the io_context itself. Each
//      firing increments an atomic counter and reschedules itself. If the
//      io_context thread is alive and pumping, the counter keeps advancing;
//      if the thread is wedged, it stops dead.
//   2. A separate std::thread — untouched by anything the io_context does —
//      polls that counter. If it has not advanced for longer than
//      stall_timeout, the io_context thread is presumed permanently wedged.
//
// On a detected stall this class does NOT attempt to gracefully unwedge the
// stuck thread in-process (that is generally unsafe/impossible in C++ once a
// deadlock has occurred). Instead it logs critical, flushes all logger sinks,
// and hard-exits the process with a distinct non-zero code so an external
// process supervisor performs a clean restart — a real Full-Stop + Re-Auth,
// since a fresh process re-authenticates from scratch. If the network is
// genuinely down, the supervisor's own restart/backoff policy means mining
// correctly stays at zero energy rather than looping.
// ─────────────────────────────────────────────────────────────────────────────
class LivenessWatchdog
{
public:
    LivenessWatchdog(chrono::Timer_factory::Sptr timer_factory,
                      std::shared_ptr<spdlog::logger> logger,
                      std::chrono::seconds heartbeat_interval = std::chrono::seconds{5},
                      std::chrono::seconds stall_timeout = std::chrono::seconds{60});
    ~LivenessWatchdog();

    LivenessWatchdog(const LivenessWatchdog&) = delete;
    LivenessWatchdog& operator=(const LivenessWatchdog&) = delete;

    /// Start the heartbeat timer (on the io_context) and the external
    /// watchdog thread. Idempotent — a second call while already started is
    /// a no-op.
    void start();

    /// Stop everything cleanly. Idempotent, and safe to call from the
    /// io_context thread itself (e.g. from a signal handler) — the watchdog
    /// thread's exit does not depend on the io_context making progress.
    /// Must be called before process shutdown so a deliberate, graceful stop
    /// (e.g. SIGINT/SIGTERM) is never mistaken for a stall.
    void stop();

private:
    void schedule_heartbeat();
    void watchdog_loop();

    chrono::Timer_factory::Sptr m_timer_factory;
    std::shared_ptr<spdlog::logger> m_logger;
    std::chrono::seconds m_heartbeat_interval;
    std::chrono::seconds m_stall_timeout;

    chrono::Timer::Uptr m_heartbeat_timer;
    std::atomic<std::uint64_t> m_heartbeat_counter{0};
    std::atomic<bool> m_shutdown{true};
    std::atomic<bool> m_started{false};

    std::mutex m_cv_mtx;
    std::condition_variable m_cv;
    std::thread m_watchdog_thread;
};

} // namespace nexusminer

#endif // NEXUSMINER_LIVENESS_WATCHDOG_HPP
