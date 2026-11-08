#ifndef NEXUSMINER_WORKER_RECOVERY_TIMEOUT_HPP
#define NEXUSMINER_WORKER_RECOVERY_TIMEOUT_HPP

#include <algorithm>
#include <chrono>
#include <cstdint>

namespace nexusminer::worker
{

class RecoveryTimeout
{
public:
    using Clock = std::chrono::steady_clock;

    void start_if_unset(Clock::time_point now)
    {
        if (m_started_at == Clock::time_point{}) {
            m_started_at = now;
        }
    }

    void observe_node_retry(bool deferred, bool session_usable, Clock::time_point now)
    {
        if (deferred) {
            m_node_retry_wait_active = true;
            return;
        }
        if (!m_node_retry_wait_active) {
            return;
        }

        m_node_retry_wait_active = false;
        if (session_usable) {
            m_started_at = now;
            m_node_retry_timeout_rebased = true;
        }
    }

    void reset()
    {
        m_started_at = {};
        m_node_retry_wait_active = false;
        m_node_retry_timeout_rebased = false;
    }

    int64_t elapsed_seconds(Clock::time_point now) const
    {
        if (m_started_at == Clock::time_point{}) {
            return 0;
        }
        return std::max<int64_t>(
            std::chrono::duration_cast<std::chrono::seconds>(now - m_started_at).count(), 0);
    }

    int64_t worst_case_outage_seconds(Clock::time_point now,
                                      int64_t since_push_seconds,
                                      int64_t lane_alive_seconds) const
    {
        const auto elapsed = elapsed_seconds(now);
        if (m_node_retry_timeout_rebased) {
            since_push_seconds = std::min(since_push_seconds, elapsed);
            lane_alive_seconds = std::min(lane_alive_seconds, elapsed);
        }
        return std::max({elapsed, since_push_seconds, lane_alive_seconds});
    }

private:
    Clock::time_point m_started_at{};
    bool m_node_retry_wait_active{false};
    bool m_node_retry_timeout_rebased{false};
};

} // namespace nexusminer::worker

#endif // NEXUSMINER_WORKER_RECOVERY_TIMEOUT_HPP
