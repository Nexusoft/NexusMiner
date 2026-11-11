#include "protocol/epoch_coordinator.hpp"
#include <algorithm>
#include <sstream>

namespace nexusminer {
namespace protocol {

EpochCoordinator::EpochCoordinator()
    : m_session_epoch{0}
    , m_recovery_epoch{0}
{
    m_logger = spdlog::get("logger");
    if (!m_logger) {
        m_logger = spdlog::default_logger();
    }
}

SessionEpoch EpochCoordinator::session_epoch() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return SessionEpoch{m_session_epoch};
}

SessionEpoch EpochCoordinator::advance_session_epoch(const char* reason)
{
    std::vector<EpochObserver> observers_copy;
    uint64_t old_val{0};
    uint64_t new_val{0};
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        old_val = m_session_epoch;
        new_val = ++m_session_epoch;
        observers_copy = m_observers;
    }
    if (m_logger) {
        m_logger->info("[EpochCoordinator] session_epoch advanced: {} → {} (reason: {})",
                       old_val, new_val, reason ? reason : "unknown");
    }
    for (auto& obs : observers_copy) {
        obs("session", old_val, new_val);
    }
    return SessionEpoch{new_val};
}

uint64_t EpochCoordinator::recovery_epoch() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_recovery_epoch;
}

uint64_t EpochCoordinator::advance_recovery_epoch(const char* reason)
{
    std::vector<EpochObserver> observers_copy;
    uint64_t old_val{0};
    uint64_t new_val{0};
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        old_val = m_recovery_epoch;
        new_val = ++m_recovery_epoch;
        observers_copy = m_observers;
    }
    if (m_logger) {
        m_logger->info("[EpochCoordinator] recovery_epoch advanced: {} → {} (reason: {})",
                       old_val, new_val, reason ? reason : "unknown");
    }
    for (auto& obs : observers_copy) {
        obs("recovery", old_val, new_val);
    }
    return new_val;
}

uint64_t EpochCoordinator::global_epoch() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return std::max(m_session_epoch, m_recovery_epoch);
}

void EpochCoordinator::add_observer(EpochObserver observer)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    m_observers.push_back(std::move(observer));
}

EpochCoordinator::Snapshot EpochCoordinator::snapshot() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return {SessionEpoch{m_session_epoch}, m_recovery_epoch, std::max(m_session_epoch, m_recovery_epoch)};
}

std::string EpochCoordinator::diagnostics() const
{
    auto snap = snapshot();
    std::ostringstream oss;
    oss << "session=" << snap.session_epoch
        << " recovery=" << snap.recovery_epoch
        << " global=" << snap.global_epoch;
    return oss.str();
}

} // namespace protocol
} // namespace nexusminer
