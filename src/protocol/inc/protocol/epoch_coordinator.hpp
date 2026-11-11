#ifndef NEXUSMINER_PROTOCOL_EPOCH_COORDINATOR_HPP
#define NEXUSMINER_PROTOCOL_EPOCH_COORDINATOR_HPP

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "protocol/session_semantic_types.hpp"
#include "spdlog/spdlog.h"

namespace nexusminer {
namespace protocol {

/**
 * @brief EpochCoordinator — single source of truth for all epoch counters.
 *
 * Prevents desync between the session_epoch (SessionManager) and the
 * recovery_epoch (Worker_manager::RecoveryContext) by centralising all
 * advance operations and distributing them via an observer callback.
 *
 * Design invariants:
 *  - Both session_epoch and recovery_epoch are MONOTONICALLY INCREASING.
 *    advance_*() never decreases the stored value.
 *  - All reads/writes are protected by m_mutex (thread-safe).
 *  - When any epoch advances, all registered observers are notified
 *    (outside m_mutex to avoid deadlocks).
 */
class EpochCoordinator : public std::enable_shared_from_this<EpochCoordinator> {
public:
    EpochCoordinator();

    // ── Epoch domains ─────────────────────────────────────────────────────────

    /// Session epoch: incremented on every successful authentication.
    /// Replaces SessionManager::SessionInfo::session_epoch, Solo::m_session_epoch,
    /// HeightTracker::m_session_epoch, MiningTemplateInterface::m_session_epoch.
    SessionEpoch session_epoch() const;
    SessionEpoch advance_session_epoch(const char* reason);

    /// Recovery epoch: incremented on every recovery phase transition (non-HEALTHY).
    /// Replaces RecoveryContext::epoch in worker_manager.hpp.
    uint64_t recovery_epoch() const;
    uint64_t advance_recovery_epoch(const char* reason);

    /// Global epoch: max(session_epoch, recovery_epoch) — a monotonic envelope
    /// any component can use for staleness checks without knowing which domain advanced.
    uint64_t global_epoch() const;

    // ── Observer pattern ──────────────────────────────────────────────────────

    /// Callback signature: (domain, old_value, new_value)
    using EpochObserver = std::function<void(const char* domain, uint64_t old_val, uint64_t new_val)>;

    /// Register an observer that is called whenever any epoch advances.
    /// Observers are called WITHOUT the internal mutex held.
    void add_observer(EpochObserver observer);

    // ── Snapshot ─────────────────────────────────────────────────────────────

    struct Snapshot {
        SessionEpoch session_epoch{};
        uint64_t recovery_epoch{0};
        uint64_t global_epoch{0};
    };

    /// Atomically snapshot all three epoch values.
    Snapshot snapshot() const;

    // ── Diagnostics ───────────────────────────────────────────────────────────

    /// Human-readable dump: "session=N recovery=M global=K"
    std::string diagnostics() const;

private:
    mutable std::mutex m_mutex;
    uint64_t m_session_epoch{0};
    uint64_t m_recovery_epoch{0};
    std::vector<EpochObserver> m_observers;
    std::shared_ptr<spdlog::logger> m_logger;
};

} // namespace protocol
} // namespace nexusminer

#endif // NEXUSMINER_PROTOCOL_EPOCH_COORDINATOR_HPP
