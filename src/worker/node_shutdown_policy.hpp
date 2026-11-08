#ifndef NEXUSMINER_WORKER_NODE_SHUTDOWN_POLICY_HPP
#define NEXUSMINER_WORKER_NODE_SHUTDOWN_POLICY_HPP

namespace nexusminer {

enum class NodeShutdownAction {
    RETRY_PRIMARY_NODE,
    SWITCH_TO_STANDBY_NODE
};

struct NodeShutdownWorkInvalidation {
    bool stop_workers;
    bool discard_template;
    bool reset_session;
    bool quarantine_current_generation;
};

inline constexpr NodeShutdownAction decide_node_shutdown_action(bool has_failover_configured) noexcept
{
    return has_failover_configured
        ? NodeShutdownAction::SWITCH_TO_STANDBY_NODE
        : NodeShutdownAction::RETRY_PRIMARY_NODE;
}

inline constexpr NodeShutdownWorkInvalidation node_shutdown_work_invalidation() noexcept
{
    return {
        true,  // stop_workers
        true,  // discard_template
        true,  // reset_session
        true   // quarantine_current_generation
    };
}

} // namespace nexusminer

#endif // NEXUSMINER_WORKER_NODE_SHUTDOWN_POLICY_HPP
