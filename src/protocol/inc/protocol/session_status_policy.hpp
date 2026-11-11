#ifndef NEXUSMINER_PROTOCOL_SESSION_STATUS_POLICY_HPP
#define NEXUSMINER_PROTOCOL_SESSION_STATUS_POLICY_HPP

#include <cstdint>
#include <string>
#include "protocol/session_semantic_types.hpp"

namespace nexusminer {
namespace protocol {

struct SessionStatusDecision
{
    bool accept_ack{false};
    bool force_reauth{false};
    bool mark_degraded{false};
    bool expire_session{false};
    bool force_reconnect{false};
    bool reset_mismatch_counter{false};
    uint32_t mismatch_count{0};
    std::string reason;
};

class SessionStatusPolicy
{
public:
    struct AckValidationInput
    {
        bool has_session_manager{false};
        SessionId local_session_id{};
        SessionId ack_session_id{};
        uint32_t consecutive_mismatches{0};
        uint32_t mismatch_expire_threshold{1};
    };

    struct AckHealthInput
    {
        uint32_t uptime_seconds{0};
        bool lane_authenticated{false};
    };

    struct DegradedSessionInput
    {
        int64_t degraded_duration_seconds{0};
        int64_t degraded_hard_limit_seconds{0};
        bool push_recent{false};
    };

    static SessionStatusDecision validate_ack(const AckValidationInput& input)
    {
        SessionStatusDecision decision;
        decision.accept_ack = true;
        decision.mismatch_count = input.consecutive_mismatches;

        if (!input.has_session_manager) {
            decision.reason = "no authoritative session manager";
            return decision;
        }

        if (input.ack_session_id.is_default()) {
            decision.reason = "ack omitted session id";
            return decision;
        }

        if (input.ack_session_id == input.local_session_id) {
            decision.reset_mismatch_counter = true;
            decision.mismatch_count = 0;
            decision.reason = "ack matched active session id";
            return decision;
        }

        decision.accept_ack = false;
        decision.mismatch_count = input.consecutive_mismatches + 1;
        // ACK mismatch is diagnostic only — PUSH notification liveness is the
        // sole authoritative signal for session health.  Do NOT expire the
        // session or force re-auth based on keepalive ACK mismatches; the node-
        // side ACK responder can lag or fail independently of the PUSH path
        // that lives in Server.cpp and auto-sends every new block.
        decision.reason = "ack session id mismatched active session (diagnostic only — PUSH is authoritative)";

        return decision;
    }

    static SessionStatusDecision evaluate_ack_health(const AckHealthInput& input)
    {
        SessionStatusDecision decision;
        decision.accept_ack = true;
        // force_reauth and mark_degraded intentionally remain false:
        // SESSION_STATUS is a telemetry probe — a single bad payload must never
        // kill mining workers.  PUSH notification liveness is the authoritative signal.
        if (input.uptime_seconds == 0 || !input.lane_authenticated) {
            decision.reason = "ack reported expired or unauthenticated session (diagnostic only -- PUSH is authoritative)";
        } else {
            decision.reason = "ack reported healthy authenticated session";
        }
        return decision;
    }

    static SessionStatusDecision evaluate_degraded_session(const DegradedSessionInput& input)
    {
        SessionStatusDecision decision;
        decision.accept_ack = true;
        decision.mark_degraded = (input.degraded_duration_seconds > 0);

        if (input.degraded_duration_seconds <= input.degraded_hard_limit_seconds) {
            decision.reason = "degraded session remains inside recovery window";
            return decision;
        }

        if (input.push_recent) {
            // Push is flowing — the TCP session and auth are operationally alive.
            // Do NOT force reauth; the miner should continue mining normally.
            // The degraded timer continues running but no destructive action is taken
            // while the node is actively pushing block notifications.
            decision.reason = "degraded timer running but push traffic is live — holding session";
            return decision;   // accept_ack=true, mark_degraded=true, nothing destructive
        }

        decision.force_reconnect = true;
        decision.reason = "degraded session exceeded hard limit without live push traffic";
        return decision;
    }
};

} // namespace protocol
} // namespace nexusminer

#endif // NEXUSMINER_PROTOCOL_SESSION_STATUS_POLICY_HPP
