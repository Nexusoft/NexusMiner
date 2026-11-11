#ifndef NEXUSMINER_PROTOCOL_SESSION_RECOVERY_POLICY_HPP
#define NEXUSMINER_PROTOCOL_SESSION_RECOVERY_POLICY_HPP

#include <string>
#include <cstdint>
#include "protocol/session_semantic_types.hpp"

namespace nexusminer {
namespace protocol {

/**
 * @brief Decision record returned by SessionRecoveryPolicy::evaluate_session_expired().
 *
 * Callers must check is_stale_replay first; when true allow_recovery will be false
 * and the SESSION_EXPIRED signal should be silently discarded.
 */
struct SessionExpiredDecision
{
    bool allow_recovery{false};  ///< True when expired signal is authentic and recovery should proceed
    bool is_stale_replay{false}; ///< True when expired session_id does not match authoritative session
    std::string reason;
};

/**
 * @brief Decision record returned by SessionRecoveryPolicy::evaluate_ingress_readiness().
 *
 * Callers should apply side-effects in this order:
 *   1. If !allow_ingress: queue deferred push (if queue_deferred_push), check auth timeout, trigger recovery (if trigger_recovery)
 *   2. If allow_ingress && resync_local_cache: resync local auth cache from authoritative state
 *   3. If allow_ingress && !resync_local_cache: proceed normally
 */
struct IngressReadinessDecision
{
    bool allow_ingress{false};       ///< True when packet may be accepted
    bool resync_local_cache{false};  ///< True when local auth cache should be refreshed from authoritative state
    bool trigger_recovery{false};    ///< True when a reauth/recovery should be initiated
    bool queue_deferred_push{false}; ///< True when a GET_BLOCK should be queued for after authentication
    std::string reason;
};

/**
 * @brief Pure static session recovery policy.
 *
 * SessionRecoveryPolicy provides authoritative-state-derived decisions for the two
 * primary session recovery code-paths in the miner protocol:
 *
 *   1. SESSION_EXPIRED packet handling — determines whether a received expiry signal
 *      is authentic (session_id matches authoritative) or a stale replay.
 *
 *   2. Packet ingress readiness — determines whether a deferred packet should be
 *      queued, whether the local auth cache needs resyncing from the authoritative
 *      SessionManager state, and whether recovery (reauth) should be triggered.
 *
 * Design principle: callers provide a snapshot of authoritative session state and
 * relevant local-cache flags; this class makes the decision without consulting any
 * mutable state.  Executing the decision (side-effects such as invoking callbacks)
 * remains the responsibility of the caller.
 *
 * @note Keeping these decisions in a pure static class makes them unit-testable in
 * isolation without instantiating Solo or SessionManager.
 */
class SessionRecoveryPolicy
{
public:
    // ─────────────────────────────────────────────────────────────────────────
    // SESSION_EXPIRED evaluation
    // ─────────────────────────────────────────────────────────────────────────

    struct SessionExpiredInput
    {
        bool has_authoritative_session{false};  ///< False when no SessionManager is wired in
        SessionId expired_session_id{};         ///< session_id carried in the SESSION_EXPIRED packet
        SessionId authoritative_session_id{};   ///< session_id from the authoritative SessionManager
        uint8_t reason_code{0};                 ///< reason byte from the SESSION_EXPIRED packet
    };

    /**
     * @brief Evaluate a received SESSION_EXPIRED signal against the authoritative session.
     *
     * Returns allow_recovery=true only when the expired_session_id matches the authoritative
     * session_id.  Any mismatch is classified as a stale replay that should be ignored.
     *
     * @param input Authoritative session state and packet fields
     * @return SessionExpiredDecision describing the correct recovery action
     */
    static SessionExpiredDecision evaluate_session_expired(const SessionExpiredInput& input)
    {
        SessionExpiredDecision decision;

        if (!input.has_authoritative_session) {
            decision.is_stale_replay = true;
            decision.reason = "no authoritative session — SESSION_EXPIRED signal is not actionable";
            return decision;
        }

        if (input.expired_session_id != input.authoritative_session_id) {
            decision.is_stale_replay = true;
            decision.reason = "expired session_id does not match authoritative session_id — stale replay";
            return decision;
        }

        decision.allow_recovery = true;
        decision.reason = "authoritative session_id confirmed expired — recovery is authorised";
        return decision;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Packet ingress readiness evaluation
    // ─────────────────────────────────────────────────────────────────────────

    struct IngressReadinessInput
    {
        bool has_session_context{false};          ///< False when no session context is wired in (legacy mode)
        bool authoritative_authenticated{false};  ///< Authoritative SessionManager reports authenticated
        bool local_auth_stale{false};             ///< Local auth cache disagrees with authoritative (local=false, auth=true)
        bool auth_not_in_flight{false};           ///< True when no auth handshake is currently pending (NOT_AUTHENTICATED state)
        bool is_push_context{false};              ///< True when called from a push notification handler — always allow ingress, never trigger recovery
    };

    /**
     * @brief Evaluate whether a deferred packet may be accepted for processing.
     *
     * Consults authoritative session state (not local cache) as the primary authority:
     *
     *   - Push context (is_push_context=true): always allow_ingress=true; never trigger
     *     recovery.  Push notifications are the authoritative liveness signal — a transient
     *     unauthenticated state must never suppress GET_BLOCK.  Cache resync is still applied
     *     when the local cache is stale so the next ingress path sees up-to-date state.
     *   - No session context (legacy mode): allow_ingress=true unconditionally.
     *   - Authoritative says not authenticated: defer the packet; trigger recovery
     *     only when no auth is already in-flight (auth_not_in_flight=true).
     *   - Authoritative authenticated but local cache is stale: allow_ingress=true
     *     with resync_local_cache=true so the caller refreshes its local cache.
     *   - Both authoritative and local agree on authenticated: allow_ingress=true.
     *
     * @param input Authoritative session state and local-cache indicators
     * @return IngressReadinessDecision describing the correct accept/defer/resync action
     */
    static IngressReadinessDecision evaluate_ingress_readiness(const IngressReadinessInput& input)
    {
        IngressReadinessDecision decision;

        if (input.is_push_context) {
            // Push notifications are the authoritative liveness signal.  They must ALWAYS
            // be allowed through regardless of transient session state — the session gate
            // applies only to SUBMIT paths.  Recovery must never be triggered by a push.
            decision.allow_ingress = true;
            decision.resync_local_cache = input.local_auth_stale;
            decision.reason = "push notification context — ingress always allowed, recovery suppressed";
            return decision;
        }

        if (!input.has_session_context) {
            // Legacy mode: session management is disabled; pass all packets through.
            decision.allow_ingress = true;
            decision.reason = "no authoritative session context — operating in legacy mode";
            return decision;
        }

        if (!input.authoritative_authenticated) {
            // Authoritative says not authenticated — defer this packet.
            decision.queue_deferred_push = true;
            if (input.auth_not_in_flight) {
                // No auth handshake is pending; initiate one.
                decision.trigger_recovery = true;
                decision.reason = "authoritative session not authenticated and no auth in-flight — triggering recovery";
            } else {
                decision.reason = "authoritative session not authenticated — auth in-flight, deferring ingress";
            }
            return decision;
        }

        if (input.local_auth_stale) {
            // Authoritative is authenticated but the local auth cache has not been
            // updated yet.  Signal the caller to resync the cache before proceeding.
            decision.allow_ingress = true;
            decision.resync_local_cache = true;
            decision.reason = "local auth cache is stale relative to authoritative session — resyncing";
            return decision;
        }

        // Both authoritative and local cache agree: authenticated.
        decision.allow_ingress = true;
        decision.reason = "authoritative session is authenticated — ingress allowed";
        return decision;
    }
};

} // namespace protocol
} // namespace nexusminer

#endif // NEXUSMINER_PROTOCOL_SESSION_RECOVERY_POLICY_HPP
