// Packet ingress preflight + session recovery policy unit tests
//
// Covers:
//   PacketIngressPreflight::evaluate()  — all accept/reject/drop-stale/reauth paths
//   SessionRecoveryPolicy::evaluate_session_expired()  — authentic vs stale replay
//   SessionRecoveryPolicy::evaluate_ingress_readiness() — authoritative-first gating,
//       stale local cache resync, recovery trigger, legacy mode
//
// The tests are pure policy exercises; no Solo, SessionManager, or network
// objects are instantiated.

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "protocol_lane.hpp"
#include "protocol/packet_ingress_preflight.hpp"
#include "protocol/session_recovery_policy.hpp"

using namespace nexusminer::protocol;
using nexusminer::ProtocolLane;

// ─────────────────────────────────────────────────────────────────────────────
// Test scaffolding
// ─────────────────────────────────────────────────────────────────────────────
namespace {

int tests_run    = 0;
int tests_passed = 0;
int tests_failed = 0;

void print_test_result(const char* name, bool passed)
{
    ++tests_run;
    if (passed) {
        ++tests_passed;
        std::cout << "  [PASS] " << name << '\n';
    } else {
        ++tests_failed;
        std::cout << "  [FAIL] " << name << '\n';
    }
}

// Helper: build a minimal authenticated Input
struct AuthSession {
    bool authenticated{true};
    uint32_t session_id{0xABCD1234};
    uint64_t session_epoch{3};
    ProtocolLane active_lane{ProtocolLane::STATELESS};
};

AuthSession make_authenticated_session(uint32_t session_id  = 0xABCD1234,
                                       uint64_t session_epoch = 3,
                                       bool /* reward_bound */ = true)
{
    return {true, session_id, session_epoch, ProtocolLane::STATELESS};
}

SessionOwnershipStamp make_stamp(uint32_t sid, uint64_t epoch)
{
    SessionOwnershipStamp stamp;
    stamp.session_id    = SessionId(sid);
    stamp.session_epoch = SessionEpoch(epoch);
    return stamp;
}

// Build a flat Input from AuthSession
PacketIngressPreflight::Input make_input(const AuthSession& s,
                                         ProtocolLane packet_lane  = ProtocolLane::STATELESS,
                                         bool validate_lane        = false,
                                         bool allow_without        = false,
                                         uint32_t packet_session_id = 0,
                                         uint64_t owner_epoch      = 0,
                                         uint32_t owner_session_id = 0)
{
    PacketIngressPreflight::Input in;
    in.has_authoritative_session      = true;
    in.authoritative_authenticated    = s.authenticated;
    in.authoritative_session_id       = SessionId(s.session_id);
    in.authoritative_session_epoch    = SessionEpoch(s.session_epoch);
    in.authoritative_lane             = s.active_lane;
    in.packet_lane                    = packet_lane;
    in.validate_lane                  = validate_lane;
    in.allow_without_active_session   = allow_without;
    in.packet_session_id              = SessionId(packet_session_id);
    in.owner_epoch                    = SessionEpoch(owner_epoch);
    in.owner_session_id               = SessionId(owner_session_id);
    return in;
}

} // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// PacketIngressPreflight::evaluate() tests
// ─────────────────────────────────────────────────────────────────────────────

void test_preflight_allows_clean_authenticated_session()
{
    std::cout << "\nTest: PacketIngressPreflight — clean authenticated session is allowed\n";

    const auto session = make_authenticated_session();
    const auto decision = PacketIngressPreflight::evaluate(make_input(session));

    print_test_result("allow_processing is true",  decision.allow_processing);
    print_test_result("force_reauth is false",     !decision.force_reauth);
    print_test_result("drop_as_stale is false",    !decision.drop_as_stale);
    print_test_result("mark_degraded is false",    !decision.mark_degraded);
}

void test_preflight_rejects_no_session_container()
{
    std::cout << "\nTest: PacketIngressPreflight — no authoritative session container\n";

    PacketIngressPreflight::Input in{};
    in.has_authoritative_session = false;
    const auto decision = PacketIngressPreflight::evaluate(in);

    print_test_result("allow_processing is false",  !decision.allow_processing);
    print_test_result("force_reauth is false",       !decision.force_reauth);
}

void test_preflight_rejects_inconsistent_session()
{
    std::cout << "\nTest: PacketIngressPreflight — unauthenticated session rejected\n";

    PacketIngressPreflight::Input in{};
    in.has_authoritative_session   = true;
    in.authoritative_authenticated = false;  // not authenticated
    const auto decision = PacketIngressPreflight::evaluate(in);

    print_test_result("allow_processing is false", !decision.allow_processing);
}

void test_preflight_forces_reauth_when_not_authenticated()
{
    std::cout << "\nTest: PacketIngressPreflight — unauthenticated session triggers reauth\n";

    AuthSession s{false, 0, 0, ProtocolLane::UNKNOWN};
    const auto decision = PacketIngressPreflight::evaluate(make_input(s));

    print_test_result("allow_processing is false", !decision.allow_processing);
    print_test_result("force_reauth is true",       decision.force_reauth);
    print_test_result("mark_degraded is true",      decision.mark_degraded);
}

void test_preflight_allows_unauthenticated_when_explicitly_permitted()
{
    std::cout << "\nTest: PacketIngressPreflight — unauthenticated allowed when allow_without_active_session\n";

    AuthSession s{false, 0, 0, ProtocolLane::UNKNOWN};
    const auto decision = PacketIngressPreflight::evaluate(
        make_input(s, ProtocolLane::UNKNOWN, false, true /*allow_without*/));

    print_test_result("allow_processing is true",  decision.allow_processing);
    print_test_result("force_reauth is false",     !decision.force_reauth);
}

void test_preflight_rejects_lane_mismatch()
{
    std::cout << "\nTest: PacketIngressPreflight — lane mismatch marks degraded\n";

    auto session = make_authenticated_session();
    session.active_lane = ProtocolLane::LEGACY;

    const auto decision = PacketIngressPreflight::evaluate(
        make_input(session, ProtocolLane::STATELESS, true /*validate_lane*/));

    print_test_result("allow_processing is false", !decision.allow_processing);
    print_test_result("mark_degraded is true",      decision.mark_degraded);
}

void test_preflight_allows_unknown_lane_without_marking_degraded()
{
    std::cout << "\nTest: PacketIngressPreflight — UNKNOWN packet lane skips lane check\n";

    auto session = make_authenticated_session();
    session.active_lane = ProtocolLane::LEGACY;

    const auto decision = PacketIngressPreflight::evaluate(
        make_input(session, ProtocolLane::UNKNOWN, true /*validate_lane*/));

    print_test_result("allow_processing is true", decision.allow_processing);
}

void test_preflight_rejects_crypto_not_ready()
{
    // require_crypto_ready is removed from the new design — this is now a no-op test
    std::cout << "\nTest: PacketIngressPreflight — crypto readiness check removed (no-op)\n";
    print_test_result("no-op: feature removed", true);
}

void test_preflight_rejects_reward_not_bound()
{
    // require_reward_binding is removed from the new design — this is now a no-op test
    std::cout << "\nTest: PacketIngressPreflight — reward binding check removed (no-op)\n";
    print_test_result("no-op: feature removed", true);
}

void test_preflight_allows_reward_bound()
{
    // require_reward_binding is removed from the new design — plain authenticated is allowed
    std::cout << "\nTest: PacketIngressPreflight — plain authenticated session is allowed\n";

    const auto session = make_authenticated_session();
    const auto decision = PacketIngressPreflight::evaluate(make_input(session));

    print_test_result("allow_processing is true", decision.allow_processing);
}

void test_preflight_drops_stale_session_id_mismatch()
{
    std::cout << "\nTest: PacketIngressPreflight — packet session_id mismatch drops as stale\n";

    auto session = make_authenticated_session(0xABCD1234);
    const auto decision = PacketIngressPreflight::evaluate(
        make_input(session, ProtocolLane::STATELESS, false, false,
                   0xDEADBEEF /*mismatched packet_session_id*/));

    print_test_result("allow_processing is false",                    !decision.allow_processing);
    print_test_result("drop_as_stale is true",                         decision.drop_as_stale);
    print_test_result("stale_reason is SESSION_ID_MISMATCH",
        decision.stale_reason == PacketStaleReason::SESSION_ID_MISMATCH);
    print_test_result("mark_degraded is true",                         decision.mark_degraded);
}

void test_preflight_drops_stale_ownership_epoch_mismatch()
{
    std::cout << "\nTest: PacketIngressPreflight — ownership epoch mismatch drops as stale\n";

    auto session = make_authenticated_session(0xABCD1234, 5 /*epoch*/);
    const auto stamp = make_stamp(0xABCD1234, 3 /*old epoch*/);

    const auto decision = PacketIngressPreflight::evaluate(
        make_input(session, ProtocolLane::STATELESS, false, false,
                   0 /*no packet_session_id*/,
                   stamp.session_epoch.get(),
                   stamp.session_id.get()));

    print_test_result("allow_processing is false",                        !decision.allow_processing);
    print_test_result("drop_as_stale is true",                             decision.drop_as_stale);
    print_test_result("stale_reason is OWNERSHIP_EPOCH_MISMATCH",
        decision.stale_reason == PacketStaleReason::OWNERSHIP_EPOCH_MISMATCH);
}

void test_preflight_drops_stale_ownership_session_id_mismatch()
{
    std::cout << "\nTest: PacketIngressPreflight — ownership session_id mismatch drops as stale\n";

    auto session = make_authenticated_session(0xABCD1234, 5 /*epoch*/);
    const auto stamp = make_stamp(0xDEAD0000 /*different sid*/, 5 /*matching epoch*/);

    const auto decision = PacketIngressPreflight::evaluate(
        make_input(session, ProtocolLane::STATELESS, false, false,
                   0 /*no packet_session_id*/,
                   stamp.session_epoch.get(),
                   stamp.session_id.get()));

    print_test_result("allow_processing is false",                            !decision.allow_processing);
    print_test_result("drop_as_stale is true",                                 decision.drop_as_stale);
    print_test_result("stale_reason is OWNERSHIP_SESSION_ID_MISMATCH",
        decision.stale_reason == PacketStaleReason::OWNERSHIP_SESSION_ID_MISMATCH);
}

void test_preflight_allows_matching_ownership_stamp()
{
    std::cout << "\nTest: PacketIngressPreflight — matching ownership stamp is allowed\n";

    const uint32_t sid   = 0xABCD1234;
    const uint64_t epoch = 5;
    auto session = make_authenticated_session(sid, epoch);
    const auto stamp = make_stamp(sid, epoch);

    const auto decision = PacketIngressPreflight::evaluate(
        make_input(session, ProtocolLane::STATELESS, false, false,
                   0, stamp.session_epoch.get(), stamp.session_id.get()));

    print_test_result("allow_processing is true",  decision.allow_processing);
    print_test_result("drop_as_stale is false",    !decision.drop_as_stale);
}

void test_preflight_drops_after_epoch_change()
{
    std::cout << "\nTest: PacketIngressPreflight — stale stamp rejected after epoch change\n";

    const uint32_t sid = 0x00001111;
    auto session = make_authenticated_session(sid, 2 /*new epoch*/);
    const auto old_stamp = make_stamp(sid, 1 /*old epoch*/);

    const auto decision = PacketIngressPreflight::evaluate(
        make_input(session, ProtocolLane::STATELESS, false, false,
                   0, old_stamp.session_epoch.get(), old_stamp.session_id.get()));

    print_test_result("allow_processing is false",                      !decision.allow_processing);
    print_test_result("drop_as_stale is true",                           decision.drop_as_stale);
    print_test_result("stale_reason is OWNERSHIP_EPOCH_MISMATCH",
        decision.stale_reason == PacketStaleReason::OWNERSHIP_EPOCH_MISMATCH);
}

// ─────────────────────────────────────────────────────────────────────────────
// SessionRecoveryPolicy::evaluate_session_expired() tests
// ─────────────────────────────────────────────────────────────────────────────

void test_recovery_allows_matching_session_expired()
{
    std::cout << "\nTest: SessionRecoveryPolicy — SESSION_EXPIRED with matching session_id allows recovery\n";

    const auto decision = SessionRecoveryPolicy::evaluate_session_expired({
        true,                       // has_authoritative_session
        SessionId(0xABCD1234u),     // expired_session_id
        SessionId(0xABCD1234u),     // authoritative_session_id (matches)
        0x01                        // reason_code
    });

    print_test_result("allow_recovery is true",     decision.allow_recovery);
    print_test_result("is_stale_replay is false",  !decision.is_stale_replay);
}

void test_recovery_ignores_stale_replay_session_expired()
{
    std::cout << "\nTest: SessionRecoveryPolicy — SESSION_EXPIRED with mismatched session_id is stale replay\n";

    const auto decision = SessionRecoveryPolicy::evaluate_session_expired({
        true,
        SessionId(0xDEADBEEFu),    // expired_session_id (stale/old)
        SessionId(0xABCD1234u),    // authoritative_session_id (current)
        0x01
    });

    print_test_result("allow_recovery is false",    !decision.allow_recovery);
    print_test_result("is_stale_replay is true",     decision.is_stale_replay);
}

void test_recovery_ignores_session_expired_when_no_authoritative_session()
{
    std::cout << "\nTest: SessionRecoveryPolicy — SESSION_EXPIRED without authoritative session is stale replay\n";

    const auto decision = SessionRecoveryPolicy::evaluate_session_expired({
        false,                      // has_authoritative_session = false
        SessionId(0x00000001u),
        SessionId(0x00000000u),
        0x01
    });

    print_test_result("allow_recovery is false",    !decision.allow_recovery);
    print_test_result("is_stale_replay is true",     decision.is_stale_replay);
}

void test_recovery_allows_zero_session_id_when_both_zero()
{
    std::cout << "\nTest: SessionRecoveryPolicy — SESSION_EXPIRED with both IDs zero is accepted as recovery\n";

    const auto decision = SessionRecoveryPolicy::evaluate_session_expired({
        true,
        SessionId(0x00000000u),    // expired matches
        SessionId(0x00000000u),    // authoritative
        0x01
    });

    print_test_result("allow_recovery is true",     decision.allow_recovery);
    print_test_result("is_stale_replay is false",  !decision.is_stale_replay);
}

// ─────────────────────────────────────────────────────────────────────────────
// SessionRecoveryPolicy::evaluate_ingress_readiness() tests
// ─────────────────────────────────────────────────────────────────────────────

void test_ingress_readiness_legacy_mode_always_allowed()
{
    std::cout << "\nTest: SessionRecoveryPolicy ingress — legacy mode always allowed\n";

    const auto decision = SessionRecoveryPolicy::evaluate_ingress_readiness({
        false,  // has_session_context = false (legacy mode)
        false,  // authoritative_authenticated
        false,  // local_auth_stale
        true    // auth_not_in_flight
    });

    print_test_result("allow_ingress is true",          decision.allow_ingress);
    print_test_result("trigger_recovery is false",     !decision.trigger_recovery);
    print_test_result("resync_local_cache is false",   !decision.resync_local_cache);
}

void test_ingress_readiness_authoritative_not_auth_triggers_recovery_when_not_in_flight()
{
    std::cout << "\nTest: SessionRecoveryPolicy ingress — authoritative not authenticated, no auth in-flight → recovery\n";

    const auto decision = SessionRecoveryPolicy::evaluate_ingress_readiness({
        true,   // has_session_context
        false,  // authoritative_authenticated = false
        false,  // local_auth_stale
        true    // auth_not_in_flight = true
    });

    print_test_result("allow_ingress is false",         !decision.allow_ingress);
    print_test_result("trigger_recovery is true",        decision.trigger_recovery);
    print_test_result("queue_deferred_push is true",     decision.queue_deferred_push);
    print_test_result("resync_local_cache is false",    !decision.resync_local_cache);
}

void test_ingress_readiness_authoritative_not_auth_defers_without_recovery_when_auth_in_flight()
{
    std::cout << "\nTest: SessionRecoveryPolicy ingress — authoritative not authenticated, auth in-flight → defer only\n";

    const auto decision = SessionRecoveryPolicy::evaluate_ingress_readiness({
        true,   // has_session_context
        false,  // authoritative_authenticated = false
        false,  // local_auth_stale
        false   // auth_not_in_flight = false
    });

    print_test_result("allow_ingress is false",         !decision.allow_ingress);
    print_test_result("trigger_recovery is false",      !decision.trigger_recovery);
    print_test_result("queue_deferred_push is true",     decision.queue_deferred_push);
}

void test_ingress_readiness_stale_local_cache_resync()
{
    std::cout << "\nTest: SessionRecoveryPolicy ingress — authoritative authenticated but local cache stale → resync\n";

    const auto decision = SessionRecoveryPolicy::evaluate_ingress_readiness({
        true,   // has_session_context
        true,   // authoritative_authenticated = true
        true,   // local_auth_stale = true
        true    // auth_not_in_flight
    });

    print_test_result("allow_ingress is true",           decision.allow_ingress);
    print_test_result("resync_local_cache is true",      decision.resync_local_cache);
    print_test_result("trigger_recovery is false",      !decision.trigger_recovery);
}

void test_ingress_readiness_both_authenticated_passes()
{
    std::cout << "\nTest: SessionRecoveryPolicy ingress — both authoritative and local authenticated → pass\n";

    const auto decision = SessionRecoveryPolicy::evaluate_ingress_readiness({
        true,   // has_session_context
        true,   // authoritative_authenticated = true
        false,  // local_auth_stale = false
        true    // auth_not_in_flight
    });

    print_test_result("allow_ingress is true",          decision.allow_ingress);
    print_test_result("resync_local_cache is false",   !decision.resync_local_cache);
    print_test_result("trigger_recovery is false",     !decision.trigger_recovery);
    print_test_result("queue_deferred_push is false",  !decision.queue_deferred_push);
}

void test_ingress_readiness_authoritative_overrides_stale_local_authenticated()
{
    std::cout << "\nTest: SessionRecoveryPolicy ingress — local says authenticated but authoritative says no → defer\n";

    const auto decision = SessionRecoveryPolicy::evaluate_ingress_readiness({
        true,   // has_session_context
        false,  // authoritative_authenticated = false
        false,  // local_auth_stale = false
        true    // auth_not_in_flight
    });

    print_test_result("allow_ingress is false (authoritative prevails)", !decision.allow_ingress);
    print_test_result("trigger_recovery is true",                         decision.trigger_recovery);
}

void test_ingress_readiness_push_context_always_allows_even_when_not_authenticated()
{
    std::cout << "\nTest: SessionRecoveryPolicy ingress — push context always allows ingress, never triggers recovery\n";

    // Worst case: authoritative says not authenticated, no auth in-flight
    // But is_push_context=true must override and always allow.
    const auto decision = SessionRecoveryPolicy::evaluate_ingress_readiness({
        true,   // has_session_context
        false,  // authoritative_authenticated = false
        false,  // local_auth_stale
        true,   // auth_not_in_flight = true (would normally trigger recovery)
        true    // is_push_context = true
    });

    print_test_result("allow_ingress is true (push context overrides)",  decision.allow_ingress);
    print_test_result("trigger_recovery is false (push never kills)",   !decision.trigger_recovery);
    print_test_result("queue_deferred_push is false",                   !decision.queue_deferred_push);
    print_test_result("resync_local_cache is false",                    !decision.resync_local_cache);
}

void test_ingress_readiness_push_context_resyncs_stale_local_cache()
{
    std::cout << "\nTest: SessionRecoveryPolicy ingress — push context with stale local cache → allow + resync, no recovery\n";

    const auto decision = SessionRecoveryPolicy::evaluate_ingress_readiness({
        true,   // has_session_context
        true,   // authoritative_authenticated = true
        true,   // local_auth_stale = true
        true,   // auth_not_in_flight
        true    // is_push_context = true
    });

    print_test_result("allow_ingress is true",          decision.allow_ingress);
    print_test_result("resync_local_cache is true",      decision.resync_local_cache);
    print_test_result("trigger_recovery is false",      !decision.trigger_recovery);
    print_test_result("queue_deferred_push is false",   !decision.queue_deferred_push);
}

// ─────────────────────────────────────────────────────────────────────────────
// main
// ─────────────────────────────────────────────────────────────────────────────
int main()
{
    std::cout << "=== packet_ingress_preflight_test ===\n";

    test_preflight_allows_clean_authenticated_session();
    test_preflight_rejects_no_session_container();
    test_preflight_rejects_inconsistent_session();
    test_preflight_forces_reauth_when_not_authenticated();
    test_preflight_allows_unauthenticated_when_explicitly_permitted();
    test_preflight_rejects_lane_mismatch();
    test_preflight_allows_unknown_lane_without_marking_degraded();
    test_preflight_rejects_crypto_not_ready();
    test_preflight_rejects_reward_not_bound();
    test_preflight_allows_reward_bound();
    test_preflight_drops_stale_session_id_mismatch();
    test_preflight_drops_stale_ownership_epoch_mismatch();
    test_preflight_drops_stale_ownership_session_id_mismatch();
    test_preflight_allows_matching_ownership_stamp();
    test_preflight_drops_after_epoch_change();

    test_recovery_allows_matching_session_expired();
    test_recovery_ignores_stale_replay_session_expired();
    test_recovery_ignores_session_expired_when_no_authoritative_session();
    test_recovery_allows_zero_session_id_when_both_zero();

    test_ingress_readiness_legacy_mode_always_allowed();
    test_ingress_readiness_authoritative_not_auth_triggers_recovery_when_not_in_flight();
    test_ingress_readiness_authoritative_not_auth_defers_without_recovery_when_auth_in_flight();
    test_ingress_readiness_stale_local_cache_resync();
    test_ingress_readiness_both_authenticated_passes();
    test_ingress_readiness_authoritative_overrides_stale_local_authenticated();
    test_ingress_readiness_push_context_always_allows_even_when_not_authenticated();
    test_ingress_readiness_push_context_resyncs_stale_local_cache();

    std::cout << "\n=== Results: " << tests_passed << "/" << tests_run << " passed";
    if (tests_failed > 0) {
        std::cout << " (" << tests_failed << " FAILED)";
    }
    std::cout << " ===\n";

    return tests_failed > 0 ? 1 : 0;
}
