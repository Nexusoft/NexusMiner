#include "protocol/node_session_context.hpp"
#include "protocol/hex_prefix_utils.hpp"
#include "protocol/session_semantic_types.hpp"
#include <cassert>
#include <iostream>
#include <string>
#include <type_traits>

namespace {
std::string format_hex_prefix_via_qualified_call(const std::vector<uint8_t>& bytes, std::size_t prefix_bytes)
{
    return nexusminer::protocol::format_hex_prefix(bytes, prefix_bytes);
}
}

using namespace nexusminer::protocol;

void test_session_lifecycle() {
    std::cout << "Testing session lifecycle..." << std::endl;

    auto session_manager = std::make_shared<SessionManager>(24, nullptr);
    NodeSessionContext context(session_manager);

    // Initially no session
    assert(context.get_session_id() == SessionId(0u));
    assert(!context.is_authenticated());
    assert(!context.is_active());
    assert(context.get_state() == SessionManager::SessionState::DISCONNECTED);

    // Start a session
    context.start_session(SessionId(12345u));
    assert(context.get_session_id() == SessionId(12345u));
    assert(context.is_authenticated());
    assert(context.is_active());

    // End session
    context.end_session();
    assert(context.get_session_id() == SessionId(0u));
    assert(!context.is_authenticated());

    std::cout << "Session lifecycle test passed!" << std::endl;
}

void test_session_constants() {
    std::cout << "Testing session constants accessors..." << std::endl;

    assert(NodeSessionContext::get_keepalive_safety_divisor() == 4);
    assert(NodeSessionContext::get_max_session_auth_retries() == 10);
    assert(NodeSessionContext::get_base_session_retry_ms() == 1000);
    // Widened from 60000 to 180000 (3 min) so a reorg-storm-driven burst of
    // session re-auth failures has more headroom to self-resolve before the
    // controlled retry budget is exhausted. See protocol_constants.hpp
    // (MAX_SESSION_RETRY_MS) and Worker_manager::next_session_auth_retry_delay_seconds().
    assert(NodeSessionContext::get_max_session_retry_ms() == 180000);

    std::cout << "Session constants test passed!" << std::endl;
}

void test_parse_session_start() {
    std::cout << "Testing parse_session_start..." << std::endl;

    // Valid SESSION_START packet (minimal: 9 bytes)
    std::vector<uint8_t> packet = {
        0x01,                    // success
        0x39, 0x30, 0x00, 0x00,  // session_id = 12345 (LE)
        0x10, 0x0E, 0x00, 0x00   // timeout = 3600 seconds (LE)
    };

    uint32_t session_id = 0;
    uint32_t timeout = 0;
    std::vector<uint8_t> genesis;

    bool result = NodeSessionContext::parse_session_start(packet, session_id, timeout, genesis);
    assert(result);
    assert(session_id == 12345);
    assert(timeout == 3600);
    assert(genesis.empty());

    // Valid SESSION_START with genesis (41 bytes)
    std::vector<uint8_t> packet_with_genesis = {
        0x01,                    // success
        0x39, 0x30, 0x00, 0x00,  // session_id = 12345 (LE)
        0x10, 0x0E, 0x00, 0x00   // timeout = 3600 seconds (LE)
    };
    // Add 32 bytes of genesis
    for (int i = 0; i < 32; ++i) {
        packet_with_genesis.push_back(static_cast<uint8_t>(i));
    }

    genesis.clear();
    result = NodeSessionContext::parse_session_start(packet_with_genesis, session_id, timeout, genesis);
    assert(result);
    assert(session_id == 12345);
    assert(timeout == 3600);
    assert(genesis.size() == 32);
    for (int i = 0; i < 32; ++i) {
        assert(genesis[i] == static_cast<uint8_t>(i));
    }

    // Invalid packet (too short)
    std::vector<uint8_t> short_packet = {0x01, 0x39, 0x30};
    result = NodeSessionContext::parse_session_start(short_packet, session_id, timeout, genesis);
    assert(!result);

    // Invalid packet (success = 0)
    std::vector<uint8_t> failed_packet = {
        0x00,                    // success = 0 (failed)
        0x39, 0x30, 0x00, 0x00,
        0x10, 0x0E, 0x00, 0x00
    };
    result = NodeSessionContext::parse_session_start(failed_packet, session_id, timeout, genesis);
    assert(!result);

    std::cout << "parse_session_start test passed!" << std::endl;
}

void test_keepalive_interval() {
    std::cout << "Testing keepalive interval..." << std::endl;

    auto session_manager = std::make_shared<SessionManager>(24, nullptr);
    NodeSessionContext context(session_manager);

    assert(context.get_keepalive_interval() == 24);

    context.set_keepalive_interval(12);
    assert(context.get_keepalive_interval() == 12);

    std::cout << "Keepalive interval test passed!" << std::endl;
}

void test_tritium_genesis() {
    std::cout << "Testing Tritium genesis..." << std::endl;

    auto session_manager = std::make_shared<SessionManager>(24, nullptr);
    NodeSessionContext context(session_manager);

    std::vector<uint8_t> genesis(32, 0xAB);
    context.set_tritium_genesis(genesis);

    auto retrieved = context.get_tritium_genesis();
    assert(retrieved.size() == 32);
    for (size_t i = 0; i < 32; ++i) {
        assert(retrieved[i] == 0xAB);
    }

    std::cout << "Tritium genesis test passed!" << std::endl;
}

void test_session_manager_access() {
    std::cout << "Testing session manager access..." << std::endl;

    auto session_manager = std::make_shared<SessionManager>(24, nullptr);
    NodeSessionContext context(session_manager);

    assert(context.get_session_manager() == session_manager);

    std::cout << "Session manager access test passed!" << std::endl;
}

void test_authoritative_miner_session_container_binding() {
    std::cout << "Testing authoritative miner session container binding..." << std::endl;

    auto session_manager = std::make_shared<SessionManager>(24, nullptr);
    NodeSessionContext context(session_manager);

    std::vector<uint8_t> genesis(32, 0x11);
    std::vector<uint8_t> falcon_pubkey(32, 0x22);
    std::vector<uint8_t> chacha_key(32, 0x33);
    std::vector<uint8_t> reward_hash(32, 0x44);

    context.set_protocol_lane(nexusminer::ProtocolLane::STATELESS);
    context.set_connection_metadata("127.0.0.1:4000", "127.0.0.1:9323", true);
    context.set_tritium_genesis(genesis);
    context.set_falcon_identity(falcon_pubkey, "2222222222222222", true);
    context.start_session(SessionId(0x12345678u), {}, genesis);
    const auto chacha_fingerprint = format_hex_prefix(chacha_key, 8);
    context.set_chacha20_session_key(chacha_key, chacha_fingerprint, true);
    context.set_reward_binding("reward-address", reward_hash, true, "config");
    context.set_channel_state(2, true, true);

    std::string reason;
    assert(context.validate_miner_session(&reason));
    assert(reason == "PASS");

    auto info = context.get_session_info();
    assert(info.connected);
    assert(info.authenticated);
    assert(info.falcon_authenticated);
    assert(info.session_id == SessionId(0x12345678u));
    assert(info.session_genesis.get() == genesis);
    assert(info.falcon_pubkey == falcon_pubkey);
    assert(info.falcon_key_id.get() == "2222222222222222");
    assert(info.chacha20_session_key == chacha_key);
    assert(info.chacha20_key_fingerprint.get() == chacha_fingerprint);
    assert(info.reward_address_string == "reward-address");
    assert(info.reward_hash.get() == reward_hash);
    assert(info.reward_bound);
    assert(info.channel == 2);
    assert(info.ready_for_submit);
    assert(info.ready_for_get_block);

    const auto diagnostics = context.build_miner_session_diagnostics();
    assert(diagnostics.find("MINER SESSION CONTAINER") != std::string::npos);
    assert(diagnostics.find("reward-address") != std::string::npos);
    assert(diagnostics.find("consistency: PASS") != std::string::npos);

    std::cout << "Authoritative miner session container binding test passed!" << std::endl;
}

void test_reward_binding_persists_across_session_restart() {
    std::cout << "Testing reward binding persistence across session restart..." << std::endl;

    auto session_manager = std::make_shared<SessionManager>(24, nullptr);
    NodeSessionContext context(session_manager);

    const std::vector<uint8_t> initial_genesis(32, 0x11);
    const std::vector<uint8_t> reconnect_genesis(32, 0x22);
    const std::vector<uint8_t> reward_hash(32, 0x44);

    context.start_session(SessionId(0x12345678u), {}, initial_genesis);
    context.set_reward_binding("reward-address", reward_hash, true, "config");
    context.set_channel_state(2, true, true);

    context.start_session(SessionId(0x87654321u), {}, reconnect_genesis);

    const auto info = context.get_session_info();
    assert(info.session_id == SessionId(0x87654321u));
    assert(info.session_genesis.get() == reconnect_genesis);
    assert(info.reward_address_string == "reward-address");
    assert(info.reward_hash.get() == reward_hash);
    assert(info.reward_bound);
    // Reward intent persists across restart, but the new session must re-establish
    // channel/work readiness before GET_BLOCK or submit may resume.
    assert(!info.ready_for_submit);
    assert(!info.ready_for_get_block);

    std::string reason;
    assert(context.validate_miner_session(&reason));
    assert(reason == "PASS");

    std::cout << "Reward binding persistence across session restart test passed!" << std::endl;
}

void test_format_hex_prefix_matches_session_validation_fingerprint() {
    std::cout << "Testing shared hex-prefix fingerprint formatting..." << std::endl;

    auto session_manager = std::make_shared<SessionManager>(24, nullptr);
    NodeSessionContext context(session_manager);

    std::vector<uint8_t> genesis(32, 0x0A);
    std::vector<uint8_t> chacha_key{0x00, 0x01, 0x0A, 0x10, 0xAB, 0xCD, 0xEF, 0xFF, 0x55};
    const auto fingerprint = format_hex_prefix(chacha_key, 8);

    assert(fingerprint == "00010a10abcdefff");

    context.set_protocol_lane(nexusminer::ProtocolLane::STATELESS);
    context.set_connection_metadata("127.0.0.1:4000", "127.0.0.1:9323", true);
    context.start_session(SessionId(0xDEADBEEFu), {}, genesis);
    context.set_falcon_identity(std::vector<uint8_t>(32, 0x77), "7777777777777777", true);
    context.set_chacha20_session_key(chacha_key, fingerprint, true);

    std::string reason;
    assert(context.validate_miner_session(&reason));
    assert(reason == "PASS");

    std::cout << "Shared hex-prefix fingerprint formatting test passed!" << std::endl;
}

void test_miner_session_container_detects_inconsistent_state() {
    std::cout << "Testing miner session container consistency checks..." << std::endl;

    auto session_manager = std::make_shared<SessionManager>(24, nullptr);
    NodeSessionContext context(session_manager);

    // Test 1: Unauthenticated session should fail validation
    std::string reason;
    assert(!context.validate_miner_session(&reason));
    assert(reason.find("not authenticated") != std::string::npos);

    // Test 2: Authenticated session with valid session_id passes validation
    std::vector<uint8_t> genesis(32, 0x55);
    context.set_protocol_lane(nexusminer::ProtocolLane::STATELESS);
    context.set_connection_metadata("127.0.0.1:4000", "127.0.0.1:9323", true);
    context.start_session(SessionId(0xABCDEF01u), {}, genesis);
    assert(context.validate_miner_session(&reason));
    assert(reason == "PASS");

    std::cout << "Miner session container consistency test passed!" << std::endl;
}

void test_atomic_authenticated_session_commit_sets_auth_fields_together() {
    std::cout << "Testing atomic authenticated session commit..." << std::endl;

    auto session_manager = std::make_shared<SessionManager>(24, nullptr);
    NodeSessionContext context(session_manager);

    const SessionId committed_session_id(0x13572468u);
    const std::vector<uint8_t> genesis(32, 0x5A);
    const std::vector<uint8_t> falcon_pubkey(32, 0x7C);

    context.set_state(SessionManager::SessionState::AUTHENTICATING);
    context.commit_authenticated_session(committed_session_id, falcon_pubkey, "atomic-session-key", genesis);

    const auto info = context.get_session_info();
    assert(info.session_id == committed_session_id);
    assert(info.session_epoch == context.get_session_epoch());
    assert(info.authenticated);
    assert(info.falcon_authenticated);
    assert(info.falcon_pubkey == falcon_pubkey);
    assert(info.falcon_key_id.get() == "atomic-session-key");
    assert(info.session_genesis.get() == genesis);
    assert(context.get_state() == SessionManager::SessionState::AUTHENTICATED);
    assert(context.is_authenticated());

    std::cout << "Atomic authenticated session commit test passed!" << std::endl;
}

void test_auth_handshake_preserves_reward_crypto_material() {
    std::cout << "Testing auth handshake preserves authoritative reward crypto material..." << std::endl;

    auto session_manager = std::make_shared<SessionManager>(24, nullptr);
    NodeSessionContext context(session_manager);

    const std::vector<uint8_t> genesis(32, 0x5A);
    const std::vector<uint8_t> chacha_key(32, 0x6B);
    const auto fingerprint = format_hex_prefix(chacha_key, 8);

    context.set_tritium_genesis(genesis);
    context.set_reward_binding("reward-address", RewardHash{}, false, "config");
    context.set_chacha20_session_key(chacha_key, fingerprint, true);

    context.begin_auth_handshake("preserve reward crypto");

    auto snapshot = context.get_runtime_snapshot();
    assert(snapshot.state == SessionManager::SessionState::AUTHENTICATING);
    assert(snapshot.session_genesis.get() == genesis);
    assert(snapshot.chacha20_session_key == chacha_key);
    assert(snapshot.chacha20_key_fingerprint.get() == fingerprint);
    assert(snapshot.chacha20_ready);
    assert(snapshot.reward_address_string == "reward-address");
    assert(snapshot.reward_state == SessionManager::RewardState::REQUIRED);

    context.commit_authenticated_session(SessionId(0x11223344u),
                                         std::vector<uint8_t>(32, 0x21),
                                         "preserved-handshake-key",
                                         genesis);

    snapshot = context.get_runtime_snapshot();
    assert(snapshot.state == SessionManager::SessionState::AUTHENTICATED);
    assert(snapshot.session_genesis.get() == genesis);
    assert(snapshot.chacha20_session_key == chacha_key);
    assert(snapshot.chacha20_key_fingerprint.get() == fingerprint);
    assert(snapshot.chacha20_ready);

    const auto readiness = context.get_reward_bind_readiness();
    assert(readiness.ready);
    assert(readiness.reason == "ready");

    std::cout << "Auth handshake reward crypto preservation test passed!" << std::endl;
}

void test_reward_bind_readiness_reports_precise_missing_crypto_reason() {
    std::cout << "Testing reward bind readiness reports precise missing-crypto reason..." << std::endl;

    auto session_manager = std::make_shared<SessionManager>(24, nullptr);
    NodeSessionContext context(session_manager);

    const std::vector<uint8_t> genesis(32, 0x4C);
    context.set_reward_binding("reward-address", RewardHash{}, false, "config");
    context.commit_authenticated_session(SessionId(0xA1B2C3D4u),
                                         std::vector<uint8_t>(32, 0x33),
                                         "missing-crypto-key",
                                         genesis);

    const auto readiness = context.get_reward_bind_readiness();
    assert(!readiness.ready);
    assert(readiness.reason.find("ChaCha20 reward/session key") != std::string::npos);

    std::cout << "Reward bind readiness missing-crypto reason test passed!" << std::endl;
}

void test_reward_bind_readiness_requires_ready_chacha20_flag() {
    std::cout << "Testing reward bind readiness requires ready ChaCha20 flag..." << std::endl;

    auto session_manager = std::make_shared<SessionManager>(24, nullptr);
    NodeSessionContext context(session_manager);

    const std::vector<uint8_t> genesis(32, 0x5D);
    const std::vector<uint8_t> chacha_key(32, 0x7E);
    context.set_reward_binding("reward-address", RewardHash{}, false, "config");
    context.commit_authenticated_session(SessionId(0x0BADF00Du),
                                         std::vector<uint8_t>(32, 0x41),
                                         "not-ready-crypto-key",
                                         genesis);
    context.set_chacha20_session_key(chacha_key, format_hex_prefix(chacha_key, 8), false);

    const auto readiness = context.get_reward_bind_readiness();
    assert(!readiness.ready);
    assert(readiness.reason.find("ChaCha20 reward/session key") != std::string::npos);

    std::cout << "Reward bind readiness ready-flag test passed!" << std::endl;
}

void test_reset_session_credentials_clears_atomic_auth_flags() {
    std::cout << "Testing atomic credential reset..." << std::endl;

    auto session_manager = std::make_shared<SessionManager>(24, nullptr);
    NodeSessionContext context(session_manager);

    const SessionId committed_session_id(0x24681357u);
    context.commit_authenticated_session(
        committed_session_id,
        std::vector<uint8_t>(32, 0x33),
        "reset-session-key",
        std::vector<uint8_t>(32, 0x44));
    context.set_chacha20_session_key(std::vector<uint8_t>(32, 0x55), "5555555555555555", true);
    context.set_reward_binding("reward-address", std::vector<uint8_t>(32, 0x66), true, "live bind");
    context.set_channel_state(3, true, true);
    context.set_prevblock_suffix({0xAA, 0xBB, 0xCC, 0xDD});

    context.reset_session_credentials();

    const auto info = context.get_session_info();
    const std::array<uint8_t, 4> cleared_suffix{0, 0, 0, 0};
    // reset_session_credentials clears credential/auth flags but NOT session identity
    // (session_id, authenticated state, etc. persist — only crypto keys and readiness are reset)
    assert(info.session_id == committed_session_id);
    assert(!info.falcon_authenticated);
    assert(info.chacha20_session_key.empty());
    assert(info.chacha20_key_fingerprint.get().empty());
    assert(!info.chacha20_ready);
    assert(!info.ready_for_submit);
    assert(!info.ready_for_get_block);

    std::cout << "Atomic credential reset test passed!" << std::endl;
}

void test_multiple_session_contexts_do_not_overlap() {
    std::cout << "Testing multiple session container isolation..." << std::endl;

    auto session_manager_a = std::make_shared<SessionManager>(24, nullptr);
    auto session_manager_b = std::make_shared<SessionManager>(24, nullptr);
    NodeSessionContext context_a(session_manager_a);
    NodeSessionContext context_b(session_manager_b);

    context_a.set_falcon_identity(std::vector<uint8_t>(32, 0x01), "aaaaaaaaaaaaaaaa", true);
    context_a.start_session(SessionId(0x11111111u), {}, std::vector<uint8_t>(32, 0x10));
    context_a.set_reward_binding("reward-a", std::vector<uint8_t>(32, 0x21), true, "config");
    context_a.set_channel_state(1, true, true);

    context_b.set_falcon_identity(std::vector<uint8_t>(32, 0x02), "bbbbbbbbbbbbbbbb", true);
    context_b.start_session(SessionId(0x22222222u), {}, std::vector<uint8_t>(32, 0x20));
    context_b.set_reward_binding("reward-b", std::vector<uint8_t>(32, 0x31), true, "config");
    context_b.set_channel_state(2, true, true);

    auto info_a = context_a.get_session_info();
    auto info_b = context_b.get_session_info();

    assert(info_a.session_id == SessionId(0x11111111u));
    assert(info_b.session_id == SessionId(0x22222222u));
    assert(info_a.reward_address_string == "reward-a");
    assert(info_b.reward_address_string == "reward-b");
    assert(info_a.falcon_key_id.get() == "aaaaaaaaaaaaaaaa");
    assert(info_b.falcon_key_id.get() == "bbbbbbbbbbbbbbbb");
    assert(info_a.session_genesis != info_b.session_genesis);
    assert(info_a.reward_hash != info_b.reward_hash);

    std::cout << "Multiple session container isolation test passed!" << std::endl;
}

void test_prevblock_suffix_is_authoritative_session_state() {
    std::cout << "Testing prevblock suffix lives in the authoritative session container..." << std::endl;

    auto session_manager = std::make_shared<SessionManager>(24, nullptr);
    NodeSessionContext context(session_manager);

    const std::array<uint8_t, 4> suffix{0x12, 0x34, 0x56, 0x78};
    context.set_prevblock_suffix(suffix);

    const auto info = context.get_session_info();
    assert(info.prevblock_suffix == suffix);

    const std::vector<uint8_t> suffix_bytes(suffix.begin(), suffix.end());
    const std::string expected_suffix_hex = format_hex_prefix_via_qualified_call(suffix_bytes, 4);
    const auto diagnostics = context.build_miner_session_diagnostics();
    assert(diagnostics.find("prevblock_suffix: " + expected_suffix_hex) != std::string::npos);

    std::cout << "Prevblock suffix authoritative-state test passed!" << std::endl;
}

void test_session_epoch_advances_across_session_restarts() {
    std::cout << "Testing authoritative session epoch advancement across restarts..." << std::endl;

    auto session_manager = std::make_shared<SessionManager>(24, nullptr);
    NodeSessionContext context(session_manager);

    assert(context.get_session_epoch() == SessionEpoch(0u));

    context.start_session(SessionId(0x11111111u));
    const auto first_info = context.get_session_info();
    assert(first_info.session_id == SessionId(0x11111111u));
    assert(first_info.session_epoch == context.get_session_epoch());
    assert(first_info.session_epoch.get() > 0u);

    context.end_session();
    assert(context.get_session_epoch() == first_info.session_epoch);

    context.start_session(SessionId(0x22222222u));
    const auto second_info = context.get_session_info();
    assert(second_info.session_id == SessionId(0x22222222u));
    assert(second_info.session_epoch == context.get_session_epoch());
    assert(second_info.session_epoch.get() > first_info.session_epoch.get());

    const auto diagnostics = context.build_miner_session_diagnostics();
    assert(diagnostics.find("session_epoch: " + std::to_string(second_info.session_epoch.get())) != std::string::npos);

    std::cout << "Session epoch advancement test passed!" << std::endl;
}

void test_runtime_snapshot_is_authoritative_copy() {
    std::cout << "Testing authoritative runtime snapshot access..." << std::endl;

    auto session_manager = std::make_shared<SessionManager>(24, nullptr);
    NodeSessionContext context(session_manager);

    context.set_state(SessionManager::SessionState::AUTHENTICATING);
    context.commit_authenticated_session(SessionId(0x1234ABCDu),
                                         std::vector<uint8_t>(32, 0x11),
                                         "snapshot-key",
                                         std::vector<uint8_t>(32, 0x22));

    const auto authenticated_snapshot = context.get_runtime_snapshot();
    const auto compatibility_snapshot = context.get_session_info();
    assert(authenticated_snapshot.session_id == SessionId(0x1234ABCDu));
    assert(authenticated_snapshot.session_epoch == compatibility_snapshot.session_epoch);
    assert(authenticated_snapshot.state == SessionManager::SessionState::AUTHENTICATED);
    assert(authenticated_snapshot.falcon_key_id.get() == "snapshot-key");

    session_manager->record_keepalive();
    const auto active_snapshot = context.get_runtime_snapshot();
    assert(active_snapshot.session_id == authenticated_snapshot.session_id);
    assert(active_snapshot.session_epoch == authenticated_snapshot.session_epoch);
    assert(active_snapshot.state == SessionManager::SessionState::ACTIVE);
    // Snapshots are read-only copies of the authoritative container, so an older
    // snapshot must not change when the live session transitions forward.
    assert(authenticated_snapshot.state == SessionManager::SessionState::AUTHENTICATED);

    context.end_session();
    const auto disconnected_snapshot = context.get_runtime_snapshot();
    assert(disconnected_snapshot.session_id == SessionId(0u));
    assert(disconnected_snapshot.state == SessionManager::SessionState::DISCONNECTED);
    assert(disconnected_snapshot.session_epoch == active_snapshot.session_epoch);

    std::cout << "Authoritative runtime snapshot test passed!" << std::endl;
}

void test_hex_prefix_header_supports_qualified_callers() {
    std::cout << "Testing hex_prefix_utils qualified-call documentation path..." << std::endl;

    const std::vector<uint8_t> bytes{0xDE, 0xAD, 0xBE, 0xEF, 0xAA};
    assert(format_hex_prefix_via_qualified_call(bytes, 4) == "deadbeef");

    std::cout << "hex_prefix_utils qualified-call test passed!" << std::endl;
}

void test_session_event_journal_tracks_current_session() {
    std::cout << "Testing session event journal tracks current session..." << std::endl;

    auto session_manager = std::make_shared<SessionManager>(24, nullptr);
    NodeSessionContext context(session_manager);

    // Use begin_auth_handshake() to properly record AUTH_INIT event,
    // rather than set_state(AUTHENTICATING) which only changes state.
    session_manager->begin_auth_handshake();
    context.start_session(SessionId(0xABCDEF01u));
    context.record_session_event(SessionManager::SessionEventKind::STATUS_ACK_ACCEPTED,
                                 "session status ack accepted");
    context.set_reward_binding("reward-address", std::vector<uint8_t>(32, 0x42), true, "live bind");

    const auto journal = context.get_session_event_journal();
    assert(journal.size() >= 4);
    assert(journal[0].kind == SessionManager::SessionEventKind::AUTH_INIT);
    assert(journal[1].kind == SessionManager::SessionEventKind::AUTH_SUCCESS);
    assert(journal[2].kind == SessionManager::SessionEventKind::SESSION_START);
    assert(journal.back().kind == SessionManager::SessionEventKind::REWARD_BIND_RESULT);
    assert(journal.back().session_id.get() == 0xABCDEF01u);
    assert(journal.back().session_epoch == context.get_session_epoch());

    const auto diagnostics = context.build_miner_session_diagnostics();
    assert(diagnostics.find("SESSION EVENT JOURNAL") != std::string::npos);
    assert(diagnostics.find("auth_init") != std::string::npos);
    assert(diagnostics.find("reward_bind_result") != std::string::npos);

    std::cout << "Session event journal tracking test passed!" << std::endl;
}

void test_session_event_journal_behaves_like_ring_buffer() {
    std::cout << "Testing session event journal ring buffer behavior..." << std::endl;

    auto session_manager = std::make_shared<SessionManager>(24, nullptr);
    NodeSessionContext context(session_manager);

    context.start_session(SessionId(0x12345678u));
    for (std::size_t i = 0; i < SessionManager::SESSION_EVENT_JOURNAL_CAPACITY + 5; ++i) {
        context.record_session_event(SessionManager::SessionEventKind::STALE_PACKET_DROPPED,
                                     "drop-" + std::to_string(i));
    }

    const auto journal = context.get_session_event_journal();
    assert(journal.size() == SessionManager::SESSION_EVENT_JOURNAL_CAPACITY);
    assert(journal.front().detail == "drop-5");
    assert(journal.back().detail ==
           "drop-" + std::to_string(SessionManager::SESSION_EVENT_JOURNAL_CAPACITY + 4));

    std::cout << "Session event journal ring buffer test passed!" << std::endl;
}

void test_session_event_journal_preserves_preflight_drop_detail() {
    std::cout << "Testing session event journal preserves ingress preflight detail..." << std::endl;

    auto session_manager = std::make_shared<SessionManager>(24, nullptr);
    NodeSessionContext context(session_manager);

    context.start_session(SessionId(0xCAFEBABEu));
    context.record_session_event(SessionManager::SessionEventKind::EPOCH_MISMATCH,
                                 "Solo SessionKeepalive: packet ownership epoch mismatched authoritative session");

    const auto diagnostics = context.build_miner_session_diagnostics();
    assert(diagnostics.find("epoch_mismatch") != std::string::npos);
    assert(diagnostics.find("packet ownership epoch mismatched authoritative session") != std::string::npos);

    std::cout << "Session event journal preflight detail test passed!" << std::endl;
}

void test_authoritative_transition_apis_drive_lifecycle_state() {
    std::cout << "Testing authoritative transition APIs drive lifecycle state..." << std::endl;

    auto session_manager = std::make_shared<SessionManager>(24, nullptr);
    NodeSessionContext context(session_manager);

    context.set_reward_binding("reward-address", RewardHash{}, false, "config");
    context.begin_auth_handshake("unit test auth");
    auto snapshot = context.get_runtime_snapshot();
    assert(snapshot.state == SessionManager::SessionState::AUTHENTICATING);
    assert(snapshot.reward_state == SessionManager::RewardState::REQUIRED);
    assert(snapshot.recovery_state == SessionManager::RecoveryState::HEALTHY);

    context.commit_authenticated_session(SessionId(0xABCDEF12u),
                                         std::vector<uint8_t>(32, 0x21),
                                         "transition-key",
                                         std::vector<uint8_t>(32, 0x34));
    snapshot = context.get_runtime_snapshot();
    assert(snapshot.state == SessionManager::SessionState::AUTHENTICATED);
    assert(context.allow_deferred_push_replay());
    assert(!context.can_request_get_block());
    assert(context.reward_binding_required());

    context.begin_reward_binding("reward-address", std::vector<uint8_t>(32, 0x45), "live bind");
    snapshot = context.get_runtime_snapshot();
    assert(snapshot.reward_state == SessionManager::RewardState::BINDING);
    assert(!context.is_reward_bound());

    context.commit_reward_bound("reward-address", std::vector<uint8_t>(32, 0x45), "live bind");
    context.set_channel_state(2, true, true);
    context.note_keepalive_ack(true, "ack ok");
    snapshot = context.get_runtime_snapshot();
    assert(snapshot.state == SessionManager::SessionState::ACTIVE);
    assert(snapshot.reward_state == SessionManager::RewardState::BOUND);
    assert(snapshot.expiry_state == SessionManager::ExpiryState::FRESH);
    assert(context.is_reward_bound());
    assert(context.can_request_get_block());
    assert(context.can_submit_work());
    assert(context.allow_get_block_replay());

    context.mark_recovery_required("epoch mismatch");
    snapshot = context.get_runtime_snapshot();
    assert(snapshot.recovery_state == SessionManager::RecoveryState::RECOVERY_PENDING);
    assert(snapshot.recovery_reason == "epoch mismatch");

    context.mark_recovery_healthy("template delivered");
    snapshot = context.get_runtime_snapshot();
    assert(snapshot.recovery_state == SessionManager::RecoveryState::HEALTHY);
    assert(snapshot.recovery_reason.empty());

    context.mark_session_expired("ack mismatch");
    snapshot = context.get_runtime_snapshot();
    assert(snapshot.state == SessionManager::SessionState::EXPIRED);
    assert(snapshot.reward_state == SessionManager::RewardState::STALE);
    assert(snapshot.recovery_state == SessionManager::RecoveryState::FORCED_REAUTH);
    assert(snapshot.expiry_state == SessionManager::ExpiryState::EXPIRED_ACCEPTED);
    assert(!context.can_submit_work());
    assert(!context.allow_get_block_replay());

    context.clear_for_reauth("reward-address", "config", "reauth requested");
    snapshot = context.get_runtime_snapshot();
    assert(snapshot.state == SessionManager::SessionState::AUTHENTICATING);
    assert(snapshot.reward_state == SessionManager::RewardState::REQUIRED);
    assert(snapshot.recovery_state == SessionManager::RecoveryState::RECOVERY_IN_PROGRESS);
    assert(snapshot.reward_address_string == "reward-address");
    assert(!snapshot.reward_bound);
    assert(!context.allow_deferred_push_replay());
    assert(!context.can_request_get_block());
    assert(!context.can_submit_work());

    const auto diagnostics = context.build_miner_session_diagnostics();
    assert(diagnostics.find("reward_state: REQUIRED") != std::string::npos);
    assert(diagnostics.find("recovery_state: RECOVERY_IN_PROGRESS") != std::string::npos);
    assert(diagnostics.find("expiry_state: FRESH") != std::string::npos);

    std::cout << "Authoritative transition API test passed!" << std::endl;
}

void test_session_semantic_wrapper_types_are_distinct() {
    std::cout << "Testing strong semantic session wrapper types..." << std::endl;

    static_assert(!std::is_same<SessionId, SessionEpoch>::value, "SessionId and SessionEpoch must differ");
    static_assert(!std::is_same<SessionGenesisHash, RewardHash>::value, "Genesis and reward hashes must differ");
    static_assert(!std::is_same<FalconHashKeyId, SessionFingerprint>::value, "Key id and fingerprint must differ");

    const SessionId session_id(0x11111111u);
    const SessionEpoch session_epoch(9);
    const SessionGenesisHash genesis_hash(std::vector<uint8_t>(32, 0xAA));
    const RewardHash reward_hash(std::vector<uint8_t>(32, 0xBB));
    const FalconHashKeyId key_id(std::string("falcon-key"));
    const SessionFingerprint fingerprint(std::string("fingerprint"));

    assert(session_id.get() == 0x11111111u);
    assert(session_epoch.get() == 9u);
    assert(genesis_hash.get().size() == 32);
    assert(reward_hash.get().size() == 32);
    assert(key_id.get() == "falcon-key");
    assert(fingerprint.get() == "fingerprint");

    std::cout << "Strong semantic session wrapper types test passed!" << std::endl;
}

int main() {
    std::cout << "Running NodeSessionContext unit tests..." << std::endl;

    try {
        test_session_lifecycle();
        test_session_constants();
        test_parse_session_start();
        test_keepalive_interval();
        test_tritium_genesis();
        test_session_manager_access();
        test_authoritative_miner_session_container_binding();
        test_reward_binding_persists_across_session_restart();
        test_format_hex_prefix_matches_session_validation_fingerprint();
        test_miner_session_container_detects_inconsistent_state();
        test_atomic_authenticated_session_commit_sets_auth_fields_together();
        test_auth_handshake_preserves_reward_crypto_material();
        test_reward_bind_readiness_reports_precise_missing_crypto_reason();
        test_reward_bind_readiness_requires_ready_chacha20_flag();
        test_reset_session_credentials_clears_atomic_auth_flags();
        test_multiple_session_contexts_do_not_overlap();
        test_prevblock_suffix_is_authoritative_session_state();
        test_session_epoch_advances_across_session_restarts();
        test_runtime_snapshot_is_authoritative_copy();
        test_hex_prefix_header_supports_qualified_callers();
        test_session_event_journal_tracks_current_session();
        test_session_event_journal_behaves_like_ring_buffer();
        test_session_event_journal_preserves_preflight_drop_detail();
        test_authoritative_transition_apis_drive_lifecycle_state();
        test_session_semantic_wrapper_types_are_distinct();

        std::cout << "\nAll NodeSessionContext tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
