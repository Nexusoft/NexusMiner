#include "protocol/node_session_context.hpp"
#include "protocol/session_binding.hpp"
#include <cassert>
#include <iostream>
#include <vector>

using namespace nexusminer::protocol;

namespace {

void test_default_binding()
{
    SessionBinding binding;
    assert(!binding.has_session());
    assert(binding.session_requires_full_recovery());
    assert(!binding.can_submit_work());
    assert(!binding.can_request_get_block());
    assert(!binding.may_request_work());
    assert(!binding.is_fully_mining_ready());
    assert(!binding.has_crypto_context());
    assert(binding.identity_matches_session());
    std::cout << "  PASS: default_binding\n";
}

void test_context_exposes_batched_binding()
{
    auto session_manager = std::make_shared<SessionManager>(24, nullptr);
    NodeSessionContext context(session_manager);

    const SessionId session_id(0x12345678u);
    const std::vector<uint8_t> genesis(32, 0x11);
    const std::vector<uint8_t> reward_hash(32, 0x22);
    const std::vector<uint8_t> chacha_key(32, 0x33);

    context.set_protocol_lane(nexusminer::ProtocolLane::STATELESS);
    context.set_falcon_identity(std::vector<uint8_t>(32, 0x44),
                                FalconHashKeyId("binding-falcon-key"),
                                true);
    context.commit_authenticated_session(session_id,
                                         std::vector<uint8_t>(32, 0x44),
                                         FalconHashKeyId("binding-falcon-key"),
                                         SessionGenesisHash(genesis));
    context.set_chacha20_session_key(chacha_key,
                                     SessionFingerprint("binding-fingerprint"),
                                     true);
    context.set_reward_binding("reward-address", RewardHash(reward_hash), true, "config");
    context.set_channel_state(2, true, true);

    const auto binding = context.get_session_binding();
    assert(binding.has_session());
    assert(binding.session_id == session_id);
    assert(binding.session_epoch == context.get_session_epoch());
    assert(binding.session_genesis == SessionGenesisHash(genesis));
    assert(binding.falcon_key_id == FalconHashKeyId("binding-falcon-key"));
    assert(binding.chacha20_session_key == chacha_key);
    assert(binding.chacha20_key_fingerprint == SessionFingerprint("binding-fingerprint"));
    assert(binding.active_lane == nexusminer::ProtocolLane::STATELESS);
    assert(binding.authenticated);
    assert(binding.chacha20_ready);
    assert(binding.reward_address == "reward-address");
    assert(binding.reward_hash == RewardHash(reward_hash));
    assert(binding.reward_bound);
    assert(binding.channel == 2);
    assert(!binding.session_requires_full_recovery());
    assert(binding.may_request_work());
    assert(binding.is_fully_mining_ready());
    assert(binding.can_submit_work());
    assert(binding.can_request_get_block());
    assert(binding.has_crypto_context());
    assert(binding.identity.is_valid());
    assert(binding.identity_matches_session());
    std::cout << "  PASS: context_exposes_batched_binding\n";
}

void test_binding_clears_crypto_after_session_expiry()
{
    auto session_manager = std::make_shared<SessionManager>(24, nullptr);
    NodeSessionContext context(session_manager);

    const SessionId session_id(0xAABBCCDDu);
    const std::vector<uint8_t> genesis(32, 0x55);
    const std::vector<uint8_t> reward_hash(32, 0x66);
    const std::vector<uint8_t> chacha_key(32, 0x77);

    context.set_protocol_lane(nexusminer::ProtocolLane::STATELESS);
    context.commit_authenticated_session(session_id,
                                         std::vector<uint8_t>(32, 0x44),
                                         FalconHashKeyId("expired-session-key"),
                                         SessionGenesisHash(genesis));
    context.set_chacha20_session_key(chacha_key,
                                     SessionFingerprint("expired-fingerprint"),
                                     true);
    context.set_reward_binding("reward-address", RewardHash(reward_hash), true, "config");
    context.mark_session_expired("ack mismatch");

    const auto binding = context.get_session_binding();
    assert(binding.session_id == session_id);
    assert(!binding.authenticated);
    assert(binding.session_requires_full_recovery());
    assert(!binding.may_request_work());
    assert(!binding.is_fully_mining_ready());
    assert(!binding.chacha20_ready);
    assert(!binding.has_crypto_context());
    assert(binding.reward_bound);
    assert(!binding.identity.is_valid());
    std::cout << "  PASS: binding_clears_crypto_after_session_expiry\n";
}

void test_binding_exposes_reauth_in_progress_as_full_recovery()
{
    auto session_manager = std::make_shared<SessionManager>(24, nullptr);
    NodeSessionContext context(session_manager);

    context.set_protocol_lane(nexusminer::ProtocolLane::STATELESS);
    context.commit_authenticated_session(SessionId(0x10203040u),
                                         std::vector<uint8_t>(32, 0x44),
                                         FalconHashKeyId("reauth-key"),
                                         SessionGenesisHash(std::vector<uint8_t>(32, 0x11)));
    context.set_reward_binding("reward-address", RewardHash(std::vector<uint8_t>(32, 0x22)), true, "config");
    context.set_channel_state(2, true, true);
    context.clear_for_reauth("reward-address", "config", "reauth requested");

    const auto binding = context.get_session_binding();
    assert(binding.session_requires_full_recovery());
    assert(!binding.may_request_work());
    assert(!binding.is_fully_mining_ready());
    std::cout << "  PASS: binding_exposes_reauth_in_progress_as_full_recovery\n";
}

void test_reward_binding_without_channel_ready_cannot_request_work()
{
    auto session_manager = std::make_shared<SessionManager>(24, nullptr);
    NodeSessionContext context(session_manager);

    context.set_protocol_lane(nexusminer::ProtocolLane::STATELESS);
    context.commit_authenticated_session(SessionId(0x99887766u),
                                         std::vector<uint8_t>(32, 0x44),
                                         FalconHashKeyId("reward-only-key"),
                                         SessionGenesisHash(std::vector<uint8_t>(32, 0x11)));
    context.set_reward_binding("reward-address", RewardHash(std::vector<uint8_t>(32, 0x22)), true, "config");

    const auto binding = context.get_session_binding();
    assert(binding.reward_bound);
    assert(!binding.ready_for_get_block);
    assert(!binding.can_request_get_block());
    std::cout << "  PASS: reward_binding_without_channel_ready_cannot_request_work\n";
}

void test_channel_state_clear_is_authoritative()
{
    auto session_manager = std::make_shared<SessionManager>(24, nullptr);
    NodeSessionContext context(session_manager);

    context.set_protocol_lane(nexusminer::ProtocolLane::STATELESS);
    context.commit_authenticated_session(SessionId(0x55667788u),
                                         std::vector<uint8_t>(32, 0x44),
                                         FalconHashKeyId("clear-state-key"),
                                         SessionGenesisHash(std::vector<uint8_t>(32, 0x11)));
    context.set_reward_binding("reward-address", RewardHash(std::vector<uint8_t>(32, 0x22)), true, "config");
    context.set_channel_state(2, true, true);

    auto ready_binding = context.get_session_binding();
    assert(ready_binding.can_request_get_block());
    assert(ready_binding.can_submit_work());

    context.set_channel_state(2, false, false);
    const auto cleared_binding = context.get_session_binding();
    assert(!cleared_binding.ready_for_get_block);
    assert(!cleared_binding.ready_for_submit);
    assert(!cleared_binding.can_request_get_block());
    assert(!cleared_binding.can_submit_work());
    std::cout << "  PASS: channel_state_clear_is_authoritative\n";
}

} // namespace

int main()
{
    test_default_binding();
    test_context_exposes_batched_binding();
    test_binding_clears_crypto_after_session_expiry();
    test_binding_exposes_reauth_in_progress_as_full_recovery();
    test_reward_binding_without_channel_ready_cannot_request_work();
    test_channel_state_clear_is_authoritative();
    std::cout << "All SessionBinding tests passed!\n";
    return 0;
}
