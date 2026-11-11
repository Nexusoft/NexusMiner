#include "include/stateless_block_utility.hpp"
#include "protocol/mining_template_interface.hpp"
#include "protocol/node_session_context.hpp"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include <openssl/sha.h>

using namespace nexusminer;
using namespace nexusminer::protocol;

namespace {

constexpr uint32_t MINER_A_SESSION_ID = 0x11111111u;
constexpr uint32_t MINER_B_SESSION_ID = 0x22222222u;
const std::string KDF_DOMAIN = "nexus-mining-chacha20-v1";

std::vector<uint8_t> make_repeated_bytes(std::size_t size, uint8_t seed)
{
    std::vector<uint8_t> bytes(size, 0);
    for (std::size_t i = 0; i < size; ++i) {
        bytes[i] = static_cast<uint8_t>(seed + static_cast<uint8_t>(i));
    }
    return bytes;
}

std::vector<uint8_t> derive_session_key(const std::vector<uint8_t>& genesis)
{
    std::vector<uint8_t> preimage;
    preimage.insert(preimage.end(), KDF_DOMAIN.begin(), KDF_DOMAIN.end());
    preimage.insert(preimage.end(), genesis.begin(), genesis.end());

    std::vector<uint8_t> key(SHA256_DIGEST_LENGTH);
    SHA256(preimage.data(), preimage.size(), key.data());
    return key;
}

uint8_t be_byte(uint32_t value, int index)
{
    return static_cast<uint8_t>((value >> (24 - 8 * index)) & 0xFF);
}

std::array<uint8_t, 128> make_prev_hash_pattern(uint8_t seed)
{
    std::array<uint8_t, 128> prev_hash{};
    for (std::size_t i = 0; i < prev_hash.size(); ++i) {
        prev_hash[i] = static_cast<uint8_t>(seed + static_cast<uint8_t>(i));
    }
    return prev_hash;
}

network::Payload make_template_payload(uint32_t unified_height,
                                       uint32_t channel_height,
                                       uint32_t difficulty,
                                       uint32_t n_version,
                                       uint32_t n_channel,
                                       uint32_t n_height,
                                       uint32_t n_bits,
                                       uint64_t n_nonce,
                                       const std::array<uint8_t, 128>& prev_hash)
{
    network::Payload buf(228, 0x00);

    buf[0] = be_byte(unified_height, 0);  buf[1] = be_byte(unified_height, 1);
    buf[2] = be_byte(unified_height, 2);  buf[3] = be_byte(unified_height, 3);
    buf[4] = be_byte(channel_height, 0);  buf[5] = be_byte(channel_height, 1);
    buf[6] = be_byte(channel_height, 2);  buf[7] = be_byte(channel_height, 3);
    buf[8] = be_byte(difficulty, 0);      buf[9] = be_byte(difficulty, 1);
    buf[10] = be_byte(difficulty, 2);     buf[11] = be_byte(difficulty, 3);

    std::size_t offset = 12;
    buf[offset] = be_byte(n_version, 0);      buf[offset + 1] = be_byte(n_version, 1);
    buf[offset + 2] = be_byte(n_version, 2);  buf[offset + 3] = be_byte(n_version, 3);
    offset += 4;

    std::copy(prev_hash.begin(), prev_hash.end(), buf.begin() + static_cast<std::ptrdiff_t>(offset));
    offset += prev_hash.size();

    for (std::size_t i = 0; i < 64; ++i) {
        buf[offset + i] = static_cast<uint8_t>(0xA0 + (i & 0x0F));
    }
    offset += 64;

    buf[offset] = be_byte(n_channel, 0);      buf[offset + 1] = be_byte(n_channel, 1);
    buf[offset + 2] = be_byte(n_channel, 2);  buf[offset + 3] = be_byte(n_channel, 3);
    offset += 4;

    buf[offset] = be_byte(n_height, 0);       buf[offset + 1] = be_byte(n_height, 1);
    buf[offset + 2] = be_byte(n_height, 2);   buf[offset + 3] = be_byte(n_height, 3);
    offset += 4;

    buf[offset] = be_byte(n_bits, 0);         buf[offset + 1] = be_byte(n_bits, 1);
    buf[offset + 2] = be_byte(n_bits, 2);     buf[offset + 3] = be_byte(n_bits, 3);
    offset += 4;

    for (int i = 0; i < 8; ++i) {
        buf[offset + static_cast<std::size_t>(i)] =
            static_cast<uint8_t>((n_nonce >> (i * 8)) & 0xFF);
    }

    return buf;
}

struct MinerFixture
{
    std::shared_ptr<SessionManager> manager;
    NodeSessionContext context;
    std::vector<uint8_t> genesis;
    std::vector<uint8_t> reward_hash;

    MinerFixture()
        : manager(std::make_shared<SessionManager>(24, nullptr))
        , context(manager)
    {
    }
};

MinerFixture make_miner_fixture(uint32_t session_id,
                                uint8_t pubkey_seed,
                                uint8_t genesis_seed,
                                const std::string& falcon_key_id)
{
    MinerFixture fixture;
    fixture.genesis = make_repeated_bytes(32, genesis_seed);
    fixture.reward_hash = make_repeated_bytes(32, 0x44);
    const auto chacha_key = derive_session_key(fixture.genesis);

    fixture.context.set_protocol_lane(ProtocolLane::STATELESS);
    fixture.context.begin_auth_handshake("multi-miner test");
    fixture.context.commit_authenticated_session(SessionId(session_id),
                                                make_repeated_bytes(32, pubkey_seed),
                                                FalconHashKeyId(falcon_key_id),
                                                SessionGenesisHash(fixture.genesis));
    fixture.context.set_chacha20_session_key(chacha_key,
                                             SessionFingerprint("fingerprint-" + falcon_key_id),
                                             true);
    fixture.context.set_reward_binding("shared-reward-address",
                                       RewardHash(fixture.reward_hash),
                                       true,
                                       "config");
    fixture.context.set_channel_state(2, true, true);
    return fixture;
}

void test_same_reward_string_still_preserves_distinct_session_ownership()
{
    std::cout << "Test 1: shared reward string does not imply shared session ownership" << std::endl;

    const auto miner_a = make_miner_fixture(MINER_A_SESSION_ID, 0x10, 0x20, "miner-a-key");
    const auto miner_b = make_miner_fixture(MINER_B_SESSION_ID, 0x40, 0x60, "miner-b-key");

    const auto binding_a = miner_a.context.get_session_binding();
    const auto binding_b = miner_b.context.get_session_binding();

    assert(binding_a.reward_address == binding_b.reward_address);
    assert(binding_a.reward_hash == binding_b.reward_hash);
    assert(binding_a.session_id != binding_b.session_id);
    assert(binding_a.identity.is_valid());
    assert(binding_b.identity.is_valid());
    assert(!binding_a.identity.matches(binding_b.identity));
    assert(!binding_a.identity.same_miner(binding_b.identity));
    assert(binding_a.identity_matches_session());
    assert(binding_b.identity_matches_session());
}

void test_reauth_of_miner_a_does_not_mutate_miner_b()
{
    std::cout << "Test 2: miner A reauth does not mutate miner B state" << std::endl;

    auto miner_a = make_miner_fixture(MINER_A_SESSION_ID, 0x10, 0x20, "miner-a-key");
    auto miner_b = make_miner_fixture(MINER_B_SESSION_ID, 0x40, 0x60, "miner-b-key");

    const auto binding_b_before = miner_b.context.get_session_binding();
    const auto journal_b_before = miner_b.context.build_session_event_journal();

    miner_a.context.clear_for_reauth("shared-reward-address", "config", "miner a reauth");
    miner_a.context.begin_auth_handshake("reauth miner a");
    const auto new_genesis = make_repeated_bytes(32, 0x70);
    miner_a.context.commit_authenticated_session(SessionId(0x33333333u),
                                                make_repeated_bytes(32, 0x10),
                                                FalconHashKeyId("miner-a-key"),
                                                SessionGenesisHash(new_genesis));
    miner_a.context.set_chacha20_session_key(derive_session_key(new_genesis),
                                             SessionFingerprint("fingerprint-miner-a-key"),
                                             true);
    miner_a.context.set_reward_binding("shared-reward-address",
                                       RewardHash(miner_a.reward_hash),
                                       true,
                                       "config");
    miner_a.context.set_channel_state(2, true, true);

    const auto binding_b_after = miner_b.context.get_session_binding();
    const auto journal_b_after = miner_b.context.build_session_event_journal();

    assert(binding_b_after.session_id == binding_b_before.session_id);
    assert(binding_b_after.session_genesis == binding_b_before.session_genesis);
    assert(binding_b_after.reward_hash == binding_b_before.reward_hash);
    assert(binding_b_after.identity.matches(binding_b_before.identity));
    assert(binding_b_after.ready_for_submit == binding_b_before.ready_for_submit);
    assert(journal_b_after == journal_b_before);
}

void test_template_ownership_stays_bound_to_each_miner()
{
    std::cout << "Test 3: template ownership stays bound to each miner" << std::endl;

    auto miner_a = make_miner_fixture(MINER_A_SESSION_ID, 0x10, 0x20, "miner-a-key");
    auto miner_b = make_miner_fixture(MINER_B_SESSION_ID, 0x40, 0x60, "miner-b-key");
    const auto binding_a = miner_a.context.get_session_binding();
    const auto binding_b = miner_b.context.get_session_binding();

    MiningTemplateInterface template_a(2, binding_a.session_id);
    MiningTemplateInterface template_b(2, binding_b.session_id);
    template_a.set_session_binding(binding_a);
    template_b.set_session_binding(binding_b);

    const auto payload = make_template_payload(6000000u, 2000000u, 0x04308519u, 8u, 2u, 6000001u, 0x04308519u, 0, make_prev_hash_pattern(0x30));
    assert(StatelessBlockUtility::decode_template(template_a, payload, 2u, nullptr).valid);
    assert(StatelessBlockUtility::decode_template(template_b, payload, 2u, nullptr).valid);
    template_a.set_session_binding(binding_a);
    template_b.set_session_binding(binding_b);

    const auto* current_a = template_a.get_current_template();
    const auto* current_b = template_b.get_current_template();
    assert(current_a != nullptr);
    assert(current_b != nullptr);
    assert(template_a.get_session_identity().matches(binding_a.identity));
    assert(template_b.get_session_identity().matches(binding_b.identity));
    assert(!template_a.get_session_identity().matches(template_b.get_session_identity()));

    const auto rebinding_a = make_miner_fixture(0x44444444u, 0x10, 0x80, "miner-a-key");
    template_a.set_session_binding(rebinding_a.context.get_session_binding());

    assert(template_a.get_session_identity().matches(rebinding_a.context.get_session_binding().identity));
    assert(template_b.get_session_identity().matches(binding_b.identity));
}

} // namespace

int main()
{
    std::cout << "Running multi-miner session isolation tests..." << std::endl;

    test_same_reward_string_still_preserves_distinct_session_ownership();
    test_reauth_of_miner_a_does_not_mutate_miner_b();
    test_template_ownership_stays_bound_to_each_miner();

    std::cout << "\nAll multi-miner session isolation tests passed!" << std::endl;
    return 0;
}
