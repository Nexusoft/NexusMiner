#include "include/stateless_block_utility.hpp"
#include "protocol/hex_prefix_utils.hpp"
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

constexpr uint32_t TEST_SESSION_ID = 0x12345678u;
constexpr uint32_t TEST_TIMEOUT_SECONDS = 7200u;
constexpr uint32_t TEST_UNIFIED_HEIGHT = 6000000u;
constexpr uint32_t TEST_CHANNEL_HEIGHT = 2000000u;
constexpr uint32_t TEST_TEMPLATE_HEIGHT = 6000001u;
constexpr uint32_t TEST_CHANNEL = 2u;
constexpr uint64_t TEST_NONCE = 0x0102030405060708ULL;
const std::string KDF_DOMAIN = "nexus-mining-chacha20-v1";

uint8_t be_byte(uint32_t value, int index)
{
    return static_cast<uint8_t>((value >> (24 - 8 * index)) & 0xFF);
}

uint8_t le_byte(uint32_t value, int index)
{
    return static_cast<uint8_t>((value >> (8 * index)) & 0xFF);
}

std::vector<uint8_t> make_repeated_bytes(std::size_t size, uint8_t seed)
{
    std::vector<uint8_t> bytes(size, 0);
    for (std::size_t i = 0; i < size; ++i) {
        bytes[i] = static_cast<uint8_t>(seed + static_cast<uint8_t>(i));
    }
    return bytes;
}

std::array<uint8_t, 128> make_prev_hash_pattern(uint8_t seed)
{
    std::array<uint8_t, 128> prev_hash{};
    for (std::size_t i = 0; i < prev_hash.size(); ++i) {
        prev_hash[i] = static_cast<uint8_t>(seed + static_cast<uint8_t>(i));
    }
    return prev_hash;
}

network::Payload make_session_start_packet(uint32_t session_id,
                                           uint32_t timeout_seconds,
                                           const std::vector<uint8_t>& genesis)
{
    network::Payload packet{
        0x01,
        le_byte(session_id, 0), le_byte(session_id, 1), le_byte(session_id, 2), le_byte(session_id, 3),
        le_byte(timeout_seconds, 0), le_byte(timeout_seconds, 1),
        le_byte(timeout_seconds, 2), le_byte(timeout_seconds, 3)
    };
    packet.insert(packet.end(), genesis.begin(), genesis.end());
    return packet;
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

std::vector<uint8_t> strip_wire_header(const std::vector<uint8_t>& framed, ProtocolLane lane)
{
    const std::size_t header_size = (lane == ProtocolLane::STATELESS) ? 6u : 5u;
    if (framed.size() <= header_size) {
        return {};
    }

    return std::vector<uint8_t>(framed.begin() + static_cast<std::ptrdiff_t>(header_size), framed.end());
}

uint32_t extract_serialized_block_height(const std::vector<uint8_t>& block_payload)
{
    constexpr std::size_t TRITIUM_HEIGHT_OFFSET = 200u;
    if (block_payload.size() < TRITIUM_HEIGHT_OFFSET + 4u) {
        return 0;
    }

    return (static_cast<uint32_t>(block_payload[TRITIUM_HEIGHT_OFFSET]) << 24) |
           (static_cast<uint32_t>(block_payload[TRITIUM_HEIGHT_OFFSET + 1]) << 16) |
           (static_cast<uint32_t>(block_payload[TRITIUM_HEIGHT_OFFSET + 2]) << 8) |
            static_cast<uint32_t>(block_payload[TRITIUM_HEIGHT_OFFSET + 3]);
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

SessionBinding make_test_binding()
{
    auto session_manager = std::make_shared<SessionManager>(24, nullptr);
    NodeSessionContext context(session_manager);

    const auto genesis = make_repeated_bytes(32, 0x10);
    const auto reward_hash = make_repeated_bytes(32, 0x44);
    const auto chacha_key = derive_session_key(genesis);

    context.set_protocol_lane(ProtocolLane::STATELESS);
    context.begin_auth_handshake("cross-arch test");
    context.commit_authenticated_session(SessionId(TEST_SESSION_ID),
                                         make_repeated_bytes(32, 0x20),
                                         FalconHashKeyId("cross-arch-key"),
                                         SessionGenesisHash(genesis));
    context.set_chacha20_session_key(chacha_key, SessionFingerprint("cross-arch-fingerprint"), true);
    context.set_reward_binding("reward-address", RewardHash(reward_hash), true, "test");
    context.set_channel_state(TEST_CHANNEL, true, true);

    return context.get_session_binding();
}

void test_session_start_packet_is_explicitly_little_endian()
{
    std::cout << "Test 1: session_start wire bytes stay explicit-endian" << std::endl;

    const auto genesis = make_repeated_bytes(32, 0x10);
    const auto packet = make_session_start_packet(TEST_SESSION_ID, TEST_TIMEOUT_SECONDS, genesis);

    assert(packet.size() == 41u);
    assert(packet[0] == 0x01);
    assert(packet[1] == 0x78 && packet[2] == 0x56 && packet[3] == 0x34 && packet[4] == 0x12);
    assert(packet[5] == 0x20 && packet[6] == 0x1C && packet[7] == 0x00 && packet[8] == 0x00);

    uint32_t parsed_session_id = 0;
    uint32_t parsed_timeout = 0;
    std::vector<uint8_t> parsed_genesis;
    assert(NodeSessionContext::parse_session_start(packet,
                                                   parsed_session_id,
                                                   parsed_timeout,
                                                   parsed_genesis));
    assert(parsed_session_id == TEST_SESSION_ID);
    assert(parsed_timeout == TEST_TIMEOUT_SECONDS);
    assert(parsed_genesis == genesis);
}

void test_session_binding_preserves_exact_genesis_reward_and_crypto_bytes()
{
    std::cout << "Test 2: SessionBinding preserves exact canonical bytes" << std::endl;

    const auto binding = make_test_binding();
    const auto expected_genesis = make_repeated_bytes(32, 0x10);
    const auto expected_reward_hash = make_repeated_bytes(32, 0x44);

    assert(binding.authenticated);
    assert(binding.has_session());
    assert(binding.has_crypto_context());
    assert(binding.session_genesis == SessionGenesisHash(expected_genesis));
    assert(binding.reward_hash == RewardHash(expected_reward_hash));
    assert(format_hex_prefix(binding.chacha20_session_key, binding.chacha20_session_key.size()) ==
           "44fb56dfc0df35ff274a9f7d33fdae20c248439b652ffa966dce3699cbffa557");
}

void test_submit_encoding_is_byte_stable_across_replay()
{
    std::cout << "Test 3: submit encoding remains byte-stable across replay" << std::endl;

    const auto binding = make_test_binding();
    const auto template_payload = make_template_payload(
        TEST_UNIFIED_HEIGHT,
        TEST_CHANNEL_HEIGHT,
        0x04308519u,
        8u,
        TEST_CHANNEL,
        TEST_TEMPLATE_HEIGHT,
        0x04308519u,
        0,
        make_prev_hash_pattern(0x30));

    MiningTemplateInterface template_a(static_cast<uint8_t>(TEST_CHANNEL), binding.session_id);
    MiningTemplateInterface template_b(static_cast<uint8_t>(TEST_CHANNEL), binding.session_id);
    template_a.set_session_binding(binding);
    template_b.set_session_binding(binding);

    const auto decoded_a = StatelessBlockUtility::decode_template(template_a, template_payload, TEST_CHANNEL, nullptr);
    const auto decoded_b = StatelessBlockUtility::decode_template(template_b, template_payload, TEST_CHANNEL, nullptr);
    assert(decoded_a.valid);
    assert(decoded_b.valid);

    const auto* current_a = template_a.get_current_template();
    const auto* current_b = template_b.get_current_template();
    assert(current_a != nullptr);
    assert(current_b != nullptr);

    ::LLP::CBlock solved_a = current_a->block;
    ::LLP::CBlock solved_b = current_b->block;
    solved_a.nNonce = TEST_NONCE;
    solved_b.nNonce = TEST_NONCE;

    SubmitContext submit_context;
    submit_context.session_id = binding.session_id;
    submit_context.session_epoch = binding.session_epoch;
    submit_context.template_height = current_a->block.nHeight;
    submit_context.chain_height = TEST_UNIFIED_HEIGHT;
    submit_context.identity = binding.identity;

    const auto submit_a = StatelessBlockUtility::encode_submit(
        template_a, solved_a, {}, nullptr, ProtocolLane::STATELESS, HeightTracker::Snapshot{}, nullptr, submit_context);
    const auto submit_b = StatelessBlockUtility::encode_submit(
        template_b, solved_b, {}, nullptr, ProtocolLane::STATELESS, HeightTracker::Snapshot{}, nullptr, submit_context);

    assert(submit_a.valid && submit_a.wire_bytes && !submit_a.wire_bytes->empty());
    assert(submit_b.valid && submit_b.wire_bytes && !submit_b.wire_bytes->empty());
    assert(*submit_a.wire_bytes == *submit_b.wire_bytes);
    assert((*submit_a.wire_bytes)[0] == 0xD0 && (*submit_a.wire_bytes)[1] == 0x01);

    const auto plaintext = strip_wire_header(*submit_a.wire_bytes, ProtocolLane::STATELESS);
    assert(extract_serialized_block_height(plaintext) == TEST_TEMPLATE_HEIGHT);
}

} // namespace

int main()
{
    std::cout << "Running cross-architecture serialization tests..." << std::endl;

    test_session_start_packet_is_explicitly_little_endian();
    test_session_binding_preserves_exact_genesis_reward_and_crypto_bytes();
    test_submit_encoding_is_byte_stable_across_replay();

    std::cout << "\nAll cross-architecture serialization tests passed!" << std::endl;
    return 0;
}
