#include "include/stateless_block_utility.hpp"
#include "miner_opcodes.hpp"
#include "protocol/chacha20_wrapper.hpp"
#include "protocol/hex_prefix_utils.hpp"
#include "protocol/mining_template_interface.hpp"
#include "protocol/node_session_context.hpp"
#include "protocol/packet_builder.hpp"
#include "worker/worker.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <openssl/sha.h>

using namespace nexusminer;
using namespace nexusminer::protocol;

namespace {

int tests_run = 0;
int tests_passed = 0;
int tests_failed = 0;

constexpr uint32_t DEFAULT_DIFFICULTY = 0x04308519u;
constexpr uint16_t DEFAULT_KEEPALIVE_INTERVAL_HOURS = 24u;
constexpr uint32_t SECONDS_PER_HOUR = 3600u;
constexpr uint32_t DEFAULT_BLOCK_VERSION = 8u;
constexpr uint32_t ACCEPTANCE_UNIFIED_HEIGHT = 6000000u;
constexpr uint32_t ACCEPTANCE_CHANNEL_HEIGHT = 2000000u;
constexpr uint32_t ACCEPTANCE_TEMPLATE_HEIGHT = 6000001u;
constexpr uint32_t ACCEPTANCE_CHANNEL = 2u;
constexpr uint64_t ACCEPTANCE_NONCE = 0x0102030405060708ULL;
constexpr uint32_t ACCEPTANCE_SESSION_ID = 0x12345678u;
constexpr uint32_t ACCEPTANCE_TIMEOUT_SECONDS = 7200u;
const std::string KDF_DOMAIN = "nexus-mining-chacha20-v1";

void print_result(const char* name, bool passed)
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

std::array<uint8_t, 4> extract_prevblock_suffix(const ::LLP::CBlock& block)
{
    std::array<uint8_t, 4> suffix{};
    const auto prev_hash = block.hashPrevBlock.GetBytes();
    if (prev_hash.size() >= suffix.size()) {
        std::copy(prev_hash.end() - static_cast<std::ptrdiff_t>(suffix.size()),
                  prev_hash.end(),
                  suffix.begin());
    }
    return suffix;
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
    constexpr std::size_t TRITIUM_HEIGHT_FIELD_SIZE = 4u;
    if (block_payload.size() < TRITIUM_HEIGHT_OFFSET + TRITIUM_HEIGHT_FIELD_SIZE) {
        return 0;
    }

    // Tritium submit payload serializes nHeight as a big-endian uint32 at bytes
    // [200..203], matching the existing block serialization contract.
    return (static_cast<uint32_t>(block_payload[TRITIUM_HEIGHT_OFFSET]) << 24) |
           (static_cast<uint32_t>(block_payload[TRITIUM_HEIGHT_OFFSET + 1]) << 16) |
           (static_cast<uint32_t>(block_payload[TRITIUM_HEIGHT_OFFSET + 2]) << 8) |
            static_cast<uint32_t>(block_payload[TRITIUM_HEIGHT_OFFSET + 3]);
}

struct AcceptedSubmissionTracker
{
    struct Outcome
    {
        uint32_t height{0};
        uint32_t channel{0};
        uint64_t nonce{0};
        std::vector<uint8_t> prev_hash;
        bool used_snapshot{false};
        bool used_fallback{false};
    };

    bool valid{false};
    uint32_t height{0};
    uint32_t channel{0};
    uint64_t nonce{0};
    std::vector<uint8_t> prev_hash;

    void snapshot_from(const MiningTemplateInterface::MiningTemplate& tmpl, uint64_t submitted_nonce)
    {
        valid = true;
        height = tmpl.block.nHeight;
        channel = tmpl.block.nChannel;
        nonce = submitted_nonce;
        prev_hash = tmpl.block.hashPrevBlock.GetBytes();
    }

    Outcome accept(const MiningTemplateInterface::MiningTemplate* current_template)
    {
        if (valid) {
            valid = false;
            return Outcome{height, channel, nonce, prev_hash, true, false};
        }

        Outcome outcome{};
        outcome.used_fallback = true;
        if (current_template) {
            outcome.height = current_template->block.nHeight;
            outcome.channel = current_template->block.nChannel;
            outcome.prev_hash = current_template->block.hashPrevBlock.GetBytes();
        }
        return outcome;
    }
};

struct HarnessOptions
{
    bool full_validation{false};
};

struct HarnessArtifacts
{
    std::vector<std::string> phases;
    std::string reward_diagnostics;
    std::string pre_submit_diagnostics;
    std::string post_accept_diagnostics;
    std::string session_event_journal;
    std::string accept_response;
    std::array<uint8_t, 4> prevblock_suffix{};
    uint32_t session_id{0};
    uint16_t keepalive_hours{0};
    uint32_t unified_height{0};
    uint32_t channel_height{0};
    uint32_t authoritative_submit_height{0};
    uint32_t template_height{0};
    uint32_t built_block_height{0};
    uint32_t submitted_payload_height{0};
    uint32_t template_channel{0};
    uint64_t submitted_nonce{0};
    std::size_t plaintext_submit_size{0};
    std::size_t encrypted_submit_size{0};
    bool session_valid_before_submit{false};
    bool submit_validation_matches_template{false};
    bool decrypt_matches_submit{false};
    bool accept_used_snapshot{false};
    bool fallback_after_snapshot_consumption{false};
    bool post_accept_session_valid{false};
    bool binding_consistency_valid{false};
    bool full_validation_event_journal_complete{false};
};

struct HarnessResult
{
    bool ok{false};
    std::string failure;
    HarnessArtifacts artifacts;
};

HarnessResult run_first_block_acceptance_harness(const HarnessOptions& options)
{
    HarnessResult result;

    auto fail = [&](const std::string& message) {
        result.ok = false;
        result.failure = message;
        return result;
    };

    auto session_manager = std::make_shared<SessionManager>(DEFAULT_KEEPALIVE_INTERVAL_HOURS, nullptr);
    NodeSessionContext context(session_manager);
    context.set_protocol_lane(ProtocolLane::STATELESS);
    context.set_connection_metadata("127.0.0.1:4000", "127.0.0.1:9323", true);

    result.artifacts.phases.push_back("auth");
    const auto falcon_pubkey = make_repeated_bytes(32, 0x20);
    context.set_falcon_identity(falcon_pubkey, "acceptance-harness-key", true);
    auto auth_info = context.get_session_info();
    if (!auth_info.falcon_authenticated || auth_info.falcon_key_id.get() != "acceptance-harness-key") {
        return fail("auth phase did not persist falcon identity");
    }

    result.artifacts.phases.push_back("session start");
    const auto genesis = make_repeated_bytes(32, 0x10);
    const auto session_start_packet =
        make_session_start_packet(ACCEPTANCE_SESSION_ID, ACCEPTANCE_TIMEOUT_SECONDS, genesis);
    uint32_t parsed_session_id = 0;
    uint32_t parsed_timeout = 0;
    std::vector<uint8_t> parsed_genesis;
    if (!NodeSessionContext::parse_session_start(session_start_packet,
                                                 parsed_session_id,
                                                 parsed_timeout,
                                                 parsed_genesis)) {
        return fail("session start packet did not parse");
    }
    if (parsed_session_id != ACCEPTANCE_SESSION_ID ||
        parsed_timeout != ACCEPTANCE_TIMEOUT_SECONDS ||
        parsed_genesis != genesis) {
        return fail("session start payload did not preserve session metadata");
    }

    context.start_session(SessionId(parsed_session_id), {}, parsed_genesis);
    context.set_state(SessionManager::SessionState::ACTIVE);
    const uint16_t keepalive_hours = static_cast<uint16_t>(
        std::max<uint32_t>(1u,
                           (parsed_timeout / NodeSessionContext::get_keepalive_safety_divisor()) / SECONDS_PER_HOUR));
    context.set_keepalive_interval(keepalive_hours);

    const auto chacha_key = derive_session_key(parsed_genesis);
    const auto chacha_fingerprint = format_hex_prefix(chacha_key, 8);
    context.set_chacha20_session_key(chacha_key, chacha_fingerprint, true);
    result.artifacts.session_id = parsed_session_id;
    result.artifacts.keepalive_hours = keepalive_hours;

    result.artifacts.phases.push_back("reward bind");
    const auto reward_hash = make_repeated_bytes(32, 0x44);
    context.set_reward_binding("reward-address", reward_hash, true, "acceptance harness");
    result.artifacts.reward_diagnostics = context.build_miner_session_diagnostics();
    if (result.artifacts.reward_diagnostics.find("reward-address") == std::string::npos) {
        return fail("reward bind diagnostics did not include reward address");
    }

    result.artifacts.phases.push_back("channel set");
    context.set_channel_state(ACCEPTANCE_CHANNEL, false, true);
    auto channel_info = context.get_session_info();
    // Channel readiness is the authoritative gate for GET_BLOCK after reward binding.
    // Submit readiness is established separately once the session is fully mining-ready.
    if (channel_info.channel != ACCEPTANCE_CHANNEL ||
        !channel_info.ready_for_get_block) {
        return fail("channel set phase did not establish ready_for_get_block state");
    }

    result.artifacts.phases.push_back("miner ready");
    context.set_channel_state(ACCEPTANCE_CHANNEL, true, true);
    auto session_status = context.build_session_status_packet(false, false, true, false);
    if (!session_status || session_status->empty()) {
        return fail("miner ready phase did not build session status packet");
    }

    result.artifacts.phases.push_back("get/push block");
    MiningTemplateInterface template_interface(static_cast<uint8_t>(ACCEPTANCE_CHANNEL), parsed_session_id);
    auto template_payload = make_template_payload(
        ACCEPTANCE_UNIFIED_HEIGHT,
        ACCEPTANCE_CHANNEL_HEIGHT,
        DEFAULT_DIFFICULTY,
        DEFAULT_BLOCK_VERSION,
        ACCEPTANCE_CHANNEL,
        ACCEPTANCE_TEMPLATE_HEIGHT,
        DEFAULT_DIFFICULTY,
        0,
        make_prev_hash_pattern(0x30));
    auto decoded = StatelessBlockUtility::decode_template(
        template_interface, template_payload, ACCEPTANCE_CHANNEL, nullptr);
    if (!decoded.valid || !decoded.metadata_consistent || !decoded.channel_consistent) {
        return fail("template decode/validation failed");
    }

    const auto* current_template = template_interface.get_current_template();
    if (!current_template || !template_interface.has_valid_template()) {
        return fail("validated template was not retained");
    }

    result.artifacts.unified_height = decoded.unified_height;
    result.artifacts.channel_height = decoded.channel_height;
    result.artifacts.authoritative_submit_height = current_template->block.nHeight;
    result.artifacts.template_height = current_template->block.nHeight;
    result.artifacts.template_channel = current_template->block.nChannel;
    result.artifacts.prevblock_suffix = extract_prevblock_suffix(current_template->block);
    context.set_prevblock_suffix(result.artifacts.prevblock_suffix);

    auto keepalive_packet = context.build_keepalive_packet();
    if (!keepalive_packet || keepalive_packet->empty()) {
        return fail("get/push block phase did not build keepalive packet");
    }

    result.artifacts.phases.push_back("solve/inject valid block");
    ::LLP::CBlock solved_block = current_template->block;
    solved_block.nNonce = ACCEPTANCE_NONCE;
    result.artifacts.built_block_height = solved_block.nHeight;
    result.artifacts.submitted_nonce = ACCEPTANCE_NONCE;

    AcceptedSubmissionTracker accepted_tracker;
    accepted_tracker.snapshot_from(*current_template, ACCEPTANCE_NONCE);

    SubmitContext submit_context;
    submit_context.session_id = SessionId(parsed_session_id);
    submit_context.session_epoch = SessionEpoch(context.get_session_epoch());
    submit_context.template_height = current_template->block.nHeight;
    submit_context.chain_height = decoded.unified_height;

    result.artifacts.phases.push_back("submit");
    context.record_session_event(SessionManager::SessionEventKind::SUBMIT_SENT,
                                 "acceptance harness submit prepared");
    auto submit_result = StatelessBlockUtility::encode_submit(
        template_interface,
        solved_block,
        {},
        nullptr,
        ProtocolLane::STATELESS,
        HeightTracker::Snapshot{},
        nullptr,
        submit_context);
    if (!submit_result.valid || !submit_result.wire_bytes || submit_result.wire_bytes->empty()) {
        return fail("submit path did not produce wire bytes");
    }

    auto plaintext_submit = strip_wire_header(*submit_result.wire_bytes, ProtocolLane::STATELESS);
    if (plaintext_submit.empty()) {
        return fail("submit path produced an empty plaintext payload");
    }
    result.artifacts.plaintext_submit_size = plaintext_submit.size();
    result.artifacts.submitted_payload_height = extract_serialized_block_height(plaintext_submit);

    Block_data solved_snapshot(solved_block);
    auto expected_block_submission = template_interface.prepare_block_submission_from_solved(
        solved_snapshot, {});
    result.artifacts.submit_validation_matches_template =
        (plaintext_submit == expected_block_submission);
    if (!result.artifacts.submit_validation_matches_template) {
        return fail("submit plaintext did not match solved-snapshot block serialization");
    }

    std::string validate_reason;
    result.artifacts.session_valid_before_submit = context.validate_miner_session(&validate_reason);
    if (!result.artifacts.session_valid_before_submit || validate_reason != "PASS") {
        return fail("miner session was not valid before submit");
    }
    result.artifacts.pre_submit_diagnostics = context.build_miner_session_diagnostics();

    auto payload_info = StatelessBlockUtility::compute_submit_payload_info(
        solved_block.nChannel, plaintext_submit.size(), 0);
    ChaCha20Wrapper wrapper;
    auto encrypted_submit = wrapper.encrypt_submit_block_payload(
        plaintext_submit, chacha_key, payload_info);
    if (!encrypted_submit.success || encrypted_submit.data.empty()) {
        return fail("ChaCha20 submit encryption failed");
    }

    auto final_submit_packet = PacketBuilder::build(
        ProtocolLane::STATELESS, nexusminer::LLP::SUBMIT_BLOCK, encrypted_submit.data);
    if (!final_submit_packet || final_submit_packet->empty()) {
        return fail("final submit packet build failed");
    }

    auto encrypted_payload = strip_wire_header(*final_submit_packet, ProtocolLane::STATELESS);
    if (encrypted_payload != encrypted_submit.data || encrypted_payload.size() <= 12) {
        return fail("transmitted encrypted payload did not survive framing");
    }
    result.artifacts.encrypted_submit_size = encrypted_payload.size();

    std::vector<uint8_t> nonce(encrypted_payload.begin(), encrypted_payload.begin() + 12);
    std::vector<uint8_t> ciphertext_and_tag(encrypted_payload.begin() + 12, encrypted_payload.end());
    auto decrypted_submit = wrapper.decrypt(ciphertext_and_tag, chacha_key, nonce, {});
    result.artifacts.decrypt_matches_submit =
        decrypted_submit.success && decrypted_submit.data == plaintext_submit;
    if (!result.artifacts.decrypt_matches_submit) {
        return fail("node-side submit decrypt/verify path failed");
    }

    result.artifacts.phases.push_back("accept");
    auto accepted = accepted_tracker.accept(current_template);
    result.artifacts.accept_used_snapshot =
        accepted.used_snapshot &&
        !accepted.used_fallback &&
        accepted.height == current_template->block.nHeight &&
        accepted.channel == current_template->block.nChannel &&
        accepted.nonce == ACCEPTANCE_NONCE &&
        accepted.prev_hash == current_template->block.hashPrevBlock.GetBytes();
    if (!result.artifacts.accept_used_snapshot) {
        return fail("accept path did not consume the submitted snapshot");
    }
    result.artifacts.accept_response = "ACCEPT";
    context.record_session_event(SessionManager::SessionEventKind::SUBMIT_ACCEPTED,
                                 "acceptance harness accept");

    if (options.full_validation) {
        auto replacement_payload = make_template_payload(
            ACCEPTANCE_UNIFIED_HEIGHT + 1,
            ACCEPTANCE_CHANNEL_HEIGHT + 1,
            DEFAULT_DIFFICULTY,
            DEFAULT_BLOCK_VERSION,
            ACCEPTANCE_CHANNEL,
            ACCEPTANCE_TEMPLATE_HEIGHT + 1,
            DEFAULT_DIFFICULTY,
            0,
            make_prev_hash_pattern(0x70));
        auto replacement_decoded = StatelessBlockUtility::decode_template(
            template_interface, replacement_payload, ACCEPTANCE_CHANNEL, nullptr);
        if (!replacement_decoded.valid) {
            return fail("full validation mode failed to load a replacement template");
        }

        const auto* replacement_template = template_interface.get_current_template();
        if (!replacement_template) {
            return fail("replacement template was not retained");
        }

        context.set_prevblock_suffix(extract_prevblock_suffix(replacement_template->block));
        auto fallback_accept = accepted_tracker.accept(replacement_template);
        result.artifacts.fallback_after_snapshot_consumption =
            fallback_accept.used_fallback &&
            !fallback_accept.used_snapshot &&
            fallback_accept.height == replacement_template->block.nHeight &&
            fallback_accept.channel == replacement_template->block.nChannel;
        if (!result.artifacts.fallback_after_snapshot_consumption) {
            return fail("snapshot consumption did not fall back to current template on second accept");
        }
    }

    context.mark_activity();
    std::string post_accept_reason;
    result.artifacts.post_accept_diagnostics = context.build_miner_session_diagnostics();
    result.artifacts.post_accept_session_valid = context.validate_miner_session(&post_accept_reason);
    if (!result.artifacts.post_accept_session_valid || post_accept_reason != "PASS") {
        return fail("post-accept session validation failed");
    }

    const auto binding = context.get_session_binding();
    result.artifacts.binding_consistency_valid =
        binding.has_session() &&
        binding.authenticated &&
        binding.has_crypto_context() &&
        binding.reward_bound &&
        binding.ready_for_submit &&
        binding.ready_for_get_block &&
        binding.identity.is_valid() &&
        binding.identity_matches_session();
    result.artifacts.session_event_journal = context.build_session_event_journal();
    result.artifacts.full_validation_event_journal_complete =
        result.artifacts.session_event_journal.find("auth_success") != std::string::npos &&
        result.artifacts.session_event_journal.find("session_start") != std::string::npos &&
        result.artifacts.session_event_journal.find("reward_bind_result") != std::string::npos &&
        result.artifacts.session_event_journal.find("submit_sent") != std::string::npos &&
        result.artifacts.session_event_journal.find("submit_accepted") != std::string::npos;

    result.ok = true;
    return result;
}

void test_first_block_acceptance_fast_mode()
{
    std::cout << "\nTest 1: deterministic first-block acceptance harness (fast mode)\n";

    const auto result = run_first_block_acceptance_harness(HarnessOptions{false});
    const auto expected_phases = std::vector<std::string>{
        "auth",
        "session start",
        "reward bind",
        "channel set",
        "miner ready",
        "get/push block",
        "solve/inject valid block",
        "submit",
        "accept"
    };

    print_result("Fast harness: completes required phase sequence",
                 result.ok && result.artifacts.phases == expected_phases);
    print_result("Fast harness: captures reward/session diagnostics",
                 result.ok &&
                 !result.artifacts.reward_diagnostics.empty() &&
                 !result.artifacts.post_accept_diagnostics.empty());
    print_result("Fast harness: captures template anchor and derived keepalive",
                 result.ok &&
                  result.artifacts.session_id == ACCEPTANCE_SESSION_ID &&
                  result.artifacts.keepalive_hours == 1 &&
                  result.artifacts.authoritative_submit_height == ACCEPTANCE_TEMPLATE_HEIGHT &&
                  result.artifacts.template_height == ACCEPTANCE_TEMPLATE_HEIGHT &&
                  result.artifacts.template_channel == ACCEPTANCE_CHANNEL &&
                  result.artifacts.unified_height == ACCEPTANCE_UNIFIED_HEIGHT &&
                  result.artifacts.channel_height == ACCEPTANCE_CHANNEL_HEIGHT &&
                  result.artifacts.prevblock_suffix != std::array<uint8_t, 4>{});
    print_result("Fast harness: authoritative submit height matches template, built block, and payload",
                 result.ok &&
                 result.artifacts.authoritative_submit_height == ACCEPTANCE_TEMPLATE_HEIGHT &&
                 result.artifacts.template_height == result.artifacts.authoritative_submit_height &&
                 result.artifacts.built_block_height == result.artifacts.authoritative_submit_height &&
                 result.artifacts.submitted_payload_height == result.artifacts.authoritative_submit_height);
    print_result("Fast harness: submit decrypt + validation + accept path succeed",
                 result.ok &&
                  result.artifacts.session_valid_before_submit &&
                  result.artifacts.submit_validation_matches_template &&
                  result.artifacts.decrypt_matches_submit &&
                  result.artifacts.accept_used_snapshot &&
                  result.artifacts.binding_consistency_valid &&
                  result.artifacts.accept_response == "ACCEPT" &&
                  result.artifacts.post_accept_session_valid);

    if (!result.ok) {
        std::cout << "    Failure: " << result.failure << '\n';
    }
}

void test_first_block_acceptance_full_validation_mode()
{
    std::cout << "\nTest 2: deterministic first-block acceptance harness (full validation mode)\n";

    const auto result = run_first_block_acceptance_harness(HarnessOptions{true});
    const auto suffix_hex = format_hex_prefix(result.artifacts.prevblock_suffix, 4);

    print_result("Full harness: retains pre-submit diagnostics with template suffix",
                 result.ok &&
                 result.artifacts.pre_submit_diagnostics.find("consistency: PASS") != std::string::npos &&
                 result.artifacts.pre_submit_diagnostics.find(suffix_hex) != std::string::npos);
    print_result("Full harness: produces deterministic submit payload sizes",
                 result.ok &&
                 result.artifacts.submitted_nonce == ACCEPTANCE_NONCE &&
                 result.artifacts.plaintext_submit_size == StatelessBlockUtility::BLOCK_BODY_SIZE &&
                 result.artifacts.encrypted_submit_size == 12u + StatelessBlockUtility::BLOCK_BODY_SIZE + 16u);
    print_result("Full harness: consumes accepted snapshot before fallback path",
                 result.ok &&
                 result.artifacts.accept_used_snapshot &&
                 result.artifacts.fallback_after_snapshot_consumption);
    print_result("Full harness: preserves session consistency after accept",
                 result.ok &&
                  result.artifacts.post_accept_session_valid &&
                  result.artifacts.post_accept_diagnostics.find("reward-address") != std::string::npos);
    print_result("Full harness: captures event journal for auth/reward/submit lifecycle",
                 result.ok &&
                  result.artifacts.binding_consistency_valid &&
                  result.artifacts.full_validation_event_journal_complete &&
                  result.artifacts.session_event_journal.find("SESSION EVENT JOURNAL") != std::string::npos);

    if (!result.ok) {
        std::cout << "    Failure: " << result.failure << '\n';
    }
}

} // namespace

int main()
{
    std::cout << "\n========================================\n";
    std::cout << "  First-Block Acceptance Harness Tests\n";
    std::cout << "========================================\n";

    test_first_block_acceptance_fast_mode();
    test_first_block_acceptance_full_validation_mode();

    std::cout << "\n========================================\n";
    std::cout << "Test Summary\n";
    std::cout << "========================================\n";
    std::cout << "Tests run:    " << tests_run << "\n";
    std::cout << "Tests passed: " << tests_passed << "\n";
    std::cout << "Tests failed: " << tests_failed << "\n";
    std::cout << "========================================\n";

    return (tests_failed == 0) ? 0 : 1;
}
