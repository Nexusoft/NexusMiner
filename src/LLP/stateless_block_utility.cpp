/**
 * @file stateless_block_utility.cpp
 * @brief Implementation of StatelessBlockUtility
 *
 * Compiled as part of the `protocol` STATIC library (listed in
 * src/protocol/CMakeLists.txt) so that it can include both LLP headers
 * and protocol headers (falcon_wrapper.hpp, mining_template_interface.hpp,
 * packet_builder.hpp).
 *
 * All encode/decode logic is delegated to MiningTemplateInterface:
 *   decode_template() -> MiningTemplateInterface::read_stateless_payload()
 *   encode_submit()   -> MiningTemplateInterface::prepare_block_submission_from_solved()
 * This file provides only the pre-check gate and Falcon-signing layer.
 */

#include "include/stateless_block_utility.hpp"

#include "miner_opcodes.hpp"
#include "protocol/falcon_wrapper.hpp"
#include "protocol/packet_builder.hpp"
#include "worker/worker.hpp"
#include "spdlog/spdlog.h"

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace nexusminer {
namespace protocol {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void append_u64_le(std::vector<uint8_t>& dest, uint64_t value) {
    for (int i = 0; i < 8; ++i)
        dest.push_back(static_cast<uint8_t>((value >> (i * 8)) & 0xFF));
}

static void append_u16_le(std::vector<uint8_t>& dest, uint16_t value) {
    dest.push_back(static_cast<uint8_t>(value & 0xFF));
    dest.push_back(static_cast<uint8_t>((value >> 8) & 0xFF));
}

// ---------------------------------------------------------------------------
// decode_template
// ---------------------------------------------------------------------------

DecodedTemplate StatelessBlockUtility::decode_template(
    MiningTemplateInterface& tmpl_iface,
    const network::Payload& raw_payload,
    uint32_t mining_channel,
    std::shared_ptr<spdlog::logger> logger,
    bool auto_feed)
{
    DecodedTemplate result;

    // ── Delegate to MiningTemplateInterface::read_stateless_payload() ─────────
    // MTI handles the size gate, metadata prefix extraction, and block body
    // decode via llp_utils::deserialize_block_header().
    auto vresult = tmpl_iface.read_stateless_payload(raw_payload, "stateless", auto_feed);
    if (!vresult.is_valid) {
        result.error_message = vresult.error_message;
        if (logger)
            logger->error("[StatelessBlockUtility::decode_template] {}",
                          result.error_message);
        return result;
    }

    // ── Extract decoded data from MTI's current template ─────────────────────
    const auto* tmpl = tmpl_iface.get_current_template();
    if (!tmpl) {
        result.error_message = "read_stateless_payload succeeded but no current template";
        if (logger)
            logger->error("[StatelessBlockUtility::decode_template] {}",
                          result.error_message);
        return result;
    }

    // Canonical mining state (from 216-byte block body)
    result.block = tmpl->block;

    // Diagnostic metadata (from 12-byte prefix, stored by read_stateless_payload)
    result.unified_height   = tmpl->nUnifiedHeightMeta;
    result.channel_height   = tmpl->nChannelHeightMeta;
    result.difficulty_nbits = tmpl->block.nBits; // echoed in prefix; use block value

    if (logger)
        logger->debug("[StatelessBlockUtility::decode_template] "
                      "metadata: unified={} channel={} nBits=0x{:08x}",
                      result.unified_height, result.channel_height,
                      result.difficulty_nbits);

    // ── Derived validation flags ─────────────────────────────────────────────
    if (result.unified_height > 0 && result.block.nHeight > 0) {
        int32_t diff = static_cast<int32_t>(result.block.nHeight) -
                       static_cast<int32_t>(result.unified_height + 1);
        result.metadata_consistent = (diff >= -1 && diff <= 1);
    }

    result.channel_consistent = (result.block.nChannel == mining_channel);

    if (logger) {
        if (!result.metadata_consistent)
            logger->warn("[StatelessBlockUtility::decode_template] "
                         "metadata_consistent=false: block.nHeight={} vs "
                         "unified_height+1={}",
                         result.block.nHeight, result.unified_height + 1);
        if (!result.channel_consistent)
            logger->warn("[StatelessBlockUtility::decode_template] "
                         "channel_consistent=false: block.nChannel={} vs "
                         "mining_channel={}",
                         result.block.nChannel, mining_channel);
    }

    result.valid = true;
    return result;
}

// ---------------------------------------------------------------------------
// encode_submit
// ---------------------------------------------------------------------------

SubmitResult StatelessBlockUtility::encode_submit(
    MiningTemplateInterface& tmpl_iface,
    const ::LLP::CBlock& solved_block,
    const std::vector<uint8_t>& vOffsets,
    FalconSignatureWrapper* falcon,
    ProtocolLane lane,
    const HeightTracker::Snapshot& ht,
    std::shared_ptr<spdlog::logger> logger,
    const SubmitContext& submit_context)
{
    SubmitResult result;

    // ── Pre-check 1: Nonce sanity ────────────────────────────────────────────
    if (solved_block.nNonce == 0) {
        result.rejection_reason = "nNonce is zero -- block not yet solved";
        if (logger)
            logger->error("[StatelessBlockUtility::encode_submit] {}",
                          result.rejection_reason);
        return result;
    }

    // ── Pre-check 2: Channel plausibility ───────────────────────────────────
    if (solved_block.nChannel != 1 && solved_block.nChannel != 2) {
        result.rejection_reason =
            "invalid nChannel " + std::to_string(solved_block.nChannel) +
            " (expected 1=Prime or 2=Hash)";
        if (logger)
            logger->error("[StatelessBlockUtility::encode_submit] {}",
                          result.rejection_reason);
        return result;
    }

    // ── Pre-check 3: Height plausibility ────────────────────────────────────
    if (solved_block.nHeight == 0) {
        result.rejection_reason =
            "nHeight is zero -- template not yet received";
        if (logger)
            logger->error("[StatelessBlockUtility::encode_submit] {}",
                          result.rejection_reason);
        return result;
    }

    // ── Pre-check 4: Authoritative submit-height consistency ─────────────────
    if (!submit_context.matches_submit_height(solved_block.nHeight)) {
        result.rejection_reason =
            "submit height mismatch -- template_height=" +
            std::to_string(submit_context.template_height) +
            " submit_height=" + std::to_string(solved_block.nHeight) +
            " chain_height=" + std::to_string(submit_context.chain_height) +
            " session_epoch=" + std::to_string(submit_context.session_epoch.get());
        if (logger)
            logger->error("[StatelessBlockUtility::encode_submit] {}",
                          result.rejection_reason);
        return result;
    }

    // ── Pre-check 5: Staleness — removed (is_template_stale structurally false) ──

    // ── Pre-check 6: Tip-moved (informational) ────────────────────────────────
    if (ht.is_tip_moved()) {
        if (logger)
            logger->warn("[StatelessBlockUtility::encode_submit] "
                         "unified tip moved (unified_height={} > "
                         "template_unified_height={}) -- submitting anyway",
                         ht.unified_height, ht.template_unified_height);
    }

    // ── Pre-check 7: Delegate serialization to MiningTemplateInterface ────────
    // Re-wrap the solved block in Block_data so the submit path reuses the same
    // worker-snapshot serializer that worker_manager uses when it first produces
    // block_data.  This keeps encode_submit() aligned with the authoritative
    // flow instead of reviving the older "template + merkle_root + nonce" path.
    Block_data solved_snapshot(solved_block);
    auto block_bytes = vOffsets.empty()
        ? tmpl_iface.prepare_block_submission_from_solved(solved_snapshot)
        : tmpl_iface.prepare_block_submission_from_solved(solved_snapshot, vOffsets);

    if (block_bytes.empty()) {
        result.rejection_reason =
            "MiningTemplateInterface::prepare_block_submission returned empty -- "
            "no valid template or block validation failed";
        if (logger)
            logger->error("[StatelessBlockUtility::encode_submit] {}",
                          result.rejection_reason);
        return result;
    }

    // ── Pre-check 8: Disposable Falcon sign (optional) ───────────────────────
    std::vector<uint8_t> plaintext;

    if (falcon != nullptr) {
        // Timestamp for replay-protection (8 bytes LE)
        uint64_t ts = static_cast<uint64_t>(
            std::chrono::system_clock::to_time_t(
                std::chrono::system_clock::now()));

        // message_to_sign = block_bytes || timestamp
        std::vector<uint8_t> msg_to_sign;
        msg_to_sign.reserve(block_bytes.size() + 8);
        msg_to_sign.insert(msg_to_sign.end(),
                           block_bytes.begin(), block_bytes.end());
        append_u64_le(msg_to_sign, ts);

        auto sig_result = falcon->sign_payload(
            msg_to_sign, FalconSignatureWrapper::SignatureType::BLOCK);

        if (!sig_result.success) {
            result.rejection_reason =
                "Falcon signing failed: " + sig_result.error_message;
            if (logger)
                logger->error("[StatelessBlockUtility::encode_submit] {}",
                              result.rejection_reason);
            return result;
        }

        // plaintext = block_bytes || timestamp(8) || sig_len(2) || sig
        uint16_t sig_len = static_cast<uint16_t>(sig_result.signature.size());
        plaintext.reserve(block_bytes.size() + 8 + 2 + sig_len);
        plaintext.insert(plaintext.end(), block_bytes.begin(), block_bytes.end());
        append_u64_le(plaintext, ts);
        append_u16_le(plaintext, sig_len);
        plaintext.insert(plaintext.end(),
                         sig_result.signature.begin(),
                         sig_result.signature.end());

        if (logger)
            logger->debug("[StatelessBlockUtility::encode_submit] "
                          "signed: block({})+ts(8)+siglen(2)+sig({}) = {} bytes",
                          block_bytes.size(), sig_len, plaintext.size());
    } else {
        // No signing -- payload is just the serialized block bytes (+ any vOffsets
        // already appended by prepare_block_submission for Prime channel)
        plaintext = std::move(block_bytes);
        if (logger)
            logger->debug("[StatelessBlockUtility::encode_submit] "
                          "unsigned submit: {} bytes (no Falcon signature)",
                          plaintext.size());
    }

    // ── Pre-check 9: Wire encode ─────────────────────────────────────────────
    auto wire = PacketBuilder::build(lane, LLP::SUBMIT_BLOCK, plaintext);
    if (!wire || wire->empty()) {
        result.rejection_reason = "PacketBuilder::build returned empty result";
        if (logger)
            logger->error("[StatelessBlockUtility::encode_submit] {}",
                          result.rejection_reason);
        return result;
    }

    result.valid      = true;
    result.wire_bytes = wire;
    return result;
}

// ─── Channel-aware payload sizing helper ─────────────────────────────────────
ChaCha20Wrapper::SubmitBlockPayloadInfo StatelessBlockUtility::compute_submit_payload_info(
    uint32_t channel,
    size_t   block_data_size,
    size_t   signature_size)
{
    ChaCha20Wrapper::SubmitBlockPayloadInfo info;
    info.channel            = channel;
    info.base_block_size    = BLOCK_BODY_SIZE;  // 216 for Tritium
    // For Prime, offset bytes = total block_data_size - base 216-byte block body.
    // For Hash, block_data_size should equal 216, so offset_bytes_count = 0.
    info.offset_bytes_count = (block_data_size > BLOCK_BODY_SIZE)
                                ? (block_data_size - BLOCK_BODY_SIZE)
                                : 0;
    info.timestamp_size     = 8;
    info.sig_len_field_size = 2;
    info.signature_size     = signature_size;
    return info;
}

} // namespace protocol
} // namespace nexusminer
