#ifndef NEXUSMINER_LLP_STATELESS_BLOCK_UTILITY_HPP
#define NEXUSMINER_LLP_STATELESS_BLOCK_UTILITY_HPP

/**
 * @file stateless_block_utility.hpp
 * @brief Pre-check gate and Falcon-signing adapter for the stateless mining lane
 *
 * StatelessBlockUtility is a thin adapter over MiningTemplateInterface that
 * provides:
 *  1. A canonical pre-check gate before SUBMIT_BLOCK (nonce, channel, height
 *     sanity; staleness and tip-moved warnings).
 *  2. Disposable Falcon signing of the serialized block payload.
 *  3. Wire-frame encoding via PacketBuilder for STATELESS vs LEGACY lanes.
 *
 * Decode (Node -> Miner)
 *   decode_template() handles the 228-byte STATELESS_GET_BLOCK payload by
 *   delegating the 12-byte metadata prefix extraction and the 216-byte block
 *   body decode to MiningTemplateInterface::read_stateless_payload().  No
 *   encode/decode logic is duplicated here.
 *
 * Encode (Miner -> Node)
 *   encode_submit() delegates block serialization (including Prime-channel
 *   vOffsets) to MiningTemplateInterface::prepare_block_submission_from_solved(),
 *   then optionally signs with Disposable Falcon and frames with PacketBuilder.
 *
 * Namespace separation (canonical vs diagnostic):
 *   Canonical inputs  -- block.nHeight, block.nBits, block.nChannel,
 *                        block.hashPrevBlock, block.nNonce.  These drive
 *                        mining and ProofHash; they are never overwritten.
 *   Diagnostic inputs -- unified_height / channel_height from the 12-byte
 *                        metadata prefix and HeightTracker snapshot fields.
 *                        For Colin / staleness detection only.
 */

#include "LLP/block.hpp"
#include "network/types.hpp"
#include "protocol/height_tracker.hpp"
#include "protocol/mining_template_interface.hpp"
#include "protocol/submit_context.hpp"
#include "protocol_lane.hpp"
#include "protocol/chacha20_wrapper.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace nexusminer { namespace protocol { class FalconSignatureWrapper; } }
namespace spdlog { class logger; }

namespace nexusminer {
namespace protocol {

/**
 * @brief Result of a decode_template() call
 *
 * Canonical fields come exclusively from the 216-byte Tritium block body
 * (after MiningTemplateInterface::read_stateless_payload() processes them).
 * Metadata fields (unified_height, channel_height, difficulty_nbits) come
 * from the 12-byte big-endian prefix and are DIAGNOSTIC/telemetry only.
 */
struct DecodedTemplate {
    // -- Diagnostic metadata (12-byte prefix, big-endian) --------------------
    /** Unified blockchain height -- diagnostic; feed only to OnTemplateMetadata() */
    uint32_t unified_height{0};
    /** Channel-specific height -- diagnostic; feed only to OnTemplateMetadata() */
    uint32_t channel_height{0};
    /** Mining difficulty (compact nBits) -- diagnostic */
    uint32_t difficulty_nbits{0};

    // -- Canonical mining state (from the 216-byte Tritium block body) --------
    /**
     * Decoded block header -- the canonical source of truth for all mining
     * decisions: nVersion, hashPrevBlock, hashMerkleRoot, nChannel, nHeight,
     * nBits, nNonce.  Never overwrite these from the metadata prefix.
     *
     * This is a copy of MiningTemplateInterface::get_current_template()->block
     * after a successful read_stateless_payload() call.
     */
    ::LLP::CBlock block;

    // -- Derived validation flags ---------------------------------------------
    /**
     * True when block.nHeight ~= unified_height + 1 (within +-1 tolerance for
     * the tip-advance race window).  A mismatch is a warning, not a hard reject.
     */
    bool metadata_consistent{false};
    /** True when block.nChannel equals the declared mining channel. */
    bool channel_consistent{false};

    /** True when the 228-byte payload was successfully decoded. */
    bool valid{false};
    /** Human-readable reason for failure (empty when valid == true). */
    std::string error_message;
};

/**
 * @brief Result of an encode_submit() call
 */
struct SubmitResult {
    /** False -- do NOT send; block failed a pre-check. */
    bool valid{false};
    /** Human-readable rejection reason (logged by caller). */
    std::string rejection_reason;
    /**
     * Wire-encoded bytes (PacketBuilder-framed plaintext payload).
     * Non-null only when valid == true.
     *
     * For the STATELESS lane the first two bytes are 0xD0, 0x01 (opcode
     * 0xD001).  For the LEGACY lane the first byte is 0x01.
     *
     * Note: ChaCha20 encryption is NOT applied here -- the caller encrypts
     * and re-frames before sending to the node.
     */
    network::Shared_payload wire_bytes;
};

/**
 * @brief Pre-check gate and Falcon-signing adapter for the stateless mining lane
 *
 * All methods are static -- no instance state.  Each method receives a
 * MiningTemplateInterface reference so that encode/decode logic is
 * delegated to the canonical MTI implementation (no duplication).
 */
class StatelessBlockUtility {
public:
    // --- Wire-format constants -----------------------------------------------
    /** Total payload size for a STATELESS_GET_BLOCK (12 metadata + 216 block) */
    static constexpr size_t TEMPLATE_PAYLOAD_SIZE = 228;
    /** Size of the metadata prefix in a STATELESS_GET_BLOCK payload */
    static constexpr size_t METADATA_PREFIX_SIZE  = 12;
    /** Size of the Tritium block body */
    static constexpr size_t BLOCK_BODY_SIZE        = 216;

    // =========================================================================
    // Template decode  (Node -> Miner)
    // =========================================================================

    /**
     * @brief Decode a 228-byte STATELESS_GET_BLOCK payload via MiningTemplateInterface.
     *
     * Delegates to tmpl_iface.read_stateless_payload() for the actual block
     * body decode and validation -- no encode/decode logic is duplicated here.
     *
     * The metadata prefix (bytes 0-11, big-endian) is stored as diagnostic
     * fields in the returned DecodedTemplate.  Route them only to
     * HeightTracker::OnTemplateMetadata(), never to OnGetRound().
     *
     * @param tmpl_iface    MiningTemplateInterface that owns the current template.
     *                      On success, its internal template state is updated.
     * @param raw_payload   Full 228-byte STATELESS_GET_BLOCK payload.
     *                      Returns invalid DecodedTemplate if size != 228 bytes.
     * @param mining_channel Mining channel the miner is configured for
     *                      (1 = Prime, 2 = Hash).  Used for channel_consistent flag.
     * @param logger        Optional spdlog logger; may be nullptr.
     * @return DecodedTemplate -- caller checks DecodedTemplate::valid before use.
     */
    static DecodedTemplate decode_template(MiningTemplateInterface& tmpl_iface,
                                           const network::Payload& raw_payload,
                                           uint32_t mining_channel,
                                           std::shared_ptr<spdlog::logger> logger,
                                           bool auto_feed = true);

    // =========================================================================
    // Submit encode  (Miner -> Node)  with pre-checks
    // =========================================================================

    /**
     * @brief Pre-check and wire-encode a solved block for submission.
     *
     * Block serialization (including Prime-channel vOffsets) is delegated to
     * tmpl_iface.prepare_block_submission_from_solved() -- no serialization logic is
     * duplicated here.
     *
     * Pre-check sequence (canonical gates, in order):
     *  1. Nonce sanity:     solved_block.nNonce != 0
     *  2. Channel validity: solved_block.nChannel in {1, 2}
     *  3. Height validity:  solved_block.nHeight > 0
     *  4. Staleness (warn, do not block -- node is authoritative)
     *  5. Tip-moved  (warn, do not block)
     *  6. MiningTemplateInterface::prepare_block_submission_from_solved(...)
     *  7. Falcon sign: if falcon != nullptr, sign serialized block bytes and append.
     *  8. PacketBuilder::build(lane, SUBMIT_BLOCK, payload).
     *
     * @param tmpl_iface    MiningTemplateInterface that owns the active template.
     *                      Must have a valid template (has_valid_template() == true).
     * @param solved_block  Block header with nNonce filled by the worker.
     * @param vOffsets      Prime chain offsets from ValidatePrimeCandidate()
     *                      (empty for Hash channel).
     * @param falcon        Disposable Falcon wrapper; nullptr = skip signing.
     * @param lane          ProtocolLane::STATELESS -> opcode 0xD001,
     *                      ProtocolLane::LEGACY    -> opcode 0x01.
     * @param ht            Read-only HeightTracker snapshot for diagnostic
     *                      staleness / tip-moved pre-checks.  Does NOT write
     *                      back to the tracker.
     * @param submit_context Canonical submit-path context captured from the
     *                      authoritative session/template flow. When populated,
     *                      template_height must match the locally built submit
     *                      height before serialization proceeds.
     * @param logger        Optional spdlog logger; may be nullptr.
     * @return SubmitResult -- check valid before sending wire_bytes.
     */
     static SubmitResult encode_submit(MiningTemplateInterface& tmpl_iface,
                                       const ::LLP::CBlock& solved_block,
                                       const std::vector<uint8_t>& vOffsets,
                                       FalconSignatureWrapper* falcon,
                                       ProtocolLane lane,
                                       const HeightTracker::Snapshot& ht,
                                       std::shared_ptr<spdlog::logger> logger,
                                       const SubmitContext& submit_context = {});

    // =========================================================================
    // Channel-aware payload sizing helpers
    // =========================================================================

    /**
     * @brief Build a SubmitBlockPayloadInfo from runtime submit data.
     *
     * This helper computes expected sizes from real inputs rather than
     * assuming a universal fixed Tritium payload size.  Hash submissions
     * are fixed-size; Prime submissions are variable because
     * prepare_block_submission_from_solved(..., vOffsets) appends the Prime tail.
     *
     * @param channel          1 = Prime, 2 = Hash
     * @param block_data_size  Total serialized block bytes returned by
     *                         prepare_block_submission_from_solved() (includes vOffsets
     *                         for Prime).  For Hash this is always 216.
     * @param signature_size   Actual Falcon signature length (0 when unsigned).
     * @return Populated SubmitBlockPayloadInfo with all size fields.
     */
    static ChaCha20Wrapper::SubmitBlockPayloadInfo compute_submit_payload_info(
        uint32_t channel,
        size_t   block_data_size,
        size_t   signature_size);

private:
    StatelessBlockUtility() = delete;
};

} // namespace protocol
} // namespace nexusminer

#endif // NEXUSMINER_LLP_STATELESS_BLOCK_UTILITY_HPP
