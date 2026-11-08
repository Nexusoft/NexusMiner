#include "protocol/mining_template_interface.hpp"
#include "protocol/protocol_constants.hpp"
#include "LLP/block_utils.hpp"
#include "worker/worker.hpp"
#include "worker/block_header_utils.hpp"
#include <cassert>
#include <chrono>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <type_traits>
#include <utility>

namespace nexusminer {
namespace protocol {

namespace {

using BlockHeightField = std::remove_cv_t<std::remove_reference_t<decltype(std::declval<::LLP::CBlock>().nHeight)>>;
static_assert(std::is_same_v<BlockHeightField, uint32_t>,
              "CBlock::nHeight must remain uint32_t for unified-height submission guards");

} // namespace

MiningTemplateInterface::MiningTemplateInterface(uint8_t channel, SessionId session_id)
    : m_channel(channel)
    , m_session_id(session_id)
    , m_current_unified_height(0)
    , m_current_channel_height(0)
    , m_template_channel_height_snapshot(0)
    , m_has_snapshot(false)
    , m_last_unified_height(0)
    , m_template_received_time(std::chrono::steady_clock::now())
    , m_feed_handler(nullptr)
    , m_logger(spdlog::get("logger"))
    , m_templates_received(0)
    , m_templates_validated(0)
    , m_templates_rejected(0)
    , m_templates_stale(0)
    , m_templates_fed(0)
    , m_blocks_verified(0)
    , m_blocks_submitted(0)
    , m_total_read_time_us(0)
    , m_total_validation_time_us(0)
{
    if (!m_logger) {
        m_logger = spdlog::default_logger();
    }
    
    // Initialize template as empty
    m_current_template.state = TemplateState::INVALID;
    m_current_template.session_id = session_id;
    m_current_template.session_epoch = m_session_epoch;
    m_current_template.timestamp_received = 0;
    m_current_template.nChannelHeight = 0;
    m_template_channel_height_snapshot = 0;
    m_has_snapshot = false;
    
    // Validate channel
    if (m_channel != 1 && m_channel != 2) {
        m_logger->warn("[TemplateInterface] Invalid channel {} specified, defaulting to 2 (hash)", 
            static_cast<int>(m_channel));
        m_channel = 2;
    }
    
    m_logger->info("[TemplateInterface] Initialized for channel {} ({})", 
        static_cast<int>(m_channel), 
        (m_channel == 1) ? "prime" : "hash");
}

MiningTemplateInterface::MiningTemplateInterface(uint8_t channel, uint32_t session_id)
    : MiningTemplateInterface(channel, SessionId{session_id})
{
}

MiningTemplateInterface::~MiningTemplateInterface()
{
    m_logger->debug("[TemplateInterface] Destroyed");
}

MiningTemplateInterface::ValidationResult 
MiningTemplateInterface::read_template(const network::Payload& data,
                                        const std::string& source_endpoint,
                                        bool auto_feed)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    
    ValidationResult result;
    result.is_valid = false;
    result.is_stale = false;
    result.merkle_valid = false;
    result.height_valid = false;
    result.bits_valid = false;
    result.channel_valid = false;
    result.prime_origins_valid = false;
    result.retry_reason = GetBlockReason::VALIDATION_FAILURE;
    
    m_templates_received.fetch_add(1, std::memory_order_relaxed);
    
    m_logger->debug("[TemplateInterface] READ: Processing template ({} bytes) from {}", 
        data.size(), source_endpoint.empty() ? "unknown" : source_endpoint);
    
    // Snapshot session fields under the lock to avoid data races with
    // set_session_id/epoch/identity() which write under m_template_mutex.
    SessionId snapshot_session_id;
    SessionEpoch snapshot_session_epoch;
    {
        std::lock_guard<std::mutex> lock(m_template_mutex);
        snapshot_session_id = m_session_id;
        snapshot_session_epoch = m_session_epoch;
    }

    // Parse the block header
    MiningTemplate tmpl;
    tmpl.state = TemplateState::PENDING;
    tmpl.session_id = snapshot_session_id;
    tmpl.session_epoch = snapshot_session_epoch;
    tmpl.source_endpoint = source_endpoint;
    tmpl.timestamp_received = static_cast<uint64_t>(
        std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
    
    // Determine block format based on size
    if (data.size() == 216) {
        tmpl.format = BlockFormat::TRITIUM;
    } else if (data.size() >= 220) {
        tmpl.format = BlockFormat::LEGACY;
    } else {
        tmpl.format = BlockFormat::COMPACT;
    }
    
    if (!parse_block_header(data, tmpl.block)) {
        result.error_message = "Failed to parse block header from template data";
        m_templates_rejected.fetch_add(1, std::memory_order_relaxed);
        m_logger->error("[TemplateInterface] READ FAILED: {}", result.error_message);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        result.validation_time = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time);
        return result;
    }
    
    // ═══════════════════════════════════════════════════════════════════════
    // 📥 TEMPLATE RECEIVED FROM NODE - Trust Node's nChannel Value
    // ═══════════════════════════════════════════════════════════════════════
    
    // NOTE: format_hash_preview uses byte-reversed (big-integer) display order,
    // matching ToString()/GetHex()/SubString() and the node's log convention.
    // This allows direct cross-reference between [TemplateInterface] and [Worker_manager] logs,
    // and between miner logs and node logs (e.g. node's hashPrevBlock SubString display).
    // Helper lambda for hex preview formatting (reduces code duplication)
    auto format_hash_preview = [](const std::vector<uint8_t>& bytes, size_t preview_len = 16) -> std::string {
        std::ostringstream hex;
        hex << std::hex << std::setfill('0');
        size_t total = bytes.size();
        for (size_t i = 0; i < std::min(preview_len, total); ++i) {
            hex << std::setw(2) << static_cast<unsigned int>(bytes[total - 1 - i]);
        }
        return hex.str();
    };
    
    m_logger->info("[TemplateInterface] 📥 Template received from node:");
    m_logger->info("[TemplateInterface]   nVersion: {}", tmpl.block.nVersion);
    m_logger->info("[TemplateInterface]   nHeight: {}", tmpl.block.nHeight);
    m_logger->info("[TemplateInterface]   nChannel: {} ({})", 
        tmpl.block.nChannel,
        (tmpl.block.nChannel == 1) ? "Prime" : 
        (tmpl.block.nChannel == 2) ? "Hash" : "INVALID");
    m_logger->info("[TemplateInterface]   nBits: 0x{:08x}", tmpl.block.nBits);
    
    auto merkle_bytes = tmpl.block.hashMerkleRoot.GetBytes();
    m_logger->info("[TemplateInterface]   hashMerkleRoot: {}... ({} bytes)", 
        format_hash_preview(merkle_bytes), merkle_bytes.size());
    
    auto prev_bytes = tmpl.block.hashPrevBlock.GetBytes();
    m_logger->info("[TemplateInterface]   hashPrevBlock: {}... ({} bytes)", 
        format_hash_preview(prev_bytes), prev_bytes.size());
    
    // ✅ VALIDATION: Check if node's channel matches our connection
    // Node is authoritative for all template fields, but we should validate
    // that what the node sent matches what we expect based on our connection.
    
    if (tmpl.block.nChannel != m_channel) {
        m_logger->warn("[TemplateInterface] ⚠️  WARNING: Channel mismatch detected!");
        m_logger->warn("[TemplateInterface]   - Node sent: {} ({})", 
            tmpl.block.nChannel,
            (tmpl.block.nChannel == 1) ? "Prime" : "Hash");
        m_logger->warn("[TemplateInterface]   - Connection expects: {} ({})",
            static_cast<int>(m_channel), (m_channel == 1) ? "Prime" : "Hash");
        m_logger->warn("[TemplateInterface]   - Mining what node sent (node is authoritative)");
    } else {
        m_logger->info("[TemplateInterface] ✓ Channel validation passed: {} ({})",
            tmpl.block.nChannel, (tmpl.block.nChannel == 1) ? "Prime" : "Hash");
    }
    
    // Initialize channel height (will be set later when GET_ROUND response arrives)
    tmpl.nChannelHeight = 0;
    tmpl.height_guard.capture_unified_height(tmpl.block.nHeight);
     
    tmpl.nBits = tmpl.block.nBits;
    
    // Validate the template
    result = validate_template(tmpl);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto read_time = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time);
    m_total_read_time_us.fetch_add(read_time.count(), std::memory_order_relaxed);
    
    if (result.is_valid) {
        tmpl.state = TemplateState::VALID;

        bool canonical_prev_hash_confirmed = false;
        if (m_height_tracker) {
            const auto snapshot = m_height_tracker->GetSnapshot();
            canonical_prev_hash_confirmed =
                (snapshot.hash_prev_block != uint1024_t(0) &&
                 snapshot.hash_prev_block == tmpl.block.hashPrevBlock);
        }
        
        // Protect template assignment with mutex
        {
            std::lock_guard<std::mutex> lock(m_template_mutex);
            m_current_template = tmpl;
            m_current_unified_height = tmpl.block.nHeight;
            m_template_channel_height_snapshot = 0;
            m_has_snapshot = false;
            
            if (m_recovery_template_pending && m_last_unified_height == 0) {
                m_has_provisional_recovery_template = true;
                m_provisional_recovery_unified_height = tmpl.block.nHeight;
                m_logger->warn("[TemplateInterface] Provisional recovery template accepted: "
                               "height={} prev_hash={}... — awaiting channel-height finalization "
                               "or canonical prev-hash confirmation before continuity promotion",
                               tmpl.block.nHeight,
                               format_hash_preview(tmpl.block.hashPrevBlock.GetBytes()));
                if (canonical_prev_hash_confirmed) {
                    promote_provisional_recovery_template_unsafe("canonical prev-hash confirmation");
                }
            } else {
                // Update height tracking for sanity checks
                m_last_unified_height = tmpl.block.nHeight;
                m_recovery_template_pending = false;
                m_has_provisional_recovery_template = false;
                m_provisional_recovery_unified_height = 0;
            }
            
            // Update template received time for age monitoring
            m_template_received_time = std::chrono::steady_clock::now();

            // Atomic swap completed — any in-flight replacement promise (set
            // by mark_replacement_pending() while the previous template was
            // still being mined) is now satisfied. Clearing here ensures
            // take_expired_replacement_pending() does not later trigger a
            // spurious HEALTH_NO_TEMPLATE recovery for a swap that succeeded.
            m_replacement_pending = false;
            m_replacement_deadline = {};
            m_replacement_reason.clear();
        }
        
        m_templates_validated.fetch_add(1, std::memory_order_relaxed);
        
        m_logger->info("[TemplateInterface] ═══════════════════════════════════════");
        m_logger->info("[TemplateInterface] ✅ TEMPLATE VALIDATION SUCCESS");
        m_logger->info("[TemplateInterface]   Height: {}", tmpl.block.nHeight);
        m_logger->info("[TemplateInterface]   Channel: {} ({})", tmpl.block.nChannel, 
            (tmpl.block.nChannel == 1) ? "Prime" : "Hash");
        m_logger->info("[TemplateInterface]   nBits: 0x{:08x}", tmpl.nBits);
        m_logger->info("[TemplateInterface]   Validation time: {} μs", read_time.count());
        m_logger->info("[TemplateInterface] ═══════════════════════════════════════");

        // NOTE: HeightTracker::OnTemplateReceived() is intentionally NOT called here.
        // block.nHeight is the UNIFIED blockchain height — not the channel target.
        // The channel target height is only known after set_channel_height() is called.
        // HeightTracker will be notified from set_channel_height() with the correct value.

        if (auto_feed) {
            feed_current_template();
        }
    } else {
        m_templates_rejected.fetch_add(1, std::memory_order_relaxed);
        
        m_logger->warn("[TemplateInterface] ═══════════════════════════════════════");
        m_logger->warn("[TemplateInterface] ❌ TEMPLATE VALIDATION FAILED");
        m_logger->warn("[TemplateInterface]   Reason: {}", result.error_message);
        m_logger->warn("[TemplateInterface]   Height valid: {}", result.height_valid);
        m_logger->warn("[TemplateInterface]   Merkle valid: {}", result.merkle_valid);
        m_logger->warn("[TemplateInterface]   Bits valid: {}", result.bits_valid);
        m_logger->warn("[TemplateInterface]   Channel valid: {}", result.channel_valid);
        m_logger->warn("[TemplateInterface]   Is stale: {}", result.is_stale);
        m_logger->warn("[TemplateInterface] ═══════════════════════════════════════");
        
        if (result.is_stale) {
            m_templates_stale.fetch_add(1, std::memory_order_relaxed);
        }
        
        // Notify validation failure handler (if registered)
        if (m_validation_failure_handler) {
            m_validation_failure_handler(result);
        }
    }
    
    result.validation_time = read_time;
    return result;
}

MiningTemplateInterface::ValidationResult 
MiningTemplateInterface::read_template(network::Shared_payload data,
                                        const std::string& source_endpoint,
                                        bool auto_feed)
{
    if (!data || data->empty()) {
        ValidationResult result;
        result.is_valid = false;
        result.error_message = "Empty or null template data";
        return result;
    }
    return read_template(*data, source_endpoint, auto_feed);
}

MiningTemplateInterface::ValidationResult
MiningTemplateInterface::read_stateless_payload(const network::Payload& payload228,
                                                 const std::string& source_endpoint,
                                                 bool auto_feed)
{
    // ── Stateless BLOCK_DATA wire-format constants ────────────────────────────
    static constexpr size_t METADATA_SIZE = 12;   // [unified_height(4)][channel_height(4)][nBits(4)]
    static constexpr size_t BLOCK_SIZE    = 216;  // Tritium Block::Serialize() output
    static constexpr size_t EXPECTED_SIZE = METADATA_SIZE + BLOCK_SIZE; // 228

    // Size gate
    if (payload228.size() != EXPECTED_SIZE) {
        ValidationResult result;
        result.is_valid = false;
        result.error_message = "STATELESS_GET_BLOCK payload size " +
            std::to_string(payload228.size()) +
            " != " + std::to_string(EXPECTED_SIZE) + " (expected)";
        m_logger->error("[TemplateInterface] read_stateless_payload: {}",
                        result.error_message);
        return result;
    }

    // ── Extract 12-byte metadata prefix (big-endian) ─────────────────────────
    // These are DIAGNOSTIC fields only — the canonical mining state comes from
    // the 216-byte block body below.
    auto read_be32 = [&](size_t off) -> uint32_t {
        return (static_cast<uint32_t>(payload228[off])     << 24) |
               (static_cast<uint32_t>(payload228[off + 1]) << 16) |
               (static_cast<uint32_t>(payload228[off + 2]) <<  8) |
                static_cast<uint32_t>(payload228[off + 3]);
    };
    uint32_t nUnifiedHeightMeta   = read_be32(0);
    uint32_t nChannelHeightMeta   = read_be32(4);
    uint32_t nDifficultyMetaEcho  = read_be32(8);  // echoed nBits — not used separately

    m_logger->debug("[TemplateInterface] read_stateless_payload: "
                    "metadata unified={} channel={} nBits=0x{:08x}",
                    nUnifiedHeightMeta, nChannelHeightMeta, nDifficultyMetaEcho);

    // ── Delegate the 216-byte block body to the canonical read_template() ────
    network::Payload block_body(payload228.begin() + METADATA_SIZE, payload228.end());

    bool require_recovery_metadata_consistency = false;
    {
        std::lock_guard<std::mutex> lock(m_template_mutex);
        require_recovery_metadata_consistency =
            (m_recovery_template_pending && m_last_unified_height == 0);
    }

    if (require_recovery_metadata_consistency) {
        try {
            const auto body = llp_utils::deserialize_block_header(block_body);
            const bool metadata_height_has_next = (nUnifiedHeightMeta < UINT32_MAX);
            const uint32_t expected_body_height = metadata_height_has_next
                ? (nUnifiedHeightMeta + 1)
                : 0;
            const bool height_consistent =
                (metadata_height_has_next && body.nHeight == expected_body_height);
            const bool nbits_consistent = (body.nBits == nDifficultyMetaEcho);
            const bool channel_sane = (nChannelHeightMeta <= nUnifiedHeightMeta);

            if (!height_consistent || !nbits_consistent || !channel_sane) {
                ValidationResult result;
                result.is_valid = false;
                result.is_stale = false;
                result.merkle_valid = true;
                result.height_valid = height_consistent && channel_sane;
                result.bits_valid = nbits_consistent;
                result.channel_valid = true;
                result.validation_time = std::chrono::microseconds{0};
                result.error_message = "Provisional recovery template metadata/body mismatch";
                if (!metadata_height_has_next) {
                    result.error_message += ": metadata.unified_height is UINT32_MAX";
                } else if (!height_consistent) {
                    result.error_message += ": metadata.unified_height+1=" +
                        std::to_string(expected_body_height) +
                        " but body.nHeight=" + std::to_string(body.nHeight);
                } else if (!nbits_consistent) {
                    std::ostringstream msg;
                    msg << ": metadata.nBits=0x" << std::hex << nDifficultyMetaEcho
                        << " but body.nBits=0x" << body.nBits;
                    result.error_message += msg.str();
                } else {
                    result.error_message += ": metadata.channel_height=" +
                        std::to_string(nChannelHeightMeta) +
                        " > metadata.unified_height=" + std::to_string(nUnifiedHeightMeta);
                }
                m_templates_rejected.fetch_add(1, std::memory_order_relaxed);
                m_logger->critical("[TemplateInterface] 🚨 PROVISIONAL RECOVERY TEMPLATE rejected: "
                                   "metadata/body consistency failed "
                                   "(metadata.unified={} metadata.channel={} metadata.nBits=0x{:08x}; "
                                   "body.height={} body.nBits=0x{:08x})",
                                   nUnifiedHeightMeta, nChannelHeightMeta, nDifficultyMetaEcho,
                                   body.nHeight, body.nBits);
                return result;
            }

            m_logger->info("[TemplateInterface] Provisional recovery metadata/body "
                           "corroboration passed (metadata.unified={} → body.height={}, "
                           "nBits=0x{:08x})",
                           nUnifiedHeightMeta, body.nHeight, body.nBits);
        } catch (const std::exception& e) {
            ValidationResult result;
            result.is_valid = false;
            result.is_stale = false;
            result.merkle_valid = false;
            result.height_valid = false;
            result.bits_valid = false;
            result.channel_valid = false;
            result.validation_time = std::chrono::microseconds{0};
            result.error_message = std::string("Failed to parse provisional recovery template body: ") + e.what();
            m_templates_rejected.fetch_add(1, std::memory_order_relaxed);
            m_logger->critical("[TemplateInterface] 🚨 PROVISIONAL RECOVERY TEMPLATE rejected: {}", result.error_message);
            return result;
        }
    }

    auto result = read_template(block_body, source_endpoint, auto_feed);

    // ── Store diagnostic metadata in the current template (if decode succeeded) ─
    if (result.is_valid) {
        std::lock_guard<std::mutex> lock(m_template_mutex);
        m_current_template.nUnifiedHeightMeta = nUnifiedHeightMeta;
        m_current_template.nChannelHeightMeta = nChannelHeightMeta;
        // NOTE: m_last_unified_height is intentionally kept as block.nHeight (set by
        // read_template() above).  block.nHeight is the NEXT unified block height
        // (tStateBest.nHeight + 1), which is what the set_channel_height() corruption
        // guard compares against m_current_template.block.nHeight.  Overriding
        // m_last_unified_height with nUnifiedHeightMeta (the CURRENT tip, i.e. one less)
        // would cause the guard to always fire for stateless templates, silently
        // discarding every template received via read_stateless_payload().

        // ── Post-decode metadata divergence diagnostics (read-only, no behavior change) ─
        // These run after validate_template() has already accepted the body (including the
        // existing nBits==0 reject), so they are pure audit — they do NOT gate acceptance.

        // Diagnostic 1: metadata nBits vs body nBits
        // Contract: nDifficultyMetaEcho is an echo of body.nBits; they must agree.
        // (validate_template() already rejects body.nBits==0, so body_nbits is non-zero here.)
        const uint32_t body_nbits = m_current_template.block.nBits;
        if (body_nbits != nDifficultyMetaEcho) {
            m_stateless_nbits_divergence_count.fetch_add(1, std::memory_order_relaxed);
            m_logger->warn("[TemplateInterface] \u26a0 STATELESS METADATA DIVERGENCE (nBits): "
                           "body=0x{:08x} metadata=0x{:08x} delta=0x{:08x} \u2014 "
                           "node may be composing stateless BLOCK_DATA from mismatched sources "
                           "(cached body + fresh metadata). Miner uses body.nBits \u2014 submitted block "
                           "may not match node's current round difficulty.",
                           body_nbits, nDifficultyMetaEcho,
                           body_nbits ^ nDifficultyMetaEcho);
        } else {
            m_logger->debug("[TemplateInterface] \u2713 Stateless metadata nBits matches body: 0x{:08x}",
                            body_nbits);
        }

        // Diagnostic 2: metadata unified_height vs body nHeight
        // Contract: nUnifiedHeightMeta is the CURRENT tip; body.nHeight is the NEXT block (tip+1).
        const uint32_t body_height = m_current_template.block.nHeight;
        const uint32_t expected_body_height = nUnifiedHeightMeta + 1;
        if (body_height != expected_body_height) {
            m_stateless_height_divergence_count.fetch_add(1, std::memory_order_relaxed);
            m_logger->warn("[TemplateInterface] \u26a0 STATELESS METADATA DIVERGENCE (height): "
                           "metadata.unified_height={} \u2192 expected body.nHeight={} but body.nHeight={} "
                           "(delta={}) \u2014 node BLOCK_DATA composer may be racing chain advance "
                           "between body capture and metadata capture.",
                           nUnifiedHeightMeta, expected_body_height, body_height,
                           static_cast<int64_t>(body_height) - static_cast<int64_t>(expected_body_height));
        } else {
            m_logger->debug("[TemplateInterface] \u2713 Stateless metadata height consistent: "
                            "tip={} body={} (tip+1)", nUnifiedHeightMeta, body_height);
        }

        // Diagnostic 3: channel_height sanity (must not exceed unified_height)
        // NOTE: body.nChannelHeight is still 0 at this point (set later by set_channel_height()),
        // so we only check the internal consistency of the metadata fields themselves.
        // Do NOT compare against m_current_channel_height here — that field may be stale.
        if (nChannelHeightMeta > nUnifiedHeightMeta) {
            m_stateless_channel_sanity_violations.fetch_add(1, std::memory_order_relaxed);
            m_logger->warn("[TemplateInterface] \u26a0 STATELESS METADATA SANITY (channel > unified): "
                           "channel_height={} > unified_height={} \u2014 node payload composer emitted "
                           "an impossible state.",
                           nChannelHeightMeta, nUnifiedHeightMeta);
        }
    }

    return result;
}

bool MiningTemplateInterface::has_valid_template() const
{
    std::lock_guard<std::mutex> lock(m_template_mutex);
    return has_valid_template_unsafe();
}

bool MiningTemplateInterface::has_valid_template_unsafe() const
{
    // ASSUMES: m_template_mutex is already locked by caller
    return m_current_template.state == TemplateState::VALID;
}

uint32_t MiningTemplateInterface::get_node_channel_height() const
{
    return m_current_channel_height;
}

const MiningTemplateInterface::MiningTemplate* 
MiningTemplateInterface::get_current_template() const
{
    std::lock_guard<std::mutex> lock(m_template_mutex);
    
    if (has_valid_template_unsafe()) {
        return &m_current_template;
    }
    return nullptr;
}

void MiningTemplateInterface::set_template_feed_handler(TemplateFeedHandler handler)
{
    m_feed_handler = std::move(handler);
    m_logger->debug("[TemplateInterface] Feed handler registered");
}

void MiningTemplateInterface::set_template_fed_handler(std::function<void()> handler)
{
    m_template_fed_handler = std::move(handler);
}

void MiningTemplateInterface::set_template_feed_failed_handler(std::function<void()> handler)
{
    m_template_feed_failed_handler = std::move(handler);
}

void MiningTemplateInterface::reset_feed_debounce()
{
    std::lock_guard<std::mutex> lock(m_template_mutex);
    m_last_feed_tp = {};
    m_last_feed_height = 0;
    m_last_feed_prev_hash = {};
}

void MiningTemplateInterface::set_template_cleared_callback(TemplateClearedCallback cb)
{
    m_template_cleared_callback = std::move(cb);
    m_logger->debug("[TemplateInterface] Template-cleared callback registered");
}

void MiningTemplateInterface::set_validation_failure_handler(ValidationFailureHandler handler)
{
    m_validation_failure_handler = std::move(handler);
    m_logger->debug("[TemplateInterface] Validation failure handler registered");
}

bool MiningTemplateInterface::feed_current_template()
{
    std::unique_lock<std::mutex> lock(m_template_mutex);

    if (!has_valid_template_unsafe()) {
        m_logger->warn("[TemplateInterface] FEED: No valid template to feed");
        return false;
    }

    if (!m_feed_handler) {
        m_logger->debug("[TemplateInterface] FEED: No handler registered");
        return false;
    }

    // ── Unified Template Feed Debounce (PR #324 evolution: central dedup gate) ──
    // Prevents duplicate template distribution when the same block arrives via
    // multiple paths (push notification + GET_BLOCK response, or automatic feed
    // from read_template() + manual BLOCK_DATA handler re-push).
    //
    // This is the SINGLE AUTHORITATIVE debounce gate for the entire system.
    // Both Solo's BLOCK_DATA handler and Worker_manager's set_block flow rely on
    // this check to suppress duplicates at the source.
    const auto now = std::chrono::steady_clock::now();
    {
        auto ms_since_last = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - m_last_feed_tp).count();

        bool same_template = (m_current_template.block.nHeight == m_last_feed_height &&
                              m_current_template.block.hashPrevBlock == m_last_feed_prev_hash);

        // Bypass debounce if chain tip changed (hashPrevBlock differs)
        // This handles StakeMinter Guard 1 scenario: new template at same height
        // but building on a different tip (reorg or multi-tip race).
        bool tip_changed = (m_current_template.block.hashPrevBlock != m_last_feed_prev_hash) &&
                           (m_last_feed_height > 0);  // Skip on first feed

        if (same_template && ms_since_last < ProtocolConstants::TEMPLATE_FEED_DEBOUNCE_MS) {
            m_logger->info("[TemplateInterface] ⏱ Duplicate template feed suppressed "
                           "(height {} already fed {}ms ago, debounce {}ms)",
                           m_current_template.block.nHeight, ms_since_last,
                           ProtocolConstants::TEMPLATE_FEED_DEBOUNCE_MS);
            m_logger->info("[TemplateInterface]   Workers still initializing — "
                           "duplicate distribution prevented at source");
            return false;  // Duplicate suppressed
        }

        if (tip_changed) {
            m_logger->info("[TemplateInterface] ⚡ Debounce bypassed: chain tip changed "
                           "(hashPrevBlock), feeding new template at height {}",
                           m_current_template.block.nHeight);
        }

    }

    m_logger->info("[TemplateInterface] FEED: Feeding template at height {} to workers",
        m_current_template.block.nHeight);

    // Feed to handlers
    const bool handler_accepted = m_feed_handler(m_current_template, m_current_template.nBits);
    if (!handler_accepted) {
        m_logger->warn("[TemplateInterface] FEED: Handler rejected template at height {}",
            m_current_template.block.nHeight);
        mark_template_stale_unsafe("feed handler rejected template");
        lock.unlock();
        if (m_template_feed_failed_handler) {
            m_template_feed_failed_handler();
        }
        return false;
    }

    // Record this feed for next duplicate check only after the worker layer
    // accepts the template.
    m_last_feed_tp = now;
    m_last_feed_height = m_current_template.block.nHeight;
    m_last_feed_prev_hash = m_current_template.block.hashPrevBlock;

    m_templates_fed.fetch_add(1, std::memory_order_relaxed);

    // Recovery completion queries the current template and must not run while
    // its non-recursive mutex is held by the feed callback.
    lock.unlock();
    if (m_template_fed_handler) {
        m_template_fed_handler();
    }

    return true;
}

void MiningTemplateInterface::mark_template_stale(const std::string& reason)
{
    std::lock_guard<std::mutex> lock(m_template_mutex);
    mark_template_stale_unsafe(reason);
}

void MiningTemplateInterface::mark_template_stale_unsafe(const std::string& reason)
{
    // ASSUMES: m_template_mutex is already locked by caller
    if (m_current_template.state != TemplateState::INVALID) {
        
        m_current_template.state = TemplateState::INVALID;
        m_templates_stale.fetch_add(1, std::memory_order_relaxed);
        
        // Emit operator-facing terminology matching the actual invalidation cause.
        const bool has_reason = !reason.empty();
        const bool is_lag = has_reason && (reason == "multi_block_lag" || reason.find("behind") != std::string::npos);
        const bool is_timeout = has_reason && (reason.find("timeout") != std::string::npos ||
                                               reason.find("age")     != std::string::npos ||
                                               reason.find("expired") != std::string::npos);
        if (reason == "same_height_chain_reorg") {
            m_logger->info("[TemplateInterface] ⚡ Unified Tip-Anchor Changed — template superseded by same-height chain reorg (channel height unchanged, canonical prev hash replaced)");
        } else if (is_lag) {
            m_logger->info("[TemplateInterface] 📉 Template behind canonical chain ({})", reason);
        } else if (is_timeout) {
            m_logger->info("[TemplateInterface] ⏱️ Template expired: {}", reason);
        } else {
            m_logger->info("[TemplateInterface] Template invalidated{}{}", 
                has_reason ? ": " : "", reason);
        }
        m_template_channel_height_snapshot = 0;
        m_has_snapshot = false;

        // Reset debounce gate so any replacement template is not suppressed.
        // Mirrors the reset in discard_template_unsafe — every path that marks
        // the template stale must clear these so the next feed_current_template()
        // is not blocked by stale debounce data from the prior template.
        m_last_feed_tp = {};
        m_last_feed_height = 0;
        m_last_feed_prev_hash = {};
    }
}

bool MiningTemplateInterface::verify_block_creation(const std::vector<uint8_t>& merkle_root,
                                                     uint64_t nonce) const
{
    if (!has_valid_template()) {
        m_logger->error("[TemplateInterface] VERIFY: No valid template for block verification");
        return false;
    }
    
    // Verify merkle root size
    if (merkle_root.size() != 32 && merkle_root.size() != 64) {
        m_logger->error("[TemplateInterface] VERIFY: Invalid merkle root size {} (expected 32 or 64)",
            merkle_root.size());
        return false;
    }
    
    // Note: Zero nonce is technically valid in cryptographic mining - the mining process
    // may legitimately produce a zero nonce as a solution. We log at debug level for
    // diagnostics but do not reject the block.
    if (nonce == 0) {
        m_logger->debug("[TemplateInterface] VERIFY: Zero nonce found - valid mining result");
    }
    
    m_logger->debug("[TemplateInterface] VERIFY: Block creation verified (merkle: {} bytes, nonce: 0x{:016x})",
        merkle_root.size(), nonce);
    
    return true;
}

std::vector<uint8_t> MiningTemplateInterface::prepare_block_submission(
    const std::vector<uint8_t>& merkle_root,
    uint64_t nonce)
{
    if (!verify_block_creation(merkle_root, nonce)) {
        return {};
    }
    
    if (!has_valid_template()) {
        m_logger->error("[TemplateInterface] Cannot prepare block submission: no valid template");
        return {};
    }
    
    std::lock_guard<std::mutex> lock(m_template_mutex);
    
    // Create a copy of the current block template with the solved merkle root and nonce
    ::LLP::CBlock solved_block = m_current_template.block;
    const auto height_guard = m_current_template.height_guard;
     
    // Update with the mined merkle root
    solved_block.hashMerkleRoot.SetBytes(merkle_root);
    
    // Update with the found nonce
    solved_block.nNonce = nonce;
    
    // CRITICAL HEIGHT AUDIT: block.nHeight must match the unified height captured
    // when GET_BLOCK / BLOCK_DATA constructed this template, never the channel tip/target.
    const bool height_guard_ok = height_guard.matches(solved_block);
    if (!height_guard_ok) {
        m_logger->error("[SUBMIT AUDIT] Height guard mismatch: actual={} expected_unified={} channel_height={}",
            solved_block.nHeight, height_guard.unified_height.get(), height_guard.channel_height.get());
    }
    assert(height_guard_ok && "block.nHeight must remain the unified GET_BLOCK height");
    m_logger->info("[SUBMIT AUDIT] solved_block.nHeight = {} (must equal unified GET_BLOCK height {})",
        solved_block.nHeight, height_guard.unified_height.get());
    m_logger->info("[SUBMIT AUDIT] nChannelHeight (metadata) = {} (channel marker={}, NOT in block bytes)",
        m_current_template.nChannelHeight, is_channel_height(height_guard.channel_height));

    if (!height_guard_ok) {
        m_logger->error("[SUBMIT AUDIT] ❌ ABORT: block.nHeight {} != unified GET_BLOCK height {}",
            solved_block.nHeight, height_guard.unified_height.get());
        m_logger->error("[SUBMIT AUDIT]   Channel height marker={} value={}",
            is_channel_height(height_guard.channel_height), height_guard.channel_height.get());
        return {};
    }
    
    // Determine serialization format from template
    bool is_tritium = (m_current_template.format == BlockFormat::TRITIUM);
    
    // Serialize the full block
    auto payload = llp_utils::serialize_full_block(solved_block, is_tritium);
    
    // SUBMISSION AUDIT: Log solved block fields for ProofHash cross-reference with node.
    // The node calls pBlock->ProofHash() on nVersion..nBits. These must match for
    // GetPrime() / prime validation to succeed.
    // CRITICAL: block.nHeight must be unified blockchain height, NOT channel height.
    {
        m_logger->info("[SUBMIT AUDIT]");
        m_logger->info("[SUBMIT AUDIT]   block.nVersion     = {}", solved_block.nVersion);
        auto prev_bytes = solved_block.hashPrevBlock.GetBytes();
        std::string prev_hex;
        for (size_t i = 0; i < std::min(prev_bytes.size(), size_t(8)); ++i)
        {
            char buf[3];
            snprintf(buf, sizeof(buf), "%02x", prev_bytes[i]);
            prev_hex += buf;
        }
        m_logger->info("[SUBMIT AUDIT]   block.hashPrevBlock = {}... (tip anchor)", prev_hex);
        m_logger->info("[SUBMIT AUDIT]   block.nChannel     = {}", solved_block.nChannel);
        m_logger->info("[SUBMIT AUDIT]   block.nHeight      = {} (unified blockchain height)", solved_block.nHeight);
        m_logger->info("[SUBMIT AUDIT]   nChannelHeight     = {} (metadata only, NOT in block bytes)", m_current_template.nChannelHeight);
        m_logger->info("[SUBMIT AUDIT]   block.nBits        = 0x{:08x}", solved_block.nBits);
        m_logger->info("[SUBMIT AUDIT]   block.nNonce       = 0x{:016x}", solved_block.nNonce);
        m_logger->info("[SUBMIT AUDIT]   serialized size    = {} bytes (expected 216 for Tritium)", payload.size());
        
        if (!height_guard_ok)
        {
            m_logger->error("[SUBMIT AUDIT]   ❌ ABORT: block.nHeight {} != unified GET_BLOCK height {}",
                solved_block.nHeight, height_guard.unified_height.get());
            m_logger->error("[SUBMIT AUDIT]   ProofHash() would mismatch — block.nHeight must never be substituted with channel height {}.",
                height_guard.channel_height.get());
            m_blocks_verified.fetch_add(1, std::memory_order_relaxed);
            return {};
        }
        
        // Verify nHeight survives serialization at offset 200 (Tritium: big-endian uint32 at [200-203])
        if (is_tritium && payload.size() >= 204)
        {
            uint32_t nHeightSerialized =
                (static_cast<uint32_t>(payload[200]) << 24) |
                (static_cast<uint32_t>(payload[201]) << 16) |
                (static_cast<uint32_t>(payload[202]) << 8) |
                static_cast<uint32_t>(payload[203]);
            if (nHeightSerialized != solved_block.nHeight)
                m_logger->error("[SUBMIT AUDIT]   ❌ CRITICAL: nHeight serialization mismatch! "
                    "block.nHeight={} but serialized[200-203]={}",
                    solved_block.nHeight, nHeightSerialized);
            else
                m_logger->info("[SUBMIT AUDIT]   ✅ nHeight verified in serialized payload: {}", nHeightSerialized);
        }
        
        // Verify nNonce survives serialization at offset 208 (Tritium: little-endian uint64 at [208-215])
        if (is_tritium && payload.size() >= 216)
        {
            uint64_t nNonceSerialized = 0;
            for (int i = 0; i < 8; ++i)
                nNonceSerialized |= static_cast<uint64_t>(payload[208 + i]) << (i * 8);
            if (nNonceSerialized != nonce)
                m_logger->error("[SUBMIT AUDIT]   ❌ CRITICAL: nNonce serialization mismatch! "
                    "expected=0x{:016x} but serialized[208-215]=0x{:016x}",
                    nonce, nNonceSerialized);
            else
                m_logger->info("[SUBMIT AUDIT]   ✅ nNonce verified in serialized payload[208-215]: 0x{:016x}", nNonceSerialized);
        }
    }
    
    m_blocks_verified.fetch_add(1, std::memory_order_relaxed);
    
    m_logger->info("[TemplateInterface] Block submission prepared: {} bytes ({} format)",
        payload.size(), is_tritium ? "Tritium" : "Legacy");
    
    return payload;
}

std::vector<uint8_t> MiningTemplateInterface::prepare_block_submission(
    const std::vector<uint8_t>& merkle_root,
    uint64_t nonce,
    const std::vector<uint8_t>& vOffsets)
{
    // Delegate the block serialization to the base overload
    auto payload = prepare_block_submission(merkle_root, nonce);
    if (payload.empty())
        return payload;

    // For Prime channel, append Cunningham-chain offsets so the node can verify
    // the prime cluster via GetPrimeDifficulty() / GetOffsets().
    // Hash channel vOffsets are always empty — no-op.
    if (!vOffsets.empty() && m_channel == 1) {
        payload.insert(payload.end(), vOffsets.begin(), vOffsets.end());
        m_logger->debug("[TemplateInterface] Appended {} vOffset bytes for Prime channel",
                        vOffsets.size());
    }

    return payload;
}

std::vector<uint8_t> MiningTemplateInterface::prepare_block_submission_from_solved(
    const Block_data& solved)
{
    if (!has_valid_template()) {
        m_logger->error("[TemplateInterface] Cannot prepare block submission: no valid template");
        return {};
    }

    // Build the submit block directly from the worker's snapshot.
    // This is the canonical "worker proved these exact bytes" hand-off used by:
    //   worker -> worker_manager -> prepare_block_submission_from_solved(...) -> Solo
    //
    // These are the exact field values the worker used when performing primality / hash
    // proof-of-work testing.  Crucially, nHeight comes from the worker, not from
    // m_current_template, so a concurrent template refresh cannot silently advance it.
    ::LLP::CBlock submit_block;
    submit_block.nVersion       = solved.nVersion;
    submit_block.hashPrevBlock  = solved.previous_hash;
    submit_block.hashMerkleRoot = solved.merkle_root;
    submit_block.nChannel       = solved.nChannel;
    submit_block.nHeight        = solved.nHeight;
    submit_block.nBits          = solved.nBits;
    submit_block.nNonce         = solved.nNonce;

    // Snapshot current-template metadata needed for serialization format and drift check.
    // Hold the mutex only long enough to copy the scalar fields.
    bool       is_tritium;
    uint32_t   tmpl_height;
    uint1024_t tmpl_prev;
    {
        std::lock_guard<std::mutex> lock(m_template_mutex);
        is_tritium  = (m_current_template.format == BlockFormat::TRITIUM);
        tmpl_height = m_current_template.block.nHeight;
        tmpl_prev   = m_current_template.block.hashPrevBlock;
    }

    // Serialize using the worker's snapshot fields.
    // The payload returned here is the authoritative 216-byte block body that Solo later
    // re-decodes before signing/encrypting, so any mismatch must be logged immediately.
    auto payload = llp_utils::serialize_full_block(submit_block, is_tritium);

    // Option C drift guard: non-tautological comparison of worker snapshot vs.
    // current template.  Logs a warning but does NOT abort — the worker's PoW was
    // proven against its snapshot, and the node may still accept it if the chain
    // has not yet advanced past submit_block.nHeight.
    {
        const bool height_drift = (submit_block.nHeight != tmpl_height);
        const bool prev_drift   = (submit_block.hashPrevBlock != tmpl_prev);

        if (height_drift || prev_drift) {
            m_logger->warn("[SUBMIT AUDIT] ⚠ Template advanced between worker-found and submit-prep:");
            m_logger->warn("[SUBMIT AUDIT]   worker_height={} current_template_height={}",
                submit_block.nHeight, tmpl_height);
            m_logger->warn("[SUBMIT AUDIT]   worker_prev_drift={} (worker block kept; PoW belongs to worker's snapshot)",
                prev_drift);
            m_logger->warn("[SUBMIT AUDIT]   This is the failure mode of legacy prepare_block_submission(merkle, nonce).");
        } else {
            m_logger->info("[SUBMIT AUDIT] ✓ Worker snapshot matches current template (height={}).",
                submit_block.nHeight);
        }
    }

    // SUBMISSION AUDIT: Log solved block fields for ProofHash cross-reference with node.
    {
        m_logger->info("[SUBMIT AUDIT]");
        m_logger->info("[SUBMIT AUDIT]   block.nVersion     = {}", submit_block.nVersion);
        auto prev_bytes = submit_block.hashPrevBlock.GetBytes();
        std::string prev_hex;
        for (size_t i = 0; i < std::min(prev_bytes.size(), size_t(8)); ++i)
        {
            char buf[3];
            snprintf(buf, sizeof(buf), "%02x", prev_bytes[i]);
            prev_hex += buf;
        }
        m_logger->info("[SUBMIT AUDIT]   block.hashPrevBlock = {}... (tip anchor)", prev_hex);
        m_logger->info("[SUBMIT AUDIT]   block.nChannel     = {}", submit_block.nChannel);
        m_logger->info("[SUBMIT AUDIT]   block.nHeight      = {} (unified blockchain height)", submit_block.nHeight);
        m_logger->info("[SUBMIT AUDIT]   block.nBits        = 0x{:08x}", submit_block.nBits);
        m_logger->info("[SUBMIT AUDIT]   block.nNonce       = 0x{:016x}", submit_block.nNonce);
        m_logger->info("[SUBMIT AUDIT]   serialized size    = {} bytes (expected 216 for Tritium)", payload.size());

        // Verify nHeight survives serialization at offset 200 (Tritium: big-endian uint32 at [200-203])
        if (is_tritium && payload.size() >= 204)
        {
            uint32_t nHeightSerialized =
                (static_cast<uint32_t>(payload[200]) << 24) |
                (static_cast<uint32_t>(payload[201]) << 16) |
                (static_cast<uint32_t>(payload[202]) << 8) |
                static_cast<uint32_t>(payload[203]);
            if (nHeightSerialized != submit_block.nHeight)
                m_logger->error("[SUBMIT AUDIT]   ❌ CRITICAL: nHeight serialization mismatch! "
                    "block.nHeight={} but serialized[200-203]={}",
                    submit_block.nHeight, nHeightSerialized);
            else
                m_logger->info("[SUBMIT AUDIT]   ✅ nHeight verified in serialized payload: {}", nHeightSerialized);
        }

        // Verify nNonce survives serialization at offset 208 (Tritium: little-endian uint64 at [208-215])
        if (is_tritium && payload.size() >= 216)
        {
            uint64_t nNonceSerialized = 0;
            for (int i = 0; i < 8; ++i)
                nNonceSerialized |= static_cast<uint64_t>(payload[208 + i]) << (i * 8);
            if (nNonceSerialized != submit_block.nNonce)
                m_logger->error("[SUBMIT AUDIT]   ❌ CRITICAL: nNonce serialization mismatch! "
                    "expected=0x{:016x} but serialized[208-215]=0x{:016x}",
                    submit_block.nNonce, nNonceSerialized);
            else
                m_logger->info("[SUBMIT AUDIT]   ✅ nNonce verified in serialized payload[208-215]: 0x{:016x}", nNonceSerialized);
        }
    }

    m_blocks_verified.fetch_add(1, std::memory_order_relaxed);

    m_logger->info("[TemplateInterface] Block submission prepared: {} bytes ({} format)",
        payload.size(), is_tritium ? "Tritium" : "Legacy");

    return payload;
}

std::vector<uint8_t> MiningTemplateInterface::prepare_block_submission_from_solved(
    const Block_data& solved,
    const std::vector<uint8_t>& vOffsets)
{
    auto payload = prepare_block_submission_from_solved(solved);
    if (payload.empty())
        return payload;

    // For Prime channel, append Cunningham-chain offsets so the node can verify
    // the prime cluster.  Hash channel vOffsets are always empty — no-op.
    // These bytes must remain glued to the same solved snapshot because the node
    // treats the Prime submit plaintext as:
    //   [216-byte solved block][vOffsets...][timestamp][sig_len][signature]
    // Use solved.nChannel (worker's snapshot) as the source of truth, not m_channel.
    if (!vOffsets.empty() && solved.nChannel == 1) {
        payload.insert(payload.end(), vOffsets.begin(), vOffsets.end());
        m_logger->debug("[TemplateInterface] Appended {} vOffset bytes for Prime channel",
                        vOffsets.size());
    }

    return payload;
}

void MiningTemplateInterface::set_session_id(SessionId session_id)
{
    std::lock_guard<std::mutex> lock(m_template_mutex);
    
    m_session_id = session_id;
    m_current_template.session_id = session_id;
    m_logger->info("[TemplateInterface] Session ID set to 0x{:08x}", session_id.get());
}

void MiningTemplateInterface::set_session_epoch(SessionEpoch session_epoch)
{
    std::lock_guard<std::mutex> lock(m_template_mutex);

    m_session_epoch = session_epoch;
    m_current_template.session_epoch = session_epoch;
    m_logger->info("[TemplateInterface] Session epoch set to {}", session_epoch.get());
}

void MiningTemplateInterface::set_session_identity(const SessionIdentity& identity)
{
    std::lock_guard<std::mutex> lock(m_template_mutex);

    m_session_identity = identity;
    // Keep backward-compat fields in sync
    m_session_id = identity.session_id();
    m_session_epoch = identity.session_epoch();
    m_current_template.identity = identity;
    m_current_template.session_id = identity.session_id();
    m_current_template.session_epoch = identity.session_epoch();
    m_logger->info("[TemplateInterface] Session identity set: {}", identity.fingerprint());
}

void MiningTemplateInterface::set_session_binding(const SessionBinding& binding)
{
    std::lock_guard<std::mutex> lock(m_template_mutex);

    m_session_id = binding.session_id;
    m_session_epoch = binding.session_epoch;
    m_current_template.session_id = binding.session_id;
    m_current_template.session_epoch = binding.session_epoch;

    if (!binding.identity.is_empty()) {
        m_session_identity = binding.identity;
        m_current_template.identity = binding.identity;
        m_logger->info("[TemplateInterface] Session binding set: {}", binding.identity.fingerprint());
        return;
    }

    m_session_identity = SessionIdentity{};
    m_current_template.identity = SessionIdentity{};
    if (!binding.has_session()) {
        m_logger->debug("[TemplateInterface] Session binding cleared: session_id=0x{:08x}, epoch={}",
                        binding.session_id.get(), binding.session_epoch.get());
        return;
    }

    m_logger->warn("[TemplateInterface] Session binding missing canonical identity: session_id=0x{:08x}, epoch={}",
                   binding.session_id.get(), binding.session_epoch.get());
}

SessionIdentity MiningTemplateInterface::get_session_identity() const
{
    std::lock_guard<std::mutex> lock(m_template_mutex);
    return m_session_identity;
}

void MiningTemplateInterface::set_channel(uint8_t channel)
{
    if (channel != 1 && channel != 2) {
        m_logger->warn("[TemplateInterface] Invalid channel {} specified, keeping current channel {}",
            static_cast<int>(channel), static_cast<int>(m_channel));
        return;
    }
    
    m_channel = channel;
    m_logger->info("[TemplateInterface] Channel set to {} ({})",
        static_cast<int>(m_channel), (m_channel == 1) ? "prime" : "hash");
}

MiningTemplateInterface::TemplateStats MiningTemplateInterface::get_stats() const
{
    TemplateStats stats;
    stats.templates_received = m_templates_received.load(std::memory_order_relaxed);
    stats.templates_validated = m_templates_validated.load(std::memory_order_relaxed);
    stats.templates_rejected = m_templates_rejected.load(std::memory_order_relaxed);
    stats.templates_stale = m_templates_stale.load(std::memory_order_relaxed);
    stats.templates_fed = m_templates_fed.load(std::memory_order_relaxed);
    stats.blocks_verified = m_blocks_verified.load(std::memory_order_relaxed);
    stats.blocks_submitted = m_blocks_submitted.load(std::memory_order_relaxed);
    stats.total_read_time_us = m_total_read_time_us.load(std::memory_order_relaxed);
    stats.total_validation_time_us = m_total_validation_time_us.load(std::memory_order_relaxed);
    stats.templates_expired_age = m_templates_expired_age.load(std::memory_order_relaxed);
    stats.templates_expired_height = m_templates_expired_height.load(std::memory_order_relaxed);
    stats.stateless_nbits_divergence_count = m_stateless_nbits_divergence_count.load(std::memory_order_relaxed);
    stats.stateless_height_divergence_count = m_stateless_height_divergence_count.load(std::memory_order_relaxed);
    stats.stateless_channel_sanity_violations = m_stateless_channel_sanity_violations.load(std::memory_order_relaxed);
    return stats;
}

void MiningTemplateInterface::reset_stats()
{
    m_templates_received.store(0, std::memory_order_relaxed);
    m_templates_validated.store(0, std::memory_order_relaxed);
    m_templates_rejected.store(0, std::memory_order_relaxed);
    m_templates_stale.store(0, std::memory_order_relaxed);
    m_templates_fed.store(0, std::memory_order_relaxed);
    m_blocks_verified.store(0, std::memory_order_relaxed);
    m_blocks_submitted.store(0, std::memory_order_relaxed);
    m_total_read_time_us.store(0, std::memory_order_relaxed);
    m_total_validation_time_us.store(0, std::memory_order_relaxed);
    m_templates_expired_age.store(0, std::memory_order_relaxed);
    m_templates_expired_height.store(0, std::memory_order_relaxed);
    m_stateless_nbits_divergence_count.store(0, std::memory_order_relaxed);
    m_stateless_height_divergence_count.store(0, std::memory_order_relaxed);
    m_stateless_channel_sanity_violations.store(0, std::memory_order_relaxed);
    
    m_logger->debug("[TemplateInterface] Statistics reset");
}

const char* MiningTemplateInterface::state_to_string(TemplateState state)
{
    switch (state) {
        case TemplateState::INVALID:    return "INVALID";
        case TemplateState::PENDING:    return "PENDING";
        case TemplateState::VALID:      return "VALID";
        default:                        return "UNKNOWN";
    }
}

MiningTemplateInterface::ValidationResult 
MiningTemplateInterface::validate_template(const MiningTemplate& tmpl)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    
    ValidationResult result;
    result.is_valid = true;
    result.is_stale = false;
    result.merkle_valid = true;
    result.height_valid = true;
    result.bits_valid = true;
    result.channel_valid = true;
    
    // TRAINING WHEELS: Show validation details
    m_logger->info("[TemplateInterface] ═══ TEMPLATE VALIDATION ═══");
    m_logger->info("[TemplateInterface] Template details:");
    m_logger->info("[TemplateInterface]   - Height: {}", tmpl.block.nHeight);
    m_logger->info("[TemplateInterface]   - Channel: {} (expected: {})", 
        tmpl.block.nChannel, m_channel);
    m_logger->info("[TemplateInterface]   - nBits: 0x{:08x}", tmpl.block.nBits);
    m_logger->info("[TemplateInterface]   - nVersion: {}", tmpl.block.nVersion);
    m_logger->info("[TemplateInterface]   - Unified height: {}", m_current_unified_height);
    m_logger->info("[TemplateInterface]   - Node channel height: {}", m_current_channel_height);
    m_logger->info("[TemplateInterface]   - Channel height: {}",
        (tmpl.nChannelHeight != 0) ? std::to_string(tmpl.nChannelHeight) : "pending");
    
    // ═══════════════════════════════════════════════════════════════════════
    // VALIDATE nChannel (CRITICAL - Do NOT overwrite, only validate)
    // ═══════════════════════════════════════════════════════════════════════
    
    // Validation: Check if channel is valid (1=Prime or 2=Hash)
    // nChannel is now properly deserialized from all block formats (Tritium/Legacy/Compact)
    if (tmpl.block.nChannel != 1 && tmpl.block.nChannel != 2) {
        result.channel_valid = false;
        result.is_valid = false;
        result.error_message = "❌ Invalid channel value: " + 
            std::to_string(tmpl.block.nChannel);
        m_logger->error("[TemplateInterface] {}", result.error_message);
        m_logger->error("[TemplateInterface]   Expected: 1 (Prime) or 2 (Hash)");
        m_logger->error("[TemplateInterface]   Got: {}", tmpl.block.nChannel);
        m_logger->error("[TemplateInterface]   This indicates deserialization failed");
        return result;  // Reject template immediately
    }
    
    // Check if channel matches our connection preference (informational only)
    if (tmpl.block.nChannel != m_channel) {
        // NOTE: This is just informational logging - we don't reject the template
        // The node is authoritative, and we mine what it sends
        m_logger->info("[TemplateInterface] ℹ️  Channel info: Node sent channel {} but connection is for channel {}",
            tmpl.block.nChannel, static_cast<int>(m_channel));
        m_logger->info("[TemplateInterface]   Mining what node sent (node is authoritative)");
    }
    
    m_logger->info("[TemplateInterface] ✓ nChannel validation passed: {} ({})", 
        tmpl.block.nChannel,
        (tmpl.block.nChannel == 1) ? "Prime" : "Hash");
    
    // ═══════════════════════════════════════════════════════════════════════
    // VALIDATE UNIFIED HEIGHT (Sanity Check for Corrupted Height)
    // ═══════════════════════════════════════════════════════════════════════
    
    // Check for unreasonable height jumps (e.g., 6.5M → 1.9B) or deep reorgs
    // Normal height changes should be within ±100 blocks
    // - Forward jumps >100: likely corrupted height
    // - Backward jumps >100: likely corrupted height (normal reorgs are shallow)
    if (m_last_unified_height > 0) {
        // Calculate absolute height difference to handle both directions
        int64_t height_diff = static_cast<int64_t>(tmpl.block.nHeight) - static_cast<int64_t>(m_last_unified_height);
        uint64_t abs_height_delta = static_cast<uint64_t>(std::abs(height_diff));
        
        if (abs_height_delta > 100) {
            result.height_valid = false;
            result.is_valid = false;
            
            std::string direction = (height_diff > 0) ? "forward" : "backward";
            result.error_message = "Unified height " + direction + " jump exceeds sanity threshold: " + 
                std::to_string(m_last_unified_height) + " → " + 
                std::to_string(tmpl.block.nHeight) + " (delta: " + 
                std::to_string(abs_height_delta) + " blocks, max: 100)";
            
            m_logger->error("[TemplateInterface] ❌ CORRUPTED HEIGHT DETECTED");
            m_logger->error("[TemplateInterface]   Previous height: {}", m_last_unified_height);
            m_logger->error("[TemplateInterface]   New height: {}", tmpl.block.nHeight);
            m_logger->error("[TemplateInterface]   Delta: {} blocks {} (max allowed: 100)", 
                           abs_height_delta, direction);
            m_logger->error("[TemplateInterface]   This indicates corrupted template data or deep reorg");
            m_logger->error("[TemplateInterface]   Mining will be stopped to prevent wasted hashrate");
            return result;  // Reject template immediately
        }
        
        // Log direction of height change for diagnostics
        if (height_diff > 0) {
            m_logger->debug("[TemplateInterface] ✓ Height advanced {} blocks (forward)", abs_height_delta);
        } else if (height_diff < 0) {
            m_logger->info("[TemplateInterface] ℹ️  Height decreased {} blocks (reorg detected)", abs_height_delta);
        } else {
            m_logger->debug("[TemplateInterface] ℹ️  Height unchanged (duplicate template)");
        }
    } else if (m_recovery_template_pending) {
        if (m_has_provisional_recovery_template && m_provisional_recovery_unified_height > 0) {
            int64_t height_diff = static_cast<int64_t>(tmpl.block.nHeight) -
                                  static_cast<int64_t>(m_provisional_recovery_unified_height);
            uint64_t abs_height_delta = static_cast<uint64_t>(std::abs(height_diff));
            if (abs_height_delta > 100) {
                result.height_valid = false;
                result.is_valid = false;

                std::string direction = (height_diff > 0) ? "forward" : "backward";
                result.error_message = "Provisional recovery height " + direction +
                    " jump exceeds sanity threshold: " +
                    std::to_string(m_provisional_recovery_unified_height) + " → " +
                    std::to_string(tmpl.block.nHeight) + " (delta: " +
                    std::to_string(abs_height_delta) + " blocks, max: 100)";

                m_logger->error("[TemplateInterface] ❌ PROVISIONAL RECOVERY HEIGHT REJECTED");
                m_logger->error("[TemplateInterface]   Provisional height: {}", m_provisional_recovery_unified_height);
                m_logger->error("[TemplateInterface]   Candidate height: {}", tmpl.block.nHeight);
                m_logger->error("[TemplateInterface]   Delta: {} blocks {} (max allowed: 100)",
                                abs_height_delta, direction);
                return result;
            }
            m_logger->info("[TemplateInterface] Provisional recovery template update accepted: "
                           "height {} is within {} blocks of unpromoted recovery candidate {}",
                           tmpl.block.nHeight, abs_height_delta,
                           m_provisional_recovery_unified_height);
        } else {
            // discard_template_unsafe() cleared the baseline during recovery.  Do
            // not fall back to HeightTracker.unified_height here: after reorg it
            // may represent a different observation point than the replacement.
            // Accept this as provisional only; read_template() will not promote
            // m_last_unified_height until a later corroboration step.
            m_logger->info("[TemplateInterface] First template after discard accepted as "
                           "provisional recovery candidate; skipping stale HeightTracker continuity "
                           "fallback until channel-height finalization or canonical prev-hash confirmation");
        }
    } else {
        // m_last_unified_height == 0 at startup: there is no previous template to
        // compare against, so skip the continuity sanity check for this template.
        m_logger->debug("[TemplateInterface] ℹ️  First template after startup - skipping height continuity sanity check");
    }
    
    // Validate channel height if available (only mark stale when THIS channel advanced)
    // Use channel height from GET_ROUND - this is the CRITICAL staleness check
    uint32_t node_channel_height = get_node_channel_height();
    
    // Template semantics:
    //   - nChannelHeight = height of block being mined (node_height + 1)
    //   - node_channel_height = current height on blockchain for this channel
    // Template is stale only if node's channel reached or passed the height we're mining for
    if (tmpl.nChannelHeight != 0 && node_channel_height > 0) {
        if (node_channel_height >= tmpl.nChannelHeight) {
            result.is_stale = true;
            result.height_valid = false;
            result.is_valid = false;
            // Use template's actual channel for accurate error messages
            std::string channel_name = (tmpl.block.nChannel == 1) ? "Prime" : "Hash";
            result.error_message = channel_name + " channel stale: template for height " + 
                std::to_string(tmpl.nChannelHeight) + " but node already at " + 
                std::to_string(node_channel_height);
            
            m_logger->warn("[TemplateInterface] {}", result.error_message);
            return result;
        }
        
        // Log success - unified height can differ and that's NORMAL
        m_logger->debug("[TemplateInterface] ✓ Template valid: node channel height {} < template height {}",
            node_channel_height, tmpl.nChannelHeight);
        
        // If unified height differs, log informational message
        if (tmpl.block.nHeight != m_current_unified_height) {
            m_logger->debug("[TemplateInterface] ℹ️  Unified height differs (template={}, current={})",
                tmpl.block.nHeight, m_current_unified_height);
            m_logger->debug("[TemplateInterface]    This is NORMAL when other channels mine blocks");
        }
    } else {
        m_logger->info("[TemplateInterface] ✓ Channel height pending, skipping staleness check");
    }
    
    // Validate nBits (difficulty) is non-zero
    if (tmpl.block.nBits == 0) {
        result.bits_valid = false;
        result.is_valid = false;
        result.error_message = "Invalid nBits (difficulty) value: 0";
        m_logger->error("[TemplateInterface] ❌ VALIDATION FAILED: {}", result.error_message);
    } else {
        m_logger->info("[TemplateInterface] ✓ Difficulty validation passed");
    }

    if (!HasSufficientPrimeOrigins(tmpl.block)) {
        result.prime_origins_valid = false;
        result.is_valid = false;
        result.retry_reason = GetBlockReason::PRIME_ORIGIN_TOO_LOW;
        result.error_message = "Prime origins below 1016-bits (ProofHash < bnPrimeMinOrigins)";
        m_logger->error("[TemplateInterface] ❌ VALIDATION FAILED: {}", result.error_message);
        return result;
    }
    result.prime_origins_valid = true;
    m_logger->debug("[TemplateInterface] ✓ Prime origins validation passed");
    
    // Validate merkle root is not all zeros (basic sanity check)
    bool merkle_all_zeros = true;
    auto merkle_bytes = tmpl.block.hashMerkleRoot.GetBytes();
    for (auto byte : merkle_bytes) {
        if (byte != 0) {
            merkle_all_zeros = false;
            break;
        }
    }
    if (merkle_all_zeros) {
        result.merkle_valid = false;
        result.is_valid = false;
        result.error_message = "Invalid merkle root: all zeros";
        m_logger->error("[TemplateInterface] ❌ VALIDATION FAILED: {}", result.error_message);
    } else {
        // Log first few bytes of merkle root for verification
        std::ostringstream merkle_hex;
        merkle_hex << std::hex << std::setfill('0');
        size_t preview_len = std::min(merkle_bytes.size(), static_cast<size_t>(16));
        for (size_t i = 0; i < preview_len; ++i) {
            merkle_hex << std::setw(2) << static_cast<unsigned int>(merkle_bytes[i]) << " ";
        }
        m_logger->info("[TemplateInterface] ✓ Merkle root validation passed");
        m_logger->debug("[TemplateInterface]   First {} bytes: {}", preview_len, merkle_hex.str());
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.validation_time = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time);
    m_total_validation_time_us.fetch_add(result.validation_time.count(), std::memory_order_relaxed);
    
    if (result.is_valid) {
        m_logger->info("[TemplateInterface] ✅ ALL VALIDATION CHECKS PASSED");
        m_logger->info("[TemplateInterface]   Validation time: {} μs", result.validation_time.count());
    } else {
        m_logger->error("[TemplateInterface] ❌ VALIDATION FAILED");
        m_logger->error("[TemplateInterface]   Reason: {}", result.error_message);
    }
    m_logger->info("[TemplateInterface] ═══════════════════════════");
    
    return result;
}

bool MiningTemplateInterface::parse_block_header(const network::Payload& data, 
                                                  ::LLP::CBlock& block)
{
    try {
        // Detect and log block type based on size
        std::string block_type;
        if (data.size() == 216) {
            block_type = "Tritium (216 bytes, nChannel at offset 211)";
        } else if (data.size() >= 220) {
            block_type = "Legacy (220+ bytes, nChannel at offset 196)";
        } else if (data.size() == 92) {
            block_type = "Compact (92 bytes, sequential format)";
        } else {
            block_type = "Unknown format (" + std::to_string(data.size()) + " bytes)";
        }
        
        m_logger->debug("[TemplateInterface] Parsing {} block header", block_type);
        
        block = llp_utils::deserialize_block_header(data);
        
        // Log parsed block details
        m_logger->debug("[TemplateInterface] Parsed block header successfully:");
        m_logger->debug("[TemplateInterface]   - Block type: {}", block_type);
        m_logger->debug("[TemplateInterface]   - nVersion: {}", block.nVersion);
        m_logger->debug("[TemplateInterface]   - nChannel: {}", block.nChannel);
        m_logger->debug("[TemplateInterface]   - nHeight: {}", block.nHeight);
        m_logger->debug("[TemplateInterface]   - nBits: 0x{:08x}", block.nBits);
        m_logger->debug("[TemplateInterface]   - nNonce: 0x{:016x}", block.nNonce);
        m_logger->debug("[TemplateInterface]   - nTime: {}", block.nTime);
        
        return true;
    }
    catch (const std::exception& e) {
        m_logger->error("[TemplateInterface] Failed to parse block header: {}", e.what());
        m_logger->error("[TemplateInterface]   - Payload size: {} bytes", data.size());
        
        // Log first few bytes for debugging using C++ stringstream
        if (data.size() > 0) {
            std::ostringstream hex_preview;
            hex_preview << std::hex << std::setfill('0');
            size_t preview_len = std::min(data.size(), static_cast<size_t>(32));
            for (size_t i = 0; i < preview_len; ++i) {
                hex_preview << std::setw(2) << static_cast<unsigned int>(data[i]) << " ";
            }
            m_logger->error("[TemplateInterface]   - First {} bytes: {}", preview_len, hex_preview.str());
        }
        
        return false;
    }
}

// =========================================================================
// Template Staleness Prevention (LLL-TAO PR #131 Client-Side Integration)
// =========================================================================

bool MiningTemplateInterface::is_template_stale() const
{
    std::lock_guard<std::mutex> lock(m_template_mutex);
    
    if (m_current_template.state == TemplateState::INVALID ||
        m_current_template.state == TemplateState::PENDING) {
        return true;
    }
    
    uint64_t age = get_template_age_unsafe();
    return age > MAX_TEMPLATE_AGE;
}

bool MiningTemplateInterface::is_template_old() const
{
    std::lock_guard<std::mutex> lock(m_template_mutex);
    
    if (m_current_template.state == TemplateState::INVALID ||
        m_current_template.state == TemplateState::PENDING) {
        return true;
    }
    
    uint64_t age = get_template_age_unsafe();
    return age > WARNING_TEMPLATE_AGE;
}

uint64_t MiningTemplateInterface::get_template_age() const
{
    std::lock_guard<std::mutex> lock(m_template_mutex);
    return get_template_age_unsafe();
}

uint64_t MiningTemplateInterface::get_template_age_unsafe() const
{
    // ASSUMES: m_template_mutex is already locked by caller
    if (m_current_template.state == TemplateState::INVALID ||
        m_current_template.timestamp_received == 0) {
        return 0;
    }
    
    auto now = std::chrono::system_clock::now();
    uint64_t current_time = static_cast<uint64_t>(
        std::chrono::system_clock::to_time_t(now));
    
    // Handle clock skew - if current time is less than timestamp, treat as stale
    if (current_time < m_current_template.timestamp_received) {
        m_logger->warn("[TemplateInterface] Clock skew detected: current time < template timestamp");
        // Return a value greater than MAX_TEMPLATE_AGE to trigger staleness
        constexpr uint64_t CLOCK_SKEW_STALENESS_VALUE = MAX_TEMPLATE_AGE + 1;
        return CLOCK_SKEW_STALENESS_VALUE;
    }
    
    return current_time - m_current_template.timestamp_received;
}

bool MiningTemplateInterface::update_height(uint32_t new_height)
{
    std::lock_guard<std::mutex> lock(m_template_mutex);
    
    m_logger->debug("[TemplateInterface] Height update: {} -> {}", m_current_unified_height, new_height);
    
    // Check if template should be discarded due to height change
    bool template_discarded = false;
    
    if (new_height > m_current_unified_height && has_valid_template_unsafe()) {
        // Height has advanced - discard current template
        uint32_t old_height = m_current_template.block.nHeight;
        
        m_logger->info("[TemplateInterface] 🔔 Height changed: {} -> {} (template at height {})",
            m_current_unified_height, new_height, old_height);
        
        if (old_height < new_height) {
            m_logger->info("[TemplateInterface] ❌ Template behind canonical chain (height advanced: {} → {}) — discarding", old_height, new_height);
            discard_template_unsafe("Height changed");
            template_discarded = true;
            m_templates_expired_height.fetch_add(1, std::memory_order_relaxed);
        }
    }
    
    // Always update current height
    m_current_unified_height = new_height;
    
    return template_discarded;
}

bool MiningTemplateInterface::update_channel_height(uint32_t channel, uint32_t new_channel_height)
{
    std::lock_guard<std::mutex> lock(m_template_mutex);
    
    m_current_channel_height = new_channel_height;

    m_logger->debug("[TemplateInterface] Channel {} height update: {}", 
        channel, new_channel_height);
    
    // Check if we have a valid template
    if (!has_valid_template_unsafe()) {
        m_logger->debug("[TemplateInterface] No active template to check");
        return false;
    }
    
    // Only check if this is the same channel as our template
    if (m_current_template.block.nChannel != channel) {
        m_logger->debug("[TemplateInterface] Template is for channel {}, update is for channel {} - no action needed",
            m_current_template.block.nChannel, channel);
        return false;
    }
    
    // Check if template is stale based on channel height.
    // Template builds the NEXT block, so nChannelHeight is our mining target.
    // Stale only when the chain tip has REACHED or PASSED our target.
    // new_channel_height < nChannelHeight means the chain is still behind our
    // target — template is valid (normal during burst push lag and async round lag).
    if (m_current_template.nChannelHeight == 0) {
        m_logger->debug("[TemplateInterface] Template channel height not yet set (pending finalization)");
        return false;
    }

    // FIXED: only discard when chain tip has reached or passed our mining target.
    // new_channel_height == nChannelHeight → block we are mining was just found → stale.
    // new_channel_height >  nChannelHeight → chain is multiple blocks ahead       → stale.
    // new_channel_height <  nChannelHeight → chain is still behind our target     → valid.
    if (new_channel_height >= m_current_template.nChannelHeight) {
        m_logger->info("[TemplateInterface] ⚠ Channel {} height {}/{} reached/passed target {} — template stale",
            channel, new_channel_height,
            (channel == 1) ? "Prime" : (channel == 2) ? "Hash" : "Stake",
            m_current_template.nChannelHeight);
        discard_template_unsafe("Channel height reached or passed target");
        m_templates_expired_height.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    m_logger->debug("[TemplateInterface] ✓ Template FRESH — channel={} still below target={}",
        new_channel_height, m_current_template.nChannelHeight);
    return false;
}

void MiningTemplateInterface::set_template_channel_height_snapshot(uint32_t channel_height)
{
    std::lock_guard<std::mutex> lock(m_template_mutex);
    // Invariant: must not be called when the template is already finalized (nChannelHeight > 0).
    // Calling this after set_channel_height() re-poisons the guard and causes false staleness
    // on the next GET_ROUND, even though the template is still 100% valid.
    if (m_current_template.nChannelHeight > 0) {
        m_logger->warn("[TemplateInterface] set_template_channel_height_snapshot({}) ignored: "
                       "template already finalized with nChannelHeight={}",
                       channel_height, m_current_template.nChannelHeight);
        return;
    }
    m_template_channel_height_snapshot = channel_height;
    m_has_snapshot = true;
    m_logger->debug("[TemplateInterface] Snapshot channel height set to {}", channel_height);
}

void MiningTemplateInterface::clear_template_channel_height_snapshot()
{
    std::lock_guard<std::mutex> lock(m_template_mutex);
    m_template_channel_height_snapshot = 0;
    m_has_snapshot = false;
    m_logger->debug("[TemplateInterface] Snapshot channel height cleared");
}

bool MiningTemplateInterface::check_staleness_by_channel_delta(uint32_t current_channel_height)
{
    std::lock_guard<std::mutex> lock(m_template_mutex);

    // When nChannelHeight is finalized, use it directly as the staleness threshold.
    // The snapshot is irrelevant once the template target is known: stale only when
    // the chain tip has met or passed the block we are mining toward.
    if (m_current_template.nChannelHeight > 0) {
        if (current_channel_height >= m_current_template.nChannelHeight) {
            m_logger->warn("[TemplateInterface] 📉 Template stale: channel {} reached or exceeded target {}",
                current_channel_height, m_current_template.nChannelHeight);
            discard_template_unsafe("Channel height reached nChannelHeight target");
            m_templates_expired_height.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
        m_logger->debug("[TemplateInterface] ✓ VALID: channel={} below target={}",
            current_channel_height, m_current_template.nChannelHeight);
        return false;
    }

    // Fallback: snapshot-based check for pending-finalization case only
    // (nChannelHeight == 0 means template arrived but GET_ROUND not yet processed).
    if (!m_has_snapshot) {
        m_logger->debug("[TemplateInterface] No snapshot and nChannelHeight not set; cannot check staleness");
        return false;
    }

    // Log the comparison for diagnostics
    m_logger->debug("[TemplateInterface] Staleness check (snapshot fallback): current={} snapshot={}",
        current_channel_height, m_template_channel_height_snapshot);

    if (current_channel_height > m_template_channel_height_snapshot) {
        m_logger->warn("[TemplateInterface] 📉 Template behind canonical chain: channel advanced {} → {}",
            m_template_channel_height_snapshot, current_channel_height);
        discard_template_unsafe("Channel height advanced past snapshot");
        m_templates_expired_height.fetch_add(1, std::memory_order_relaxed);
        return true;
    }

    m_logger->debug("[TemplateInterface] ✓ VALID: channel unchanged at {}",
        current_channel_height);
    return false;
}

void MiningTemplateInterface::set_channel_height(uint32_t channel_height)
{
    std::lock_guard<std::mutex> lock(m_template_mutex);
    
    if (m_current_template.state == TemplateState::INVALID) {
        m_logger->warn("[TemplateInterface] Cannot set channel height - no active template");
        return;
    }
    
    // DEFENSIVE: Verify block.nHeight was NOT corrupted (must remain unified height).
    // set_channel_height() must ONLY update metadata — never block.nHeight.
    // m_last_unified_height is always set to tmpl.block.nHeight in read_template()
    // (read_stateless_payload() deliberately does NOT override it).  For stateless
    // templates, nUnifiedHeightMeta is the current chain tip (block.nHeight - 1);
    // using it here would cause a false mismatch on every stateless template.
    // So both should be identical; any difference indicates in-flight corruption.
    const auto expected_unified_height = m_current_template.height_guard.unified_height.get();
    if (expected_unified_height > 0 && !m_current_template.height_guard.matches(m_current_template.block)) {
        m_logger->error("[TemplateInterface] Height guard mismatch: actual={} expected_unified={}",
            m_current_template.block.nHeight, expected_unified_height);
    }
    assert(m_current_template.height_guard.matches(m_current_template.block) &&
           "set_channel_height() must not overwrite block.nHeight");
    if (expected_unified_height > 0 && !m_current_template.height_guard.matches(m_current_template.block)) {
        m_logger->error("[TemplateInterface] ❌ CRITICAL: block.nHeight ({}) != last_unified_height ({})!",
            m_current_template.block.nHeight, expected_unified_height);
        m_logger->error("[TemplateInterface]   block.nHeight was corrupted — discarding template");
        discard_template_unsafe("block.nHeight corruption detected");
        return;
    }
    
    // Only update metadata field, NEVER block.nHeight
    m_current_template.nChannelHeight = channel_height;
    m_current_template.height_guard.capture_channel_height(channel_height);
    m_current_channel_height = (channel_height > 0) ? (channel_height - 1) : 0;
    m_template_channel_height_snapshot = 0;
    m_has_snapshot = false;
    m_logger->info("[TemplateInterface] ✓ Template channel height (metadata) set to {} (block.nHeight={} unchanged)",
        channel_height, m_current_template.block.nHeight);

    promote_provisional_recovery_template_unsafe("channel-height finalization");

    // Notify HeightTracker with the correct channel target height (channel_height is node tip + 1).
    // This must be called here (not in read_template()) because block.nHeight is the UNIFIED
    // blockchain height and would cause HeightTracker::channel_target to be set incorrectly.
    if (m_height_tracker && channel_height > 0) {
        m_height_tracker->OnTemplateReceived(m_current_template.block.nChannel, channel_height);
        std::string drift_msg = m_height_tracker->ExplainMismatch();
        if (!drift_msg.empty()) {
            m_logger->info("{}", drift_msg);
        }
    }
}

void MiningTemplateInterface::promote_provisional_recovery_template_unsafe(const char* reason)
{
    // ASSUMES: m_template_mutex is already locked by caller
    if (!m_recovery_template_pending || !m_has_provisional_recovery_template) {
        return;
    }

    m_last_unified_height = m_current_template.block.nHeight;
    m_recovery_template_pending = false;
    m_has_provisional_recovery_template = false;
    m_provisional_recovery_unified_height = 0;

    m_logger->info("[TemplateInterface] Provisional recovery template promoted after {}: "
                   "height={} is now the continuity baseline",
                   reason ? reason : "corroboration", m_last_unified_height);
}

void MiningTemplateInterface::discard_template(const std::string& reason)
{
    {
        std::lock_guard<std::mutex> lock(m_template_mutex);
        discard_template_unsafe(reason);
    }
    // Invoke the cleared callback outside the mutex so callers can safely
    // zero derived state (e.g. SESSION_KEEPALIVE prevblock_suffix).
    if (m_template_cleared_callback) {
        m_template_cleared_callback();
    }
}

void MiningTemplateInterface::discard_template_unsafe(const std::string& reason)
{
    // ASSUMES: m_template_mutex is already locked by caller

    // Always reset the debounce gate BEFORE the early-return guard.
    // A rapid second discard on an already-EMPTY template (e.g. three burst blocks in
    // quick succession) must still clear the gate so the next feed_current_template()
    // is not blocked by stale debounce data from the last successful feed.
    m_last_feed_tp = {};        // Reset timestamp — effectively disables time-based debounce
    m_last_feed_height = 0;     // Reset last feed height to default
    m_last_feed_prev_hash = {}; // Reset last prev-hash used for duplicate detection

    if (m_current_template.state == TemplateState::INVALID) {
        m_logger->debug("[TemplateInterface] No template to discard");
        return;
    }

    // Reset the unified-height baseline so the next template passes the height sanity check
    // regardless of how many blocks the chain advanced while the miner was in degraded mode.
    // Without this reset, validate_template() rejects every BLOCK_DATA response with
    // "Corrupted Height Detected" (abs_height_delta > 100) when the chain has advanced more
    // than 100 unified blocks during a degraded-mode recovery period, causing
    // get_block_sent_total to increment indefinitely with zero successful template installations.
    //
    // Do this only when a real active template is discarded.  Startup/burst paths can ask
    // to discard while no template has ever been installed; treating that no-op as recovery
    // makes the initial block template look like a reorg in operator logs.  Repeated no-op
    // startup discards should remain a clean slate: debounce is cleared above, but recovery
    // state must not be initialized until there is actual installed work to recover from.
    m_last_unified_height = 0;
    m_recovery_template_pending = true;
    m_has_provisional_recovery_template = false;
    m_provisional_recovery_unified_height = 0;
    
    m_logger->info("[TemplateInterface] Discarding template: {}", reason);
    m_logger->info("[TemplateInterface]   - Height: {}", m_current_template.block.nHeight);
    m_logger->info("[TemplateInterface]   - Age: {}s", get_template_age_unsafe());
    m_logger->info("[TemplateInterface]   - State: {}", state_to_string(m_current_template.state));
    
    // Mark as stale (mark_template_stale_unsafe also resets the debounce triple,
    // but we already reset above so this is a harmless double-clear).
    mark_template_stale_unsafe(reason);
    m_template_channel_height_snapshot = 0;
    m_has_snapshot = false;

    // Any pending-replacement promise is moot once we explicitly discard
    // (either we just timed it out via take_expired_replacement_pending(),
    // or some other path has decided the template is unrecoverable).
    m_replacement_pending = false;
    m_replacement_deadline = {};
    m_replacement_reason.clear();
}

void MiningTemplateInterface::mark_replacement_pending(const std::string& reason,
                                                      int64_t timeout_ms)
{
    if (timeout_ms <= 0) {
        timeout_ms = REPLACEMENT_PENDING_DEFAULT_TIMEOUT_MS;
    }

    std::lock_guard<std::mutex> lock(m_template_mutex);

    // Reset the feed-debounce triple so the imminent BLOCK_DATA reply is not
    // suppressed as a duplicate by feed_current_template(). This mirrors the
    // reset performed by discard_template_unsafe() / mark_template_stale_unsafe()
    // but without the destructive state change those methods imply.
    m_last_feed_tp = {};
    m_last_feed_height = 0;
    m_last_feed_prev_hash = {};

    const bool was_pending = m_replacement_pending;
    m_replacement_pending = true;
    m_replacement_deadline = std::chrono::steady_clock::now()
                           + std::chrono::milliseconds(timeout_ms);
    m_replacement_reason = reason;

    if (!was_pending) {
        m_logger->info("[TemplateInterface] \u26a1 Replacement pending: {} "
                       "(template stays VALID, deadline {}ms) \u2014 workers continue mining "
                       "until BLOCK_DATA atomically swaps in",
                       reason, timeout_ms);
    } else {
        m_logger->debug("[TemplateInterface] Replacement-pending deadline refreshed: {} ({}ms)",
                        reason, timeout_ms);
    }
}

bool MiningTemplateInterface::is_replacement_pending() const
{
    std::lock_guard<std::mutex> lock(m_template_mutex);
    if (!m_replacement_pending) {
        return false;
    }
    return std::chrono::steady_clock::now() < m_replacement_deadline;
}

bool MiningTemplateInterface::take_expired_replacement_pending(std::string& reason)
{
    std::lock_guard<std::mutex> lock(m_template_mutex);
    reason.clear();
    if (!m_replacement_pending) {
        return false;
    }
    if (std::chrono::steady_clock::now() < m_replacement_deadline) {
        return false;
    }
    reason = m_replacement_reason;
    m_replacement_pending = false;
    m_replacement_deadline = {};
    m_replacement_reason.clear();
    return true;
}

std::optional<std::chrono::steady_clock::time_point>
MiningTemplateInterface::get_replacement_pending_deadline() const
{
    std::lock_guard<std::mutex> lock(m_template_mutex);
    if (!m_replacement_pending) {
        return std::nullopt;
    }
    return m_replacement_deadline;
}

bool MiningTemplateInterface::needs_channel_height_finalization() const
{
    std::lock_guard<std::mutex> lock(m_template_mutex);
    
    // Template needs finalization if it's valid but channel height not set
    return has_valid_template_unsafe() && m_current_template.nChannelHeight == 0;
}

uint32_t MiningTemplateInterface::get_template_height() const
{
    std::lock_guard<std::mutex> lock(m_template_mutex);
    
    if (m_current_template.state == TemplateState::INVALID) {
        return 0;
    }
    return m_current_template.block.nHeight;
}

void MiningTemplateInterface::set_height_tracker(HeightTracker* tracker)
{
    m_height_tracker = tracker;
}

} // namespace protocol
} // namespace nexusminer
