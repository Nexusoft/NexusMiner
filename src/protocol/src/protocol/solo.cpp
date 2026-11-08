#include "protocol/solo.hpp"
#include "protocol/protocol.hpp"
#include "protocol/falcon_constants.hpp"
#include "protocol/protocol_constants.hpp"
#include "protocol/push_notification_handler.hpp"
#include "protocol/packet_builder.hpp"
#include "protocol/genesis_utils.hpp"
#include "protocol/hex_prefix_utils.hpp"
#include "protocol/serialization_helpers.hpp"
#include "protocol/session_status_policy.hpp"
#include "protocol/session_start_parser.hpp"
#include "protocol/get_block_control.hpp"
#include "packet.hpp"
#include "network/connection.hpp"
#include "stats/stats_collector.hpp"
#include "LLP/block_utils.hpp"
#include "LLP/llp_logging.hpp"
#include "LLP/utils.hpp"
#include <LLC/hash/SK.h>
#include "include/stateless_block_utility.hpp"
#include "../miner_keys.hpp"
#include "hex_utils.h"
#include <openssl/sha.h>
#include <asio/error.hpp>
#include <chrono>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <cstring>

namespace nexusminer
{
namespace protocol
{

bool Solo::queue_payload(const std::shared_ptr<network::Connection>& connection,
                         const network::Shared_payload& payload,
                         const char* context)
{
    if (!connection) {
        m_logger->error("{} transmit failed: no connection available", context ? context : "[Solo]");
        return false;
    }
    if (!payload || payload->empty()) {
        m_logger->error("{} transmit failed: payload is null or empty", context ? context : "[Solo]");
        return false;
    }

    try {
        if (connection->transmit(payload)) {
            return true;
        }
        m_logger->warn("{} transmit rejected — payload was not queued", context ? context : "[Solo]");
    } catch (const std::exception& e) {
        m_logger->error("{} transmit failed: {}", context ? context : "[Solo]", e.what());
    }
    return false;
}

bool Solo::request_and_queue_get_block(const std::shared_ptr<network::Connection>& connection,
                                       GetBlockReason reason,
                                       const char* context)
{
    const auto snap = m_height_tracker.GetSnapshot();
    uint32_t template_target = 0;
    if (m_template_interface) {
        if (auto const* tmpl = m_template_interface->get_current_template()) {
            template_target = tmpl->block.nHeight;
        }
    }
    const auto now = std::chrono::steady_clock::now();
    const int64_t since_last_block_data_ms =
        (m_last_template_adopted_at == std::chrono::steady_clock::time_point::min())
            ? -1
            : std::chrono::duration_cast<std::chrono::milliseconds>(
                  now - m_last_template_adopted_at).count();

    const char* caller_stack = context ? context : "(none)";
    m_logger->info(
        "[Solo GET_BLOCK SOURCE] reason={} context=\"{}\" template_target={} canonical_unified={} "
        "pending_for={} since_last_block_data_ms={} caller_stack=\"{}\"",
        reason_name(reason),
        context ? context : "(none)",
        template_target,
        snap.canonical_unified_height,
        m_pending_get_block.unified_height,
        since_last_block_data_ms,
        caller_stack);

    auto payload = get_work(reason);
    if (!payload || payload->empty()) {
        return false;
    }

    auto reasons_bypass_post_adoption = [](GetBlockReason r) {
        return should_bypass_all_dedup(r) ||
               r == GetBlockReason::BLOCK_REJECTED ||
               r == GetBlockReason::SESSION_REAUTH ||
               r == GetBlockReason::INITIAL_REQUEST;
    };

    if (m_last_template_adopted_at != std::chrono::steady_clock::time_point::min() &&
        !reasons_bypass_post_adoption(reason))
    {
        const auto since_adopt = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - m_last_template_adopted_at);
        if (since_adopt < POST_ADOPTION_SUPPRESSION_WINDOW)
        {
            m_logger->info(
                "[Solo GET_BLOCK] Suppressed: template adopted {}ms ago (window {}ms) "
                "— redundant request from reason={} context={}",
                since_adopt.count(),
                POST_ADOPTION_SUPPRESSION_WINDOW.count(),
                reason_name(reason),
                context ? context : "(none)");
            return false;
        }
    }

    if (!queue_payload(connection, payload, context)) {
        return false;
    }
    mark_get_block_pending(reason);
    return true;
}

void Solo::dispatch_recovery_or_fallback(const std::shared_ptr<network::Connection>& connection,
                                         GetBlockReason reason,
                                         const char* fallback_context)
{
    // Production chokepoint: Worker_manager owns the GET_BLOCK transmission
    // schedule (burst guard, hashprev backoff, auth/session gates).  If the
    // handler indicates it scheduled a request, do NOT also fire a local one —
    // that was the "double-tap" storm.
    if (m_recovery_handler) {
        m_logger->info("[Solo] Delegating recovery GET_BLOCK to handler (reason={})",
                       reason_name(reason));
        const bool handler_scheduled = m_recovery_handler(reason);
        if (handler_scheduled) {
            return;
        }
        m_logger->debug("[Solo] Recovery handler declined (suppressed/deferred); "
                        "no local fallback issued (reason={})", reason_name(reason));
        return;
    }

    // Fallback path: no handler registered (tests, standalone Solo, or a
    // NodeSession that has not yet wired Worker_manager).  Issue a direct
    // GET_BLOCK on the supplied connection so unit-test expectations are
    // preserved and standalone modes can still recover.
    if (!connection) {
        m_logger->warn("[Solo] Recovery GET_BLOCK suppressed: no handler and no connection (reason={})",
                       reason_name(reason));
        return;
    }
    if (!request_and_queue_get_block(connection, reason, fallback_context)) {
        m_logger->error("[Solo] Recovery fallback GET_BLOCK failed (reason={})", reason_name(reason));
    }
}

uint32_t Solo::get_blocks_accepted() const
{
    return m_stats_collector ? m_stats_collector->get_global_stats().m_accepted_blocks : 0;
}

uint32_t Solo::get_blocks_rejected() const
{
    return m_stats_collector ? m_stats_collector->get_global_stats().m_rejected_blocks : 0;
}

SessionOwnershipStamp Solo::get_last_submitted_owner_snapshot() const
{
    std::scoped_lock lock(m_last_submitted_mutex);
    return m_last_submitted_state.owner;
}

void Solo::store_last_submitted_state(const SessionOwnershipStamp& owner,
                                      uint64_t nonce,
                                      const uint1024_t& prev_hash,
                                      uint32_t height,
                                      uint32_t channel)
{
    std::scoped_lock lock(m_last_submitted_mutex);
    m_last_submitted_state.valid = true;
    m_last_submitted_state.owner = owner;
    m_last_submitted_state.nonce = nonce;
    m_last_submitted_state.prev_hash = prev_hash;
    m_last_submitted_state.height = height;
    m_last_submitted_state.channel = channel;
}

Solo::LastSubmittedBlockState Solo::get_last_submitted_state_snapshot() const
{
    std::scoped_lock lock(m_last_submitted_mutex);
    return m_last_submitted_state;
}

Solo::LastSubmittedBlockState Solo::consume_last_submitted_state()
{
    std::scoped_lock lock(m_last_submitted_mutex);
    auto state = m_last_submitted_state;
    m_last_submitted_state.clear();
    return state;
}

void Solo::clear_last_submitted_state()
{
    std::scoped_lock lock(m_last_submitted_mutex);
    m_last_submitted_state.clear();
}

void Solo::clear_pending_submit_result_state()
{
    m_submit_result_gate.clear();
    clear_last_submitted_state();
}

void Solo::note_block_discarded(const std::string& reason)
{
    m_logger->critical("[Solo Submit] BLOCK DISCARDED (not submitted — solved work lost): {}", reason);
    if (m_stats_collector) {
        stats::Global_delta delta{};
        delta.m_discarded_blocks = 1;
        m_stats_collector->update_global_stats(delta);
    }
}

namespace {

bool is_expected_cached_session_resync(bool local_has_state, bool authoritative_has_state)
{
    return !local_has_state && authoritative_has_state;
}

FalconHashKeyId falcon_pubkey_to_hash_key_id(const std::vector<uint8_t>& pubkey)
{
    if (pubkey.empty()) {
        return FalconHashKeyId{};
    }

    return FalconHashKeyId(keys::to_hex(LLC::SK256(pubkey).GetBytes()));
}

std::vector<uint8_t> strip_submit_wire_header(const network::Payload& framed,
                                              ProtocolLane lane)
{
    const std::size_t header_size = (lane == ProtocolLane::STATELESS) ? 6u : 5u;
    if (framed.size() <= header_size) {
        return {};
    }

    return std::vector<uint8_t>(framed.begin() + header_size, framed.end());
}

/// Returns true when a BLOCK_AVAILABLE push for at least `unified_height` was
/// received within the last `guard_ms` milliseconds — implying BLOCK_DATA is
/// still in transit and a GET_BLOCK would be redundant.
bool push_implies_block_data_in_transit(
    const HeightTracker::Snapshot& snap,
    uint32_t unified_height,
    int64_t guard_ms,
    std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now())
{
    if (snap.last_push_notification_at == std::chrono::steady_clock::time_point{}) {
        return false;
    }
    if (snap.push_unified_height < unified_height) {
        return false;
    }
    int64_t age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - snap.last_push_notification_at).count();
    return age_ms >= 0 && age_ms < guard_ms;
}

}

// Protocol constants
constexpr size_t GENESIS_HASH_SIZE = 32;  // Tritium genesis hash size
constexpr size_t ADDRESS_DISPLAY_TRUNCATE = 40;  // Max characters to display for addresses in logs
constexpr size_t MIN_GENESIS_LOG_SIZE = 8;  // Minimum genesis bytes to log (for sanity check)
constexpr size_t MAX_GENESIS_LOG_BYTES = 32;  // Maximum genesis bytes to log (avoid excessive output)

// Push notification payload offsets (12-byte format, big-endian)
constexpr size_t PUSH_NOTIFICATION_UNIFIED_HEIGHT_OFFSET = 0;   // Unified blockchain height (4 bytes)
constexpr size_t PUSH_NOTIFICATION_CHANNEL_HEIGHT_OFFSET = 4;   // Channel-specific height (4 bytes)
constexpr size_t PUSH_NOTIFICATION_DIFFICULTY_OFFSET = 8;       // Mining difficulty (4 bytes)
constexpr size_t PUSH_NOTIFICATION_PAYLOAD_SIZE = 12;           // Total payload size

// ChaCha20 key derivation domain separator
static const std::string KDF_DOMAIN = "nexus-mining-chacha20-v1";

// ChaCha20 AAD for Falcon public key encryption (as vector for efficiency)
static const std::vector<uint8_t> AAD_DOMAIN_VEC{'F','A','L','C','O','N','_','P','U','B','K','E','Y'};

/* AAD (Additional Authenticated Data) constants for ChaCha20-Poly1305 AEAD
 * These MUST match the node's expectations for domain separation.
 * See: LLL-TAO/src/LLP/stateless_miner.cpp line 48-50 */

/** AAD for encrypting MINER_SET_REWARD payload (reward address)
 *  Node expects: "REWARD_ADDRESS" (14 bytes) */
static const std::vector<uint8_t> AAD_REWARD_ADDRESS{
    'R','E','W','A','R','D','_',
    'A','D','D','R','E','S','S'
};

/** AAD for decrypting MINER_REWARD_RESULT response
 *  Node uses: "REWARD_RESULT" (13 bytes) */
static const std::vector<uint8_t> AAD_REWARD_RESULT{
    'R','E','W','A','R','D','_',
    'R','E','S','U','L','T'
};

// Helper function to parse uint32 from big-endian bytes
static uint32_t read_uint32_be(const std::vector<uint8_t>& src, size_t offset = 0) {
    if (src.size() < offset + 4) {
        return 0;
    }
    return (static_cast<uint32_t>(src[offset]) << 24) |
           (static_cast<uint32_t>(src[offset + 1]) << 16) |
           (static_cast<uint32_t>(src[offset + 2]) << 8) |
           static_cast<uint32_t>(src[offset + 3]);
}

// Helper function to serialize uint16 to little-endian bytes  
static void append_uint16_le(std::vector<uint8_t>& dest, uint16_t value) {
    dest.push_back(value & 0xFF);
    dest.push_back((value >> 8) & 0xFF);
}

// Helper function to get channel name string for logging
static std::string get_channel_name(uint32_t channel) {
    switch(channel) {
        case mining::CHANNEL_PRIME: return "Prime";
        case mining::CHANNEL_HASH:  return "Hash";
        default:                     return "Unknown";
    }
}

Solo::Solo(std::uint8_t channel, std::shared_ptr<stats::Collector> stats_collector,
           std::shared_ptr<NodeSessionContext> session_context,
           std::shared_ptr<asio::io_context> io_context)
: m_channel{channel}
, m_logger{spdlog::get("logger")}
, m_current_height{0}
, m_current_reward{0}
, m_set_block_handler{}
, m_stats_collector{std::move(stats_collector)}
, m_authenticated{false}
, m_session_id{0}
, m_address{"127.0.0.1"}  // Default address, can be overridden
, m_auth_timestamp{0}
, m_auth_state{AuthState::NOT_AUTHENTICATED}
, m_miner_id{"NexusMiner"}  // Default miner ID
, m_falcon_wrapper{nullptr}
, m_disposable_falcon_enabled{true}  // ALWAYS ON - Disposable Falcon (core protocol, accepts both F-512/F-1024)
, m_chacha20_wrapper{nullptr}  // Lazy initialization when needed
, m_enable_chacha20{true}  // ALWAYS ON - Core implementation (localhost miners, SessionID protection, etc.)
, m_session_context{std::move(session_context)}
, m_template_interface{nullptr}
, m_connection{nullptr}
, m_reward_address{""}  // Empty until configured
, m_reward_bound{false}  // Not bound until successful MINER_REWARD_RESULT
, m_last_round_status{false, 0, 0, 0, 0, 0, false}  // Initialize GET_ROUND status (with difficulty and channel heights)
, m_last_get_round_time{std::chrono::steady_clock::now()}  // Initialize to now
, m_current_poll_interval_ms{POLL_INTERVAL_MIN_MS}  // Start at minimum interval (configured)
, m_needs_initial_round_check{false}  // No template yet
, m_template_unified_height{0}  // No template yet
, m_protocol_lane{ProtocolLane::UNKNOWN}  // Will be determined from connection port
, m_dedup_guard{spdlog::get("logger")}
, m_io_context{std::move(io_context)}
{
    if (!m_logger) {
        m_logger = spdlog::default_logger();
    }
    // Initialise the 5s recovery-debounce timer when an io_context is available.
    // Without one (test environments that only pass 3 args), the timer is null and
    // on_get_round_response falls back to the legacy immediate-GET_BLOCK behaviour.
    if (m_io_context) {
        m_recovery_timer = std::make_unique<asio::steady_timer>(*m_io_context);
        m_get_block_control_timer = std::make_unique<asio::steady_timer>(*m_io_context);
    }
   // Log constructor call with requested channel value
    m_logger->info("Solo::Solo: ctor called, channel={}", static_cast<int>(m_channel));

    // Clamp channel to valid LLL-TAO channels: 1 = prime, 2 = hash
    if (m_channel != 1 && m_channel != 2) {
        m_logger->warn("Invalid channel {} specified. Valid channels: 1 (prime), 2 (hash). Defaulting to 2 (hash).",
            static_cast<int>(m_channel));
        m_channel = 2;
    }

    // Initialize unified push notification handler
    m_push_handler = std::make_unique<PushNotificationHandler>(m_logger, m_channel);

    // Initialize Colin AI Diagnostic PING/PONG handler with the shared logger
    m_colin_ping_handler.set_logger(m_logger);
    m_colin_ping_handler.set_channel(m_channel);

    // Note: ChaCha20 wrapper is lazily initialized when enable_chacha20_wrapping() is called
    // This avoids unnecessary resource allocation when ChaCha20 is not needed

    // Session manager is now provided via NodeSessionContext (shared across primary/secondary protocols)
    if (!get_session_manager()) {
        m_logger->error("[Solo] Session manager not available from NodeSessionContext - session management disabled");
    } else {
        m_logger->info("[Solo] Using shared session manager from NodeSessionContext");
    }

    // Initialize client-side channel managers (mirrors NODE's PR #136)
    m_prime_manager = std::make_unique<mining::PrimeClientManager>();
    m_hash_manager = std::make_unique<mining::HashClientManager>();
    m_logger->info("[Solo] Client channel managers initialized (Prime + Hash)");
    
    // Initialize the Mining Template Interface for unified READ/FEED operations
    // Session ID starts at 0 (unauthenticated) and will be updated after MINER_AUTH_RESULT
    // The session ID binds the template interface to the FALCON authenticated tunnel
    m_template_interface = std::make_unique<MiningTemplateInterface>(m_channel, SessionId{});
    m_logger->info("[Solo] Mining Template Interface initialized for unified READ/FEED system");
    
    // Wire centralized HeightTracker into MiningTemplateInterface (non-owning pointer)
    m_template_interface->set_height_tracker(&m_height_tracker);
    if (m_session_context) {
        m_session_epoch = m_session_context->get_session_epoch();
        m_has_seen_session_epoch = true;
        m_cached_runtime_state_generation = m_session_context->get_runtime_snapshot().runtime_state_generation;
    }
    // Keep MTI/HeightTracker aligned with the authoritative session epoch even
    // before authentication. With no session context the epoch remains 0,
    // which is the explicit "no active session ownership" baseline.
    m_template_interface->set_session_epoch(m_session_epoch);
    m_height_tracker.set_session_epoch(m_session_epoch);
    m_logger->info("[Solo] HeightTracker wired into MiningTemplateInterface");
    
    // Setup template feed handler - called automatically when templates are validated
    m_template_interface->set_template_feed_handler(
        [this](const MiningTemplateInterface::MiningTemplate& tmpl, uint32_t nBits) -> bool {
            // Log new template (infrequent: once per block, typically every few minutes)
            m_logger->info("[Solo] ═══════════════════════════════════════");
            m_logger->info("[Solo] 🆕 NEW MINING TEMPLATE RECEIVED");
            m_logger->info("[Solo]   Channel:         {} ({})", tmpl.block.nChannel, 
                get_channel_name(tmpl.block.nChannel));
            
            // block.nHeight is the UNIFIED blockchain height (tStateBest.nHeight + 1).
            // This is a TARGET (the block we are trying to mine). Displayed as-is.
            m_logger->info("[Solo]   Template height: {} (unified target — block being mined)", tmpl.block.nHeight);
            
            // Show best-known unified TIP from all sources (canonical, push, round).
            // During rapid block production, GET_ROUND may lag behind BLOCK_DATA by
            // Log unified tip from canonical BLOCK_DATA. Template height = tip + 1.
            auto feed_snap = m_height_tracker.GetSnapshot();
            m_logger->info("[Solo]   Unified tip:     {} (BLOCK_DATA canonical)", feed_snap.unified_height);
            if (m_last_round_status.height > 0 && m_last_round_status.height < feed_snap.unified_height) {
                m_logger->debug("[Solo]   GET_ROUND tip:   {} (lagging — polled data, not authoritative)",
                    m_last_round_status.height);
            }
            
            if (tmpl.nChannelHeight > 0) {
                m_logger->info("[Solo]   Channel height:  {} (channel target from BLOCK_DATA metadata)", tmpl.nChannelHeight);
            } else {
                // nChannelHeight == 0: the node did not provide channel height in BLOCK_DATA
                // metadata, or this is a genesis/startup edge case.
                if (feed_snap.channel_height > 0) {
                    m_logger->info("[Solo]   Channel height:  ~{} (BLOCK_DATA canonical, pending node confirmation)",
                        feed_snap.channel_height);
                } else {
                    m_logger->info("[Solo]   Channel height:  (pending — awaiting first BLOCK_DATA)");
                }
            }
            m_logger->info("[Solo]   Difficulty:      0x{:08x}", nBits);
            m_logger->info("[Solo] ═══════════════════════════════════════");
            
            // Feed to worker threads via set_block_handler
            // Note: Debounce is now handled in MiningTemplateInterface::feed_current_template()
            // This handler is only invoked if the template passes the unified debounce gate
            if (m_set_block_handler) {
                m_logger->info("[Solo] Distributing template to worker threads...");
                const bool accepted = m_set_block_handler(tmpl.block, nBits);
                if (accepted) {
                    m_logger->info("[Solo] ✓ Template distributed - workers should start mining");
                } else {
                    m_logger->error("[Solo] Template feed rejected by worker layer");
                }
                return accepted;
            } else {
                m_logger->error("[Solo] CRITICAL: No block handler registered!");
                return false;
            }
        }
    );
    m_logger->info("[Solo] Template feed handler registered - ready to distribute work to workers");

    // Register template-cleared callback: zero prevblock_suffix in SESSION_KEEPALIVE
    // so the node sees a clean slate when a template is discarded.
    m_template_interface->set_template_cleared_callback(
        [this]() {
            if (get_session_manager()) {
                get_session_manager()->set_prevblock_suffix({0, 0, 0, 0});
                m_logger->debug("[Solo] Template cleared — prevblock_suffix zeroed in session keepalive");
            }
        }
    );

    // Register all packet handlers with the table-driven router
    register_packet_handlers();
}

void Solo::register_packet_handlers()
{
    using Pkt = Packet;

    // Auth-related opcodes all route to the same handler (on_miner_auth_response
    // discriminates internally based on the opcode).
    auto auth_handler = [this](Pkt const& p, std::shared_ptr<network::Connection> c) {
        on_miner_auth_response(p, c);
    };
    m_packet_router.register_handler(Pkt::MINER_AUTH_CHALLENGE, auth_handler);
    m_packet_router.register_handler(Pkt::MINER_AUTH_RESULT,    auth_handler);
    m_packet_router.register_handler(Pkt::CHANNEL_ACK,          auth_handler);
    m_packet_router.register_handler(Pkt::SESSION_START,        auth_handler);
    m_packet_router.register_handler(Pkt::SESSION_KEEPALIVE,    auth_handler);
    m_packet_router.register_handler(Pkt::MINER_REWARD_RESULT,  auth_handler);

    // BLOCK_DATA → on_block_data
    m_packet_router.register_handler(Pkt::BLOCK_DATA, [this](Pkt const& p, std::shared_ptr<network::Connection> c) {
        on_block_data(p, c);
    });

    // ACCEPT / GOOD_BLOCK → on_block_accepted
    auto accept_handler = [this](Pkt const& p, std::shared_ptr<network::Connection> c) {
        on_block_accepted(p, c);
    };
    m_packet_router.register_handler(Pkt::ACCEPT, accept_handler);
    m_packet_router.register_handler(LLP::GOOD_BLOCK, accept_handler);

    // REJECT / ORPHAN_BLOCK → on_block_rejected
    auto reject_handler = [this](Pkt const& p, std::shared_ptr<network::Connection> c) {
        on_block_rejected(p, c);
    };
    m_packet_router.register_handler(Pkt::REJECT, reject_handler);
    m_packet_router.register_handler(LLP::ORPHAN_BLOCK, reject_handler);

    // NEW_ROUND / OLD_ROUND → on_get_round_response
    auto round_handler = [this](Pkt const& p, std::shared_ptr<network::Connection> c) {
        on_get_round_response(p, c);
    };
    m_packet_router.register_handler(Pkt::NEW_ROUND, round_handler);
    m_packet_router.register_handler(Pkt::OLD_ROUND, round_handler);

    // SESSION_EXPIRED → on_session_expired
    m_packet_router.register_handler(Pkt::SESSION_EXPIRED, [this](Pkt const& p, std::shared_ptr<network::Connection> c) {
        on_session_expired(p, c);
    });

    // Push notifications → on_push_notification with channel
    m_packet_router.register_handler(Pkt::PRIME_BLOCK_AVAILABLE, [this](Pkt const& p, std::shared_ptr<network::Connection> c) {
        on_push_notification(p, c, mining::CHANNEL_PRIME);
    });
    m_packet_router.register_handler(Pkt::HASH_BLOCK_AVAILABLE, [this](Pkt const& p, std::shared_ptr<network::Connection> c) {
        on_push_notification(p, c, mining::CHANNEL_HASH);
    });

    // GET_BLOCK template response. Stateless uses 0xD081; legacy-compatible nodes
    // may answer the 0x81 request with the same 228-byte template payload.
    m_packet_router.register_handler(Pkt::GET_BLOCK, [this](Pkt const& p, std::shared_ptr<network::Connection> c) {
        if (p.m_is_uint16_opcode || p.m_length > 0) {
            on_get_block_template(p, c);
        }
    });

    // Colin AI Diagnostic PING (stateless-only diagnostic opcode)
    m_packet_router.register_handler(::LLP::ColinDiagOpcodes::PING_DIAG_LEGACY, [this](Pkt const& p, std::shared_ptr<network::Connection> c) {
        on_ping_diag(p, c);
    });

    // SESSION_STATUS_ACK (raw handlers — these opcodes have no legacy mirror)
    auto status_ack_handler = [this](Pkt const& p, std::shared_ptr<network::Connection> c) {
        on_session_status_ack(p, c);
    };
    m_packet_router.register_raw_handler(
        static_cast<uint16_t>(::LLP::SessionStatusOpcodes::SESSION_STATUS_ACK), status_ack_handler);
    m_packet_router.register_raw_handler(
        static_cast<uint16_t>(::LLP::SessionStatusOpcodes::SESSION_STATUS_ACK_LEGACY), status_ack_handler);

    // NODE_SHUTDOWN → handled inline in process_messages (it reads frame data and
    // fires the m_node_shutdown_handler callback, which is tightly coupled to the
    // pre-dispatch preamble).  Registered here for completeness so the router
    // recognizes it and does not log "invalid header".
    m_packet_router.register_handler(Pkt::NODE_SHUTDOWN, [this](Pkt const& p, std::shared_ptr<network::Connection> /*c*/) {
        ::LLP::NodeShutdownFrame frame;
        static const std::vector<uint8_t> empty_vec;
        const auto& payload = p.m_data ? *p.m_data : empty_vec;
        if (frame.Parse(payload)) {
            m_logger->warn("[Solo] ════════════════════════════════════════════════");
            m_logger->warn("[Solo] Node sent graceful shutdown notice (reason={}) — stopping workers",
                frame.ReasonString());
            m_logger->warn("[Solo] Reconnect backoff: {}s", NODE_SHUTDOWN_BACKOFF_S);
            m_logger->warn("[Solo] ════════════════════════════════════════════════");
        } else {
            m_logger->warn("[Solo] ════════════════════════════════════════════════");
            m_logger->warn("[Solo] Node sent graceful shutdown notice (reason=UNKNOWN, no payload) — stopping workers");
            m_logger->warn("[Solo] Reconnect backoff: {}s", NODE_SHUTDOWN_BACKOFF_S);
            m_logger->warn("[Solo] ════════════════════════════════════════════════");
        }
        if (m_node_shutdown_handler)
            m_node_shutdown_handler(frame.reason);
    });

    // BLOCK_REWARD → inline (small, self-contained)
    m_packet_router.register_handler(Pkt::BLOCK_REWARD, [this](Pkt const& p, std::shared_ptr<network::Connection> /*c*/) {
        if (!p.m_data || p.m_length < 8) {
            m_logger->warn("Solo::process_messages: BLOCK_REWARD packet has invalid data or length < 8");
            return;
        }
        m_current_reward = bytes2uint64(*p.m_data);
        m_logger->info("[Solo] Received BLOCK_REWARD: reward={}", m_current_reward);
    });

    // BLOCK_HEIGHT (opcode 0x02 / 0xD002) — with compat disambiguation.
    // 0xD002 with length 0 is BLOCK_ACCEPTED_COMPAT (routed to on_block_accepted).
    // 0xD003 with length ≤ 1 is BLOCK_REJECTED_COMPAT — but 0xD003 unmirrors to
    // SET_CHANNEL (0x03), so it doesn't collide with BLOCK_HEIGHT (0x02).
    // We only need to check for BLOCK_ACCEPTED_COMPAT here.
    m_packet_router.register_handler(Pkt::BLOCK_HEIGHT, [this](Pkt const& p, std::shared_ptr<network::Connection> c) {
        // Check for BLOCK_ACCEPTED_COMPAT (0xD002 with zero-length payload)
        if (p.m_is_uint16_opcode &&
            p.m_header == LLP::StatelessMining::BLOCK_ACCEPTED_COMPAT &&
            p.m_length == 0) {
            on_block_accepted(p, c);
            return;
        }

        // Normal BLOCK_HEIGHT processing
        if (!p.m_data || p.m_length < 4) {
            m_logger->warn("Solo::process_messages: BLOCK_HEIGHT packet has invalid data or length < 4");
            return;
        }

        auto const height = bytes2uint(*p.m_data);
        m_logger->info("[Solo] Received BLOCK_HEIGHT: height={}", height);

        auto snap = m_height_tracker.GetSnapshot();
        uint32_t known_height = snap.unified_height > 0 ? snap.unified_height : m_current_height;

        if (height > known_height) {
            m_logger->info("Nexus Network: New height {} (old height: {})", height, known_height);
            m_current_height = height;  // diagnostic only

            m_logger->info("[Solo] Height updated, requesting work via GET_BLOCK");
            if (!request_and_queue_get_block(c, GetBlockReason::INITIAL_REQUEST, "[Solo] GET_BLOCK")) {
                m_logger->warn("[Solo] GET_BLOCK rate-limited or unavailable — will wait for next node push");
            }
        } else if (height == known_height) {
            m_logger->debug("[Solo] Height unchanged ({}), no action needed", height);
        } else {
            m_logger->warn("[Solo] Received older height {} (current: {})", height, known_height);
        }
    });

    // SET_CHANNEL (opcode 0x03 / 0xD003) — with BLOCK_REJECTED_COMPAT disambiguation.
    // 0xD003 with length ≤ 1 is BLOCK_REJECTED_COMPAT (routed to on_block_rejected).
    // Normal SET_CHANNEL packets have different lengths and are not currently handled,
    // but this registration ensures the compat case is dispatched correctly.
    m_packet_router.register_handler(Pkt::SET_CHANNEL, [this](Pkt const& p, std::shared_ptr<network::Connection> c) {
        if (p.m_is_uint16_opcode &&
            p.m_header == LLP::StatelessMining::BLOCK_REJECTED_COMPAT &&
            p.m_length <= 1) {
            on_block_rejected(p, c);
            return;
        }
        // Normal SET_CHANNEL — currently no handler (node-initiated channel acknowledgement)
        m_logger->debug("[Solo] Received SET_CHANNEL packet — no action");
    });
}
std::vector<uint8_t> Solo::derive_chacha20_session_key(const std::vector<uint8_t>& genesis)
{
    // IMPORTANT: Use genesis bytes exactly as parsed/configured; do not reverse them
    // before the KDF step. This must match the node's hashGenesis.GetBytes() ordering.
    std::vector<uint8_t> preimage;
    preimage.insert(preimage.end(), KDF_DOMAIN.begin(), KDF_DOMAIN.end());
    preimage.insert(preimage.end(), genesis.begin(), genesis.end());
    
    // Use OpenSSL SHA256 - output is always SHA256_DIGEST_LENGTH (32) bytes
    std::vector<uint8_t> key(SHA256_DIGEST_LENGTH);
    unsigned char* result = SHA256(preimage.data(), preimage.size(), key.data());
    if (!result) {
        // This should never happen - SHA256 only fails on internal OpenSSL errors
        m_logger->error("[Solo] CRITICAL: OpenSSL SHA256 internal error during key derivation");
        throw std::runtime_error("OpenSSL SHA256 internal error");
    }
    
    // DIAGNOSTIC: Log key derivation details for comparison with node logs
    m_logger->info("╔═══════════════════════════════════════════════════════════╗");
    m_logger->info("║  ChaCha20 KEY DERIVATION DIAGNOSTIC (Miner Side)          ║");
    m_logger->info("╠═══════════════════════════════════════════════════════════╣");
    m_logger->info("║ Domain: {}", KDF_DOMAIN);
    m_logger->info("║ Genesis size: {} bytes", genesis.size());
    
    // Log genesis bytes for comparison (sanity check: only if we have a reasonable amount)
    if (genesis.size() >= MIN_GENESIS_LOG_SIZE) {
        // Use existing keys::to_hex with truncated vector to limit log output
        size_t log_length = std::min(genesis.size(), MAX_GENESIS_LOG_BYTES);
        std::vector<uint8_t> genesis_truncated(genesis.begin(), genesis.begin() + log_length);
        m_logger->info("║ Genesis (hex, configured order / no reversal): {}", nexusminer::keys::to_hex(genesis_truncated));
    }
    
    // Log derived key for comparison with node's "Derived Key (32 bytes):" log
    m_logger->info("║ Derived Key (hex): {}", nexusminer::keys::to_hex(key));
    m_logger->info("╚═══════════════════════════════════════════════════════════╝");
    
    return key;
}

std::vector<uint8_t> Solo::derive_or_get_cached_chacha20_session_key(const std::vector<uint8_t>& genesis)
{
    if (!genesis.empty() &&
        !m_cached_chacha20_key.empty() &&
        !m_cached_chacha20_key_genesis.empty() &&
        m_cached_chacha20_key_genesis == genesis) {
        m_logger->debug("[Solo Auth] Reusing cached ChaCha20 session key for unchanged genesis");
        return m_cached_chacha20_key;
    }

    auto key = derive_chacha20_session_key(genesis);
    m_cached_chacha20_key_genesis = genesis;
    m_cached_chacha20_key = key;
    return key;
}

std::vector<uint8_t> Solo::load_tritium_genesis()
{
    // Try to get genesis from session manager first
    if (get_session_manager() && !get_session_manager()->get_tritium_genesis().empty()) 
    {
        m_logger->info("[Solo Auth] Using genesis from session manager");
        return get_session_manager()->get_tritium_genesis();
    }
    
    // If session manager doesn't have it, reload from persistent storage (handles reconnection)
    if (!m_persistent_tritium_genesis.empty())
    {
        auto genesis = m_persistent_tritium_genesis;
        
        // Restore to session manager for future use
        if (get_session_manager())
        {
            get_session_manager()->set_tritium_genesis(genesis);
        }
        
        if (genesis.size() >= 4)
            m_logger->info("[Solo Auth] Reloaded tritium_genesis from persistent storage ({} bytes), first 4 bytes: {:02x}{:02x}{:02x}{:02x}",
                genesis.size(), genesis[0], genesis[1], genesis[2], genesis[3]);
        else
            m_logger->info("[Solo Auth] Reloaded tritium_genesis from persistent storage ({} bytes)", genesis.size());
        return genesis;
    }
    
    // No genesis configured
    m_logger->warn("[Solo Auth] No tritium_genesis configured - using zero genesis");
    m_logger->warn("[Solo Auth] ChaCha20 encryption unavailable without valid genesis");
    return std::vector<uint8_t>(GENESIS_HASH_SIZE, 0);  // 32 zero bytes
}

void Solo::reset()
{
    cancel_get_block_control_retry();
    m_current_height = 0;
    m_current_reward = 0;
    m_authenticated = false;
    m_session_id.clear();
    m_auth_timestamp = 0;
    m_auth_state = AuthState::NOT_AUTHENTICATED;
    m_auth_in_flight_since = {};
    m_reward_bound = false;  // Reset reward binding for new session
    m_reward_genesis_mismatch = false;  // Re-evaluated on next SESSION_START
    m_subscribed_to_notifications = false;  // Reset push notification subscription
    m_pending_push_after_auth = false;

    // Reset HashCheckpoint Guard state for new session
    m_hash_checkpoint_guard.reset();

    // Reset MerkleRoot Feed Guard for new session
    m_merkle_root_feed_guard.reset();

    // Note: m_chacha20_wrapper is intentionally NOT cleared here — the wrapper object
    // is stateless (no per-session state) and can be reused across reconnects.

    // Reset session manager
    if (m_session_context) {
        m_session_context->set_chacha20_session_key({}, "", false);
        m_session_context->clear_for_disconnect(m_reward_address,
                                                m_reward_address.empty() ? "" : "config",
                                                "solo session reset");
    }

    // Reset template interface for new session
    if (m_template_interface) {
        propagate_session_to_template_interface("Solo Reset");
        m_logger->info("[Solo] MiningTemplateInterface session ID cleared for session reset");
        m_template_interface->reset_stats();
        m_template_interface->clear_template_channel_height_snapshot();
    }
}

bool Solo::session_context_is_authenticated() const
{
    return m_session_context && m_session_context->is_authenticated();
}

void Solo::propagate_session_to_template_interface(const char* log_scope)
{
    if (!m_template_interface) {
        return;
    }

    SessionBinding binding;
    if (m_session_context) {
        binding = m_session_context->get_session_binding();
    } else {
        binding.session_id = get_session_id();
        binding.session_epoch = m_session_epoch;
        binding.active_lane = m_protocol_lane;
        binding.identity = m_cached_identity;
    }

    m_template_interface->set_session_binding(binding);

    m_logger->debug("[{}] Propagated session binding to MiningTemplateInterface: session_id=0x{:08x}, epoch={}",
                    log_scope, binding.session_id.get(), binding.session_epoch.get());
}

void Solo::resync_auth_from_session_context(const char* log_scope)
{
    if (m_authenticated || !session_context_is_authenticated()) {
        return;
    }

    m_authenticated = true;
    m_auth_state = AuthState::AUTHENTICATED;
    m_auth_in_flight_since = {};
    m_logger->warn("[{}] Resynced m_authenticated from session_context — local flag was stale",
                   log_scope);

    refresh_cached_session_state(log_scope);
    if (!m_session_id.is_default()) {
        propagate_session_to_template_interface(log_scope);
    }
}

void Solo::refresh_cached_session_state(const char* log_scope)
{
    if (!m_session_context) {
        return;
    }

    if (auto* session_manager = get_session_manager()) {
        const auto authoritative_generation = session_manager->peek_runtime_state_generation();
        if (authoritative_generation != 0 &&
            authoritative_generation == m_cached_runtime_state_generation) {
            return;
        }
    }

    const auto session = m_session_context->get_runtime_snapshot();
    const auto binding = m_session_context->get_session_binding();
    m_cached_runtime_state_generation = session.runtime_state_generation;

    if (!m_has_seen_session_epoch || m_session_epoch != binding.session_epoch) {
        if (!m_has_seen_session_epoch) {
            m_logger->info("[{}] Resyncing local session epoch from authoritative session container: local={} authoritative={}",
                           log_scope, m_session_epoch.get(), binding.session_epoch.get());
        } else {
            m_logger->warn("[{}] Session epoch advanced: local={} authoritative={} — invalidating generation-bound cached state",
                           log_scope, m_session_epoch.get(), binding.session_epoch.get());
            clear_generation_bound_state("authoritative session epoch advanced");
        }

        m_session_epoch = binding.session_epoch;
        m_has_seen_session_epoch = true;
        m_height_tracker.set_session_epoch(m_session_epoch);
    }

    if (m_authenticated != binding.authenticated) {
        m_authenticated = binding.authenticated;
    }

    m_session_id = binding.session_id;
    m_cached_identity = binding.identity;

    if (binding.authenticated && !binding.session_id.is_default()) {
        propagate_session_to_template_interface(log_scope);
    }

    m_reward_bound = binding.reward_bound;

    if (m_protocol_lane == ProtocolLane::UNKNOWN &&
        binding.active_lane != ProtocolLane::UNKNOWN) {
        m_protocol_lane = binding.active_lane;
    }
}

SessionOwnershipStamp Solo::capture_session_ownership() const
{
    if (!m_session_context) {
        // No authoritative session context means there is no correlatable owner.
        // Callers treat the zero-initialized stamp as "ownership unavailable".
        return {};
    }

    const auto binding = m_session_context->get_session_binding();
    return { binding.session_id, binding.session_epoch };
}

SubmitContext Solo::capture_submit_context(uint32_t template_height,
                                           uint32_t chain_height) const
{
    SubmitContext context;
    context.template_height = template_height;
    context.chain_height = chain_height;

    if (!m_session_context) {
        return context;
    }

    const auto binding = m_session_context->get_session_binding();
    context.session_id = binding.session_id;
    context.session_epoch = binding.session_epoch;
    context.identity = binding.identity;
    return context;
}

void Solo::record_session_event(SessionManager::SessionEventKind kind,
                                const std::string& detail) const
{
    if (m_session_context) {
        m_session_context->record_session_event(kind, detail);
    }
}

void Solo::finalize_keepalive_ack(const char* detail)
{
    if (m_session_context) {
        m_session_context->note_keepalive_ack(true, detail ? detail : "keepalive ack accepted");
    }
    if (auto* session_manager = get_session_manager()) {
        session_manager->record_keepalive();
    }
}

void Solo::clear_generation_bound_state(const char* reason)
{
    cancel_get_block_control_retry();
    m_last_keepalive_request_owner.clear();
    m_last_session_status_request_owner.clear();
    m_last_reward_request_owner.clear();
    m_last_get_block_request_owner.clear();
    clear_pending_submit_result_state();
    m_session_id_mismatch_count = 0;
    m_preflight_reject_count = 0;
    m_unanswered_get_round_count.store(0, std::memory_order_release);
    m_earliest_unanswered_get_round_at = {};
    m_last_get_round_transmitted_at = {};
    m_last_session_status_ack = {};
    m_last_session_status_ack_time = {};
    m_last_known_hash_prev_block = uint1024_t(0);
    m_last_keepalive_prevhash_lo32 = 0;
    m_get_round_push_silent_fallback_active = false;

    // Bug 1 fix: Clear in-flight GET_BLOCK flag so recovery is not blocked
    // for up to TIMEOUT_SECONDS (4s) after session invalidation.
    m_pending_get_block.clear();

    if (m_template_interface) {
        m_template_interface->discard_template(reason);
        m_template_interface->clear_template_channel_height_snapshot();
    }
}

bool Solo::finalize_and_feed_current_template(uint32_t unified_height,
                                              uint32_t effective_channel_height,
                                              const char* log_scope,
                                              bool snapshot_round_channel_height)
{
    if (!m_template_interface) {
        return false;
    }

    if (effective_channel_height > 0) {
        m_template_interface->set_channel_height(effective_channel_height + 1);
    }

    auto const* tmpl = m_template_interface->get_current_template();
    if (!tmpl) {
        m_logger->error("[{}] No valid template available after finalization", log_scope);
        return false;
    }

    // Update the canonical tip anchor BEFORE validate_current_template() so that
    // the freshly decoded BLOCK_DATA's hashPrevBlock becomes the canonical value
    // against which validate checks.  This ordering is critical: if placed AFTER
    // validate, the canonical anchor would still hold the previous tip's hash,
    // and the advisory mismatch logging in validate_current_template() would
    // fire on every legitimate chain-tip advance.  With the softened policy
    // (1B: always-accept), this wouldn't cause a doom loop, but it would
    // produce spurious advisory warnings on every normal tip change.
    m_height_tracker.UpdateWithHashPrevBlock(tmpl->block.hashPrevBlock);

    // *** 1C: Record HashCheckpoint unconditionally on every BLOCK_DATA ***
    // The checkpoint guard tracks reality during reorgs. Unlike the canonical
    // anchor (which is a single value), checkpoints form an immutable rolling
    // window that cannot be reorged away. This enables shallow-vs-deep reorg
    // classification in validate_current_template() and Colin diagnostics.
    m_hash_checkpoint_guard.record_checkpoint(tmpl->block.hashPrevBlock);

    // Clear the push tip anchor now that a fresh BLOCK_DATA template is in hand.
    // This must happen BEFORE validate_current_template() so that any residual
    // push_hash_prev_block from a prior same-height push does not cause
    // validate_current_template() to log a spurious "push tip-anchor differs" note.
    // The anchor was already used in the push handler to trigger the soft refresh;
    // at this point BLOCK_DATA is authoritative and the anchor is stale.
    m_height_tracker.ClearPushTipAnchor();

    if (!validate_current_template()) {
        m_logger->warn("[{}] Template invalidated by final adoption gate before worker feed", log_scope);
        // Reset GET_BLOCK dedup state so the recovery request is not suppressed
        // by stale height values from the prior (now-rejected) template request.
        reset_get_block_dedup_state();
        return false;
    }

    // Template passed validation — reset the consecutive hashPrevBlock mismatch counter
    // so the chain-in-flux guard doesn't carry over stale state to the next validate cycle.
    m_hashprev_mismatch_consecutive.store(0, std::memory_order_relaxed);
    m_hash_checkpoint_guard.reset_consecutive();

    m_current_height = unified_height;

    auto format_hex8 = [](const uint1024_t& h) -> std::string {
        auto bytes = h.GetBytes();
        std::string s;
        for (size_t i = 0; i < std::min(bytes.size(), size_t(8)); ++i) {
            char buf[3];
            snprintf(buf, sizeof(buf), "%02x", bytes[i]);
            s += buf;
        }
        return s;
    };

    if (m_last_known_hash_prev_block != uint1024_t(0)) {
        if (tmpl->block.hashPrevBlock != m_last_known_hash_prev_block) {
            m_logger->warn("[TEMPLATE ANCHOR] ⚡ CHAIN TIP CHANGED: old={} new={}",
                format_hex8(m_last_known_hash_prev_block), format_hex8(tmpl->block.hashPrevBlock));
        } else {
            m_logger->info("[TEMPLATE ANCHOR] ✅ Chain tip unchanged (channel advanced, same tip)");
        }
    }

    m_last_known_hash_prev_block = tmpl->block.hashPrevBlock;

    if (get_session_manager()) {
        auto prev_bytes = m_last_known_hash_prev_block.GetBytes();
        std::array<uint8_t, 4> suffix{};
        if (prev_bytes.size() >= 4) {
            suffix = { prev_bytes[0], prev_bytes[1], prev_bytes[2], prev_bytes[3] };
        }
        get_session_manager()->set_prevblock_suffix(suffix);
        m_last_keepalive_prevhash_lo32 =
            (uint32_t(suffix[0]) << 24) | (uint32_t(suffix[1]) << 16)
          | (uint32_t(suffix[2]) <<  8) | uint32_t(suffix[3]);
    }

    m_logger->info("[TEMPLATE ANCHOR] hashPrevBlock = {}... (tip anchor at template creation)",
                   format_hex8(m_last_known_hash_prev_block));
    m_logger->info("[TEMPLATE ANCHOR] block.nHeight = {} (unified blockchain height)", tmpl->block.nHeight);

    // The snapshot is only valid/useful while nChannelHeight == 0 (pending finalization).
    // Once set_channel_height() has been called (effective_channel_height > 0), the snapshot
    // must remain cleared — any subsequent call to set_template_channel_height_snapshot()
    // re-poisons the guard and causes false staleness on the next GET_ROUND.
    if (snapshot_round_channel_height && effective_channel_height == 0 && m_last_round_status.has_channel_heights) {
        uint32_t snapshot_height = m_last_round_status.get_channel_height(m_channel);
        if (snapshot_height > 0) {
            m_template_interface->set_template_channel_height_snapshot(snapshot_height);
            m_logger->info("[Solo] 📸 Snapshot: {} at height {} (template is for height {})",
                get_channel_name(m_channel), snapshot_height, tmpl->block.nHeight);
        }
    }

    on_template_received(tmpl->block.nHeight);

    if (m_template_interface->needs_channel_height_finalization()) {
        m_logger->debug("[{}] Template pending channel height finalization", log_scope);
    }

    if (!m_set_block_handler) {
        m_logger->error("[{}] CRITICAL: No block handler set - cannot process template", log_scope);
        return false;
    }

    // *** Unified-height feed guard: suppress same-height re-feeds within cooldown ***
    // This is the ultimate backstop against the "Double Feed" race.  Even if
    // transmit-side and MerkleRoot guards both pass, this guard prevents workers
    // from being restarted with a duplicate template at the same chain-tip position.
    // Legitimate reorgs (hashPrevBlock change) bypass the guard so the miner
    // always switches to the correct chain tip.
    {
        // Skip on first-ever feed (no prior state to compare against).
        if (m_last_fed_time != std::chrono::steady_clock::time_point{}) {
            bool same_tip = (unified_height == m_last_fed_unified_height &&
                             tmpl->block.hashPrevBlock == m_last_fed_hash_prev_block);
            if (same_tip) {
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::steady_clock::now() - m_last_fed_time).count();
                if (elapsed < SAME_HEIGHT_FEED_COOLDOWN_SECONDS) {
                    ++m_same_tip_reconfirmation_count;
                    m_logger->info("[{}] Feed suppressed: same unified height {} within {}s cooldown "
                                   "(elapsed {}s, same hashPrevBlock, reconfirmation #{})",
                                   log_scope, unified_height, SAME_HEIGHT_FEED_COOLDOWN_SECONDS, elapsed,
                                   m_same_tip_reconfirmation_count);

                    // The node just proved the currently-loaded tip is still canonical —
                    // this is a legitimate proof-of-liveness even though no re-feed is
                    // needed. If a recovery cycle (e.g. WAITING_TEMPLATE after a
                    // submit-side hashPrevBlock mismatch) is waiting specifically for
                    // confirmation that the current template is still good, it would
                    // otherwise never learn about this response, because the normal
                    // feed-handler chokepoint (and its clear_recovery_state() call) is
                    // never reached when the guard above suppresses the re-feed.
                    if (m_recovery_confirmed_handler) {
                        m_recovery_confirmed_handler();
                    }

                    return true;  // receive succeeded, feed intentionally suppressed
                }
            }
        }
    }

    // *** MerkleRoot Feed Guard: suppress duplicate worker restarts ***
    // Tier 1: same (height, hashMerkleRoot) within 2s → suppressed.
    // Tier 2: same height, different merkle root within 500ms → suppressed
    //         (mempool-variant duplicates).
    // hashPrevBlock change (reorg) always passes through both tiers.
    if (!m_merkle_root_feed_guard.should_feed(tmpl->block.hashMerkleRoot,
                                               tmpl->block.nHeight,
                                               tmpl->block.hashPrevBlock)) {
        m_logger->info("[{}] Feed suppressed by MerkleRoot/height guard (suppressed count: {})",
                       log_scope, m_merkle_root_feed_guard.suppressed_count());
        return true;  // receive succeeded, feed intentionally suppressed
    }

    if (!m_template_interface->feed_current_template()) {
        m_logger->debug("[{}] Template feed not accepted; TemplateInterface logged whether this was "
                        "debounce suppression, missing handler, or worker-layer rejection",
                        log_scope);
        return false;
    } else {
        m_last_template_adopted_at = std::chrono::steady_clock::now();
        // Record successful feed for the unified-height guard.
        m_last_fed_unified_height = unified_height;
        m_last_fed_hash_prev_block = tmpl->block.hashPrevBlock;
        m_last_fed_time = std::chrono::steady_clock::now();
    }

    auto stats = m_template_interface->get_stats();
    if (stats.templates_received % 10 == 0) {
        m_logger->debug("[Solo Template Stats] Received: {}, Validated: {}, Rejected: {}, Fed: {}",
            stats.templates_received, stats.templates_validated,
            stats.templates_rejected, stats.templates_fed);
    }

    return true;
}

void Solo::mark_authoritative_recovery_required(const std::string& reason)
{
    if (m_session_context) {
        m_session_context->mark_recovery_required(reason);
    }
}

void Solo::mark_authoritative_recovery_healthy(const std::string& reason)
{
    if (m_session_context) {
        m_session_context->mark_recovery_healthy(reason);
    }
}

void Solo::set_epoch_coordinator(std::shared_ptr<EpochCoordinator> coordinator)
{
    m_epoch_coordinator = std::move(coordinator);
}

const Solo::PacketIngressPreflightOptions Solo::kDefaultPacketIngressPreflightOptions{};

bool Solo::run_packet_ingress_preflight(const char* log_scope,
                                        const PacketIngressPreflightOptions& options)
{
    if (!m_session_context) {
        return true;
    }

    std::string validation_reason;
    const bool session_valid = m_session_context->validate_miner_session(&validation_reason);
    const auto binding = m_session_context->get_session_binding();
    const SessionEpoch owner_epoch     = options.owner ? options.owner->session_epoch : SessionEpoch{};
    const SessionId owner_session_id = options.owner ? options.owner->session_id : SessionId{};
    const auto decision = PacketIngressPreflight::evaluate({
        true,
        binding.authenticated,
        binding.session_id,
        binding.session_epoch,
        binding.active_lane,
        m_protocol_lane,
        options.validate_lane,
        options.allow_without_active_session,
        options.packet_session_id,
        owner_epoch,
        owner_session_id
    });

    if (decision.allow_processing) {
        // Reset preflight reject counter on any successful pass
        m_preflight_reject_count = 0;
        return true;
    }

    // ── Shadow-ban detection ────────────────────────────────────────────
    // Track consecutive preflight rejections.  When the threshold is
    // exceeded, force re-auth to break out of the silent drop cycle.
    ++m_preflight_reject_count;

    m_logger->warn("[{}] Session ingress preflight rejected packet: {} (reject_count={})",
                   log_scope, decision.reason, m_preflight_reject_count);
    if (decision.drop_as_stale) {
        const auto kind = decision.stale_reason == PacketStaleReason::OWNERSHIP_EPOCH_MISMATCH
                        ? SessionManager::SessionEventKind::EPOCH_MISMATCH
                        : SessionManager::SessionEventKind::STALE_PACKET_DROPPED;
        record_session_event(kind, std::string(log_scope) + ": " + decision.reason);
    }
    if (!session_valid) {
        m_logger->warn("[{}] Authoritative session validation failed: {}", log_scope, validation_reason);
        m_logger->warn("[{}] {}", log_scope, m_session_context->build_miner_session_diagnostics());
    }

    // Clear pending GET_BLOCK so recovery is not blocked by a stale
    // in-flight marker when the response was preflight-rejected.
    if (m_pending_get_block.active) {
        m_logger->warn("[{}] Clearing stale pending GET_BLOCK after preflight rejection", log_scope);
        m_pending_get_block.clear();
    }

    // Force re-auth on threshold-triggered escalation or explicit force_reauth
    const bool threshold_exceeded = m_preflight_reject_count >= SHADOW_BAN_PREFLIGHT_THRESHOLD;
    if (threshold_exceeded && m_session_expired_handler) {
        m_logger->error("[{}] SHADOW BAN DETECTED: {} consecutive preflight rejections — forcing re-auth",
                        log_scope, m_preflight_reject_count);
        record_session_event(SessionManager::SessionEventKind::FORCED_REAUTH,
                             "shadow_ban_detected: " + std::to_string(m_preflight_reject_count) +
                             " consecutive preflight rejections");
        m_preflight_reject_count = 0;
        m_session_expired_handler();
    } else if (decision.force_reauth && options.trigger_reauth && m_session_expired_handler) {
        record_session_event(SessionManager::SessionEventKind::FORCED_REAUTH,
                             std::string(log_scope) + ": " + decision.reason);
        m_logger->warn("[{}] Triggering session-expired handler after preflight rejection", log_scope);
        m_session_expired_handler();
    }

    return false;
}

bool Solo::ensure_session_ready_for_ingress(const char* log_scope,
                                            const char* packet_name,
                                            bool queue_post_auth_get_block)
{
    // Derive ingress readiness from authoritative session state first.
    // The local m_authenticated flag is a cache that may lag behind the
    // authoritative SessionManager; the policy treats authoritative state as
    // the primary authority and local_auth_stale as a secondary indicator
    // to trigger a cache resync before accepting the packet.
    const bool authoritative_authenticated =
        m_session_context && m_session_context->is_authenticated();
    const auto decision = SessionRecoveryPolicy::evaluate_ingress_readiness({
        m_session_context != nullptr,                              // has_session_context
        authoritative_authenticated,                               // authoritative_authenticated
        false,                                                     // local_auth_stale (eliminated: single source of truth)
        m_auth_state == AuthState::NOT_AUTHENTICATED               // auth_not_in_flight
    });

    if (!decision.allow_ingress) {
        if (queue_post_auth_get_block && decision.queue_deferred_push) {
            queue_pending_push_after_auth(log_scope);
        }

        const std::string reason = std::string(packet_name) + " deferred: " + decision.reason;
        m_logger->warn("[{}] Session ingress deferred: {} (auth_state={})",
                       log_scope, reason, static_cast<int>(m_auth_state));

        // Always check whether an in-flight auth has timed out before deciding
        // whether to trigger recovery; this may advance m_auth_state.
        check_auth_in_flight_timeout(log_scope);
        if (decision.trigger_recovery && m_session_expired_handler) {
            record_session_event(SessionManager::SessionEventKind::FORCED_REAUTH,
                                 std::string(log_scope) + ": " + decision.reason);
            m_logger->warn("[{}] Triggering session-expired handler after ingress readiness failure",
                           log_scope);
            m_session_expired_handler();
        }
        return false;
    }

    if (decision.resync_local_cache) {
        m_logger->info("[{}] Session ingress resyncing stale local auth cache before processing {}",
                       log_scope, packet_name);
        resync_auth_from_session_context(log_scope);
        if (!is_authenticated()) {
            m_logger->warn("[{}] Session ingress deferred: failed to resync local auth cache for {}",
                           log_scope, packet_name);
            if (m_session_context) {
                m_logger->warn("[{}] {}", log_scope, m_session_context->build_miner_session_diagnostics());
            }
            return false;
        }
    }

    return true;
}

void Solo::queue_pending_push_after_auth(const char* log_scope)
{
    if (m_pending_push_after_auth) {
        return;
    }

    m_pending_push_after_auth = true;
    m_logger->info("[{}] Queued GET_BLOCK to run after authentication flow completes", log_scope);
}

void Solo::flush_pending_push_after_auth(const std::shared_ptr<network::Connection>& connection,
                                         const char* log_scope)
{
    if (!m_pending_push_after_auth) {
        return;
    }

    // NODE auto-sends BLOCK_DATA after PUSH — no GET_BLOCK request needed.
    // The push arrived during the auth handshake; the node will auto-send
    // fresh block data now that the miner is authenticated and ready.
    m_pending_push_after_auth = false;
    m_logger->info("[{}] Push arrived during auth handshake — node will auto-send block data (no GET_BLOCK needed)",
                   log_scope);
}

void Solo::update_connection_metadata(const std::shared_ptr<network::Connection>& connection)
{
    if (!m_session_context) {
        return;
    }

    if (!connection) {
        m_session_context->set_connection_metadata("", "", false);
        return;
    }

    const auto& remote_ep = connection->remote_endpoint();
    const auto& local_ep = connection->local_endpoint();
    m_session_context->set_connection_metadata(local_ep.to_string(), remote_ep.to_string(), true);
}

bool Solo::validate_authoritative_session(const char* log_scope, bool require_reward_binding) const
{
    if (!m_session_context) {
        return true;
    }

    std::string reason;
    if (!m_session_context->validate_miner_session(&reason)) {
        m_logger->error("[{}] Authoritative miner session container consistency failure: {}", log_scope, reason);
        m_logger->error("[{}] {}", log_scope, m_session_context->build_miner_session_diagnostics());
        return false;
    }

    const auto binding = m_session_context->get_session_binding();
    if (require_reward_binding && !binding.reward_address.empty() && !binding.reward_bound) {
        m_logger->error("[{}] Authoritative miner session container requires reward binding before continuing", log_scope);
        m_logger->error("[{}] {}", log_scope, m_session_context->build_miner_session_diagnostics());
        return false;
    }

    return true;
}

void Solo::log_session_container_summary(const char* log_scope) const
{
    if (!m_session_context) {
        return;
    }

    m_logger->info("[{}] {}", log_scope, m_session_context->build_miner_session_diagnostics());
}

network::Shared_payload Solo::login(Login_handler handler)
{
    // Clamp channel to valid values as safety net
    if (m_channel != 1 && m_channel != 2) {
        m_logger->warn("Solo::login: Invalid channel {}, clamping to 2 (hash)", static_cast<int>(m_channel));
        m_channel = 2;
    }
    
    // Falcon authentication is mandatory - no legacy fallback
    if (m_miner_pubkey.empty() || m_miner_privkey.empty()) {
        m_logger->error("[Solo Auth] CRITICAL: Falcon miner keys are required for authentication");
        m_logger->error("[Solo Auth] Legacy authentication mode has been removed");
        m_logger->error("[Solo Auth] Please configure Falcon keys in miner.conf:");
        m_logger->error("[Solo Auth]   1. Generate keys: ./NexusMiner --create-keys");
        m_logger->error("[Solo Auth]   2. Add falcon_miner_pubkey and falcon_miner_privkey to miner.conf");
        m_logger->error("[Solo Auth]   3. Whitelist your public key on the node:");
        m_logger->error("[Solo Auth]      - Config file: Add 'minerallowkey=<pubkey>' to nexus.conf");
        m_logger->error("[Solo Auth]      - Command line: Start nexus with -minerallowkey=<pubkey>");
        handler(false);
        return network::Shared_payload{};
    }
    
    if (!m_falcon_wrapper || !m_falcon_wrapper->is_valid()) {
        m_logger->error("[Solo Phase 2] Cannot authenticate - Falcon keys not loaded");
        handler(false);
        return network::Shared_payload{};
    }
    
    m_logger->info("[Solo Phase 2] Starting Falcon authentication (challenge-response)");
    m_logger->info("[Solo Auth] Using public key ({} bytes)", m_miner_pubkey.size());
    
    // Initialize authentication timestamp for this login attempt
    auto current_time = std::chrono::system_clock::now();
    m_auth_timestamp = static_cast<uint64_t>(
        std::chrono::system_clock::to_time_t(current_time));
    m_logger->info("[Solo Auth] Authentication timestamp set: {} (0x{:016x})", 
                   m_auth_timestamp, m_auth_timestamp);
    
    if (m_protocol_lane == ProtocolLane::UNKNOWN) {
        m_logger->warn("[Solo Auth] Protocol lane unknown - defaulting to legacy auth opcode");
    }
    
    // Build MINER_AUTH_INIT payload
    network::Payload auth_payload;
    
    // ═══════════════════════════════════════════════════════════
    // STEP 1: hashGenesis FIRST (32 bytes) - enables key derivation
    // ═══════════════════════════════════════════════════════════
    std::vector<uint8_t> tritium_genesis = load_tritium_genesis();
    if (m_session_context) {
        m_session_context->set_tritium_genesis(tritium_genesis);
        m_session_context->set_falcon_identity(m_miner_pubkey, falcon_pubkey_to_hash_key_id(m_miner_pubkey), false);
        m_session_context->set_channel_state(m_channel, false, false);
        m_session_context->mark_activity();
    }
    
    // Genesis goes FIRST in the packet
    auth_payload.insert(auth_payload.end(), tritium_genesis.begin(), tritium_genesis.end());
    
    // ═══════════════════════════════════════════════════════════
    // STEP 2: Prepare pubkey (optionally ChaCha20 wrapped)
    // ═══════════════════════════════════════════════════════════
    std::vector<uint8_t> pubkey_to_send = m_miner_pubkey;
    bool wrapped = false;

    // Only wrap if we have a valid genesis (non-zero)
    bool has_valid_genesis = genesis_utils::is_valid_genesis(tritium_genesis);

    if (m_enable_chacha20 && has_valid_genesis)
    {
        m_logger->info("[Solo Auth] ChaCha20 wrapping ENABLED (genesis-derived key)");
        
        try {
            // Derive session key from genesis
            auto session_key = derive_or_get_cached_chacha20_session_key(tritium_genesis);
            auto nonce = ChaCha20Wrapper::generate_nonce();  // Random 12 bytes

            // Log the nonce being used for encryption
            m_logger->info("[Solo Auth] ChaCha20 nonce (12 bytes): {}", nexusminer::keys::to_hex(nonce));

            if (!m_chacha20_wrapper)
                m_chacha20_wrapper = std::make_unique<ChaCha20Wrapper>();

            // Use AAD for domain separation
            auto wrap_result = m_chacha20_wrapper->encrypt(m_miner_pubkey, session_key, nonce, AAD_DOMAIN_VEC);

            if (wrap_result.success)
            {
                // Build wrapped format: nonce(12) + ciphertext+tag(897+16)
                pubkey_to_send.clear();
                pubkey_to_send.insert(pubkey_to_send.end(), nonce.begin(), nonce.end());
                pubkey_to_send.insert(pubkey_to_send.end(), wrap_result.data.begin(), wrap_result.data.end());

                wrapped = true;

                m_logger->info("[Solo Auth] ChaCha20 key fingerprint (first 8 bytes): {}",
                               format_hex_prefix(session_key, 8));
                if (m_session_context) {
                    m_session_context->set_chacha20_session_key(session_key,
                                                                format_hex_prefix(session_key, 8),
                                                                true);
                    m_logger->info("[Solo Auth] ✓ Session key stored in authoritative session container");
                } else {
                    m_logger->error("[Solo Auth] Unable to store session key: session context not initialized");
                }

                m_logger->info("[Solo Auth] ✓ Pubkey wrapped: {} → {} bytes (genesis-derived key)",
                               m_miner_pubkey.size(), pubkey_to_send.size());
            }
            else
            {
                m_logger->warn("[Solo Auth] ChaCha20 wrap failed: {}", wrap_result.error_message);
                m_logger->warn("[Solo Auth] Falling back to unwrapped pubkey");
            }
        }
        catch (const std::exception& e) {
            m_logger->error("[Solo Auth] Key derivation failed: {}", e.what());
            m_logger->warn("[Solo Auth] Falling back to unwrapped pubkey");
        }
    }
    else if (m_enable_chacha20 && !has_valid_genesis)
    {
        m_logger->warn("[Solo Auth] ChaCha20 requested but no genesis for key derivation");
        m_logger->warn("[Solo Auth] Sending unwrapped pubkey");
    }
    
    // ═══════════════════════════════════════════════════════════
    // STEP 3: pubkey_len + pubkey
    // ═══════════════════════════════════════════════════════════
    uint16_t pubkey_len = static_cast<uint16_t>(pubkey_to_send.size());
    auth_payload.push_back(static_cast<uint8_t>((pubkey_len >> 8) & 0xFF));
    auth_payload.push_back(static_cast<uint8_t>(pubkey_len & 0xFF));
    auth_payload.insert(auth_payload.end(), pubkey_to_send.begin(), pubkey_to_send.end());
    
    // ═══════════════════════════════════════════════════════════
    // STEP 4: miner_id_len + miner_id
    // ═══════════════════════════════════════════════════════════
    std::string miner_id = m_miner_id.empty() ? "NexusMiner" : m_miner_id;
    uint16_t miner_id_len = static_cast<uint16_t>(miner_id.size());
    auth_payload.push_back(static_cast<uint8_t>((miner_id_len >> 8) & 0xFF));
    auth_payload.push_back(static_cast<uint8_t>(miner_id_len & 0xFF));
    auth_payload.insert(auth_payload.end(), miner_id.begin(), miner_id.end());
    
    m_logger->debug("[Solo Auth] Packet built: length={}", auth_payload.size());
    
    auto bytes = PacketBuilder::build(m_protocol_lane, LLP::MINER_AUTH_INIT, auth_payload);
    if (!bytes || bytes->empty())
    {
        m_logger->error("[Solo Auth] CRITICAL: PacketBuilder::build returned null/empty!");
        m_logger->error("[Solo Auth] Cannot transmit - payload is null or empty");
        return network::Shared_payload{};
    }
    
    m_logger->debug("[Solo Auth] Packet validation: SUCCESS - encoded {} bytes", bytes->size());
    
    // TRAINING WHEELS: Show hex dump of MINER_AUTH_INIT packet
    m_logger->info("[Solo Auth] MINER_AUTH_INIT packet hex dump (wire format):");
    m_logger->info("\n{}", format_llp_payload_hexdump(bytes, 128));
    
    // ═══════════════════════════════════════════════════════════
    // Log summary
    // ═══════════════════════════════════════════════════════════
    m_logger->info("");
    m_logger->info("═══════════════════════════════════════════════════════════");
    m_logger->info("    MINER_AUTH_INIT (Genesis-First Protocol)");
    m_logger->info("═══════════════════════════════════════════════════════════");
    m_logger->info("  Genesis:     {} bytes ({})", tritium_genesis.size(),
                   has_valid_genesis ? "VALID - key derivation enabled" : "ZERO");
    m_logger->info("  Public Key:  {} bytes {}", pubkey_len,
                   wrapped ? "(ChaCha20 wrapped)" : "(unwrapped)");
    m_logger->info("  Miner ID:    '{}'", miner_id);
    m_logger->info("  Total Size:  {} bytes", auth_payload.size());
    m_logger->info("═══════════════════════════════════════════════════════════");
    m_logger->info("");
    
    // Set state to waiting for challenge
    m_auth_state = AuthState::WAITING_FOR_CHALLENGE;
    m_auth_in_flight_since = std::chrono::steady_clock::now();
    if (m_session_context) {
        m_session_context->begin_auth_handshake("falcon auth handshake started");
    }
    
    // Login handler will be called after successful authentication in MINER_AUTH_RESULT
    // For now, mark as "in progress"
    handler(true);
    
    return bytes;
}

network::Shared_payload Solo::get_work()
{
    return get_work(GetBlockReason::INITIAL_REQUEST);
}

bool Solo::is_get_block_retry_deferred() const
{
    const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    return now_ms < m_get_block_retry_at_ms.load(std::memory_order_acquire);
}

network::Shared_payload Solo::get_work(GetBlockReason reason)
{
    /// Request a fresh mining template via GET_BLOCK.
    /// Authentication-guarded; returns null if not authenticated or reward not bound.
    /// Node rate limit: 25 GET_BLOCK per 60 seconds (with 1-second per-request cooldown).

    refresh_cached_session_state("Solo GET_BLOCK");

    if (is_get_block_retry_deferred()) {
        m_last_get_block_request_status.store(GetBlockRequestStatus::NODE_RETRY_DELAY);
        return nullptr;
    }

    /* Validate prerequisites */
    if (!is_authenticated()) {
        m_last_get_block_request_status.store(GetBlockRequestStatus::UNAUTHENTICATED);
        m_logger->error("[Solo] Cannot request work - not authenticated");
        m_logger->error("[Solo]   Current auth state: {}",
            m_auth_state == AuthState::NOT_AUTHENTICATED ? "NOT_AUTHENTICATED" :
            m_auth_state == AuthState::WAITING_FOR_CHALLENGE ? "WAITING_FOR_CHALLENGE" :
            m_auth_state == AuthState::WAITING_FOR_RESULT ? "WAITING_FOR_RESULT" :
            "AUTHENTICATED");
        m_logger->error("[Solo]   Waiting for Falcon authentication to complete");
        return nullptr;
    }

    if (!validate_authoritative_session("Solo GET_BLOCK", !m_reward_address.empty())) {
        m_last_get_block_request_status.store(GetBlockRequestStatus::SESSION_INVALID);
        return nullptr;
    }

    if (m_session_context && !m_session_context->can_request_get_block()) {
        m_last_get_block_request_status.store(GetBlockRequestStatus::SESSION_INVALID);
        m_logger->info("[Solo] Cannot request work - authoritative session is not yet ready for GET_BLOCK");
        return nullptr;
    }

    // Only validate reward binding if a reward address was configured
    // (Reward binding is optional for localhost/testing, but required for production)
    if (!m_reward_address.empty() &&
        !m_reward_bound) {
        m_last_get_block_request_status.store(GetBlockRequestStatus::REWARD_NOT_BOUND);
        m_logger->error("[Solo] Cannot request work - reward address not bound");
        return nullptr;
    }

    // ── Pending GET_BLOCK in-flight guard ───────────────────────────────────
    // If a GET_BLOCK is already in-flight for the same-or-higher unified height,
    // suppress this request to prevent two concurrent responses triggering
    // duplicate worker feeds (the "Double Feed" race).  Recovery reasons that
    // bypass all dedup are exempt so degraded-mode retries always make progress.
    {
        auto snap = m_height_tracker.GetSnapshot();
        uint32_t canonical_unified = snap.canonical_unified_height;
        if (!should_bypass_all_dedup(reason) &&
            m_pending_get_block.is_pending_for(canonical_unified)) {
            m_last_get_block_request_status.store(GetBlockRequestStatus::DUPLICATE_WINDOW);
            m_logger->debug("[Solo] GET_BLOCK suppressed: already in-flight for height {} "
                           "(pending reason={}, new reason={})",
                           m_pending_get_block.unified_height,
                           reason_name(m_pending_get_block.reason),
                           reason_name(reason));
            return nullptr;
        }
    }

    // ── GET_BLOCK deduplication guard ────────────────────────────────────────
    // Bug #3 fix: use canonical_unified_height for the height key (not composite).
    // Push/GET_ROUND reasons already bypass height dedup via should_bypass_height_dedup(),
    // so the height key only matters for same-height BLOCK_DATA responses.
    // Using composite caused inconsistent dedup decisions when push/round raced ahead.
    {
        auto snap = m_height_tracker.GetSnapshot();
        uint32_t canonical_unified = snap.canonical_unified_height;
        bool have_valid_template = m_template_interface &&
                                   m_template_interface->has_valid_template();

        // Pass current hashPrevBlock so the guard can detect same-height reorgs.
        uint1024_t current_hash_prev{};
        if (have_valid_template && m_template_interface) {
            auto const* tmpl = m_template_interface->get_current_template();
            if (tmpl) {
                current_hash_prev = tmpl->block.hashPrevBlock;
            }
        }

        auto verdict = m_dedup_guard.check(reason, canonical_unified, have_valid_template, current_hash_prev);
        if (verdict != GetBlockDedupGuard::Verdict::ALLOW) {
            m_last_get_block_request_status.store(GetBlockRequestStatus::DUPLICATE_WINDOW);
            m_logger->warn("[Solo] GET_BLOCK suppressed by dedup guard: verdict={}, reason={}, "
                           "unified={}, have_template={}",
                           verdict == GetBlockDedupGuard::Verdict::SUPPRESS_RAPID_BURST
                               ? "RAPID_BURST" : "HEIGHT_MATCH",
                           reason_name(reason), canonical_unified, have_valid_template);
            return nullptr;
        }
    }

    m_logger->debug("[Solo] Requesting mining template via GET_BLOCK");
    m_logger->debug("[Solo]   Session ID: 0x{:08x}", get_session_id().get());
    m_logger->debug("[Solo]   Authenticated: {}", is_authenticated() ? "YES" : "NO");
    m_logger->debug("[Solo]   Reward bound: {}", is_reward_bound() ? "YES" : "NO");
    if (m_session_context) {
        m_session_context->mark_activity();
    }

    /* Build GET_BLOCK packet via PacketBuilder (header-only, no payload) */
    auto payload = PacketBuilder::build(m_protocol_lane, LLP::GET_BLOCK);

    if (payload && !payload->empty()) {
        m_logger->debug("[Solo] GET_BLOCK encoded payload size: {} bytes", payload->size());
        // TRAINING WHEELS: Show GET_BLOCK packet (should be just header byte)
        m_logger->debug("[Solo] GET_BLOCK packet hex dump:");
        m_logger->debug("\n{}", format_llp_payload_hexdump(payload, 16));
    } else {
        m_last_get_block_request_status.store(GetBlockRequestStatus::BUILD_EMPTY);
        m_logger->error("[Solo] GET_BLOCK PacketBuilder::build returned null or empty payload!");
    }

    return payload;
}

void Solo::reset_get_block_dedup_state()
{
    m_dedup_guard.reset();
}

void Solo::mark_get_block_pending(GetBlockReason reason)
{
    auto snap = m_height_tracker.GetSnapshot();
    uint1024_t hash_prev{};
    if (m_template_interface) {
        auto const* tmpl = m_template_interface->get_current_template();
        if (tmpl) {
            hash_prev = tmpl->block.hashPrevBlock;
        }
    }
    m_last_get_block_request_owner = capture_session_ownership();
    m_dedup_guard.record_transmission(snap.canonical_unified_height, hash_prev);
    m_last_get_block_request_status.store(GetBlockRequestStatus::SENT);
    m_pending_get_block.mark_pending(snap.unified_height, reason);
    m_logger->debug("[Solo] GET_BLOCK in-flight marked: unified={} reason={}",
                    snap.unified_height, reason_name(reason));
}

network::Shared_payload Solo::send_get_round()
{
    // GET_ROUND — pure informational/sanity probe.
    //
    // Sends GET_ROUND (legacy: opcode 0x85; stateless: mirror-mapped 0xD085) on
    // all lanes.  The node responds with NEW_ROUND or OLD_ROUND containing height
    // and difficulty info — it does NOT push a block template.
    //
    // To request a fresh mining template use send_recovery_work_request() instead.

    if (!is_authenticated()) {
        m_logger->warn("[Solo GET_ROUND] Cannot send GET_ROUND - not authenticated yet");
        m_logger->debug("[Solo GET_ROUND]   Current auth state: {}",
            m_auth_state == AuthState::NOT_AUTHENTICATED ? "NOT_AUTHENTICATED" :
            m_auth_state == AuthState::WAITING_FOR_CHALLENGE ? "WAITING_FOR_CHALLENGE" :
            m_auth_state == AuthState::WAITING_FOR_RESULT ? "WAITING_FOR_RESULT" :
            "AUTHENTICATED");
        return nullptr;
    }

    // Always send GET_ROUND on all lanes (legacy: 0x85, stateless: 0xD085).
    m_logger->debug("[Solo GET_ROUND] Sending GET_ROUND ({} {})",
        get_lane_name(m_protocol_lane), format_lane_opcode(m_protocol_lane, LLP::GET_ROUND));
    auto payload = PacketBuilder::build(m_protocol_lane, LLP::GET_ROUND);
    if (payload && !payload->empty()) {
        m_logger->debug("[Solo GET_ROUND] Encoded payload size: {} bytes (header-only)", payload->size());
    } else {
        m_logger->error("[Solo GET_ROUND] PacketBuilder::build returned null or empty payload!");
    }
    return payload;
}

void Solo::note_get_round_transmitted()
{
    auto new_val = m_unanswered_get_round_count.fetch_add(1, std::memory_order_relaxed) + 1;
    m_last_get_round_transmitted_at = std::chrono::steady_clock::now();
    if (m_earliest_unanswered_get_round_at == std::chrono::steady_clock::time_point{}) {
        m_earliest_unanswered_get_round_at = m_last_get_round_transmitted_at;
    }
    m_logger->info("[Solo GET_ROUND] \u2192 Sent (unanswered={}, lane={})",
                   new_val, get_lane_name(m_protocol_lane));
}

network::Shared_payload Solo::send_recovery_work_request()
{
    // Recovery work request — requests a fresh mining template via GET_BLOCK.
    //
    // Sends GET_BLOCK on all lanes (legacy: 0x81; stateless: mirror-mapped 0xD081).
    // Delegates to get_work() which enforces the miner-side 1s rate limiter and
    // the authentication / reward-binding guards.  Callers must check for a null
    // or empty return value (rate-limited or not yet authenticated).
    //
    // Use this method — not send_get_round() — whenever the goal is to force a
    // template refresh (e.g. Timer_manager recovery, Worker_manager emergency).

    m_logger->debug("[Solo Recovery] Requesting fresh template via GET_BLOCK ({} {})",
        get_lane_name(m_protocol_lane), format_lane_opcode(m_protocol_lane, LLP::GET_BLOCK));
    return get_work(GetBlockReason::RECOVERY_FORCED);
}

network::Shared_payload Solo::submit_block(std::vector<std::uint8_t> const& block_data, std::uint64_t nonce)
{
    refresh_cached_session_state("Solo Submit");

    if (block_data.empty()) {
        m_logger->error("[Solo Submit] CRITICAL: block_data is empty! Cannot submit block.");
        note_block_discarded("empty block_data");
        return network::Shared_payload{};
    }

    if (!validate_authoritative_session("Solo Submit", !m_reward_address.empty())) {
        note_block_discarded("authoritative session validation failed");
        return network::Shared_payload{};
    }

    // Refuse to submit when the node-confirmed session genesis (echoed in
    // SESSION_START) did not match the configured reward_address. Submitting
    // anyway would always be rejected by the node's Coinbase::Verify with no
    // reason byte. See SESSION_START handler where this flag is set.
    if (m_reward_genesis_mismatch) {
        m_logger->error("[Solo Submit] Refusing to submit block: reward_address does not match"
                        " the node's session genesis. Fix mining.reward_address to match the"
                        " genesis hash returned by 'system/get/info' on the node and restart.");
        note_block_discarded("reward_address/session genesis mismatch");
        return network::Shared_payload{};
    }

    // ── Delegate to StatelessBlockUtility::encode_submit() ──────────────────
    // encode_submit() handles all pre-checks (nonce, channel, height, staleness),
    // worker-snapshot serialization, Disposable Falcon signing, and PacketBuilder
    // framing (0xD001 vs 0x01). Solo::submit_block() adds ChaCha20 encryption on
    // top of the signed payload.
    if (!m_template_interface || !m_template_interface->has_valid_template()) {
        m_logger->error("[Solo Submit] No valid template — cannot submit block");
        note_block_discarded("no valid template");
        return network::Shared_payload{};
    }

    const auto* tmpl = m_template_interface->get_current_template();
    if (!tmpl) {
        m_logger->error("[Solo Submit] get_current_template() returned null");
        note_block_discarded("get_current_template() returned null");
        return network::Shared_payload{};
    }

    if (block_data.size() < StatelessBlockUtility::BLOCK_BODY_SIZE) {
        m_logger->error("[Solo Submit] Block payload too small: {} bytes (need at least {} for block body)",
                        block_data.size(), StatelessBlockUtility::BLOCK_BODY_SIZE);
        note_block_discarded("block payload smaller than BLOCK_BODY_SIZE");
        return network::Shared_payload{};
    }

    // worker_manager passes block_data as:
    //   [216-byte solved block body][optional Prime vOffsets tail]
    // Solo re-decodes only the fixed 216-byte block body here so the authoritative
    // header fields (height / prevhash / bits / nonce) come from the worker-owned
    // snapshot, while the variable-length Prime tail is forwarded unchanged below.
    network::Payload block_body(block_data.begin(),
                                block_data.begin() + StatelessBlockUtility::BLOCK_BODY_SIZE);

    ::LLP::CBlock block_to_submit;
    try {
        block_to_submit = nexusminer::llp_utils::deserialize_block_header(block_body);
    } catch (const std::exception& e) {
        m_logger->error("[Solo Submit] Failed to decode solved block body: {}", e.what());
        note_block_discarded(std::string("failed to decode solved block body: ") + e.what());
        return network::Shared_payload{};
    }

    if (block_to_submit.nNonce != nonce) {
        m_logger->error("[Solo Submit] Nonce mismatch between serialized block (0x{:016x}) and callback argument (0x{:016x})",
                        block_to_submit.nNonce, nonce);
        note_block_discarded("nonce mismatch between serialized block and callback argument");
        return network::Shared_payload{};
    }

    const auto submit_snapshot = m_height_tracker.GetSnapshot();
    const auto submit_context = capture_submit_context(block_to_submit.nHeight, submit_snapshot.unified_height);
    const auto tracker_channel_tip = submit_snapshot.channel_tip_height.get();
    const auto template_channel_target = tmpl->nChannelHeight;
    const auto template_age_seconds = m_template_interface ? m_template_interface->get_template_age() : 0u;
    const auto prev_hash_bytes = block_to_submit.hashPrevBlock.GetBytes();
    std::array<uint8_t, 4> prevblock_suffix{};
    const auto prevblock_suffix_size = static_cast<std::ptrdiff_t>(prevblock_suffix.size());
    if (prev_hash_bytes.size() >= prevblock_suffix.size()) {
        std::copy(prev_hash_bytes.end() - prevblock_suffix_size,
                  prev_hash_bytes.end(),
                  prevblock_suffix.begin());
    }

    m_logger->info("[Solo Submit][Authoritative]");
    m_logger->info("[Solo Submit][Authoritative]   template_height      = {}", submit_context.template_height);
    m_logger->info("[Solo Submit][Authoritative]   central_tracker_height = {}", submit_context.chain_height);
    m_logger->info("[Solo Submit][Authoritative]   submit_height        = {}", block_to_submit.nHeight);
    m_logger->info("[Solo Submit][Authoritative]   prevblock_hash       = {}", format_hex_prefix(prev_hash_bytes, 8));
    m_logger->info("[Solo Submit][Authoritative]   prevblock_suffix     = {}", format_hex_prefix(prevblock_suffix, 4));
    m_logger->info("[Solo Submit][Authoritative]   session_epoch        = {}", submit_context.session_epoch.get());
    m_logger->info("[Solo Submit][Authoritative]   template_age         = {}s", template_age_seconds);
    if (const auto* session_manager = get_session_manager()) {
        const auto current_epoch = session_manager->get_session_epoch();
        if (current_epoch != submit_context.session_epoch) {
            // Bug 8 fix: Reject submit with stale epoch — NODE may silently drop
            // the block if session credentials don't match the current epoch.
            const std::string detail =
                "snap_epoch=" + std::to_string(submit_context.session_epoch.get()) +
                " current_epoch=" + std::to_string(current_epoch.get()) +
                " height=" + std::to_string(block_to_submit.nHeight);
            m_logger->error("[Solo Submit] Session epoch mismatch — rejecting stale submit: {}", detail);
            record_session_event(SessionManager::SessionEventKind::SUBMIT_REJECTED, detail);
            note_block_discarded("session epoch mismatch: " + detail);
            return network::Shared_payload{};
        }
    }

    if (!submit_context.matches_submit_height(block_to_submit.nHeight)) {
        const std::string detail =
            "template_height=" + std::to_string(submit_context.template_height) +
            " submit_height=" + std::to_string(block_to_submit.nHeight) +
            " chain_height=" + std::to_string(submit_context.chain_height) +
            " session_epoch=" + std::to_string(submit_context.session_epoch.get());
        m_logger->error("[Solo Submit] Authoritative submit-height validation failed: {}", detail);
        record_session_event(SessionManager::SessionEventKind::SUBMIT_REJECTED, detail);
        note_block_discarded("submit-height validation failed: " + detail);
        return network::Shared_payload{};
    }

    if (!tmpl->height_guard.matches(block_to_submit)) {
        // The solved snapshot is authoritative here: worker_manager already serialized
        // block_data from the exact Block_data snapshot the worker proved. A later
        // template refresh can legitimately advance the live template between "found"
        // and "submit" without invalidating the solved bytes, so keep logging the
        // drift but do not rewrite or discard the worker-owned submission payload.
        const std::string detail =
            "submit_height=" + std::to_string(block_to_submit.nHeight) +
            " current_template_height=" + std::to_string(tmpl->height_guard.unified_height.get()) +
            " channel_height=" + std::to_string(tracker_channel_tip) +
            " channel_target=" + std::to_string(template_channel_target) +
            " channel_height_marker=" +
            std::string(is_channel_height(submit_snapshot.channel_tip_height) ? "true" : "false");
        m_logger->warn("[Solo Submit] Height guard drift detected — keeping solved snapshot authoritative: {}", detail);
    }

    // Snapshot submitted block state for the ACCEPT/GOOD_BLOCK handler
    // so it doesn't need to re-read from a potentially-replaced template.
    // This metadata must mirror the solved snapshot, not the live template.
    store_last_submitted_state(capture_session_ownership(),
                               block_to_submit.nNonce,
                               block_to_submit.hashPrevBlock,
                               block_to_submit.nHeight,
                               block_to_submit.nChannel);
    m_submit_result_gate.mark_pending();

    // Extract Prime channel vOffsets from block_data (bytes after the fixed 216-byte body).
    // For Hash channel block_data is exactly 216 bytes so this stays empty.
    // The key rule is "split, do not rebuild": header fields come from block_body above,
    // variable-length Prime bytes come from this tail, and both originated from the same
    // worker snapshot in worker_manager.
    std::vector<uint8_t> vOffsets;
    if (block_data.size() > StatelessBlockUtility::BLOCK_BODY_SIZE)
        vOffsets.assign(block_data.begin() + StatelessBlockUtility::BLOCK_BODY_SIZE, block_data.end());

    auto submit_result = StatelessBlockUtility::encode_submit(
        *m_template_interface, block_to_submit, vOffsets,
        m_falcon_wrapper.get(), m_protocol_lane,
        submit_snapshot, m_logger, submit_context);

    if (!submit_result.valid) {
        m_logger->error("[Solo Submit] encode_submit() rejected block: {}",
                        submit_result.rejection_reason);
        record_session_event(SessionManager::SessionEventKind::SUBMIT_REJECTED,
                             submit_result.rejection_reason);
        note_block_discarded("encode_submit() rejected block: " + submit_result.rejection_reason);
        clear_pending_submit_result_state();
        return network::Shared_payload{};
    }

    // ── Extract plaintext payload from PacketBuilder-framed wire_bytes ────────
    // STATELESS wire format: [opcode(2 BE)][length(4 BE)][plaintext_payload]
    // LEGACY wire format:    [opcode(1)   ][length(4 BE)][plaintext_payload]
    // The ChaCha20 submit payload must exclude BOTH the opcode and the LLP length
    // prefix.  Encrypting the 4-byte length field breaks the node-side submit
    // parser and causes lane-specific framing drift.
    const auto& framed = *submit_result.wire_bytes;
    auto plaintextPayload = strip_submit_wire_header(framed, m_protocol_lane);
    if (plaintextPayload.empty()) {
        const size_t header_size = (m_protocol_lane == ProtocolLane::STATELESS) ? 6u : 5u;
        m_logger->error("[Solo Submit] Wire frame too small: {} bytes (header+length={})",
                        framed.size(), header_size);
        note_block_discarded("wire frame too small after stripping submit header");
        clear_pending_submit_result_state();
        return network::Shared_payload{};
    }
    record_session_event(SessionManager::SessionEventKind::SUBMIT_SENT,
                         "unified_height=" + std::to_string(block_to_submit.nHeight) +
                         " channel_height=" + std::to_string(tracker_channel_tip) +
                         " channel_target=" + std::to_string(template_channel_target) +
                         " channel=" + std::to_string(block_to_submit.nChannel) +
                         " channel_height_marker=" +
                         std::string(is_channel_height(submit_snapshot.channel_tip_height) ? "true" : "false"));

    // ── Channel-aware payload diagnostics ────────────────────────────────────
    // Compute payload metadata from live data — offset_bytes_count is derived
    // from real block_data, not defaulted to 0.
    const size_t live_offset_bytes = vOffsets.size();
    // Signature size: in the signed path, plaintext includes
    // block + offsets + timestamp(8) + sig_len(2) + signature.
    // If unsigned (no Falcon), signature_size = 0 and plaintext = block + offsets only.
    const size_t block_plus_offsets = StatelessBlockUtility::BLOCK_BODY_SIZE + live_offset_bytes;
    // Minimum signed overhead = timestamp(8) + sig_len_field(2) = 10 bytes
    constexpr size_t MIN_SIGNED_OVERHEAD = 8 + 2;
    const size_t sig_size = (plaintextPayload.size() > block_plus_offsets + MIN_SIGNED_OVERHEAD)
        ? (plaintextPayload.size() - block_plus_offsets - 8 - 2)  // signed: subtract ts + sig_len
        : 0;  // unsigned or too small for signature fields

    auto payload_info = StatelessBlockUtility::compute_submit_payload_info(
        tmpl->block.nChannel, block_plus_offsets, sig_size);

    m_logger->info("[Solo Submit] Channel-aware payload diagnostics:");
    m_logger->info("[Solo Submit]   channel            = {} ({})",
                   payload_info.channel,
                   payload_info.channel == 1 ? "Prime" : "Hash");
    m_logger->info("[Solo Submit]   base_block_size    = {} bytes", payload_info.base_block_size);
    m_logger->info("[Solo Submit]   offset_bytes_count = {} bytes", payload_info.offset_bytes_count);
    m_logger->info("[Solo Submit]   timestamp_size     = {} bytes", payload_info.timestamp_size);
    m_logger->info("[Solo Submit]   sig_len_field_size = {} bytes", payload_info.sig_len_field_size);
    m_logger->info("[Solo Submit]   signature_size     = {} bytes", payload_info.signature_size);
    m_logger->info("[Solo Submit]   plaintext (actual) = {} bytes", plaintextPayload.size());
    m_logger->info("[Solo Submit]   plaintext (expect) = {} bytes", payload_info.expected_plaintext_size());
    m_logger->info("[Solo Submit]   encrypted (expect) = {} bytes", payload_info.expected_encrypted_size());

    // ── ChaCha20-Poly1305 encryption ─────────────────────────────────────────
    if (!m_enable_chacha20) {
        m_logger->error("[Solo Submit] ChaCha20 not enabled");
        note_block_discarded("ChaCha20 not enabled");
        clear_pending_submit_result_state();
        return network::Shared_payload{};
    }

    const auto binding = m_session_context ? m_session_context->get_session_binding()
                                           : SessionBinding{};
    const auto& submit_session_key = binding.chacha20_session_key;

    // Use the authoritative session key from the session container.
    if (!binding.has_crypto_context()) {
        m_logger->critical("[Solo Submit] CRITICAL: authoritative session.chacha20_session_key is not ready");
        note_block_discarded("chacha20 session key not ready");
        clear_pending_submit_result_state();
        return network::Shared_payload{};
    }

    try {
        if (!m_chacha20_wrapper)
            m_chacha20_wrapper = std::make_unique<ChaCha20Wrapper>();

        // Canonical wrapper path: nonce generation, size validation, and
        // [nonce(12)][ciphertext][tag(16)] assembly are all internal.
        auto enc_result = m_chacha20_wrapper->encrypt_submit_block_payload(
            plaintextPayload, submit_session_key, payload_info);

        if (!enc_result.success || enc_result.data.empty()) {
            m_logger->error("[Solo Submit] ChaCha20 encryption failed: {}",
                            enc_result.error_message);
            note_block_discarded("ChaCha20 encryption failed: " + enc_result.error_message);
            clear_pending_submit_result_state();
            return network::Shared_payload{};
        }

        // enc_result.data is already [nonce(12)][ciphertext(plaintext.size())][tag(16)]
        const auto& encryptedPayload = enc_result.data;

        m_logger->info("[Solo Submit] Encrypted payload: {} bytes "
                       "(expected {} from payload_info) → SUBMIT_BLOCK",
                       encryptedPayload.size(),
                       payload_info.expected_encrypted_size());

        auto result = PacketBuilder::build(m_protocol_lane, LLP::SUBMIT_BLOCK, encryptedPayload);
        if (!result || result->empty()) {
            m_logger->error("[Solo Submit] PacketBuilder::build() returned empty packet");
            note_block_discarded("PacketBuilder::build() returned empty packet");
            clear_pending_submit_result_state();
            return network::Shared_payload{};
        }

        m_logger->info("[Solo Submit] SUBMIT_BLOCK ({}) wire format: {} bytes",
            format_lane_opcode(m_protocol_lane, LLP::SUBMIT_BLOCK),
            result->size());
        if (m_session_context) {
            m_session_context->set_channel_state(m_channel, true, true);
            m_session_context->mark_activity();
        }
        return result;
    }
    catch (const std::exception& e) {
        m_logger->error("[Solo Submit] Exception during encryption: {}", e.what());
        note_block_discarded(std::string("exception during encryption: ") + e.what());
        clear_pending_submit_result_state();
        return network::Shared_payload{};
    }
}

void Solo::process_messages(Packet packet, std::shared_ptr<network::Connection> connection)  
{
    // Store connection for multi-packet authentication flow
    if (connection) {
        set_connection(connection);
        update_connection_metadata(connection);
        
        // Initialize protocol lane from connection port (once, on first message)
        if (m_protocol_lane == ProtocolLane::UNKNOWN) {
            initialize_protocol_lane(connection);
        }
    }

    refresh_cached_session_state("Solo ProcessMessages");

    // ═══════════════════════════════════════════════════════════════════════
    // STRICT LANE VALIDATION (NO FALLBACK)
    // ═══════════════════════════════════════════════════════════════════════
    // Validate that received packet matches expected lane framing
    if (m_protocol_lane != ProtocolLane::UNKNOWN) {
        bool expected_uint16 = (m_protocol_lane == ProtocolLane::STATELESS);
        bool received_uint16 = packet.m_is_uint16_opcode;
        
        if (expected_uint16 != received_uint16) {
            // LANE MISMATCH: Server speaking wrong protocol on this port
            auto const& remote_ep = connection->remote_endpoint();
            m_logger->error("═══════════════════════════════════════════════════════════");
            m_logger->error("PROTOCOL LANE MISMATCH - DISCONNECTING");
            m_logger->error("═══════════════════════════════════════════════════════════");
            m_logger->error("Remote:          {}", remote_ep.to_string());
            m_logger->error("Remote Port:     {}", remote_ep.port());
            m_logger->error("Expected Lane:   {} ({}-bit header)", 
                get_lane_name(m_protocol_lane), expected_uint16 ? 16 : 8);
            m_logger->error("Received Header: {} (0x{:04x}) - {}-bit format",
                get_llp_header_name(packet.m_header), packet.m_header, 
                received_uint16 ? 16 : 8);
            m_logger->error("═══════════════════════════════════════════════════════════");
            m_logger->error("Server is speaking the WRONG protocol on this port!");
            m_logger->error("- Port {} should use {} lane", remote_ep.port(), 
                get_lane_name(m_protocol_lane));
            m_logger->error("- NO FALLBACK AVAILABLE (strict port-lane separation)");
            m_logger->error("═══════════════════════════════════════════════════════════");
            
            // Close connection and abort processing
            if (connection) {
                connection->close();
            }
            return;
        }
    }
    
    // Protocol lane is strictly determined by port at connection time (no negotiation/timeout)
    
    // Reject invalid packets at the start
    if (!packet.m_is_valid) {
        m_logger->warn("Solo::process_messages: Received invalid packet - header=0x{:04X} ({}), length={}", 
            packet.m_header, get_llp_header_name(packet.m_header), packet.m_length);
        return;
    }
    
    /* Guard: Un-mirrored stateless opcodes must never arrive on legacy lane */
    if(m_protocol_lane == ProtocolLane::LEGACY &&
       ::LLP::IsUnmirroredDataOpcode(static_cast<uint16_t>(packet.m_header)))
    {
        m_logger->error("[Colin] REJECTED un-mirrored stateless opcode 0x{:04x} ({}) on legacy lane"
                        " — stateless port required",
            packet.m_header, ::LLP::GetUnmirroredOpcodeName(static_cast<uint16_t>(packet.m_header)));
        if(connection) connection->close();
        return;
    }

    if (packet.m_length == 8 && matches_opcode(packet, Packet::REJECT) &&
        !m_pending_get_block.active) {
        m_logger->debug("[Solo GET_BLOCK] Ignoring stale control response: no request in flight");
        return;
    }

    // Template-lifeline packets (push notifications and template deliveries) must
    // stay open even when older session-debugging logic would otherwise defer or
    // stale-drop ingress. Dedicated handlers apply their own template validity
    // checks; authoritative-session preflight remains reserved for packets whose
    // semantics truly depend on the currently active session.
    if (requires_active_session_packet(packet)) {
        PacketIngressPreflightOptions preflight;
        preflight.validate_lane = true;
        preflight.trigger_reauth = true;
        if (!run_packet_ingress_preflight("Solo ProcessMessages", preflight)) {
            return;
        }
    }
    
    // Log received packet for diagnostics with port information
    if (connection) {
        auto const& remote_ep = connection->remote_endpoint();
        auto const& local_ep = connection->local_endpoint();
        m_logger->info("[Solo] ══════════════════════════════════════════════");
        // Use appropriate format based on opcode type
        if (packet.m_is_uint16_opcode) {
            m_logger->info("[Solo] RECEIVED PACKET: {} (0x{:04x})", 
                get_llp_header_name(packet.m_header), packet.m_header);
        } else {
            m_logger->info("[Solo] RECEIVED PACKET: {} (0x{:02x})", 
                get_llp_header_name(static_cast<uint8_t>(packet.m_header)), 
                static_cast<uint8_t>(packet.m_header));
        }
        m_logger->info("[Solo]   Length: {} bytes", packet.m_length);
        m_logger->info("[Solo]   Remote: {} | Local: {}", 
            remote_ep.to_string(), local_ep.to_string());
        
        // TRAINING WHEELS: Show hex dump of received packet payload (first 128 bytes)
        if (packet.m_data && !packet.m_data->empty()) {
            m_logger->info("[Solo] Payload hex dump:");
            m_logger->info("\n{}", format_llp_payload_hexdump(packet.m_data, 128));
        }
        m_logger->info("[Solo] ══════════════════════════════════════════════");
    } else {
        m_logger->info("[Solo] ══════════════════════════════════════════════");
        if (packet.m_is_uint16_opcode) {
            m_logger->info("[Solo] RECEIVED PACKET: {} (0x{:04x}), length={}", 
                get_llp_header_name(packet.m_header), packet.m_header,
                packet.m_length);
        } else {
            m_logger->info("[Solo] RECEIVED PACKET: {} (0x{:02x}), length={}", 
                get_llp_header_name(static_cast<uint8_t>(packet.m_header)), 
                static_cast<uint8_t>(packet.m_header), packet.m_length);
        }
        
        // TRAINING WHEELS: Show hex dump even without connection
        if (packet.m_data && !packet.m_data->empty()) {
            m_logger->info("[Solo] Payload hex dump:");
            m_logger->info("\n{}", format_llp_payload_hexdump(packet.m_data, 128));
        }
        m_logger->info("[Solo] ══════════════════════════════════════════════");
    }
    
    // ═══════════════════════════════════════════════════════════════════════
    // TABLE-DRIVEN DISPATCH (via PacketRouter)
    // ═══════════════════════════════════════════════════════════════════════
    // All handler registrations live in register_packet_handlers() (called
    // once from the constructor).  The router canonicalizes uint16_t opcodes
    // to their legacy mirror before lookup, and handles raw (unmirror-able)
    // opcodes via a separate table.
    if (!m_packet_router.dispatch(packet, connection)) {
        m_logger->debug("Invalid header received: 0x{:04x}", packet.m_header);
    } 
}

bool Solo::matches_opcode(Packet const& packet, uint16_t legacy_opcode)
{
    if (packet.m_is_uint16_opcode) {
        return packet.m_header == LLP::MirrorOpcode(static_cast<uint8_t>(legacy_opcode));
    }
    return packet.m_header == legacy_opcode;
}

bool Solo::matches_stateless_opcode(Packet const& packet, uint16_t legacy_opcode)
{
    return packet.m_is_uint16_opcode &&
        packet.m_header == LLP::MirrorOpcode(static_cast<uint8_t>(legacy_opcode));
}

bool Solo::requires_active_session_packet(Packet const& packet)
{
    return matches_opcode(packet, Packet::ACCEPT) ||
           matches_opcode(packet, LLP::GOOD_BLOCK) ||
           matches_opcode(packet, Packet::REJECT) ||
           matches_opcode(packet, LLP::ORPHAN_BLOCK) ||
           matches_opcode(packet, Packet::NEW_ROUND) ||
           matches_opcode(packet, Packet::OLD_ROUND) ||
           matches_opcode(packet, Packet::MINER_REWARD_RESULT) ||
           packet.m_header == static_cast<uint32_t>(::LLP::SessionStatusOpcodes::SESSION_STATUS_ACK) ||
           packet.m_header == static_cast<uint32_t>(::LLP::SessionStatusOpcodes::SESSION_STATUS_ACK_LEGACY);
}

bool Solo::activate_push_lane_after_channel_ack(std::shared_ptr<network::Connection> connection)
{
    // Unified push-subscription path: both protocol lanes share the same mining
    // behavior after CHANNEL_ACK.  PacketBuilder is the only difference: legacy
    // emits 8-bit opcodes, stateless emits 16-bit mirror-mapped opcodes.
    const char* lane_name = get_lane_name(m_protocol_lane);
    const auto opcode_str = format_lane_opcode(m_protocol_lane, LLP::MINER_READY);

    m_logger->info("[Solo Protocol] ═══════════════════════════════════════");
    m_logger->info("[Solo Protocol] {} LANE: Using push protocol", lane_name);
    m_logger->info("[Solo Protocol] ═══════════════════════════════════════");
    m_logger->info("[Solo Protocol] Sending MINER_READY ({})", opcode_str);

    auto miner_ready_payload = send_miner_ready();
    if (!miner_ready_payload || miner_ready_payload->empty()) {
        m_logger->error("[Solo Protocol] Failed to encode MINER_READY on {} lane", lane_name);
        if (connection) {
            connection->close();
        }
        return false;
    }

    if (!connection) {
        m_logger->error("[Solo Protocol] No connection available");
        return false;
    }

    if (!queue_payload(connection, miner_ready_payload, "[Solo Push] MINER_READY")) {
        m_logger->error("[Solo Protocol] MINER_READY payload was not queued on {} lane", lane_name);
        return false;
    }

    m_logger->info("[Solo Protocol] ✓ MINER_READY ({}) transmitted on {} lane", opcode_str, lane_name);
    if (m_session_context) {
        m_session_context->set_channel_state(m_channel, false, true);
        m_session_context->mark_activity();
    }
    validate_authoritative_session("Solo ChannelAck", false);
    log_session_container_summary("Solo ChannelAck");
    flush_pending_push_after_auth(connection, "Solo Protocol");
    if (m_work_ready_handler &&
        (!m_session_context || m_session_context->can_request_get_block())) {
        m_work_ready_handler();
    }
    return true;
}

void Solo::on_block_data(Packet const& packet, std::shared_ptr<network::Connection> connection)
{
        // ─────────────────────────────────────────────────────────────────────
        // Phase A2: validate payload BEFORE clearing the in-flight guard.
        // If a node sends back-to-back malformed BLOCK_DATA frames, clearing
        // the guard up-front would let each one trigger a fresh recovery
        // request before the previous one has even been accounted for.
        // ─────────────────────────────────────────────────────────────────────
        if (!packet.m_data) {
            m_logger->error("[Solo] CRITICAL: BLOCK_DATA received with null payload");
            m_logger->error("[Solo] Recovery: Empty BLOCK_DATA indicates node issue — delegating to recovery handler");
            m_pending_get_block.clear();
            dispatch_recovery_or_fallback(connection,
                                          GetBlockReason::VALIDATION_FAILURE,
                                          "[Solo] Null-payload recovery GET_BLOCK");
            return;
        }

        // Clear in-flight GET_BLOCK state — the node has responded with at least
        // a non-null payload. Even invalid responses past this point mean the
        // pending request has been serviced; leaving the flag set strands it for
        // up to TIMEOUT_SECONDS, suppressing legitimate future GET_BLOCK requests.
        m_pending_get_block.clear();

        
        // ═══════════════════════════════════════════════════════════════════
        // CRITICAL FIX: Accept BLOCK_DATA as initial template after MINER_READY
        // ═══════════════════════════════════════════════════════════════════
        // Node sends BLOCK_DATA (opcode 0) instead of STATELESS_GET_BLOCK (0xD081)
        // This fixes the 5-second timeout that causes 0.00 GIPS (no mining work)
        handle_initial_template_response("BLOCK_DATA (0x00)");

        // (m_pending_get_block.clear() is now at the top of on_block_data)
        
        // ═══════════════════════════════════════════════════════════════════
        // ENHANCED DIAGNOSTICS: Template delivery tracking
        // ═══════════════════════════════════════════════════════════════════
        m_logger->debug("[Solo Template Delivery] ═══════════════════════════════════");
        m_logger->debug("[Solo Template Delivery] 📥 TEMPLATE RECEIVED VIA: BLOCK_DATA (0x00)");
        m_logger->debug("[Solo Template Delivery]   Delivery Method: Legacy 8-bit opcode");
        m_logger->debug("[Solo Template Delivery]   Payload Size: {} bytes", packet.m_data->size());
        m_logger->debug("[Solo Template Delivery]   Packet Length: {} bytes", packet.m_length);
        m_logger->debug("[Solo Template Delivery]   Protocol Lane: {}", 
            get_lane_name(m_protocol_lane));
        m_logger->debug("[Solo Template Delivery] ═══════════════════════════════════");
        
        // TRAINING WHEELS: Full hex dump of BLOCK_DATA payload for debugging
        m_logger->debug("[Solo] BLOCK_DATA hex dump:");
        m_logger->debug("\n{}", format_llp_payload_hexdump(packet.m_data, 256));
        
        // Validate packet has minimum required data — guard BOTH the framer-declared
        // length AND the actual buffer size so the BE metadata read below cannot
        // dereference past the end of m_data.
        if (packet.m_length < MIN_BLOCK_HEADER_SIZE ||
            packet.m_data->size() < MIN_BLOCK_HEADER_SIZE) {
            m_logger->error("[Solo] CRITICAL: BLOCK_DATA packet has invalid size "
                            "(declared={} actual={} minimum={})",
                packet.m_length, packet.m_data->size(), MIN_BLOCK_HEADER_SIZE);
            m_logger->error("[Solo]   - This indicates corrupted or incomplete block data");
            m_logger->error("[Solo] Recovery: Invalid BLOCK_DATA — delegating to recovery handler");

            dispatch_recovery_or_fallback(connection,
                                          GetBlockReason::VALIDATION_FAILURE,
                                          "[Solo] Short-payload recovery GET_BLOCK");
            return;
        }
        
        // Use the Mining Template Interface for unified READ/FEED operations
        // This provides reliable template verification as part of the FALCON tunnel
        std::string source_endpoint;
        if (connection) {
            source_endpoint = connection->remote_endpoint().to_string();
        }

        // Extract 12-byte metadata prefix (big-endian) for HeightTracker, then
        // pass the full metadata+body payload to MiningTemplateInterface so
        // provisional recovery can require metadata/body consistency.
        uint32_t nUnifiedHeight = 0, nChannelHeight = 0, nBitsMeta = 0;
        {
            const auto& d = *packet.m_data;
            nUnifiedHeight = (uint32_t(d[0]) << 24) | (uint32_t(d[1]) << 16)
                           | (uint32_t(d[2]) <<  8) |  uint32_t(d[3]);
            nChannelHeight = (uint32_t(d[4]) << 24) | (uint32_t(d[5]) << 16)
                           | (uint32_t(d[6]) <<  8) |  uint32_t(d[7]);
            nBitsMeta      = (uint32_t(d[8]) << 24) | (uint32_t(d[9]) << 16)
                           | (uint32_t(d[10]) << 8) |  uint32_t(d[11]);
        }
        if (!m_template_interface) {
            // m_template_interface is constructed unconditionally in Solo's
            // constructor; this branch is unreachable in production.  Treat it
            // as a hard initialization failure rather than running a parallel
            // legacy decode path (the old fallback skipped the 12-byte metadata
            // prefix strip and would silently corrupt nHeight/nBits/nNonce).
            m_logger->error("[Solo] CRITICAL: Template interface missing — cannot process BLOCK_DATA");
            dispatch_recovery_or_fallback(connection,
                                          GetBlockReason::VALIDATION_FAILURE,
                                          "[Solo] Missing template-interface recovery GET_BLOCK");
            return;
        }

        m_logger->info("[Solo READ/FEED] Processing template via Mining Template Interface");

        auto validation_result = m_template_interface->read_stateless_payload(*packet.m_data, source_endpoint, false);

        if (!validation_result.is_valid) {
            m_logger->error("[Solo READ] Template validation failed: {}", validation_result.error_message);

            if (validation_result.is_stale) {
                m_logger->warn("[Solo READ] Template is stale - requesting fresh work");
            }

            dispatch_recovery_or_fallback(connection,
                                          GetBlockReason::VALIDATION_FAILURE,
                                          "[Solo] Validation recovery GET_BLOCK");
            return;
        }

        // ── HeightTracker BLOCK_DATA feed (Step 1/2) ───────────────────────────────
        // Feed HeightTracker only after MiningTemplateInterface accepted the full
        // metadata+body payload.  During provisional recovery, rejected metadata
        // must not advance monotonic canonical tracker state.
        // Use TEMPLATE source (not PUSH) so last_template_update timestamp is set,
        // enabling the post-push guard in check_template_health() to suppress false-positive
        // emergency stops when the GET_BLOCK response arrives after a push notification.
        update_height_state(nUnifiedHeight, nChannelHeight, nBitsMeta, HeightTracker::UpdateSource::TEMPLATE);

        // ── HeightTracker BLOCK_DATA feed (Step 2/2) ───────────────────────────────
        // Record channel_target = channel_height + 1 so is_template_stale() can
        // detect when the node's channel tip reaches or passes this template's target.
        // Skip genesis (channel_height == 0) to avoid false-positive staleness at startup.
        //
        // Bug #5 fix: use raw BLOCK_DATA metadata channel_height directly.
        // Previously used max(metadata, composite tracker) which allowed stale
        // push data to inflate the channel_target beyond what BLOCK_DATA reported.
        // BLOCK_DATA metadata is authoritative after payload validation succeeds.
        if (nChannelHeight > 0) {
            m_height_tracker.OnTemplateReceived(m_channel, nChannelHeight + 1);
            m_logger->info("[Solo BLOCK_DATA] HeightTracker fed: unified={} channel={} nBits=0x{:08x} → channel_target={}",
                nUnifiedHeight, nChannelHeight, nBitsMeta, nChannelHeight + 1);
        }

        m_logger->info("[Solo READ] Template validated successfully in {} μs",
            validation_result.validation_time.count());
        if (!finalize_and_feed_current_template(nUnifiedHeight,
                                                nChannelHeight,
                                                "Solo FEED",
                                                true)) {
            m_logger->error("[Solo FEED] Recovery: Block will be discarded, requesting new work");
            dispatch_recovery_or_fallback(connection,
                                          GetBlockReason::TEMPLATE_FEED_FAILURE,
                                          "[Solo] Template feed recovery GET_BLOCK");
            return;
        }
        cancel_get_block_control_retry(false);
}

void Solo::on_block_accepted(Packet const& packet, std::shared_ptr<network::Connection> connection)
{
    const auto last_submitted_owner = get_last_submitted_owner_snapshot();
    PacketIngressPreflightOptions preflight;
    preflight.owner = &last_submitted_owner;
    if (!run_packet_ingress_preflight("Solo BlockAccepted", preflight)) {
        return;
    }

    const bool is_block_accepted_compat =
        packet.m_is_uint16_opcode &&
        packet.m_header == LLP::StatelessMining::BLOCK_ACCEPTED_COMPAT &&
        packet.m_length == 0;

    if (matches_opcode(packet, Packet::ACCEPT) || is_block_accepted_compat)
    {
        if (!consume_pending_submit_result_or_warn("BLOCK_ACCEPTED", TriggerRecoveryOnStray::No)) {
            return;
        }

        stats::Global_delta global_stats{};
        global_stats.m_accepted_blocks = 1;
        m_stats_collector->update_global_stats(global_stats);

        // Use submitted block state (snapshotted at submit_block time) so we
        // don't depend on a template that may have been replaced since submission.
        const auto submitted = consume_last_submitted_state();
        uint32_t accepted_height  = submitted.height;
        uint32_t accepted_channel = submitted.channel;
        if (!submitted.valid) {
            // Fallback: submission state not populated (e.g. legacy path).
            // Warning: template may have been replaced since submission.
            m_logger->warn("BLOCK_ACCEPTED fallback: m_last_submitted_valid=false — "
                           "reading height/channel from current template (may reflect a newer block)");
            if (m_template_interface) {
                auto const* tmpl = m_template_interface->get_current_template();
                if (tmpl) {
                    accepted_height  = tmpl->block.nHeight;
                    accepted_channel = tmpl->block.nChannel;
                }
            }
        }
        m_logger->info("✅ BLOCK ACCEPTED by node — height={} channel={}", accepted_height, accepted_channel);
        m_logger->info("Block Accepted By Nexus Network.");
        record_session_event(SessionManager::SessionEventKind::SUBMIT_ACCEPTED,
                             "height=" + std::to_string(accepted_height) +
                             " channel=" + std::to_string(accepted_channel));

        // Notify Worker_manager to record in the mined-block cache.
        // Use submitted prev_hash and nonce rather than re-reading from template.
        if (m_block_accepted_handler) {
            m_block_accepted_handler(accepted_height, submitted.prev_hash,
                                     accepted_channel, submitted.nonce);
        }
        
        // Enhanced diagnostics: Log connection info for accepted block
        if (connection) {
            auto const& remote_ep = connection->remote_endpoint();
            std::string remote_addr;
            remote_ep.address(remote_addr);
            uint16_t actual_port = remote_ep.port();
            m_logger->info("[Solo] Block accepted on connection {}:{}", remote_addr, actual_port);
        }

        // Record BLOCK_ACCEPTED timestamp and cancel any pending NEW_ROUND recovery
        // debounce.  A fresh BLOCK_DATA push is about to arrive — no need to also
        // fire a deferred recovery GET_BLOCK scheduled by a concurrent NEW_ROUND.
        m_last_block_accepted_time = std::chrono::steady_clock::now();
        cancel_recovery_timer("BLOCK_ACCEPTED");
        
        // Reset dedup state so the follow-up GET_BLOCK is not suppressed by stale
        // height values — the node has not sent a new push notification yet.
        reset_get_block_dedup_state();

        // Request new work with recovery logic.
        // BLOCK_ACCEPTED bypasses all dedup: the accepted-block template is now
        // spent at the previous height, and legacy/stateless lanes must both force
        // an authoritative replacement even if PUSH/BLOCK_DATA has not arrived yet.
        auto work_payload = get_work(GetBlockReason::BLOCK_ACCEPTED);
        if (!work_payload || work_payload->empty()) {
            m_logger->error("[Solo] CRITICAL: GET_BLOCK request after ACCEPT returned empty payload!");
            m_logger->error("[Solo] Recovery: Retrying work request");
            // Retry once
            work_payload = get_work(GetBlockReason::RECOVERY_FORCED);
            if (!work_payload || work_payload->empty()) {
                m_logger->error("[Solo] CRITICAL: GET_BLOCK retry also failed - mining may stall");
            } else {
                if (!queue_payload(connection, work_payload, "[Solo] Accepted-block GET_BLOCK")) {
                    return;
                }
                mark_get_block_pending(GetBlockReason::RECOVERY_FORCED);
            }
        } else {
            if (!queue_payload(connection, work_payload, "[Solo] Accepted-block GET_BLOCK")) {
                return;
            }
            mark_get_block_pending(GetBlockReason::BLOCK_ACCEPTED);
        }
    }
    // Handle legacy GOOD_BLOCK (opcode 6): some legacy nodes send this for valid-but-not-best blocks.
    // Treat as accepted for counter purposes.
    else if (matches_opcode(packet, LLP::GOOD_BLOCK))
    {
        if (!consume_pending_submit_result_or_warn("GOOD_BLOCK", TriggerRecoveryOnStray::No)) {
            return;
        }

        stats::Global_delta global_stats{};
        global_stats.m_accepted_blocks = 1;
        m_stats_collector->update_global_stats(global_stats);

        // Use submitted block state (snapshotted at submit_block time).
        const auto submitted = consume_last_submitted_state();
        uint32_t accepted_height  = submitted.height;
        uint32_t accepted_channel = submitted.channel;
        if (!submitted.valid) {
            m_logger->warn("GOOD_BLOCK fallback: m_last_submitted_valid=false — "
                           "reading height/channel from current template (may reflect a newer block)");
            if (m_template_interface) {
                auto const* tmpl = m_template_interface->get_current_template();
                if (tmpl) {
                    accepted_height  = tmpl->block.nHeight;
                    accepted_channel = tmpl->block.nChannel;
                }
            }
        }
        m_logger->info("✅ BLOCK ACCEPTED by node (Legacy Lane, GOOD_BLOCK) — height={} channel={}",
            accepted_height, accepted_channel);
        record_session_event(SessionManager::SessionEventKind::SUBMIT_ACCEPTED,
                             "height=" + std::to_string(accepted_height) +
                             " channel=" + std::to_string(accepted_channel) +
                             " via GOOD_BLOCK");

        // Notify Worker_manager to record in the mined-block cache.
        // Use submitted prev_hash and nonce rather than re-reading from template.
        if (m_block_accepted_handler) {
            m_block_accepted_handler(accepted_height, submitted.prev_hash,
                                     accepted_channel, submitted.nonce);
        }

        m_last_block_accepted_time = std::chrono::steady_clock::now();
        cancel_recovery_timer("GOOD_BLOCK");
        reset_get_block_dedup_state();
        // BLOCK_ACCEPTED bypasses all dedup: the accepted template is spent.
        request_and_queue_get_block(connection,
                                    GetBlockReason::BLOCK_ACCEPTED,
                                    "[Solo] GOOD_BLOCK GET_BLOCK");
    }
}

bool Solo::consume_pending_submit_result_or_warn(const char* opcode_name, TriggerRecoveryOnStray trigger_recovery)
{
    if (m_submit_result_gate.consume_pending()) {
        return true;
    }

    const bool had_pending_get_block = m_pending_get_block.active;
    m_logger->warn("[Solo] {} received with no submitted block pending "
                   "(pending_get_block={}) — not counting duplicate/stray block result",
                   opcode_name, had_pending_get_block ? "true" : "false");

    if (trigger_recovery == TriggerRecoveryOnStray::Yes) {
        m_pending_get_block.clear();
        reset_get_block_dedup_state();
        m_logger->warn("[Solo] Legacy lane recovery: cleared pending GET_BLOCK; health/recovery monitor will retry");
        if (m_recovery_handler) {
            // Return value discarded — this is a "kick recovery" signal, not a
            // chokepoint substitute. There is no direct local GET_BLOCK to skip
            // in this path; the health/recovery monitor decides what to do next.
            (void) m_recovery_handler(GetBlockReason::RECOVERY_FORCED);
        }
    }

    return false;
}

void Solo::on_block_rejected(Packet const& packet, std::shared_ptr<network::Connection> connection)
{
    if (packet.m_length == 8 && matches_opcode(packet, Packet::REJECT)) {
        on_get_block_control(packet, connection);
        return;
    }

    const auto last_submitted_owner = get_last_submitted_owner_snapshot();
    PacketIngressPreflightOptions preflight;
    preflight.owner = &last_submitted_owner;
    if (!run_packet_ingress_preflight("Solo BlockRejected", preflight)) {
        return;
    }

    const bool is_block_rejected_compat =
        packet.m_is_uint16_opcode &&
        packet.m_header == LLP::StatelessMining::BLOCK_REJECTED_COMPAT &&
        packet.m_length <= 1;

    if (matches_opcode(packet, Packet::REJECT) || is_block_rejected_compat)
    {
        if (!consume_pending_submit_result_or_warn("BLOCK_REJECTED", TriggerRecoveryOnStray::Yes)) {
            return;
        }

        stats::Global_delta global_stats{};
        global_stats.m_rejected_blocks = 1;
        m_stats_collector->update_global_stats(global_stats);
        const auto submitted = consume_last_submitted_state();
        uint32_t rejected_height = submitted.height;
        uint32_t rejected_channel = submitted.channel;

        if (!submitted.valid) {
            // Fallback: submission state not populated (e.g. legacy path).
            // Warning: template may have been replaced since submission.
            m_logger->warn("BLOCK_REJECTED fallback: m_last_submitted_valid=false — "
                           "reading height/channel from current template (may reflect a newer block)");
            rejected_height = 0;
            rejected_channel = m_channel;
            if (m_template_interface) {
                auto const* tmpl = m_template_interface->get_current_template();
                if (tmpl) {
                    rejected_height = tmpl->block.nHeight;
                    rejected_channel = tmpl->block.nChannel;
                }
            }
        }

        // Both protocol lanes may carry a one-byte submission rejection reason.
        std::string reason_str = "NONE";
        bool is_fork_rejection = false;
        if (packet.m_data && !packet.m_data->empty()) {
            uint8_t reason_byte = (*packet.m_data)[0];
            switch (reason_byte) {
                case static_cast<uint8_t>(LLP::StatelessMining::RejectionReason::STALE):       reason_str = "STALE"; break;
                case static_cast<uint8_t>(LLP::StatelessMining::RejectionReason::INVALID_POW): reason_str = "INVALID_POW"; break;
                case static_cast<uint8_t>(LLP::StatelessMining::RejectionReason::INVALID_SIG): reason_str = "INVALID_SIG"; break;
                case static_cast<uint8_t>(LLP::StatelessMining::RejectionReason::DUPLICATE):   reason_str = "DUPLICATE"; break;
                case static_cast<uint8_t>(LLP::StatelessMining::RejectionReason::FORK):
                    reason_str = "FORK";
                    is_fork_rejection = true;
                    break;
                default: { char buf[16]; snprintf(buf, sizeof(buf), "0x%02x", reason_byte); reason_str = buf; break; }
            }
        }
        m_logger->warn("❌ BLOCK REJECTED by node — height={} channel={} reason={}", rejected_height, rejected_channel, reason_str);
        m_logger->warn("Block Rejected by Nexus Network.");
        record_session_event(SessionManager::SessionEventKind::SUBMIT_REJECTED,
                             "height=" + std::to_string(rejected_height) +
                             " channel=" + std::to_string(rejected_channel) +
                             " reason=" + reason_str);

        // Enhanced diagnostics: Log connection info and possible reasons
        if (connection) {
            auto const& remote_ep = connection->remote_endpoint();
            std::string remote_addr;
            remote_ep.address(remote_addr);
            uint16_t actual_port = remote_ep.port();
            m_logger->warn("[Solo] Block rejected on connection {}:{}", remote_addr, actual_port);
        }

        if (reason_str == "NONE") {
            m_logger->info("[Solo] Node sent no rejection reason code; check node logs for the"
                           " submission failure. Older nodes may not send a reason.");
        }

        // Special handling for FORK rejections: invalidate template and trigger recovery
        if (is_fork_rejection) {
            m_logger->warn("[Solo FORK] FORK rejection detected — invalidating template and initiating recovery");

            // Get current height for fork detection handler
            auto snap = m_height_tracker.GetSnapshot();
            uint32_t unified_height = snap.unified_height;

            // Call handle_fork_detected to invalidate the template
            auto* pManager = get_channel_manager();
            if (pManager) {
                handle_fork_detected(pManager, unified_height);
            } else {
                // Fallback: Invalidate template directly if no channel manager
                if (m_template_interface && m_template_interface->has_valid_template()) {
                    m_template_interface->discard_template("Fork detected - BLOCK_REJECTED:FORK");
                    m_logger->info("[Solo FORK] ✗ Template invalidated due to fork rejection");
                }
            }

            // Notify Worker_manager to mark recovery initiated (same pattern as push handler).
            // Return value is intentionally discarded — the rejected-block path below issues
            // its own GET_BLOCK directly because the dedup state has just been reset and we
            // need to push the request through promptly to recover the chain tip.
            if (m_recovery_handler) {
                m_logger->info("[Solo FORK] Recovery initiated (fork_rejection) — notifying Worker_manager");
                (void) m_recovery_handler(GetBlockReason::RECOVERY_FORCED);
            }
        }

        // Reset dedup state so the follow-up GET_BLOCK is not suppressed by stale
        // height values — the node has not sent a new push notification yet.
        reset_get_block_dedup_state();

        // Request new work with recovery logic (REJECT path)
        auto work_payload = get_work(GetBlockReason::BLOCK_REJECTED);
        if (!work_payload || work_payload->empty()) {
            m_logger->error("[Solo] CRITICAL: GET_BLOCK request after REJECT returned empty payload!");
            m_logger->error("[Solo] Recovery: Retrying work request");
            // Retry once
            work_payload = get_work(GetBlockReason::RECOVERY_FORCED);
            if (!work_payload || work_payload->empty()) {
                m_logger->error("[Solo] CRITICAL: GET_BLOCK retry also failed - will wait for next node push");
                // NOTE: Do NOT send MINER_READY here. MINER_READY is a one-time subscription
                // handshake; the node keeps the miner subscribed for the session lifetime.
                // The next push from the node will trigger a fresh GET_BLOCK request.
            } else {
                if (!queue_payload(connection, work_payload, "[Solo] Rejected-block GET_BLOCK")) {
                    return;
                }
                mark_get_block_pending(GetBlockReason::RECOVERY_FORCED);
            }
        } else {
            if (!queue_payload(connection, work_payload, "[Solo] Rejected-block GET_BLOCK")) {
                return;
            }
            mark_get_block_pending(GetBlockReason::BLOCK_REJECTED);
        }
    }
    // Handle legacy ORPHAN_BLOCK (opcode 7): some legacy nodes send this for orphaned blocks.
    // Treat as rejected for counter purposes.
    else if (matches_opcode(packet, LLP::ORPHAN_BLOCK))
    {
        if (!consume_pending_submit_result_or_warn("ORPHAN_BLOCK", TriggerRecoveryOnStray::Yes)) {
            return;
        }

        stats::Global_delta global_stats{};
        global_stats.m_rejected_blocks = 1;
        m_stats_collector->update_global_stats(global_stats);

        const auto submitted = consume_last_submitted_state();
        uint32_t rejected_height = submitted.height;
        uint32_t rejected_channel = submitted.channel;
        if (!submitted.valid) {
            m_logger->warn("ORPHAN_BLOCK fallback: m_last_submitted_valid=false — "
                           "reading height/channel from current template (may reflect a newer block)");
            rejected_height = 0;
            rejected_channel = m_channel;
            if (m_template_interface) {
                auto const* tmpl = m_template_interface->get_current_template();
                if (tmpl) {
                    rejected_height = tmpl->block.nHeight;
                    rejected_channel = tmpl->block.nChannel;
                }
            }
        }
        m_logger->warn("❌ BLOCK REJECTED by node (Legacy Lane, ORPHAN_BLOCK) — height={} channel={}",
            rejected_height, rejected_channel);

        reset_get_block_dedup_state();
        request_and_queue_get_block(connection,
                                    GetBlockReason::BLOCK_REJECTED,
                                    "[Solo] ORPHAN_BLOCK GET_BLOCK");
    }
}

void Solo::cancel_get_block_control_retry(bool clear_deadline)
{
    ++m_get_block_control_generation;
    if (clear_deadline) {
        m_get_block_retry_at_ms.store(0, std::memory_order_release);
    }
    if (m_get_block_control_timer) {
        m_get_block_control_timer->cancel();
    }
}

void Solo::on_get_block_control(Packet const& packet,
                                const std::shared_ptr<network::Connection>& connection)
{
    if (!m_pending_get_block.active) {
        m_logger->debug("[Solo GET_BLOCK] Ignoring stale control response: no request in flight");
        return;
    }
    PacketIngressPreflightOptions preflight;
    preflight.owner = &m_last_get_block_request_owner;
    if (!run_packet_ingress_preflight("Solo GET_BLOCK control", preflight)) {
        return;
    }
    if (!packet.m_data) {
        m_logger->error("[Solo GET_BLOCK] Control response has no payload");
        return;
    }
    const auto control = GetBlockControl::parse(*packet.m_data);
    if (!control) {
        m_logger->error("[Solo GET_BLOCK] Invalid control response (version, reason, or reserved bytes)");
        return;
    }

    m_pending_get_block.clear();
    m_logger->warn("[Solo GET_BLOCK] Node returned {} (retry_after_ms={}); "
                   "check system/get/info.mining_readiness on the node",
                   control->reason_name(), control->retry_after_ms);
    if (!control->retryable()) {
        cancel_get_block_control_retry();
        if (control->reason == GetBlockControlReason::SESSION_INVALID ||
            control->reason == GetBlockControlReason::UNAUTHENTICATED) {
            if (m_session_context) {
                handle_session_expired(get_session_id(), static_cast<uint8_t>(control->reason), connection);
            } else {
                reset_auth_state();
                if (m_session_expired_handler) {
                    m_session_expired_handler();
                }
            }
        } else {
            m_logger->error("[Solo GET_BLOCK] Non-retryable node response requires configuration attention");
        }
        return;
    }

    const auto delay = control->retry_delay();
    const auto due = std::chrono::steady_clock::now() + delay;
    const auto due_ms = std::chrono::ceil<std::chrono::milliseconds>(
        due.time_since_epoch()).count();
    const auto previous_due_ms = m_get_block_retry_at_ms.load(std::memory_order_acquire);
    const auto effective_due_ms = std::max<int64_t>(due_ms, previous_due_ms);
    m_get_block_retry_at_ms.store(effective_due_ms, std::memory_order_release);
    const auto generation = ++m_get_block_control_generation;
    if (!m_get_block_control_timer || weak_from_this().expired()) {
        m_logger->warn("[Solo GET_BLOCK] No retry timer available; health monitor must retry");
        return;
    }
    m_get_block_control_timer->expires_at(
        std::chrono::steady_clock::time_point(std::chrono::milliseconds(effective_due_ms)));
    m_get_block_control_timer->async_wait(
        [weak_self = weak_from_this(), weak_connection = std::weak_ptr<network::Connection>(connection),
         generation](const asio::error_code& ec) {
            auto self = weak_self.lock();
            if (!self || ec == asio::error::operation_aborted) return;
            if (ec) {
                self->m_logger->error("[Solo GET_BLOCK] Retry timer failed: {}", ec.message());
                return;
            }
            if (generation != self->m_get_block_control_generation.load()) return;
            self->m_get_block_retry_at_ms.store(0, std::memory_order_release);
            if (self->m_pending_get_block.active && !self->m_pending_get_block.has_timed_out()) {
                self->m_logger->debug("[Solo GET_BLOCK] Retry unnecessary: request already in flight");
                return;
            }
            self->dispatch_recovery_or_fallback(weak_connection.lock(), GetBlockReason::RECOVERY_FORCED,
                                                 "[Solo GET_BLOCK] Node control retry");
        });
}

// ─────────────────────────────────────────────────────────────────────────────
// NEW_ROUND recovery debounce helpers
// ─────────────────────────────────────────────────────────────────────────────

void Solo::schedule_recovery_get_block(
    std::shared_ptr<network::Connection> connection,
    uint32_t unified_height)
{
    const auto now = std::chrono::steady_clock::now();

    auto ms_since = [&](std::chrono::steady_clock::time_point tp) -> int64_t {
        if (tp == std::chrono::steady_clock::time_point::min()) return -1;
        return std::chrono::duration_cast<std::chrono::milliseconds>(now - tp).count();
    };

    const int64_t since_accept_ms = ms_since(m_last_block_accepted_time);
    const int64_t since_push_ms   = ms_since(m_last_push_received_time);

    // Phase C1: PUSH × NEW_ROUND race guard.
    // A PUSH for ≥ unified_height arrived within the BLOCK_DATA-in-transit
    // window means the node has already signalled and is about to auto-deliver
    // a fresh template; scheduling a 2 s recovery GET_BLOCK on top of that is
    // pure noise.  Note: cancel_recovery_timer() in on_push_notification kills
    // an *already-scheduled* timer when PUSH arrives second; this guard covers
    // the inverse race where PUSH arrived first and we are about to schedule
    // a redundant timer.
    {
        auto snap = m_height_tracker.GetSnapshot();
        if (push_implies_block_data_in_transit(snap, unified_height,
                                               push_in_transit_guard_for_channel().count(), now)) {
            m_logger->info(
                "[NEW_ROUND] Recovery GET_BLOCK NOT scheduled: PUSH for height ≥ {} arrived "
                "{}ms ago — BLOCK_DATA is in transit, no recovery needed",
                unified_height, since_push_ms);
            return;
        }
    }

    if (!m_recovery_timer) {
        // No io_context supplied (e.g. unit tests without a timer) — fall back
        // to the legacy immediate behaviour so pre-existing tests are unaffected.
        m_logger->info("[NEW_ROUND] No debounce timer (no io_context) — recovery GET_BLOCK firing immediately "
                       "(since_accept={}ms, since_push={}ms)",
                       since_accept_ms, since_push_ms);
        if (connection) {
            request_and_queue_get_block(connection,
                                        GetBlockReason::GET_ROUND_NO_TEMPLATE,
                                        "[Solo GET_ROUND] Template refresh GET_BLOCK (immediate)");
        }
        return;
    }

    m_logger->info(
        "[NEW_ROUND] Template invalidated; deferring recovery GET_BLOCK for {}ms "
        "(since_block_accepted={}ms, since_push={}ms)",
        std::chrono::duration_cast<std::chrono::milliseconds>(kRecoveryDebounceWindow).count(),
        since_accept_ms,
        since_push_ms);

    m_recovery_deferred_at = now;

    // Cancel any prior pending recovery — coalesce rapid NEW_ROUND bursts so that
    // only the LAST NEW_ROUND in a burst starts the 5s countdown.
    m_recovery_timer->cancel();
    m_recovery_timer->expires_after(kRecoveryDebounceWindow);
    m_recovery_timer->async_wait(
        [this, conn = connection, unified_height]
        (const asio::error_code& ec) {
            if (ec == asio::error::operation_aborted) {
                // Cancelled — either a newer NEW_ROUND superseded us, or a
                // PUSH / BLOCK_ACCEPTED arrived and called cancel_recovery_timer().
                return;
            }
            if (ec) {
                m_logger->warn("[NEW_ROUND] Recovery timer error: {}", ec.message());
                return;
            }
            // 5 seconds elapsed — did a template arrive in the meantime?
            if (m_template_interface && m_template_interface->has_valid_template()) {
                m_logger->info(
                    "[NEW_ROUND] Recovery NOT fired: template installed during {}ms debounce",
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        kRecoveryDebounceWindow).count());
                m_recovery_deferred_at = std::chrono::steady_clock::time_point::min();
                return;
            }
            // Final push-in-transit guard at fire time.
            //
            // The schedule-time guard above (Phase C1) covers PUSH-before-NEW_ROUND.
            // cancel_recovery_timer() in on_push_notification covers PUSH-during-defer
            // by aborting this async_wait.  This third guard closes the residual
            // race window where a PUSH arrives so close to timer expiry that
            // m_recovery_timer->cancel() returns 0 (the wait already completed
            // and is queued on the io_context) — the cancel log fires but the
            // callback still runs and would otherwise send a redundant GET_BLOCK
            // on top of the BLOCK_DATA the node is already auto-sending.
            //
            // Re-evaluating push_implies_block_data_in_transit() right before
            // the send keeps this path consistent with all other channel-aware
            // BLOCK_DATA-in-transit guards in solo.cpp.
            {
                auto snap_now = m_height_tracker.GetSnapshot();
                if (push_implies_block_data_in_transit(snap_now, unified_height,
                                                       push_in_transit_guard_for_channel().count())) {
                    m_logger->info(
                        "[NEW_ROUND] Recovery GET_BLOCK suppressed at fire time: "
                        "PUSH for height ≥ {} arrived during {}ms debounce — "
                        "BLOCK_DATA in transit, no recovery needed",
                        unified_height,
                        std::chrono::duration_cast<std::chrono::milliseconds>(
                            kRecoveryDebounceWindow).count());
                    m_recovery_deferred_at = std::chrono::steady_clock::time_point::min();
                    return;
                }
            }
            m_logger->info(
                "[NEW_ROUND] Recovery GET_BLOCK firing after {}ms debounce: "
                "no push arrived, template still invalid",
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    kRecoveryDebounceWindow).count());
            m_recovery_deferred_at = std::chrono::steady_clock::time_point::min();
            ++m_recovery_fired_count;
            // Prefer the connection captured when NEW_ROUND fired; fall back to
            // the stored session connection if the first one expired.
            auto active_conn = conn ? conn : m_connection;
            if (active_conn) {
                request_and_queue_get_block(active_conn,
                                            GetBlockReason::GET_ROUND_NO_TEMPLATE,
                                            "[Solo GET_ROUND] Recovery GET_BLOCK (5s debounce)");
            } else {
                m_logger->warn("[NEW_ROUND] Recovery GET_BLOCK: no connection available after debounce");
            }
        });
}

void Solo::cancel_recovery_timer(const char* handler_name)
{
    if (!m_recovery_timer) return;
    if (m_recovery_deferred_at == std::chrono::steady_clock::time_point::min()) return;

    const auto cancelled = m_recovery_timer->cancel();
    if (cancelled > 0) {
        const auto deferred_age_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - m_recovery_deferred_at).count();
        m_logger->info(
            "[{}] Cancelling pending NEW_ROUND recovery GET_BLOCK "
            "(deferred {}ms ago — push won the race)",
            handler_name, deferred_age_ms);
    }
    m_recovery_deferred_at = std::chrono::steady_clock::time_point::min();
}

void Solo::on_get_round_response(Packet const& packet, std::shared_ptr<network::Connection> connection)
{
    const char* lane_label = get_lane_name(m_protocol_lane);

    if (matches_opcode(packet, Packet::NEW_ROUND))
    {
        m_logger->info("[Solo GET_ROUND] NEW_ROUND response received (lane={})", lane_label);

        if (get_session_manager()) {
            if (!get_session_manager()->is_active()) {
                m_logger->error("[Solo GET_ROUND] NEW_ROUND received but no active session");
            } else {
                auto session_id = get_session_manager()->get_session_id();
                m_logger->info("[Solo GET_ROUND] NEW_ROUND received, keeping session 0x{:08X}", session_id.get());
            }
        }
        
        bool get_block_sent_in_handler = false;  // Track whether GET_BLOCK was already requested in this handler
        
        // ✅ ACCEPTED FORMAT: 16 bytes only — full height picture (unified + prime + hash + stake)
        if (!packet.m_data || packet.m_length != 16) {
            m_logger->error("[Solo GET_ROUND] ❌ PROTOCOL ERROR: Invalid packet length");
            m_logger->error("[Solo GET_ROUND]   Expected:  16 bytes (unified + prime + hash + stake)");
            m_logger->error("[Solo GET_ROUND]   Received:  {} bytes", packet.m_length);
            m_logger->error("[Solo GET_ROUND]   Node may be running incompatible version");
            m_logger->error("[Solo GET_ROUND]   Required:  LLL-TAO legacy 16-byte GET_ROUND format");
            return;
        }
        
        // Parse 16-byte full-height-picture response (all big-endian)
        uint32_t unified_height = bytes2uint(*packet.m_data, 0);
        uint32_t prime_height   = bytes2uint(*packet.m_data, 4);
        uint32_t hash_height    = bytes2uint(*packet.m_data, 8);
        uint32_t stake_height   = bytes2uint(*packet.m_data, 12);
        
        // ── GET_ROUND heights are already TIP semantics — no normalization ──
        // The NamecoinGithub/LLL-TAO node GET_ROUND (handle_get_round_stateless)
        // sends tStateBest.nHeight for unified and stateChannel.nChannelHeight
        // for per-channel values.  These are the current chain TIP — NOT target
        // (tip+1).  No subtraction needed.
        //
        // Note: GET_HEIGHT sends nBestHeight + 1 (target), but GET_ROUND does
        // NOT follow that convention.  BLOCK_DATA metadata bytes [0-3] and [4-7]
        // also use TIP semantics.  block.nHeight inside the 216-byte block is
        // TARGET (tStateBest.nHeight + 1).
        m_logger->debug("[Solo GET_ROUND] Parsed heights (TIP): unified={} prime={} hash={} stake={}",
            unified_height, prime_height, hash_height, stake_height);

        // Derive active-channel height from full picture
        uint32_t channel_height = 0;
        if (m_channel == mining::CHANNEL_PRIME) {
            channel_height = prime_height;
        } else if (m_channel == mining::CHANNEL_HASH) {
            channel_height = hash_height;
        } else {
            m_logger->error("[Solo GET_ROUND] Invalid channel: {}", m_channel);
            m_logger->error("[Solo GET_ROUND] Expected 1 (Prime) or 2 (Hash), got {}", m_channel);
            return;
        }
        
        uint32_t previous_unified_height = m_last_round_unified_height;
        uint32_t previous_channel_height = m_last_round_channel_height;

        // Determine channel name for logging
        std::string channel_name = get_channel_name(m_channel);
        
        // Log response details (normalized to tip semantics)
        // Show canonical (BLOCK_DATA) height alongside for context — during rapid
        // block production GET_ROUND may lag behind BLOCK_DATA by several blocks.
        auto canonical_snap = m_height_tracker.GetCanonicalSnapshot();
        m_logger->info("[Solo GET_ROUND] 🔔 NEW_ROUND (16-byte full height picture, lane={}):", lane_label);
        m_logger->info("[Solo GET_ROUND]   Unified height:  {} (tip{})",
            unified_height,
            canonical_snap.is_initialized()
                ? fmt::format("; canonical={}", canonical_snap.canonical_unified_height)
                : std::string{});
        m_logger->info("[Solo GET_ROUND]   Prime height:    {} (tip)", prime_height);
        m_logger->info("[Solo GET_ROUND]   Hash height:     {} (tip)", hash_height);
        m_logger->info("[Solo GET_ROUND]   Stake height:    {} (tip)", stake_height);
        m_logger->info("[Solo GET_ROUND]   {} height:      {} (derived tip)", channel_name, channel_height);
        m_logger->info("[Solo GET_ROUND]   Difficulty:      (unchanged; not in 16-byte payload)");
        
        // Update RoundStatus
        m_last_round_status.is_new_round = true;
        m_last_round_status.height = unified_height;
        m_last_round_status.has_channel_heights = true;
        m_last_round_status.prime_height = prime_height;
        m_last_round_status.hash_height  = hash_height;
        m_last_round_status.stake_height = stake_height;
        
        // Update HeightTracker with full height picture (direct call — bypasses
        // update_height_state so all 4 heights reach the diagnostic state).
        m_height_tracker.OnGetRound(unified_height, prime_height, hash_height, stake_height);
        // Update ClientChannelManager for fork/phantom-stake detection (exactly once).
        apply_channel_manager_update(unified_height, channel_height);

        // NEW_ROUND means the tip changed.  GET_ROUND_* reasons already bypass
        // the height-based dedup guard via should_bypass_height_dedup(), and the
        // 100ms rapid-burst guard remains active to prevent a simultaneous PUSH
        // from racing and sending a duplicate GET_BLOCK.  Do NOT call
        // reset_get_block_dedup_state() here — that clears the burst guard
        // timestamp, enabling a PUSH arriving <100ms later to bypass burst
        // suppression entirely (Bug #2 fix).

        // Pass channel height to template interface for staleness validation.
        // update_channel_height() is the primary staleness gate (uses nChannelHeight).
        // check_staleness_by_channel_delta() is a secondary snapshot-based check used ONLY
        // when the template is still pending finalization (nChannelHeight == 0); once the
        // template is finalized, sync_template_state() + update_channel_height() are
        // the sole arbiters and the snapshot check must be skipped to prevent false stales.
        if (m_template_interface) {
            m_template_interface->update_channel_height(m_channel, channel_height);

            // Only use snapshot-based delta check while template awaits finalization.
            if (m_template_interface->needs_channel_height_finalization()) {
                bool is_stale = m_template_interface->check_staleness_by_channel_delta(channel_height);
                if (is_stale) {
                    // Phase C2: PUSH × NEW_ROUND race guard.  If a PUSH for
                    // height ≥ unified_height arrived within the BLOCK_DATA
                    // in-transit window, BLOCK_DATA is already on its way —
                    // suppress this immediate GET_BLOCK so we don't race a
                    // duplicate request that the node will rate-limit.
                    auto snap_push = m_height_tracker.GetSnapshot();
                    const bool push_in_transit = push_implies_block_data_in_transit(
                        snap_push, unified_height, push_in_transit_guard_for_channel().count());
                    if (push_in_transit) {
                        m_logger->info("[Solo GET_ROUND] ⚡ CHAIN TIP CHANGED: {} channel advanced "
                            "(staleness) — GET_BLOCK suppressed: PUSH for height ≥ {} in transit",
                            get_channel_name(m_channel), unified_height);
                    } else {
                        m_logger->info("[Solo GET_ROUND] ⚡ CHAIN TIP CHANGED: {} channel advanced (awaiting finalization → GET_BLOCK)",
                            get_channel_name(m_channel));
                        if (connection) {
                            if (request_and_queue_get_block(connection,
                                                            GetBlockReason::GET_ROUND_STALE,
                                                            "[Solo GET_ROUND] GET_BLOCK")) {
                                get_block_sent_in_handler = true;
                                m_logger->info("[Solo GET_ROUND] ✓ GET_BLOCK request sent - waiting for new template...");
                            } else {
                                m_logger->debug("[Solo GET_ROUND] GET_BLOCK suppressed by dedup guard (staleness path)");
                            }
                        } else {
                            m_logger->error("[Solo GET_ROUND] Cannot request fresh template - connection is null");
                        }
                    }
                    // No early return (previous early return removed): fall through to
                    // sync_template_state() so any pending template channel-height metadata
                    // is finalized regardless of whether GET_BLOCK was also requested.
                }
            }

            m_logger->debug("[Solo] Channel height for staleness validation: {} ({})",
                channel_height, get_channel_name(m_channel));
        }

        // ── WHY GET_ROUND REMAINS USEFUL EVEN WITH PUSH/BLOCK_DATA ────────────────────
        //
        // The Nexus node currently sends PUSH notifications (PRIME/HASH_BLOCK_AVAILABLE)
        // only when Prime or Hash channel blocks are found. Stake blocks DO advance the
        // unified blockchain height (and change hashPrevBlock), but they do NOT trigger
        // a PUSH to mining channels.
        //
        // GET_ROUND is therefore still the miner's backstop for discovering unified-tip
        // advances that arrive without a matching PUSH/BLOCK_DATA autosend. In the common
        // case PUSH/BLOCK_DATA keep templates fresh and GET_ROUND remains informational.
        // When PUSH is absent or local template state says stale, the same poll stream can
        // still trigger a corrective GET_BLOCK.
        //
        // Future improvement: Add node-side PUSH subscription that fires on ANY unified
        // height change (including Stake), eliminating the need for polling entirely.
        // Until then, GET_ROUND remains the fallback signal for cross-channel tip detection.
        // ────────────────────────────────────────────────────────────────────────────────

        // ── Stake/cross-channel unified tip detection ──────────────────────────────
        // Detect when unified height advanced but our channel height did NOT change.
        // This happens when a Stake block or the opposite PoW channel finds a block.
        // PUSH does not currently fire for Stake blocks, so GET_ROUND is the ONLY
        // mechanism to detect these tip advances.
        // When this occurs, hashPrevBlock in the current template is stale — we must
        // discard and request a fresh template.
        //
        // CANONICAL GUARD: GET_ROUND polls every 15s and can lag behind BLOCK_DATA
        // arrivals during rapid block production. If the canonical path (BLOCK_DATA)
        // already knows about a unified height >= what GET_ROUND reports, the template
        // is NOT stale — GET_ROUND is simply lagging. Skip the discard to avoid
        // unnecessary template churn and wasted work.
        {
            bool unified_advanced = (unified_height > m_last_round_unified_height) &&
                                    (m_last_round_unified_height > 0);
            bool channel_unchanged = (channel_height == m_last_round_channel_height);

            auto canonical = m_height_tracker.GetCanonicalSnapshot();
            bool canonical_already_ahead = canonical.is_initialized() &&
                                           canonical.canonical_unified_height >= unified_height;
            auto snap = m_height_tracker.GetSnapshot();
            bool recent_push_for_this_height = push_implies_block_data_in_transit(
                snap, unified_height, push_in_transit_guard_for_channel().count());

            if (unified_advanced && channel_unchanged && !get_block_sent_in_handler
                && !canonical_already_ahead && !recent_push_for_this_height) {
                m_logger->info("[Solo GET_ROUND] ⚡ CHAIN TIP CHANGED: Stake/cross-channel advance unified {} → {} "
                               "({} channel height unchanged — discarding stale template)",
                               m_last_round_unified_height, unified_height,
                               get_channel_name(m_channel));

                // Discard template: hashPrevBlock is now stale (different tip)
                if (m_template_interface && m_template_interface->has_valid_template()) {
                    m_template_interface->discard_template("Stake/cross-channel tip advance detected via GET_ROUND");
                    m_logger->info("[Solo GET_ROUND] ✗ Template discarded (Stake-advance stale hashPrevBlock)");
                }

                // Reset dedup guard so this GET_BLOCK is not suppressed
                reset_get_block_dedup_state();

                if (connection) {
                    if (request_and_queue_get_block(connection,
                                                    GetBlockReason::GET_ROUND_NO_TEMPLATE,
                                                    "[Solo GET_ROUND] Stake refresh GET_BLOCK")) {
                        get_block_sent_in_handler = true;
                        m_logger->info("[Solo GET_ROUND] ✓ GET_BLOCK sent for Stake/cross-channel refresh");
                    }
                }
            } else if (unified_advanced && channel_unchanged && canonical_already_ahead) {
                m_logger->debug("[Solo GET_ROUND] Stake/cross-channel advance unified {} → {} "
                                "suppressed: canonical already at {} (BLOCK_DATA ahead of GET_ROUND)",
                                m_last_round_unified_height, unified_height,
                                canonical.canonical_unified_height);
            } else if (unified_advanced && channel_unchanged && !get_block_sent_in_handler
                       && !canonical_already_ahead && recent_push_for_this_height) {
                m_logger->info("[Solo GET_ROUND] ⚡ CHAIN TIP CHANGED: Stake/cross-channel advance "
                               "unified {} → {} suppressed — PUSH received within {}ms for height {} ≥ {}, "
                               "BLOCK_DATA in transit",
                               m_last_round_unified_height, unified_height,
                               push_in_transit_guard_for_channel().count(),
                               snap.push_unified_height, unified_height);
            }
        }

        // Use sync_template_state to handle: channel manager updates, fork detection,
        // template finalization, and template validation.
        // This must always run — even when a GET_BLOCK was already requested above —
        // so that any pending template channel-height metadata is finalized.
        bool template_valid = sync_template_state(unified_height, channel_height);

        // ── Self-induced NEW_ROUND tagging ──────────────────────────────────────────
        // A NEW_ROUND at the same height as our last submission that arrives within
        // 2s of a BLOCK_ACCEPTED is almost certainly our own block advancing the chain.
        // Log this so post-mortem analysis is a one-grep job (grep "SELF-induced").
        {
            const auto submitted = get_last_submitted_state_snapshot();
            const auto since_accept_ms =
                (m_last_block_accepted_time == std::chrono::steady_clock::time_point::min())
                ? int64_t{-1}
                : std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::steady_clock::now() - m_last_block_accepted_time).count();
            const bool likely_self_induced =
                (unified_height == submitted.height) &&
                (since_accept_ms >= 0) && (since_accept_ms < 2000);
            m_logger->info("[NEW_ROUND] {}-induced: unified_height={}, since_accept={}ms",
                           likely_self_induced ? "SELF" : "external",
                           unified_height, since_accept_ms);
        }

        // CRITICAL FIX: After NEW_ROUND, check if we have a valid template
        // If not, defer a recovery GET_BLOCK by 5s so that the BLOCK_DATA push that
        // the node almost always sends within ~50ms has a chance to arrive first.
        // Skip if a GET_BLOCK was already sent in this handler (staleness check above)
        // — the in-flight response will provide the replacement template.
        // Also skip if another handler (e.g. PUSH) already has a GET_BLOCK in-flight
        // for the same or higher height — prevents the duplicate-request race condition
        // where PUSH sends GET_BLOCK, resets dedup, then GET_ROUND fires 0.5-3s later
        // and sends a second GET_BLOCK before the first response arrives.
        bool needs_template = !template_valid || 
                             (m_template_interface && !m_template_interface->has_valid_template());

        // Log when a previous GET_BLOCK request timed out without a replacement
        // template arriving via PUSH/BLOCK_DATA/GET_BLOCK response. Previously this
        // expired silently, leaving operators with no diagnostic trail for lost requests.
        if (m_pending_get_block.has_timed_out()) {
            m_logger->warn("[Solo GET_ROUND] ⏱️  Previous GET_BLOCK timed out after {}ms "
                           "(reason={}, height={}) — allowing new request",
                           m_pending_get_block.elapsed_ms(),
                           reason_name(m_pending_get_block.reason),
                           m_pending_get_block.unified_height);
        }

        bool get_block_already_in_flight = m_pending_get_block.is_pending_for(unified_height);
        
        if (needs_template && !get_block_sent_in_handler && !get_block_already_in_flight) {
            if (!template_valid && m_template_interface) {
                m_logger->info("[Solo GET_ROUND] ⚡ CHAIN TIP CHANGED: {} channel advanced "
                    "(template stale → deferred recovery GET_BLOCK in 5s)",
                    get_channel_name(m_channel));
            } else {
                m_logger->info("[Solo GET_ROUND] 📭 NEW_ROUND received but no template — "
                               "deferring recovery GET_BLOCK for 5s");
                m_logger->info("[Solo GET_ROUND]   This handles legacy nodes that send NEW_ROUND without BLOCK_DATA");
            }
            schedule_recovery_get_block(connection, unified_height);
            // Note: get_block_sent_in_handler stays false — we have not sent a GET_BLOCK
            // yet, only scheduled a deferred one.  The height-parity backup path below
            // still guards on has_valid_template() so it won't double-fire.
        } else if (needs_template && get_block_sent_in_handler) {
            m_logger->debug("[Solo GET_ROUND] Template needed but GET_BLOCK already sent in this handler — waiting for response");
        } else if (needs_template && get_block_already_in_flight) {
            m_logger->debug("[Solo GET_ROUND] Template needed but GET_BLOCK already in-flight "
                            "(unified={}, reason={}, age={}ms) — suppressing duplicate",
                            m_pending_get_block.unified_height,
                            reason_name(m_pending_get_block.reason),
                            std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - m_pending_get_block.sent_at).count());
        } else {
            m_logger->debug("[Solo GET_ROUND] ✓ Template valid, continuing to mine");
        }

        // Event-driven: only request GET_BLOCK when template state or missing PUSH/BLOCK_DATA
        // says we truly need one. No unconditional GET_BLOCK here - avoids feedback loops.
        if (!get_block_sent_in_handler) {
            m_logger->debug("[Solo GET_ROUND] ✓ Template valid after NEW_ROUND, no GET_BLOCK needed");
        }

        // Height parity backup GET_BLOCK: fire when GET_ROUND confirms the node's channel tip
        // has met or passed our template's target AND push has been silent — catching
        // missed-push scenarios (especially on long Prime blocks of 300–330s).
        if (!get_block_sent_in_handler &&
            m_template_interface &&
            m_template_interface->has_valid_template() &&
            channel_height != 0)
        {
            auto tmpl = m_template_interface->get_current_template();
            if (tmpl && tmpl->nChannelHeight != 0 &&
                channel_height >= tmpl->nChannelHeight)
            {
                auto snap = m_height_tracker.GetSnapshot();
                auto now  = std::chrono::steady_clock::now();
                auto last_push = snap.last_push_notification_at;
                bool push_silent = (last_push == std::chrono::steady_clock::time_point{}) ||
                    (std::chrono::duration_cast<std::chrono::seconds>(now - last_push).count()
                         >= PUSH_ABSENT_FOR_GET_ROUND_FALLBACK_SECONDS);
                if (push_silent)
                {
                    int64_t elapsed = (last_push == std::chrono::steady_clock::time_point{})
                        ? -1
                        : std::chrono::duration_cast<std::chrono::seconds>(now - last_push).count();
                    arm_get_round_fallback(elapsed);
                    m_logger->info("[Solo GET_ROUND] ⚡ TIP CHANGE — fallback parity: node tip {} >= template target {} "
                        "(push silent {}s) — GET_ROUND triggering GET_BLOCK",
                        channel_height, tmpl->nChannelHeight, elapsed);
                    m_template_interface->discard_template(
                        "GET_ROUND height parity: node tip met template target");
                    if (connection) {
                        if (request_and_queue_get_block(connection,
                                                        GetBlockReason::GET_ROUND_HEIGHT_PARITY,
                                                        "[Solo GET_ROUND] Height parity GET_BLOCK")) {
                            get_block_sent_in_handler = true;
                            mark_authoritative_recovery_required("get_round_height_parity");
                            m_logger->info("[Solo GET_ROUND] ✓ GET_BLOCK sent (height parity backup)");
                        } else {
                            m_logger->error("[Solo GET_ROUND] Failed to generate GET_BLOCK request (height parity)");
                        }
                    } else {
                        m_logger->error("[Solo GET_ROUND] Cannot send GET_BLOCK (height parity) — connection is null");
                    }
                } else if (m_get_round_push_silent_fallback_active) {
                    disarm_get_round_fallback(
                        "PUSH active again",
                        std::chrono::duration_cast<std::chrono::seconds>(now - last_push).count());
                }
            }
        }

        // Update intelligent polling state. Treat NEW_ROUND as authoritative when
        // the unified tip advanced, even if this miner's channel height did not.
        if (unified_height == 0 || unified_height == previous_unified_height) {
            m_logger->info("[Solo GET_ROUND] NEW_ROUND received but unified height unchanged "
                           "(unified={} previous_unified={} channel={}) — polling continues at fixed {}ms interval",
                           unified_height, previous_unified_height, channel_height, POLL_INTERVAL_MIN_MS);
        }
        // Always reset to fixed interval (backoff disabled)
        on_new_round_received(unified_height);

        // Record the current channel and unified heights after the NEW_ROUND polling
        // decision so diagnostic logging can show how heights changed (or stayed flat)
        // when the authoritative unified height advanced on another channel.
        m_last_round_channel_height = channel_height;
        m_last_round_unified_height = unified_height;
    }
    else if (matches_opcode(packet, Packet::OLD_ROUND))
    {
        m_logger->info("[Solo GET_ROUND] OLD_ROUND response received (lane={})", lane_label);
        
        bool get_block_sent_in_handler = false;  // Track whether GET_BLOCK was already requested in this handler

        // ✅ ACCEPTED FORMAT: 16 bytes only — full height picture (unified + prime + hash + stake)
        if (!packet.m_data || packet.m_length != 16) {
            m_logger->error("[Solo GET_ROUND] ❌ PROTOCOL ERROR: Invalid packet length");
            m_logger->error("[Solo GET_ROUND]   Expected:  16 bytes (unified + prime + hash + stake)");
            m_logger->error("[Solo GET_ROUND]   Received:  {} bytes", packet.m_length);
            return;
        }
        
        // Parse 16-byte full-height-picture response (all big-endian)
        uint32_t unified_height = bytes2uint(*packet.m_data, 0);
        uint32_t prime_height   = bytes2uint(*packet.m_data, 4);
        uint32_t hash_height    = bytes2uint(*packet.m_data, 8);
        uint32_t stake_height   = bytes2uint(*packet.m_data, 12);
        
        // ── Normalize GET_ROUND heights: target (tip+1) → tip ──────────────
        // (see NEW_ROUND handler for rationale)
        m_logger->debug("[Solo GET_ROUND] Raw heights (target/round): unified={} prime={} hash={} stake={}",
            unified_height, prime_height, hash_height, stake_height);
        if (unified_height > 0) --unified_height;
        if (prime_height   > 0) --prime_height;
        if (hash_height    > 0) --hash_height;
        if (stake_height   > 0) --stake_height;

        // Derive active-channel height from full picture
        uint32_t channel_height = 0;
        if (m_channel == mining::CHANNEL_PRIME) {
            channel_height = prime_height;
        } else if (m_channel == mining::CHANNEL_HASH) {
            channel_height = hash_height;
        } else {
            m_logger->error("[Solo GET_ROUND] Invalid channel: {}", m_channel);
            m_logger->error("[Solo GET_ROUND] Expected 1 (Prime) or 2 (Hash), got {}", m_channel);
            return;
        }
        
        std::string channel_name = get_channel_name(m_channel);
        
        m_logger->info("[Solo GET_ROUND] ✓ OLD_ROUND (16-byte full height picture, lane={}):", lane_label);
        {
            auto canonical_snap = m_height_tracker.GetCanonicalSnapshot();
            m_logger->info("[Solo GET_ROUND]   Unified:       {} (tip{})", unified_height,
                canonical_snap.is_initialized()
                    ? fmt::format("; canonical={}", canonical_snap.canonical_unified_height)
                    : std::string{});
        }
        m_logger->info("[Solo GET_ROUND]   Prime height:  {} (tip)", prime_height);
        m_logger->info("[Solo GET_ROUND]   Hash height:   {} (tip)", hash_height);
        m_logger->info("[Solo GET_ROUND]   Stake height:  {} (tip)", stake_height);
        m_logger->info("[Solo GET_ROUND]   {} height:   {} (derived tip)", channel_name, channel_height);
        m_logger->info("[Solo GET_ROUND]   Difficulty:    (unchanged; not in 16-byte payload)");
        
        // Update RoundStatus
        m_last_round_status.is_new_round = false;
        m_last_round_status.height = unified_height;
        m_last_round_status.has_channel_heights = true;
        m_last_round_status.prime_height = prime_height;
        m_last_round_status.hash_height  = hash_height;
        m_last_round_status.stake_height = stake_height;
        
        // Update HeightTracker with full height picture (direct call — bypasses
        // update_height_state so all 4 heights reach the diagnostic state).
        m_height_tracker.OnGetRound(unified_height, prime_height, hash_height, stake_height);
        // Update ClientChannelManager for fork/phantom-stake detection (exactly once).
        apply_channel_manager_update(unified_height, channel_height);
        
        // Pass channel height to template interface for staleness validation.
        // Same invariant as NEW_ROUND: snapshot-based delta check only while pending finalization.
        if (m_template_interface) {
            m_template_interface->update_channel_height(m_channel, channel_height);

            // Only use snapshot-based delta check while template awaits finalization.
            if (m_template_interface->needs_channel_height_finalization()) {
                bool is_stale = m_template_interface->check_staleness_by_channel_delta(channel_height);
                if (is_stale) {
                    // Phase C2: PUSH × NEW_ROUND race guard (OLD_ROUND mirror).
                    auto snap_push = m_height_tracker.GetSnapshot();
                    const bool push_in_transit = push_implies_block_data_in_transit(
                        snap_push, unified_height, push_in_transit_guard_for_channel().count());
                    if (push_in_transit) {
                        m_logger->info("[Solo GET_ROUND] ⚡ CHAIN TIP CHANGED: {} channel advanced "
                            "(OLD_ROUND staleness) — GET_BLOCK suppressed: PUSH for height ≥ {} in transit",
                            get_channel_name(m_channel), unified_height);
                    } else {
                        m_logger->info("[Solo GET_ROUND] ⚡ CHAIN TIP CHANGED: {} channel advanced (awaiting finalization → GET_BLOCK)",
                            get_channel_name(m_channel));
                        if (connection) {
                            if (request_and_queue_get_block(connection,
                                                            GetBlockReason::GET_ROUND_STALE,
                                                            "[Solo GET_ROUND] OLD_ROUND GET_BLOCK")) {
                                get_block_sent_in_handler = true;
                                m_logger->info("[Solo GET_ROUND] ✓ GET_BLOCK request sent - waiting for new template...");
                            } else {
                                m_logger->debug("[Solo GET_ROUND] GET_BLOCK suppressed by dedup guard (OLD_ROUND staleness path)");
                            }
                        } else {
                            m_logger->error("[Solo GET_ROUND] Cannot request fresh template - connection is null");
                        }
                    }
                    // No early return (previous early return removed): fall through to sync_template_state().
                }
            }

            m_logger->debug("[Solo] Channel height for staleness validation: {} ({})",
                channel_height, get_channel_name(m_channel));
        }

        // ── Stake/cross-channel unified tip detection ──────────────────────────────
        // (see NEW_ROUND handler for the full explanation comment)
        // Detect when unified height advanced but our channel height did NOT change.
        // Even on OLD_ROUND responses, the unified height can advance if a Stake
        // block was mined — hashPrevBlock is now stale and a fresh template is needed.
        //
        // CANONICAL GUARD: same as NEW_ROUND — skip discard when BLOCK_DATA is ahead.
        {
            bool unified_advanced = (unified_height > m_last_round_unified_height) &&
                                    (m_last_round_unified_height > 0);
            bool channel_unchanged = (channel_height == m_last_round_channel_height);

            auto canonical = m_height_tracker.GetCanonicalSnapshot();
            bool canonical_already_ahead = canonical.is_initialized() &&
                                           canonical.canonical_unified_height >= unified_height;

            if (unified_advanced && channel_unchanged && !get_block_sent_in_handler && !canonical_already_ahead) {
                // Phase C2: PUSH × NEW_ROUND race guard (OLD_ROUND stake mirror).
                auto snap_push = m_height_tracker.GetSnapshot();
                const bool push_in_transit = push_implies_block_data_in_transit(
                    snap_push, unified_height, push_in_transit_guard_for_channel().count());
                if (push_in_transit) {
                    m_logger->info("[Solo GET_ROUND] ⚡ CHAIN TIP CHANGED: Stake/cross-channel advance "
                                   "unified {} → {} (OLD_ROUND) suppressed — PUSH for height ≥ {} in transit",
                                   m_last_round_unified_height, unified_height, unified_height);
                } else {
                    m_logger->info("[Solo GET_ROUND] ⚡ CHAIN TIP CHANGED: Stake/cross-channel advance unified {} → {} "
                                   "({} channel height unchanged — discarding stale template)",
                                   m_last_round_unified_height, unified_height,
                                   get_channel_name(m_channel));

                    // Discard template: hashPrevBlock is now stale (different tip)
                    if (m_template_interface && m_template_interface->has_valid_template()) {
                        m_template_interface->discard_template("Stake/cross-channel tip advance detected via GET_ROUND");
                        m_logger->info("[Solo GET_ROUND] ✗ Template discarded (Stake-advance stale hashPrevBlock)");
                    }

                    // Reset dedup guard so this GET_BLOCK is not suppressed
                    reset_get_block_dedup_state();

                    if (connection) {
                        if (request_and_queue_get_block(connection,
                                                        GetBlockReason::GET_ROUND_NO_TEMPLATE,
                                                        "[Solo GET_ROUND] OLD_ROUND stake refresh")) {
                            get_block_sent_in_handler = true;
                            m_logger->info("[Solo GET_ROUND] ✓ GET_BLOCK sent for Stake/cross-channel refresh");
                        }
                    }
                }
            } else if (unified_advanced && channel_unchanged && canonical_already_ahead) {
                m_logger->debug("[Solo GET_ROUND] Stake/cross-channel advance unified {} → {} "
                                "suppressed: canonical already at {} (BLOCK_DATA ahead of GET_ROUND)",
                                m_last_round_unified_height, unified_height,
                                canonical.canonical_unified_height);
            }
        }

        // Use sync_template_state to handle: channel manager updates, fork detection,
        // template finalization, and template validation
        bool template_valid = sync_template_state(unified_height, channel_height);
        
        // BUG #5 fix: Log timeout on the OLD_ROUND path as well
        if (m_pending_get_block.has_timed_out()) {
            m_logger->warn("[Solo GET_ROUND] ⏱️  Previous GET_BLOCK timed out after {}ms "
                           "(reason={}, height={}) — allowing new request (OLD_ROUND path)",
                           m_pending_get_block.elapsed_ms(),
                           reason_name(m_pending_get_block.reason),
                           m_pending_get_block.unified_height);
        }

        if (!template_valid && m_template_interface && !get_block_sent_in_handler &&
            !m_pending_get_block.is_pending_for(unified_height)) {
            // Phase C2: PUSH × NEW_ROUND race guard (OLD_ROUND template-invalid path).
            auto snap_push = m_height_tracker.GetSnapshot();
            const bool push_in_transit = push_implies_block_data_in_transit(
                snap_push, unified_height, push_in_transit_guard_for_channel().count());
            if (push_in_transit) {
                m_logger->info("[Solo GET_ROUND] ⚡ CHAIN TIP CHANGED: template invalidated on OLD_ROUND — "
                               "GET_BLOCK suppressed: PUSH for height ≥ {} in transit", unified_height);
            } else {
                m_logger->info("[Solo GET_ROUND] ⚡ CHAIN TIP CHANGED: template invalidated on OLD_ROUND → GET_BLOCK");

                // Request fresh template
                if (connection) {
                    if (request_and_queue_get_block(connection,
                                                    GetBlockReason::GET_ROUND_NO_TEMPLATE,
                                                    "[Solo GET_ROUND] OLD_ROUND template refresh")) {
                        get_block_sent_in_handler = true;
                        m_logger->info("[Solo GET_ROUND] ✓ GET_BLOCK request sent - waiting for new template...");
                    } else {
                        m_logger->debug("[Solo GET_ROUND] GET_BLOCK suppressed by dedup guard (OLD_ROUND template path)");
                    }
                }
            }
        }
        
        // Event-driven: only request GET_BLOCK when template is actually stale (handled above).
        // OLD_ROUND alone does not force a template refresh; only stale-template / parity /
        // push-silence paths above may escalate to GET_BLOCK.
        if (!get_block_sent_in_handler) {
            m_logger->debug("[Solo GET_ROUND] ✓ OLD_ROUND: no change, no GET_BLOCK needed");
        }

        // Height parity backup GET_BLOCK: fire when GET_ROUND confirms the node's channel tip
        // has met or passed our template's target AND push has been silent — catching
        // missed-push scenarios (especially on long Prime blocks of 300–330s).
        if (!get_block_sent_in_handler &&
            m_template_interface &&
            m_template_interface->has_valid_template() &&
            channel_height != 0)
        {
            auto tmpl = m_template_interface->get_current_template();
            if (tmpl && tmpl->nChannelHeight != 0 &&
                channel_height >= tmpl->nChannelHeight)
            {
                auto snap = m_height_tracker.GetSnapshot();
                auto now  = std::chrono::steady_clock::now();
                auto last_push = snap.last_push_notification_at;
                bool push_silent = (last_push == std::chrono::steady_clock::time_point{}) ||
                    (std::chrono::duration_cast<std::chrono::seconds>(now - last_push).count()
                         >= PUSH_ABSENT_FOR_GET_ROUND_FALLBACK_SECONDS);
                if (push_silent)
                {
                    int64_t elapsed = (last_push == std::chrono::steady_clock::time_point{})
                        ? -1
                        : std::chrono::duration_cast<std::chrono::seconds>(now - last_push).count();
                    arm_get_round_fallback(elapsed);
                    m_logger->info("[Solo GET_ROUND] ⚡ TIP CHANGE — fallback parity: node tip {} >= template target {} "
                        "(push silent {}s) — GET_ROUND triggering GET_BLOCK",
                        channel_height, tmpl->nChannelHeight, elapsed);
                    m_template_interface->discard_template(
                        "GET_ROUND height parity: node tip met template target");
                    if (connection) {
                        if (request_and_queue_get_block(connection,
                                                        GetBlockReason::GET_ROUND_HEIGHT_PARITY,
                                                        "[Solo GET_ROUND] OLD_ROUND height parity")) {
                            get_block_sent_in_handler = true;
                            mark_authoritative_recovery_required("get_round_height_parity");
                            m_logger->info("[Solo GET_ROUND] ✓ GET_BLOCK sent (height parity backup)");
                        } else {
                            m_logger->error("[Solo GET_ROUND] Failed to generate GET_BLOCK request (height parity)");
                        }
                    } else {
                        m_logger->error("[Solo GET_ROUND] Cannot send GET_BLOCK (height parity) — connection is null");
                    }
                } else if (m_get_round_push_silent_fallback_active) {
                    disarm_get_round_fallback(
                        "PUSH active again",
                        std::chrono::duration_cast<std::chrono::seconds>(now - last_push).count());
                }
            }
        }

        // Update intelligent polling state
        on_old_round_received();

        // Update the dedicated dedup fields so the next NEW_ROUND compares against
        // the most recent GET_ROUND heights (regardless of NEW/OLD opcode).
        m_last_round_channel_height = channel_height;
        m_last_round_unified_height = unified_height;
    }
}

void Solo::on_miner_auth_response(Packet const& packet, std::shared_ptr<network::Connection> connection)
{
    if (matches_opcode(packet, Packet::MINER_AUTH_CHALLENGE))
    {
        // Phase 2 Challenge-Response Protocol:
        // Handle MINER_AUTH_CHALLENGE from node and respond with signed nonce
        m_logger->info("[Solo Auth] Received MINER_AUTH_CHALLENGE from node");
        handle_miner_auth_challenge(packet);
    }
    else if (matches_opcode(packet, Packet::MINER_AUTH_RESULT))
    {
        // Phase 2: Handle MINER_AUTH_RESULT (auth result from node)
        // The node sends: [status(1)][session_id(4, optional, LE)]
        m_logger->info("[Solo Phase 2] Received MINER_AUTH_RESULT from node");
        
        // Enhanced diagnostics: Validate packet
        if (!packet.m_data || packet.m_length < 1) {
            m_logger->error("[Solo Auth] CRITICAL: MINER_AUTH_RESULT packet has invalid data");
            m_logger->error("[Solo Auth]   - Packet data null: {}", packet.m_data == nullptr);
            m_logger->error("[Solo Auth]   - Packet length: {}", packet.m_length);
            m_logger->error("[Solo Auth] Cannot proceed - authentication protocol error");
            return;
        }
        
        bool auth_success = (*packet.m_data)[0] != 0;
        
        m_logger->info("[Solo Auth] Authentication result:");
        m_logger->info("[Solo Auth]   - Status byte: 0x{:02x} ({})", 
            (*packet.m_data)[0], auth_success ? "SUCCESS" : "FAILURE");
        m_logger->info("[Solo Auth]   - Packet length: {} bytes", packet.m_length);
        
        if (auth_success) {
            m_authenticated = true;
            m_auth_state = AuthState::AUTHENTICATED;
            m_auth_in_flight_since = {};  // Auth complete — clear in-flight timestamp

            // Extract session ID if present (4 bytes, little-endian)
            if (packet.m_length >= 5) {
                // Read little-endian uint32
                m_session_id = SessionId(static_cast<uint32_t>((*packet.m_data)[1]) |
                               (static_cast<uint32_t>((*packet.m_data)[2]) << 8) |
                               (static_cast<uint32_t>((*packet.m_data)[3]) << 16) |
                               (static_cast<uint32_t>((*packet.m_data)[4]) << 24));

                // Validate session ID: must be non-zero for a valid session
                // Zero session ID indicates a protocol error or node-side issue
                if (m_session_id.is_default()) {
                    m_logger->error("[Solo Auth] CRITICAL: Node sent session_id = 0 (invalid)");
                    m_logger->error("[Solo Auth] This indicates a node-side bug or protocol violation");
                    m_logger->error("[Solo Auth] Valid session IDs must be non-zero");
                    m_logger->error("[Solo Auth] Cannot proceed with mining - session establishment failed");

                    // Reset authentication state
                    m_authenticated = false;
                    m_auth_state = AuthState::NOT_AUTHENTICATED;
                    m_auth_in_flight_since = {};
                    m_pending_push_after_auth = false;
                    if (m_session_context) {
                        m_session_context->reset_session_credentials();
                        m_session_context->set_falcon_identity(
                            m_miner_pubkey,
                            falcon_pubkey_to_hash_key_id(m_miner_pubkey),
                            false);
                        m_session_context->set_chacha20_session_key({}, "", false);
                    }

                    // Close connection to force re-authentication
                    if (connection) {
                        connection->close();
                    }
                    return;
                }

                // Visual box logging for success
                std::string genesis_status = (get_session_manager() && !get_session_manager()->get_tritium_genesis().empty())
                                             ? "CONFIGURED" : "NOT CONFIGURED";
                std::string chacha20_status = m_enable_chacha20 ? "ENABLED" : "DISABLED";

                // Prepare formatted strings with safe alignment
                std::stringstream pubkey_line, genesis_line, chacha20_line, session_line;
                pubkey_line << "║ Public Key:  " << m_miner_pubkey.size() << " bytes";
                genesis_line << "║ Genesis:     " << genesis_status;
                chacha20_line << "║ ChaCha20:    " << chacha20_status;
                session_line << "║ Session ID:  0x" << std::hex << std::setw(8) << std::setfill('0') << m_session_id;

                // Calculate padding (box width = 59 chars, '║' takes 1 char at end)
                auto pad_line = [](std::stringstream& ss) -> std::string {
                    std::string line = ss.str();
                    int padding = 58 - static_cast<int>(line.length());
                    if (padding < 0) padding = 0;  // Safety: never negative
                    return line + std::string(padding, ' ') + "║";
                };

                m_logger->info("╔═════════════════════════════════════════════════════════╗");
                m_logger->info("║       FALCON AUTHENTICATION SUCCESSFUL                  ║");
                m_logger->info("╠═════════════════════════════════════════════════════════╣");
                m_logger->info(pad_line(pubkey_line));
                m_logger->info(pad_line(genesis_line));
                m_logger->info(pad_line(chacha20_line));
                m_logger->info(pad_line(session_line));
                m_logger->info("╚═════════════════════════════════════════════════════════╝");

                m_logger->debug("[Solo Auth]   - Session ID bytes (LE): {:02x} {:02x} {:02x} {:02x}",
                    (*packet.m_data)[1], (*packet.m_data)[2], (*packet.m_data)[3], (*packet.m_data)[4]);

                propagate_session_to_template_interface("Solo Auth");

                // Start session in session manager
                if (m_session_context) {
                    m_session_context->commit_authenticated_session(
                        m_session_id,
                        m_miner_pubkey,
                        falcon_pubkey_to_hash_key_id(m_miner_pubkey),
                        SessionGenesisHash(load_tritium_genesis()));
                    refresh_cached_session_state("Solo Auth");
                    m_session_context->set_channel_state(m_channel, false, false);
                    m_session_context->start_keepalive_timer();
                    m_session_context->mark_activity();
                    m_logger->info("[Solo Session] Session started in session manager");
                    m_logger->info("[Solo Session] Keepalive timer started (early ping + regular cadence)");
                }

                // Update template interface with authenticated session ID (FALCON tunnel established)
                if (m_template_interface) {
                    m_template_interface->clear_template_channel_height_snapshot();
                    m_logger->info("[Solo Phase 2] FALCON tunnel established - Template interface bound to session");
                }

                // BUG FIX (Bug 2): Invoke session_authenticated handler AFTER session_id is fully set.
                // This allows Worker_manager to check session_id=0 and trigger retry at the correct time
                // (after MINER_AUTH_RESULT processing, not at login callback which fires too early).
                if (m_session_authenticated_handler) {
                    m_session_authenticated_handler(m_session_id);
                }

                if (!validate_authoritative_session("Solo Auth", false)) {
                    if (connection) {
                        connection->close();
                    }
                    return;
                }
                log_session_container_summary("Solo Auth");
            } else {
                m_logger->info("[Solo Phase 2] ✓ Authentication SUCCEEDED");
                m_logger->warn("[Solo Auth]   - WARNING: No session ID provided by node (expected 5 bytes, got {})",
                    packet.m_length);
                if (m_session_context) {
                    m_session_context->set_falcon_identity(m_miner_pubkey, falcon_pubkey_to_hash_key_id(m_miner_pubkey), true);
                    m_session_context->set_channel_state(m_channel, false, false);
                }

                // BUG FIX (Bug 2): Invoke handler even when no session ID provided (session_id will be 0).
                if (m_session_authenticated_handler) {
                    m_session_authenticated_handler(m_session_id);
                }
            }
            
            // Log port information for authenticated session
            if (connection) {
                auto const& remote_ep = connection->remote_endpoint();
                std::string remote_addr;
                remote_ep.address(remote_addr);
                uint16_t actual_port = remote_ep.port();
                
                m_logger->info("[Solo Phase 2] Authenticated session established on {}:{}", 
                    remote_addr, actual_port);
                m_logger->debug("[Solo] Port Validation: Authenticated mining session using LLP port {}", 
                    actual_port);
                    
                // Enhanced connection trace
                auto const& local_ep = connection->local_endpoint();
                std::string local_addr;
                local_ep.address(local_addr);
                uint16_t local_port = local_ep.port();
                m_logger->info("[Solo Connection] Session details:");
                m_logger->info("[Solo Connection]   - Local endpoint: {}:{}", local_addr, local_port);
                m_logger->info("[Solo Connection]   - Remote endpoint: {}:{}", remote_addr, actual_port);
                m_logger->info("[Solo Connection]   - Session ID: 0x{:08x}", m_session_id.get());
            }
            
            // Check if we have a reward address to bind
            if (!m_reward_address.empty())
            {
                m_logger->info("[Solo Phase 2] Authentication successful - binding reward address");
                m_logger->info("[Solo Phase 2] Sending MINER_SET_REWARD (required before GET_BLOCK)");
                auto reward_payload = send_set_reward();
                if (reward_payload && !reward_payload->empty() && connection)
                {
                    if (!queue_payload(connection, reward_payload, "[Solo Reward]")) {
                        m_logger->error("[Solo Reward] Failed to queue MINER_SET_REWARD payload");
                        if (connection) {
                            connection->close();
                        }
                        return;
                    }
                    // Note: SET_CHANNEL and GET_BLOCK will be sent after receiving MINER_REWARD_RESULT
                    // This is handled in handle_reward_result()
                    return;
                }
                else
                {
                    // CRITICAL: Failed to send reward address - cannot proceed with mining
                    m_logger->error("[Solo Reward] CRITICAL: Failed to send reward address");
                    m_logger->error("[Solo Reward] Mining cannot proceed without reward address binding");
                    m_logger->error("[Solo Reward] This typically indicates:");
                    m_logger->error("[Solo Reward]   - Invalid reward address format");
                    m_logger->error("[Solo Reward]   - ChaCha20 encryption failure");
                    m_logger->error("[Solo Reward]   - Missing tritium genesis for encryption");
                    
                    // Close connection to prevent invalid mining state
                    if (connection) {
                        connection->close();
                    }
                    return;
                }
            }
            
            // Now send SET_CHANNEL since we're authenticated (no reward binding)
            m_logger->info("[Solo Phase 2] No reward address configured - proceeding to SET_CHANNEL");
            send_set_channel(connection);
        }
        else {
            m_authenticated = false;
            m_auth_state = AuthState::NOT_AUTHENTICATED;
            m_auth_in_flight_since = {};
            m_pending_push_after_auth = false;
            if (m_session_context) {
                m_session_context->reset_session_credentials();
                m_session_context->set_falcon_identity(
                    m_miner_pubkey,
                    falcon_pubkey_to_hash_key_id(m_miner_pubkey),
                    false);
                m_session_context->set_chacha20_session_key({}, "", false);
                m_session_context->set_channel_state(m_channel, false, false);
            }
            
            // Visual box logging for failure
            uint8_t error_code = (packet.m_length >= 2) ? (*packet.m_data)[1] : 0x00;
            std::string error_message;
            std::string troubleshooting;
            
            // Interpret error codes based on LLL-TAO implementation
            switch (error_code) {
                case 0x01:
                    error_message = "Public key not whitelisted";
                    troubleshooting = "Add to nexus.conf minerallowkey";
                    break;
                case 0x02:
                    error_message = "Signature verification failed";
                    troubleshooting = "Check key pair in miner.conf";
                    break;
                case 0x03:
                    error_message = "Invalid message format";
                    troubleshooting = "Check protocol version";
                    break;
                case 0x04:
                    error_message = "Timestamp out of range";
                    troubleshooting = "Synchronize system clocks";
                    break;
                default: {
                    std::stringstream ss;
                    ss << "Unknown error 0x" << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(error_code);
                    error_message = ss.str();
                    troubleshooting = "Check node logs";
                    break;
                }
            }
            
            // Prepare formatted strings with safe alignment
            std::stringstream status_line, error_line, action_line;
            status_line << "║ Status:  0x" << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>((*packet.m_data)[0]);
            error_line << "║ Error:   " << error_message;
            action_line << "║ Action:  " << troubleshooting;
            
            // Calculate padding (box width = 59 chars)
            auto pad_line = [](std::stringstream& ss) -> std::string {
                std::string line = ss.str();
                int padding = 58 - static_cast<int>(line.length());
                if (padding < 0) padding = 0;  // Safety: never negative
                return line + std::string(padding, ' ') + "║";
            };
            
            m_logger->error("╔═════════════════════════════════════════════════════════╗");
            m_logger->error("║       FALCON AUTHENTICATION FAILED                      ║");
            m_logger->error("╠═════════════════════════════════════════════════════════╣");
            m_logger->error(pad_line(status_line));
            m_logger->error(pad_line(error_line));
            m_logger->error(pad_line(action_line));
            m_logger->error("╚═════════════════════════════════════════════════════════╝");
            
            // Log authentication attempt details for debugging
            m_logger->error("[Solo Auth] Authentication attempt details:");
            m_logger->error("[Solo Auth]   - Address used: '{}'", m_address);
            m_logger->error("[Solo Auth]   - Timestamp used: {} (0x{:016x})", m_auth_timestamp, m_auth_timestamp);
            m_logger->error("[Solo Auth]   - Public key size: {} bytes", m_miner_pubkey.size());
            
            // Enhanced diagnostics: Log timestamp synchronization info
            uint64_t current_time = static_cast<uint64_t>(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
            int64_t time_delta = static_cast<int64_t>(current_time) - static_cast<int64_t>(m_auth_timestamp);
            m_logger->error("[Solo Auth] Timestamp synchronization check:");
            m_logger->error("[Solo Auth]   - Current time: {} (0x{:016x})", current_time, current_time);
            m_logger->error("[Solo Auth]   - Auth timestamp: {} (0x{:016x})", m_auth_timestamp, m_auth_timestamp);
            m_logger->error("[Solo Auth]   - Time delta: {} seconds", time_delta);
            if (std::abs(time_delta) > 60) {
                m_logger->error("[Solo Auth] WARNING: Large time delta detected! This may indicate clock drift or replay attack protection.");
            }
            
            // Enhanced diagnostics: Log public key fingerprint for troubleshooting
            if (m_miner_pubkey.size() >= 16) {
                m_logger->error("[Solo Auth] Public key fingerprint (first 16 bytes):");
                m_logger->error("[Solo Auth]   {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x}",
                    m_miner_pubkey[0], m_miner_pubkey[1], m_miner_pubkey[2], m_miner_pubkey[3],
                    m_miner_pubkey[4], m_miner_pubkey[5], m_miner_pubkey[6], m_miner_pubkey[7],
                    m_miner_pubkey[8], m_miner_pubkey[9], m_miner_pubkey[10], m_miner_pubkey[11],
                    m_miner_pubkey[12], m_miner_pubkey[13], m_miner_pubkey[14], m_miner_pubkey[15]);
            }
            
            m_logger->error("[Solo Auth] Possible causes:");
            m_logger->error("[Solo Auth]   - Public key not whitelisted on node (check nexus.conf -minerallowkey)");
            m_logger->error("[Solo Auth]   - Invalid key format in miner.conf (must be valid hex strings)");
            m_logger->error("[Solo Auth]   - Falcon signature verification failed (key mismatch or corruption)");
            m_logger->error("[Solo Auth]   - Node missing Phase 2 stateless miner support");
            m_logger->error("[Solo Auth]   - Timestamp drift between miner and node (check system clocks)");
            m_logger->error("[Solo Auth] Mining cannot proceed without valid authentication");
            m_logger->error("[Solo Auth] Please verify:");
            m_logger->error("[Solo Auth]   1. Your public key is whitelisted: nexus.conf -minerallowkey=<pubkey>");
            m_logger->error("[Solo Auth]   2. Keys in miner.conf match the whitelisted key");
            m_logger->error("[Solo Auth]   3. Node is running LLL-TAO with Phase 2 miner support");
            m_logger->error("[Solo Auth]   4. System clocks are synchronized between miner and node");
            
            // Connection will fail - no fallback available
        }
    }
    else if (matches_opcode(packet, Packet::CHANNEL_ACK))
    {
        // Phase 2: Handle CHANNEL_ACK response with dynamic port detection
        m_logger->info("[Solo Phase 2] Received CHANNEL_ACK from node");
        
        // Enhanced connection trace: Log actual connected port information
        if (connection) {
            auto const& remote_ep = connection->remote_endpoint();
            auto const& local_ep = connection->local_endpoint();
            std::string remote_addr;
            std::string local_addr;
            remote_ep.address(remote_addr);
            local_ep.address(local_addr);
            uint16_t actual_port = remote_ep.port();
            uint16_t local_port = local_ep.port();
            
            m_logger->info("[Solo Connection] CHANNEL_ACK connection details:");
            m_logger->info("[Solo Connection]   - Local: {}:{}", local_addr, local_port);
            m_logger->info("[Solo Connection]   - Remote: {}:{}", remote_addr, actual_port);
            m_logger->info("[Solo] Dynamic Port Detection: Successfully connected to {}:{}", 
                remote_addr, actual_port);
            m_logger->debug("[Solo] Port Validation: Using dynamically detected LLP port {}", actual_port);
        } else {
            m_logger->warn("[Solo Connection] WARNING: Connection object is null during CHANNEL_ACK");
        }
        
        // Parse channel acknowledgment with enhanced diagnostics
        if (packet.m_data && packet.m_length >= 1) {
            uint8_t acked_channel = (*packet.m_data)[0];
            m_logger->info("[Solo] Channel acknowledged: {} ({})", 
                static_cast<int>(acked_channel), 
                (acked_channel == 1) ? "prime" : "hash");
            
            // Validate acknowledged channel matches requested channel
            if (acked_channel != m_channel) {
                m_logger->warn("[Solo] WARNING: Channel mismatch detected!");
                m_logger->warn("[Solo]   - Requested channel: {} ({})", 
                    static_cast<int>(m_channel), (m_channel == 1) ? "prime" : "hash");
                m_logger->warn("[Solo]   - Acknowledged channel: {} ({})", 
                    static_cast<int>(acked_channel), (acked_channel == 1) ? "prime" : "hash");
            }
            
            // Update template interface with confirmed channel
            if (m_template_interface) {
                m_template_interface->set_channel(acked_channel);
                m_logger->info("[Solo] Template interface updated with confirmed channel");
            }
            
            // Extended CHANNEL_ACK: Check for optional port information (future protocol enhancement)
            // Format (if extended): [channel(1)][port(2, big-endian, optional)]
            if (packet.m_length >= 3) {
                // Node is sending back port information (future enhancement)
                uint16_t node_port = (static_cast<uint16_t>((*packet.m_data)[1]) << 8) | 
                                     static_cast<uint16_t>((*packet.m_data)[2]);
                
                m_logger->info("[Solo] Extended CHANNEL_ACK: Node communicated LLP port: {}", node_port);
                
                // Validate against actual connected port
                if (connection) {
                    uint16_t actual_port = connection->remote_endpoint().port();
                    if (node_port != actual_port) {
                        m_logger->warn("[Solo] Port mismatch detected:");
                        m_logger->warn("[Solo]   - Node advertised port: {}", node_port);
                        m_logger->warn("[Solo]   - Actually connected to port: {}", actual_port);
                        m_logger->info("[Solo] Recovery: Continuing with actual connection (port {})", actual_port);
                    } else {
                        m_logger->info("[Solo] Port validation successful: Both using port {}", actual_port);
                    }
                }
            }
        } else {
            m_logger->warn("[Solo] WARNING: CHANNEL_ACK packet has no data or insufficient length");
            m_logger->warn("[Solo]   - Packet data null: {}", packet.m_data == nullptr);
            m_logger->warn("[Solo]   - Packet length: {}", packet.m_length);
        }
        
        // Verify reward address is bound before requesting work (if configured)
        if (!m_reward_address.empty() && !m_reward_bound)
        {
            m_logger->error("[Solo] Cannot GET_BLOCK - reward address not bound yet");
            m_logger->error("[Solo] This indicates a protocol flow error:");
            m_logger->error("[Solo]   - MINER_REWARD_RESULT may have been lost or not received");
            m_logger->error("[Solo]   - SET_CHANNEL sent before reward binding completed");
            m_logger->error("[Solo]   - Network issue during reward binding handshake");
            m_logger->error("[Solo] Expected flow: Auth → MINER_SET_REWARD → MINER_REWARD_RESULT → SET_CHANNEL → CHANNEL_ACK → GET_BLOCK");
            m_logger->error("[Solo] Recovery: Disconnect and reconnect to retry authentication");
            
            // Close connection to force re-authentication
            if (connection) {
                connection->close();
            }
            return;
        }
        
        // ═══════════════════════════════════════════════════════════════════
        // LANE-BASED PROTOCOL FLOW (NO NEGOTIATION)
        // ═══════════════════════════════════════════════════════════════════
        
        if (m_protocol_lane == ProtocolLane::UNKNOWN) {
            m_logger->error("[Solo Protocol] UNKNOWN protocol lane - cannot proceed");
            if (connection) {
                connection->close();
            }
            return;
        }

        if (!activate_push_lane_after_channel_ack(connection)) {
            return;
        }
    }
    else if (matches_opcode(packet, Packet::SESSION_START))
    {
        // LLL-TAO SESSION_START handler (session parameters from node)
        // ACTUAL Wire Format: [success(1B: 0x01)][session_id(4B, LE)][timeout(4B, LE)][optional: genesis_hash(32B)]
        // NOTE: Earlier documentation incorrectly stated success byte was only in MINER_AUTH_RESULT.
        //       The node actually sends it again in SESSION_START along with the session_id.
        m_logger->info("[Solo Session] Received SESSION_START from node");

        // Defensive check: SESSION_START should only be processed after successful authentication
        // This guards against node-side bugs where SESSION_START might be sent after auth rejection
        if (!is_authenticated()) {
            m_logger->error("[Solo Session] Rejecting SESSION_START - not authenticated");
            m_logger->error("[Solo Session] This indicates a node-side protocol violation");
            m_logger->error("[Solo Session] SESSION_START should only be sent after MINER_AUTH_RESULT success");
            return;
        }

        // Parse SESSION_START packet using dedicated parser
        if (!packet.m_data) {
            m_logger->error("[Solo Session] SESSION_START packet has null data");
            return;
        }

        auto parsed = parse_session_start(packet.m_data->data(), packet.m_length);
        if (!parsed) {
            m_logger->error("[Solo Session] Failed to parse SESSION_START packet");
            m_logger->error("[Solo Session]   - Packet length: {} bytes", packet.m_length);
            m_logger->error("[Solo Session]   - Expected: minimum 9 bytes (success+session_id+timeout), or 41 bytes (with genesis)");
            return;
        }

        // Validate success byte
        if (parsed->success != 0x01) {
            m_logger->error("[Solo Session] Invalid success byte in SESSION_START: 0x{:02x} (expected 0x01)",
                          parsed->success);
            return;
        }

        // Validate session ID matches what we received in MINER_AUTH_RESULT
        if (parsed->session_id != get_session_id()) {
            m_logger->error("[Solo Session] Session ID mismatch in SESSION_START:");
            m_logger->error("[Solo Session]   - Expected: 0x{:08x} (from MINER_AUTH_RESULT)", m_session_id.get());
            m_logger->error("[Solo Session]   - Received: 0x{:08x} (from SESSION_START)", parsed->session_id.get());
            return;
        }

        // Log parsed session parameters
        m_logger->info("[Solo Session] Session parameters:");
        m_logger->info("[Solo Session]   - Success: 0x{:02x}", parsed->success);
        m_logger->info("[Solo Session]   - Session ID: 0x{:08x}", parsed->session_id.get());
        m_logger->info("[Solo Session]   - Timeout: {} seconds ({} hours)",
                      parsed->timeout_seconds, parsed->timeout_seconds / 3600);

        if (parsed->has_genesis_hash()) {
            m_logger->info("[Solo Session]   - Tritium Genesis from node: {} bytes",
                          parsed->genesis_hash->size());
            if (m_session_context) {
                m_session_context->set_tritium_genesis(*parsed->genesis_hash);
            }

            // Cross-check node-echoed session genesis against configured reward_address.
            // The miner currently requires the session signer to own the coinbase
            // recipient (no end-to-end DynamicGenesis). A mismatch produces a node-side
            // Coinbase::Verify / signature failure that, on the wire, is just a
            // BLOCK_REJECTED with no reason byte. Catch it here, refuse to submit, and
            // log a precise message instead.
            //
            // Wire-format note (coordinated fix with LLL-TAO PR #578):
            //
            //   Prior to LLL-TAO PR #578 the node emitted the genesis hash via
            //   base_uint::GetBytes(), which serialises pn[0..7] LSW-first — opposite
            //   to the GetHex() display order used by 'system/get/info'.  The miner was
            //   compensating with genesis_utils::swap_genesis_word_order() to convert
            //   the wire bytes to display order before the comparison.
            //
            //   LLL-TAO PR #578 fixed SessionStartPacket::BuildPayload to emit genesis
            //   in display/MSW-first order (identical to GetHex()).  Both sides now use
            //   the same byte order, so the comparison below is a straight equality
            //   check with no word-swap needed.  swap_genesis_word_order() is retained
            //   in genesis_utils.hpp for historical reference and testing but is no
            //   longer called on this code path.
            if (!m_reward_address.empty()
                && parsed->genesis_hash->size() == 32
                && m_reward_address.length() == 64)
            {
                std::vector<uint8_t> reward_bytes =
                    genesis_utils::hex_decode_genesis_hash(m_reward_address);
                // Node sends genesis in display/MSW-first order (LLL-TAO PR #578).
                // Direct comparison is correct; no word-swap needed.
                const auto& node_genesis_display = *parsed->genesis_hash;
                if (reward_bytes.size() == 32 && reward_bytes != node_genesis_display) {
                    auto bytes_to_hex = [](const std::vector<uint8_t>& b) {
                        std::string s; s.reserve(b.size() * 2);
                        static const char* kHex = "0123456789abcdef";
                        for (uint8_t v : b) {
                            s.push_back(kHex[v >> 4]);
                            s.push_back(kHex[v & 0x0f]);
                        }
                        return s;
                    };
                    const std::string node_hex = bytes_to_hex(node_genesis_display);
                    m_reward_genesis_mismatch = true;
                    m_logger->error("[Solo Session] reward_address GENESIS MISMATCH:");
                    m_logger->error("[Solo Session]   - reward_address (configured): {}", m_reward_address);
                    m_logger->error("[Solo Session]   - node session genesis:        {}", node_hex);
                    m_logger->error("[Solo Session] The miner will REFUSE to submit blocks until this is"
                                    " corrected. The session signer (node auto-login) must own the coinbase"
                                    " recipient, otherwise mined blocks fail Coinbase::Verify and are"
                                    " rejected with no reason byte. Set mining.reward_address to the"
                                    " genesis hash reported by 'system/get/info' on the node.");
                } else if (reward_bytes.size() == 32) {
                    m_reward_genesis_mismatch = false;
                    m_logger->debug("[Solo Session] reward_address matches node session genesis ✓");
                }
            }
        }

        // Adjust keepalive interval based on timeout (ping at 1/N of timeout)
        // Using KEEPALIVE_SAFETY_DIVISOR=4 ensures 4 keepalives per node timeout window.
        // Example: 24h node timeout → keepalive every 6h (4 pings/window)
        if (parsed->timeout_seconds > 0 && get_session_manager()) {
            uint16_t keepalive_hours = calculate_keepalive_hours(
                parsed->timeout_seconds, ProtocolConstants::KEEPALIVE_SAFETY_DIVISOR);
            get_session_manager()->set_keepalive_interval(keepalive_hours);
            m_logger->info("[Solo Session] Keepalive interval adjusted to {} hours (node timeout={}s, {} pings/window)",
                          keepalive_hours, parsed->timeout_seconds, ProtocolConstants::KEEPALIVE_SAFETY_DIVISOR);
            if (m_session_start_handler) {
                m_session_start_handler(keepalive_hours);
            }
        }

        refresh_cached_session_state("Solo SessionStart");
        if (!validate_authoritative_session("Solo SessionStart", false)) {
            return;
        }
        log_session_container_summary("Solo SessionStart");
    }
    else if (matches_opcode(packet, Packet::SESSION_KEEPALIVE))
    {
        // KEEPALIVE receive handler: unified 32-byte response
        //   Provides unified_height / prime_height / hash_height / stake_height / fork_score
        m_logger->debug("[Solo Session] Received SESSION_KEEPALIVE response ({} bytes)", packet.m_length);

        if (packet.m_data && packet.m_length == 32) {
            // ── Unified 32-byte keepalive reply — parse using KeepaliveAckFrame ──
            ::LLP::KeepaliveAckFrame unified;
            if (unified.Parse(*packet.m_data))
            {
                PacketIngressPreflightOptions preflight;
                preflight.owner = &m_last_keepalive_request_owner;
                preflight.packet_session_id = SessionId(unified.session_id);
                if (!run_packet_ingress_preflight("Solo SessionKeepalive", preflight)) {
                    return;
                }

                m_height_tracker.OnKeepaliveResponse(unified.unified_height,
                                                      unified.prime_height,
                                                      unified.hash_height,
                                                      unified.stake_height,
                                                      unified.hash_tip_lo32,
                                                      unified.fork_score);

                m_logger->debug("[Solo Keepalive] Unified reply received:"
                               " session=0x{:08x}"
                               " unified={} prime={} hash={} stake={}"
                               " hash_tip_lo32=0x{:08x} fork_score={}",
                    unified.session_id,
                    unified.unified_height, unified.prime_height,
                    unified.hash_height, unified.stake_height,
                    unified.hash_tip_lo32, unified.fork_score);

                finalize_keepalive_ack("keepalive ack accepted");

                // Fork canary cross-check
                // Diagnostic-only: PUSH notification system handles real chain tip advances.
                if (unified.IsForkDetected(m_last_keepalive_prevhash_lo32))
                {
                    m_logger->warn("[SESSION_KEEPALIVE] Fork canary triggered:"
                                   " miner_prevHash_lo32=0x{:08x} node_tip_lo32=0x{:08x} fork_score={}",
                        m_last_keepalive_prevhash_lo32, unified.hash_tip_lo32, unified.fork_score);
                }

            }
        } else if (packet.m_length != 0) {
            // Unexpected payload length — ignore gracefully
            m_logger->debug("[Solo Session] Unexpected KEEPALIVE payload length {} — ignored", packet.m_length);
        }
    }
    else if (matches_opcode(packet, Packet::MINER_REWARD_RESULT))
    {
        // Phase 2: Handle MINER_REWARD_RESULT (reward binding result from node)
        handle_reward_result(packet);
    }
}

void Solo::on_session_expired(Packet const& packet, std::shared_ptr<network::Connection> connection)
{
        // SESSION_EXPIRED handler (LLL-TAO PR #354)
        // Node notifies miner that session has expired (inactivity timeout)
        // Wire format: [session_id(4B, LE)][reason(1B)]
        m_logger->info("[Solo Session] Received SESSION_EXPIRED from node");

        // Validate payload size
        if (!packet.m_data || packet.m_length < 5) {
            m_logger->error("[Solo Session] SESSION_EXPIRED packet too small: {} bytes (expected 5)",
                          packet.m_length);
            m_logger->error("[Solo Session]   Expected: [session_id(4B LE)][reason(1B)]");
            return;
        }

        // Parse session_id (little-endian uint32)
        SessionId expired_sid(static_cast<uint32_t>((*packet.m_data)[0]) |
                               (static_cast<uint32_t>((*packet.m_data)[1]) << 8) |
                               (static_cast<uint32_t>((*packet.m_data)[2]) << 16) |
                               (static_cast<uint32_t>((*packet.m_data)[3]) << 24));

        // Parse reason code
        uint8_t reason = (*packet.m_data)[4];

        // Delegate to handler
        handle_session_expired(expired_sid, reason, connection);
}

void Solo::on_push_notification(Packet const& packet, std::shared_ptr<network::Connection> connection, uint32_t channel)
{
    disarm_get_round_fallback("PUSH re-established");

    // Bug 11 fix: Lightweight session validation for PUSH notifications.
    // PUSH is processed regardless (it's a broadcast), but log a warning if the
    // session is not authenticated — this detects stale/mismatched PUSH data
    // from a previous session that could inject incorrect template data.
    if (!is_authenticated()) {
        m_logger->warn("[Solo PUSH] Received push notification while NOT authenticated — "
                       "data may be from a stale session (session_id=0x{:08x})", get_session_id().get());
    }

    const char* push_opcode_name = (channel == mining::CHANNEL_PRIME) ? "PRIME_BLOCK_AVAILABLE"
                                 : (channel == mining::CHANNEL_HASH)  ? "HASH_BLOCK_AVAILABLE"
                                 : "STAKE_BLOCK_AVAILABLE";

    // Capture whether the handler substantively processed the push
    // (heights/state updated) vs just recorded liveness.
    // Same-channel: always true (every same-channel PUSH updates state).
    // Cross-channel tip advance: true (unified height moved → state updated).
    // Cross-channel liveness-only (same unified height): false — no state change.
    //
    // NODE auto-sends BLOCK_DATA after PUSH — no GET_BLOCK request needed.
    bool push_processed = m_push_handler->handle_push_notification(
        packet, channel, m_protocol_lane,
        m_template_interface.get(),
        &m_height_tracker,
        // update_height_fn — Updates cached height state
        [this](uint32_t u, uint32_t c, uint32_t d) {
            update_height_state(u, c, d, HeightTracker::UpdateSource::PUSH);
        });

    if (push_processed) {
        m_logger->debug("[Solo Push] PUSH processed ({} channel) — "
                        "node will auto-send fresh BLOCK_DATA",
            (channel == m_channel) ? "same" : "cross");
    }

    // If this push left the template marked replacement-pending (same-height
    // tip update), notify Worker_manager with the deadline so it can arm a
    // precise one-shot timeout check rather than waiting on the next slower
    // general-purpose template-health poll tick.
    if (m_template_interface && m_replacement_pending_handler) {
        if (auto deadline = m_template_interface->get_replacement_pending_deadline()) {
            m_replacement_pending_handler(*deadline);
        }
    }

    // Record push timestamp and cancel any pending NEW_ROUND recovery debounce.
    // A PUSH arriving means the node has signalled a new block — BLOCK_DATA will
    // follow automatically, so a deferred recovery GET_BLOCK is no longer needed.
    m_last_push_received_time = std::chrono::steady_clock::now();
    cancel_recovery_timer(push_opcode_name);
}

void Solo::on_get_block_template(Packet const& packet, std::shared_ptr<network::Connection> connection)
{
    // Phase A2: clear in-flight GET_BLOCK state only AFTER validating the
    // payload is non-null.  Clearing up-front would let a node spamming
    // null GET_BLOCK responses reset the in-flight guard on every frame.
    if (!packet.m_data) {
        m_logger->error("[Solo GET_BLOCK] Null packet data — empty response");
        m_pending_get_block.clear();
        dispatch_recovery_or_fallback(connection,
                                      GetBlockReason::VALIDATION_FAILURE,
                                      "[Solo GET_BLOCK] Null-data recovery GET_BLOCK");
        return;
    }
    m_pending_get_block.clear();

    const bool stateless = packet.m_is_uint16_opcode;
    const char* source_name = stateless ? "STATELESS_GET_BLOCK (0xD081)" : "LEGACY_GET_BLOCK (0x81)";
    const char* protocol_mode = stateless ? "Stateless Push" : "Legacy Push";

    // Unified handler for initial template response
    handle_initial_template_response(source_name);

    // (m_pending_get_block.clear() is now at the top of on_get_block_template)
    
    // ═══════════════════════════════════════════════════════════════════
    // ENHANCED DIAGNOSTICS: Template delivery tracking
    // ═══════════════════════════════════════════════════════════════════
    m_logger->debug("[Solo Template Delivery] ═══════════════════════════════════");
    m_logger->debug("[Solo Template Delivery] 📥 TEMPLATE RECEIVED VIA: {}", source_name);
        m_logger->debug("[Solo Template Delivery]   Delivery Method: {} opcode",
                        stateless ? "Stateless 16-bit" : "Legacy 8-bit");
        m_logger->debug("[Solo Template Delivery]   Payload Size: {} bytes", packet.m_length);
        m_logger->debug("[Solo Template Delivery]   Expected Format: 12 metadata + 216 block");
        m_logger->debug("[Solo Template Delivery]   Protocol Mode: {}", protocol_mode);
        m_logger->debug("[Solo Template Delivery] ═══════════════════════════════════");
        
        m_logger->debug("[Solo GET_BLOCK] ✨ {} received! {} bytes", source_name, packet.m_length);

        if (!m_template_interface) {
            m_logger->error("[Solo GET_BLOCK] No template interface available!");
            return;
        }

        // ── Decode via StatelessBlockUtility::decode_template() ─────────────────
        // decode_template() validates the 228-byte size, extracts the 12-byte
        // metadata prefix (diagnostic: unified_height, channel_height, nBits), and
        // delegates the 216-byte block body decode to
        // MiningTemplateInterface::read_stateless_payload().  All canonical mining
        // state (nHeight, nChannel, nBits, hashPrevBlock) comes from the block body.
        // (Null-data already rejected at the top of this handler.)
        auto decoded = StatelessBlockUtility::decode_template(
            *m_template_interface, *packet.m_data, m_channel, m_logger, false);

        if (!decoded.valid) {
            m_logger->error("[Solo GET_BLOCK] Template decode failed: {}", decoded.error_message);
            m_logger->error("[Solo GET_BLOCK] Recovery: Delegating to recovery handler");
            dispatch_recovery_or_fallback(connection,
                                          GetBlockReason::VALIDATION_FAILURE,
                                          "[Solo GET_BLOCK] Decode-fail recovery GET_BLOCK");
            return;
        }

        uint32_t unified_height = decoded.unified_height;
        uint32_t channel_height = decoded.channel_height;
        uint32_t difficulty     = decoded.difficulty_nbits;

        m_logger->info("[Solo GET_BLOCK] 📦 Metadata: unified={} channel={} nBits=0x{:08x}",
                       unified_height, channel_height, difficulty);
        if (!decoded.channel_consistent) {
            m_logger->warn("[Solo GET_BLOCK] ⚠️  Channel mismatch: block.nChannel={} vs mining channel={}",
                            decoded.block.nChannel, m_channel);
        }
        if (!decoded.metadata_consistent) {
            m_logger->warn("[Solo GET_BLOCK] ⚠️  Height mismatch: block.nHeight={} vs unified_height+1={}",
                            decoded.block.nHeight, unified_height + 1);
        }


        // ── HeightTracker feed (TEMPLATE source) ────────────────────────────────
        // Registers unified/channel heights as TEMPLATE source so last_template_update
        // timestamp is set — the post-push guard in check_template_health() uses this
        // to suppress false-positive emergency stops when a GET_BLOCK response arrives
        // after a push notification.
        update_height_state(unified_height, channel_height, difficulty,
                            HeightTracker::UpdateSource::TEMPLATE);

        // ── channel_target: use raw BLOCK_DATA metadata channel_height (Bug #5 fix) ─
        // Previously used max(metadata, composite tracker) which allowed stale push
        // data to inflate the channel_target. BLOCK_DATA metadata is authoritative.
        if (channel_height > 0) {
            m_height_tracker.OnTemplateReceived(m_channel, channel_height + 1);
            m_logger->info("[Solo GET_BLOCK] HeightTracker fed: unified={} channel={} nBits=0x{:08x} → channel_target={}",
                unified_height, channel_height, difficulty, channel_height + 1);
        }

        if (!finalize_and_feed_current_template(unified_height,
                                                 channel_height,
                                                 "Solo GET_BLOCK",
                                                 false)) {
            m_logger->error("[Solo GET_BLOCK] Failed to finalize decoded template");
            return;
        }
        cancel_get_block_control_retry(false);

        m_logger->info("[Solo GET_BLOCK] 🎯 Template ready! Mining for height {} (channel {})",
                       unified_height, channel_height);
}

void Solo::on_ping_diag(Packet const& packet, std::shared_ptr<network::Connection> connection)
{
        /* PING_DIAG is stateless-only — reject on legacy lane */
        if(m_protocol_lane != ProtocolLane::STATELESS)
        {
            m_logger->warn("[Colin PING] PING_DIAG rejected on non-stateless lane — stateless port required");
            return;
        }

        /* Exact payload size enforcement for fixed-size PING_DIAG opcode */
        static const std::vector<uint8_t> empty_ping_vec;
        const auto& payload = packet.m_data ? *packet.m_data : empty_ping_vec;
        uint32_t nExpected = ::LLP::GetExpectedPayloadSize(::LLP::ColinDiagOpcodes::PING_DIAG);
        if(payload.size() != nExpected)
        {
            m_logger->warn("[Colin PING] Payload size mismatch for PING_DIAG:"
                           " expected {} bytes, got {} — discarding",
                nExpected, payload.size());
            return;
        }

        /* Parse 64-byte PingFrame, build telemetry-enriched PongFrame, reply immediately */
        auto pong_bytes = m_colin_ping_handler.HandlePing(payload, true /* stateless */);
        if(!pong_bytes.empty() && connection)
        {
            /* PONG opcode: stateless-only diagnostic pong (mirror-mapped by PacketBuilder) */
            if (queue_payload(connection,
                              PacketBuilder::build(m_protocol_lane,
                                                   static_cast<uint8_t>(::LLP::ColinDiagOpcodes::PONG_DIAG_LEGACY),
                                                   pong_bytes),
                              "[Solo PONG]")) {
                m_logger->debug("[Colin PING] PongFrame transmitted (seq #{})",
                    m_colin_ping_handler.last_received_ping().sequence);
            }
        }
}

void Solo::on_session_status_ack(Packet const& packet, std::shared_ptr<network::Connection> connection)
{
        static const std::vector<uint8_t> empty_ack_vec;
        const auto& data = packet.m_data ? *packet.m_data : empty_ack_vec;
        ::LLP::SessionStatusAckFrame ack;
        if(ack.Parse(data))
        {
            // SESSION_STATUS_ACK is a slow-cycle (60s) telemetry probe.  Do NOT run
            // preflight ownership checks or session-id mismatch guards here — a slightly
            // late ACK (ownership epoch rolled between send and receive) would silently
            // drop the entire ACK and score a false mismatch against the session.
            // PUSH notification liveness is the authoritative session-alive signal.

            m_logger->info("[Solo] SESSION_STATUS_ACK: lane_health=0x{:04x} uptime={}s "
                           "primary={} secondary={} simlink={} auth={}",
                           ack.lane_health_flags, ack.uptime_seconds,
                           ack.IsPrimaryAlive(), ack.IsSecondaryAlive(),
                           ack.IsSimLinkActive(), ack.IsAuthenticated());

            if (m_session_context) {
                m_session_context->note_keepalive_ack(true, "session status ack accepted");
            }
            record_session_event(SessionManager::SessionEventKind::STATUS_ACK_ACCEPTED,
                                 "session status ack accepted");

            m_last_session_status_ack      = ack;
            m_last_session_status_ack_time = std::chrono::steady_clock::now();

            const auto decision = SessionStatusPolicy::evaluate_ack_health(
                { ack.uptime_seconds, ack.IsAuthenticated() });
            if (decision.force_reauth) {
                // SESSION_STATUS is a telemetry probe.  A single bad payload (uptime==0 or
                // auth==false from a transient node race) must NOT kill mining workers.
                // PUSH notification liveness is the authoritative session-alive signal.
                record_session_event(SessionManager::SessionEventKind::STATUS_ACK_REJECTED,
                                     "session status ack unhealthy (logged only — push is authoritative): "
                                     + decision.reason);
                m_logger->warn("[Solo] SESSION_STATUS_ACK reports unhealthy session "
                               "(uptime={}s auth={}) — noted, PUSH is authoritative, session preserved",
                               ack.uptime_seconds, ack.IsAuthenticated());
                // DO NOT call m_session_expired_handler() here
            }
        }
        else
        {
            m_logger->warn("[Solo] SESSION_STATUS_ACK: malformed payload (size={})", data.size());
        }
}

void Solo::set_miner_keys(std::vector<uint8_t> const& pubkey, std::vector<uint8_t> const& privkey)
{
    m_miner_pubkey = pubkey;
    m_miner_privkey = privkey;
    
    // Detect Falcon version from key size
    bool is_falcon1024 = (pubkey.size() == FalconConstants::FALCON1024_PUBKEY_SIZE);
    std::string version = is_falcon1024 ? "Falcon-1024" : "Falcon-512";
    size_t expected_sig_size = is_falcon1024 ? 
        FalconConstants::FALCON1024_SIG_CT_SIZE : FalconConstants::FALCON512_SIG_CT_SIZE;
    
    m_logger->info("[Solo] Miner {} keys configured", version);
    m_logger->info("[Solo]   Public key:  {} bytes", pubkey.size());
    m_logger->info("[Solo]   Private key: {} bytes", privkey.size());
    m_logger->info("[Solo]   Signature:   {} bytes (CT)", expected_sig_size);
    
    // Initialize the Unified Falcon Signature Wrapper
    try {
        m_falcon_wrapper = std::make_unique<FalconSignatureWrapper>(pubkey, privkey);
        if (m_falcon_wrapper->is_valid()) {
            m_logger->info("[Solo] ✓ Falcon wrapper initialized and valid");
            m_logger->info("[Solo]   Signature size: {} bytes", m_falcon_wrapper->get_signature_size());
        } else {
            m_logger->error("[Solo] ✗ Falcon wrapper NOT valid - block submission will fail!");
            m_logger->error("[Solo]   Check miner.conf for valid falcon_public_key and falcon_private_key");
            m_falcon_wrapper.reset();
        }
    } catch (const std::exception& e) {
        m_logger->error("[Solo] ✗ Failed to initialize Falcon Signature Wrapper: {}", e.what());
        m_logger->error("[Solo]   Check miner.conf for valid falcon_public_key and falcon_private_key");
        m_falcon_wrapper.reset();
    }
}

void Solo::set_protocol_lane(ProtocolLane lane)
{
    m_protocol_lane = lane;
    
    // Also set protocol lane in SessionManager for keepalive packet generation
    if (m_session_context) {
        m_session_context->set_protocol_lane(lane);
        m_session_context->mark_activity();
    }
}

network::Shared_payload Solo::send_session_keepalive()
{
    m_logger->debug("[Solo Session] Sending SESSION_KEEPALIVE for session 0x{:08x}", get_session_id().get());

    // Delegate to SessionManager which builds the correct 8-byte v2 payload:
    //   [0..3] session_id             (u32 little-endian)
    //   [4..7] miner_prevblock_suffix (last 4 bytes of hashPrevBlock, raw bytes;
    //                                  zeros when no valid template is available)
    // This causes the node to reply with the 32-byte unified KeepAliveV2AckFrame
    // (unified_height / prime_height / hash_height / stake_height / hash_tip_lo32 / fork_score).
    auto payload = get_session_manager()->build_keepalive_packet();
    if (payload && !payload->empty()) {
        m_last_keepalive_request_owner = capture_session_ownership();
    }
    return payload;
}

network::Shared_payload Solo::build_session_status_packet(
    bool degraded, bool workers_running, bool secondary_up) const
{
    bool has_tmpl = m_template_interface && m_template_interface->has_valid_template();
    auto payload = get_session_manager()->build_session_status_packet(
        degraded, has_tmpl, workers_running, secondary_up);
    if (payload && !payload->empty()) {
        m_last_session_status_request_owner = capture_session_ownership();
    }
    return payload;
}

void Solo::send_set_channel(std::shared_ptr<network::Connection> connection)
{
    std::string channel_name = (m_channel == 1) ? "prime" : "hash";
    m_logger->info("[Solo] Sending SET_CHANNEL channel={} ({})", static_cast<int>(m_channel), channel_name);
    if (m_session_context) {
        m_session_context->set_channel_state(m_channel, false, false);
        m_session_context->mark_activity();
    }
    
    std::vector<uint8_t> channel_data(1, m_channel);
    auto payload = PacketBuilder::build(m_protocol_lane, LLP::SET_CHANNEL, channel_data);
    if (!queue_payload(connection, payload, "[Solo] SET_CHANNEL")) {
        m_logger->error("[Solo] SET_CHANNEL payload was not queued");
    }
}

void Solo::set_tritium_genesis(std::vector<uint8_t> const& genesis)
{
    if (genesis.size() != 32) {
        m_logger->warn("[Solo] Invalid Tritium genesis size: {} (expected 32 bytes)", genesis.size());
        return;
    }
    
    // Store persistently to survive reconnections
    m_persistent_tritium_genesis = genesis;
    
    if (m_session_context) {
        m_session_context->set_tritium_genesis(genesis);
        m_logger->info("[Solo] Tritium genesis hash configured for reward binding");
    }
}

void Solo::set_reward_address(std::string const& address)
{
    m_reward_address = address;
    if (m_session_context) {
        m_session_context->set_reward_binding(address,
                                              RewardHash{},
                                              false,
                                              address.empty() ? "" : "config");
    }
}

bool Solo::has_tritium_genesis() const
{
    // Check persistent storage first
    if (!m_persistent_tritium_genesis.empty()) {
        return true;
    }
    
    if (get_session_manager()) {
        return !get_session_manager()->get_tritium_genesis().empty();
    }
    return false;
}

void Solo::set_keepalive_interval(std::uint16_t hours)
{
    if (get_session_manager()) {
        get_session_manager()->set_keepalive_interval(hours);
        m_logger->info("[Solo] Keepalive interval set to {} hours", hours);
    }
}

SessionId Solo::get_session_id() const
{
    if (m_session_context) {
        auto* sm = m_session_context->get_session_manager().get();
        if (sm) return sm->get_session_id();
    }
    return m_session_id;  // Fallback (pre-commit or no session context)
}

bool Solo::is_session_active() const
{
    if (m_session_context) {
        auto* sm = m_session_context->get_session_manager().get();
        if (sm) return sm->is_active();
    }
    return m_authenticated;  // Fallback
}

void Solo::set_connection(std::shared_ptr<network::Connection> connection)
{
    m_connection = std::move(connection);
    if (m_session_context) {
        m_session_context->set_connection(m_connection);
        update_connection_metadata(m_connection);
    }
}

void Solo::reset_auth_state()
{
    cancel_get_block_control_retry();
    m_auth_state = AuthState::NOT_AUTHENTICATED;
    m_auth_in_flight_since = {};
    m_authenticated = false;
    if (m_session_context) {
        m_session_context->reset_session_credentials();
        m_session_context->set_falcon_identity(
            m_miner_pubkey,
            falcon_pubkey_to_hash_key_id(m_miner_pubkey),
            false);
    }
    m_logger->info("[Solo] Auth state reset (in-band re-auth prep)");
}

bool Solo::check_auth_in_flight_timeout(const char* context)
{
    if (is_auth_in_progress() &&
        m_auth_in_flight_since != std::chrono::steady_clock::time_point{} &&
        std::chrono::steady_clock::now() - m_auth_in_flight_since > std::chrono::seconds(AUTH_IN_FLIGHT_TIMEOUT_S)) {
        m_logger->warn("[{}] Auth in-flight timeout (>{}s) — resetting to NOT_AUTHENTICATED",
                       context, AUTH_IN_FLIGHT_TIMEOUT_S);
        m_auth_state = AuthState::NOT_AUTHENTICATED;
        m_auth_in_flight_since = {};
        return true;
    }
    return false;
}

bool Solo::handle_session_id_mismatch(SessionId ack_session_id)
{
    auto* session_manager = get_session_manager();
    const SessionId authoritative_session_id = session_manager ? session_manager->get_session_id() : SessionId{};
    const auto decision = SessionStatusPolicy::validate_ack({
        session_manager != nullptr,
        authoritative_session_id,
        ack_session_id,
        m_session_id_mismatch_count,
        protocol::ProtocolConstants::SESSION_MISMATCH_EXPIRE_THRESHOLD
    });

    m_session_id_mismatch_count = decision.mismatch_count;
    if (decision.accept_ack) {
        return false;
    }

    record_session_event(SessionManager::SessionEventKind::STATUS_ACK_REJECTED, decision.reason);
    if (m_session_context) {
        m_session_context->note_keepalive_ack(false, decision.reason);
    }

    // Bug 3 fix: After N consecutive mismatches, trigger soft re-authentication.
    // Persistent mismatches indicate the NODE assigned a new session_id (e.g., after
    // a session sweep) and the miner's cached session is stale.  Without re-auth,
    // the session silently dies once PUSH notifications also stop.
    if (m_session_id_mismatch_count >= protocol::ProtocolConstants::SESSION_MISMATCH_EXPIRE_THRESHOLD) {
        m_logger->error("[KEEPALIVE_V2] {} #{}: ack=0x{:08x} != authoritative=0x{:08x}"
                       " — threshold reached, triggering re-authentication",
            decision.reason, m_session_id_mismatch_count, ack_session_id.get(), authoritative_session_id.get());

        // Reset counter to prevent re-triggering on every subsequent mismatch
        m_session_id_mismatch_count = 0;

        // Force re-auth via the same path as SESSION_EXPIRED
        if (m_session_expired_handler) {
            m_session_expired_handler();
        }
    } else {
        m_logger->warn("[KEEPALIVE_V2] {} #{}: ack=0x{:08x} != authoritative=0x{:08x}"
                       " — tracking mismatch (threshold={})",
            decision.reason, m_session_id_mismatch_count, ack_session_id.get(), authoritative_session_id.get(),
            protocol::ProtocolConstants::SESSION_MISMATCH_EXPIRE_THRESHOLD);
    }

    return true;  // mismatch detected — caller must return to skip further ACK processing
}

void Solo::handle_session_expired(SessionId expired_sid, uint8_t reason, std::shared_ptr<network::Connection> connection)
{
    // ═══════════════════════════════════════════════════════════════════════════
    // SESSION_EXPIRED HANDLER (5-step response flow per LLL-TAO PR #354)
    // ═══════════════════════════════════════════════════════════════════════════

    // STEP 1: LOG & VERIFY session_id against authoritative session state.
    // Use SessionRecoveryPolicy to make the stale-replay guard explicit and
    // consistent with the authoritative session machine.
    m_logger->warn("[Solo] SESSION_EXPIRED received: session_id=0x{:08x} reason=0x{:02x}",
                   expired_sid.get(), reason);

    const SessionId authoritative_session_id = get_session_id();
    const auto recovery_decision = SessionRecoveryPolicy::evaluate_session_expired({
        m_session_context != nullptr,  // has_authoritative_session
        expired_sid,                   // expired_session_id
        authoritative_session_id,      // authoritative_session_id
        reason                         // reason_code
    });

    if (recovery_decision.is_stale_replay) {
        m_logger->warn("[Solo] SESSION_EXPIRED stale or replay — ignoring: {} "
                       "(expired=0x{:08x} authoritative=0x{:08x})",
                       recovery_decision.reason, expired_sid.get(), authoritative_session_id.get());
        return;
    }

    // Log reason code
    const char* reason_str = "UNKNOWN";
    if (reason == static_cast<uint8_t>(LLP::StatelessMining::SessionExpiredReason::EXPIRED_INACTIVITY)) {
        reason_str = "EXPIRED_INACTIVITY";
    }
    m_logger->warn("[Solo] Session 0x{:08x} expired: reason={} ({}) — {}",
                  authoritative_session_id.get(), reason_str, reason, recovery_decision.reason);

    // STEP 2: CLEAR LOCAL SESSION STATE (mirror reset_auth_state)
    m_logger->info("[Solo] Clearing local session state");
    cancel_get_block_control_retry();
    m_current_height = 0;
    m_current_reward = 0;
    m_session_id.clear();
    m_authenticated = false;
    m_auth_state = AuthState::NOT_AUTHENTICATED;
    m_auth_in_flight_since = {};
    m_current_height = 0;
    m_current_reward = 0;
    m_reward_bound = false;  // Reward binding dies with session
    m_reward_genesis_mismatch = false;  // Re-evaluated after re-auth + SESSION_START
    m_subscribed_to_notifications = false;

    // Bug 1 fix: Clear in-flight GET_BLOCK so recovery can immediately
    // request a new template instead of being blocked for up to 4 seconds.
    m_pending_get_block.clear();

    // Clear the authoritative session context
    if (m_session_context) {
        m_session_context->set_chacha20_session_key({}, "", false);
        m_session_context->set_falcon_identity(m_miner_pubkey, falcon_pubkey_to_hash_key_id(m_miner_pubkey), false);
        m_session_context->clear_for_reauth(m_reward_address,
                                            m_reward_address.empty() ? "" : "config",
                                            "node signalled session expiry");
        m_logger->info("[Solo] Session context cleared");
    }

    // DO NOT close the connection — the node kept it open deliberately
    // (The whole point of PR-C is to re-auth on the same TCP connection)

    // STEP 3: STOP WORKERS (via callback)
    // The m_session_expired_handler is registered by Worker_manager to pause workers
    // (no valid template/session to work on)
    if (m_session_expired_handler) {
        m_logger->info("[Solo] Invoking session_expired_handler to stop workers");
        m_session_expired_handler();
    }

    // STEP 4 & 5: EXPONENTIAL BACKOFF THEN RE-AUTH
    // The Worker_manager's session_expired_handler callback will handle:
    // - Exponential backoff using MAX_SESSION_AUTH_RETRIES / BASE_SESSION_RETRY_MS / MAX_SESSION_RETRY_MS
    // - Incrementing retry counter
    // - Scheduling io_context timer to delay re-auth
    // - Calling login(m_login_handler) to re-send MINER_AUTH_INIT on same TCP connection
    //
    // The retry counter will be reset to 0 on next successful SESSION_START (handled separately)

    m_logger->warn("[Solo] SESSION_EXPIRED handling complete — waiting for Worker_manager to re-authenticate");
}

void Solo::handle_miner_auth_challenge(const Packet& packet)
{
    m_logger->info("[Solo Phase 2] Received MINER_AUTH_CHALLENGE");
    
    // TRAINING WHEELS: Show hex dump of MINER_AUTH_CHALLENGE
    if (packet.m_data) {
        m_logger->info("[Solo Auth] MINER_AUTH_CHALLENGE hex dump:");
        m_logger->info("\n{}", format_llp_payload_hexdump(packet.m_data, 128));
    }
    
    // Defensive bounds check
    if (!packet.m_data || packet.m_data->size() < 2) {
        m_logger->error("[Solo Phase 2] MINER_AUTH_CHALLENGE too small: {} bytes", 
                       packet.m_data ? packet.m_data->size() : 0);
        reset_auth_state();
        return;
    }
    
    // Parse nonce_len (2 bytes, big-endian)
    uint16_t nonce_len = (static_cast<uint16_t>((*packet.m_data)[0]) << 8) |
                          static_cast<uint16_t>((*packet.m_data)[1]);
    
    m_logger->info("[Solo Auth] Challenge nonce length: {} bytes (big-endian encoding)", nonce_len);
    m_logger->info("[Solo Auth]   Bytes 0-1: {:02x} {:02x}", (*packet.m_data)[0], (*packet.m_data)[1]);

    if (nonce_len == 0) {
        m_logger->error("[Solo Phase 2] MINER_AUTH_CHALLENGE: nonce_len is zero (invalid)");
        reset_auth_state();
        return;
    }
    
    if (packet.m_data->size() < static_cast<size_t>(2 + nonce_len)) {
        m_logger->error("[Solo Phase 2] MINER_AUTH_CHALLENGE: incomplete nonce (expected {} bytes, got {})", 
                       2 + nonce_len, packet.m_data->size());
        reset_auth_state();
        return;
    }
    
    // Extract nonce
    std::vector<uint8_t> nonce(packet.m_data->begin() + 2, packet.m_data->begin() + 2 + nonce_len);
    
    m_logger->info("[Solo Phase 2] Extracted nonce: {} bytes", nonce.size());
    
    // Show first 16 bytes of nonce for verification
    if (nonce.size() > 0) {
        std::ostringstream nonce_hex;
        nonce_hex << std::hex << std::setfill('0');
        size_t preview_len = std::min(nonce.size(), static_cast<size_t>(16));
        for (size_t i = 0; i < preview_len; ++i) {
            nonce_hex << std::setw(2) << static_cast<unsigned int>(nonce[i]) << " ";
        }
        m_logger->info("[Solo Auth] Nonce (first {} bytes): {}", preview_len, nonce_hex.str());
    }
    
    // Sign the NONCE (not address+timestamp!)
    if (!m_falcon_wrapper || !m_falcon_wrapper->is_valid()) {
        m_logger->error("[Solo Phase 2] Falcon wrapper not initialized");
        reset_auth_state();
        return;
    }
    
    auto sign_result = m_falcon_wrapper->sign_payload(nonce, FalconSignatureWrapper::SignatureType::AUTHENTICATION);
    if (!sign_result.success) {
        m_logger->error("[Solo Phase 2] Failed to sign nonce: {}", sign_result.error_message);
        reset_auth_state();
        return;
    }
    
    // Validate signature size
    size_t expected_sig_size = m_falcon_wrapper->get_signature_size();
    std::string falcon_version = m_falcon_wrapper->is_falcon1024() ? "Falcon-1024" : "Falcon-512";
    
    m_logger->info("[Solo Phase 2] Signed nonce with {}", falcon_version);
    m_logger->info("[Solo Phase 2]   Signature: {} bytes (expected: {})", 
                   sign_result.signature.size(), expected_sig_size);
    
    if (sign_result.signature.size() != expected_sig_size) {
        m_logger->error("[Solo Phase 2] SIGNATURE SIZE MISMATCH!");
        m_logger->error("[Solo Phase 2]   Expected: {} bytes ({})", expected_sig_size, falcon_version);
        m_logger->error("[Solo Phase 2]   Got: {} bytes", sign_result.signature.size());
        m_logger->error("[Solo Phase 2]   This indicates a key version mismatch");
        reset_auth_state();
        return;
    }
    
    // Build MINER_AUTH_RESPONSE packet using PacketBuilder
    network::Payload response_payload;
    
    // NOTE: MINER_AUTH_RESPONSE uses little-endian encoding per protocol specification
    // sig_len (2 bytes, little-endian)
    uint16_t sig_len = static_cast<uint16_t>(sign_result.signature.size());
    response_payload.push_back(static_cast<uint8_t>(sig_len & 0xFF));
    response_payload.push_back(static_cast<uint8_t>((sig_len >> 8) & 0xFF));
    
    // signature
    response_payload.insert(response_payload.end(), 
                             sign_result.signature.begin(), 
                             sign_result.signature.end());
    
    // Debug: Verify payload size before transmission
    m_logger->debug("[Solo Auth] MINER_AUTH_RESPONSE payload: length={}", response_payload.size());
    
    m_logger->info("[Solo Phase 2] Sending MINER_AUTH_RESPONSE: sig_len={}, total_size={}", 
                   sig_len, response_payload.size());
    
    // Build and validate the response packet
    auto bytes = PacketBuilder::build(m_protocol_lane, LLP::MINER_AUTH_RESPONSE, response_payload);
    if (!bytes || bytes->empty()) {
        m_logger->error("[Solo Auth] CRITICAL: MINER_AUTH_RESPONSE PacketBuilder::build returned null/empty!");
        m_logger->error("[Solo Phase 2] Failed to serialize MINER_AUTH_RESPONSE packet");
        reset_auth_state();
        return;
    }
    
    m_logger->debug("[Solo Auth] MINER_AUTH_RESPONSE validation: SUCCESS - encoded {} bytes", bytes->size());
    
    // TRAINING WHEELS: Show hex dump of MINER_AUTH_RESPONSE packet
    m_logger->info("[Solo Auth] MINER_AUTH_RESPONSE packet hex dump (wire format):");
    m_logger->info("\n{}", format_llp_payload_hexdump(bytes, 128));
    
    // Set state to waiting for result
    m_auth_state = AuthState::WAITING_FOR_RESULT;
    
    // Transmit the response using stored connection
    if (m_connection) {
        if (!queue_payload(m_connection, bytes, "[Solo Auth] MINER_AUTH_RESPONSE")) {
            reset_auth_state();
        }
    } else {
        m_logger->error("[Solo Phase 2] Cannot send MINER_AUTH_RESPONSE - no connection stored");
        reset_auth_state();
    }
}

network::Shared_payload Solo::send_set_reward()
{
    refresh_cached_session_state("Solo RewardSend");

    // Verify we have a reward address configured
    if (m_reward_address.empty()) {
        m_logger->warn("[Solo Reward] No reward address configured - skipping MINER_SET_REWARD");
        return nullptr;
    }
    
    // Verify we are authenticated (ChaCha20 encryption requires established session)
    if (!is_authenticated()) {
        m_logger->error("[Solo Reward] Cannot send reward address - not authenticated");
        return nullptr;
    }
    
    m_logger->info("[Solo Reward] Sending MINER_SET_REWARD (encrypted)");
    m_logger->info("[Solo Reward]   Genesis Hash: {}", m_reward_address);

    // reward_address must be a 64-character hex Tritium genesis hash
    if (m_reward_address.length() != 64) {
        m_logger->error("[Solo Reward] reward_address must be a 64-char hex genesis hash (got {} chars)",
                        m_reward_address.length());
        if (genesis_utils::looks_like_base58_address(m_reward_address)) {
            m_logger->error("[Solo Reward] This looks like a Base58 account address — use the 64-char"
                            " genesis hash from 'system/get/info' instead");
        }
        return nullptr;
    }

    // Hex-decode the 64-char genesis hash to 32 bytes
    std::vector<uint8_t> vHash = genesis_utils::hex_decode_genesis_hash(m_reward_address);
    if (vHash.size() != 32) {
        m_logger->error("[Solo Reward] Failed to hex-decode genesis hash — ensure it contains only hex characters");
        return nullptr;
    }

    // Validate genesis user-type byte: Coinbase::Verify on the node hard-rejects any
    // genesis whose leading byte does not match the network user-type (0xa1 mainnet,
    // 0xb1 testnet — see genesis_utils.hpp). Fail fast here instead of letting the
    // node reject a mined block with no reason byte.
    if (!genesis_utils::has_mainnet_genesis_type(vHash)
        && !genesis_utils::has_testnet_genesis_type(vHash)) {
        m_logger->error("[Solo Reward] Genesis hash leading byte is 0x{:02x} — expected 0xa1 (mainnet)"
                        " or 0xb1 (testnet). Block rewards would be rejected by the node's"
                        " Coinbase::Verify. Refusing to send MINER_SET_REWARD."
                        " Verify reward_address matches the genesis hash returned by"
                        " 'system/get/info' on the node you are connected to.",
                        static_cast<unsigned int>(vHash[0]));
        return nullptr;
    }

    // Build the payload - the hash bytes (will be encrypted by ChaCha20)
    std::vector<uint8_t> payload_data;

    if (m_session_context) {
        m_session_context->begin_reward_binding(m_reward_address, vHash, "config");
        m_session_context->mark_activity();
    }
    
    // Log the decoded genesis hash for debugging
    m_logger->info("[Solo Reward] Genesis hash (32 bytes): {}", m_reward_address);
    m_logger->info("[Solo Reward] Hash size: {} bytes", vHash.size());
    
    // If ChaCha20 encryption is enabled, encrypt the address
    if (m_enable_chacha20)
    {
        if (!m_chacha20_wrapper)
        {
            m_logger->error("[Solo Reward] ChaCha20 enabled but wrapper not initialized — cannot send reward unencrypted");
            return nullptr;  // hard fail — do NOT send unencrypted
        }
        // Use the authoritative session key from the session container.
        const auto reward_readiness = m_session_context
            ? m_session_context->get_reward_bind_readiness()
            : SessionManager::RewardBindReadiness{false, "session context unavailable"};
        if (!reward_readiness.ready) {
            m_logger->error("[Solo Reward] Cannot send MINER_SET_REWARD: {}",
                            reward_readiness.reason);
            return nullptr;
        }
        const auto binding = m_session_context ? m_session_context->get_session_binding()
                                               : SessionBinding{};
        const auto& reward_session_key = binding.chacha20_session_key;
        if (!binding.has_crypto_context()) {
            m_logger->error("[Solo Reward] Cannot send MINER_SET_REWARD: authoritative session key is not ready");
            return nullptr;
        }

        try {
            auto nonce = ChaCha20Wrapper::generate_nonce();

            // Encrypt the 32-byte hash (NOT the 37-byte address!)
            auto encrypt_result = m_chacha20_wrapper->encrypt(vHash, reward_session_key, nonce, AAD_REWARD_ADDRESS);

            if (encrypt_result.success)
            {
                // Build encrypted format: nonce(12) + ciphertext+tag
                payload_data.insert(payload_data.end(), nonce.begin(), nonce.end());
                payload_data.insert(payload_data.end(), encrypt_result.data.begin(), encrypt_result.data.end());

                m_logger->info("[Solo Reward] Address encrypted: {} → {} bytes (using authoritative session key)",
                               vHash.size(), payload_data.size());
                m_logger->debug("[Solo Reward] Encrypted with AAD: REWARD_ADDRESS ({} bytes)", AAD_REWARD_ADDRESS.size());
            }
            else
            {
                m_logger->error("[Solo Reward] ChaCha20 encryption failed: {}", encrypt_result.error_message);
                return nullptr;
            }
        }
        catch (const std::exception& e) {
            m_logger->error("[Solo Reward] Encryption failed: {}", e.what());
            return nullptr;
        }
    }
    else
    {
        // Send unencrypted (only valid for localhost connections)
        m_logger->warn("[Solo Reward] ChaCha20 not enabled - sending reward address unencrypted");
        payload_data = vHash;
    }
    
    if (!validate_authoritative_session("Solo RewardSend", false)) {
        return nullptr;
    }
    log_session_container_summary("Solo RewardSend");

    // Build the MINER_SET_REWARD packet via PacketBuilder only after the
    // authoritative session guard passes, so invalid session state cannot
    // silently prepare an outbound reward packet.
    m_logger->info("[Solo Reward] MINER_SET_REWARD packet built: {} bytes", payload_data.size());
    auto payload = PacketBuilder::build(m_protocol_lane, LLP::MINER_SET_REWARD, payload_data);
    if (payload && !payload->empty()) {
        m_last_reward_request_owner = capture_session_ownership();
        record_session_event(SessionManager::SessionEventKind::REWARD_BIND_SENT,
                             "reward bind request sent");
    }
    return payload;
}

network::Shared_payload Solo::send_miner_ready()
{
    const char* lane_name = get_lane_name(m_protocol_lane);
    const auto opcode_str = format_lane_opcode(m_protocol_lane, LLP::MINER_READY);

    m_logger->info("[Solo Push] Sending MINER_READY (subscribe to push notifications)");
    m_logger->info("[Solo Push]   Lane: {}, Opcode: {}", lane_name, opcode_str);
    m_logger->info("[Solo Push]   Channel: {} ({})", 
                   m_channel, 
                   m_channel == mining::CHANNEL_PRIME ? "Prime" : "Hash");
    
    // MINER_READY is a header-only packet (no payload); use PacketBuilder for lane-aware framing
    auto payload = PacketBuilder::build(m_protocol_lane, LLP::MINER_READY);
    if (payload && !payload->empty()) {
        m_subscribed_to_notifications = true;
        m_logger->info("[Solo Push] ✓ Subscribed to push notifications ({} lane)", lane_name);
        
        m_logger->info("[Solo Push] MINER_READY packet hex dump:");
        m_logger->debug("\n{}", format_llp_payload_hexdump(payload, 16));
    } else {
        m_logger->error("[Solo Push] Failed to encode MINER_READY packet");
    }
    
    return payload;
}

void Solo::resubscribe_push_notifications()
{
    if (!m_connection) {
        m_logger->warn("[Solo Push] resubscribe_push_notifications: no connection available — skipping");
        return;
    }
    if (!is_authenticated()) {
        m_logger->warn("[Solo Push] resubscribe_push_notifications: not authenticated — skipping");
        return;
    }
    m_logger->warn("[Solo Push] Re-subscribing to push notifications (MINER_READY re-send)");
    auto payload = send_miner_ready();
    if (payload && !payload->empty()) {
        if (queue_payload(m_connection, payload, "[Solo Push] MINER_READY re-subscription")) {
            m_logger->warn("[Solo Push] ✓ MINER_READY re-subscription transmitted");
        }
    } else {
        m_logger->error("[Solo Push] Failed to build MINER_READY for re-subscription");
    }
}

bool Solo::finalize_template_with_channel_height(uint32_t node_channel_height, const std::string& context)
{
    if (!m_template_interface || node_channel_height == 0) {
        return false;
    }
    
    // Sets MiningTemplate::nChannelHeight metadata for staleness detection only.
    // DOES NOT modify block.nHeight (which must remain unified height from 216-byte template bytes).
    if (m_template_interface->needs_channel_height_finalization()) {
        // Template builds NEXT block, so channel height = node height + 1
        uint32_t template_channel_height = node_channel_height + 1;
        m_template_interface->set_channel_height(template_channel_height);
        m_logger->info("[Solo GET_ROUND] ✓ Template channel metadata set to {} ({})", 
            template_channel_height, context);
        
        // Diagnostic: confirm block.nHeight (unified) was NOT overwritten by set_channel_height()
        auto const* tmpl = m_template_interface->get_current_template();
        if (tmpl)
        {
            m_logger->debug("[Solo GET_ROUND]   block.nHeight = {} (unified, unchanged)", tmpl->block.nHeight);
            m_logger->debug("[Solo GET_ROUND]   nChannelHeight = {} (metadata only, NOT in block bytes)", template_channel_height);

            // Gap 1 (legacy lane): snapshot hashPrevBlock for parity with stateless lane.
            m_last_known_hash_prev_block = tmpl->block.hashPrevBlock;
            m_height_tracker.UpdateWithHashPrevBlock(tmpl->block.hashPrevBlock);

            // Gap 3: Update session keepalive prevblock_suffix on legacy GET_ROUND lane
            // (mirrors the stateless lane at BLOCK_DATA parse time) so the fork-canary
            // suffix is always current regardless of which protocol lane delivered the template.
            if (get_session_manager()) {
                auto suffix_bytes = m_last_known_hash_prev_block.GetBytes();
                std::array<uint8_t, 4> suffix{};
                if (suffix_bytes.size() >= 4) {
                    // BUG FIX: Use bytes 0-3 (same as node's hash_tip_lo32) for fork canary alignment
                    suffix = { suffix_bytes[0], suffix_bytes[1], suffix_bytes[2], suffix_bytes[3] };
                }
                get_session_manager()->set_prevblock_suffix(suffix);
                m_last_keepalive_prevhash_lo32 =
                    (uint32_t(suffix[0]) << 24) | (uint32_t(suffix[1]) << 16)
                  | (uint32_t(suffix[2]) <<  8) | uint32_t(suffix[3]);
            }

            auto prev_bytes = m_last_known_hash_prev_block.GetBytes();
            std::string prev_hex;
            for (size_t i = 0; i < std::min(prev_bytes.size(), size_t(8)); ++i) {
                char buf[3];
                snprintf(buf, sizeof(buf), "%02x", prev_bytes[i]);
                prev_hex += buf;
            }
            m_logger->info("[TEMPLATE ANCHOR] hashPrevBlock = {}... (tip anchor at template creation)", prev_hex);
            m_logger->info("[TEMPLATE ANCHOR] block.nHeight = {} (unified blockchain height)", tmpl->block.nHeight);
        }
        return true;
    }
    
    return false;
}

mining::ClientChannelManager* Solo::get_channel_manager() const
{
    return get_channel_manager(m_channel);
}

mining::ClientChannelManager* Solo::get_channel_manager(uint32_t channel) const
{
    switch (channel) {
        case mining::CHANNEL_PRIME:
            return m_prime_manager.get();
        case mining::CHANNEL_HASH:
            return m_hash_manager.get();
        default:
            return nullptr;
    }
}

bool Solo::sync_template_state(uint32_t unified_height, uint32_t channel_height)
{
    // Get the appropriate channel manager for this miner's channel
    auto* pManager = get_channel_manager();
    if (!pManager) {
        m_logger->error("[Solo Sync] No channel manager for channel {}", m_channel);
        return false;
    }
    
    std::string channel_name = pManager->GetChannelName();
    m_logger->debug("[Solo Sync] Synchronizing {} channel state:", channel_name);
    m_logger->debug("[Solo Sync]   Node unified height: {}", unified_height);
    m_logger->debug("[Solo Sync]   Node channel height: {}", channel_height);
    
    // Note: Heights are updated in update_height_state() before this is called.
    // Fork detection flag is set there too; check and handle it here.
    
    // Step 1: Check for fork detection (height regression detected by update_height_state)
    if (pManager->IsForkDetected()) {
        handle_fork_detected(pManager, unified_height);
        return false;  // Template was invalidated
    }
    
    // Step 2: Finalize template channel height if needed
    if (m_template_interface && m_template_interface->needs_channel_height_finalization()) {
        uint32_t template_channel_height = channel_height + 1;
        m_template_interface->set_channel_height(template_channel_height);
        m_logger->info("[Solo Sync] ✓ Template finalized: channel metadata set to {} (NOT written to block.nHeight)", 
            template_channel_height);
        
        // Diagnostic: confirm block.nHeight (unified) was NOT overwritten
        auto const* tmpl_ptr = m_template_interface->get_current_template();
        if (tmpl_ptr)
        {
            m_logger->debug("[Solo Sync]   block.nHeight = {} (unified, unchanged)", tmpl_ptr->block.nHeight);
            m_logger->debug("[Solo Sync]   nChannelHeight = {} (metadata only, NOT in block bytes)", template_channel_height);
        }
    }
    
    // Step 3: Validate current template using HeightTracker (single source of truth)
    return validate_current_template();
}

bool Solo::validate_current_template()
{
    if (!m_template_interface || !m_template_interface->has_valid_template()) {
        return true;  // No template to validate - that's OK
    }
    
    // Get current template
    auto const* tmpl = m_template_interface->get_current_template();
    if (!tmpl) {
        return true;
    }
    
    // Use HeightTracker snapshot for staleness decision (single source of truth).
    // ClientChannelManager heights are kept in sync by update_height_state() but
    // are NOT used for staleness decisions; only HeightTracker is authoritative here.
    auto snap = m_height_tracker.GetSnapshot();
    uint32_t expectedChannel = snap.expected_template_target();
    
    // ── PRIMARY GATE: Unified Height regression guard ─────────────────────
    // BUG #6 fix: Unified Height from BLOCK_DATA is the essential gate for
    // template swap decisions.  Channel height is secondary and non-blocking
    // because we must allow same-channel-height template refresh when
    // hashPrevBlock changes (e.g., PRIME/HASH channel templates swapped for
    // fresh ones with a new tip).
    //
    // block.nHeight = TARGET (TIP + 1).  canonical_unified_height = TIP.
    // A template is stale if its TARGET ≤ the canonical TIP (the block it
    // targets has already been mined or is at the current tip level).
    if (snap.canonical_unified_height > 0 && tmpl->block.nHeight > 0) {
        if (tmpl->block.nHeight <= snap.canonical_unified_height) {
            m_logger->warn("[Solo Validate] Template unified height {} (TARGET) already at or below "
                           "canonical tip {} — discarding (unified height regression)",
                           tmpl->block.nHeight, snap.canonical_unified_height);
            m_template_interface->discard_template("Unified height regression");
            return false;
        }
    }

    // ── SECONDARY: Channel height check (INFORMATIONAL ONLY, not a rejection gate) ──
    // BUG #6 fix: Previously this was a hard rejection gate using canonical_channel_height.
    // This caused false rejections when canonical_channel_height was inflated by a
    // partially-processed BLOCK_DATA (metadata extracted but template rejected).
    // Same-height channel templates with a new hashPrevBlock were incorrectly rejected
    // as "already surpassed."
    //
    // Now: Unified Height is the primary gate (above).  Channel height is logged
    // for operator diagnostics but never blocks template adoption.  The hashPrevBlock
    // guard (below) handles fork/reorg detection.
    uint32_t authoritative_channel_height = snap.canonical_channel_height;
    if (expectedChannel != 0 && tmpl->nChannelHeight != 0) {
        if (tmpl->nChannelHeight <= authoritative_channel_height) {
            // Informational only — do NOT discard.  The unified height gate above
            // is the authoritative rejection mechanism.
            m_logger->info("[Solo Validate] Channel height note: template target {} <= canonical "
                           "channel tip {} (non-blocking — unified height gate is authoritative)",
                           tmpl->nChannelHeight, authoritative_channel_height);
        }
        // Note: nChannelHeight > expectedChannel (more than 1 ahead) is VALID during burst
        // recovery — log informationally so operators can observe burst lag without alarming.
        if (tmpl->nChannelHeight != expectedChannel) {
            m_logger->info("[Solo Validate] ℹ️  nChannelHeight={} is {} blocks ahead of canonical channel tip={} "
                           "(normal during burst recovery — template is valid)",
                           tmpl->nChannelHeight,
                           tmpl->nChannelHeight - authoritative_channel_height,
                           authoritative_channel_height);
        }
    }

    // NOTE: The has_same_height_push_tip_replacement() check was intentionally removed
    // from this gate.  Keeping it here created an infinite soft-refresh / template-swap
    // pending loop: a push set push_hash_prev_block = H_new, but the BLOCK_DATA response
    // to the subsequent GET_BLOCK returned a template with hashPrevBlock = H_old (the node's
    // canonical template, which may legitimately differ from the push hint when the push was
    // premature or from a competing chain tip).  The veto prevented adoption, ClearPushTipAnchor()
    // was never reached, and each new GET_BLOCK response was rejected by the same check,
    // looping indefinitely while Worker_manager reported valid_template=no.
    //
    // Architecture rule: BLOCK_DATA is canonical; push is a hint to request a refresh.
    // Once the node responds to GET_BLOCK with BLOCK_DATA, that template is the ground truth.
    // The push handler already discarded the prior template and triggered the soft refresh;
    // there is no need to veto the replacement here.
    //
    // A push that genuinely represents a same-height chain reorg will be caught by the NEXT
    // push notification after the new template is adopted (the push handler checks
    // has_same_height_push_tip_replacement against the ACTIVE template at that point).
    if (snap.has_same_height_push_tip_replacement(tmpl->block.hashPrevBlock, tmpl->nChannelHeight)) {
        m_logger->info("[Solo Validate] ℹ️  Push tip-anchor differs from incoming BLOCK_DATA "
                       "(push_channel_height={} template_channel_target={}) — "
                       "accepting BLOCK_DATA as canonical (push was a hint, not authoritative)",
                       snap.push_channel_height, tmpl->nChannelHeight);
        // Accept the template — do not discard, do not re-trigger soft refresh.
        // The push anchor will be cleared by ClearPushTipAnchor() in finalize_and_feed_current_template()
        // after successful adoption, preventing this informational log from repeating.
    }

    // *** 1B: SOFTENED hashPrevBlock Mismatch Policy ***
    // hashPrevBlock staleness check (primary anchor, StakeMinter pattern).
    // Only active when HeightTracker has a known hashPrevBlock (non-zero).
    //
    // POLICY CHANGE: Accept ALL templates from the node immediately.
    // The NODE already validates hashPrevBlock; the miner should trust node data
    // during reorgs. We shouldn't make hashPrevBlock a make-or-break to accept
    // new Block Data — the node is authoritative.
    //
    // The HashCheckpointGuard provides advisory diagnostics:
    //   - Shallow reorg: new hashPrevBlock matches a recent checkpoint (depth 1-9)
    //   - Deep reorg: new hashPrevBlock is unknown (not in checkpoint history)
    // HashCheckpoints are immutable once recorded and cannot go through the same
    // reorg process as hashPrevBlock, making them reliable reference points.
    //
    // Tiered advisory logging:
    //   Mismatch count 1:   INFO  — node may be processing reorg
    //   Mismatch count 2-3: WARN  — consecutive hashPrevBlock drift; chain tip churning
    //   Mismatch count 4+:  WARN  — sustained chain flux; node authoritative, accepting
    //   ALL cases:          ACCEPT — the NODE is authoritative
    if (snap.hash_prev_block != uint1024_t(0) &&
        tmpl->block.hashPrevBlock != snap.hash_prev_block) {

        ++m_hashprev_mismatch_consecutive;

        // Use HashCheckpointGuard for reorg depth classification
        auto cp_result = m_hash_checkpoint_guard.evaluate(tmpl->block.hashPrevBlock);

        auto mismatch_count = m_hashprev_mismatch_consecutive.load(std::memory_order_relaxed);

        const char* reorg_type = cp_result.is_shallow_reorg ? "shallow" : "deep/unknown";

        if (mismatch_count == 1) {
            m_logger->info("[ValidateTemplate] ℹ️  hashPrevBlock differs from canonical — "
                           "node may be processing reorg (type={}, depth_est={}) "
                           "(canonical={}, template={}) [mismatch #1]",
                           reorg_type, cp_result.reorg_depth,
                           snap.hash_prev_block.SubString(), tmpl->block.hashPrevBlock.SubString());
        } else if (mismatch_count <= MAX_CONSECUTIVE_HASHPREV_MISMATCHES) {
            m_logger->warn("[ValidateTemplate] ⚡ Consecutive hashPrevBlock drift — "
                           "chain tip churning (type={}, depth_est={}) "
                           "(canonical={}, template={}) [consecutive mismatch #{}/{}]",
                           reorg_type, cp_result.reorg_depth,
                           snap.hash_prev_block.SubString(), tmpl->block.hashPrevBlock.SubString(),
                           mismatch_count, MAX_CONSECUTIVE_HASHPREV_MISMATCHES);
        } else {
            m_logger->warn("[ValidateTemplate] ⚠️  Sustained chain flux — "
                           "{} consecutive hashPrevBlock mismatches (type={}, depth_est={}) "
                           "(canonical={}, template={}) — node authoritative, accepting",
                           mismatch_count, reorg_type, cp_result.reorg_depth,
                           snap.hash_prev_block.SubString(), tmpl->block.hashPrevBlock.SubString());
        }
        // *** ALWAYS ACCEPT — the NODE is authoritative ***
        // The node already validates hashPrevBlock against hashBestChain.
        // During reorgs the node may temporarily serve templates with a different
        // hashPrevBlock; rejecting these causes the doom-loop. Trust the node.
    } else {
        // Hashes match (or canonical is zero): reset the consecutive counter.
        // Note: finalize_and_feed_current_template() also resets the counter after a
        // successful validation to cover the mismatch acceptance path (where the if
        // condition above was true but we fell through without discarding).  The reset
        // here handles all other validate callers and the normal (no-mismatch) path.
        m_hashprev_mismatch_consecutive.store(0, std::memory_order_relaxed);
        m_hash_checkpoint_guard.reset_consecutive();
    }

    // Advisory: log if push_hash_prev_block differs (informational only, not a discard trigger)
    if (snap.push_hash_prev_block != uint1024_t(0) &&
        tmpl->block.hashPrevBlock != snap.push_hash_prev_block) {
        m_logger->info("[ValidateTemplate] ℹ️  Push tip-anchor differs from live template "
                       "(push={}, template={}) — monitoring; canonical check is authoritative",
                       snap.push_hash_prev_block.SubString(), tmpl->block.hashPrevBlock.SubString());
    }

    // Note: Age timeout validation (60s safety net) is handled internally by
    // MiningTemplateInterface. No additional validation needed here.

    m_logger->debug("[Solo Validate] ✓ Template valid (channel_target={}, unified_height={})", 
        tmpl->nChannelHeight, snap.unified_height);
    return true;
}

bool Solo::apply_channel_manager_update(uint32_t unified_height, uint32_t channel_height)
{
    auto* pManager = get_channel_manager();
    if (!pManager) return false;

    pManager->UpdateFromGetRound(unified_height, channel_height);

    if (pManager->IsForkDetected())
    {
        handle_fork_detected(pManager, unified_height);
        return true;
    }

    if (pManager->IsPhantomStakeRegression())
    {
        auto prevHeights = pManager->GetPreviousHeights();
        m_logger->info("[Solo GET_ROUND] ⚡ PHANTOM STAKE REGRESSION — unified {}→{}",
                       prevHeights.first, unified_height);
        m_logger->info("[Solo GET_ROUND]   (cross-channel Stake tip oscillation — template preserved)");
        pManager->ClearForkFlag();
    }

    return false;
}

void Solo::handle_fork_detected(mining::ClientChannelManager* pManager, uint32_t current_height)
{
    if (!pManager) return;
    
    auto prevHeights = pManager->GetPreviousHeights();
    uint32_t nPrevHeight = prevHeights.first;
    uint32_t nRollback = (nPrevHeight > current_height) ? (nPrevHeight - current_height) : 0;
    
    if (nRollback <= 1) {
        // 1-block regression: likely Phantom Stake (GET_ROUND normalization artifact)
        // Log at INFO with ⚡ (tip change semantic) not ⚠ (genuine fork)
        m_logger->info("[Solo Fork] ⚡ PHANTOM STAKE — unified {}→{} "
                       "(1-block GET_ROUND normalization, NOT a real fork — template refresh)",
                       nPrevHeight, current_height);
    } else {
        // 2+ block regression: genuine blockchain rollback
        m_logger->warn("[Solo Fork] ⚠ FORK DETECTED on {} channel!", pManager->GetChannelName());
        m_logger->warn("[Solo Fork]    Previous unified height: {}", nPrevHeight);
        m_logger->warn("[Solo Fork]    Current unified height:  {}", current_height);
        m_logger->warn("[Solo Fork]    Blocks rolled back:      {}", nRollback);
    }
    
    // Template discard applies to BOTH cases — hashPrevBlock is stale either way
    if (m_template_interface && m_template_interface->has_valid_template()) {
        m_template_interface->discard_template(
            nRollback <= 1
                ? "Phantom Stake tip oscillation — template refresh"
                : "Fork detected - blockchain rollback");
        m_logger->info("[Solo Fork] ✗ Template invalidated due to {}",
                       nRollback <= 1 ? "Phantom Stake tip refresh" : "fork");
        // Reset the dedup guard so the immediate replacement GET_BLOCK from the caller
        // (on_get_round_response → sync_template_state → needs_template path) is never
        // suppressed by the 100ms rapid-burst guard.  Without this reset, a PUSH or
        // prior GET_ROUND poll that fired a GET_BLOCK within the last 100ms will block
        // the replacement request, leaving the miner at NO VALID TEMPLATE for up to 30s
        // until HEALTH_NO_TEMPLATE fires with bypass_all.  Every other discard site that
        // is immediately followed by a GET_BLOCK (BLOCK_REJECTED, Stake-advance path,
        // finalize_and_feed_current_template validation gate) already calls this reset.
        reset_get_block_dedup_state();
    }
    
    pManager->ClearForkFlag();
}

void Solo::update_height_state(uint32_t unified_height, uint32_t channel_height,
                                uint32_t difficulty_nbits, HeightTracker::UpdateSource source)
{
    // Update HeightTracker (single source of truth for staleness decisions)
    if (source == HeightTracker::UpdateSource::PUSH) {
        m_height_tracker.OnPushNotification(unified_height, channel_height, difficulty_nbits);
    } else if (source == HeightTracker::UpdateSource::GET_ROUND) {
        // GET_ROUND responses are handled via direct OnGetRound() calls in
        // on_get_round_response() which pass the full 4-height picture.
        // This fallback path (no per-channel data available) passes zeros for
        // the secondary heights so at least the unified height is recorded.
        m_height_tracker.OnGetRound(unified_height, channel_height, 0, 0);
    } else if (source == HeightTracker::UpdateSource::TEMPLATE) {
        // Template metadata may arrive after pushes/keepalives have already
        // advanced the tracker beyond the metadata heights.  Use the monotonic
        // OnTemplateMetadata() so that stale BLOCK_DATA metadata cannot regress
        // unified_height / channel_height, which would hide true staleness from
        // is_template_stale() and is_tip_moved().
        // OnTemplateReceived() is called separately by the handler (after read_template()
        // succeeds) to set channel_target — see the STATELESS_GET_BLOCK handler in
        // process_messages() and the legacy BLOCK_DATA handler.
        m_height_tracker.OnTemplateMetadata(unified_height, channel_height, difficulty_nbits);
    } else {
        m_logger->warn("[Solo] update_height_state: unexpected source {}, defaulting to GET_ROUND",
                       static_cast<int>(source));
        m_height_tracker.OnGetRound(unified_height, channel_height, 0, 0);
    }
}

void Solo::handle_reward_result(const Packet& packet)
{
    m_logger->info("[Solo Reward] Received MINER_REWARD_RESULT");
    refresh_cached_session_state("Solo RewardResult");

    PacketIngressPreflightOptions preflight;
    preflight.owner = &m_last_reward_request_owner;
    preflight.require_crypto_ready = m_enable_chacha20;
    if (!run_packet_ingress_preflight("Solo RewardResult", preflight)) {
        m_reward_bound = false;
        return;
    }
    
    // Validate packet data
    if (!packet.m_data || packet.m_length < 1) {
        m_logger->error("[Solo Reward] Invalid MINER_REWARD_RESULT packet - no data");
        m_reward_bound = false;
        if (m_session_context) {
            m_session_context->commit_reward_rejected(m_reward_address,
                                                      m_reward_address.empty() ? "" : "live bind",
                                                      "empty reward result payload");
        }
        return;
    }
    
    std::vector<uint8_t> result_data;
    
    // Decrypt if ChaCha20 is enabled
    if (m_enable_chacha20 && m_chacha20_wrapper && packet.m_length > 13)
    {
        const auto session = m_session_context ? m_session_context->get_runtime_snapshot()
                                               : SessionManager::SessionInfo{};
        const auto& reward_session_key = session.chacha20_session_key;
        if (session.chacha20_ready && !reward_session_key.empty())
        {
            try {
                // Extract nonce (first 12 bytes)
                std::vector<uint8_t> nonce(packet.m_data->begin(), packet.m_data->begin() + 12);
                std::vector<uint8_t> ciphertext(packet.m_data->begin() + 12, packet.m_data->end());
                
                // Decrypt response using matching AAD
                auto decrypt_result = m_chacha20_wrapper->decrypt(ciphertext, reward_session_key, nonce, AAD_REWARD_RESULT);
                
                if (decrypt_result.success) {
                    result_data = decrypt_result.data;
                    m_logger->debug("[Solo Reward] Decrypted with AAD: REWARD_RESULT ({} bytes)", AAD_REWARD_RESULT.size());
                } else {
                    m_logger->error("[Solo Reward] Failed to decrypt result: {}", decrypt_result.error_message);
                    m_reward_bound = false;
                    return;
                }
            }
            catch (const std::exception& e) {
                m_logger->error("[Solo Reward] Decryption failed: {}", e.what());
                m_reward_bound = false;
                return;
            }
        }
        else
        {
            if (!reward_session_key.empty() && !session.chacha20_ready) {
                m_logger->warn("[Solo Reward] Reward result arrived with non-ready ChaCha20 key; treating payload as unencrypted");
            }
            // No genesis, try unencrypted
            result_data.assign(packet.m_data->begin(), packet.m_data->end());
        }
    }
    else
    {
        // Unencrypted result
        result_data.assign(packet.m_data->begin(), packet.m_data->end());
    }
    
    if (result_data.empty()) {
        m_logger->error("[Solo Reward] Empty result data after processing");
        m_reward_bound = false;
        return;
    }
    
    // Parse status byte
    uint8_t status = result_data[0];
    
    if (status == 0x01)  // Success
    {
        m_logger->info("╔═════════════════════════════════════════════════════════╗");
        m_logger->info("║       REWARD ADDRESS BINDING SUCCESSFUL                 ║");
        m_logger->info("╠═════════════════════════════════════════════════════════╣");
        m_logger->info("║ Address: {}                                              ║", 
            m_reward_address.substr(0, std::min(m_reward_address.length(), ADDRESS_DISPLAY_TRUNCATE)));
        m_logger->info("╚═════════════════════════════════════════════════════════╝");
        
        m_reward_bound = true;
        std::vector<uint8_t> reward_hash;
        // reward_address is a 64-char hex genesis hash — decode directly
        reward_hash = genesis_utils::hex_decode_genesis_hash(m_reward_address);
        if (reward_hash.size() != 32) {
            m_logger->warn("[Solo Reward] Could not decode reward_address to 32 bytes for session commit");
            reward_hash.clear();
        }
        if (m_session_context) {
            m_session_context->commit_reward_bound(m_reward_address, reward_hash, "live bind");
            m_session_context->set_channel_state(m_channel, false, false);
            m_session_context->mark_activity();
        }
        validate_authoritative_session("Solo RewardResult", false);
        log_session_container_summary("Solo RewardResult");
        
        // CONTINUE MINING FLOW: Now send SET_CHANNEL and GET_BLOCK
        if (m_connection)
        {
            m_logger->info("[Solo Phase 2] Reward bound - continuing with mining flow");
            m_logger->info("[Solo Phase 2] Sending SET_CHANNEL");
            send_set_channel(m_connection);
            
            // Note: GET_BLOCK will be sent after receiving CHANNEL_ACK
            // This is handled in the existing CHANNEL_ACK handler
        }
        else
        {
            m_logger->error("[Solo Reward] Cannot continue - no connection available");
        }
    }
    else
    {
        // Parse optional error message
        std::string error_message = "Unknown error";
        if (result_data.size() >= 2)
        {
            uint8_t msg_len = result_data[1];
            if (result_data.size() >= 2 + msg_len)
            {
                error_message = std::string(result_data.begin() + 2, result_data.begin() + 2 + msg_len);
            }
        }
        
        m_logger->error("╔═════════════════════════════════════════════════════════╗");
        m_logger->error("║       REWARD ADDRESS BINDING FAILED                     ║");
        m_logger->error("╠═════════════════════════════════════════════════════════╣");
        m_logger->error("║ Error: {}                                                ║", error_message);
        m_logger->error("╚═════════════════════════════════════════════════════════╝");
        
        m_reward_bound = false;
        if (m_session_context) {
            m_session_context->commit_reward_rejected(m_reward_address, "live bind", error_message);
            m_session_context->set_channel_state(m_channel, false, false);
        }
        
        // Reward binding failed - cannot proceed with mining
        m_logger->error("[Solo Reward] Cannot mine without reward address bound");
        
        // Close connection to prevent further issues
        if (m_connection)
        {
            m_logger->error("[Solo Reward] Closing connection due to reward binding failure");
            m_connection->close();
            m_connection = nullptr;  // Reset to prevent use-after-close
        }
    }
}

void Solo::handle_initial_template_response(const char* opcode_name)
{
    // Log template reception for diagnostic purposes
    // No timing or state tracking - lane is authoritative, determined at connection time
    m_logger->info("[Solo Protocol] ✅ Template received via {}", opcode_name);
    m_logger->debug("[Solo Protocol]   Protocol lane: {}", get_lane_name(m_protocol_lane));
}

// ═══════════════════════════════════════════════════════════════════════
// INTELLIGENT GET_ROUND POLLING IMPLEMENTATION
// ═══════════════════════════════════════════════════════════════════════

bool Solo::should_poll_get_round()
{
    // GET_ROUND polling is DISABLED by default.
    // Push notifications are the primary mechanism for height detection:
    // - Legacy: PRIME_BLOCK_AVAILABLE (0xD9) / HASH_BLOCK_AVAILABLE (0xDA)
    // - Stateless: STATELESS_PRIME_BLOCK_AVAILABLE (0xD0D9) / STATELESS_HASH_BLOCK_AVAILABLE (0xD0DA)
    // Event-driven GET_BLOCK handles template requests on demand.
    // Template health monitor (300s timeout) provides emergency safety net.
    if (!POLLING_ENABLED) {
        return false;
    }

    auto now = std::chrono::steady_clock::now();
    auto elapsed_ms = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            now - m_last_get_round_time).count());

    if (elapsed_ms < static_cast<uint64_t>(m_current_poll_interval_ms)) {
        return false;
    }

    m_last_get_round_time = now;
    m_logger->debug("[Solo Poll] GET_ROUND sanity-check poll (interval {}ms)", m_current_poll_interval_ms);
    return true;
}

std::chrono::milliseconds Solo::push_in_transit_guard_for_channel() const
{
    if (m_channel == mining::CHANNEL_PRIME) {
        return std::chrono::milliseconds(PUSH_BLOCK_DATA_IN_TRANSIT_GUARD_PRIME_MS);
    }
    return std::chrono::milliseconds(PUSH_BLOCK_DATA_IN_TRANSIT_GUARD_HASH_MS);
}

void Solo::arm_get_round_fallback(int64_t push_silent_seconds)
{
    if (m_get_round_push_silent_fallback_active) {
        return;
    }
    m_get_round_push_silent_fallback_active = true;
    m_logger->warn("[Solo GET_ROUND] PUSH silent for {}s (threshold {}s) — arming GET_ROUND fallback GET_BLOCK mode",
        push_silent_seconds,
        PUSH_ABSENT_FOR_GET_ROUND_FALLBACK_SECONDS);
}

void Solo::disarm_get_round_fallback(const char* reason, int64_t push_age_seconds)
{
    if (!m_get_round_push_silent_fallback_active) {
        return;
    }
    m_get_round_push_silent_fallback_active = false;
    const char* reason_text = reason ? reason : (push_age_seconds >= 0 ? "PUSH active again" : "PUSH re-established");
    if (push_age_seconds >= 0) {
        m_logger->info("[Solo GET_ROUND] {} ({}s ago) — disabling GET_ROUND fallback GET_BLOCK mode",
            reason_text,
            push_age_seconds);
    } else {
        m_logger->info("[Solo GET_ROUND] {} — disabling GET_ROUND fallback GET_BLOCK mode",
            reason_text);
    }
}

void Solo::on_new_round_received(uint32_t new_unified_height)
{
    // NEW_ROUND = block was found, fixed polling interval (backoff disabled)
    m_current_poll_interval_ms = POLL_INTERVAL_MIN_MS;
    m_logger->info("[Solo Poll] ⚡ NEW_ROUND: chain tip changed — poll interval reset to {}ms",
        m_current_poll_interval_ms);
    
    // GET_ROUND response received — reset unanswered counter and timestamps
    m_unanswered_get_round_count.store(0, std::memory_order_release);
    m_earliest_unanswered_get_round_at = {};
    m_last_get_round_transmitted_at = {};

    // Check unified height delta
    check_unified_height_delta(new_unified_height);
}

void Solo::on_old_round_received()
{
    // Backoff disabled: GET_ROUND is the primary mechanism for detecting Stake
    // block tip advances, which do NOT trigger PUSH notifications. A fixed
    // interval ensures the miner polls for template freshness consistently.
    // m_current_poll_interval_ms stays at POLL_INTERVAL_MIN_MS always.
    m_current_poll_interval_ms = POLL_INTERVAL_MIN_MS;

    // GET_ROUND response received — reset unanswered counter and timestamps
    m_unanswered_get_round_count.store(0, std::memory_order_release);
    m_earliest_unanswered_get_round_at = {};
    m_last_get_round_transmitted_at = {};
}

void Solo::on_template_received(uint32_t template_height)
{
    // Track template reception for height delta detection
    m_needs_initial_round_check = true;
    m_template_unified_height = template_height;
    m_last_get_round_time = std::chrono::steady_clock::now();  // Reset timer
    m_logger->debug("[Solo Poll] Template received (height {})", template_height);
}

void Solo::check_unified_height_delta(uint32_t current_unified_height)
{
    if (m_template_unified_height == 0) {
        return;  // No template yet
    }
    
    // When the unified tip moves, hashPrevBlock in the current template becomes
    // stale even if the channel height hasn't changed.  The NODE auto-sends
    // BLOCK_DATA after PUSH, and GET_ROUND is the backup for tip changes.
    // Health monitor logs the tip movement but does NOT send GET_BLOCK.
    if (current_unified_height > m_template_unified_height) {
        uint32_t delta = current_unified_height - m_template_unified_height;
        
        m_logger->info("[Solo Poll] ↑ Unified tip moved {} blocks ({} → {}) [reason: tip_moved] — "
                       "node will auto-send fresh template via PUSH; GET_ROUND backup active",
            delta, m_template_unified_height, current_unified_height);
        // Update to avoid repeated log spam
        m_template_unified_height = current_unified_height;
    }
}

void Solo::initialize_protocol_lane(std::shared_ptr<network::Connection> connection)
{
    if (!connection) {
        m_logger->error("[Lane Init] No connection available - cannot determine protocol lane");
        return;
    }
    
    auto const& remote_ep = connection->remote_endpoint();
    uint16_t remote_port = remote_ep.port();
    
    // Determine lane from port (authoritative)
    m_protocol_lane = determine_lane_from_port(remote_port);
    
    const char* lane_name = get_lane_name(m_protocol_lane);
    const char* framing = (m_protocol_lane == ProtocolLane::STATELESS)
                            ? "16-bit header (0xD000-0xD0FF)" : "8-bit header (legacy)";
    const char* behavior = (m_protocol_lane == ProtocolLane::STATELESS)
                            ? "Push (MINER_READY / GET_BLOCK)"
                            : "Push (MINER_READY / GET_BLOCK) with 8-bit opcodes";

    // Log lane selection
    m_logger->info("═══════════════════════════════════════════════════════════");
    m_logger->info("PROTOCOL LANE INITIALIZATION (Solo Protocol Layer)");
    m_logger->info("═══════════════════════════════════════════════════════════");
    m_logger->info("Remote:          {}", remote_ep.to_string());
    m_logger->info("Remote Port:     {}", remote_port);
    m_logger->info("Selected Lane:   {}", lane_name);
    m_logger->info("Lane Source:     Port-determined (authoritative)");
    m_logger->info("Framing:         {}", framing);
    m_logger->info("Behavior:        {}", behavior);
    m_logger->info("Authentication:  Falcon + ChaCha20 (required)");
    m_logger->info("═══════════════════════════════════════════════════════════");
    m_logger->info("STRICT LANE SEPARATION: NO FALLBACK BETWEEN LANES");
    m_logger->info("═══════════════════════════════════════════════════════════");
}

}
}
