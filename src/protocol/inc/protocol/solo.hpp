#ifndef NEXUSMINER_PROTOCOL_SOLO_HPP
#define NEXUSMINER_PROTOCOL_SOLO_HPP

#include "protocol/protocol.hpp"
#include "protocol/falcon_wrapper.hpp"
#include "protocol/chacha20_wrapper.hpp"
#include "protocol/session_manager.hpp"
#include "protocol/node_session_context.hpp"
#include "protocol/mining_template_interface.hpp"
#include "protocol/push_notification_handler.hpp"
#include "protocol/height_tracker.hpp"
#include "protocol/session_ingress_gate.hpp"
#include "protocol/session_recovery_policy.hpp"
#include "protocol/submit_context.hpp"
#include "protocol/submit_result_gate.hpp"
#include "protocol/epoch_coordinator.hpp"
#include "protocol/get_block_reason.hpp"
#include "protocol/get_block_dedup_guard.hpp"
#include "protocol/hash_checkpoint_guard.hpp"
#include "protocol/merkle_root_feed_guard.hpp"
#include "protocol/packet_router.hpp"
#include "mining/client_channel_manager.h"
#include "protocol_lane.hpp"
#include "LLP/colin_ping_handler.h"
#include "spdlog/spdlog.h"
#include <asio/io_context.hpp>
#include <asio/steady_timer.hpp>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>

namespace nexusminer {
namespace network { class Connection; }
namespace stats { class Collector; }
namespace mining { class PrimeClientManager; class HashClientManager; }
namespace protocol
{

enum class AuthState {
    NOT_AUTHENTICATED,
    WAITING_FOR_CHALLENGE,
    WAITING_FOR_RESULT,
    AUTHENTICATED
};

class Solo : public Protocol, public std::enable_shared_from_this<Solo> {
public:
    enum class GetBlockRequestStatus : uint8_t {
        NONE = 0,
        SENT,
        DUPLICATE_WINDOW,
        NODE_RETRY_DELAY,
        UNAUTHENTICATED,
        REWARD_NOT_BOUND,
        SESSION_INVALID,
        BUILD_EMPTY
    };

    Solo(std::uint8_t channel, std::shared_ptr<stats::Collector> stats_collector,
         std::shared_ptr<NodeSessionContext> session_context,
         std::shared_ptr<asio::io_context> io_context = nullptr);

    void reset() override;
    network::Shared_payload login(Login_handler handler) override;
    /// Request a fresh mining template via GET_BLOCK.
    /// Authentication-guarded; returns null if not authenticated or reward not bound.
    /// Node retry hints are enforced across all GET_BLOCK request paths.
    network::Shared_payload get_work() override;
    network::Shared_payload get_work(GetBlockReason reason);
    GetBlockRequestStatus get_last_get_block_request_status() const { return m_last_get_block_request_status.load(); }

    /// Returns the number of consecutive hashPrevBlock mismatches detected by
    /// validate_current_template() since the last successful template adoption.
    /// Worker_manager uses this to apply exponential GET_BLOCK backoff during
    /// node-attack / chain-flux scenarios that would otherwise cause a rapid-fire
    /// discard-and-retry doom loop.
    uint32_t get_hashprev_mismatch_consecutive() const { return m_hashprev_mismatch_consecutive.load(std::memory_order_relaxed); }

    /// Returns how many times the same-height feed guard has suppressed a re-feed
    /// because the node re-served an unchanged (same height, same hashPrevBlock)
    /// template. Diagnostic counter only — see m_same_tip_reconfirmation_count.
    uint64_t get_same_tip_reconfirmation_count() const { return m_same_tip_reconfirmation_count; }

    /// Returns a const reference to the HashCheckpointGuard for diagnostic queries.
    /// The guard maintains a rolling window of recent canonical hashPrevBlock values
    /// and provides reorg depth estimation for Colin diagnostics.
    const HashCheckpointGuard& get_hash_checkpoint_guard() const { return m_hash_checkpoint_guard; }

    /// Reset the GET_BLOCK deduplication timestamp so the next get_work() call will
    /// not be suppressed.  Must be called whenever the canonical tip-anchor changes
    /// (same-height chain reorg) or a new degraded-recovery epoch begins, because the
    /// outstanding dedup state refers to a request for the *old* canonical tip and is
    /// therefore no longer valid as a duplicate guard.
    void reset_get_block_dedup_state();
    network::Shared_payload submit_block(std::vector<std::uint8_t> const& block_data, std::uint64_t nonce) override;
    void set_block_handler(Set_block_handler handler) override { m_set_block_handler = std::move(handler); }

    void process_messages(Packet packet, std::shared_ptr<network::Connection> connection) override;
    
    // GET_ROUND protocol support (12-byte response from LLL-TAO PR #151)
    struct RoundStatus {
        bool is_new_round;          // true if NEW_ROUND (204), false if OLD_ROUND (205)
        uint32_t height;            // Unified blockchain height (reference only)
        uint32_t difficulty;        // Mining difficulty in compact nBits format (big-endian)
        
        // Channel-specific heights (miner's channel only)
        uint32_t prime_height;      // Prime channel height (if mining Prime)
        uint32_t hash_height;       // Hash channel height (if mining Hash)
        uint32_t stake_height;      // Stake channel height (unused by stateless miners)
        bool has_channel_heights;   // True if 12-byte response parsed successfully
        
        /**
         * @brief Get channel-specific height
         * @param channel Channel number (1=Prime, 2=Hash, 3=Stake)
         * @return Channel height, or 0 if not set
         */
        uint32_t get_channel_height(uint32_t channel) const {
            switch(channel) {
                case 1:  return prime_height;
                case 2:  return hash_height;
                case 3:  return stake_height;
                default: return 0;
            }
        }
    };
    /// Send GET_ROUND on all lanes (legacy: 0x85; stateless: 0xD085).
    /// This is a pure informational/sanity probe — the node responds with
    /// NEW_ROUND or OLD_ROUND containing height and difficulty info.
    /// It does NOT request a block template.  Use send_recovery_work_request()
    /// when the intent is to force a fresh template retrieval.
    network::Shared_payload send_get_round();

    // GET_ROUND polling configuration (public so callers can log the intervals)
    static constexpr bool POLLING_ENABLED = true;              // Enabled: GET_ROUND sanity probe for both lanes
    static constexpr uint32_t POLL_INTERVAL_MIN_MS = 30000;    // 30 seconds — Stake block detection cadence
    static constexpr uint32_t POLL_INTERVAL_MAX_MS = 30000;    // Same: backoff disabled, fixed 30s interval
    // Minimum push-silence duration before GET_ROUND fallback may trigger GET_BLOCK.
    // Policy: while PUSH is active, GET_ROUND is informational only. Once PUSH has
    // been silent for this threshold, each GET_ROUND poll (POLL_INTERVAL_MIN_MS minimum
    // cadence) may request GET_BLOCK.  Set to match TEMPLATE_AGE_WARNING_SECONDS so
    // the GET_ROUND fallback arms BEFORE the 600s emergency timeout fires.
    static constexpr int64_t PUSH_ABSENT_FOR_GET_ROUND_FALLBACK_SECONDS = 480;
    /// Guard window (ms) during which a received BLOCK_AVAILABLE push implies BLOCK_DATA
    /// is already in transit from the node. Prime-channel traces show PUSH→BLOCK_DATA
    /// delivery jitter up to ~4.7s in production, so Prime uses 6000ms while Hash keeps
    /// the previous 3000ms guard.
    static constexpr int64_t PUSH_BLOCK_DATA_IN_TRANSIT_GUARD_PRIME_MS = 6000;
    static constexpr int64_t PUSH_BLOCK_DATA_IN_TRANSIT_GUARD_HASH_MS  = 3000;
    /// Send GET_BLOCK on all lanes (legacy: 0x81; stateless: 0xD081) to request
    /// a fresh mining template.  Authentication-guarded; delegates to get_work().
    /// Returns null/empty if not yet authenticated — callers must guard for this.
    /// Use this method — not send_get_round() — for template recovery actions.
    network::Shared_payload send_recovery_work_request();
    RoundStatus get_last_round_status() const { return m_last_round_status; }
    uint32_t get_current_poll_interval_ms() const { return m_current_poll_interval_ms; }
    
    // Intelligent polling: Check if GET_ROUND should be sent now
    // Note: This modifies internal timing state, so cannot be truly const
    bool should_send_get_round() { return should_poll_get_round(); }
    
    // Falcon miner authentication
    void set_miner_keys(std::vector<uint8_t> const& pubkey, std::vector<uint8_t> const& privkey);
    bool is_authenticated() const { return m_session_context ? m_session_context->is_authenticated() : m_authenticated; }
    bool can_request_get_block() const {
        return m_session_context ? m_session_context->can_request_get_block() : m_authenticated;
    }
    void set_address(std::string const& address) { m_address = address; }
    void set_protocol_lane(ProtocolLane lane);
    
    // Tritium GenesisHash for reward binding
    void set_tritium_genesis(std::vector<uint8_t> const& genesis);
    bool has_tritium_genesis() const;
    
    // Session management configuration
    void set_keepalive_interval(std::uint16_t hours);
    void enable_chacha20_wrapping(bool enable) { m_enable_chacha20 = enable; }
    bool is_chacha20_enabled() const { return m_enable_chacha20; }
    
    // Disposable Falcon is ALWAYS ON (core protocol) - method kept for backward compatibility only
    void enable_disposable_falcon(bool enable) { m_disposable_falcon_enabled = enable; }
    bool is_disposable_falcon_enabled() const { return m_disposable_falcon_enabled; }
    
    // Legacy method names (deprecated - kept for backward compatibility)
    void enable_block_signing(bool enable) { m_disposable_falcon_enabled = enable; }
    bool is_block_signing_enabled() const { return m_disposable_falcon_enabled; }
    
    // Session management (LLL-TAO PR #22 / PR #217)
    // Sends SESSION_KEEPALIVE via SessionManager (8-byte v2 payload: session_id + prevhash_lo32).
    // Node replies with the 32-byte unified KeepAliveV2AckFrame carrying all channel heights.
    network::Shared_payload send_session_keepalive();
    SessionId get_session_id() const;
    bool is_session_active() const;

    // Build a SESSION_STATUS packet for transmission on this lane.
    // Uses the SessionManager and current template/worker state internally.
    // Returns null if session is not active or lane is UNKNOWN.
    network::Shared_payload build_session_status_packet(
        bool degraded, bool workers_running, bool secondary_up) const;
    
    // SessionManager keepalive needs connection context
    void set_connection(std::shared_ptr<network::Connection> connection);
    
    // Mining Template Interface access (unified READ/FEED system)
    MiningTemplateInterface* get_template_interface() { return m_template_interface.get(); }
    const MiningTemplateInterface* get_template_interface() const { return m_template_interface.get(); }
    
    // Stateless mining reward address binding (MINER_SET_REWARD protocol)
    void set_reward_address(std::string const& address);
    bool has_reward_address() const { return !m_reward_address.empty(); }
    bool is_reward_bound() const {
        return m_session_context ? m_session_context->get_session_binding().reward_bound : m_reward_bound;
    }
    network::Shared_payload send_set_reward();
    
    // Push notification subscription (LLL-TAO PR #156)
    network::Shared_payload send_miner_ready();
    /// Re-subscribe to push notifications by sending MINER_READY on the stored connection.
    /// Safe to call after prolonged push silence (e.g. after degraded recovery) to restore
    /// push flow when the node's subscription state may have been lost during TCP disruption.
    /// No-op when not yet authenticated or no connection is available.
    void resubscribe_push_notifications();
    
    // HeightTracker snapshot (single source of truth for height/staleness decisions)
    HeightTracker::Snapshot get_height_tracker_snapshot() const { return m_height_tracker.GetSnapshot(); }

    // Canonical-only snapshot (authoritative block-data state)
    HeightTracker::CanonicalChainState get_canonical_snapshot() const { return m_height_tracker.GetCanonicalSnapshot(); }

    // Diagnostic-only snapshot (push/keepalive/GET_ROUND telemetry)
    HeightTracker::DiagnosticObserverState get_diagnostic_snapshot() const { return m_height_tracker.GetDiagnosticSnapshot(); }

    // HeightTracker reference (for direct read access by ColinAgent)
    const HeightTracker& get_height_tracker() const { return m_height_tracker; }

    // Helper to access SessionManager through NodeSessionContext
    SessionManager* get_session_manager() const {
        return m_session_context ? m_session_context->get_session_manager().get() : nullptr;
    }

    void mark_authoritative_recovery_required(const std::string& reason);
    void mark_authoritative_recovery_healthy(const std::string& reason = "");

    // Recovery callback: called by the push handler when a channel-stale recovery GET_BLOCK
    // is triggered (is_template_stale() is true at request_work_fn invocation time),
    // and by deserialize/validation failure paths in on_block_data / on_get_block_template.
    //
    // The handler MUST return true iff it successfully scheduled / transmitted a fresh
    // GET_BLOCK request.  Solo uses the return value as the single chokepoint signal:
    // if the handler succeeded, Solo will NOT also issue its own local
    // request_and_queue_get_block (preventing the "double-tap" storm where one failure
    // event fires two concurrent GET_BLOCKs — see fix for PR #698 regression).
    //
    // A void-returning legacy handler can be wired with a lambda that returns true.
    using Recovery_handler = std::function<bool(protocol::GetBlockReason)>;
    void set_recovery_initiated_handler(Recovery_handler h) { m_recovery_handler = std::move(h); }

    // Session-expired callback: called when a KEEPALIVE ACK session_id mismatch is detected.
    // Worker_manager registers this to trigger recovery on stale session (same pattern as
    // Recovery_handler above).
    using Session_expired_handler = std::function<void()>;
    void set_session_expired_handler(Session_expired_handler h) { m_session_expired_handler = std::move(h); }

    // Recovery-confirmed callback: called when the node re-serves BLOCK_DATA that turns
    // out to be identical to the template already loaded (same unified height AND same
    // hashPrevBlock) while a recovery-triggering GET_BLOCK request is in flight.
    //
    // This is distinct from the normal template-feed path: the same-height feed guard in
    // finalize_and_feed_current_template() intentionally suppresses re-distributing an
    // unchanged template to workers (to avoid redundant worker restarts), which means
    // Worker_manager's feed handler — the only place that normally clears recovery state
    // via clear_recovery_state() — never runs for this response.
    //
    // Without this signal, a recovery cycle that legitimately confirms "the tip I already
    // have is still the canonical one" is indistinguishable from one that got no answer at
    // all, and stays parked in WAITING_TEMPLATE until a genuinely new tip arrives or the
    // controlled-recovery hard-stop escalates it into DEGRADED_MODE. Since the node just
    // proved it is alive and the current template is still valid, Worker_manager should be
    // allowed to clear recovery immediately on this signal (subject to its own session/
    // template validity gates in clear_recovery_state()) instead of waiting.
    using Recovery_confirmed_handler = std::function<void()>;
    void set_recovery_confirmed_handler(Recovery_confirmed_handler h) { m_recovery_confirmed_handler = std::move(h); }

    // Session-authenticated callback: called after MINER_AUTH_RESULT is fully processed and session_id is set.
    // Worker_manager registers this to check session_id=0 and trigger retry if needed.
    // Parameter: session_id (0 if node rejected authentication).
    using Session_authenticated_handler = std::function<void(SessionId session_id)>;
    void set_session_authenticated_handler(Session_authenticated_handler h) { m_session_authenticated_handler = std::move(h); }

    // Work-ready callback: called once the authoritative session container says GET_BLOCK may resume.
    using Work_ready_handler = std::function<void()>;
    void set_work_ready_handler(Work_ready_handler h) { m_work_ready_handler = std::move(h); }

    // Session-start callback: called when SESSION_START is received and keepalive interval
    // has been auto-adjusted from the node-advertised timeout.
    // Parameter: keepalive_hours (the newly derived interval, e.g. session_timeout / 2 / 3600)
    using Session_start_handler = std::function<void(uint16_t keepalive_hours)>;
    void set_session_start_handler(Session_start_handler h) { m_session_start_handler = std::move(h); }

    // Block-result counters (Gap 3)
    uint32_t get_blocks_accepted() const;
    uint32_t get_blocks_rejected() const;

    // Shadow-ban telemetry: consecutive unanswered GET_ROUNDs and preflight rejections
    uint32_t get_unanswered_get_round_count() const { return m_unanswered_get_round_count.load(std::memory_order_acquire); }
    uint32_t get_preflight_reject_count() const { return m_preflight_reject_count; }

    // Returns the time of the earliest still-unanswered GET_ROUND, or the default
    // time_point{} if no unanswered GET_ROUNDs are outstanding.
    std::chrono::steady_clock::time_point get_earliest_unanswered_get_round_at() const {
        return m_earliest_unanswered_get_round_at;
    }

    /// Call ONLY after a GET_ROUND packet has been successfully handed to transmit().
    /// Increments the unanswered counter that feeds shadow-ban detection.
    void note_get_round_transmitted();

    // Block-result callback: invoked on BLOCK_ACCEPTED with (height, hashPrevBlock, channel, nonce).
    // Worker_manager registers this to record accepted blocks in the mined-block cache.
    using Block_accepted_handler = std::function<void(uint32_t height, uint1024_t hash_prev_block,
                                                      uint32_t channel, uint64_t nonce)>;
    void set_block_accepted_handler(Block_accepted_handler h) { m_block_accepted_handler = std::move(h); }

    // Node shutdown callback: invoked when NODE_SHUTDOWN (0xD0FF) is received.
    // Worker_manager registers this to stop workers and set reconnect backoff.
    // Parameter: shutdown reason (GRACEFUL=0x01, MAINTENANCE=0x02).
    using Node_shutdown_handler = std::function<void(uint8_t reason)>;
    void set_node_shutdown_handler(Node_shutdown_handler h) { m_node_shutdown_handler = std::move(h); }

    // Replacement-pending callback: invoked after a PUSH notification leaves the
    // current template marked replacement-pending (mark_replacement_pending()),
    // with the deadline by which the promised BLOCK_DATA must arrive. Lets
    // Worker_manager schedule a dedicated one-shot timer that fires exactly at
    // the deadline, instead of relying solely on the slower general-purpose
    // template-health polling cadence to eventually notice an expired window
    // (see take_expired_replacement_pending()).
    using Replacement_pending_handler =
        std::function<void(std::chrono::steady_clock::time_point deadline)>;
    void set_replacement_pending_handler(Replacement_pending_handler h) { m_replacement_pending_handler = std::move(h); }

    // Reconnect backoff (seconds) after receiving NODE_SHUTDOWN from the node.
    static constexpr uint32_t NODE_SHUTDOWN_BACKOFF_S = 60;

    // ── Recovery-debounce test/diagnostic interface ──────────────────────────────
    // These lightweight read-only accessors let unit tests (and future diagnostics)
    // observe the state of the 5s deferred-recovery timer without needing a network
    // connection or real workers.

    /// True when a deferred recovery GET_BLOCK has been scheduled and is still pending.
    bool is_recovery_pending() const noexcept {
        return m_recovery_deferred_at != std::chrono::steady_clock::time_point::min();
    }

    /// How many times the 5s debounce timer fired and attempted to send a recovery
    /// GET_BLOCK (regardless of whether the connection was available).  Used by unit
    /// tests to assert that the timer did / did not fire.
    int get_recovery_fired_count() const noexcept { return m_recovery_fired_count; }

    // Colin AI Diagnostic PING/PONG handler
    const ::LLP::ReceivedPingFrame& last_received_ping() const
    {
        return m_colin_ping_handler.last_received_ping();
    }

    // Non-owning reference to the Colin ping handler for diagnostics (ping_count, last_rtt_us).
    // Lifetime: as long as this Solo instance is alive.
    const ::LLP::ColinPingHandler& get_ping_handler() const { return m_colin_ping_handler; }

    // SESSION_STATUS_ACK tracking — updated whenever a SESSION_STATUS_ACK is received
    /** Returns the most recently received SESSION_STATUS_ACK frame (zero if never received) **/
    const ::LLP::SessionStatusAckFrame& last_session_status_ack() const { return m_last_session_status_ack; }
    /** Returns time of last SESSION_STATUS_ACK (default time_point if never received) **/
    std::chrono::steady_clock::time_point last_session_status_ack_time() const { return m_last_session_status_ack_time; }

    // Returns true if authentication is currently in-flight (waiting for challenge or result).
    // Used by Worker_manager to avoid sending duplicate login() calls.
    bool is_auth_in_progress() const {
        return m_auth_state == AuthState::WAITING_FOR_CHALLENGE ||
               m_auth_state == AuthState::WAITING_FOR_RESULT;
    }

    // Returns how many seconds auth has been in-flight (0 if not in-flight).
    // Used by Worker_manager to detect stuck in-flight auth from a dead TCP session.
    double auth_in_flight_seconds() const {
        if (!is_auth_in_progress() || m_auth_in_flight_since == std::chrono::steady_clock::time_point{})
            return 0.0;
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - m_auth_in_flight_since).count();
    }

    // Reset only the authentication state machine (auth_state, auth_in_flight_since, authenticated flag).
    // Call this before triggering in-band re-authentication to clear any stale in-flight state.
    // Does NOT clear session key, genesis, or connection.
    void reset_auth_state();

    // Maximum time (seconds) auth is allowed to stay in-flight before being treated as failed.
    static constexpr int AUTH_IN_FLIGHT_TIMEOUT_S = 30;

    /// Wire the shared EpochCoordinator (called by Worker_manager after construction).
    void set_epoch_coordinator(std::shared_ptr<EpochCoordinator> coordinator);

private:
    
    // Derive ChaCha20 session key from genesis hash
    std::vector<uint8_t> derive_chacha20_session_key(const std::vector<uint8_t>& genesis);
    std::vector<uint8_t> derive_or_get_cached_chacha20_session_key(const std::vector<uint8_t>& genesis);
    
    // Load tritium genesis (session manager or persistent storage)
    std::vector<uint8_t> load_tritium_genesis();
    
    // Helper method to send SET_CHANNEL packet
    void send_set_channel(std::shared_ptr<network::Connection> connection);

    // If auth is in-flight and has exceeded AUTH_IN_FLIGHT_TIMEOUT_S, reset to NOT_AUTHENTICATED.
    // Returns true if a timeout reset occurred.
    bool check_auth_in_flight_timeout(const char* context);

    // Session ID mismatch check — shared by KEEPALIVE_V2_ACK and SESSION_STATUS_ACK handlers.
    // Returns true if a mismatch was detected (state set to EXPIRED, handler called);
    // caller must return immediately when true is returned.
    bool handle_session_id_mismatch(SessionId ack_session_id);

    // Session expired handler — called when SESSION_EXPIRED (0xDD / 0xD0DD) packet is received
    // Implements 5-step response: log, clear state, stop workers, prepare for re-auth
    void handle_session_expired(SessionId expired_sid, uint8_t reason, std::shared_ptr<network::Connection> connection);

    // Challenge-response authentication methods
    void handle_miner_auth_challenge(const Packet& packet);
    
    // Handle reward result response from node (MINER_REWARD_RESULT)
    void handle_reward_result(const Packet& packet);
    
    // Unified handler for initial template reception (called by both BLOCK_DATA and STATELESS_GET_BLOCK handlers)
    void handle_initial_template_response(const char* opcode_name);
    
    // Helper method to finalize template with channel height
    // Returns true if template was finalized, false if already finalized or no template
    bool finalize_template_with_channel_height(uint32_t node_channel_height, const std::string& context);
    
    // Helper method to get channel manager for current channel
    mining::ClientChannelManager* get_channel_manager() const;
    mining::ClientChannelManager* get_channel_manager(uint32_t channel) const;

    // NodeSessionContext is the authoritative session source; use it to detect
    // whether a stale local auth flag needs resynchronization.
    bool session_context_is_authenticated() const;
    void propagate_session_to_template_interface(const char* log_scope);
    void resync_auth_from_session_context(const char* log_scope);
    void refresh_cached_session_state(const char* log_scope);
    void update_connection_metadata(const std::shared_ptr<network::Connection>& connection);
    bool validate_authoritative_session(const char* log_scope, bool require_reward_binding) const;
    void log_session_container_summary(const char* log_scope) const;
    struct PacketIngressPreflightOptions {
        const SessionOwnershipStamp* owner{nullptr};
        SessionId packet_session_id{};
        bool allow_without_active_session{false};
        bool validate_lane{false};
        bool require_crypto_ready{false};
        bool require_reward_binding{false};
        bool trigger_reauth{false};
    };
    SessionOwnershipStamp capture_session_ownership() const;
    SubmitContext capture_submit_context(uint32_t template_height,
                                         uint32_t chain_height) const;
    void record_session_event(SessionManager::SessionEventKind kind,
                              const std::string& detail) const;
    void finalize_keepalive_ack(const char* detail);
    void clear_generation_bound_state(const char* reason);
    bool finalize_and_feed_current_template(uint32_t unified_height,
                                            uint32_t effective_channel_height,
                                            const char* log_scope,
                                            bool snapshot_round_channel_height);
    static const PacketIngressPreflightOptions kDefaultPacketIngressPreflightOptions;
    bool run_packet_ingress_preflight(
        const char* log_scope,
        const PacketIngressPreflightOptions& options = kDefaultPacketIngressPreflightOptions);
    bool ensure_session_ready_for_ingress(const char* log_scope,
                                          const char* packet_name,
                                          bool queue_post_auth_get_block);
    void queue_pending_push_after_auth(const char* log_scope);
    void flush_pending_push_after_auth(const std::shared_ptr<network::Connection>& connection,
                                       const char* log_scope);
    // Integration helper functions (bridge MiningTemplateInterface and ClientChannelManager)
    /**
     * @brief Synchronize channel manager state with template interface
     * 
     * Called after GET_ROUND response to:
     * 1. Update channel manager heights
     * 2. Check for forks (auto-invalidate template if detected)
     * 3. Finalize template channel height if needed
     * 4. Validate current template against channel manager state
     * 
     * @param unified_height Current unified height from GET_ROUND
     * @param channel_height Current channel height from GET_ROUND (for THIS channel)
     * @return true if template is still valid, false if invalidated
     */
    bool sync_template_state(uint32_t unified_height, uint32_t channel_height);
    
    /**
     * @brief Check if current template is valid using HeightTracker snapshot state
     * 
     * Validates channel-specific staleness using a directional guard:
     * - Discards only when snap.channel_height >= template.nChannelHeight
     *   (chain tip has already met or passed our mining target).
     * - nChannelHeight > expectedChannel (multiple blocks ahead of local tracker) is
     *   intentionally accepted — normal during burst recovery when push notifications
     *   for intermediate blocks are still queued.
     * - Unified height differences are informational (other channels may advance)
     * 
     * @return true if template valid, false if stale/invalid
     */
    bool validate_current_template();
    
    /**
     * @brief Handle fork detection and template invalidation
     * 
     * Called when fork is detected to:
     * - Log rollback information
     * - Invalidate current template in MiningTemplateInterface
     * - Clear fork flag in channel manager
     * 
     * @param pManager Channel manager that detected the fork
     * @param current_height Current unified height after rollback
     */
    void handle_fork_detected(mining::ClientChannelManager* pManager, uint32_t current_height);

    /**
     * @brief Shared helper: update ClientChannelManager exactly once and dispatch
     *        fork or phantom-stake handling as appropriate.
     *
     * Calls pManager->UpdateFromGetRound(unified, channel), then:
     * - If IsForkDetected(): calls handle_fork_detected() and returns true
     *   (template discarded)
     * - If IsPhantomStakeRegression(): logs ⚡ PHANTOM STAKE info, clears both
     *   detection flags, returns false (template preserved — NOT a fork)
     * - Otherwise (normal advance or same-height): returns false
     *
     * @param unified_height  Unified blockchain height (tip-normalised)
     * @param channel_height  Channel-specific height
     * @return true if a genuine fork was detected and handled (template discarded),
     *         false for phantom stake regressions and normal advances
     */
    bool apply_channel_manager_update(uint32_t unified_height, uint32_t channel_height);

    struct LastSubmittedBlockState {
        bool valid{false};
        SessionOwnershipStamp owner{};
        uint64_t nonce{0};
        uint1024_t prev_hash{0};
        uint32_t height{0};
        uint32_t channel{0};

        void clear()
        {
            valid = false;
            owner.clear();
            nonce = 0;
            prev_hash = uint1024_t(0);
            height = 0;
            channel = 0;
        }
    };

    SessionOwnershipStamp get_last_submitted_owner_snapshot() const;
    void store_last_submitted_state(const SessionOwnershipStamp& owner,
                                    uint64_t nonce,
                                    const uint1024_t& prev_hash,
                                    uint32_t height,
                                    uint32_t channel);
    LastSubmittedBlockState get_last_submitted_state_snapshot() const;
    LastSubmittedBlockState consume_last_submitted_state();
    void clear_last_submitted_state();
    void clear_pending_submit_result_state();

    /// Called from every early-return / discard path inside submit_block()
    /// where a worker-solved block is refused before (or instead of) being
    /// placed on the wire as SUBMIT_BLOCK. Unlike BLOCK_REJECTED (the node
    /// rejecting a block it received), this counts blocks the NODE NEVER
    /// SAW — logs at critical level with a single greppable tag and
    /// increments stats::Global_delta::m_discarded_blocks so a solved-but-
    /// dropped block can never be silent (see Global.md / stats printer
    /// "discarded:" field). Bounded, additive change — no submission
    /// behaviour is altered, only visibility.
    void note_block_discarded(const std::string& reason);

    /**
     * @brief Unified height-state updater (single source of truth for both HeightTracker and ClientChannelManager)
     *
     * Called from push notification handlers and GET_ROUND handlers to ensure
     * HeightTracker and the active ClientChannelManager are always updated from
     * the same parsed data.  Fork detection is run immediately after the manager
     * update so that template invalidation is consistent across both sources.
     *
     * @param unified_height  Unified blockchain height from the parsed packet
     * @param channel_height  Channel-specific height from the parsed packet
     * @param difficulty_nbits Compact nBits difficulty from the parsed packet
     * @param source          Update origin (PUSH or GET_ROUND)
     */
    void update_height_state(uint32_t unified_height, uint32_t channel_height,
                             uint32_t difficulty_nbits, HeightTracker::UpdateSource source);

    // Opcode matching helpers (promoted from process_messages() lambdas so all on_* methods can use them)
    static bool matches_opcode(Packet const& packet, uint16_t legacy_opcode);
    static bool matches_stateless_opcode(Packet const& packet, uint16_t legacy_opcode);
    static bool requires_active_session_packet(Packet const& packet);

    /// Register all packet handlers with m_packet_router (called once from constructor).
    void register_packet_handlers();

    // Called from process_messages() after the lane/validity guards pass
    void on_miner_auth_response(Packet const& packet, std::shared_ptr<network::Connection> connection);
    void on_session_expired(Packet const& packet, std::shared_ptr<network::Connection> connection);
    void on_block_accepted(Packet const& packet, std::shared_ptr<network::Connection> connection);
    void on_block_rejected(Packet const& packet, std::shared_ptr<network::Connection> connection);
    enum class TriggerRecoveryOnStray { No, Yes };
    bool consume_pending_submit_result_or_warn(const char* opcode_name, TriggerRecoveryOnStray trigger_recovery);
    void on_block_data(Packet const& packet, std::shared_ptr<network::Connection> connection);
    void on_push_notification(Packet const& packet, std::shared_ptr<network::Connection> connection, uint32_t channel);
    void on_ping_diag(Packet const& packet, std::shared_ptr<network::Connection> connection);
    void on_session_status_ack(Packet const& packet, std::shared_ptr<network::Connection> connection);
    void on_get_round_response(Packet const& packet, std::shared_ptr<network::Connection> connection);
    void on_get_block_template(Packet const& packet, std::shared_ptr<network::Connection> connection);
    bool activate_push_lane_after_channel_ack(std::shared_ptr<network::Connection> connection);

    std::uint8_t m_channel;
    std::shared_ptr<spdlog::logger> m_logger;
    std::uint32_t m_current_height; ///< Diagnostic-only: used for BLOCK_DATA legacy fallback; NOT authoritative for staleness
    std::uint64_t m_current_reward;
    Set_block_handler m_set_block_handler;
    std::shared_ptr<stats::Collector> m_stats_collector;

    // hashPrevBlock snapshot (Gap 1): tip anchor captured at template parse time.
    // Equivalent to StakeMinter::hashLastBlock — a new template with a different
    // hashPrevBlock signals that the chain tip has moved.
    uint1024_t m_last_known_hash_prev_block;

    // Block-result counters are sourced from m_stats_collector Global stats.
    
    // Falcon miner authentication state (Phase 2)
    std::vector<uint8_t> m_miner_pubkey;
    std::vector<uint8_t> m_miner_privkey;
    bool m_authenticated;
    SessionId m_session_id;
    SessionEpoch m_session_epoch{};
    bool m_has_seen_session_epoch{false};
    uint64_t m_cached_runtime_state_generation{0};
    SessionIdentity m_cached_identity{};  // Cached canonical identity from SessionManager
    std::shared_ptr<EpochCoordinator> m_epoch_coordinator;
    std::string m_address;  // Miner's network address for auth message
    std::uint64_t m_auth_timestamp;  // Timestamp for auth message
    AuthState m_auth_state;  // Authentication state machine
    std::chrono::steady_clock::time_point m_auth_in_flight_since{};  // When auth entered in-flight; default-constructed = not in-flight
    std::string m_miner_id;  // Miner identifier (optional)

    // Consecutive KEEPALIVE_V2_ACK session-ID mismatch counter.
    // Incremented each time handle_session_id_mismatch() detects a mismatch;
    // reset to zero on a successful (matching) ACK.  The session is only
    // self-expired once this reaches SESSION_MISMATCH_EXPIRE_THRESHOLD,
    // preventing premature expiry on late/replayed ACKs or node-side races.
    uint32_t m_session_id_mismatch_count{0};

    // ── Shadow-ban detection ────────────────────────────────────────────────
    // Counts consecutive packet-ingress preflight rejections for response
    // packets (GET_ROUND replies, BLOCK_ACCEPTED/REJECTED, keepalive ACK).
    // When the counter reaches SHADOW_BAN_PREFLIGHT_THRESHOLD the miner
    // forces a full re-auth to break out of the "shadow ban" state where
    // PUSH still arrives but all miner-initiated round-trips are silently
    // dropped by the miner's own preflight logic.
    uint32_t m_preflight_reject_count{0};
    static constexpr uint32_t SHADOW_BAN_PREFLIGHT_THRESHOLD = 5;

    // Unanswered GET_ROUND counter: incremented ONLY after a GET_ROUND packet has
    // been successfully handed to transmit() (via note_get_round_transmitted()).
    // Reset to zero when a NEW_ROUND / OLD_ROUND response is processed.
    // std::atomic because Worker_manager reads via get_unanswered_get_round_count()
    // from its own thread while Solo modifies it on the io_context thread.
    std::atomic<uint32_t> m_unanswered_get_round_count{0};
    // Time of the earliest GET_ROUND that is still unanswered (set by
    // note_get_round_transmitted() when the counter goes from 0→1; cleared on reset).
    std::chrono::steady_clock::time_point m_earliest_unanswered_get_round_at{};
    // Time of the most recent successfully-transmitted GET_ROUND.
    std::chrono::steady_clock::time_point m_last_get_round_transmitted_at{};

    // ── NEW_ROUND recovery debounce (operator-directed 5s symmetric gate) ────────
    // Prevents the "panic GET_BLOCK" that fires immediately when NEW_ROUND arrives
    // with an invalid template, racing the BLOCK_DATA push that almost always
    // arrives within 50–500ms.  Both the "PUSH-before-NEW_ROUND" and
    // "NEW_ROUND-before-PUSH" orderings are handled symmetrically.
    //
    // m_io_context: ASIO context for the async timer.  Null in test environments
    //   that do not supply one — in that case the debounce falls back to legacy
    //   immediate behaviour so existing tests are unaffected.
    // m_recovery_timer: fires the deferred recovery GET_BLOCK after 5s.  Null
    //   when m_io_context is null.
    // m_last_block_accepted_time / m_last_push_received_time: timestamps of the
    //   most recent BLOCK_ACCEPTED and PUSH arrivals (used for diagnostic logging).
    // m_recovery_deferred_at: set when the timer is scheduled; cleared on fire or
    //   cancel.  sentinel = time_point::min() (not scheduled).
    static constexpr auto kRecoveryDebounceWindow = std::chrono::seconds(5);
    // Time at which the most recent BLOCK_DATA template was successfully adopted.
    // Used to suppress redundant GET_BLOCK requests from polling races immediately
    // after a fresh template was already fed to workers.
    std::chrono::steady_clock::time_point m_last_template_adopted_at{
        std::chrono::steady_clock::time_point::min()};
    static constexpr auto POST_ADOPTION_SUPPRESSION_WINDOW = std::chrono::milliseconds(2500);
    std::shared_ptr<asio::io_context> m_io_context;
    std::unique_ptr<asio::steady_timer> m_recovery_timer;
    std::unique_ptr<asio::steady_timer> m_get_block_control_timer;
    std::atomic<int64_t> m_get_block_retry_at_ms{0};
    std::atomic<uint64_t> m_get_block_control_generation{0};
    void cancel_get_block_control_retry(bool clear_deadline = true);
    void on_get_block_control(Packet const& packet, const std::shared_ptr<network::Connection>& connection);
    std::chrono::steady_clock::time_point m_last_block_accepted_time{
        std::chrono::steady_clock::time_point::min()};
    std::chrono::steady_clock::time_point m_last_push_received_time{
        std::chrono::steady_clock::time_point::min()};
    std::chrono::steady_clock::time_point m_recovery_deferred_at{
        std::chrono::steady_clock::time_point::min()};
    int m_recovery_fired_count{0}; ///< Incremented each time the 5s timer fires (for tests)

    // Schedule a deferred (5s) recovery GET_BLOCK in response to NEW_ROUND with
    // no valid template.  Cancels any previously pending deferred recovery so
    // rapid NEW_ROUND bursts are coalesced.  Falls back to immediate behaviour
    // when no io_context is available (legacy / test environments).
    void schedule_recovery_get_block(std::shared_ptr<network::Connection> connection,
                                     uint32_t unified_height);

    // Cancel any pending deferred recovery GET_BLOCK (called from PUSH /
    // BLOCK_ACCEPTED handlers when a response has arrived before the 5s window
    // elapsed — "push won the race").
    void cancel_recovery_timer(const char* handler_name);

    // Consecutive hashPrevBlock mismatch counter (chain-in-flux doom-loop guard).
    // Incremented each time validate_current_template() detects a hashPrevBlock mismatch
    // and would normally discard the template.  After MAX_CONSECUTIVE_HASHPREV_MISMATCHES
    // consecutive mismatches the template is accepted (not discarded) to prevent the
    // NO VALID TEMPLATE doom loop that occurs when the node is under attack or its chain
    // tip is churning rapidly (e.g. orphan limit exceeded by DDoS peer).
    // Reset to zero whenever a template is successfully validated and fed to workers.
    // std::atomic because Worker_manager reads via get_hashprev_mismatch_consecutive()
    // from its own thread while Solo modifies it on the io_context thread.
    std::atomic<uint32_t> m_hashprev_mismatch_consecutive{0};
    static constexpr uint32_t MAX_CONSECUTIVE_HASHPREV_MISMATCHES = 3;

    // HashCheckpoint Guard: rolling window of recent canonical hashPrevBlock values.
    // Advisory-only — NEVER blocks template acceptance. The NODE is authoritative.
    // Provides reorg depth estimation and shallow-vs-deep reorg classification
    // for Colin diagnostics. HashCheckpoints are immutable once recorded; unlike
    // hashPrevBlock (which can change during node reorgs), checkpoints represent
    // confirmed chain tips that the node has built on.
    HashCheckpointGuard m_hash_checkpoint_guard;

    // MerkleRoot Feed Guard: suppresses duplicate worker feeds when the same
    // hashMerkleRoot arrives within 5 seconds.  Only the feed (worker distribution)
    // is suppressed — the receive and validation still proceed normally.
    MerkleRootFeedGuard m_merkle_root_feed_guard;

    // Unified-height feed guard: prevents same-height re-feeds within a cooldown.
    // This is the ultimate backstop — even if transmit-side dedup fails, this guard
    // prevents the actual worker restart for the same (height, hashPrevBlock) pair.
    // Cooldown is long enough to suppress all duplicate arrivals but short enough
    // to allow legitimate template refreshes (e.g., fee-optimization updates).
    static constexpr int64_t SAME_HEIGHT_FEED_COOLDOWN_SECONDS = 10;
    uint32_t     m_last_fed_unified_height{0};
    uint1024_t   m_last_fed_hash_prev_block{};
    std::chrono::steady_clock::time_point m_last_fed_time{};

    // Counts same-tip re-confirmations observed while the guard above suppresses a
    // re-feed (node re-served BLOCK_DATA that is identical to what is already loaded).
    // Surfaced in logs so operators can distinguish "miner alive, node just re-confirmed
    // the same still-valid tip" from a genuinely stalled recovery. Also read by
    // Worker_manager for diagnostics via get_same_tip_reconfirmation_count().
    uint64_t m_same_tip_reconfirmation_count{0};
    
    // Unified Falcon Signature Wrapper (Phase 2 enhancement)
    std::unique_ptr<FalconSignatureWrapper> m_falcon_wrapper;
    bool m_disposable_falcon_enabled;     // Disposable Falcon signing (ALWAYS ON - core protocol, 0 blockchain overhead)
    
    // ChaCha20 encryption wrapper for Falcon pubkey protection
    std::unique_ptr<ChaCha20Wrapper> m_chacha20_wrapper;
    bool m_enable_chacha20;  // ChaCha20 encryption (ALWAYS ON - core security for localhost + SessionID)

    // Session context for centralized session management (passed from NodeSession)
    // This is the authoritative source for session state shared across primary/secondary protocols
    std::shared_ptr<NodeSessionContext> m_session_context;

    // KEEPALIVE_V2 (0xD100) send-side tracking:
    // The lo32 of hashPrevBlock that the miner put in its last KEEPALIVE_V2 frame.
    // Stored locally so the ACK handler can compare against the node's echoed value
    // without trusting the potentially-tampered echo in ack.hashPrevBlock_lo32.
    uint32_t m_last_keepalive_prevhash_lo32{0};
    SessionOwnershipStamp m_last_keepalive_request_owner{};
    mutable SessionOwnershipStamp m_last_session_status_request_owner{};
    SessionOwnershipStamp m_last_reward_request_owner{};
    SessionOwnershipStamp m_last_get_block_request_owner{};
    mutable std::mutex m_last_submitted_mutex;
    LastSubmittedBlockState m_last_submitted_state{};

    // Mining Template Interface for unified READ/FEED operations
    std::unique_ptr<MiningTemplateInterface> m_template_interface;
    
    // Centralized height tracker (single source of truth for heights)
    HeightTracker m_height_tracker;
    
    // Connection for multi-packet authentication flow
    std::shared_ptr<network::Connection> m_connection;
    
    // Persistent tritium genesis (preserved across reconnections)
    std::vector<uint8_t> m_persistent_tritium_genesis;
    std::vector<uint8_t> m_cached_chacha20_key_genesis;
    std::vector<uint8_t> m_cached_chacha20_key;

    // Stateless mining reward address binding (MINER_SET_REWARD protocol)
    std::string m_reward_address;  // NXS account address for mining rewards
    bool m_reward_bound;  // True after successful MINER_REWARD_RESULT

    // True when the node-confirmed session genesis (echoed in SESSION_START) does
    // not match the configured reward_address. The miner refuses to submit blocks
    // in this state because the node's Coinbase::Verify / signature check will
    // reject them — see Solo::submit_block. Cleared on disconnect/reauth.
    bool m_reward_genesis_mismatch{false};
    
    // Push notification subscription state (MINER_READY sent after auth)
    std::atomic<bool> m_subscribed_to_notifications{false};
    bool m_pending_push_after_auth{false};

    // Recovery callback — invoked when a push handler fires GET_BLOCK for a stale template
    // (channel_advanced staleness), signalling Worker_manager to enter recovery_pending state.
    Recovery_handler m_recovery_handler;

    // Session-expired callback — invoked when a keepalive ACK carries a mismatched session_id,
    // signalling Worker_manager to trigger recovery for the stale session.
    Session_expired_handler m_session_expired_handler;

    // Recovery-confirmed callback — invoked when the same-height feed guard in
    // finalize_and_feed_current_template() suppresses a re-feed because the node re-served
    // an unchanged template (same unified height AND hashPrevBlock already loaded). Lets
    // Worker_manager clear an in-flight recovery immediately on this proof-of-liveness
    // instead of waiting for a genuinely new tip or escalating toward degraded mode.
    Recovery_confirmed_handler m_recovery_confirmed_handler;

    // Session-authenticated callback — invoked after MINER_AUTH_RESULT processing is complete.
    // Worker_manager uses this to check session_id=0 and trigger retry if needed.
    Session_authenticated_handler m_session_authenticated_handler;
    Work_ready_handler m_work_ready_handler;

    // Session-start callback — invoked when SESSION_START is received and keepalive interval
    // has been auto-adjusted from the node-advertised timeout.
    // Worker_manager uses this to cache the node-advertised interval for future connections.
    Session_start_handler m_session_start_handler;  // Notifies Worker_manager of node-advertised keepalive

    // Block-accepted callback — invoked on BLOCK_ACCEPTED to record the mined block.
    Block_accepted_handler m_block_accepted_handler;

    // Node-shutdown callback — invoked on NODE_SHUTDOWN (0xD0FF) to stop workers
    // and set reconnect backoff.
    Node_shutdown_handler m_node_shutdown_handler;

    // Replacement-pending callback — invoked after a PUSH notification marks the
    // current template replacement-pending, carrying the deadline so
    // Worker_manager can schedule a precise one-shot timeout check.
    Replacement_pending_handler m_replacement_pending_handler;

    // Last submitted block gate/state — carried forward from submit_block() so the
    // ACCEPT/GOOD_BLOCK handler uses the actual submitted values rather than
    // re-reading from a potentially-replaced template (Priority 2 fix).
    SubmitResultGate m_submit_result_gate{};
    
    // GET_ROUND status tracking (Template Staleness Prevention - LLL-TAO PR #131)
    RoundStatus m_last_round_status;  // Last received round status
    // Channel height from the last GET_ROUND response. Updated after the NEW_ROUND
    // polling-state decision so logs can show how the active channel changed (or
    // stayed flat) when the authoritative unified height advanced.
    uint32_t m_last_round_channel_height{0};
    // Unified height from the last GET_ROUND response. Used as the primary
    // NEW_ROUND/OLD_ROUND discriminator: a new block on ANY channel (Prime, Hash,
    // or Stake) advances unified height and means all miners need fresh templates.
    // Stored separately from m_last_round_status.height for clarity.
    uint32_t m_last_round_unified_height{0};
    
    // Client-side fork-aware channel managers (mirrors NODE's PR #136)
    // INTEGRATION PATTERN:
    // - MiningTemplateInterface (m_template_interface): Manages template storage and worker distribution
    // - ClientChannelManagers (m_prime_manager, m_hash_manager): Track heights and detect forks
    // 
    // Division of Responsibilities:
    // 1. MiningTemplateInterface:
    //    - Stores current mining template (MiningTemplate with LLP::CBlock)
    //    - Feeds templates to worker threads
    //    - Handles template age tracking for the WAITING_TEMPLATE recovery window
    //    - Manages channel height (set via set_channel_height())
    // 
    // 2. ClientChannelManagers:
    //    - Track node heights from GET_ROUND (unified + channel)
    //    - Detect blockchain forks (height regression)
    //    - Provide validation logic (ValidateTemplate mirrors NODE's Block::Accept)
    //    - Do NOT store templates in production (use MiningTemplateInterface instead)
    // 
    // Fork Detection Flow:
    //   GET_ROUND response → Update managers → Fork detected? → Clear MiningTemplateInterface
    // 
    // This separation mirrors NODE's architecture where ChannelStateManager tracks state
    // but doesn't duplicate Block storage - blocks live in the blockchain database.
    std::unique_ptr<mining::PrimeClientManager> m_prime_manager;
    std::unique_ptr<mining::HashClientManager> m_hash_manager;
    
    // ═══════════════════════════════════════════════════════════════════════
    // GET_ROUND INTELLIGENT POLLING STATE
    // ═══════════════════════════════════════════════════════════════════════
    
    // Timing state
    std::chrono::steady_clock::time_point m_last_get_round_time;
    uint32_t m_current_poll_interval_ms;  // Current interval (adaptive)
    
    // Note: When POLLING_ENABLED is true, backoff multiplier is 1.5x via integer arithmetic: interval + (interval >> 1)
    
    // State flags
    bool m_needs_initial_round_check;  // Set true when new template received
    uint32_t m_template_unified_height;    // Informational only: unified height at last template receipt (logging unified drift).
                                           // NOT used for staleness decisions (channel height is authoritative).
    bool m_get_round_push_silent_fallback_active{false};  // Armed after PUSH silence >= 600s; cleared by new PUSH.
    
    // Helper methods for intelligent polling
    bool should_poll_get_round();
    std::chrono::milliseconds push_in_transit_guard_for_channel() const;
    void arm_get_round_fallback(int64_t push_silent_seconds);
    void disarm_get_round_fallback(const char* reason, int64_t push_age_seconds = -1);
    void on_new_round_received(uint32_t new_unified_height);
    void on_old_round_received();
    void on_template_received(uint32_t template_height);
    void check_unified_height_delta(uint32_t current_unified_height);
    
    // ═══════════════════════════════════════════════════════════════════════
    // PORT-LANE SEPARATION STATE (STRICT - NO FALLBACK)
    // ═══════════════════════════════════════════════════════════════════════
    
    // Protocol lane (determined once from connection port, never changes)
    ProtocolLane m_protocol_lane;
    
    // Unified push notification handler (consolidates 4 duplicate handlers)
    std::unique_ptr<PushNotificationHandler> m_push_handler;
    
    // Helper method to determine and log lane from connection
    void initialize_protocol_lane(std::shared_ptr<network::Connection> connection);

    // ═══════════════════════════════════════════════════════════════════════
    // COLIN AI DIAGNOSTIC PING/PONG HANDLER
    // ═══════════════════════════════════════════════════════════════════════
    // Handles PING_DIAG (0xE0 legacy / 0xD0E0 stateless) from node and
    // replies with a 64-byte PongFrame containing live miner telemetry.
    ::LLP::ColinPingHandler m_colin_ping_handler;

    // ── SESSION_STATUS_ACK tracking ─────────────────────────────────────────
    // Updated whenever SESSION_STATUS_ACK (0xD0DC / legacy 220) is received.
    ::LLP::SessionStatusAckFrame m_last_session_status_ack{};
    std::chrono::steady_clock::time_point m_last_session_status_ack_time{};

    // ── GET_BLOCK deduplication ──────────────────────────────────────────────
    // Centralized dedup guard: reason-aware three-tier policy (bypass_all,
    // bypass_height, full dedup).  See get_block_dedup_guard.hpp.
    GetBlockDedupGuard m_dedup_guard;
    std::atomic<GetBlockRequestStatus> m_last_get_block_request_status{GetBlockRequestStatus::NONE};

    // ── Packet dispatch router ──────────────────────────────────────────────
    // Table-driven dispatch replacing the if/else-if chain in process_messages().
    PacketRouter m_packet_router;

    // ── In-flight GET_BLOCK awareness (cross-handler dedup tier) ────────────
    // Bridges the gap between PUSH and GET_ROUND handlers: when PUSH sends a
    // GET_BLOCK, the pending state tells GET_ROUND (arriving 0.5-3s later) that
    // a response is already expected, preventing a duplicate request that would
    // cause workers to restart and waste ~5 seconds of mining work.
    struct PendingGetBlock {
        bool                                     active{false};
        uint32_t                                 unified_height{0};
        std::chrono::steady_clock::time_point    sent_at{};
        GetBlockReason                           reason{GetBlockReason::INITIAL_REQUEST};

        static constexpr int64_t TIMEOUT_SECONDS = 4;  // Auto-expire if no response — allows for WAN/burst latency

        /// Returns true if a GET_BLOCK is already in-flight for the given height
        /// (or a higher height).  Auto-expires after TIMEOUT_SECONDS.
        bool is_pending_for(uint32_t height) const {
            if (!active) return false;
            // Pending at same or higher height — a response for unified_height
            // will produce a template valid for 'height' as well.
            if (unified_height < height) return false;
            // Timeout check: auto-expire stale pending state
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - sent_at).count();
            return elapsed < TIMEOUT_SECONDS;
        }

        void mark_pending(uint32_t height, GetBlockReason r) {
            active         = true;
            unified_height = height;
            sent_at        = std::chrono::steady_clock::now();
            reason         = r;
        }

        void clear() { active = false; }

        /// Returns elapsed milliseconds since the request was sent, or -1 if not active.
        /// Useful for timeout diagnostics at call sites.
        int64_t elapsed_ms() const {
            if (!active) return -1;
            return std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - sent_at).count();
        }

        /// Returns true if a request was marked active but has expired (timed out
        /// without a BLOCK_DATA response clearing it).  Does NOT clear the flag —
        /// that happens in on_block_data() / on_get_block_template() or via is_pending_for().
        bool has_timed_out() const {
            if (!active) return false;
            return elapsed_ms() >= TIMEOUT_SECONDS * 1000;
        }
    };
    PendingGetBlock m_pending_get_block;

    friend struct PostAdoptionSuppressionHarness;
    friend struct SameTipReconfirmationHarness;
    friend struct GetBlockControlHarness;

public:
    /// Mark a GET_BLOCK request as successfully transmitted using the current
    /// height-tracker snapshot. Must be called immediately after the payload has
    /// been accepted by Connection::transmit() so dedup/pending state only moves
    /// forward on confirmed outbound progress.
    void mark_get_block_pending(GetBlockReason reason);

    /// Clear the in-flight GET_BLOCK marker (exposed for testing / forced reset).
    void clear_get_block_pending() { m_pending_get_block.clear(); }

private:
    bool queue_payload(const std::shared_ptr<network::Connection>& connection,
                       const network::Shared_payload& payload,
                       const char* context);
    bool request_and_queue_get_block(const std::shared_ptr<network::Connection>& connection,
                                     GetBlockReason reason,
                                     const char* context);

    // Single chokepoint used by failure-recovery paths in on_block_data,
    // on_get_block_template, on_block_rejected (fork), and stray-result
    // recovery.  Delegates to m_recovery_handler when registered (the normal
    // production path through Worker_manager::retry_template_request, which
    // applies the 500 ms burst guard, hashprev backoff, auth/session/in-flight
    // gates, and rate accounting).  Only if no handler is wired (e.g. unit
    // tests, or NodeSession-less mode) does it fall back to a direct local
    // request_and_queue_get_block.
    //
    // This prevents the "double tap" storm where the same failure event used
    // to both call m_recovery_handler() AND immediately call
    // request_and_queue_get_block() — racing two concurrent GET_BLOCKs through
    // partially-disjoint dedup guards.
    void dispatch_recovery_or_fallback(const std::shared_ptr<network::Connection>& connection,
                                       GetBlockReason reason,
                                       const char* fallback_context);
    
    // ═══════════════════════════════════════════════════════════════════════
    // PROTOCOL LANE DETERMINATION
    // ═══════════════════════════════════════════════════════════════════════
    // Protocol lane is strictly determined by connection port (no negotiation/fallback):
    // - Port 8323: Legacy lane (8-bit opcodes, polling)
    // - Port 9323+: Stateless lane (16-bit opcodes, push notifications)
    // 
    // State is anchored on:
    // - m_protocol_lane: Authoritative lane identifier (set at connection time)
    // - m_auth_state: Authentication state (managed by session manager)
    // - Push readiness: Inferred from actual template delivery events
};

}
}
#endif
