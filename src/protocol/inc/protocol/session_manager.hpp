#ifndef NEXUSMINER_PROTOCOL_SESSION_MANAGER_HPP
#define NEXUSMINER_PROTOCOL_SESSION_MANAGER_HPP

#include <vector>
#include <cstdint>
#include <string>
#include <memory>
#include <chrono>
#include <atomic>
#include <mutex>
#include <shared_mutex>
#include <array>
#include <deque>
#include <functional>
#include <optional>
#include "asio/io_context.hpp"
#include "asio/steady_timer.hpp"
#include "network/types.hpp"
#include "protocol/session_binding.hpp"
#include "protocol/session_semantic_types.hpp"
#include "protocol/session_identity.hpp"
#include "protocol_lane.hpp"
#include "spdlog/spdlog.h"
#include "LLP/include/colin_ping_protocol.h"
#include "protocol/epoch_coordinator.hpp"

namespace nexusminer {
namespace network { class Connection; }
namespace protocol {

class SessionManager : public std::enable_shared_from_this<SessionManager> {
public:
    static constexpr std::size_t SESSION_EVENT_JOURNAL_CAPACITY = 16;

    // 4 clean states
    enum class SessionState {
        DISCONNECTED,    // No TCP connection to node
        AUTHENTICATING,  // Handshake in progress
        AUTHENTICATED,   // Session ID valid, mining can proceed
        DEGRADED,        // Connection lost / keepalive failed; workers should pause
        // Backward-compat aliases
        ACTIVE   = AUTHENTICATED,
        EXPIRED  = DEGRADED
    };

    // Kept for backward compat
    enum class RewardState {
        NONE,
        REQUIRED,
        BINDING,
        BOUND,
        REJECTED,
        STALE
    };

    enum class RecoveryState {
        HEALTHY,
        RECOVERY_PENDING,
        RECOVERY_IN_PROGRESS,
        FORCED_REAUTH,
        RECONNECT_REQUIRED
    };

    enum class ExpiryState {
        FRESH,
        KEEPALIVE_MISMATCH_WARNING,
        STALE_ACK_IGNORED,
        EXPIRED_ACCEPTED,
        EXPIRED_REJECTED,
        AUTH_TIMEOUT,
        DEAD_SESSION_TIMEOUT
    };

    using SessionExpiredHandler = std::function<void()>;

    enum class SessionEventKind {
        AUTH_INIT,
        AUTH_SUCCESS,
        SESSION_START,
        REWARD_BIND_SENT,
        REWARD_BOUND,
        REWARD_BIND_RESULT = REWARD_BOUND,
        KEEPALIVE_ACK,
        KEEPALIVE_MISSED,
        STATUS_ACK_ACCEPTED,
        STATUS_ACK_REJECTED,
        DEGRADED,
        FORCED_REAUTH = DEGRADED,
        DISCONNECTED,
        SESSION_RESET,
        SUBMIT_SENT,
        SUBMIT_ACCEPTED,
        SUBMIT_REJECTED,
        // Backward-compat event kinds used by solo.cpp and tests
        STALE_PACKET_DROPPED,
        EPOCH_MISMATCH,
        RECOVERY_REQUESTED,
        RECOVERY_HEALTHY
    };

    struct SessionEvent {
        uint64_t timestamp{0};
        SessionEventKind kind{SessionEventKind::AUTH_INIT};
        SessionId session_id{};
        SessionEpoch session_epoch{};
        std::string detail;
    };

    // Full session info — new minimal fields plus backward-compat fields
    struct SessionInfo {
        // ── Core minimal fields (new design) ──────────────────────────────
        SessionId session_id{};
        SessionEpoch session_epoch{};
        uint64_t runtime_state_generation{0};
        SessionState state{SessionState::DISCONNECTED};
        ProtocolLane active_lane{ProtocolLane::UNKNOWN};
        std::string reward_address;   // canonical name (new design)
        bool reward_bound{false};
        std::array<uint8_t, 4> prevblock_suffix{};
        uint64_t session_start{0};
        uint64_t last_activity{0};
        std::chrono::system_clock::time_point last_keepalive{};
        uint32_t keepalive_count{0};
        bool authenticated{false};   // Convenience: derived from state

        // ── Backward-compat fields (used by solo.cpp and existing tests) ──
        std::string remote_endpoint;
        std::string local_endpoint;
        bool connected{false};
        std::vector<uint8_t> falcon_pubkey;
        FalconHashKeyId falcon_key_id{};
        bool falcon_authenticated{false};
        std::vector<uint8_t> session_key;
        SessionGenesisHash session_genesis{};
        std::vector<uint8_t> chacha20_session_key;
        SessionFingerprint chacha20_key_fingerprint{};
        bool chacha20_ready{false};
        std::string reward_address_string;  // backward-compat alias; prefer reward_address in new code
        RewardHash reward_hash{};
        RewardState reward_state{RewardState::NONE};
        std::string reward_binding_source;
        uint32_t channel{0};
        bool ready_for_submit{false};
        bool ready_for_get_block{false};
        RecoveryState recovery_state{RecoveryState::HEALTHY};
        std::string recovery_reason;
        ExpiryState expiry_state{ExpiryState::FRESH};
        std::string expiry_reason;
        bool deferred_push_replay_allowed{false};
        bool get_block_replay_allowed{false};
        bool queued_replay_survives_partial_readiness{false};
        uint64_t created_at{0};
        uint64_t last_auth_time{0};
        uint64_t last_reward_bind_time{0};
        std::chrono::system_clock::time_point session_start_tp{};
    };
    using MinerSessionContainer = SessionInfo;
    using RuntimeSessionSnapshot = SessionInfo;

    struct RewardBindReadiness {
        bool ready{false};
        std::string reason;
    };

    // ── Constructors ──────────────────────────────────────────────────────────
    // New minimal constructor
    explicit SessionManager(std::shared_ptr<asio::io_context> io_context = nullptr);
    // Backward-compat constructor (tests use SessionManager(24, nullptr))
    explicit SessionManager(uint16_t keepalive_interval_hours,
                            std::shared_ptr<asio::io_context> io_context = nullptr);
    ~SessionManager();

    // ── Core session lifecycle (new minimal API) ──────────────────────────────
    void begin_auth();
    void commit_authenticated(SessionId session_id, ProtocolLane lane,
                              const std::string& reward_address = {});
    void commit_reward_bound(const std::string& reward_address,
                             const std::string& source = "");
    void mark_degraded(const std::string& reason = {});
    void end_session();

    // ── Backward-compat lifecycle ─────────────────────────────────────────────
    void begin_auth_handshake(const std::string& detail = "");
    void commit_authenticated_session(SessionId session_id,
                                      const std::vector<uint8_t>& pubkey = {},
                                      FalconHashKeyId key_id = {},
                                      SessionGenesisHash tritium_genesis = {});
    void commit_authenticated_session(SessionId session_id,
                                      const std::vector<uint8_t>& pubkey,
                                      const std::string& key_id,
                                      const std::vector<uint8_t>& tritium_genesis);
    void start_session(SessionId session_id,
                       const std::vector<uint8_t>& session_key = {},
                       SessionGenesisHash tritium_genesis = {});
    void start_session(SessionId session_id,
                       const std::vector<uint8_t>& session_key,
                       const std::vector<uint8_t>& tritium_genesis);
    void mark_session_expired(const std::string& reason);
    void mark_recovery_required(const std::string& reason);
    void mark_recovery_healthy(const std::string& reason = "");
    void clear_for_disconnect(const std::string& reward_address = {},
                              const std::string& reward_source = "",
                              const std::string& reason = "",
                              bool preserve_genesis = true);
    void clear_for_reauth(const std::string& reward_address = {},
                          const std::string& reward_source = "",
                          const std::string& reason = "",
                          bool preserve_genesis = true);

    void begin_reward_binding(const std::string& addr,
                              RewardHash hash = {},
                              const std::string& src = "");
    void begin_reward_binding(const std::string& addr,
                              const std::vector<uint8_t>& hash,
                              const std::string& src);
    // Note: commit_reward_bound(addr, source) is the new API
    // Backward-compat overload that also accepts reward_hash
    void commit_reward_bound(const std::string& reward_address,
                             RewardHash reward_hash,
                             const std::string& source = "");
    void commit_reward_bound(const std::string& reward_address,
                             const std::vector<uint8_t>& reward_hash,
                             const std::string& source);
    void commit_reward_rejected(const std::string& addr,
                                const std::string& src = "",
                                const std::string& rsn = "");
    void note_keepalive_ack(bool accepted, const std::string& detail = "");
    void record_keepalive();
    void record_keepalive_ack(bool accepted);

    // No-ops (callers will be removed in future PRs)
    void set_connection_metadata(const std::string& local, const std::string& remote,
                                 bool connected);
    void set_falcon_identity(const std::vector<uint8_t>& pubkey,
                             FalconHashKeyId key_id, bool authenticated);
    void set_falcon_identity(const std::vector<uint8_t>& pubkey,
                             const std::string& key_id, bool authenticated);
    void reset_session_credentials();
    void set_chacha20_session_key(const std::vector<uint8_t>& key,
                                  SessionFingerprint fingerprint, bool ready);
    void set_chacha20_session_key(const std::vector<uint8_t>& key,
                                  const std::string& fingerprint, bool ready);
    void set_reward_binding(const std::string& addr,
                            RewardHash hash,
                            bool bound, const std::string& src);
    void set_reward_binding(const std::string& addr,
                            const std::vector<uint8_t>& hash,
                            bool bound, const std::string& src);
    void set_channel_state(uint32_t channel, bool ready_for_submit,
                           bool ready_for_get_block);
    void mark_activity();
    void set_tritium_genesis(SessionGenesisHash genesis);
    void set_tritium_genesis(const std::vector<uint8_t>& genesis);
    void set_keepalive_interval(uint16_t hours);
    void set_prevblock_suffix(const std::array<uint8_t, 4>& suffix);
    void set_protocol_lane(ProtocolLane lane);
    void set_connection(std::shared_ptr<network::Connection> connection);
    void set_state(SessionState state);

    // ── State queries ─────────────────────────────────────────────────────────
    bool is_authenticated() const;
    bool is_active() const { return is_authenticated(); }
    bool is_degraded() const;
    bool is_reward_bound() const;
    bool can_submit() const;
    bool can_submit_work() const;
    bool can_request_get_block() const;
    bool session_requires_full_recovery() const;
    bool session_may_request_work() const;
    bool session_is_fully_mining_ready() const;
    bool allow_deferred_push_replay() const;
    bool allow_get_block_replay() const;
    bool reward_binding_required() const;
    RewardBindReadiness get_reward_bind_readiness() const;
    bool validate_miner_session(std::string* reason = nullptr) const;
    std::string build_miner_session_diagnostics() const;
    SessionId get_session_id() const;
    SessionEpoch get_session_epoch() const;
    uint64_t peek_runtime_state_generation() const noexcept;
    SessionState get_state() const;
    SessionInfo get_session_info() const;
    RuntimeSessionSnapshot get_runtime_snapshot() const;
    SessionIdentity get_canonical_identity() const;
    SessionBinding get_session_binding() const;
    std::chrono::seconds get_session_uptime() const;
    std::vector<uint8_t> get_session_key() const;
    SessionGenesisHash get_typed_tritium_genesis() const;
    std::vector<uint8_t> get_tritium_genesis() const;
    std::chrono::seconds get_time_until_keepalive() const { return std::chrono::seconds(0); }
    uint16_t get_keepalive_interval() const { return m_keepalive_interval_hours; }
    std::chrono::seconds get_keepalive_timer_interval() const;
    uint16_t map_auth_opcode(uint8_t legacy_opcode) const;

    // ── Session Keepalive ────────────────────────────────────────────────────
    // Single keepalive system: SESSION_KEEPALIVE packets at node-derived interval.
    // Early ping at +10s after auth, then every m_keepalive_interval_hours.
    void start_keepalive_timer();
    void stop_keepalive_timer();
    network::Shared_payload build_keepalive_packet() const;
    network::Shared_payload build_session_status_packet(bool degraded, bool has_template,
                                                        bool workers_running,
                                                        bool secondary_up) const;

    // ── Session expired callback ──────────────────────────────────────────────
    void set_session_expired_handler(SessionExpiredHandler h) {
        m_session_expired_handler = std::move(h);
    }

    // ── Event journal ─────────────────────────────────────────────────────────
    void record_session_event(SessionEventKind kind, const std::string& detail = "");
    std::vector<SessionEvent> get_session_event_journal() const;
    std::string build_session_event_journal() const;

    // Static name helpers (used by diagnostics)
    static const char* session_event_kind_name(SessionEventKind kind);
    static const char* reward_state_name(RewardState state);
    static const char* recovery_state_name(RecoveryState state);
    static const char* expiry_state_name(ExpiryState state);

    /// Wire the shared EpochCoordinator (called by Worker_manager before sessions begin).
    void set_epoch_coordinator(std::shared_ptr<EpochCoordinator> coordinator);

private:
    void transition_to_authenticated_locked(SessionId session_id,
                                            SessionGenesisHash tritium_genesis);
    void clear_runtime_session_locked(bool preserve_genesis, bool clear_prevblock_suffix);
    void update_replay_allowances_locked();
    void bump_runtime_state_generation_locked();
    void clear_session_event_journal_locked();
    void record_session_event_locked(SessionEventKind kind, const std::string& detail);
    void schedule_regular_keepalives(const std::shared_ptr<SessionManager>& self);
    void send_keepalive(const char* cadence);
    std::chrono::seconds get_session_uptime_locked() const;
    static bool validate_miner_session_container_locked(const SessionInfo& session,
                                                        std::string* reason);

    mutable std::shared_mutex m_session_mutex;
    SessionInfo m_session;
    SessionIdentity m_canonical_identity;   // Frozen at auth time; cleared on disconnect/reauth
    std::deque<SessionEvent> m_session_event_journal;

    // Bug 10 fix: Archive previous journal entries across re-auth instead of
    // dropping them.  Preserves the failure/degradation events that triggered
    // re-auth for post-mortem debugging.  Capped at MAX_ARCHIVED_EVENTS to
    // prevent unbounded memory growth.
    static constexpr size_t MAX_ARCHIVED_EVENTS = 64;
    std::deque<SessionEvent> m_archived_event_journal;

    uint16_t m_keepalive_interval_hours{12};
    bool m_preserve_genesis_on_disconnect{true};
    ProtocolLane m_protocol_lane{ProtocolLane::UNKNOWN};

    std::shared_ptr<asio::io_context> m_io_context;
    std::shared_ptr<asio::steady_timer> m_keepalive_timer;
    std::atomic_bool m_keepalive_active{false};
    std::atomic<uint64_t> m_keepalive_generation{0};
    std::weak_ptr<network::Connection> m_connection;

    std::shared_ptr<spdlog::logger> m_logger;
    SessionExpiredHandler m_session_expired_handler;
    std::shared_ptr<EpochCoordinator> m_epoch_coordinator;
    std::atomic<uint64_t> m_runtime_state_generation{0};
};

} // namespace protocol
} // namespace nexusminer

#endif // NEXUSMINER_PROTOCOL_SESSION_MANAGER_HPP
