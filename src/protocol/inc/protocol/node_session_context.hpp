#ifndef NEXUSMINER_PROTOCOL_NODE_SESSION_CONTEXT_HPP
#define NEXUSMINER_PROTOCOL_NODE_SESSION_CONTEXT_HPP

#include "protocol/session_manager.hpp"
#include "protocol/protocol_constants.hpp"
#include "protocol/session_binding.hpp"
#include "protocol_lane.hpp"
#include <memory>
#include <functional>
#include <vector>
#include <cstdint>

namespace nexusminer {
namespace protocol {

/**
 * @brief NodeSessionContext — Thin façade over SessionManager
 *
 * This utility owns all session-domain decisions and provides a stable interface
 * for session management across NodeSession, Solo, and Worker_manager.
 *
 * Responsibilities:
 * - Authoritative runtime session snapshot access
 * - Session state machine (disconnected/authenticating/authenticated/active/expired)
 * - Lane-aware packet building (keepalive + session status)
 * - Session constants (keepalive cadence rules, retry caps)
 * - Expiry detection & "invalidate session" actions
 * - Wire parsing helpers for session protocols
 *
 * Key Design Principle:
 * NodeSessionContext is the SOLE authoritative source for session_id.
 * Components should query this context rather than maintaining their own copies.
 */
class NodeSessionContext {
public:
    /**
     * @brief Constructor
     * @param session_manager Shared SessionManager instance
     */
    explicit NodeSessionContext(std::shared_ptr<SessionManager> session_manager);

    /**
     * @brief Get the current session ID (authoritative)
     * @return Session ID (0 if no active session)
     */
    SessionId get_session_id() const;

    /**
     * @brief Get the current authoritative session epoch/generation
     * @return Session epoch (0 before the first session is established)
     */
    SessionEpoch get_session_epoch() const;

    /**
     * @brief Check if session is authenticated
     * @return True if session is in AUTHENTICATED or ACTIVE state
     */
    bool is_authenticated() const;

    /**
     * @brief Check if session is active
     * @return True if session is in AUTHENTICATED or ACTIVE state
     */
    bool is_active() const;

    /**
     * @brief Get current session state
     * @return SessionState enum value
     */
    SessionManager::SessionState get_state() const;

    /**
     * @brief Start a new session after successful authentication
     * @param session_id Session ID from MINER_AUTH_RESULT
     * @param session_key Falcon session key (optional)
     * @param tritium_genesis Tritium genesis hash (optional)
     */
    void start_session(SessionId session_id,
                       const std::vector<uint8_t>& session_key = {},
                       SessionGenesisHash tritium_genesis = {});
    void start_session(SessionId session_id,
                       const std::vector<uint8_t>& session_key,
                       const std::vector<uint8_t>& tritium_genesis);

    void commit_authenticated_session(SessionId session_id,
                                      const std::vector<uint8_t>& pubkey,
                                      FalconHashKeyId key_id,
                                      SessionGenesisHash tritium_genesis = {});
    void commit_authenticated_session(SessionId session_id,
                                      const std::vector<uint8_t>& pubkey,
                                      const std::string& key_id,
                                      const std::vector<uint8_t>& tritium_genesis);

    void begin_auth_handshake(const std::string& detail = "");

    void begin_reward_binding(const std::string& reward_address,
                              RewardHash reward_hash = {},
                              const std::string& source = "");
    void begin_reward_binding(const std::string& reward_address,
                              const std::vector<uint8_t>& reward_hash,
                              const std::string& source = "");

    void commit_reward_bound(const std::string& reward_address,
                             RewardHash reward_hash,
                             const std::string& source = "");
    void commit_reward_bound(const std::string& reward_address,
                             const std::vector<uint8_t>& reward_hash,
                             const std::string& source = "");

    void commit_reward_rejected(const std::string& reward_address,
                                const std::string& source = "",
                                const std::string& reason = "");

    void note_keepalive_ack(bool accepted, const std::string& detail = "");

    void mark_recovery_required(const std::string& reason);

    void mark_recovery_healthy(const std::string& reason = "");

    void mark_session_expired(const std::string& reason);

    void clear_for_disconnect(const std::string& reward_address = {},
                              const std::string& reward_source = "",
                              const std::string& reason = "",
                              bool preserve_genesis = true);

    void clear_for_reauth(const std::string& reward_address = {},
                          const std::string& reward_source = "",
                          const std::string& reason = "",
                          bool preserve_genesis = true);

    /**
     * @brief End current session
     */
    void end_session();

    /**
     * @brief Update session state
     * @param state New session state
     */
    void set_state(SessionManager::SessionState state);

    /**
     * @brief Build SESSION_KEEPALIVE packet
     * @return Shared payload for transmission
     */
    network::Shared_payload build_keepalive_packet() const;

    /**
     * @brief Build SESSION_STATUS packet
     * @param degraded True if workers are in degraded mode
     * @param has_template True if valid mining template is held
     * @param workers_running True if mining workers are active
     * @param secondary_up Compatibility status bit for any optional secondary path
     * @return Shared payload for transmission
     */
    network::Shared_payload build_session_status_packet(
        bool degraded,
        bool has_template,
        bool workers_running,
        bool secondary_up) const;

    /**
     * @brief Start keepalive timer
     */
    void start_keepalive_timer();

    /**
     * @brief Stop keepalive timer
     */
    void stop_keepalive_timer();

    /**
     * @brief Set keepalive interval
     * @param hours Interval in hours (1-168)
     */
    void set_keepalive_interval(uint16_t hours);

    /**
     * @brief Get keepalive interval
     * @return Interval in hours
     */
    uint16_t get_keepalive_interval() const;

    /**
     * @brief Set protocol lane for packet generation
     * @param lane Protocol lane (LEGACY or STATELESS)
     */
    void set_protocol_lane(ProtocolLane lane);

    /**
     * @brief Set connection for keepalive traffic
     * @param connection Network connection
     */
    void set_connection(std::shared_ptr<network::Connection> connection);

    /**
     * @brief Set Tritium genesis hash
     * @param genesis Genesis hash (32 bytes)
     */
    void set_tritium_genesis(SessionGenesisHash genesis);
    void set_tritium_genesis(const std::vector<uint8_t>& genesis);

    void set_connection_metadata(const std::string& local_endpoint,
                                 const std::string& remote_endpoint,
                                 bool connected);

    void set_falcon_identity(const std::vector<uint8_t>& pubkey,
                             FalconHashKeyId key_id,
                             bool authenticated);
    void set_falcon_identity(const std::vector<uint8_t>& pubkey,
                             const std::string& key_id,
                             bool authenticated);

    void reset_session_credentials();

    void set_chacha20_session_key(const std::vector<uint8_t>& session_key,
                                  SessionFingerprint fingerprint,
                                  bool ready);
    void set_chacha20_session_key(const std::vector<uint8_t>& session_key,
                                  const std::string& fingerprint,
                                  bool ready);

    void set_reward_binding(const std::string& reward_address,
                            RewardHash reward_hash,
                            bool bound,
                            const std::string& source);
    void set_reward_binding(const std::string& reward_address,
                            const std::vector<uint8_t>& reward_hash,
                            bool bound,
                            const std::string& source);

    void set_channel_state(uint32_t channel,
                           bool ready_for_submit,
                           bool ready_for_get_block);

    void mark_activity();

    bool is_reward_bound() const;

    bool reward_binding_required() const;

    SessionManager::RewardBindReadiness get_reward_bind_readiness() const;

    bool can_submit_work() const;

    bool can_request_get_block() const;

    bool allow_deferred_push_replay() const;

    bool allow_get_block_replay() const;

    bool validate_miner_session(std::string* reason = nullptr) const;

    std::string build_miner_session_diagnostics() const;

    void record_session_event(SessionManager::SessionEventKind kind,
                              const std::string& detail = "");

    std::vector<SessionManager::SessionEvent> get_session_event_journal() const;

    std::string build_session_event_journal() const;

    /**
     * @brief Get Tritium genesis hash
     * @return Genesis hash vector
     */
    SessionGenesisHash get_typed_tritium_genesis() const;
    std::vector<uint8_t> get_tritium_genesis() const;

    /**
     * @brief Get session key
     * @return Session key vector
     */
    std::vector<uint8_t> get_session_key() const;

    /**
     * @brief Get session uptime
     * @return Duration since session start
     */
    std::chrono::seconds get_session_uptime() const;

    /**
     * @brief Get session information
     * @return Copy of the authoritative runtime session snapshot
     */
    SessionManager::RuntimeSessionSnapshot get_runtime_snapshot() const;

    /**
     * @brief Backward-compatible alias for get_runtime_snapshot()
     * @return SessionInfo structure with current session data
     */
    SessionManager::SessionInfo get_session_info() const;

    /**
     * @brief Set session expired handler
     * @param handler Callback invoked when session expires
     */
    void set_session_expired_handler(SessionManager::SessionExpiredHandler handler);

    /**
     * @brief Set prevblock suffix for keepalive packets
     * @param suffix Last 4 bytes of hashPrevBlock
     */
    void set_prevblock_suffix(const std::array<uint8_t, 4>& suffix);

    /**
     * @brief Get the canonical session identity bundle.
     *
     * Returns the frozen SessionIdentity constructed at authentication time.
     * This bundle binds session_id, epoch, crypto context, and Falcon identity
     * into a single immutable snapshot.  Returns an empty (invalid) identity
     * if no authenticated session exists.
     *
     * @return SessionIdentity (value copy, safe for cross-thread use)
     */
    SessionIdentity get_canonical_identity() const;
    SessionBinding get_session_binding() const;

    /**
     * @brief Get the underlying SessionManager (for advanced use cases)
     * @return Shared pointer to SessionManager
     */
    std::shared_ptr<SessionManager> get_session_manager() const { return m_session_manager; }

    // Session constants accessors (from ProtocolConstants)

    /**
     * @brief Get keepalive safety divisor
     * @return Safety divisor (number of keepalives per session window)
     */
    static constexpr uint32_t get_keepalive_safety_divisor() {
        return ProtocolConstants::KEEPALIVE_SAFETY_DIVISOR;
    }

    /**
     * @brief Get maximum session authentication retries
     * @return Maximum retry count
     */
    static constexpr uint32_t get_max_session_auth_retries() {
        return ProtocolConstants::MAX_SESSION_AUTH_RETRIES;
    }

    /**
     * @brief Get base session retry delay
     * @return Base delay in milliseconds
     */
    static constexpr uint32_t get_base_session_retry_ms() {
        return ProtocolConstants::BASE_SESSION_RETRY_MS;
    }

    /**
     * @brief Get maximum session retry delay
     * @return Maximum delay in milliseconds
     */
    static constexpr uint32_t get_max_session_retry_ms() {
        return ProtocolConstants::MAX_SESSION_RETRY_MS;
    }

    // Wire parsing helpers

    /**
     * @brief Parse SESSION_START packet
     * @param packet_data Packet payload
     * @param out_session_id Parsed session ID (output)
     * @param out_timeout Parsed timeout in seconds (output)
     * @param out_genesis Optional genesis hash (output)
     * @return True if parsing succeeded
     */
    static bool parse_session_start(
        const std::vector<uint8_t>& packet_data,
        uint32_t& out_session_id,
        uint32_t& out_timeout,
        std::vector<uint8_t>& out_genesis);

private:
    std::shared_ptr<SessionManager> m_session_manager;
};

} // namespace protocol
} // namespace nexusminer

#endif // NEXUSMINER_PROTOCOL_NODE_SESSION_CONTEXT_HPP
