#ifndef NEXUSMINER_NODE_SESSION_HPP
#define NEXUSMINER_NODE_SESSION_HPP

#include "network/connection.hpp"
#include "network/socket.hpp"
#include "network/types.hpp"
#include "protocol/protocol.hpp"
#include "protocol/solo.hpp"
#include "protocol/node_session_context.hpp"
#include "protocol_lane.hpp"
#include "block.hpp"
#include "LLC/types/uint1024.h"
#include "spdlog/spdlog.h"

#include <memory>
#include <string>
#include <functional>
#include <atomic>
#include <chrono>
#include <deque>

namespace asio { class io_context; }

namespace nexusminer {
    class DualConnectionManager;
}

namespace nexusminer {

namespace config { class Config; }
namespace stats { class Collector; }

/**
 * @brief NodeSession — Unified Active Session Outer Wrapper
 *
 * NodeSession is the outer wrapper for one configured mining-node session. It
 * presents a single authenticated identity to Worker_manager on the lane selected
 * by the configured endpoint instead of assuming a paired same-node opposite lane.
 *
 * Design Principles:
 * - Outer Wrapper, Not Protocol Replacement: Wraps the active Solo protocol
 *   path plus compatibility plumbing without replacing existing auth logic.
 * - OPCODE Firewall Preserved: Each Solo instance retains its ProtocolLane.
 *   Packet lane enforcement is never bypassed.
 * - Session ID is Node-Scoped: One Falcon handshake is performed. The resulting
 *   session_id is authoritative for that node and lane.
 * - Simple Surface Area: Worker_manager calls connect(), transmit(),
 *   session_id(), is_authenticated().
 * - Failover Topology Correct: Worker_manager chooses when to switch from
 *   Node A to optional failover Node B; NodeSession itself stays lane-bound.
 */
class NodeSession : public std::enable_shared_from_this<NodeSession>
{
    friend struct NodeSession_recovery_test_access;
public:
    using Config = config::Config;

    /**
     * @brief Connection result callback
     * @param success True if connection and authentication succeeded
     */
    using Connection_callback = std::function<void(bool success)>;

    /**
     * @brief Template feed handler
     * @param block Block data
     * @param nBits Difficulty bits
     */
    using Template_handler = std::function<bool(const ::LLP::CBlock& block, uint32_t nBits)>;

    /**
     * @brief Block accepted handler
     * @param height Block height
     * @param hashPrevBlock Previous block hash
     * @param channel Mining channel
     * @param nonce Nonce value
     */
    using Block_accepted_handler = std::function<void(uint32_t height, uint1024_t hashPrevBlock,
                                                      uint32_t channel, uint64_t nonce)>;

    /**
     * @brief Recovery handler (called when template staleness is detected or
     *        BLOCK_DATA / GET_BLOCK response deserialization fails).
     *
     * The handler MUST return true iff it successfully scheduled / transmitted
     * a fresh GET_BLOCK request.  Solo uses the return value as a chokepoint
     * signal: when true, Solo will NOT also issue its own local recovery
     * GET_BLOCK (prevents the "double-tap" storm).
     */
    using Recovery_handler = std::function<bool(protocol::GetBlockReason)>;

    /**
     * @brief Session expired handler (called when session mismatch is detected)
     */
    using Session_expired_handler = std::function<void()>;

    /**
     * @brief Recovery-confirmed handler
     *
     * Fired when the node re-serves BLOCK_DATA identical to the template already
     * loaded (same unified height and hashPrevBlock) while the same-height feed
     * guard suppresses redistributing it to workers. This is a proof-of-liveness
     * signal for any in-flight recovery: the currently loaded template is still
     * canonical, so Worker_manager may clear recovery immediately rather than
     * waiting for a genuinely new template or escalating toward degraded mode.
     */
    using Recovery_confirmed_handler = std::function<void()>;

    /**
     * @brief Session authenticated handler
     * @param session_id Session ID (0 if authentication failed)
     */
    using Session_authenticated_handler = std::function<void(protocol::SessionId session_id)>;

    /**
     * @brief Session start handler
     * @param keepalive_hours Keepalive interval in hours
     */
    using Session_start_handler = std::function<void(uint16_t keepalive_hours)>;

    /**
     * @brief Work-ready handler
     *
     * Fired after the authoritative session container says outbound GET_BLOCK is allowed again.
     */
    using Work_ready_handler = std::function<void()>;

    /**
     * @brief Node shutdown handler: invoked when NODE_SHUTDOWN (0xD0FF) is received from node.
     * Worker_manager registers this to stop workers gracefully.
     * Reconnect backoff is handled automatically by the Solo protocol layer.
     * @param reason Shutdown reason (GRACEFUL=0x01, MAINTENANCE=0x02)
     */
    using Node_shutdown_handler = std::function<void(uint8_t reason)>;

    /**
     * @brief Replacement-pending handler
     *
     * Fired after a PUSH notification leaves the current template marked
     * replacement-pending (same-height tip update), carrying the deadline by
     * which the promised BLOCK_DATA must arrive. Lets Worker_manager arm a
     * precise one-shot timer instead of relying solely on the slower
     * general-purpose template-health polling cadence.
     * @param deadline steady_clock deadline for the pending replacement
     */
    using Replacement_pending_handler =
        std::function<void(std::chrono::steady_clock::time_point deadline)>;

    /**
     * @brief Constructor
     * @param io_context ASIO io_context for async operations
     * @param config Configuration reference
     * @param socket Network socket for connections
     * @param stats_collector Statistics collector
     * @param node_label Human-readable label for this node (e.g., "PRIMARY", "FAILOVER")
     * @param dcm DualConnectionManager for tracking lane health (optional)
     */
    NodeSession(
        std::shared_ptr<asio::io_context> io_context,
        Config& config,
        network::Socket::Sptr socket,
        std::shared_ptr<stats::Collector> stats_collector,
        const std::string& node_label,
        DualConnectionManager* dcm = nullptr);

    /**
     * @brief Connect to the configured node/lane
     * @param node_endpoint Endpoint selected from config (legacy or stateless)
     * @param callback Invoked only after the session is fully authenticated and
     *        ready for session-bound mining flow
     * @return True if connection initiation succeeded
     */
    bool connect(const network::Endpoint& node_endpoint, Connection_callback callback);

    /**
     * @brief Transmit data on the active connection
     * @param data Data to transmit
     * @return True if transmission was initiated on the configured active lane
     */
    bool transmit(network::Shared_payload data);

    /**
     * @brief Get the session ID
     * @return Session ID (0 if not authenticated)
     */
    protocol::SessionId session_id() const;

    /**
     * @brief Check if authenticated
     * @return True if the configured session lane is authenticated
     */
    bool is_authenticated() const;

    /**
     * @brief Check if the primary TCP connection is currently established
     * @return True if the primary connection is up (may be authenticated or pending auth)
     */
    bool is_primary_connected() const;

    /**
     * @brief Check if the secondary TCP connection is currently established
     * @return True if the secondary connection is up (may be authenticated or pending auth)
     */
    bool is_secondary_connected() const;

    /**
     * @brief Check if session is active
     * @return True if session is active
     */
    bool is_session_active() const;

    /**
     * @brief Stop the node session and close all connections
     */
    void stop();

    /**
     * @brief Reset the node session for reconnection
     *
     * Clears protocol/session state and closes any open lane sockets so the
     * next connect() starts from a clean transport state.
     */
    void reset();

    /**
     * @brief Set template feed handler
     * @param handler Template handler callback
     */
    void set_template_handler(Template_handler handler);
    void set_template_fed_handler(std::function<void()> handler);

    /**
     * @brief Set block accepted handler
     * @param handler Block accepted callback
     */
    void set_block_accepted_handler(Block_accepted_handler handler);

    /**
     * @brief Set recovery initiated handler
     * @param handler Recovery callback
     */
    void set_recovery_initiated_handler(Recovery_handler handler);

    /**
     * @brief Set session expired handler
     * @param handler Session expired callback
     */
    void set_session_expired_handler(Session_expired_handler handler);

    /**
     * @brief Set recovery-confirmed handler
     * @param handler Recovery-confirmed callback
     */
    void set_recovery_confirmed_handler(Recovery_confirmed_handler handler);

    /**
     * @brief Set session authenticated handler
     * @param handler Session authenticated callback
     */
    void set_session_authenticated_handler(Session_authenticated_handler handler);

    /**
     * @brief Set work-ready handler
     * @param handler Work-ready callback
     */
    void set_work_ready_handler(Work_ready_handler handler);

    /**
     * @brief Set session start handler
     * @param handler Session start callback
     */
    void set_session_start_handler(Session_start_handler handler);

    /**
     * @brief Set node shutdown handler
     * @param handler Node shutdown callback
     */
    void set_node_shutdown_handler(Node_shutdown_handler handler);

    /**
     * @brief Set replacement-pending handler
     * @param handler Replacement-pending callback
     */
    void set_replacement_pending_handler(Replacement_pending_handler handler);

    /**
     * @brief Set Falcon miner keys
     * @param pubkey Public key
     * @param privkey Private key
     */
    void set_miner_keys(const std::vector<uint8_t>& pubkey, const std::vector<uint8_t>& privkey);

    /**
     * @brief Set mining reward address
     * @param address NXS address for mining rewards
     */
    void set_reward_address(const std::string& address);

    /**
     * @brief Set Tritium genesis hash
     * @param genesis Genesis hash (32 bytes)
     */
    void set_tritium_genesis(const std::vector<uint8_t>& genesis);

    /**
     * @brief Set keepalive interval
     * @param hours Interval in hours
     */
    void set_keepalive_interval(uint16_t hours);

    /**
     * @brief Get the primary protocol instance (for direct access if needed)
     * @return Shared pointer to primary Solo protocol
     */
    std::shared_ptr<protocol::Solo> get_primary_protocol() const { return m_primary_protocol; }

    /**
     * @brief Get the protocol instance matching the active connection.
     *
     * Mirrors transmit()'s active-lane selection so callers can build payloads
     * with the correct lane framing.
     *
     * @return Protocol instance matching the active connection, or nullptr if none available
     */
    std::shared_ptr<protocol::Solo> get_active_protocol() const;

    /**
     * @brief Get the primary TCP connection (for timer wiring)
     * @return Shared pointer to primary network connection (may be null)
     */
    std::shared_ptr<network::Connection> get_primary_connection() const { return m_primary_connection; }

    /**
     * @brief Get the secondary protocol instance (for direct access if needed)
     * @return Shared pointer to reserved secondary-path Solo protocol (may be null)
     */
    std::shared_ptr<protocol::Solo> get_secondary_protocol() const { return m_secondary_protocol; }

    /**
     * @brief Request fresh mining template
     * @param reason The semantic reason for the request (controls dedup bypass policy)
     * @return Payload to transmit (null if not authenticated)
     */
    network::Shared_payload request_work(protocol::GetBlockReason reason = protocol::GetBlockReason::INITIAL_REQUEST);

    /**
     * @brief Submit a solved block
     * @param block_data Block data
     * @param nonce Nonce value
     * @return Payload to transmit (null if not authenticated)
     */
    network::Shared_payload submit_block(const std::vector<uint8_t>& block_data, uint64_t nonce);

    /**
     * @brief Send GET_ROUND request
     * @return Payload to transmit
     */
    network::Shared_payload send_get_round();

    /**
     * @brief Send session keepalive
     * @return Payload to transmit
     */
    network::Shared_payload send_session_keepalive();

    /**
     * @brief Perform in-band re-authentication on the active connection.
     *
     * Selects the configured primary protocol+connection pairing via
     * select_active_pair(), calls login() on that protocol, and transmits the
     * resulting auth payload on the matching connection.
     *
     * IMPORTANT: Solo::login() may return an empty payload if PacketBuilder::build()
     * fails without invoking the callback.  When this happens login_on_active_connection()
     * invokes login_callback(false) itself so callers are always notified.
     *
     * @param login_callback  Invoked with true when auth payload is queued, false on error
     * @return True if the auth payload was generated and transmitted successfully
     */
    bool login_on_active_connection(std::function<void(bool)> login_callback);

    /**
     * @brief Wire the shared EpochCoordinator to the session manager.
     * Called by Worker_manager after construction, before any connections are made.
     * @param coordinator Shared EpochCoordinator instance
     */
    void set_epoch_coordinator(std::shared_ptr<protocol::EpochCoordinator> coordinator);

private:
    enum class LaneSlot {
        Primary,
        Secondary
    };

    struct LaneDescriptor {
        LaneSlot slot;
        const char* label;
        network::Connection::Sptr* connection;
        std::shared_ptr<protocol::Solo>* protocol;
        std::atomic<bool>* connected;
        ProtocolLane* requested_lane;
    };

    struct SessionResetPlan {
        const char* log_reason;
        const char* lane_down_reason;
        bool fail_pending_connect{true};
        bool close_connections{true};
        bool reset_protocols{true};
        bool end_session{true};
        bool clear_accumulators{true};
        uint64_t transport_generation{0};
    };

    /**
     * @brief Select the active connection+protocol pair used by transmit().
     *
     * Returns {primary_connection, primary_protocol} if the configured lane is
     * up, else {nullptr, nullptr}. All three guards (connection, protocol,
     * connected flag) are checked atomically in one place so that transmit(),
     * get_active_protocol(), and login_on_active_connection() can never diverge.
     *
     * @return Pair of (connection, protocol); both are nullptr when no lane is active.
     */
    std::pair<network::Connection::Sptr, std::shared_ptr<protocol::Solo>> select_active_pair() const;

    LaneDescriptor lane(LaneSlot slot);
    bool is_transport_generation_current(uint64_t transport_generation) const;
    SessionResetPlan make_session_reset_plan(const char* log_reason, const char* lane_down_reason) const;
    void apply_session_reset(const SessionResetPlan& plan);
    ProtocolLane resolve_lane(LaneSlot slot) const;
    std::shared_ptr<protocol::Solo> ensure_protocol(LaneSlot slot);
    void sync_protocol_state(LaneSlot slot);
    void rewire_protocol_handlers();
    void connect_lane(LaneSlot slot, const network::Endpoint& node_endpoint);
    void handle_lane_event(LaneSlot slot, uint64_t transport_generation,
                           network::Result::Code result, network::Shared_payload&& receive_buffer);
    void finalize_lane_connection(LaneSlot slot, uint64_t transport_generation, bool deferred);
    void apply_protocol_handlers(LaneSlot slot);
    void mark_lane_socket_connected(LaneSlot slot);
    void mark_lane_socket_failed(LaneSlot slot);
    void mark_lane_authenticated(LaneSlot slot, protocol::SessionId sid);
    bool begin_lane_authentication(LaneSlot slot, uint64_t transport_generation);
    void complete_pending_connect(bool success);
    void mark_all_lanes_down(const char* reason);

    /**
     * @brief Initialize the configured primary connection lane
     * @param node_endpoint Node endpoint
     */
    void connect_primary(const network::Endpoint& node_endpoint);

    /**
     * @brief Initialize the secondary connection lane
     * @param node_endpoint Explicit secondary-node endpoint (if ever used)
     */
    void connect_secondary(const network::Endpoint& node_endpoint);

    /**
     * @brief Process data received on a lane connection (primary or secondary)
     *
     * Unified RX processing: accumulates bytes, extracts framed packets using
     * the connection's protocol lane, applies one-byte resync on malformed frames,
     * and dispatches to the lane's Solo protocol instance.
     *
     * @param slot  Which lane (Primary or Secondary)
     * @param receive_buffer  Received data
     */
    void process_lane_data(LaneSlot slot, network::Shared_payload&& receive_buffer);

    // Core components
    std::shared_ptr<asio::io_context> m_io_context;
    Config& m_config;
    network::Socket::Sptr m_socket;
    std::shared_ptr<stats::Collector> m_stats_collector;
    std::shared_ptr<spdlog::logger> m_logger;
    std::string m_node_label;

    // Connections
    network::Connection::Sptr m_primary_connection;    // Configured mining lane/session
    network::Connection::Sptr m_secondary_connection;  // Reserved for explicit secondary-node use only

    // Protocol instances
    std::shared_ptr<protocol::Solo> m_primary_protocol;
    std::shared_ptr<protocol::Solo> m_secondary_protocol; // Reserved for explicit secondary-node use only

    // Session management - AUTHORITATIVE source for session state
    std::shared_ptr<protocol::NodeSessionContext> m_session_context;

    // Receive accumulators for TCP stream reassembly
    std::deque<uint8_t> m_primary_rx_accumulator;
    std::deque<uint8_t> m_secondary_rx_accumulator;

    // Handlers
    Template_handler m_template_handler;
    std::function<void()> m_template_fed_handler;
    Block_accepted_handler m_block_accepted_handler;
    Recovery_handler m_recovery_handler;
    Session_expired_handler m_session_expired_handler;
    Recovery_confirmed_handler m_recovery_confirmed_handler;
    Session_authenticated_handler m_session_authenticated_handler;
    Work_ready_handler m_work_ready_handler;
    Session_start_handler m_session_start_handler;
    Node_shutdown_handler m_node_shutdown_handler;
    Replacement_pending_handler m_replacement_pending_handler;
    std::vector<uint8_t> m_miner_pubkey;
    std::vector<uint8_t> m_miner_privkey;
    std::string m_reward_address;
    std::vector<uint8_t> m_tritium_genesis;
    uint16_t m_keepalive_interval_hours{24};
    Connection_callback m_pending_connect_callback;
    ProtocolLane m_primary_requested_lane{ProtocolLane::UNKNOWN};
    ProtocolLane m_secondary_requested_lane{ProtocolLane::UNKNOWN};

    // State flags
    std::atomic<bool> m_primary_connected{false};
    std::atomic<bool> m_secondary_connected{false};
    std::atomic<bool> m_stopped{false};
    std::atomic<uint64_t> m_transport_generation{0};

    // Lane health tracking
    DualConnectionManager* m_dcm{nullptr};

    // Post-accept zero-pad window: after a BLOCK_ACCEPTED/BLOCK_REJECTED parse the
    // node (LLL-TAO commit 206d1a2c) emits a 6-byte framed packet; Fix A routes these
    // through the compat parser so no orphan bytes remain.  This state machine is the
    // belt-and-braces safety net: if a future or older node build ever leaves zero
    // padding in the RX accumulator, the MALFORMED logs are demoted to DEBUG instead
    // of ERROR/WARN, since the bytes are harmless.
    std::chrono::steady_clock::time_point m_last_block_result_parsed_at{};
    static constexpr std::chrono::milliseconds POST_ACCEPT_ZERO_PAD_WINDOW{10};
};

} // namespace nexusminer

#endif // NEXUSMINER_NODE_SESSION_HPP
