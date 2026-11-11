#include "node_session/node_session.hpp"
#include "protocol/solo.hpp"
#include "protocol/node_session_context.hpp"
#include "config/config.hpp"
#include "packet.hpp"
#include "miner_opcodes.hpp"
#include "dual_connection_manager.hpp"
#include "asio/post.hpp"
#include <algorithm>

namespace nexusminer {

namespace {

const char* lane_name(ProtocolLane lane)
{
    switch (lane) {
    case ProtocolLane::STATELESS:
        return "STATELESS";
    case ProtocolLane::LEGACY:
        return "LEGACY";
    default:
        return "UNKNOWN";
    }
}

std::string format_prefix_hex(const std::deque<uint8_t>& bytes, std::size_t max_bytes = 8)
{
    static constexpr char hex[] = "0123456789abcdef";
    std::string out;
    const std::size_t count = std::min(bytes.size(), max_bytes);
    out.reserve(count * 3);
    for (std::size_t i = 0; i < count; ++i) {
        if (i != 0) {
            out.push_back(' ');
        }
        const auto b = bytes[i];
        out.push_back(hex[(b >> 4) & 0x0F]);
        out.push_back(hex[b & 0x0F]);
    }
    return out;
}

bool is_submit_result_packet(const Packet& packet)
{
    if (packet.m_length > 1) {
        return false;
    }
    if (packet.m_is_uint16_opcode) {
        return packet.m_header == LLP::StatelessMining::BLOCK_ACCEPTED ||
               packet.m_header == LLP::StatelessMining::BLOCK_REJECTED ||
               packet.m_header == LLP::StatelessMining::BLOCK_ACCEPTED_COMPAT ||
               packet.m_header == LLP::StatelessMining::BLOCK_REJECTED_COMPAT;
    }

    return packet.m_header == LLP::BLOCK_ACCEPTED ||
           packet.m_header == LLP::BLOCK_REJECTED;
}

} // namespace

NodeSession::NodeSession(
    std::shared_ptr<asio::io_context> io_context,
    Config& config,
    network::Socket::Sptr socket,
    std::shared_ptr<stats::Collector> stats_collector,
    const std::string& node_label,
    DualConnectionManager* dcm)
    : m_io_context(std::move(io_context))
    , m_config(config)
    , m_socket(std::move(socket))
    , m_stats_collector(std::move(stats_collector))
    , m_node_label(node_label)
    , m_dcm(dcm)
{
    m_logger = spdlog::get("logger");
    if (!m_logger) {
        m_logger = spdlog::default_logger();
    }

    m_logger->info("[NodeSession:{}] Created", m_node_label);

    // Create session manager (shared across both protocols)
    auto session_manager = std::make_shared<protocol::SessionManager>(24, m_io_context);

    // Wrap session manager in NodeSessionContext (AUTHORITATIVE session state)
    m_session_context = std::make_shared<protocol::NodeSessionContext>(session_manager);

    uint8_t channel = (m_config.get_mining_mode() == config::Mining_mode::PRIME) ? 1U : 2U;
    ensure_protocol(LaneSlot::Primary);

    m_logger->info("[NodeSession:{}] Initialized with channel {}", m_node_label, channel);
}

bool NodeSession::connect(const network::Endpoint& node_endpoint, Connection_callback callback)
{
    if (m_stopped.load(std::memory_order_acquire)) {
        m_logger->warn("[NodeSession:{}] Cannot connect - session is stopped", m_node_label);
        return false;
    }

    m_transport_generation.fetch_add(1, std::memory_order_acq_rel);
    m_logger->info("[NodeSession:{}] Connecting to node at {}",
                   m_node_label, node_endpoint.to_string());

    m_pending_connect_callback = std::move(callback);

    // Start with the configured primary lane.
    connect_primary(node_endpoint);

    return true;
}

void NodeSession::connect_primary(const network::Endpoint& node_endpoint)
{
    m_primary_requested_lane = determine_lane_from_port(node_endpoint.port());
    connect_lane(LaneSlot::Primary, node_endpoint);
}

void NodeSession::connect_secondary(const network::Endpoint& node_endpoint)
{
    if (m_secondary_connected.load(std::memory_order_acquire) ||
        m_stopped.load(std::memory_order_acquire)) {
        return;
    }

    m_secondary_requested_lane = determine_lane_from_port(node_endpoint.port());
    connect_lane(LaneSlot::Secondary, node_endpoint);
}

NodeSession::LaneDescriptor NodeSession::lane(LaneSlot slot)
{
    if (slot == LaneSlot::Primary) {
        return LaneDescriptor{
            LaneSlot::Primary,
            "Primary",
            &m_primary_connection,
            &m_primary_protocol,
            &m_primary_connected,
            &m_primary_requested_lane
        };
    }

    return LaneDescriptor{
        LaneSlot::Secondary,
        "Secondary",
        &m_secondary_connection,
        &m_secondary_protocol,
        &m_secondary_connected,
        &m_secondary_requested_lane
    };
}

ProtocolLane NodeSession::resolve_lane(LaneSlot slot) const
{
    switch (slot) {
    case LaneSlot::Primary:
        return m_primary_connection ? m_primary_connection->get_protocol_lane() : m_primary_requested_lane;
    case LaneSlot::Secondary:
        return m_secondary_connection ? m_secondary_connection->get_protocol_lane() : m_secondary_requested_lane;
    }

    return ProtocolLane::UNKNOWN;
}

std::shared_ptr<protocol::Solo> NodeSession::ensure_protocol(LaneSlot slot)
{
    auto descriptor = lane(slot);
    if (!*descriptor.protocol) {
        uint8_t channel = (m_config.get_mining_mode() == config::Mining_mode::PRIME) ? 1U : 2U;
        *descriptor.protocol = std::make_shared<protocol::Solo>(
            channel, m_stats_collector, m_session_context, m_io_context);
    }

    sync_protocol_state(slot);
    return *descriptor.protocol;
}

void NodeSession::sync_protocol_state(LaneSlot slot)
{
    auto descriptor = lane(slot);
    auto protocol = *descriptor.protocol;
    if (!protocol) {
        return;
    }

    if (!m_miner_pubkey.empty() && !m_miner_privkey.empty()) {
        protocol->set_miner_keys(m_miner_pubkey, m_miner_privkey);
    }
    if (!m_reward_address.empty()) {
        protocol->set_reward_address(m_reward_address);
    }
    if (!m_tritium_genesis.empty()) {
        protocol->set_tritium_genesis(m_tritium_genesis);
    }
    protocol->set_keepalive_interval(m_keepalive_interval_hours);
    protocol->enable_chacha20_wrapping(true);
    protocol->enable_disposable_falcon(true);
    apply_protocol_handlers(slot);
}

void NodeSession::rewire_protocol_handlers()
{
    apply_protocol_handlers(LaneSlot::Primary);
    apply_protocol_handlers(LaneSlot::Secondary);
}

void NodeSession::connect_lane(LaneSlot slot, const network::Endpoint& node_endpoint)
{
    auto descriptor = lane(slot);
    ensure_protocol(slot);

    m_logger->info("[NodeSession:{}] Connecting {} ({}) to {}",
                   m_node_label,
                   descriptor.label,
                   lane_name(*descriptor.requested_lane),
                   node_endpoint.to_string());

    struct ConnectObservation {
        bool saw_initial_result{false};
        network::Result::Code result{network::Result::connection_ok};
    };

    auto observation = std::make_shared<ConnectObservation>();
    const auto transport_generation = m_transport_generation.load(std::memory_order_acquire);
    auto weak_self = std::weak_ptr<NodeSession>(shared_from_this());
    auto connection_callback = [weak_self, slot, transport_generation](auto result, auto receive_buffer) {
        auto self = weak_self.lock();
        if (!self) return;
        self->handle_lane_event(slot, transport_generation, result, std::move(receive_buffer));
    };

    auto observed_callback = [connection_callback, observation](auto result, auto receive_buffer) mutable {
        if (network::Result::category(result) == network::Result::Category::connection) {
            observation->saw_initial_result = true;
            observation->result = result;
        }
        connection_callback(result, std::move(receive_buffer));
    };

    auto connection = m_socket->connect(node_endpoint, observed_callback);
    if (!connection) {
        m_logger->error("[NodeSession:{}] {} connection setup returned null connection",
                        m_node_label, descriptor.label);
        mark_lane_socket_failed(slot);
        return;
    }

    if (observation->saw_initial_result &&
        observation->result != network::Result::connection_ok) {
        return;
    }

    *descriptor.connection = std::move(connection);
    if (auto protocol = *descriptor.protocol) {
        protocol->set_connection(*descriptor.connection);
    }
}

bool NodeSession::is_transport_generation_current(uint64_t transport_generation) const
{
    return transport_generation == m_transport_generation.load(std::memory_order_acquire);
}

NodeSession::SessionResetPlan NodeSession::make_session_reset_plan(const char* log_reason,
                                                                   const char* lane_down_reason) const
{
    SessionResetPlan plan;
    plan.log_reason = log_reason;
    plan.lane_down_reason = lane_down_reason;
    plan.transport_generation =
        m_transport_generation.load(std::memory_order_acquire) + 1;
    return plan;
}

void NodeSession::apply_session_reset(const SessionResetPlan& plan)
{
    m_transport_generation.store(plan.transport_generation, std::memory_order_release);

    if (plan.fail_pending_connect) {
        complete_pending_connect(false);
    }

    m_logger->info("[NodeSession:{}] {}", m_node_label, plan.log_reason);

    if (plan.close_connections) {
        if (m_primary_connection) {
            m_primary_connection->close();
            m_primary_connection.reset();
        }
        if (m_secondary_connection) {
            m_secondary_connection->close();
            m_secondary_connection.reset();
        }
    }

    if (plan.reset_protocols) {
        if (m_primary_protocol) {
            m_primary_protocol->reset();
        }
        if (m_secondary_protocol) {
            m_secondary_protocol->reset();
        }
    }

    mark_all_lanes_down(plan.lane_down_reason);

    if (plan.end_session && m_session_context) {
        m_session_context->end_session();
    }

    if (plan.clear_accumulators) {
        m_primary_rx_accumulator.clear();
        m_secondary_rx_accumulator.clear();
    }
}

void NodeSession::handle_lane_event(LaneSlot slot, uint64_t transport_generation, network::Result::Code result,
                                    network::Shared_payload&& receive_buffer)
{
    if (m_stopped.load(std::memory_order_acquire) ||
        !is_transport_generation_current(transport_generation)) {
        return;
    }

    auto descriptor = lane(slot);

    if (result == network::Result::connection_ok) {
        if (!*descriptor.connection) {
            ::asio::post(*m_io_context, [weak_self = std::weak_ptr<NodeSession>(shared_from_this()),
                                         slot, transport_generation]() {
                auto self = weak_self.lock();
                if (!self) return;
                self->finalize_lane_connection(slot, transport_generation, true);
            });
            return;
        }

        finalize_lane_connection(slot, transport_generation, false);
        return;
    }

    if (result == network::Result::connection_declined ||
        result == network::Result::connection_aborted ||
        result == network::Result::connection_closed ||
        result == network::Result::connection_error) {
        if (slot == LaneSlot::Primary) {
            m_logger->error("[NodeSession:{}] {} connection failed: {}",
                            m_node_label, descriptor.label, static_cast<int>(result));
        } else {
            m_logger->warn("[NodeSession:{}] {} connection failed: {}",
                           m_node_label, descriptor.label, static_cast<int>(result));
        }
        mark_lane_socket_failed(slot);
        return;
    }

    process_lane_data(slot, std::move(receive_buffer));
}

void NodeSession::finalize_lane_connection(LaneSlot slot, uint64_t transport_generation, bool deferred)
{
    if (m_stopped.load(std::memory_order_acquire) ||
        !is_transport_generation_current(transport_generation)) {
        return;
    }

    auto descriptor = lane(slot);
    if (!*descriptor.connection || !*descriptor.protocol) {
        return;
    }
    m_logger->info("[NodeSession:{}] {} connection established{}",
                   m_node_label,
                   descriptor.label,
                   deferred ? " (deferred)" : "");
    mark_lane_socket_connected(slot);
    begin_lane_authentication(slot, transport_generation);
}

void NodeSession::complete_pending_connect(bool success)
{
    if (!m_pending_connect_callback) {
        return;
    }

    auto callback = std::move(m_pending_connect_callback);
    m_pending_connect_callback = {};
    callback(success);
}

void NodeSession::mark_lane_socket_connected(LaneSlot slot)
{
    auto descriptor = lane(slot);
    descriptor.connected->store(true);

    auto lane_kind = resolve_lane(slot);
    if (*descriptor.protocol) {
        (*descriptor.protocol)->set_protocol_lane(lane_kind);
    }

    m_logger->info("[NodeSession:{}] {} lane: {}", m_node_label, descriptor.label, lane_name(lane_kind));

    if (!m_dcm || lane_kind == ProtocolLane::UNKNOWN) {
        return;
    }

    if (lane_kind == ProtocolLane::STATELESS) {
        m_dcm->set_stateless_alive(true);
    } else {
        m_dcm->set_legacy_alive(true);
    }

    if (slot == LaneSlot::Primary && m_dcm->mining_lane() == ProtocolLane::UNKNOWN) {
        m_dcm->set_mining_lane(lane_kind);
        m_logger->info("[NodeSession:{}] Mining lane stamped: {} (immutable for session lifetime)",
                       m_node_label, lane_name(lane_kind));
    }

    m_logger->info("[NodeSession:{}] {} socket connected → DualConnectionManager updated",
                   m_node_label, descriptor.label);
}

void NodeSession::mark_lane_socket_failed(LaneSlot slot)
{
    auto descriptor = lane(slot);
    const bool was_connected = descriptor.connected->exchange(false);
    if (slot == LaneSlot::Primary) {
        m_primary_rx_accumulator.clear();
    } else {
        m_secondary_rx_accumulator.clear();
    }

    auto lane_kind = resolve_lane(slot);
    if (m_dcm && lane_kind != ProtocolLane::UNKNOWN) {
        m_dcm->on_lane_failed(lane_kind);
        m_logger->warn("[NodeSession:{}] {} lane ({}) failed → DualConnectionManager updated",
                       m_node_label, descriptor.label, lane_name(lane_kind));
    }

    if (slot != LaneSlot::Primary) {
        return;
    }

    // A connect() attempt still owns the failure. Completing that callback is
    // what schedules the first retry; firing transport-lost as well would
    // reconnect twice.
    if (m_pending_connect_callback) {
        complete_pending_connect(false);
        return;
    }

    if (!was_connected) {
        return;
    }

    // The socket is gone. Leave the session unauthenticated immediately so
    // submits and GET_BLOCK cannot treat a stale push timestamp as liveness.
    if (m_session_context) {
        m_session_context->end_session();
    }

    if (m_transport_lost_handler && m_io_context) {
        auto handler = m_transport_lost_handler;
        ::asio::post(*m_io_context, [handler = std::move(handler)]() {
            if (handler) {
                handler();
            }
        });
    }
}

void NodeSession::mark_lane_authenticated(LaneSlot slot, protocol::SessionId sid)
{
    if (sid.is_default()) {
        if (slot == LaneSlot::Primary) {
            complete_pending_connect(false);
        }
        return;
    }

    auto descriptor = lane(slot);
    auto lane_kind = resolve_lane(slot);

    if (m_dcm && lane_kind != ProtocolLane::UNKNOWN) {
        if (lane_kind == ProtocolLane::STATELESS) {
            m_dcm->set_stateless_alive(true);
        } else {
            m_dcm->set_legacy_alive(true);
        }
        m_logger->info("[NodeSession:{}] {} lane ({}) authenticated → DualConnectionManager updated",
                       m_node_label, descriptor.label, lane_name(lane_kind));
    }

    if (slot == LaneSlot::Primary) {
        complete_pending_connect(true);
    }
}

bool NodeSession::begin_lane_authentication(LaneSlot slot, uint64_t transport_generation)
{
    if (m_stopped.load(std::memory_order_acquire) ||
        !is_transport_generation_current(transport_generation)) {
        return false;
    }

    auto descriptor = lane(slot);
    sync_protocol_state(slot);
    auto connection = *descriptor.connection;
    auto protocol = *descriptor.protocol;
    if (!connection || !protocol) {
        m_logger->error("[NodeSession:{}] {} authentication cannot start - lane not fully initialized",
                        m_node_label, descriptor.label);
        if (slot == LaneSlot::Primary) {
            complete_pending_connect(false);
        }
        return false;
    }

    auto weak_self = std::weak_ptr<NodeSession>(shared_from_this());
    auto auth_payload = protocol->login([weak_self, slot, transport_generation](bool login_result) {
        auto self = weak_self.lock();
        if (!self) return;
        if (self->m_stopped.load(std::memory_order_acquire) ||
            !self->is_transport_generation_current(transport_generation)) {
            return;
        }

        auto descriptor = self->lane(slot);
        if (!login_result) {
            self->m_logger->error("[NodeSession:{}] {} authentication failed",
                                  self->m_node_label, descriptor.label);
            if (slot == LaneSlot::Primary) {
                self->complete_pending_connect(false);
            }
            return;
        }
    });

    if (!auth_payload || auth_payload->empty()) {
        m_logger->error("[NodeSession:{}] {} authentication payload generation failed",
                        m_node_label, descriptor.label);
        if (slot == LaneSlot::Primary) {
            complete_pending_connect(false);
        }
        return false;
    }

    if (!connection->transmit(auth_payload)) {
        m_logger->error("[NodeSession:{}] {} failed to queue authentication payload — "
                        "connection may be closed or TX queue full",
                        m_node_label, descriptor.label);
        if (slot == LaneSlot::Primary) {
            complete_pending_connect(false);
        }
        return false;
    }

    m_logger->info("[NodeSession:{}] {} authentication initiated", m_node_label, descriptor.label);

    return true;
}

void NodeSession::apply_protocol_handlers(LaneSlot slot)
{
    auto descriptor = lane(slot);
    auto protocol = *descriptor.protocol;
    if (!protocol) {
        return;
    }

    protocol->set_block_handler([this](const ::LLP::CBlock& block, uint32_t nBits) -> bool {
        if (m_template_handler) {
            return m_template_handler(block, nBits);
        }
        return false;
    });

    protocol->get_template_interface()->set_template_fed_handler([this]() {
        if (m_template_fed_handler) {
            m_template_fed_handler();
        }
    });
    protocol->get_template_interface()->set_template_feed_failed_handler([this]() {
        if (m_template_feed_failed_handler) {
            m_template_feed_failed_handler();
        }
    });

    protocol->set_block_accepted_handler(
        [this](uint32_t height, uint1024_t hash_prev, uint32_t channel, uint64_t nonce) {
            if (m_block_accepted_handler) {
                m_block_accepted_handler(height, hash_prev, channel, nonce);
            }
        });

    protocol->set_recovery_initiated_handler([this](protocol::GetBlockReason reason) -> bool {
        if (m_recovery_handler) {
            return m_recovery_handler(reason);
        }
        return false;
    });

    protocol->set_session_expired_handler([this]() {
        if (m_session_expired_handler) {
            m_session_expired_handler();
        }
    });

    protocol->set_recovery_confirmed_handler([this]() {
        if (m_recovery_confirmed_handler) {
            m_recovery_confirmed_handler();
        }
    });

    protocol->set_session_authenticated_handler([this, slot](protocol::SessionId sid) {
        mark_lane_authenticated(slot, sid);
        if (m_session_authenticated_handler) {
            m_session_authenticated_handler(sid);
        }
    });

    protocol->set_work_ready_handler([this]() {
        if (m_work_ready_handler) {
            m_work_ready_handler();
        }
    });

    protocol->set_session_start_handler([this](uint16_t hours) {
        if (m_session_start_handler) {
            m_session_start_handler(hours);
        }
    });

    protocol->set_node_shutdown_handler([this](uint8_t reason) {
        if (m_node_shutdown_handler) {
            m_node_shutdown_handler(reason);
        }
    });

    protocol->set_replacement_pending_handler([this](std::chrono::steady_clock::time_point deadline) {
        if (m_replacement_pending_handler) {
            m_replacement_pending_handler(deadline);
        }
    });
}

void NodeSession::mark_all_lanes_down(const char* reason)
{
    m_primary_connected = false;
    m_secondary_connected = false;

    if (!m_dcm) {
        return;
    }

    m_dcm->set_stateless_alive(false);
    m_dcm->set_legacy_alive(false);
    m_logger->info("[NodeSession:{}] {} → DualConnectionManager lanes marked down",
                   m_node_label, reason);
}

void NodeSession::process_lane_data(LaneSlot slot, network::Shared_payload&& receive_buffer)
{
    auto descriptor = lane(slot);
    auto& connection = *descriptor.connection;
    auto& protocol   = *descriptor.protocol;
    auto& accumulator = (slot == LaneSlot::Primary)
                            ? m_primary_rx_accumulator
                            : m_secondary_rx_accumulator;

    if (m_stopped.load(std::memory_order_acquire) || !connection || !protocol || !receive_buffer) {
        return;
    }

    accumulator.insert(accumulator.end(), receive_buffer->begin(), receive_buffer->end());

    const ProtocolLane lane_kind = connection->get_protocol_lane();

    while (!accumulator.empty()) {
        auto buffer_shared = std::make_shared<std::vector<uint8_t>>(
            accumulator.begin(), accumulator.end());

        size_t bytes_consumed = 0;
        ParseResult parse_result;

        auto packet = extract_packet_from_buffer_with_result(
            buffer_shared, bytes_consumed, 0, lane_kind, parse_result);

        if (parse_result == ParseResult::NEED_MORE_DATA) {
            break;
        }
        if (parse_result == ParseResult::MALFORMED) {
            if (looks_like_cross_lane_frame_prefix(buffer_shared, 0, lane_kind)) {
                m_logger->error("[NodeSession:{}] Cross-lane packet prefix on {} connection "
                                "(lane={}, prefix=[{}]) — closing instead of byte-drop resync",
                                m_node_label, descriptor.label, lane_name(lane_kind),
                                format_prefix_hex(accumulator));
                accumulator.clear();
                if (connection) {
                    connection->close();
                }
                return;
            }

            const bool is_zero_pad_byte = !accumulator.empty() && accumulator.front() == 0x00;
            const auto now = std::chrono::steady_clock::now();
            const bool in_post_accept_window =
                (m_last_block_result_parsed_at != std::chrono::steady_clock::time_point{}) &&
                ((now - m_last_block_result_parsed_at) <= POST_ACCEPT_ZERO_PAD_WINDOW);
            const bool benign_zero_pad = is_zero_pad_byte && in_post_accept_window;

            if (benign_zero_pad) {
                m_logger->debug("[NodeSession:{}] Malformed byte on {} (reason=post_accept_zero_pad) — absorbing",
                                m_node_label, descriptor.label);
            } else {
                m_logger->error("[NodeSession:{}] Malformed packet on {} connection",
                                m_node_label, descriptor.label);
                m_logger->error("[NodeSession:{}] Closing {} connection instead of byte-drop resync "
                                "(lane={}, prefix=[{}])",
                                m_node_label, descriptor.label, lane_name(lane_kind),
                                format_prefix_hex(accumulator));
                accumulator.clear();
                if (connection) {
                    connection->close();
                }
                return;
            }
            if (!accumulator.empty()) {
                accumulator.pop_front();
                if (benign_zero_pad) {
                    m_logger->debug("[NodeSession:{}] Absorbed 1 zero-pad byte post-accept on {} ({} bytes remain)",
                                    m_node_label, descriptor.label, accumulator.size());
                } else {
                    m_logger->warn("[NodeSession:{}] Dropped 1 byte from {} RX accumulator for resync "
                                   "({} bytes remain)",
                                   m_node_label, descriptor.label, accumulator.size());
                }
                continue;
            }
            break;
        }

        accumulator.erase(accumulator.begin(), accumulator.begin() + bytes_consumed);
        // Stamp the post-accept window for both lanes so residual zero-pad bytes
        // from old stateless or legacy submit-result writers are absorbed at
        // DEBUG level rather than tearing down an otherwise healthy session.
        if (is_submit_result_packet(packet)) {
            m_last_block_result_parsed_at = std::chrono::steady_clock::now();
        }
        protocol->process_messages(packet, connection);
    }
}

std::pair<network::Connection::Sptr, std::shared_ptr<protocol::Solo>>
NodeSession::select_active_pair() const
{
    // Single authoritative source for active-lane selection.
    // NodeSession is now a one-configured-lane session wrapper, so reconnect and
    // re-auth stay on the configured primary lane instead of falling through to
    // any dormant secondary-node plumbing.
    if (m_primary_connection && m_primary_protocol && m_primary_connected) {
        return {m_primary_connection, m_primary_protocol};
    }
    return {nullptr, nullptr};
}

bool NodeSession::transmit(network::Shared_payload data)
{
    if (m_stopped.load(std::memory_order_acquire) || !data || data->empty()) {
        return false;
    }

    auto [conn, proto] = select_active_pair();
    if (conn) {
        return conn->transmit(data);
    }

    m_logger->warn("[NodeSession:{}] No active connection to transmit on", m_node_label);
    return false;
}

std::shared_ptr<protocol::Solo> NodeSession::get_active_protocol() const
{
    return select_active_pair().second;
}

protocol::SessionId NodeSession::session_id() const
{
    // Query the authoritative session context
    return m_session_context ? m_session_context->get_session_id() : protocol::SessionId{};
}

bool NodeSession::is_authenticated() const
{
    // Query the authoritative session context
    return m_session_context ? m_session_context->is_authenticated() : false;
}

bool NodeSession::is_primary_connected() const
{
    return m_primary_connected.load();
}

bool NodeSession::is_secondary_connected() const
{
    return m_secondary_connected.load();
}

bool NodeSession::is_session_active() const
{
    // Query the authoritative session context
    return m_session_context ? m_session_context->is_active() : false;
}

void NodeSession::stop()
{
    if (m_stopped.exchange(true, std::memory_order_acq_rel)) {
        return;
    }

    auto plan = make_session_reset_plan("Stopping", "Session stopped");
    apply_session_reset(plan);
}

void NodeSession::reset()
{
    if (m_stopped.load(std::memory_order_acquire)) {
        m_logger->warn("[NodeSession:{}] Reset ignored - session is stopped", m_node_label);
        return;
    }

    auto plan = make_session_reset_plan("Resetting", "Session reset");
    apply_session_reset(plan);
}

void NodeSession::set_template_handler(Template_handler handler)
{
    m_template_handler = std::move(handler);
    rewire_protocol_handlers();
}

void NodeSession::set_template_fed_handler(std::function<void()> handler)
{
    m_template_fed_handler = std::move(handler);
    rewire_protocol_handlers();
}

void NodeSession::set_template_feed_failed_handler(std::function<void()> handler)
{
    m_template_feed_failed_handler = std::move(handler);
    rewire_protocol_handlers();
}

void NodeSession::set_block_accepted_handler(Block_accepted_handler handler)
{
    m_block_accepted_handler = std::move(handler);
    rewire_protocol_handlers();
}

void NodeSession::set_recovery_initiated_handler(Recovery_handler handler)
{
    m_recovery_handler = std::move(handler);
    rewire_protocol_handlers();
}

void NodeSession::set_session_expired_handler(Session_expired_handler handler)
{
    m_session_expired_handler = std::move(handler);
    rewire_protocol_handlers();
}

void NodeSession::set_recovery_confirmed_handler(Recovery_confirmed_handler handler)
{
    m_recovery_confirmed_handler = std::move(handler);
    rewire_protocol_handlers();
}

void NodeSession::set_session_authenticated_handler(Session_authenticated_handler handler)
{
    m_session_authenticated_handler = std::move(handler);
    rewire_protocol_handlers();
}

void NodeSession::set_work_ready_handler(Work_ready_handler handler)
{
    m_work_ready_handler = std::move(handler);
    rewire_protocol_handlers();
}

void NodeSession::set_session_start_handler(Session_start_handler handler)
{
    m_session_start_handler = std::move(handler);
    rewire_protocol_handlers();
}

void NodeSession::set_node_shutdown_handler(Node_shutdown_handler handler)
{
    m_node_shutdown_handler = std::move(handler);
    rewire_protocol_handlers();
}

void NodeSession::set_replacement_pending_handler(Replacement_pending_handler handler)
{
    m_replacement_pending_handler = std::move(handler);
    rewire_protocol_handlers();
}

void NodeSession::set_transport_lost_handler(Transport_lost_handler handler)
{
    m_transport_lost_handler = std::move(handler);
}

void NodeSession::set_miner_keys(const std::vector<uint8_t>& pubkey, const std::vector<uint8_t>& privkey)
{
    m_miner_pubkey = pubkey;
    m_miner_privkey = privkey;

    if (m_primary_protocol) {
        m_primary_protocol->set_miner_keys(pubkey, privkey);
    }
    if (m_secondary_protocol) {
        m_secondary_protocol->set_miner_keys(pubkey, privkey);
    }
}

void NodeSession::set_reward_address(const std::string& address)
{
    m_reward_address = address;

    if (m_primary_protocol) {
        m_primary_protocol->set_reward_address(address);
    }
    if (m_secondary_protocol) {
        m_secondary_protocol->set_reward_address(address);
    }
}

void NodeSession::set_tritium_genesis(const std::vector<uint8_t>& genesis)
{
    m_tritium_genesis = genesis;

    if (m_primary_protocol) {
        m_primary_protocol->set_tritium_genesis(genesis);
    }
    if (m_secondary_protocol) {
        m_secondary_protocol->set_tritium_genesis(genesis);
    }
}

void NodeSession::set_keepalive_interval(uint16_t hours)
{
    m_keepalive_interval_hours = hours;

    if (m_primary_protocol) {
        m_primary_protocol->set_keepalive_interval(hours);
    }
    if (m_secondary_protocol) {
        m_secondary_protocol->set_keepalive_interval(hours);
    }
}

network::Shared_payload NodeSession::request_work(protocol::GetBlockReason reason)
{
    auto protocol = get_active_protocol();
    if (!protocol) return nullptr;
    return protocol->get_work(reason);
}

network::Shared_payload NodeSession::submit_block(const std::vector<uint8_t>& block_data, uint64_t nonce)
{
    auto protocol = get_active_protocol();
    if (!protocol || !protocol->is_authenticated()) return nullptr;
    return protocol->submit_block(block_data, nonce);
}

network::Shared_payload NodeSession::send_get_round()
{
    auto protocol = get_active_protocol();
    if (!protocol) return nullptr;
    return protocol->send_get_round();
}

network::Shared_payload NodeSession::send_session_keepalive()
{
    auto protocol = get_active_protocol();
    if (!protocol) return nullptr;
    return protocol->send_session_keepalive();
}

bool NodeSession::login_on_active_connection(std::function<void(bool)> login_callback)
{
    if (m_stopped.load(std::memory_order_acquire)) {
        m_logger->warn("[NodeSession:{}] Cannot login - session is stopped", m_node_label);
        return false;
    }

    // Use select_active_pair() for atomic connection+protocol selection.
    // Both are checked together so the auth payload is always framed for
    // the correct lane (no TOCTOU between protocol selection and transmit).
    auto [conn, proto] = select_active_pair();
    if (conn && proto) {
        auto auth_payload = proto->login(login_callback);
        if (auth_payload && !auth_payload->empty()) {
            if (conn->transmit(auth_payload)) {
                return true;
            }
            m_logger->error("[NodeSession:{}] Failed to queue login payload on active connection", m_node_label);
            if (login_callback) login_callback(false);
            return false;
        }
        // Solo::login() can return empty (without invoking the callback) when
        // PacketBuilder::build() fails — e.g. if the Falcon keys are not yet
        // configured or the session state is inconsistent.  Fire the callback
        // ourselves so callers always receive notification and can schedule a retry.
        if (login_callback) login_callback(false);
        return false;
    }

    m_logger->warn("[NodeSession:{}] No active connection for login", m_node_label);
    return false;
}

void NodeSession::set_epoch_coordinator(std::shared_ptr<protocol::EpochCoordinator> coordinator)
{
    if (!m_session_context) {
        m_logger->error("[NodeSession:{}] set_epoch_coordinator: m_session_context is null - "
                        "coordinator cannot be wired. Check initialization order.", m_node_label);
        return;
    }
    auto session_mgr = m_session_context->get_session_manager();
    if (!session_mgr) {
        m_logger->error("[NodeSession:{}] set_epoch_coordinator: session_manager is null - "
                        "coordinator cannot be wired. Check initialization order.", m_node_label);
        return;
    }
    session_mgr->set_epoch_coordinator(std::move(coordinator));
}

} // namespace nexusminer
