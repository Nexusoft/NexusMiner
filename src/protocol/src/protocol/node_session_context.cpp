#include "protocol/node_session_context.hpp"
#include "network/types.hpp"
#include <cstring>

namespace nexusminer {
namespace protocol {

NodeSessionContext::NodeSessionContext(std::shared_ptr<SessionManager> session_manager)
    : m_session_manager(std::move(session_manager))
{
}

SessionId NodeSessionContext::get_session_id() const
{
    return m_session_manager ? m_session_manager->get_session_id() : SessionId{};
}

SessionEpoch NodeSessionContext::get_session_epoch() const
{
    return m_session_manager ? m_session_manager->get_session_epoch() : SessionEpoch{};
}

bool NodeSessionContext::is_authenticated() const
{
    return m_session_manager ? m_session_manager->is_authenticated() : false;
}

bool NodeSessionContext::is_active() const
{
    return m_session_manager ? m_session_manager->is_active() : false;
}

SessionManager::SessionState NodeSessionContext::get_state() const
{
    return m_session_manager ? m_session_manager->get_state()
                             : SessionManager::SessionState::DISCONNECTED;
}

void NodeSessionContext::start_session(SessionId session_id,
                                       const std::vector<uint8_t>& session_key,
                                       SessionGenesisHash tritium_genesis)
{
    if (m_session_manager) {
        m_session_manager->start_session(session_id, session_key, tritium_genesis);
    }
}

void NodeSessionContext::start_session(SessionId session_id,
                                       const std::vector<uint8_t>& session_key,
                                       const std::vector<uint8_t>& tritium_genesis)
{
    start_session(session_id, session_key, SessionGenesisHash(tritium_genesis));
}

void NodeSessionContext::commit_authenticated_session(SessionId session_id,
                                                      const std::vector<uint8_t>& pubkey,
                                                      FalconHashKeyId key_id,
                                                      SessionGenesisHash tritium_genesis)
{
    if (m_session_manager) {
        m_session_manager->commit_authenticated_session(session_id, pubkey, key_id, tritium_genesis);
    }
}

void NodeSessionContext::commit_authenticated_session(SessionId session_id,
                                                      const std::vector<uint8_t>& pubkey,
                                                      const std::string& key_id,
                                                      const std::vector<uint8_t>& tritium_genesis)
{
    commit_authenticated_session(session_id, pubkey, FalconHashKeyId(key_id),
                                 SessionGenesisHash(tritium_genesis));
}

void NodeSessionContext::begin_auth_handshake(const std::string& detail)
{
    if (m_session_manager) {
        m_session_manager->begin_auth_handshake(detail);
    }
}

void NodeSessionContext::begin_reward_binding(const std::string& reward_address,
                                              RewardHash reward_hash,
                                              const std::string& source)
{
    if (m_session_manager) {
        m_session_manager->begin_reward_binding(reward_address, reward_hash, source);
    }
}

void NodeSessionContext::begin_reward_binding(const std::string& reward_address,
                                              const std::vector<uint8_t>& reward_hash,
                                              const std::string& source)
{
    begin_reward_binding(reward_address, RewardHash(reward_hash), source);
}

void NodeSessionContext::commit_reward_bound(const std::string& reward_address,
                                             RewardHash reward_hash,
                                             const std::string& source)
{
    if (m_session_manager) {
        m_session_manager->commit_reward_bound(reward_address, reward_hash, source);
    }
}

void NodeSessionContext::commit_reward_bound(const std::string& reward_address,
                                             const std::vector<uint8_t>& reward_hash,
                                             const std::string& source)
{
    commit_reward_bound(reward_address, RewardHash(reward_hash), source);
}

void NodeSessionContext::commit_reward_rejected(const std::string& reward_address,
                                                const std::string& source,
                                                const std::string& reason)
{
    if (m_session_manager) {
        m_session_manager->commit_reward_rejected(reward_address, source, reason);
    }
}

void NodeSessionContext::note_keepalive_ack(bool accepted, const std::string& detail)
{
    if (m_session_manager) {
        m_session_manager->note_keepalive_ack(accepted, detail);
    }
}

void NodeSessionContext::mark_recovery_required(const std::string& reason)
{
    if (m_session_manager) {
        m_session_manager->mark_recovery_required(reason);
    }
}

void NodeSessionContext::mark_recovery_healthy(const std::string& reason)
{
    if (m_session_manager) {
        m_session_manager->mark_recovery_healthy(reason);
    }
}

void NodeSessionContext::mark_session_expired(const std::string& reason)
{
    if (m_session_manager) {
        m_session_manager->mark_session_expired(reason);
    }
}

void NodeSessionContext::clear_for_disconnect(const std::string& reward_address,
                                              const std::string& reward_source,
                                              const std::string& reason,
                                              bool preserve_genesis)
{
    if (m_session_manager) {
        m_session_manager->clear_for_disconnect(reward_address, reward_source, reason, preserve_genesis);
    }
}

void NodeSessionContext::clear_for_reauth(const std::string& reward_address,
                                          const std::string& reward_source,
                                          const std::string& reason,
                                          bool preserve_genesis)
{
    if (m_session_manager) {
        m_session_manager->clear_for_reauth(reward_address, reward_source, reason, preserve_genesis);
    }
}

void NodeSessionContext::end_session()
{
    if (m_session_manager) {
        m_session_manager->end_session();
    }
}

void NodeSessionContext::set_state(SessionManager::SessionState state)
{
    if (m_session_manager) {
        m_session_manager->set_state(state);
    }
}

network::Shared_payload NodeSessionContext::build_keepalive_packet() const
{
    return m_session_manager ? m_session_manager->build_keepalive_packet() : nullptr;
}

network::Shared_payload NodeSessionContext::build_session_status_packet(
    bool degraded,
    bool has_template,
    bool workers_running,
    bool secondary_up) const
{
    return m_session_manager ? m_session_manager->build_session_status_packet(
        degraded, has_template, workers_running, secondary_up) : nullptr;
}

void NodeSessionContext::start_keepalive_timer()
{
    if (m_session_manager) {
        m_session_manager->start_keepalive_timer();
    }
}

void NodeSessionContext::stop_keepalive_timer()
{
    if (m_session_manager) {
        m_session_manager->stop_keepalive_timer();
    }
}

void NodeSessionContext::set_keepalive_interval(uint16_t hours)
{
    if (m_session_manager) {
        m_session_manager->set_keepalive_interval(hours);
    }
}

uint16_t NodeSessionContext::get_keepalive_interval() const
{
    return m_session_manager ? m_session_manager->get_keepalive_interval() : 0;
}

void NodeSessionContext::set_protocol_lane(ProtocolLane lane)
{
    if (m_session_manager) {
        m_session_manager->set_protocol_lane(lane);
    }
}

void NodeSessionContext::set_connection(std::shared_ptr<network::Connection> connection)
{
    if (m_session_manager) {
        m_session_manager->set_connection(std::move(connection));
    }
}

void NodeSessionContext::set_tritium_genesis(SessionGenesisHash genesis)
{
    if (m_session_manager) {
        m_session_manager->set_tritium_genesis(genesis);
    }
}

void NodeSessionContext::set_tritium_genesis(const std::vector<uint8_t>& genesis)
{
    set_tritium_genesis(SessionGenesisHash(genesis));
}

void NodeSessionContext::set_connection_metadata(const std::string& local_endpoint,
                                                 const std::string& remote_endpoint,
                                                 bool connected)
{
    if (m_session_manager) {
        m_session_manager->set_connection_metadata(local_endpoint, remote_endpoint, connected);
    }
}

void NodeSessionContext::set_falcon_identity(const std::vector<uint8_t>& pubkey,
                                             FalconHashKeyId key_id,
                                             bool authenticated)
{
    if (m_session_manager) {
        m_session_manager->set_falcon_identity(pubkey, key_id, authenticated);
    }
}

void NodeSessionContext::set_falcon_identity(const std::vector<uint8_t>& pubkey,
                                             const std::string& key_id,
                                             bool authenticated)
{
    set_falcon_identity(pubkey, FalconHashKeyId(key_id), authenticated);
}

void NodeSessionContext::reset_session_credentials()
{
    if (m_session_manager) {
        m_session_manager->reset_session_credentials();
    }
}

void NodeSessionContext::set_chacha20_session_key(const std::vector<uint8_t>& session_key,
                                                  SessionFingerprint fingerprint,
                                                  bool ready)
{
    if (m_session_manager) {
        m_session_manager->set_chacha20_session_key(session_key, fingerprint, ready);
    }
}

void NodeSessionContext::set_chacha20_session_key(const std::vector<uint8_t>& session_key,
                                                  const std::string& fingerprint,
                                                  bool ready)
{
    set_chacha20_session_key(session_key, SessionFingerprint(fingerprint), ready);
}

void NodeSessionContext::set_reward_binding(const std::string& reward_address,
                                            RewardHash reward_hash,
                                            bool bound,
                                            const std::string& source)
{
    if (m_session_manager) {
        m_session_manager->set_reward_binding(reward_address, reward_hash, bound, source);
    }
}

void NodeSessionContext::set_reward_binding(const std::string& reward_address,
                                            const std::vector<uint8_t>& reward_hash,
                                            bool bound,
                                            const std::string& source)
{
    set_reward_binding(reward_address, RewardHash(reward_hash), bound, source);
}

void NodeSessionContext::set_channel_state(uint32_t channel,
                                           bool ready_for_submit,
                                           bool ready_for_get_block)
{
    if (m_session_manager) {
        m_session_manager->set_channel_state(channel, ready_for_submit, ready_for_get_block);
    }
}

void NodeSessionContext::mark_activity()
{
    if (m_session_manager) {
        m_session_manager->mark_activity();
    }
}

bool NodeSessionContext::is_reward_bound() const
{
    return m_session_manager ? m_session_manager->is_reward_bound() : false;
}

bool NodeSessionContext::reward_binding_required() const
{
    return m_session_manager ? m_session_manager->reward_binding_required() : false;
}

SessionManager::RewardBindReadiness NodeSessionContext::get_reward_bind_readiness() const
{
    if (m_session_manager) {
        return m_session_manager->get_reward_bind_readiness();
    }
    return SessionManager::RewardBindReadiness{false, "session manager unavailable"};
}

bool NodeSessionContext::can_submit_work() const
{
    return m_session_manager ? m_session_manager->can_submit_work() : false;
}

bool NodeSessionContext::can_request_get_block() const
{
    return m_session_manager ? m_session_manager->can_request_get_block() : false;
}

bool NodeSessionContext::allow_deferred_push_replay() const
{
    return m_session_manager ? m_session_manager->allow_deferred_push_replay() : false;
}

bool NodeSessionContext::allow_get_block_replay() const
{
    return m_session_manager ? m_session_manager->allow_get_block_replay() : false;
}

bool NodeSessionContext::validate_miner_session(std::string* reason) const
{
    return m_session_manager ? m_session_manager->validate_miner_session(reason) : false;
}

std::string NodeSessionContext::build_miner_session_diagnostics() const
{
    return m_session_manager ? m_session_manager->build_miner_session_diagnostics() : std::string{};
}

void NodeSessionContext::record_session_event(SessionManager::SessionEventKind kind,
                                              const std::string& detail)
{
    if (m_session_manager) {
        m_session_manager->record_session_event(kind, detail);
    }
}

std::vector<SessionManager::SessionEvent> NodeSessionContext::get_session_event_journal() const
{
    return m_session_manager ? m_session_manager->get_session_event_journal()
                             : std::vector<SessionManager::SessionEvent>{};
}

std::string NodeSessionContext::build_session_event_journal() const
{
    return m_session_manager ? m_session_manager->build_session_event_journal() : std::string{};
}

SessionGenesisHash NodeSessionContext::get_typed_tritium_genesis() const
{
    return m_session_manager ? m_session_manager->get_typed_tritium_genesis() : SessionGenesisHash{};
}

std::vector<uint8_t> NodeSessionContext::get_tritium_genesis() const
{
    return get_typed_tritium_genesis().get();
}

std::vector<uint8_t> NodeSessionContext::get_session_key() const
{
    return m_session_manager ? m_session_manager->get_session_key() : std::vector<uint8_t>{};
}

std::chrono::seconds NodeSessionContext::get_session_uptime() const
{
    return m_session_manager ? m_session_manager->get_session_uptime() : std::chrono::seconds(0);
}

SessionManager::SessionInfo NodeSessionContext::get_session_info() const
{
    return get_runtime_snapshot();
}

SessionManager::RuntimeSessionSnapshot NodeSessionContext::get_runtime_snapshot() const
{
    if (m_session_manager) {
        return m_session_manager->get_runtime_snapshot();
    }
    return SessionManager::RuntimeSessionSnapshot{};
}

SessionIdentity NodeSessionContext::get_canonical_identity() const
{
    if (m_session_manager) {
        return m_session_manager->get_canonical_identity();
    }
    return SessionIdentity{};
}

SessionBinding NodeSessionContext::get_session_binding() const
{
    if (m_session_manager) {
        return m_session_manager->get_session_binding();
    }
    return SessionBinding{};
}

void NodeSessionContext::set_session_expired_handler(SessionManager::SessionExpiredHandler handler)
{
    if (m_session_manager) {
        m_session_manager->set_session_expired_handler(std::move(handler));
    }
}

void NodeSessionContext::set_prevblock_suffix(const std::array<uint8_t, 4>& suffix)
{
    if (m_session_manager) {
        m_session_manager->set_prevblock_suffix(suffix);
    }
}

bool NodeSessionContext::parse_session_start(
    const std::vector<uint8_t>& packet_data,
    uint32_t& out_session_id,
    uint32_t& out_timeout,
    std::vector<uint8_t>& out_genesis)
{
    // SESSION_START wire format:
    // [0]      success (0x01)
    // [1..4]   session_id (little-endian uint32)
    // [5..8]   timeout (little-endian uint32, seconds)
    // [9..40]  genesis hash (optional, 32 bytes)

    if (packet_data.size() < 9) {
        return false;  // Minimum size: 1 + 4 + 4
    }

    // Check success byte
    if (packet_data[0] != 0x01) {
        return false;
    }

    // Parse session_id (little-endian)
    out_session_id = static_cast<uint32_t>(packet_data[1]) |
                    (static_cast<uint32_t>(packet_data[2]) << 8) |
                    (static_cast<uint32_t>(packet_data[3]) << 16) |
                    (static_cast<uint32_t>(packet_data[4]) << 24);

    // Parse timeout (little-endian)
    out_timeout = static_cast<uint32_t>(packet_data[5]) |
                 (static_cast<uint32_t>(packet_data[6]) << 8) |
                 (static_cast<uint32_t>(packet_data[7]) << 16) |
                 (static_cast<uint32_t>(packet_data[8]) << 24);

    // Parse optional genesis hash
    if (packet_data.size() >= 41) {
        out_genesis.resize(32);
        std::memcpy(out_genesis.data(), packet_data.data() + 9, 32);
    } else {
        out_genesis.clear();
    }

    return true;
}

} // namespace protocol
} // namespace nexusminer
