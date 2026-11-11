#include "worker_manager.hpp"
#include "node_session/node_session.hpp"
#include "protocol/solo.hpp"
#include "network/connection.hpp"
#include "config/config.hpp"
#include "worker/worker.hpp"

#include <asio/io_context.hpp>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <memory>
#include <stdexcept>

namespace nexusminer {
class DeadWorker final : public Worker {
public:
    void set_block(::LLP::CBlock, std::uint32_t, Block_found_handler) override {}
    void update_statistics(stats::Collector&) override {}
    bool uses_template_feed() const override { return true; }
    bool is_running() const override { return false; }
};

struct Worker_manager_recovery_test_access {
    static std::shared_ptr<NodeSession> session(Worker_manager& manager) {
        return manager.m_primary_node_session;
    }
    static void start_old_recovery(Worker_manager& manager) {
        manager.mark_recovery_initiated("startup_test");
        manager.m_primary_endpoint = network::Endpoint{
            network::Transport_protocol::tcp, "127.0.0.1", 9323};
        const auto old = std::chrono::steady_clock::now() - std::chrono::minutes(10);
        manager.m_recovery.degraded_since = old;
        manager.m_recovery.timeout.reset();
        manager.m_recovery.timeout.start_if_unset(old);
    }
    static RecoveryPhase phase(const Worker_manager& manager) {
        return manager.m_recovery.phase.load();
    }
    static size_t workers(const Worker_manager& manager) {
        return manager.m_workers.size();
    }
    static void stop_workers(Worker_manager& manager) {
        manager.mark_recovery_initiated("replacement_worker_test");
        manager.stop_all_workers();
    }
    static void install_dead_worker(Worker_manager& manager) {
        stop_workers(manager);
        manager.m_workers.push_back(std::make_shared<DeadWorker>());
        manager.m_recovery_workers_spawned = true;
    }
    static bool valid_transition(RecoveryPhase from, RecoveryPhase to) {
        return Worker_manager::is_valid_transition(from, to);
    }
    static uint32_t auth_failures(const Worker_manager& manager) {
        return manager.m_session_auth_fail_count;
    }
    static uint32_t connection_retries(const Worker_manager& manager) {
        return manager.m_connection_retry_count;
    }
    static void node_shutdown(Worker_manager& manager) {
        manager.handle_node_shutdown(0);
    }
    static void set_endpoint(Worker_manager& manager) {
        manager.m_primary_endpoint = network::Endpoint{
            network::Transport_protocol::tcp, "127.0.0.1", 9323};
    }
    static void ensure_transport_handler(Worker_manager& manager) {
        manager.ensure_transport_lost_handler();
    }
    static void transport_lost(Worker_manager& manager, const char* reason) {
        manager.handle_transport_lost(reason);
    }
    static bool retry_scheduled(const Worker_manager& manager) {
        return manager.m_connection_retry_scheduled;
    }
    static void enter_reconnecting(Worker_manager& manager) {
        manager.transition_to(RecoveryPhase::RECONNECTING, "review_comment_test");
    }
    static void age_reconnect(Worker_manager& manager,
                              std::chrono::seconds age = std::chrono::seconds(91)) {
        manager.m_recovery.reconnect_started_at = std::chrono::steady_clock::now() - age;
    }
    static bool has_pending_submit(const Worker_manager& manager) {
        return manager.m_pending_submit.has_value();
    }
    static uint32_t pending_height(const Worker_manager& manager) {
        return manager.m_pending_submit && manager.m_pending_submit->solved
            ? manager.m_pending_submit->solved->nHeight : 0;
    }
    static uint64_t pending_nonce(const Worker_manager& manager) {
        return manager.m_pending_submit && manager.m_pending_submit->solved
            ? manager.m_pending_submit->solved->nNonce : 0;
    }
    static void accept_solved(Worker_manager& manager, std::unique_ptr<Block_data> block) {
        manager.accept_solved_block(std::move(block));
    }
    static bool has_pending_submit_timer(const Worker_manager& manager) {
        return static_cast<bool>(manager.m_pending_submit_expiry_timer);
    }
    static std::chrono::steady_clock::duration pending_submit_ttl(const Worker_manager& manager) {
        return manager.m_pending_submit_expiry_timer->expiry() - manager.m_pending_submit->found_at;
    }
    static std::weak_ptr<Block_data> pending_snapshot(const Worker_manager& manager) {
        return manager.m_pending_submit->solved;
    }
    static void set_pending_time_remaining(Worker_manager& manager, std::chrono::milliseconds remaining) {
        manager.m_pending_submit->found_at = std::chrono::steady_clock::now() -
            std::chrono::seconds(Worker_manager::PENDING_SUBMIT_MAX_AGE_SECONDS) + remaining;
        manager.schedule_pending_submit_expiry();
    }
    static void flush_pending(Worker_manager& manager) {
        manager.flush_pending_submit("test_recovery");
    }
};

struct NodeSession_recovery_test_access {
    static void connect(NodeSession& session, std::shared_ptr<network::Connection> connection) {
        session.m_primary_connection = std::move(connection);
        session.m_primary_connected.store(true);
    }
    static void restore_template_handler(NodeSession& session) {
        session.set_template_handler(session.m_template_handler);
    }
    static bool has_transport_lost_handler(const NodeSession& session) {
        return static_cast<bool>(session.m_transport_lost_handler);
    }
    static void drop_primary(NodeSession& session) {
        session.mark_lane_socket_failed(NodeSession::LaneSlot::Primary);
    }
};

namespace protocol {
struct GetBlockControlHarness {
    static void sync_session(Solo& solo) {
        solo.refresh_cached_session_state("startup recovery test");
    }
    static void expire_retry(Solo& solo) {
        solo.m_get_block_retry_at_ms.store(1);
    }
    static void report_auth_failure(Solo& solo, std::shared_ptr<network::Connection> connection,
                                    bool zero_session_id = false) {
        auto payload = std::make_shared<network::Payload>(
            zero_session_id ? network::Payload{1, 0, 0, 0, 0} : network::Payload{0, 1});
        Packet response(Packet::STATELESS_MINER_AUTH_RESULT, payload);
        solo.process_messages(response, std::move(connection));
        assert(solo.m_auth_state == AuthState::NOT_AUTHENTICATED);
    }
    static void report_auth_success(Solo& solo) {
        assert(solo.m_session_authenticated_handler);
        solo.m_session_authenticated_handler(SessionId{123});
    }
    static void report_session_expired(Solo& solo) {
        assert(solo.m_session_expired_handler);
        solo.m_session_expired_handler();
    }
};
}
}

using namespace nexusminer;

class MockConnection final : public network::Connection {
public:
    network::Endpoint const& remote_endpoint() const override { return m_remote; }
    network::Endpoint const& local_endpoint() const override { return m_local; }
    bool transmit(network::Shared_payload data) override { return data && !data->empty(); }
    void close() override {
        closed = true;
        if (close_handler) {
            auto handler = std::move(close_handler);
            handler(network::Result::connection_closed, {});
        }
    }
    ProtocolLane get_protocol_lane() const override { return ProtocolLane::STATELESS; }
    bool closed{false};
    Handler close_handler;
private:
    network::Endpoint m_remote{network::Transport_protocol::tcp, "127.0.0.1", 9323};
    network::Endpoint m_local{network::Transport_protocol::tcp, "127.0.0.1", 0};
};

class PendingConnectSocket final : public network::Socket {
public:
    network::Result::Code listen(Connect_handler) override {
        return network::Result::socket_ok;
    }
    void stop_listen() override {}
    network::Endpoint const& local_endpoint() const override { return m_local; }
    network::Connection::Sptr connect(network::Endpoint, network::Connection::Handler handler) override {
        auto connection = std::make_shared<MockConnection>();
        connection->close_handler = handler;
        connections.push_back(connection);
        handlers.push_back(std::move(handler));
        return connection;
    }
    void emit_connect_result(size_t index, network::Result::Code result) {
        handlers.at(index)(result, {});
    }
    std::vector<std::shared_ptr<MockConnection>> connections;
private:
    network::Endpoint m_local{network::Transport_protocol::tcp, "127.0.0.1", 0};
    std::vector<network::Connection::Handler> handlers;
};

void wait_for_connections(asio::io_context& io, const PendingConnectSocket& socket,
                          size_t expected)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (socket.connections.size() < expected && std::chrono::steady_clock::now() < deadline) {
        io.restart();
        io.run_one_for(std::chrono::milliseconds(20));
    }
    if (socket.connections.size() != expected) {
        throw std::runtime_error("Timed out waiting for the scheduled test connection");
    }
}

void test_stalled_connect_recovery(config::Config& config)
{
    auto io = std::make_shared<asio::io_context>();
    auto socket = std::make_shared<PendingConnectSocket>();
    auto manager = std::make_shared<Worker_manager>(
        io, config, std::make_shared<chrono::Timer_factory>(io), socket);
    auto session = Worker_manager_recovery_test_access::session(*manager);
    session->set_miner_keys(std::vector<uint8_t>(897, 0xaa), std::vector<uint8_t>(1281, 0xbb));
    session->set_tritium_genesis(std::vector<uint8_t>(32, 0x11));
    const network::Endpoint endpoint{network::Transport_protocol::tcp, "127.0.0.1", 9323};
    Worker_manager_recovery_test_access::enter_reconnecting(*manager);
    const bool started = manager->connect(endpoint);
    assert(started);
    assert(session->has_pending_connect());
    assert(!session->is_primary_connected());

    Worker_manager_recovery_test_access::age_reconnect(*manager, std::chrono::seconds(90));
    manager->check_template_health();
    assert(session->has_pending_connect());
    assert(!socket->connections[0]->closed);
    assert(Worker_manager_recovery_test_access::connection_retries(*manager) == 0);

    Worker_manager_recovery_test_access::age_reconnect(*manager);
    manager->check_template_health();
    assert(socket->connections[0]->closed);
    assert(!session->has_pending_connect());
    assert(Worker_manager_recovery_test_access::connection_retries(*manager) == 1);
    assert(Worker_manager_recovery_test_access::retry_scheduled(*manager));
    manager->check_template_health();
    assert(Worker_manager_recovery_test_access::connection_retries(*manager) == 1);

    wait_for_connections(*io, *socket, 2);
    assert(socket->connections.size() == 2);
    assert(session->has_pending_connect());
    socket->emit_connect_result(0, network::Result::connection_ok);
    assert(!session->is_primary_connected());
    assert(session->has_pending_connect());

    // The TCP socket may connect while authentication never completes.
    socket->emit_connect_result(1, network::Result::connection_ok);
    assert(session->is_primary_connected());
    assert(session->has_pending_connect());
    Worker_manager_recovery_test_access::age_reconnect(*manager);
    manager->check_template_health();
    assert(socket->connections[1]->closed);
    assert(!session->has_pending_connect());
    assert(!session->is_primary_connected());
    assert(Worker_manager_recovery_test_access::connection_retries(*manager) == 2);
    assert(Worker_manager_recovery_test_access::retry_scheduled(*manager));

    wait_for_connections(*io, *socket, 3);
    assert(socket->connections.size() == 3);
    socket->emit_connect_result(2, network::Result::connection_ok);
    auto solo = session->get_primary_protocol();
    solo->get_session_manager()->commit_authenticated(protocol::SessionId{123}, ProtocolLane::STATELESS);
    protocol::GetBlockControlHarness::sync_session(*solo);
    protocol::GetBlockControlHarness::report_auth_success(*solo);
    assert(!session->has_pending_connect());
    assert(Worker_manager_recovery_test_access::connection_retries(*manager) == 0);

    // Work readiness can be slow after authentication without making the socket dead.
    Worker_manager_recovery_test_access::age_reconnect(*manager);
    manager->check_template_health();
    assert(session->is_primary_connected());
    assert(!socket->connections[2]->closed);
    assert(Worker_manager_recovery_test_access::phase(*manager) == RecoveryPhase::WAITING_TEMPLATE);
    manager->stop();
}

void test_pending_submit_expiry(config::Config& config)
{
    auto io = std::make_shared<asio::io_context>();
    auto manager = std::make_shared<Worker_manager>(
        io, config, std::make_shared<chrono::Timer_factory>(io), nullptr);
    auto session = Worker_manager_recovery_test_access::session(*manager);
    auto make_solved = [](uint64_t nonce) {
        auto solved = std::make_unique<Block_data>();
        solved->nNonce = nonce;
        return solved;
    };

    Worker_manager_recovery_test_access::accept_solved(*manager, make_solved(1));
    assert(Worker_manager_recovery_test_access::has_pending_submit(*manager));
    assert(Worker_manager_recovery_test_access::has_pending_submit_timer(*manager));
    assert(Worker_manager_recovery_test_access::pending_submit_ttl(*manager) == std::chrono::seconds(180));
    const auto old_snapshot = Worker_manager_recovery_test_access::pending_snapshot(*manager);

    // An overdue callback belonging to the replaced snapshot must not clear its successor.
    Worker_manager_recovery_test_access::set_pending_time_remaining(*manager, std::chrono::milliseconds(0));
    Worker_manager_recovery_test_access::accept_solved(*manager, make_solved(2));
    io->poll();
    assert(old_snapshot.expired());
    assert(Worker_manager_recovery_test_access::pending_nonce(*manager) == 2);
    assert(Worker_manager_recovery_test_access::pending_submit_ttl(*manager) == std::chrono::seconds(180));

    // Exercise the real expiry callback without waiting 180 seconds or delivering recovery events.
    const auto expiring_snapshot = Worker_manager_recovery_test_access::pending_snapshot(*manager);
    Worker_manager_recovery_test_access::flush_pending(*manager);
    assert(Worker_manager_recovery_test_access::has_pending_submit(*manager));
    assert(Worker_manager_recovery_test_access::pending_submit_ttl(*manager) == std::chrono::seconds(180));
    assert(!session->is_authenticated());
    Worker_manager_recovery_test_access::set_pending_time_remaining(*manager, std::chrono::milliseconds(0));
    io->restart();
    io->run_for(std::chrono::milliseconds(200));
    assert(!Worker_manager_recovery_test_access::has_pending_submit(*manager));
    assert(!Worker_manager_recovery_test_access::has_pending_submit_timer(*manager));
    assert(expiring_snapshot.expired());
    assert(!session->is_authenticated());

    // A recovery callback at the TTL boundary must not get an extra truncated second.
    Worker_manager_recovery_test_access::accept_solved(*manager, make_solved(3));
    Worker_manager_recovery_test_access::set_pending_time_remaining(*manager, std::chrono::milliseconds(0));
    Worker_manager_recovery_test_access::flush_pending(*manager);
    assert(!Worker_manager_recovery_test_access::has_pending_submit(*manager));
    assert(!Worker_manager_recovery_test_access::has_pending_submit_timer(*manager));

    Worker_manager_recovery_test_access::accept_solved(*manager, make_solved(4));
    const auto stopped_snapshot = Worker_manager_recovery_test_access::pending_snapshot(*manager);
    manager->stop();
    assert(!Worker_manager_recovery_test_access::has_pending_submit(*manager));
    assert(!Worker_manager_recovery_test_access::has_pending_submit_timer(*manager));
    assert(stopped_snapshot.expired());
    io->restart();
    io->poll();
    Worker_manager_recovery_test_access::accept_solved(*manager, make_solved(5));
    assert(!Worker_manager_recovery_test_access::has_pending_submit(*manager));
}

int main()
{
    // Verify core recovery FSM invariants used by this startup regression test:
    // once the manager reaches DEGRADED_TERMINAL, it must not transition back.
    assert(!Worker_manager_recovery_test_access::valid_transition(
        RecoveryPhase::DEGRADED_TERMINAL, RecoveryPhase::HEALTHY));
    assert(!Worker_manager_recovery_test_access::valid_transition(
        RecoveryPhase::DEGRADED_TERMINAL, RecoveryPhase::RECONNECTING));

    // Install a null logger to keep test output deterministic and quiet.
    auto sink = std::make_shared<spdlog::sinks::null_sink_mt>();
    auto logger = std::make_shared<spdlog::logger>("logger", sink);
    spdlog::register_logger(logger);
    spdlog::set_default_logger(logger);

    // Create isolated runtime/config state for startup and recovery exercises.
    auto io = std::make_shared<asio::io_context>();
    config::Config config(logger);
    config.set_mining_mode(config::Mining_mode::HASH);
    config.set_miner_falcon_pubkey("00");
    config.set_miner_falcon_privkey("00");
    config.set_worker_count(1);
    config.set_connection_retry_interval(60);
    auto manager = std::make_shared<Worker_manager>(
        io, config, std::make_shared<chrono::Timer_factory>(io), nullptr);
    auto session = Worker_manager_recovery_test_access::session(*manager);
    auto connection = std::make_shared<MockConnection>();
    NodeSession_recovery_test_access::connect(*session, connection);

    auto solo = session->get_primary_protocol();
    solo->set_protocol_lane(ProtocolLane::STATELESS);
    auto* sessions = solo->get_session_manager();
    sessions->commit_authenticated(protocol::SessionId{123}, ProtocolLane::STATELESS);
    sessions->set_channel_state(2, true, true);
    protocol::GetBlockControlHarness::sync_session(*solo);
    solo->get_template_interface()->set_session_binding(sessions->get_session_binding());
    assert(solo->is_authenticated());
    assert(sessions->get_session_binding().may_request_work());

    Worker_manager_recovery_test_access::start_old_recovery(*manager);
    solo->mark_get_block_pending(protocol::GetBlockReason::INITIAL_REQUEST);
    auto payload = std::make_shared<network::Payload>(
        network::Payload{1, 4, 0, 0, 0, 0, 3, 0xe8});
    Packet response(static_cast<uint16_t>(0xd0c9), payload);
    solo->process_messages(response, connection);
    assert(solo->is_get_block_retry_deferred());
    assert(solo->is_node_template_wait_active());
    protocol::GetBlockControlHarness::expire_retry(*solo);
    assert(!solo->is_get_block_retry_deferred());

    manager->check_template_health();
    assert(Worker_manager_recovery_test_access::phase(*manager) == RecoveryPhase::WAITING_TEMPLATE);
    assert(Worker_manager_recovery_test_access::workers(*manager) == 1);
    assert(session->is_primary_connected());

    auto block = std::make_shared<network::Payload>(228, 0);
    auto write_u32 = [&](size_t offset, uint32_t value) {
        for (size_t i = 0; i < 4; ++i)
            (*block)[offset + i] = static_cast<uint8_t>(value >> (24 - 8 * i));
    };
    write_u32(0, 100);
    write_u32(4, 50);
    write_u32(8, 0x1d00ffff);
    write_u32(12, 8);
    std::fill(block->begin() + 16, block->begin() + 144, 0xaa);
    std::fill(block->begin() + 144, block->begin() + 208, 0xbb);
    write_u32(208, 2);
    write_u32(212, 101);
    write_u32(216, 0x1d00ffff);
    Packet template_response(static_cast<uint16_t>(0xd000), block);
    solo->mark_get_block_pending(protocol::GetBlockReason::RECOVERY_FORCED);
    solo->process_messages(template_response, connection);
    assert(Worker_manager_recovery_test_access::phase(*manager) == RecoveryPhase::HEALTHY);
    assert(Worker_manager_recovery_test_access::workers(*manager) == 1);

    Worker_manager_recovery_test_access::stop_workers(*manager);
    assert(Worker_manager_recovery_test_access::workers(*manager) == 0);
    solo->mark_get_block_pending(protocol::GetBlockReason::RECOVERY_FORCED);
    solo->process_messages(template_response, connection);
    assert(Worker_manager_recovery_test_access::phase(*manager) == RecoveryPhase::HEALTHY);
    assert(Worker_manager_recovery_test_access::workers(*manager) == 1);

    // A non-empty generation with no running pool threads must be destroyed outside the feed lock.
    Worker_manager_recovery_test_access::install_dead_worker(*manager);
    solo->mark_get_block_pending(protocol::GetBlockReason::RECOVERY_FORCED);
    solo->process_messages(template_response, connection);
    assert(Worker_manager_recovery_test_access::workers(*manager) == 0);
    assert(Worker_manager_recovery_test_access::phase(*manager) == RecoveryPhase::WAITING_TEMPLATE);
    solo->mark_get_block_pending(protocol::GetBlockReason::RECOVERY_FORCED);
    solo->process_messages(template_response, connection);
    assert(Worker_manager_recovery_test_access::workers(*manager) == 1);
    assert(Worker_manager_recovery_test_access::phase(*manager) == RecoveryPhase::HEALTHY);

    solo->set_block_handler([](const ::LLP::CBlock&, std::uint32_t) { return false; });
    std::fill(block->begin() + 16, block->begin() + 144, 0xcc);
    solo->mark_get_block_pending(protocol::GetBlockReason::RECOVERY_FORCED);
    solo->process_messages(template_response, connection);
    assert(Worker_manager_recovery_test_access::phase(*manager) == RecoveryPhase::WAITING_TEMPLATE);

    Worker_manager_recovery_test_access::start_old_recovery(*manager);
    manager->check_template_health();
    assert(Worker_manager_recovery_test_access::phase(*manager) == RecoveryPhase::RECONNECTING);
    assert(Worker_manager_recovery_test_access::workers(*manager) == 0);
    const auto retries_before_auth = Worker_manager_recovery_test_access::connection_retries(*manager);
    protocol::GetBlockControlHarness::report_auth_failure(*solo, connection);
    assert(Worker_manager_recovery_test_access::workers(*manager) == 0);
    io->run_for(std::chrono::milliseconds(1300));
    assert(Worker_manager_recovery_test_access::connection_retries(*manager) > retries_before_auth);
    for (int i = 0; i < 6; ++i) {
        protocol::GetBlockControlHarness::report_auth_failure(*solo, connection, i % 2 == 0);
    }
    assert(Worker_manager_recovery_test_access::auth_failures(*manager) == 7);
    assert(Worker_manager_recovery_test_access::phase(*manager) != RecoveryPhase::DEGRADED_TERMINAL);

    // Authentication alone must not resume stale work; usable work creates a new generation.
    NodeSession_recovery_test_access::connect(*session, connection);
    NodeSession_recovery_test_access::restore_template_handler(*session);
    sessions->commit_authenticated(protocol::SessionId{123}, ProtocolLane::STATELESS);
    sessions->set_channel_state(2, true, true);
    protocol::GetBlockControlHarness::sync_session(*solo);
    solo->get_template_interface()->set_session_binding(sessions->get_session_binding());
    protocol::GetBlockControlHarness::report_auth_success(*solo);
    assert(Worker_manager_recovery_test_access::workers(*manager) == 0);
    solo->mark_get_block_pending(protocol::GetBlockReason::RECOVERY_FORCED);
    solo->process_messages(template_response, connection);
    assert(Worker_manager_recovery_test_access::workers(*manager) == 1);
    assert(Worker_manager_recovery_test_access::phase(*manager) == RecoveryPhase::HEALTHY);
    protocol::GetBlockControlHarness::report_auth_failure(*solo, connection);
    assert(Worker_manager_recovery_test_access::workers(*manager) == 0);
    assert(Worker_manager_recovery_test_access::phase(*manager) == RecoveryPhase::SESSION_RECOVERY);
    Worker_manager_recovery_test_access::node_shutdown(*manager);
    assert(Worker_manager_recovery_test_access::phase(*manager) == RecoveryPhase::RECONNECTING);
    assert(Worker_manager_recovery_test_access::workers(*manager) == 0);
    manager->stop();

    auto io2 = std::make_shared<asio::io_context>();
    auto manager2 = std::make_shared<Worker_manager>(
        io2, config, std::make_shared<chrono::Timer_factory>(io2), nullptr);
    Worker_manager_recovery_test_access::set_endpoint(*manager2);
    auto session2 = Worker_manager_recovery_test_access::session(*manager2);
    NodeSession_recovery_test_access::connect(*session2, std::make_shared<MockConnection>());
    protocol::GetBlockControlHarness::report_session_expired(*session2->get_primary_protocol());
    assert(Worker_manager_recovery_test_access::phase(*manager2) == RecoveryPhase::SESSION_RECOVERY);
    Worker_manager_recovery_test_access::node_shutdown(*manager2);
    io2->run_for(std::chrono::milliseconds(50));
    assert(Worker_manager_recovery_test_access::phase(*manager2) == RecoveryPhase::RECONNECTING);
    assert(Worker_manager_recovery_test_access::connection_retries(*manager2) == 0);
    manager2->stop();

    // Review: transport-lost must be registered after shared ownership exists.
    // A constructor-time weak_ptr stays empty, so the posted handler never reconnects.
    auto io3 = std::make_shared<asio::io_context>();
    auto manager3 = std::make_shared<Worker_manager>(
        io3, config, std::make_shared<chrono::Timer_factory>(io3), nullptr);
    auto session3 = Worker_manager_recovery_test_access::session(*manager3);
    assert(!NodeSession_recovery_test_access::has_transport_lost_handler(*session3));
    Worker_manager_recovery_test_access::ensure_transport_handler(*manager3);
    assert(NodeSession_recovery_test_access::has_transport_lost_handler(*session3));
    Worker_manager_recovery_test_access::set_endpoint(*manager3);
    NodeSession_recovery_test_access::connect(*session3, std::make_shared<MockConnection>());
    NodeSession_recovery_test_access::drop_primary(*session3);
    io3->poll();
    assert(Worker_manager_recovery_test_access::phase(*manager3) == RecoveryPhase::RECONNECTING);
    assert(Worker_manager_recovery_test_access::connection_retries(*manager3) == 1);
    assert(Worker_manager_recovery_test_access::retry_scheduled(*manager3));
    Worker_manager_recovery_test_access::transport_lost(*manager3, "primary_socket_closed");
    assert(Worker_manager_recovery_test_access::connection_retries(*manager3) == 1);
    manager3->stop();

    // Review: RECONNECTING after the connect callback is not itself an outstanding
    // retry. A later socket loss must schedule a new reconnect.
    auto io4 = std::make_shared<asio::io_context>();
    auto manager4 = std::make_shared<Worker_manager>(
        io4, config, std::make_shared<chrono::Timer_factory>(io4), nullptr);
    Worker_manager_recovery_test_access::set_endpoint(*manager4);
    Worker_manager_recovery_test_access::enter_reconnecting(*manager4);
    assert(Worker_manager_recovery_test_access::phase(*manager4) == RecoveryPhase::RECONNECTING);
    assert(!Worker_manager_recovery_test_access::retry_scheduled(*manager4));
    Worker_manager_recovery_test_access::transport_lost(*manager4, "primary_socket_closed");
    assert(Worker_manager_recovery_test_access::connection_retries(*manager4) == 1);
    assert(Worker_manager_recovery_test_access::retry_scheduled(*manager4));
    Worker_manager_recovery_test_access::transport_lost(*manager4, "primary_socket_closed");
    assert(Worker_manager_recovery_test_access::connection_retries(*manager4) == 1);
    manager4->stop();

    // Review: a solved block that arrives after the session is gone must be
    // retained, not dropped at the authentication guard.
    auto io5 = std::make_shared<asio::io_context>();
    auto manager5 = std::make_shared<Worker_manager>(
        io5, config, std::make_shared<chrono::Timer_factory>(io5), nullptr);
    auto solved = std::make_unique<Block_data>();
    solved->nHeight = 424242;
    solved->nNonce = 0xabcdu;
    solved->nChannel = 2;
    Worker_manager_recovery_test_access::accept_solved(*manager5, std::move(solved));
    assert(Worker_manager_recovery_test_access::has_pending_submit(*manager5));
    assert(Worker_manager_recovery_test_access::pending_height(*manager5) == 424242);
    assert(Worker_manager_recovery_test_access::pending_nonce(*manager5) == 0xabcdu);
    manager5->stop();

    config.set_connection_retry_interval(0);
    config.set_colin_enabled(false);
    test_stalled_connect_recovery(config);
    test_pending_submit_expiry(config);
}
