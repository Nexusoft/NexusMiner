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
};

struct NodeSession_recovery_test_access {
    static void connect(NodeSession& session, std::shared_ptr<network::Connection> connection) {
        session.m_primary_connection = std::move(connection);
        session.m_primary_connected.store(true);
    }
    static void restore_template_handler(NodeSession& session) {
        session.set_template_handler(session.m_template_handler);
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
    static void report_auth_failure(Solo& solo) {
        assert(solo.m_session_authenticated_handler);
        solo.m_session_authenticated_handler(SessionId{});
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
    void close() override {}
    ProtocolLane get_protocol_lane() const override { return ProtocolLane::STATELESS; }
private:
    network::Endpoint m_remote{network::Transport_protocol::tcp, "127.0.0.1", 9323};
    network::Endpoint m_local{network::Transport_protocol::tcp, "127.0.0.1", 0};
};

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

    solo->set_block_handler([](::LLP::CBlock, std::uint32_t) { return false; });
    std::fill(block->begin() + 16, block->begin() + 144, 0xcc);
    solo->mark_get_block_pending(protocol::GetBlockReason::RECOVERY_FORCED);
    solo->process_messages(template_response, connection);
    assert(Worker_manager_recovery_test_access::phase(*manager) == RecoveryPhase::WAITING_TEMPLATE);

    Worker_manager_recovery_test_access::start_old_recovery(*manager);
    manager->check_template_health();
    assert(Worker_manager_recovery_test_access::phase(*manager) == RecoveryPhase::RECONNECTING);
    assert(Worker_manager_recovery_test_access::workers(*manager) == 0);
    const auto retries_before_auth = Worker_manager_recovery_test_access::connection_retries(*manager);
    protocol::GetBlockControlHarness::report_auth_failure(*solo);
    assert(Worker_manager_recovery_test_access::workers(*manager) == 0);
    io->run_for(std::chrono::milliseconds(1300));
    assert(Worker_manager_recovery_test_access::connection_retries(*manager) > retries_before_auth);
    for (int i = 0; i < 6; ++i) {
        protocol::GetBlockControlHarness::report_auth_failure(*solo);
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
    protocol::GetBlockControlHarness::report_auth_failure(*solo);
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
}
