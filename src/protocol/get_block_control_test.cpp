#include "protocol/get_block_control.hpp"
#include "protocol/solo.hpp"
#include "network/connection.hpp"
#include "config/config.hpp"
#include "stats/stats_collector.hpp"

#include <asio/io_context.hpp>
#include <spdlog/sinks/ostream_sink.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <memory>
#include <sstream>
#include <vector>

using namespace nexusminer;
using namespace nexusminer::protocol;

namespace nexusminer::protocol {
struct GetBlockControlHarness {
    static void authenticate(Solo& solo) { solo.m_authenticated = true; }
    static bool locally_authenticated(const Solo& solo) { return solo.m_authenticated; }
    static void sync_session(Solo& solo) { solo.refresh_cached_session_state("GetBlockControlHarness"); }
    static void mark_submit_pending(Solo& solo) { solo.m_submit_result_gate.mark_pending(); }
    static bool submit_pending(const Solo& solo) { return solo.m_submit_result_gate.has_pending(); }
    static bool get_block_pending(const Solo& solo) { return solo.m_pending_get_block.active; }
    static int64_t retry_at(const Solo& solo) { return solo.m_get_block_retry_at_ms.load(); }
    static uint64_t retry_generation(const Solo& solo) { return solo.m_get_block_control_generation.load(); }
    static void expire_retry(Solo& solo) { solo.m_get_block_retry_at_ms.store(1); }
    static void expire_node_wait(Solo& solo) { solo.m_node_template_wait_until_ms.store(1); }
};
}

class MockConnection final : public network::Connection {
public:
    explicit MockConnection(ProtocolLane lane)
        : m_lane(lane), m_remote(network::Transport_protocol::tcp, "127.0.0.1",
                                 lane == ProtocolLane::STATELESS ? 9323 : 9325),
          m_local(network::Transport_protocol::tcp, "127.0.0.1", 0) {}
    network::Endpoint const& remote_endpoint() const override { return m_remote; }
    network::Endpoint const& local_endpoint() const override { return m_local; }
    bool transmit(network::Shared_payload p) override { return p && !p->empty(); }
    void close() override {}
    ProtocolLane get_protocol_lane() const override { return m_lane; }
private:
    ProtocolLane m_lane;
    network::Endpoint m_remote, m_local;
};

Packet parse_frame(ProtocolLane lane, std::vector<uint8_t> payload)
{
    network::Payload wire = lane == ProtocolLane::STATELESS
        ? network::Payload{0xD0, 0xC9, 0, 0, 0, 8}
        : network::Payload{0xC9, 0, 0, 0, 8};
    wire.insert(wire.end(), payload.begin(), payload.end());
    std::size_t consumed = 0;
    ParseResult result;
    auto packet = extract_packet_from_buffer_with_result(
        std::make_shared<network::Payload>(wire), consumed, 0, lane, result);
    assert(result == ParseResult::SUCCESS && consumed == wire.size());
    return packet;
}

void test_submit_rejection(ProtocolLane lane, const std::vector<uint8_t>& reason,
                           const std::string& expected)
{
    std::ostringstream logs;
    auto sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(logs);
    auto logger = std::make_shared<spdlog::logger>("rejection-test", sink);
    auto previous = spdlog::default_logger();
    spdlog::set_default_logger(logger);
    {
        config::Config config(logger);
        auto stats = std::make_shared<stats::Hash_collector>(config);
        auto io = std::make_shared<asio::io_context>();
        auto solo = std::make_shared<Solo>(1, stats, nullptr, io);
        auto connection = std::make_shared<MockConnection>(lane);
        solo->set_protocol_lane(lane);
        GetBlockControlHarness::authenticate(*solo);
        GetBlockControlHarness::mark_submit_pending(*solo);

        network::Payload wire = lane == ProtocolLane::STATELESS
            ? network::Payload{0xD0, 0xC9}
            : network::Payload{0xC9};
        wire.insert(wire.end(), {0, 0, 0, static_cast<uint8_t>(reason.size())});
        wire.insert(wire.end(), reason.begin(), reason.end());
        std::size_t consumed = 0;
        ParseResult result;
        auto packet = extract_packet_from_buffer_with_result(
            std::make_shared<network::Payload>(wire), consumed, 0, lane, result);
        assert(result == ParseResult::SUCCESS && consumed == wire.size());
        solo->process_messages(packet, connection);
        logger->flush();
        assert(logs.str().find("reason=" + expected) != std::string::npos);
        assert(stats->get_global_stats().m_rejected_blocks == 1);
        assert(!GetBlockControlHarness::submit_pending(*solo));
        if (reason.empty()) {
            assert(logs.str().find("check node logs") != std::string::npos);
            assert(logs.str().find("most common cause") == std::string::npos);
        } else {
            assert(logs.str().find(fmt::format("code=0x{:02x}", reason.front())) != std::string::npos);
        }
    }
    spdlog::set_default_logger(previous);
}

void test_lane(ProtocolLane lane)
{
    auto io = std::make_shared<asio::io_context>();
    auto solo = std::make_shared<Solo>(1, nullptr, nullptr, io);
    solo->set_protocol_lane(lane);
    GetBlockControlHarness::authenticate(*solo);
    auto connection = std::make_shared<MockConnection>(lane);
    int retries = 0;
    solo->set_recovery_initiated_handler([&](GetBlockReason reason) {
        auto work = solo->get_work(reason);
        assert(work && connection->transmit(work));
        solo->mark_get_block_pending(reason);
        ++retries;
        return true;
    });
    solo->mark_get_block_pending(GetBlockReason::INITIAL_REQUEST);
    GetBlockControlHarness::mark_submit_pending(*solo);
    solo->process_messages(parse_frame(lane, {2, 4, 0, 0, 0, 0, 3, 0xE8}), connection);
    assert(GetBlockControlHarness::get_block_pending(*solo));
    assert(GetBlockControlHarness::submit_pending(*solo));
    auto packet = parse_frame(lane, {1, 4, 0, 0, 0, 0, 3, 0xE8});
    solo->process_messages(packet, connection);
    assert(!GetBlockControlHarness::get_block_pending(*solo));
    assert(solo->is_node_template_wait_active());
    assert(GetBlockControlHarness::submit_pending(*solo));
    assert(!solo->get_work(GetBlockReason::RECOVERY_FORCED));
    assert(solo->get_last_get_block_request_status() == Solo::GetBlockRequestStatus::NODE_RETRY_DELAY);
    io->run_for(std::chrono::milliseconds(100));
    assert(retries == 0);
    io->restart();
    io->run_for(std::chrono::seconds(6));
    assert(retries == 1);
    assert(solo->is_node_template_wait_active());
    GetBlockControlHarness::expire_node_wait(*solo);
    assert(!solo->is_node_template_wait_active());
    assert(GetBlockControlHarness::get_block_pending(*solo));
    assert(solo->get_work(GetBlockReason::RECOVERY_FORCED));

    solo->process_messages(parse_frame(lane, {1, 1, 0, 0, 0, 0, 0xEA, 0x60}), connection);
    assert(!solo->get_work(GetBlockReason::RECOVERY_FORCED));
    const auto rate_limit_deadline = GetBlockControlHarness::retry_at(*solo);
    const auto rate_limit_generation = GetBlockControlHarness::retry_generation(*solo);
    solo->process_messages(packet, connection);
    assert(GetBlockControlHarness::retry_at(*solo) == rate_limit_deadline);
    assert(GetBlockControlHarness::retry_generation(*solo) == rate_limit_generation);
    solo->process_messages(parse_frame(lane, {1, 2, 0, 0, 0, 0, 0, 0}), connection);
    assert(GetBlockControlHarness::retry_at(*solo) == rate_limit_deadline);
    assert(solo->is_authenticated());
    assert(!solo->get_work(GetBlockReason::RECOVERY_FORCED));
    solo->reset();
    assert(!solo->is_node_template_wait_active());
    io->restart();
    io->poll();
    assert(retries == 1);
    GetBlockControlHarness::authenticate(*solo);
    solo->mark_get_block_pending(GetBlockReason::INITIAL_REQUEST);
    solo->process_messages(parse_frame(lane, {1, 2, 0, 0, 0, 0, 0, 0}), connection);
    assert(!GetBlockControlHarness::get_block_pending(*solo));
    assert(GetBlockControlHarness::retry_at(*solo) == 0);
}

Packet template_packet(ProtocolLane lane, uint8_t opcode)
{
    auto data = std::make_shared<network::Payload>(228, 0);
    auto write_u32 = [&](std::size_t offset, uint32_t value) {
        for (int i = 0; i < 4; ++i) {
            (*data)[offset + i] = static_cast<uint8_t>(value >> (24 - 8 * i));
        }
    };
    write_u32(0, 100);
    write_u32(4, 50);
    write_u32(8, 0x03a4166d);
    write_u32(12, 8);
    std::fill(data->begin() + 16, data->begin() + 144, 0xAA);
    std::fill(data->begin() + 144, data->begin() + 208, 0xBB);
    write_u32(208, 1);
    write_u32(212, 101);
    write_u32(216, 0x03a4166d);
    return lane == ProtocolLane::STATELESS
        ? Packet(static_cast<uint16_t>(0xD000 | opcode), data)
        : Packet(opcode, data);
}

void test_template_acceptance_cancels_retry(ProtocolLane lane, uint8_t opcode)
{
    auto io = std::make_shared<asio::io_context>();
    auto solo = std::make_shared<Solo>(1, nullptr, nullptr, io);
    solo->set_protocol_lane(lane);
    GetBlockControlHarness::authenticate(*solo);
    auto connection = std::make_shared<MockConnection>(lane);
    int feeds = 0;
    int completed_feeds = 0;
    int retries = 0;
    solo->set_block_handler([&](const ::LLP::CBlock&, std::uint32_t) { ++feeds; return true; });
    solo->get_template_interface()->set_template_fed_handler([&] {
        assert(solo->get_template_interface()->has_valid_template());
        ++completed_feeds;
    });
    solo->set_recovery_initiated_handler([&](GetBlockReason) { ++retries; return true; });
    solo->mark_get_block_pending(GetBlockReason::INITIAL_REQUEST);
    solo->process_messages(parse_frame(lane, {1, 4, 0, 0, 0, 0, 3, 0xE8}), connection);
    const auto deadline = GetBlockControlHarness::retry_at(*solo);
    const auto generation = GetBlockControlHarness::retry_generation(*solo);
    assert(solo->is_get_block_retry_deferred());

    auto invalid = template_packet(lane, opcode);
    invalid.m_data = std::make_shared<network::Payload>(1, 0);
    invalid.m_length = 1;
    solo->process_messages(invalid, connection);
    assert(feeds == 0);
    assert(GetBlockControlHarness::retry_generation(*solo) == generation);
    retries = 0;

    auto packet = template_packet(lane, opcode);
    solo->process_messages(packet, connection);
    assert(feeds == 1);
    assert(completed_feeds == 1);
    assert(!solo->is_node_template_wait_active());
    assert(GetBlockControlHarness::retry_generation(*solo) > generation);
    assert(GetBlockControlHarness::retry_at(*solo) == deadline);
    assert(solo->is_get_block_retry_deferred());
    auto work = solo->get_work(GetBlockReason::RECOVERY_FORCED);
    assert(!work);
    assert(solo->get_last_get_block_request_status() == Solo::GetBlockRequestStatus::NODE_RETRY_DELAY);

    const auto template_generation = GetBlockControlHarness::retry_generation(*solo);
    solo->process_messages(parse_frame(lane, {1, 4, 0, 0, 0, 0, 3, 0xE8}), connection);
    assert(GetBlockControlHarness::retry_at(*solo) == deadline);
    const auto reconfirmation_generation = GetBlockControlHarness::retry_generation(*solo);
    assert(reconfirmation_generation == template_generation);
    solo->mark_get_block_pending(GetBlockReason::INITIAL_REQUEST);
    solo->process_messages(parse_frame(lane, {1, 4, 0, 0, 0, 0, 3, 0xE8}), connection);
    const auto active_generation = GetBlockControlHarness::retry_generation(*solo);
    assert(active_generation > reconfirmation_generation);
    solo->process_messages(packet, connection);
    assert(feeds == 1);
    assert(completed_feeds == 1);
    assert(GetBlockControlHarness::retry_generation(*solo) > active_generation);
    const auto accepted_generation = GetBlockControlHarness::retry_generation(*solo);
    const auto accepted_deadline = GetBlockControlHarness::retry_at(*solo);
    GetBlockControlHarness::mark_submit_pending(*solo);
    solo->process_messages(parse_frame(lane, {1, 4, 0, 0, 0, 9, 0x27, 0xC0}), connection);
    assert(GetBlockControlHarness::retry_generation(*solo) == accepted_generation);
    assert(GetBlockControlHarness::retry_at(*solo) == accepted_deadline);
    assert(GetBlockControlHarness::submit_pending(*solo));
    assert(solo->get_template_interface()->has_valid_template());
    io->run_for(std::chrono::seconds(6));
    assert(retries == 0);

    solo->process_messages(parse_frame(lane, {1, 4, 0, 0, 0, 9, 0x27, 0xC0}), connection);
    assert(solo->is_get_block_retry_deferred());
    GetBlockControlHarness::expire_retry(*solo);
    assert(!solo->is_get_block_retry_deferred());
    solo->reset();
    assert(!solo->is_get_block_retry_deferred());
}

void test_rejected_template_retry(ProtocolLane lane, uint8_t opcode)
{
    auto io = std::make_shared<asio::io_context>();
    auto solo = std::make_shared<Solo>(1, nullptr, nullptr, io);
    solo->set_protocol_lane(lane);
    GetBlockControlHarness::authenticate(*solo);
    auto connection = std::make_shared<MockConnection>(lane);
    int feeds = 0;
    int completions = 0;
    solo->set_block_handler([&](const ::LLP::CBlock&, std::uint32_t) {
        return ++feeds > 1;
    });
    auto* templates = solo->get_template_interface();
    assert(templates);
    templates->set_template_fed_handler([&] { ++completions; });
    solo->set_recovery_initiated_handler([](GetBlockReason) { return true; });
    solo->mark_get_block_pending(GetBlockReason::INITIAL_REQUEST);
    solo->process_messages(parse_frame(lane, {1, 4, 0, 0, 0, 0, 3, 0xE8}), connection);
    const auto generation = GetBlockControlHarness::retry_generation(*solo);
    auto packet = template_packet(lane, opcode);
    solo->process_messages(packet, connection);
    assert(feeds == 1 && completions == 0);
    assert(!templates->has_valid_template());
    assert(GetBlockControlHarness::retry_generation(*solo) == generation);

    // The same candidate must reach the worker again, not report guard suppression as success.
    solo->process_messages(packet, connection);
    assert(feeds == 2 && completions == 1);
    assert(templates->has_valid_template());
    assert(GetBlockControlHarness::retry_generation(*solo) > generation);
}

void test_first_template_after_node_startup(ProtocolLane lane)
{
    // Block deserialization retains its first logger across test cases.
    static std::ostringstream logs;
    logs.str("");
    logs.clear();
    auto sink = std::make_shared<spdlog::sinks::ostream_sink_mt>(logs);
    auto logger = std::make_shared<spdlog::logger>("startup-test", sink);
    auto previous = spdlog::default_logger();
    spdlog::set_default_logger(logger);
    {
        config::Config config(logger);
        auto stats = std::make_shared<stats::Hash_collector>(config);
        auto io = std::make_shared<asio::io_context>();
        auto solo = std::make_shared<Solo>(1, stats, nullptr, io);
        auto connection = std::make_shared<MockConnection>(lane);
        solo->set_protocol_lane(lane);
        GetBlockControlHarness::authenticate(*solo);
        int feeds = 0;
        solo->set_block_handler([&](const ::LLP::CBlock&, uint32_t) {
            ++feeds;
            return true;
        });
        solo->set_recovery_initiated_handler([&](GetBlockReason reason) {
            auto work = solo->get_work(reason);
            assert(work && connection->transmit(work));
            solo->mark_get_block_pending(reason);
            return true;
        });

        solo->mark_get_block_pending(GetBlockReason::INITIAL_REQUEST);
        solo->process_messages(parse_frame(lane, {1, 4, 0, 0, 0, 0, 3, 0xE8}), connection);
        assert(feeds == 0);
        assert(stats->get_global_stats().m_rejected_blocks == 0);
        assert(solo->is_authenticated());
        assert(!GetBlockControlHarness::submit_pending(*solo));
        assert(solo->is_node_template_wait_active());
        logger->flush();
        assert(logs.str().find("RECEIVED PACKET: GET_BLOCK_CONTROL_RESPONSE") != std::string::npos);
        assert(logs.str().find("no submitted block was rejected") != std::string::npos);

        io->run_for(std::chrono::seconds(6));
        assert(GetBlockControlHarness::get_block_pending(*solo));
        solo->process_messages(template_packet(lane, Packet::BLOCK_DATA), connection);
        assert(feeds == 1);
        assert(solo->get_template_interface()->has_valid_template());
        assert(!solo->is_node_template_wait_active());
        assert(!GetBlockControlHarness::get_block_pending(*solo));
        assert(stats->get_global_stats().m_rejected_blocks == 0);
    }
    spdlog::set_default_logger(previous);
}

void test_session_controls(ProtocolLane lane, bool authoritative)
{
    for (uint8_t reason : {2, 3, 9}) {
        auto io = std::make_shared<asio::io_context>();
        auto context = authoritative
            ? std::make_shared<NodeSessionContext>(std::make_shared<SessionManager>(24, nullptr))
            : nullptr;
        auto solo = std::make_shared<Solo>(1, nullptr, context, io);
        solo->set_protocol_lane(lane);
        if (context) {
            context->start_session(SessionId(12345u));
        }
        GetBlockControlHarness::authenticate(*solo);
        GetBlockControlHarness::sync_session(*solo);
        auto connection = std::make_shared<MockConnection>(lane);
        int reauths = 0;
        int retries = 0;
        solo->set_session_expired_handler([&] { ++reauths; });
        solo->set_recovery_initiated_handler([&](GetBlockReason) { ++retries; return true; });
        solo->mark_get_block_pending(GetBlockReason::INITIAL_REQUEST);
        solo->process_messages(parse_frame(lane, {1, 4, 0, 0, 0, 0, 3, 0xE8}), connection);
        assert(solo->is_get_block_retry_deferred());
        solo->mark_get_block_pending(GetBlockReason::RECOVERY_FORCED);
        solo->process_messages(parse_frame(lane, {1, reason, 0, 0, 0, 0, 0, 0}), connection);
        const bool session_error = reason != 9;
        assert(reauths == (session_error ? 1 : 0));
        assert(solo->is_authenticated() == !session_error);
        assert(GetBlockControlHarness::locally_authenticated(*solo) == !session_error);
        assert(!GetBlockControlHarness::get_block_pending(*solo));
        assert(GetBlockControlHarness::retry_at(*solo) == 0);
        if (context && session_error) {
            assert(context->get_session_id() == SessionId{});
        }
        solo->process_messages(parse_frame(lane, {1, reason, 0, 0, 0, 0, 0, 0}), connection);
        assert(reauths == (session_error ? 1 : 0));
        io->run_for(std::chrono::seconds(6));
        assert(retries == 0);
    }
}

int main()
{
    auto valid = GetBlockControl::parse({1, 4, 0, 0, 0, 0, 3, 0xE8});
    assert(valid && valid->retryable() && valid->retry_after_ms == 1000);
    assert(valid->retry_delay() == std::chrono::seconds(5));
    auto rate = GetBlockControl::parse({1, 1, 0, 0, 0, 0, 0xEA, 0x60});
    assert(rate && rate->retry_delay() == std::chrono::seconds(60));
    auto invalid_session = GetBlockControl::parse({1, 2, 0, 0, 0, 0, 0, 0});
    assert(invalid_session && !invalid_session->retryable());
    assert(!GetBlockControl::parse({2, 4, 0, 0, 0, 0, 0, 0}));
    assert(!GetBlockControl::parse({1, 10, 0, 0, 0, 0, 0, 0}));
    assert(!GetBlockControl::parse({1, 4, 1, 0, 0, 0, 0, 0}));
    assert(!GetBlockControl::parse({1, 4}));
    test_lane(ProtocolLane::STATELESS);
    test_lane(ProtocolLane::LEGACY);
    for (auto lane : {ProtocolLane::STATELESS, ProtocolLane::LEGACY}) {
        test_first_template_after_node_startup(lane);
        test_session_controls(lane, false);
        test_session_controls(lane, true);
        test_submit_rejection(lane, {}, "NONE");
        test_submit_rejection(lane, {1}, "STALE");
        test_submit_rejection(lane, {2}, "INVALID_POW");
        test_submit_rejection(lane, {3}, "INVALID_SIG");
        test_submit_rejection(lane, {4}, "DUPLICATE");
        test_submit_rejection(lane, {5}, "FORK");
        test_submit_rejection(lane, {0x06}, "SESSION_INVALID");
        test_submit_rejection(lane, {0x07}, "REWARD_NOT_BOUND");
        test_submit_rejection(lane, {0x08}, "RATE_LIMITED_OR_STALE_TIMESTAMP");
        test_submit_rejection(lane, {0x09}, "UNKNOWN_TEMPLATE");
        test_submit_rejection(lane, {0x0a}, "LOCAL_TEMPLATE_REJECT");
        test_submit_rejection(lane, {0x0b}, "MALFORMED_PRODUCER_OR_DECRYPTION_FAILED");
        test_submit_rejection(lane, {0x0c}, "ENCRYPTION_REQUIRED_OR_SIGNATURE_FAILED");
        test_submit_rejection(lane, {0x0d}, "NO_SESSION_KEY");
        test_submit_rejection(lane, {0x0e}, "WORK_REVOKED");
        test_submit_rejection(lane, {0x0f}, "PACKET_TOO_SMALL");
        test_submit_rejection(lane, {0x11}, "VALIDATION_FAILED");
        test_submit_rejection(lane, {0x12}, "ACCEPTANCE_FAILED");
        test_submit_rejection(lane, {0x13}, "RATE_LIMITED");
        test_submit_rejection(lane, {0x14}, "SESSION_INCONSISTENT");
        test_submit_rejection(lane, {0x15}, "PACKET_TOO_LARGE");
        test_submit_rejection(lane, {0x16}, "CANDIDATE_PREPARATION_FAILED");
        test_submit_rejection(lane, {0x17}, "WRONG_BLOCK_TYPE");
        test_submit_rejection(lane, {0x18}, "MALFORMED_SUBMISSION");
        test_submit_rejection(lane, {0xff}, "INTERNAL_ERROR");
        test_submit_rejection(lane, {0xfe}, "0xfe");
        test_template_acceptance_cancels_retry(lane, Packet::BLOCK_DATA);
        test_template_acceptance_cancels_retry(lane, Packet::GET_BLOCK);
        test_rejected_template_retry(lane, Packet::BLOCK_DATA);
        test_rejected_template_retry(lane, Packet::GET_BLOCK);
    }
}
