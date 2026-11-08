#include "protocol/get_block_control.hpp"
#include "protocol/solo.hpp"
#include "network/connection.hpp"

#include <asio/io_context.hpp>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <memory>
#include <vector>

using namespace nexusminer;
using namespace nexusminer::protocol;

namespace nexusminer::protocol {
struct GetBlockControlHarness {
    static void authenticate(Solo& solo) { solo.m_authenticated = true; }
    static void mark_submit_pending(Solo& solo) { solo.m_submit_result_gate.mark_pending(); }
    static bool submit_pending(const Solo& solo) { return solo.m_submit_result_gate.has_pending(); }
    static bool get_block_pending(const Solo& solo) { return solo.m_pending_get_block.active; }
    static int64_t retry_at(const Solo& solo) { return solo.m_get_block_retry_at_ms.load(); }
    static uint64_t retry_generation(const Solo& solo) { return solo.m_get_block_control_generation.load(); }
    static void expire_retry(Solo& solo) { solo.m_get_block_retry_at_ms.store(1); }
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
    assert(GetBlockControlHarness::submit_pending(*solo));
    assert(!solo->get_work(GetBlockReason::RECOVERY_FORCED));
    assert(solo->get_last_get_block_request_status() == Solo::GetBlockRequestStatus::NODE_RETRY_DELAY);
    io->run_for(std::chrono::milliseconds(100));
    assert(retries == 0);
    io->restart();
    io->run_for(std::chrono::seconds(6));
    assert(retries == 1);
    assert(GetBlockControlHarness::get_block_pending(*solo));
    assert(solo->get_work(GetBlockReason::RECOVERY_FORCED));

    solo->process_messages(parse_frame(lane, {1, 1, 0, 0, 0, 0, 0xEA, 0x60}), connection);
    assert(!solo->get_work(GetBlockReason::RECOVERY_FORCED));
    const auto rate_limit_deadline = GetBlockControlHarness::retry_at(*solo);
    solo->process_messages(packet, connection);
    assert(GetBlockControlHarness::retry_at(*solo) >= rate_limit_deadline);
    solo->process_messages(parse_frame(lane, {1, 2, 0, 0, 0, 0, 0, 0}), connection);
    assert(GetBlockControlHarness::retry_at(*solo) >= rate_limit_deadline);
    assert(!solo->get_work(GetBlockReason::RECOVERY_FORCED));
    solo->reset();
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
    int retries = 0;
    solo->set_block_handler([&](::LLP::CBlock, std::uint32_t) { ++feeds; return true; });
    solo->set_recovery_initiated_handler([&](GetBlockReason) { ++retries; return true; });
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
    assert(GetBlockControlHarness::retry_generation(*solo) > generation);
    assert(GetBlockControlHarness::retry_at(*solo) == deadline);
    assert(solo->is_get_block_retry_deferred());
    auto work = solo->get_work(GetBlockReason::RECOVERY_FORCED);
    assert(!work);
    assert(solo->get_last_get_block_request_status() == Solo::GetBlockRequestStatus::NODE_RETRY_DELAY);

    solo->process_messages(parse_frame(lane, {1, 4, 0, 0, 0, 0, 3, 0xE8}), connection);
    const auto reconfirmation_generation = GetBlockControlHarness::retry_generation(*solo);
    solo->process_messages(packet, connection);
    assert(feeds == 1);
    assert(GetBlockControlHarness::retry_generation(*solo) > reconfirmation_generation);
    io->run_for(std::chrono::seconds(6));
    assert(retries == 0);

    solo->process_messages(parse_frame(lane, {1, 4, 0, 0, 0, 9, 0x27, 0xC0}), connection);
    assert(solo->is_get_block_retry_deferred());
    GetBlockControlHarness::expire_retry(*solo);
    assert(!solo->is_get_block_retry_deferred());
    solo->reset();
    assert(!solo->is_get_block_retry_deferred());
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
        test_template_acceptance_cancels_retry(lane, Packet::BLOCK_DATA);
        test_template_acceptance_cancels_retry(lane, Packet::GET_BLOCK);
    }
}
