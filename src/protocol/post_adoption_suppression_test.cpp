/**
 * @file post_adoption_suppression_test.cpp
 * @brief Verifies GET_BLOCK post-adoption suppression in Solo::request_and_queue_get_block().
 */

#include "protocol/solo.hpp"
#include "network/connection.hpp"
#include "spdlog/spdlog.h"
#include "spdlog/sinks/ringbuffer_sink.h"

#include <asio/io_context.hpp>
#include <cassert>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace nexusminer;
using namespace nexusminer::protocol;

namespace {

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

void print_test_result(const char* name, bool passed)
{
    ++tests_run;
    if (passed) {
        ++tests_passed;
        std::cout << "  [PASS] " << name << '\n';
    } else {
        ++tests_failed;
        std::cout << "  [FAIL] " << name << '\n';
    }
}

class MockConnection final : public network::Connection {
public:
    MockConnection()
        : m_remote(network::Transport_protocol::tcp, "127.0.0.1", 9323)
        , m_local(network::Transport_protocol::tcp, "127.0.0.1", 0)
    {}

    network::Endpoint const& remote_endpoint() const override { return m_remote; }
    network::Endpoint const& local_endpoint() const override { return m_local; }

    bool transmit(network::Shared_payload tx_buffer) override
    {
        if (!tx_buffer || tx_buffer->empty()) {
            return false;
        }
        ++m_transmit_count;
        return true;
    }

    void close() override {}
    ProtocolLane get_protocol_lane() const override { return ProtocolLane::STATELESS; }

    int transmit_count() const { return m_transmit_count; }

private:
    network::Endpoint m_remote;
    network::Endpoint m_local;
    int m_transmit_count{0};
};

bool has_log_message(const std::shared_ptr<spdlog::sinks::ringbuffer_sink_mt>& ring,
                     const std::string& needle)
{
    const auto msgs = ring->last_formatted(256);
    for (const auto& msg : msgs) {
        if (msg.find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}

} // namespace

namespace nexusminer {
namespace protocol {

struct PostAdoptionSuppressionHarness {
    static void set_authenticated(Solo& solo, bool v) { solo.m_authenticated = v; }
    static void set_reward_bound(Solo& solo, bool v) { solo.m_reward_bound = v; }
    static void set_last_adopted_now(Solo& solo) { solo.m_last_template_adopted_at = std::chrono::steady_clock::now(); }
    static void set_last_adopted(Solo& solo, std::chrono::steady_clock::time_point tp) { solo.m_last_template_adopted_at = tp; }
    static std::chrono::milliseconds window() { return Solo::POST_ADOPTION_SUPPRESSION_WINDOW; }
    static bool request(Solo& solo, const std::shared_ptr<network::Connection>& conn, GetBlockReason reason, const char* context)
    {
        return solo.request_and_queue_get_block(conn, reason, context);
    }
};

} // namespace protocol
} // namespace nexusminer

namespace {

std::unique_ptr<Solo> make_solo()
{
    auto io = std::make_shared<asio::io_context>();
    // Session context/stats are intentionally null here: this test targets the
    // request_and_queue_get_block chokepoint logic in isolation.
    auto solo = std::make_unique<Solo>(static_cast<uint8_t>(mining::CHANNEL_PRIME), nullptr, nullptr, io);
    solo->set_protocol_lane(ProtocolLane::STATELESS);
    PostAdoptionSuppressionHarness::set_authenticated(*solo, true);
    PostAdoptionSuppressionHarness::set_reward_bound(*solo, true);
    return solo;
}

void test_non_bypass_reasons_suppressed(const std::shared_ptr<spdlog::sinks::ringbuffer_sink_mt>& ring)
{
    std::cout << "\nTest 1: non-bypass reasons are suppressed inside post-adoption window\n";
    auto conn = std::make_shared<MockConnection>();
    auto solo = make_solo();

    const std::vector<GetBlockReason> reasons = {
        GetBlockReason::PUSH_TIP_MOVED,
        GetBlockReason::PUSH_SAME_HEIGHT_TIP,
        GetBlockReason::HEALTH_CHANNEL_STALE,
        GetBlockReason::HEALTH_CHANNEL_ADVANCE,
        GetBlockReason::HEALTH_STALE_SUPPRESSED,
        GetBlockReason::TEMPLATE_AGE_WARNING,
        GetBlockReason::TEMPLATE_AGE_EMERGENCY,
        GetBlockReason::TEMPLATE_AGE_DEFERRED,
        GetBlockReason::VALIDATION_FAILURE,
        GetBlockReason::HEIGHT_DRIFT,
        GetBlockReason::GET_ROUND_STALE,
        GetBlockReason::GET_ROUND_NO_TEMPLATE,
        GetBlockReason::GET_ROUND_HEIGHT_PARITY,
        GetBlockReason::TEMPLATE_FEED_FAILURE
    };

    bool all_suppressed = true;
    for (auto reason : reasons) {
        solo->clear_get_block_pending();
        solo->reset_get_block_dedup_state();
        PostAdoptionSuppressionHarness::set_last_adopted_now(*solo);
        const bool sent = PostAdoptionSuppressionHarness::request(
            *solo, conn, reason, "[post_adoption_test] non_bypass");
        all_suppressed = all_suppressed && !sent;
    }

    print_test_result("All non-bypass reasons return false inside suppression window", all_suppressed);
    print_test_result("Suppression log emitted",
                      has_log_message(ring, "Suppressed: template adopted"));
}

void test_bypass_reasons_allowed()
{
    std::cout << "\nTest 2: bypass reasons are allowed inside post-adoption window\n";
    auto conn = std::make_shared<MockConnection>();
    auto solo = make_solo();

    const std::vector<GetBlockReason> reasons = {
        GetBlockReason::RECOVERY_FORCED,
        GetBlockReason::RECOVERY_TIMER,
        GetBlockReason::BLOCK_ACCEPTED,
        GetBlockReason::HEALTH_NO_TEMPLATE,
        GetBlockReason::BLOCK_REJECTED,
        GetBlockReason::SESSION_REAUTH,
        GetBlockReason::INITIAL_REQUEST
    };

    bool all_sent = true;
    for (auto reason : reasons) {
        solo->clear_get_block_pending();
        solo->reset_get_block_dedup_state();
        PostAdoptionSuppressionHarness::set_last_adopted_now(*solo);
        const bool sent = PostAdoptionSuppressionHarness::request(
            *solo, conn, reason, "[post_adoption_test] bypass");
        all_sent = all_sent && sent;
    }

    print_test_result("All bypass reasons return true inside suppression window", all_sent);
    print_test_result("Bypass reasons queued expected number of GET_BLOCK requests",
                      conn->transmit_count() == static_cast<int>(reasons.size()));
}

void test_non_bypass_allowed_after_window()
{
    std::cout << "\nTest 3: non-bypass reason allowed after suppression window\n";
    auto conn = std::make_shared<MockConnection>();
    auto solo = make_solo();
    solo->clear_get_block_pending();
    solo->reset_get_block_dedup_state();

    const auto expired_adopt_time = std::chrono::steady_clock::now() -
        PostAdoptionSuppressionHarness::window() - std::chrono::milliseconds(10);
    PostAdoptionSuppressionHarness::set_last_adopted(*solo, expired_adopt_time);

    const bool sent = PostAdoptionSuppressionHarness::request(
        *solo, conn, GetBlockReason::GET_ROUND_STALE, "[post_adoption_test] after_window");
    print_test_result("Non-bypass reason returns true after suppression window", sent);
}

} // namespace

int main()
{
    auto ring_sink = std::make_shared<spdlog::sinks::ringbuffer_sink_mt>(512);
    auto logger = std::make_shared<spdlog::logger>("post_adoption_suppression_test_logger", ring_sink);
    spdlog::register_logger(logger);
    auto prev_default = spdlog::default_logger();
    spdlog::set_default_logger(logger);

    test_non_bypass_reasons_suppressed(ring_sink);
    test_bypass_reasons_allowed();
    test_non_bypass_allowed_after_window();

    spdlog::set_default_logger(prev_default);
    spdlog::drop("post_adoption_suppression_test_logger");

    std::cout << "\n=== post_adoption_suppression_test Summary ===\n";
    std::cout << "Tests run:    " << tests_run << '\n';
    std::cout << "Tests passed: " << tests_passed << '\n';
    std::cout << "Tests failed: " << tests_failed << '\n';

    return tests_failed == 0 ? 0 : 1;
}
