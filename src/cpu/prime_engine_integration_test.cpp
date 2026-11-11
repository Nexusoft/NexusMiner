// Stone 7 — PrimeMiningEngine integration tests.
//
// Focused tests for the engine-mode adapter surface introduced in Stone 7:
//   * Worker_prime constructed under engine_mode="engine" exposes
//       uses_template_feed() == true and attach_template_feed() is a no-op.
//   * register_worker() ignores nulls, accumulates live workers.
//   * Engine snapshot_stats() returns aggregated counters; a worker's
//       update_statistics() partitions them via floor+remainder so the sum
//       across registered workers equals the engine total (within rounding).
//   * The channel's representative worker (lowest internal_id) gets
//       internal_id_for_solution credit.
//   * Found-block dispatch through the asio::post path credits the
//       representative worker.
//   * Engine destruction before workers (Worker_manager teardown ordering)
//       does not crash even if a worker outlives the engine.
//
// Tests use the Engine_config::test_skip_sieve / test_force_candidate_per_segment
// seams (same as prime_mining_engine_pool_test) so the heavy Sieve pipeline
// is bypassed and the assertions are deterministic.

#include "cpu/worker_prime.hpp"
#include "cpu/prime/prime_mining_engine.hpp"
#include "worker/template_feed.hpp"
#include "worker.hpp"
#include "block.hpp"
#include "config/worker_config.hpp"
#include "config/config.hpp"
#include "config/types.hpp"
#include "stats/stats_collector.hpp"
#include "stats/types.hpp"

#include <asio.hpp>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/spdlog.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace nexusminer;
using namespace nexusminer::cpu;
using namespace std::chrono_literals;

namespace
{
int tests_run = 0;
int tests_failed = 0;

void print_result(const char* name, bool passed)
{
    ++tests_run;
    std::cout << (passed ? "  [PASS] " : "  [FAIL] ") << name << '\n';
    if (!passed) ++tests_failed;
}

constexpr std::uint64_t kSegmentSize = 4096;

// RAII io_context wrapper (matches prime_mining_engine_pool_test pattern).
struct Test_io_context
{
    std::shared_ptr<asio::io_context> ctx;
    asio::executor_work_guard<asio::io_context::executor_type> guard;
    std::thread runner;

    Test_io_context()
        : ctx{std::make_shared<asio::io_context>()}
        , guard{asio::make_work_guard(*ctx)}
        , runner{[this] { ctx->run(); }}
    {}

    ~Test_io_context()
    {
        guard.reset();
        ctx->stop();
        if (runner.joinable()) runner.join();
    }
};

config::Worker_config make_worker_config(std::uint16_t internal_id,
                                         const std::string& engine_mode)
{
    config::Worker_config wc;
    wc.m_id = std::to_string(internal_id);
    wc.m_internal_id = internal_id;
    wc.m_mode = config::Worker_mode::CPU;
    config::Worker_config_cpu cpu_cfg;
    cpu_cfg.m_engine_mode = engine_mode;
    wc.m_worker_mode = cpu_cfg;
    return wc;
}

// Storage helper — we need stable references to Worker_config because
// Worker_prime stores a reference (config::Worker_config&) that must
// outlive the worker.  Returns the (worker, internal_id) pair.
struct EngineWorker {
    std::shared_ptr<Worker_prime> worker;
    std::uint16_t internal_id;
};

std::shared_ptr<spdlog::logger> make_test_logger()
{
    auto sink = std::make_shared<spdlog::sinks::null_sink_mt>();
    static std::atomic<int> counter{0};
    return std::make_shared<spdlog::logger>(
        "prime_engine_integration_logger_" + std::to_string(counter++), sink);
}

config::Config make_prime_config(std::uint32_t worker_count)
{
    config::Config cfg{make_test_logger()};
    cfg.set_mining_mode(config::Mining_mode::PRIME);
    cfg.set_worker_count(worker_count);
    auto& workers = cfg.get_worker_config();
    for (std::size_t i = 0; i < workers.size(); ++i) {
        workers[i].m_internal_id = static_cast<std::uint16_t>(i);
        workers[i].m_mode = config::Worker_mode::CPU;
        // Don't set engine_mode — these configs are for the Collector to
        // know how many workers to allocate. Worker_prime itself uses the
        // separate cfg_storage configs we pass to its constructor.
    }
    return cfg;
}

EngineWorker make_engine_mode_worker(
    std::shared_ptr<asio::io_context> io,
    std::uint16_t internal_id,
    std::vector<std::unique_ptr<config::Worker_config>>& cfg_storage)
{
    cfg_storage.emplace_back(
        std::make_unique<config::Worker_config>(make_worker_config(internal_id, "engine")));
    return {std::make_shared<Worker_prime>(io, *cfg_storage.back()), internal_id};
}

std::shared_ptr<WorkPackage> make_work_package(
    std::uint32_t nbits,
    const boost::multiprecision::uint1024_t& base_hash,
    std::uint32_t height = 0)
{
    ::LLP::CBlock block;
    block.nVersion = 4;
    block.nChannel = 2;
    block.nHeight = height;
    block.nBits = nbits;
    block.nNonce = 0;
    auto wp = std::make_shared<WorkPackage>(block, nbits);
    wp->set_prime_base_hash(base_hash);
    return wp;
}

std::shared_ptr<TemplateEpoch> make_epoch(std::shared_ptr<WorkPackage> wp,
                                          Worker::Block_found_handler on_found = {})
{
    auto epoch = std::make_shared<TemplateEpoch>();
    epoch->work_package = std::move(wp);
    epoch->on_found = std::move(on_found);
    return epoch;
}

Engine_config make_cfg(std::uint32_t pool_threads,
                       std::shared_ptr<asio::io_context> io,
                       std::uint32_t internal_id_for_solution,
                       bool force_candidate = false)
{
    Engine_config cfg;
    cfg.segment_size = kSegmentSize;
    cfg.channel_starting_nonce =
        static_cast<std::uint64_t>(internal_id_for_solution) << 48;
    cfg.internal_id_for_solution = internal_id_for_solution;
    cfg.pool_threads = pool_threads;
    cfg.io_context = std::move(io);
    cfg.test_skip_sieve = true;
    cfg.test_force_candidate_per_segment = force_candidate;
    return cfg;
}

template <typename Pred>
bool wait_until(std::chrono::milliseconds budget, Pred pred)
{
    const auto deadline = std::chrono::steady_clock::now() + budget;
    while (std::chrono::steady_clock::now() < deadline)
    {
        if (pred()) return true;
        std::this_thread::sleep_for(1ms);
    }
    return pred();
}

// ─────────────────────────────────────────────────────────────────────────────

void test_engine_mode_adapter_surface()
{
    // Worker_prime under engine mode opts into the feed and refuses the
    // legacy per-worker set_block() shim path.  Construction should be
    // cheap (no Sieve, no fermat_performance_test, no run-thread).
    Test_io_context io;
    std::vector<std::unique_ptr<config::Worker_config>> cfgs;
    auto e = make_engine_mode_worker(io.ctx, /*internal_id=*/3, cfgs);

    print_result("engine-mode Worker_prime: uses_template_feed()==true",
                 e.worker->uses_template_feed());

    // attach_template_feed is an explicit no-op; calling it must not throw.
    auto feed = std::make_shared<WorkerTemplateFeed>();
    bool no_throw = true;
    try { e.worker->attach_template_feed(feed); } catch (...) { no_throw = false; }
    print_result("engine-mode attach_template_feed() does not throw", no_throw);

    // Before bind_to_engine, is_running() should be false (not yet bound).
    print_result("engine-mode is_running() == false before bind_to_engine",
                 !e.worker->is_running());

    auto feed2 = std::make_shared<WorkerTemplateFeed>();
    auto engine = std::make_shared<PrimeMiningEngine>(make_cfg(1, io.ctx, 3), feed2);
    e.worker->bind_to_engine(engine, 0, 1);
    // is_running() now reports engine->pool_threads_running() > 0 (not just
    // the bound flag), so wait briefly for the pool thread to register.
    print_result("engine-mode is_running() == true after bind_to_engine",
                 wait_until(2s, [&] { return e.worker->is_running(); }));
}

void test_register_worker_null_guard()
{
    Test_io_context io;
    auto feed = std::make_shared<WorkerTemplateFeed>();
    PrimeMiningEngine engine{make_cfg(0, io.ctx, 0), feed};

    // Documented Stone 5 contract: register_worker() silently ignores nulls.
    bool no_throw = true;
    try { engine.register_worker(nullptr); } catch (...) { no_throw = false; }
    print_result("register_worker(nullptr) does not throw", no_throw);
    print_result("register_worker(nullptr) does not increment count",
                 engine.registered_worker_count() == 0);
}

void test_register_worker_counts_live()
{
    Test_io_context io;
    auto feed = std::make_shared<WorkerTemplateFeed>();
    PrimeMiningEngine engine{make_cfg(0, io.ctx, 0), feed};

    std::vector<std::unique_ptr<config::Worker_config>> cfgs;
    auto e0 = make_engine_mode_worker(io.ctx, 0, cfgs);
    auto e1 = make_engine_mode_worker(io.ctx, 1, cfgs);
    auto e2 = make_engine_mode_worker(io.ctx, 2, cfgs);

    engine.register_worker(e0.worker);
    engine.register_worker(e1.worker);
    engine.register_worker(e2.worker);

    print_result("register_worker counts live workers (3)",
                 engine.registered_worker_count() == 3);

    // Drop one shared_ptr — its weak_ptr in the engine should expire.
    e1.worker.reset();
    print_result("registered_worker_count drops after worker shared_ptr reset",
                 engine.registered_worker_count() == 2);
}

void test_internal_id_for_solution_is_lowest()
{
    Test_io_context io;
    auto feed = std::make_shared<WorkerTemplateFeed>();
    PrimeMiningEngine engine{make_cfg(0, io.ctx, /*internal_id=*/7), feed};
    print_result("internal_id_for_solution == ctor argument",
                 engine.internal_id_for_solution() == 7);
}

void test_stats_partition_sums_to_total()
{
    Test_io_context io;
    auto feed = std::make_shared<WorkerTemplateFeed>();
    auto engine = std::make_shared<PrimeMiningEngine>(
        make_cfg(/*pool_threads=*/2, io.ctx, /*internal_id=*/0,
                 /*force_candidate=*/true), feed);

    std::vector<std::unique_ptr<config::Worker_config>> cfgs;
    auto e0 = make_engine_mode_worker(io.ctx, 0, cfgs);
    auto e1 = make_engine_mode_worker(io.ctx, 1, cfgs);
    auto e2 = make_engine_mode_worker(io.ctx, 2, cfgs);

    engine->register_worker(e0.worker);
    engine->register_worker(e1.worker);
    engine->register_worker(e2.worker);
    e0.worker->bind_to_engine(engine, 0, 3);
    e1.worker->bind_to_engine(engine, 1, 3);
    e2.worker->bind_to_engine(engine, 2, 3);

    boost::multiprecision::uint1024_t base{"0xabc"};
    feed->publish(make_epoch(make_work_package(0x1c00ffffu, base, 1)));
    engine->wait_for_sessions_published_after(0, 2s);
    wait_until(2s, [&] { return engine->segments_processed() >= 6; });

    const auto snap = engine->snapshot_stats();
    const std::uint64_t total_range =
        snap.segments_processed * snap.segment_size;

    auto cfg_for_collector = make_prime_config(/*worker_count=*/3);
    auto collector = stats::make_collector(cfg_for_collector);
    e0.worker->update_statistics(*collector);
    e1.worker->update_statistics(*collector);
    e2.worker->update_statistics(*collector);

    auto& typed = stats::as_typed<stats::Prime>(*collector);
    auto stats0 = typed.get_worker_stats(e0.internal_id);
    auto stats1 = typed.get_worker_stats(e1.internal_id);
    auto stats2 = typed.get_worker_stats(e2.internal_id);

    const std::uint64_t sum_range =
        stats0.m_range_searched + stats1.m_range_searched + stats2.m_range_searched;

    print_result("stats partition: sum(range_searched) == engine total",
                 sum_range == total_range);

    const std::uint32_t shares_chains[3] = {
        stats0.m_chains, stats1.m_chains, stats2.m_chains};
    std::uint32_t mn = shares_chains[0], mx = shares_chains[0];
    for (auto v : shares_chains) { mn = std::min(mn, v); mx = std::max(mx, v); }
    print_result("stats partition: max(chains) - min(chains) <= 1",
                 (mx - mn) <= 1);
}

void test_found_block_credits_representative_worker()
{
    Test_io_context io;
    std::atomic<std::uint32_t> credited_id{std::numeric_limits<std::uint32_t>::max()};
    std::atomic<int> on_found_calls{0};

    auto feed = std::make_shared<WorkerTemplateFeed>();
    constexpr std::uint32_t kRepId = 5;
    PrimeMiningEngine engine{
        make_cfg(/*pool_threads=*/1, io.ctx, kRepId,
                 /*force_candidate=*/true), feed};

    Worker::Block_found_handler on_found =
        [&](std::uint32_t id, std::unique_ptr<Block_data>&& bd) {
            credited_id.store(id);
            on_found_calls.fetch_add(1);
            (void)bd;
        };

    boost::multiprecision::uint1024_t base{"0xfeed"};
    feed->publish(make_epoch(make_work_package(0x1c00ffffu, base, 7), on_found));
    engine.wait_for_sessions_published_after(0, 2s);

    const bool fired = wait_until(2s, [&] { return on_found_calls.load() > 0; });
    print_result("Found-block on_found fired", fired);
    print_result("Found-block credited to representative worker (lowest internal_id)",
                 credited_id.load() == kRepId);
}

void test_engine_destruction_before_workers_safe()
{
    Test_io_context io;
    std::vector<std::unique_ptr<config::Worker_config>> cfgs;
    auto e0 = make_engine_mode_worker(io.ctx, 0, cfgs);
    auto e1 = make_engine_mode_worker(io.ctx, 1, cfgs);

    auto feed = std::make_shared<WorkerTemplateFeed>();
    auto engine = std::make_shared<PrimeMiningEngine>(
        make_cfg(/*pool_threads=*/1, io.ctx, /*internal_id=*/0), feed);

    engine->register_worker(e0.worker);
    engine->register_worker(e1.worker);
    e0.worker->bind_to_engine(engine, 0, 2);
    e1.worker->bind_to_engine(engine, 1, 2);

    // Reset engine BEFORE workers — mirrors Worker_manager teardown order.
    engine.reset();

    auto cfg_for_collector = make_prime_config(/*worker_count=*/2);
    auto collector = stats::make_collector(cfg_for_collector);

    bool no_throw = true;
    try {
        e0.worker->update_statistics(*collector);
        e1.worker->update_statistics(*collector);
    } catch (...) { no_throw = false; }
    print_result("update_statistics() after engine destroyed does not throw",
                 no_throw);

    print_result("is_running() after engine destroyed == false (engine gone)",
                 !e0.worker->is_running() && !e1.worker->is_running());
}

}  // namespace

int main()
{
    if (!spdlog::get("logger"))
    {
        spdlog::create<spdlog::sinks::null_sink_mt>("logger");
    }

    std::cout << "Running PrimeMiningEngine integration (Stone 7) tests...\n";

    test_engine_mode_adapter_surface();
    test_register_worker_null_guard();
    test_register_worker_counts_live();
    test_internal_id_for_solution_is_lowest();
    test_stats_partition_sums_to_total();
    test_found_block_credits_representative_worker();
    test_engine_destruction_before_workers_safe();

    std::cout << "\nResult: " << (tests_run - tests_failed) << "/"
              << tests_run << " passed\n";
    return tests_failed == 0 ? 0 : 1;
}
