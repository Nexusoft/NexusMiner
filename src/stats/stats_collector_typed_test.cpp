// Unit tests for the typed-Collector refactor (Issues 3A + 6A) and the
// centralized prime view helper (Issue 5A).
#include "stats/stats_collector.hpp"
#include "stats/prime_view.hpp"
#include "stats/types.hpp"
#include "config/config.hpp"
#include "config/types.hpp"
#include <spdlog/spdlog.h>
#include <spdlog/sinks/null_sink.h>

#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

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

std::shared_ptr<spdlog::logger> make_logger()
{
    auto sink   = std::make_shared<spdlog::sinks::null_sink_mt>();
    static std::atomic<int> counter{0};
    auto logger = std::make_shared<spdlog::logger>(
        "stats_collector_test_logger_" + std::to_string(counter++), sink);
    return logger;
}

nexusminer::config::Config make_config(nexusminer::config::Mining_mode mode,
                                       std::uint32_t worker_count)
{
    nexusminer::config::Config config{make_logger()};
    config.set_mining_mode(mode);
    config.set_worker_count(worker_count);
    // set_worker_count() builds CPU workers but doesn't assign m_internal_id;
    // Worker_manager normally does that downstream. Assign here so tests that
    // exercise update_worker_stats(internal_id, …) match production indexing.
    auto& workers = config.get_worker_config();
    for (std::size_t i = 0; i < workers.size(); ++i) {
        workers[i].m_internal_id = static_cast<std::uint16_t>(i);
    }
    return config;
}

// ----------------------------------------------------------------------
// Issue 3A — typed Collector
// ----------------------------------------------------------------------

void test_factory_returns_correct_typed_collector()
{
    auto cfg_hash  = make_config(nexusminer::config::Mining_mode::HASH, 2);
    auto cfg_prime = make_config(nexusminer::config::Mining_mode::PRIME, 3);

    auto hash_c  = nexusminer::stats::make_collector(cfg_hash);
    auto prime_c = nexusminer::stats::make_collector(cfg_prime);

    bool passed =
        hash_c  != nullptr && prime_c != nullptr &&
        hash_c->get_mining_mode()  == nexusminer::config::Mining_mode::HASH &&
        prime_c->get_mining_mode() == nexusminer::config::Mining_mode::PRIME &&
        hash_c->worker_count()  == 2 &&
        prime_c->worker_count() == 3 &&
        // Downcast must succeed for the matching mode.
        dynamic_cast<nexusminer::stats::Hash_collector*>(hash_c.get())   != nullptr &&
        dynamic_cast<nexusminer::stats::Prime_collector*>(prime_c.get()) != nullptr &&
        // …and fail for the other.
        dynamic_cast<nexusminer::stats::Prime_collector*>(hash_c.get())  == nullptr &&
        dynamic_cast<nexusminer::stats::Hash_collector*>(prime_c.get())  == nullptr;

    print_result("make_collector returns typed collector matching mining mode", passed);
}

void test_typed_worker_stats_round_trip()
{
    auto cfg = make_config(nexusminer::config::Mining_mode::PRIME, 2);
    nexusminer::stats::Prime_collector pc{cfg};

    nexusminer::stats::Prime stats0;
    stats0.m_primes = 10;
    stats0.m_chains = 1;
    stats0.m_range_searched = 1000;
    stats0.m_chain_histogram[5] = 7;

    nexusminer::stats::Prime stats1;
    stats1.m_primes = 20;
    stats1.m_chains = 2;
    stats1.m_range_searched = 2000;
    stats1.m_chain_histogram[6] = 3;

    pc.update_worker_stats(0, stats0);
    pc.update_worker_stats(1, stats1);

    auto got0 = pc.get_worker_stats(0);
    auto got1 = pc.get_worker_stats(1);
    auto all  = pc.get_workers_stats();

    bool passed =
        got0.m_primes == 10 && got0.m_chains == 1 && got0.m_chain_histogram[5] == 7 &&
        got1.m_primes == 20 && got1.m_chains == 2 && got1.m_chain_histogram[6] == 3 &&
        all.size() == 2 &&
        all[0].m_range_searched == 1000 &&
        all[1].m_range_searched == 2000;

    print_result("Typed Worker_stats_collector<Prime> round-trips per-worker stats", passed);
}

void test_out_of_range_worker_id_does_not_crash()
{
    auto cfg = make_config(nexusminer::config::Mining_mode::HASH, 1);
    nexusminer::stats::Hash_collector hc{cfg};

    nexusminer::stats::Hash h{};
    h.m_hash_count = 99;

    hc.update_worker_stats(0, h);

    // Out-of-range worker ids are programming errors: debug builds assert
    // (see Worker_stats_collector::update_worker_stats / get_worker_stats),
    // release builds fall back safely (silent drop on write,
    // default-constructed T on read). Only the release-mode safe-fallback
    // is exercisable as a unit test — under debug the assert is the
    // documented contract.
#ifdef NDEBUG
    hc.update_worker_stats(99, h);                                  // silent drop
    bool oob_get_safe = hc.get_worker_stats(99).m_hash_count == 0;  // default T{}
#else
    bool oob_get_safe = true;
#endif

    bool in_range_ok = hc.get_worker_stats(0).m_hash_count == 99;
    print_result("Out-of-range worker_id is safe in release (assert in debug)",
                 in_range_ok && oob_get_safe);
}

// ----------------------------------------------------------------------
// Issue 6A — split mutexes (global vs worker)
// ----------------------------------------------------------------------

void test_global_and_worker_writes_run_concurrently()
{
    auto cfg = make_config(nexusminer::config::Mining_mode::PRIME, 4);
    nexusminer::stats::Prime_collector pc{cfg};

    // The intent here is exercise, not timing: we want to confirm that
    // global writers and worker writers can both make forward progress
    // simultaneously without deadlocking, which is the principal benefit
    // of the split-mutex design (Issue 6A).
    constexpr int iterations = 5000;
    std::atomic<bool> start{false};

    std::thread global_writer([&] {
        while (!start.load(std::memory_order_acquire)) {}
        for (int i = 0; i < iterations; ++i) {
            nexusminer::stats::Global_delta delta{};
            delta.m_accepted_blocks = 1;
            pc.update_global_stats(delta);
        }
    });

    std::thread worker_writer([&] {
        while (!start.load(std::memory_order_acquire)) {}
        for (int i = 0; i < iterations; ++i) {
            nexusminer::stats::Prime p{};
            p.m_range_searched = static_cast<std::uint64_t>(i);
            pc.update_worker_stats(static_cast<std::uint16_t>(i % 4), p);
        }
    });

    std::thread global_reader([&] {
        while (!start.load(std::memory_order_acquire)) {}
        std::uint32_t observed = 0;
        for (int i = 0; i < iterations; ++i) {
            observed = pc.get_global_stats().m_accepted_blocks;
        }
        (void)observed;
    });

    std::thread worker_reader([&] {
        while (!start.load(std::memory_order_acquire)) {}
        for (int i = 0; i < iterations; ++i) {
            auto v = pc.get_workers_stats();
            (void)v;
        }
    });

    start.store(true, std::memory_order_release);
    global_writer.join();
    worker_writer.join();
    global_reader.join();
    worker_reader.join();

    auto final_global = pc.get_global_stats();
    bool passed = final_global.m_accepted_blocks == static_cast<std::uint32_t>(iterations);
    print_result("Concurrent global+worker writers complete without deadlock and global count is exact",
                 passed);
}

void test_degraded_mode_setter_does_not_reaccumulate_counters()
{
    auto cfg = make_config(nexusminer::config::Mining_mode::HASH, 1);
    nexusminer::stats::Hash_collector hc{cfg};

    nexusminer::stats::Global_delta delta{};
    delta.m_accepted_blocks = 1;
    delta.m_rejected_blocks = 2;
    delta.m_connection_retries = 3;
    hc.update_global_stats(delta);

    hc.set_degraded_mode(true);
    hc.set_degraded_mode(false);
    hc.set_degraded_mode(true);

    auto global = hc.get_global_stats();
    bool passed = global.m_accepted_blocks == 1 &&
                  global.m_rejected_blocks == 2 &&
                  global.m_connection_retries == 3 &&
                  global.m_degraded_mode;
    print_result("set_degraded_mode toggles state without re-adding global counters", passed);
}

void test_reset_global_counters_preserves_degraded_mode()
{
    auto cfg = make_config(nexusminer::config::Mining_mode::HASH, 1);
    nexusminer::stats::Hash_collector hc{cfg};

    nexusminer::stats::Global_delta delta{};
    delta.m_accepted_blocks = 4;
    delta.m_rejected_blocks = 5;
    delta.m_accepted_shares = 6;
    delta.m_rejected_shares = 7;
    delta.m_connection_retries = 8;
    hc.update_global_stats(delta);
    hc.set_degraded_mode(true);

    hc.reset_global_counters();

    auto global = hc.get_global_stats();
    bool passed = global.m_accepted_blocks == 0 &&
                  global.m_rejected_blocks == 0 &&
                  global.m_accepted_shares == 0 &&
                  global.m_rejected_shares == 0 &&
                  global.m_connection_retries == 0 &&
                  global.m_degraded_mode;
    print_result("reset_global_counters zeroes counters while preserving degraded mode", passed);
}

// ----------------------------------------------------------------------
// Issue 5A — centralized prime view computation
// ----------------------------------------------------------------------

void test_compute_prime_view_basic_gisps()
{
    nexusminer::stats::Prime previous;
    previous.m_range_searched = 1'000'000'000ull;  // 1e9

    nexusminer::stats::Prime current;
    current.m_range_searched = 11'000'000'000ull;  // +1e10
    current.m_difficulty     = 95'000'000;         // 9.5
    current.m_cpu_load       = 0.42;

    auto v = nexusminer::stats::compute_prime_view(current, previous,
                                                   /*interval_s=*/10.0,
                                                   /*degraded=*/false);

    bool passed = v.gisps == 1.0 &&
                  v.difficulty == 9.5 &&
                  v.cpu_load == 0.42;
    print_result("compute_prime_view: GISPS = (range_delta) / (1e9 * interval_s)", passed);
}

void test_compute_prime_view_degraded_zeros_gisps_only()
{
    nexusminer::stats::Prime previous;
    previous.m_range_searched = 1'000'000'000ull;
    nexusminer::stats::Prime current;
    current.m_range_searched = 5'000'000'000ull;
    current.m_difficulty     = 80'000'000;  // 8.0
    current.m_cpu_load       = 0.7;

    auto v = nexusminer::stats::compute_prime_view(current, previous,
                                                   /*interval_s=*/4.0,
                                                   /*degraded=*/true);

    bool passed = v.gisps == 0.0 &&
                  v.difficulty == 8.0 &&
                  v.cpu_load == 0.7;
    print_result("compute_prime_view: degraded mode zeros GISPS but preserves difficulty/cpu_load",
                 passed);
}

void test_compute_prime_view_handles_counter_reset()
{
    // Worker reset: previous range > current range (saturating subtract).
    nexusminer::stats::Prime previous;
    previous.m_range_searched = 100'000'000'000ull;
    nexusminer::stats::Prime current;
    current.m_range_searched = 500'000'000ull;  // smaller than previous

    auto v = nexusminer::stats::compute_prime_view(current, previous,
                                                   /*interval_s=*/1.0,
                                                   /*degraded=*/false);

    // Falls back to current.m_range_searched / (1e9 * interval_s) = 0.5
    bool passed = v.gisps == 0.5;
    print_result("compute_prime_view: worker counter reset uses current range (no underflow)",
                 passed);
}

void test_compute_prime_view_clamps_cpu_load_and_handles_zero_interval()
{
    nexusminer::stats::Prime previous;
    nexusminer::stats::Prime current;
    current.m_range_searched = 1'000'000'000ull;
    current.m_cpu_load = 1.5;  // out of range, must clamp

    auto v = nexusminer::stats::compute_prime_view(current, previous,
                                                   /*interval_s=*/0.0,
                                                   /*degraded=*/false);

    // interval_s == 0 must produce 0 GISPS (no division by zero).
    bool passed = v.gisps == 0.0 && v.cpu_load == 1.0;
    print_result("compute_prime_view: clamps cpu_load to [0,1] and guards interval_s<=0",
                 passed);
}

}  // namespace

int main()
{
    test_factory_returns_correct_typed_collector();
    test_typed_worker_stats_round_trip();
    test_out_of_range_worker_id_does_not_crash();
    test_global_and_worker_writes_run_concurrently();
    test_degraded_mode_setter_does_not_reaccumulate_counters();
    test_reset_global_counters_preserves_degraded_mode();

    test_compute_prime_view_basic_gisps();
    test_compute_prime_view_degraded_zeros_gisps_only();
    test_compute_prime_view_handles_counter_reset();
    test_compute_prime_view_clamps_cpu_load_and_handles_zero_interval();

    if (tests_failed != 0) {
        std::cout << "\nstats_collector_typed_test: " << tests_failed << " of " << tests_run
                  << " test(s) failed.\n";
        return 1;
    }

    std::cout << "\nstats_collector_typed_test: all " << tests_run << " test(s) passed.\n";
    return 0;
}
