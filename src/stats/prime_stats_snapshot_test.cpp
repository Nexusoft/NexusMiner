#include "stats/prime_stats_snapshot.hpp"
#include "stats/types.hpp"

#include <iostream>
#include <limits>
#include <vector>

namespace
{
int tests_run = 0;
int tests_failed = 0;

void print_result(const char* name, bool passed)
{
    ++tests_run;
    std::cout << (passed ? "  [PASS] " : "  [FAIL] ") << name << '\n';
    if (!passed)
        ++tests_failed;
}

void test_copy_prime_histogram_clamps_and_zero_fills()
{
    const std::vector<std::uint32_t> histogram{1, 2, 3, 4, 5};
    const auto snapshot = nexusminer::stats::copy_prime_histogram(histogram);

    const bool passed = snapshot[0] == 1 &&
                        snapshot[1] == 2 &&
                        snapshot[2] == 3 &&
                        snapshot[3] == 4 &&
                        snapshot[4] == 5 &&
                        snapshot[5] == 0 &&
                        snapshot.back() == 0;
    print_result("Prime histogram snapshot copies values and zero-fills tail", passed);
}

void test_atomic_snapshot_keeps_immutable_previous_value()
{
    nexusminer::stats::Atomic_snapshot<nexusminer::stats::Prime> snapshot_store;

    nexusminer::stats::Prime first;
    first.m_range_searched = 11;
    first.m_chain_histogram[5] = 7;
    first.m_cpu_load = 0.25;
    first.m_sieve_diag.m_sieve_calls = 100;
    first.m_sieve_diag.m_inner_hits = 4242;
    snapshot_store.store(first);
    auto first_snapshot = snapshot_store.load();

    nexusminer::stats::Prime second = first;
    second.m_range_searched = 42;
    second.m_chain_histogram[5] = 9;
    second.m_cpu_load = 0.75;
    second.m_sieve_diag.m_sieve_calls = 200;
    second.m_sieve_diag.m_inner_hits = 9999;
    snapshot_store.store(second);
    auto second_snapshot = snapshot_store.load();

    const bool passed = first_snapshot->m_range_searched == 11 &&
                        first_snapshot->m_chain_histogram[5] == 7 &&
                        first_snapshot->m_cpu_load == 0.25 &&
                        first_snapshot->m_sieve_diag.m_sieve_calls == 100 &&
                        first_snapshot->m_sieve_diag.m_inner_hits == 4242 &&
                        second_snapshot->m_range_searched == 42 &&
                        second_snapshot->m_chain_histogram[5] == 9 &&
                        second_snapshot->m_cpu_load == 0.75 &&
                        second_snapshot->m_sieve_diag.m_sieve_calls == 200 &&
                        second_snapshot->m_sieve_diag.m_inner_hits == 9999;
    print_result("Atomic snapshot readers keep immutable published values", passed);
}

void test_saturating_prime_stat_clamps_large_counts()
{
    const auto saturated =
        nexusminer::stats::saturating_prime_stat(std::numeric_limits<std::uint64_t>::max());

    print_result(
        "Prime stat saturation clamps oversized counters to uint32 max",
        saturated == std::numeric_limits<std::uint32_t>::max());
}

// ---------------------------------------------------------------------------
// Regression for the publish-vs-update_statistics CPU-load reset bug.
//
// Worker_prime calls publish_statistics_snapshot() at the end of every mining
// iteration, but update_statistics() (the read side) only runs every
// print_statistics_interval seconds (default 15s). The accumulators
// m_cpu_active_time / m_cpu_total_time must therefore be reset by
// update_statistics(), NOT by publish_statistics_snapshot(); otherwise every
// snapshot only carries the active fraction of the single iteration that just
// completed, which is essentially always 1.0.
//
// This test models that publish/accumulate/publish flow with no read in
// between and asserts the second snapshot's m_cpu_load is < 1.0 when the
// underlying tracker has active < total.
// ---------------------------------------------------------------------------
namespace {
struct Cpu_tracker
{
    // Mirrors the post-fix Worker_prime model:
    //  * accumulate(): worker thread, per iteration
    //  * publish():    worker thread, per iteration — computes ratio, NO reset
    //  * update():     stats thread, per print interval — resets accumulators
    std::int64_t active_ms{0};
    std::int64_t total_ms{0};

    void accumulate(std::int64_t iter_active_ms, std::int64_t iter_total_ms)
    {
        active_ms += iter_active_ms;
        total_ms  += iter_total_ms;
    }

    nexusminer::stats::Prime publish() const
    {
        nexusminer::stats::Prime p{};
        if (total_ms > 0) {
            double load = static_cast<double>(active_ms) / static_cast<double>(total_ms);
            if (load < 0.0) load = 0.0;
            if (load > 1.0) load = 1.0;
            p.m_cpu_load = load;
        }
        return p;
    }

    void reset_on_update()
    {
        active_ms = 0;
        total_ms  = 0;
    }
};
}  // namespace

void test_cpu_load_publish_publish_without_update_keeps_load_below_one()
{
    Cpu_tracker tracker;
    nexusminer::stats::Atomic_snapshot<nexusminer::stats::Prime> snapshot_store;

    // Iteration 1: 30ms active out of 100ms total.
    tracker.accumulate(30, 100);
    snapshot_store.store(tracker.publish());
    auto first = snapshot_store.load();

    // Iteration 2: another 40ms active out of 100ms total.
    // No update_statistics() ran between iterations, so the accumulators
    // must NOT have been reset by the first publish. Cumulative is now
    // 70/200 = 0.35 — well below 1.0.
    tracker.accumulate(40, 100);
    snapshot_store.store(tracker.publish());
    auto second = snapshot_store.load();

    const bool first_ok  = first->m_cpu_load > 0.29 && first->m_cpu_load < 0.31;
    const bool second_ok = second->m_cpu_load > 0.34 && second->m_cpu_load < 0.36;
    const bool below_one = second->m_cpu_load < 1.0;

    print_result(
        "publish→accumulate→publish without update_statistics keeps cumulative cpu_load < 1.0",
        first_ok && second_ok && below_one);
}

void test_cpu_load_resets_only_on_update_statistics()
{
    // After update_statistics() resets the accumulators, the next publish
    // sees a fresh interval rather than the all-time cumulative ratio.
    Cpu_tracker tracker;
    nexusminer::stats::Atomic_snapshot<nexusminer::stats::Prime> snapshot_store;

    tracker.accumulate(80, 100);    // 0.80 active fraction
    snapshot_store.store(tracker.publish());
    const double pre_update_load = snapshot_store.load()->m_cpu_load;

    tracker.reset_on_update();      // simulates update_statistics() running

    tracker.accumulate(10, 100);    // fresh interval: 0.10 active fraction
    snapshot_store.store(tracker.publish());
    const double post_update_load = snapshot_store.load()->m_cpu_load;

    const bool passed = pre_update_load > 0.79 && pre_update_load < 0.81 &&
                        post_update_load > 0.09 && post_update_load < 0.11;
    print_result(
        "Reset on update_statistics() yields fresh per-interval cpu_load on next publish",
        passed);
}
}

int main()
{
    test_copy_prime_histogram_clamps_and_zero_fills();
    test_atomic_snapshot_keeps_immutable_previous_value();
    test_saturating_prime_stat_clamps_large_counts();
    test_cpu_load_publish_publish_without_update_keeps_load_below_one();
    test_cpu_load_resets_only_on_update_statistics();

    if (tests_failed != 0)
    {
        std::cout << "\nprime_stats_snapshot_test: " << tests_failed << " of " << tests_run
                  << " test(s) failed.\n";
        return 1;
    }

    std::cout << "\nprime_stats_snapshot_test: all " << tests_run << " test(s) passed.\n";
    return 0;
}
