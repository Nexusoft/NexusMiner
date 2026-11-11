// Stone 6 — PrimeMiningEngine pool sieve threads tests.
//
// Covers:
//   * Lifecycle: pool threads spawn and join cleanly with no template
//     ever published.
//   * One-template advances: segments_processed counter advances after
//     a single publish.
//   * Heavy churn: rapid republishes drive
//     segments_discarded_epoch_changed > 0.
//   * Candidates dispatched: forced-candidate seam yields candidates_dispatched
//     equal to the number of segments processed before the session is consumed.
//   * Explicit pool_threads count is honored as-is.
//   * pool_threads == 0 auto-derives within the auto-cap.
//   * segments_skipped_consumed advances after the session is consumed.
//   * Destruction during heavy publish + heavy candidate dispatch joins
//     cleanly within a 5s bound.
//   * Same-base-hash republish does NOT discard work
//     (segments_discarded_epoch_changed only advances on actual base change).
//
// The tests use the Engine_config::test_skip_sieve and
// test_force_candidate_per_segment seams so they don't depend on the actual
// prime distribution of an arbitrary base hash.  The full sieve pipeline
// is exercised in production paths (Stone 7) and by the existing
// chain_sieve_test / prime_validation_test targets.

#include "cpu/prime/prime_mining_engine.hpp"
#include "cpu/prime/segment_allocator.hpp"
#include "cpu/prime/chain_sieve.hpp"
#include "worker/template_feed.hpp"
#include "worker.hpp"
#include "block.hpp"

#include <asio.hpp>
#include <spdlog/sinks/null_sink.h>
#include <spdlog/spdlog.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <thread>

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
constexpr std::uint64_t kStartingNonce = 0x1234'0000'0000ULL;
constexpr std::uint32_t kRepWorkerId = 7;

// RAII wrapper: an io_context with a work guard, run on a background thread,
// shut down cleanly on destruction.  Pool threads asio::post their found-block
// callbacks to this io_context.
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

Engine_config make_cfg(std::uint32_t pool_threads,
                       std::shared_ptr<asio::io_context> io,
                       bool force_candidate = false,
                       std::chrono::microseconds simulated_latency = std::chrono::microseconds{0})
{
    Engine_config cfg;
    cfg.segment_size = kSegmentSize;
    cfg.channel_starting_nonce = kStartingNonce;
    cfg.internal_id_for_solution = kRepWorkerId;
    cfg.pool_threads = pool_threads;
    cfg.io_context = std::move(io);
    cfg.test_skip_sieve = true;
    cfg.test_force_candidate_per_segment = force_candidate;
    cfg.simulated_segment_latency = simulated_latency;
    return cfg;
}

std::shared_ptr<WorkPackage> make_work_package(std::uint32_t nbits,
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

// Wait until predicate() returns true OR the deadline expires.  Returns the
// final predicate value.  Used to avoid sleep-and-pray timing in tests.
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

void test_no_template_lifecycle()
{
    Test_io_context io;
    auto feed = std::make_shared<WorkerTemplateFeed>();
    {
        PrimeMiningEngine engine{make_cfg(2, io.ctx), feed};
        // Give pool threads a moment to actually park on m_pool_cv.
        const bool running = wait_until(500ms,
            [&] { return engine.pool_threads_running() == 2; });
        print_result("Pool threads start (count==2 with no template)", running);
        print_result("No-template: segments_processed == 0",
                     engine.segments_processed() == 0);
        print_result("No-template: candidates_dispatched == 0",
                     engine.candidates_dispatched() == 0);
        print_result("No-template: pool_threads_crashed == 0",
                     engine.pool_threads_crashed() == 0);
    }
    print_result("Destructor joins pool threads cleanly with no template", true);
}

void test_one_template_segments_advance()
{
    Test_io_context io;
    auto feed = std::make_shared<WorkerTemplateFeed>();
    PrimeMiningEngine engine{make_cfg(2, io.ctx), feed};

    boost::multiprecision::uint1024_t base{"0xabc"};
    feed->publish(make_epoch(make_work_package(0x1c00ffffu, base, 1)));
    engine.wait_for_sessions_published_after(0, 2s);

    const bool advanced = wait_until(2s, [&] {
        return engine.segments_processed() >= 8;
    });
    print_result("segments_processed advances after a publish", advanced);
    print_result("Pool did not crash", engine.pool_threads_crashed() == 0);
    print_result("No epoch-discards on a single stable template",
                 engine.segments_discarded_epoch_changed() == 0);
}

void test_heavy_churn_drives_discards()
{
    Test_io_context io;
    auto feed = std::make_shared<WorkerTemplateFeed>();
    // 2ms simulated per-segment latency widens the race window between
    // segment draw and the post-segment session re-check, making the
    // discard observation deterministic on fast hosts.  Without this,
    // the test_skip_sieve pool loop completes faster than the publisher
    // can rotate base_hash and the test can pass with zero discards.
    PrimeMiningEngine engine{
        make_cfg(/*pool_threads=*/4, io.ctx, /*force_candidate=*/false,
                 /*simulated_latency=*/std::chrono::milliseconds{2}),
        feed};

    // Publish many distinct base hashes back-to-back.  Each different base
    // forces an allocator reset on the consumer side; the post-segment
    // re-check on pool threads catches the base-hash change and increments
    // segments_discarded_epoch_changed.
    constexpr int kPublishes = 50;
    std::uint64_t last_seen = 0;
    for (int i = 0; i < kPublishes; ++i)
    {
        boost::multiprecision::uint1024_t base{i + 1};
        feed->publish(make_epoch(make_work_package(0x1c00ffffu, base, i)));
        last_seen = engine.wait_for_sessions_published_after(last_seen, 1s);
        std::this_thread::sleep_for(2ms);
    }

    // Give pool threads a beat to observe the final session and process.
    wait_until(1s, [&] { return engine.segments_processed() > 0; });

    print_result("Heavy churn: allocator_resets >= kPublishes",
                 engine.allocator_resets() >= static_cast<std::uint64_t>(kPublishes));
    print_result("Heavy churn: segments_discarded_epoch_changed > 0",
                 engine.segments_discarded_epoch_changed() > 0);
    print_result("Heavy churn: pool_threads_crashed == 0",
                 engine.pool_threads_crashed() == 0);
}

void test_candidates_dispatched_accuracy()
{
    Test_io_context io;
    std::atomic<int> on_found_calls{0};
    auto feed = std::make_shared<WorkerTemplateFeed>();
    PrimeMiningEngine engine{make_cfg(/*pool_threads=*/2, io.ctx,
                                      /*force_candidate=*/true), feed};

    boost::multiprecision::uint1024_t base{"0xfeed"};
    feed->publish(make_epoch(make_work_package(0x1c00ffffu, base, 42),
        [&](std::uint32_t /*id*/, std::unique_ptr<Block_data>&& bd) {
            (void)bd;
            on_found_calls.fetch_add(1, std::memory_order_relaxed);
        }));
    engine.wait_for_sessions_published_after(0, 2s);

    // With test_force_candidate_per_segment + single-found-block-wins, the
    // first segment any pool thread completes for this session marks the
    // session consumed; every subsequent pool-thread iteration observes the
    // consumed bit and idles.  So candidates_dispatched MUST equal exactly 1
    // per published session, regardless of pool thread count.
    const bool dispatched = wait_until(2s, [&] {
        return engine.candidates_dispatched() == 1;
    });
    print_result("Forced candidate: candidates_dispatched == 1 per session",
                 dispatched);

    // Wait for the io_context to drain the posted callback.
    const bool callback_fired = wait_until(2s, [&] {
        return on_found_calls.load() == 1;
    });
    print_result("Forced candidate: on_found callback fired exactly once",
                 callback_fired);

    print_result("Forced candidate: pool did not crash",
                 engine.pool_threads_crashed() == 0);
}

void test_explicit_pool_thread_count_honored()
{
    Test_io_context io;
    auto feed = std::make_shared<WorkerTemplateFeed>();
    PrimeMiningEngine engine{make_cfg(/*pool_threads=*/3, io.ctx), feed};
    print_result("Explicit pool_threads=3 honored",
                 engine.pool_thread_count() == 3);
    const bool running = wait_until(500ms,
        [&] { return engine.pool_threads_running() == 3; });
    print_result("3 pool threads actually running", running);
}

void test_auto_pool_thread_count_within_cap()
{
    Test_io_context io;
    auto feed = std::make_shared<WorkerTemplateFeed>();
    PrimeMiningEngine engine{make_cfg(/*pool_threads=*/0, io.ctx), feed};
    const auto count = engine.pool_thread_count();
    print_result("Auto pool count >= 1",
                 count >= 1);
    print_result("Auto pool count <= max auto cap",
                 count <= PrimeMiningEngine::pool_threads_max_auto_cap);
}

void test_skipped_consumed_advances()
{
    Test_io_context io;
    auto feed = std::make_shared<WorkerTemplateFeed>();
    PrimeMiningEngine engine{make_cfg(/*pool_threads=*/2, io.ctx,
                                      /*force_candidate=*/true), feed};

    boost::multiprecision::uint1024_t base{"0xc0ffee"};
    feed->publish(make_epoch(make_work_package(0x1c00ffffu, base, 1),
        [](std::uint32_t, std::unique_ptr<Block_data>&&) {}));
    engine.wait_for_sessions_published_after(0, 2s);

    // Wait for the candidate to be dispatched (session marked consumed).
    wait_until(2s, [&] { return engine.candidates_dispatched() == 1; });

    // After consumption, both pool threads should observe is_consumed() and
    // skip-park.  The idle path increments segments_skipped_consumed each
    // time a pool thread re-checks while the session is still consumed.
    const bool skipped = wait_until(2s, [&] {
        return engine.segments_skipped_consumed() > 0;
    });
    print_result("segments_skipped_consumed advances after session consumed",
                 skipped);
}

void test_destruction_during_heavy_churn()
{
    Test_io_context io;
    auto feed = std::make_shared<WorkerTemplateFeed>();
    auto cfg = make_cfg(/*pool_threads=*/4, io.ctx, /*force_candidate=*/true);

    auto start = std::chrono::steady_clock::now();
    {
        PrimeMiningEngine engine{cfg, feed};

        // Background publisher: hammer the feed with new templates while the
        // engine is mining and dispatching.  Each new template re-arms the
        // session (clears consumed) so candidates keep flowing.
        std::atomic<bool> stop{false};
        std::thread publisher([&] {
            int i = 1;
            while (!stop.load(std::memory_order_acquire))
            {
                boost::multiprecision::uint1024_t base{i++};
                feed->publish(make_epoch(make_work_package(0x1c00ffffu, base, i),
                    [](std::uint32_t, std::unique_ptr<Block_data>&&) {}));
                std::this_thread::sleep_for(500us);
            }
        });

        std::this_thread::sleep_for(200ms);
        stop.store(true, std::memory_order_release);
        publisher.join();
        // Engine destructor runs on scope exit.
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    print_result("Destruction during heavy churn joins within 5s",
                 elapsed_ms < 5000);
}

void test_same_base_republish_preserves_work()
{
    Test_io_context io;
    auto feed = std::make_shared<WorkerTemplateFeed>();
    PrimeMiningEngine engine{make_cfg(/*pool_threads=*/2, io.ctx), feed};

    boost::multiprecision::uint1024_t base{"0xbeef"};
    feed->publish(make_epoch(make_work_package(0x1c00ffffu, base, 1)));
    auto p1 = engine.wait_for_sessions_published_after(0, 2s);

    // Let pool threads do real work (no candidates because force=false).
    wait_until(500ms, [&] { return engine.segments_processed() >= 8; });
    const auto discards_before = engine.segments_discarded_epoch_changed();
    const auto resets_before = engine.allocator_resets();

    // Republish with SAME height (same proof-hash space) — simulates a
    // KEEPALIVE where only hashMerkleRoot rotates.  The engine consumer
    // must NOT reset the cursor; the pool-thread post-segment re-check
    // must NOT discard (hashPrevBlock + nHeight match).
    feed->publish(make_epoch(make_work_package(0x1c00ffffu, base, 1)));
    engine.wait_for_sessions_published_after(p1, 2s);

    // Let a few more segments drain so any spurious discards would surface.
    const auto baseline_segments = engine.segments_processed();
    wait_until(500ms, [&] {
        return engine.segments_processed() >= baseline_segments + 8;
    });

    print_result("Same-base republish: allocator_resets unchanged",
                 engine.allocator_resets() == resets_before);
    print_result("Same-base republish: same_base_short_circuits == 1",
                 engine.same_base_short_circuits() == 1);
    print_result("Same-base republish: no extra discards",
                 engine.segments_discarded_epoch_changed() == discards_before);
}

// Option 1 — Merkle-rotation republish: a KEEPALIVE-style update that
// rotates hashMerkleRoot (so prime base_hash changes) at the SAME chain tip
// (same hashPrevBlock + same nHeight) must NOT cause the pool-thread
// post-segment re-check to discard the in-flight chunk.  The publish-layer
// short-circuit (PR #674) preserves the cooperative cursor on this case;
// before Option 1 the pool thread STILL discarded because its discriminator
// was base_hash.  With Option 1 the discriminator is the stable
// (previous_hash, nHeight) pair so the chunk completes and dispatch uses
// the bound session's block_data (the OLD merkle root the candidate was
// sieved against — which is the cryptographically correct thing to do).
void test_merkle_rotation_republish_preserves_chunk()
{
    Test_io_context io;
    // Use simulated_segment_latency to widen the race window between
    // segment draw and the post-segment re-check, so the regression
    // (without Option 1) would deterministically surface as a discard.
    auto feed = std::make_shared<WorkerTemplateFeed>();
    PrimeMiningEngine engine{
        make_cfg(/*pool_threads=*/4, io.ctx, /*force_candidate=*/false,
                 /*simulated_latency=*/std::chrono::milliseconds{2}),
        feed};

    // Initial template at height=42 with base hash A.
    boost::multiprecision::uint1024_t base_a{"0xa11ce"};
    feed->publish(make_epoch(make_work_package(0x1c00ffffu, base_a, 42)));
    auto p1 = engine.wait_for_sessions_published_after(0, 2s);
    wait_until(500ms, [&] { return engine.segments_processed() >= 8; });

    const auto discards_before = engine.segments_discarded_epoch_changed();

    // Republish 30 times with DIFFERENT base hashes but the SAME height
    // (and zero previous_hash on both — make_work_package only sets the
    // base_hash override and height).  Each republish simulates a Merkle
    // rotation: prime base_hash changes, chain tip does not.
    constexpr int kRotations = 30;
    auto last_seen = p1;
    for (int i = 0; i < kRotations; ++i)
    {
        boost::multiprecision::uint1024_t merkle_rotated{0xb00b00 + i};
        feed->publish(make_epoch(make_work_package(0x1c00ffffu, merkle_rotated, 42)));
        last_seen = engine.wait_for_sessions_published_after(last_seen, 1s);
        std::this_thread::sleep_for(2ms);
    }

    // Drain a few more segments so any pending discard would surface.
    const auto baseline_segments = engine.segments_processed();
    wait_until(1s, [&] {
        return engine.segments_processed() >= baseline_segments + 16;
    });

    // The publish layer's same-base short-circuit MUST fire for every
    // Merkle rotation (height + previous_hash unchanged).  Before PR #674
    // this was already true at the publish layer.
    print_result(
        "Merkle rotation: same_base_short_circuits == kRotations",
        engine.same_base_short_circuits() == static_cast<std::uint64_t>(kRotations));

    // The KEY Option 1 invariant: even with 30 base_hash rotations and
    // a 2ms simulated latency widening the race window per segment, the
    // pool-thread post-segment discard counter must NOT advance, because
    // the stable (previous_hash, nHeight) discriminator says "same tip".
    print_result(
        "Merkle rotation: pool threads do NOT discard in-flight chunks "
        "(Option 1 — bound-session post-segment discriminator)",
        engine.segments_discarded_epoch_changed() == discards_before);

    // Sanity: pool threads kept running and made forward progress — the
    // republishes did not deadlock or stall the cooperative cursor.
    print_result(
        "Merkle rotation: segments_processed advanced after rotations",
        engine.segments_processed() > baseline_segments);
    print_result("Merkle rotation: no pool-thread crashes",
                 engine.pool_threads_crashed() == 0);
}

void test_session_only_mode_null_io_context()
{
    // With a null io_context the engine runs in session-only mode: the
    // consumer still publishes sessions, but zero pool threads are spawned
    // and all pool-thread counters stay at zero — even if pool_threads was
    // configured > 0.  Templates can still be published and observed via
    // current_session() (Stone 5 surface).
    auto feed = std::make_shared<WorkerTemplateFeed>();
    PrimeMiningEngine engine{make_cfg(/*pool_threads=*/4, /*io=*/nullptr), feed};
    print_result("Session-only mode: pool_thread_count == 0",
                 engine.pool_thread_count() == 0);

    // Publish one template and confirm the consumer thread still works.
    boost::multiprecision::uint1024_t base{"0xfacade"};
    feed->publish(make_epoch(make_work_package(0x1c00ffffu, base, 1)));
    engine.wait_for_sessions_published_after(0, 2s);
    print_result("Session-only mode: consumer still publishes sessions",
                 engine.sessions_published() >= 1);

    // Give any (incorrectly-spawned) pool thread a moment to do work.
    std::this_thread::sleep_for(50ms);

    print_result("Session-only mode: pool_threads_running == 0",
                 engine.pool_threads_running() == 0);
    print_result("Session-only mode: segments_processed == 0",
                 engine.segments_processed() == 0);
    print_result("Session-only mode: segments_discarded_epoch_changed == 0",
                 engine.segments_discarded_epoch_changed() == 0);
    print_result("Session-only mode: segments_skipped_consumed == 0",
                 engine.segments_skipped_consumed() == 0);
    print_result("Session-only mode: candidates_dispatched == 0",
                 engine.candidates_dispatched() == 0);
    print_result("Session-only mode: pool_threads_crashed == 0",
                 engine.pool_threads_crashed() == 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Stone 6.5 — regression test: chunk-based amortisation of
// Sieve::calculate_starting_multiples() under the REAL sieve path.
//
// This test would FAIL on the Stone 6 codebase (which called
// calculate_starting_multiples() per segment) and PASSES on Stone 6.5
// (one call per chunk).  It is the test-coverage gap that allowed the
// production performance regression to ship.
//
// Unlike every other test in this file, this one does NOT set
// test_skip_sieve.  It pays the cost of the real sieving-primes singleton
// build (~2s on first construction) so it can observe the actual ratio
// of starting_multiples_calls : segments_processed.
// ─────────────────────────────────────────────────────────────────────────────
void test_chunk_amortizes_starting_multiples_calls()
{
    // Populate the sieving-primes singleton once on a throwaway Sieve.
    // Subsequent Sieves (one per pool thread) share the populated table.
    {
        Sieve seed_sieve;
        seed_sieve.generate_sieving_primes();
    }

    Test_io_context io;
    auto feed = std::make_shared<WorkerTemplateFeed>();

    // Use the production default chunk size (64) and a small pool so the
    // test runs in bounded time even with the real (heavy) sieve.
    Engine_config cfg;
    cfg.segment_size = Sieve::get_segment_size();
    cfg.channel_starting_nonce = kStartingNonce;
    cfg.internal_id_for_solution = kRepWorkerId;
    cfg.pool_threads = 2;
    cfg.io_context = io.ctx;
    cfg.test_skip_sieve = false;                     // ← REAL sieve path
    cfg.test_force_candidate_per_segment = false;
    // Stone 6.5: smaller-than-default chunk size keeps per-chunk wall-clock
    // bounded under the test's grind window while still demonstrating the
    // amortisation ratio.  Production default is 64; here we use 4 so the
    // test completes in well under 30 seconds even on a modest CI runner
    // and still distinguishes Stone 6 (1:1 ratio → would FAIL this assertion
    // by a factor of 4) from Stone 6.5 (~1:4 ratio → PASSES with margin).
    cfg.pool_chunk_segments = 4;
    const std::uint64_t chunk_size = cfg.pool_chunk_segments;

    PrimeMiningEngine engine{cfg, feed};

    // Publish a single template; let pool threads grind for a bounded time.
    boost::multiprecision::uint1024_t base{
        "0x123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0"};
    feed->publish(make_epoch(make_work_package(0x1c00ffffu, base, 1)));
    engine.wait_for_sessions_published_after(0, 2s);

    // Bounded grind window.  The real sieve is heavy (each segment_size
    // covers ~30 MB of wheel-30 odds and includes a Sieve::sieve_segment +
    // find_chains + test_chains pass), so we wait long enough that several
    // full chunks complete on each pool thread and partial-chunk overhead
    // does not dominate the ratio.
    std::this_thread::sleep_for(30s);

    const auto snap = engine.snapshot_stats();

    // Sanity: the engine is doing real work.
    print_result("Stone 6.5 regression: segments_processed > 0",
                 snap.segments_processed > 0);

    // Core assertion: chunk amortisation is in effect.
    //
    //   starting_multiples_calls * pool_chunk_segments  ≤  segments_processed * 1.5
    //
    // The 1.5× slack accounts for partial chunks at session boundaries and
    // re-bind on base-hash changes.  Pre-fix this ratio would have been ~1:1
    // (one calculate_starting_multiples call per segment), so the assertion
    // would fail by a ~64× margin.
    const std::uint64_t lhs = snap.starting_multiples_calls * chunk_size;
    const std::uint64_t rhs_15x = (snap.segments_processed * 3) / 2;
    const bool amortized = lhs <= rhs_15x;
    if (!amortized)
    {
        std::cout << "    [diag] segments_processed=" << snap.segments_processed
                  << " starting_multiples_calls=" << snap.starting_multiples_calls
                  << " chunks_drawn=" << snap.chunks_drawn
                  << " chunk_size=" << chunk_size
                  << " lhs=" << lhs << " rhs(1.5x)=" << rhs_15x
                  << '\n';
    }
    print_result("Stone 6.5 regression: chunk amortisation holds "
                 "(starting_multiples_calls * chunk_size ≤ segments_processed * 1.5)",
                 amortized);

    // Pool did not crash on the real sieve.
    print_result("Stone 6.5 regression: pool_threads_crashed == 0",
                 engine.pool_threads_crashed() == 0);

    // chunks_drawn > 0 — confirms next_segment_chunk() is in use.
    print_result("Stone 6.5 regression: chunks_drawn > 0",
                 snap.chunks_drawn > 0);
}

void test_invalid_chunk_size_rejected()
{
    Test_io_context io;
    auto feed = std::make_shared<WorkerTemplateFeed>();

    // 0 is rejected.
    bool threw_zero = false;
    try
    {
        Engine_config cfg = make_cfg(1, io.ctx);
        cfg.pool_chunk_segments = 0;
        PrimeMiningEngine engine{cfg, feed};
    }
    catch (const std::invalid_argument&) { threw_zero = true; }
    print_result("pool_chunk_segments=0 rejected", threw_zero);

    // > 1024 is rejected.
    bool threw_huge = false;
    try
    {
        Engine_config cfg = make_cfg(1, io.ctx);
        cfg.pool_chunk_segments = 1025;
        PrimeMiningEngine engine{cfg, feed};
    }
    catch (const std::invalid_argument&) { threw_huge = true; }
    print_result("pool_chunk_segments=1025 rejected", threw_huge);

    // 1 is allowed (regression-test-only seam).
    bool ok_one = false;
    try
    {
        Engine_config cfg = make_cfg(1, io.ctx);
        cfg.pool_chunk_segments = 1;
        PrimeMiningEngine engine{cfg, feed};
        ok_one = engine.pool_chunk_segments() == 1;
    }
    catch (...) { ok_one = false; }
    print_result("pool_chunk_segments=1 accepted (regression-test seam)", ok_one);
}

}  // namespace

int main()
{
    if (!spdlog::get("logger"))
    {
        spdlog::create<spdlog::sinks::null_sink_mt>("logger");
    }

    std::cout << "Running cpu::PrimeMiningEngine pool-thread tests...\n";

    test_no_template_lifecycle();
    test_one_template_segments_advance();
    test_heavy_churn_drives_discards();
    test_candidates_dispatched_accuracy();
    test_explicit_pool_thread_count_honored();
    test_auto_pool_thread_count_within_cap();
    test_skipped_consumed_advances();
    test_destruction_during_heavy_churn();
    test_same_base_republish_preserves_work();
    test_merkle_rotation_republish_preserves_chunk();
    test_session_only_mode_null_io_context();
    test_invalid_chunk_size_rejected();
    test_chunk_amortizes_starting_multiples_calls();

    std::cout << "\nResult: " << (tests_run - tests_failed) << "/" << tests_run << " passed\n";
    return tests_failed == 0 ? 0 : 1;
}
