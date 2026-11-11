// Stone 5 — PrimeMiningEngine session creation tests.
//
// Covers:
//   * Construction/destruction lifecycle (consumer thread joins cleanly).
//   * Empty engine: current_session() == nullptr; counters are zero.
//   * Single publish: consumer builds a session whose epoch_id matches the
//     feed's epoch_id, and copies block + nbits + on_found out of the
//     WorkPackage.  Allocator is reset on first publish.
//   * Same-base short-circuit: a second publish with the same
//     height+hashPrevBlock does NOT reset the cooperative cursor;
//     same_base_short_circuits counter advances.
//   * Different-height publish DOES reset the cursor and advance the
//     allocator_resets counter.
//   * KEEPALIVE Merkle-rotation: same height+hashPrevBlock, different
//     hashMerkleRoot (no precomputed hash) still triggers the short-circuit
//     because the predicate does NOT include hashMerkleRoot.
//   * Null-payload publish is a no-op (no session built; no counters move).
//   * Shutdown wakes a consumer parked on wait_for_epoch_after even with
//     no publish.
//   * Shared_segment_allocator: concurrent fetch_add returns a contiguous
//     duplicate-free sequence.

#include "cpu/prime/prime_mining_engine.hpp"
#include "cpu/prime/segment_allocator.hpp"
#include "worker/template_feed.hpp"
#include "worker.hpp"
#include "block.hpp"

#include <spdlog/sinks/null_sink.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <set>
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
    if (!passed)
        ++tests_failed;
}

constexpr std::uint64_t kSegmentSize = 4096;
constexpr std::uint64_t kStartingNonce = 0x1234'0000'0000ULL;
constexpr std::uint32_t kRepWorkerId = 7;

Engine_config default_cfg()
{
    Engine_config cfg;
    cfg.segment_size = kSegmentSize;
    cfg.channel_starting_nonce = kStartingNonce;
    cfg.internal_id_for_solution = kRepWorkerId;
    return cfg;
}

// Build a WorkPackage with a deterministic precomputed prime base hash so we
// can exercise the same-base-hash short-circuit without involving the
// Skein/Keccak hash chain.
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

// ─────────────────────────────────────────────────────────────────────────────

void test_lifecycle_no_publish()
{
    auto feed = std::make_shared<WorkerTemplateFeed>();
    {
        PrimeMiningEngine engine{default_cfg(), feed};
        print_result("Empty engine current_session() is nullptr",
                     engine.current_session() == nullptr);
        print_result("Empty engine sessions_published() == 0",
                     engine.sessions_published() == 0);
        print_result("Empty engine allocator_resets() == 0",
                     engine.allocator_resets() == 0);
        print_result("Empty engine same_base_short_circuits() == 0",
                     engine.same_base_short_circuits() == 0);
        // Stone 6 (cursor amendment): cooperative cursor is RELATIVE — it
        // always seeds at 0 on a fresh engine.  Pool threads add
        // session->starting_nonce themselves at use time.
        print_result("Allocator seeded at relative 0",
                     engine.segment_allocator().current() == 0);
    }
    // Destructor must join the consumer thread cleanly — if it doesn't, the
    // test process will hang here forever, which CTest will flag as a timeout.
    print_result("Engine destructor joined consumer cleanly", true);
}

void test_first_publish_builds_session()
{
    auto feed = std::make_shared<WorkerTemplateFeed>();
    PrimeMiningEngine engine{default_cfg(), feed};

    boost::multiprecision::uint1024_t base_hash{
        "0x0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"};
    std::atomic<int> on_found_calls{0};
    auto wp = make_work_package(/*nbits=*/0x1d00ffffu, base_hash, /*height=*/2024000);

    const auto epoch_id = feed->publish(make_epoch(wp,
        [&](std::uint32_t /*id*/, std::unique_ptr<Block_data>&& /*bd*/) {
            on_found_calls.fetch_add(1);
        }));

    // Wait for the consumer thread to publish the corresponding session.
    const auto published = engine.wait_for_sessions_published_after(0, 2s);
    print_result("First publish produces sessions_published()==1", published == 1);

    auto session = engine.current_session();
    print_result("current_session() is non-null after first publish", session != nullptr);
    if (!session) return;

    print_result("Session epoch_id matches feed epoch_id",
                 session->epoch_id == epoch_id);
    print_result("Session base_hash equals precomputed value",
                 session->base_hash == base_hash);
    print_result("Session nbits copied from WorkPackage",
                 session->nbits == 0x1d00ffffu);
    print_result("Session starting_nonce equals channel_starting_nonce",
                 session->starting_nonce == kStartingNonce);
    print_result("Session internal_id_for_solution comes from Engine_config",
                 session->internal_id_for_solution == kRepWorkerId);
    print_result("Session block_data height copied from CBlock",
                 session->block_data.nHeight == 2024000);
    print_result("Session on_found stored and invokable",
                 static_cast<bool>(session->on_found));
    if (session->on_found) {
        session->on_found(0, nullptr);
        print_result("Stored on_found callable forwards to original",
                     on_found_calls.load() == 1);
    }
    print_result("First publish triggered exactly one allocator reset",
                 engine.allocator_resets() == 1);
    print_result("First publish did not short-circuit",
                 engine.same_base_short_circuits() == 0);
    // Stone 6 cursor amendment: reset takes the cursor to 0 (relative).
    print_result("Allocator cursor reset to relative 0",
                 engine.segment_allocator().current() == 0);
}

void test_same_base_hash_short_circuit()
{
    auto feed = std::make_shared<WorkerTemplateFeed>();
    PrimeMiningEngine engine{default_cfg(), feed};

    boost::multiprecision::uint1024_t base_hash{"0xdeadbeef"};
    constexpr std::uint32_t kHeight = 100;

    feed->publish(make_epoch(make_work_package(0x1c00ffffu, base_hash, kHeight)));
    auto p1 = engine.wait_for_sessions_published_after(0, 2s);
    print_result("First publish produces 1 session (short-circuit setup)", p1 == 1);

    // Drain a few segments so the cursor has visibly advanced.
    auto& alloc = engine.segment_allocator();
    (void)alloc.next_segment_start();
    (void)alloc.next_segment_start();
    (void)alloc.next_segment_start();
    const auto cursor_before_repub = alloc.current();
    // Stone 6 cursor amendment: relative cursor advances 0 → 3*S.
    print_result("Cursor advanced 3*S after three next_segment_start() calls",
                 cursor_before_repub == 3 * kSegmentSize);

    // Republish with the same height and hashPrevBlock (same-proof-hash-space),
    // keeping the same precomputed base_hash.  The engine should keep the cursor.
    feed->publish(make_epoch(make_work_package(0x1c00ffffu, base_hash, kHeight)));
    auto p2 = engine.wait_for_sessions_published_after(p1, 2s);
    print_result("Second publish produces 2 sessions total", p2 == 2);

    print_result("same_base_short_circuits() advanced to 1",
                 engine.same_base_short_circuits() == 1);
    print_result("allocator_resets() did NOT advance on same-base republish",
                 engine.allocator_resets() == 1);
    print_result("Cursor preserved across same-base republish",
                 alloc.current() == cursor_before_repub);

    auto session = engine.current_session();
    print_result("Session block_data has expected height",
                 session && session->block_data.nHeight == kHeight);
}

void test_different_base_hash_resets_cursor()
{
    auto feed = std::make_shared<WorkerTemplateFeed>();
    PrimeMiningEngine engine{default_cfg(), feed};

    boost::multiprecision::uint1024_t h1{"0x11"};
    boost::multiprecision::uint1024_t h2{"0x22"};

    feed->publish(make_epoch(make_work_package(0x1c00ffffu, h1, 1)));
    auto p1 = engine.wait_for_sessions_published_after(0, 2s);
    print_result("First publish produces 1 session (reset setup)", p1 == 1);

    auto& alloc = engine.segment_allocator();
    (void)alloc.next_segment_start();
    (void)alloc.next_segment_start();
    // Stone 6 cursor amendment: relative cursor advances 0 → 2*S.
    print_result("Cursor advanced 2*S after pool draws",
                 alloc.current() == 2 * kSegmentSize);

    feed->publish(make_epoch(make_work_package(0x1c00ffffu, h2, 2)));
    auto p2 = engine.wait_for_sessions_published_after(p1, 2s);
    print_result("Second publish produces 2 sessions total", p2 == 2);

    print_result("allocator_resets() advanced to 2 on different base",
                 engine.allocator_resets() == 2);
    print_result("same_base_short_circuits() stayed at 0",
                 engine.same_base_short_circuits() == 0);
    // Stone 6 cursor amendment: reset is to relative 0.
    print_result("Cursor reset to relative 0 on new base",
                 alloc.current() == 0);
}

void test_null_payload_publish_is_noop()
{
    auto feed = std::make_shared<WorkerTemplateFeed>();
    PrimeMiningEngine engine{default_cfg(), feed};

    // Publish an epoch whose work_package is null — heartbeat-style.  The
    // engine consumer must wake up, see the null payload, and skip without
    // building a session or moving any counter.
    auto epoch = std::make_shared<TemplateEpoch>();  // work_package = nullptr
    feed->publish(epoch);

    // Give the consumer a chance to observe and discard.  We can't use
    // wait_for_sessions_published_after here because the engine is supposed
    // to NOT publish a session; instead poll for ~200ms then assert.
    std::this_thread::sleep_for(200ms);

    print_result("Null-payload publish does not advance sessions_published()",
                 engine.sessions_published() == 0);
    print_result("Null-payload publish does not move allocator_resets()",
                 engine.allocator_resets() == 0);
    print_result("Null-payload publish leaves current_session() == nullptr",
                 engine.current_session() == nullptr);
}

void test_shutdown_wakes_parked_consumer()
{
    auto feed = std::make_shared<WorkerTemplateFeed>();
    auto start = std::chrono::steady_clock::now();
    {
        PrimeMiningEngine engine{default_cfg(), feed};
        // Give the consumer a moment to park inside wait_for_epoch_after.
        std::this_thread::sleep_for(100ms);
        // Destructor exits scope → must wake and join in bounded time even
        // though no publish has occurred.
    }
    auto elapsed = std::chrono::steady_clock::now() - start;
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    print_result("Shutdown joins consumer within 1s without any publish",
                 elapsed_ms < 1000);
}

void test_register_worker_accepts_null()
{
    auto feed = std::make_shared<WorkerTemplateFeed>();
    PrimeMiningEngine engine{default_cfg(), feed};
    // Null is silently ignored — Worker_manager creation can be partial in
    // error paths and the engine must not crash on a half-built worker list.
    engine.register_worker(nullptr);
    print_result("register_worker(nullptr) does not crash", true);
}

void test_construction_rejects_null_feed()
{
    bool threw = false;
    try {
        PrimeMiningEngine engine{default_cfg(), nullptr};
        (void)engine;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    print_result("Constructor throws std::invalid_argument on null feed", threw);
}

void test_construction_rejects_zero_segment_size()
{
    auto feed = std::make_shared<WorkerTemplateFeed>();
    Engine_config cfg = default_cfg();
    cfg.segment_size = 0;
    bool threw = false;
    try {
        PrimeMiningEngine engine{cfg, feed};
        (void)engine;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    print_result("Constructor throws std::invalid_argument on segment_size==0", threw);
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared_segment_allocator: cooperative cursor concurrency tests.
// ─────────────────────────────────────────────────────────────────────────────

void test_shared_allocator_single_threaded()
{
    Shared_segment_allocator alloc{1024};
    alloc.reset(0);
    print_result("Shared reset(0): first next_segment_start() returns 0",
                 alloc.next_segment_start() == 0);
    print_result("Shared second next_segment_start() returns 1024",
                 alloc.next_segment_start() == 1024);
    print_result("Shared third next_segment_start() returns 2048",
                 alloc.next_segment_start() == 2048);
    print_result("Shared current() reflects next pending offset",
                 alloc.current() == 3072);
    alloc.reset(1'000'000);
    print_result("Shared reset(1e6) re-seeds cursor",
                 alloc.next_segment_start() == 1'000'000);
}

void test_shared_allocator_concurrent_no_dupes_no_gaps()
{
    constexpr std::uint64_t S = 32;
    constexpr int kThreads = 8;
    constexpr int kPerThread = 1000;
    Shared_segment_allocator alloc{S};
    alloc.reset(0);

    std::vector<std::vector<std::uint64_t>> per_thread(kThreads);
    for (auto& v : per_thread) v.reserve(kPerThread);

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < kPerThread; ++i) {
                per_thread[t].push_back(alloc.next_segment_start());
            }
        });
    }
    for (auto& th : threads) th.join();

    // Flatten and sort.
    std::vector<std::uint64_t> all;
    all.reserve(static_cast<std::size_t>(kThreads) * kPerThread);
    for (auto& v : per_thread) {
        all.insert(all.end(), v.begin(), v.end());
    }
    std::sort(all.begin(), all.end());

    // Expect exactly {0, S, 2S, ..., (N-1)*S} with no duplicates and no gaps.
    bool ok = true;
    for (std::size_t i = 0; i < all.size(); ++i) {
        if (all[i] != static_cast<std::uint64_t>(i) * S) {
            ok = false;
            break;
        }
    }
    print_result("Shared allocator: 8 threads × 1000 iterations yield 0..(N-1)*S, no dupes, no gaps", ok);
    print_result("Shared allocator: post-run cursor equals N*S",
                 alloc.current() == static_cast<std::uint64_t>(kThreads) * kPerThread * S);
}

void test_same_height_different_merkle_short_circuit()
{
    // Regression test for the KEEPALIVE Merkle-rotation bug:
    // LLL-TAO rotates hashMerkleRoot on every KEEPALIVE (coinbase update).
    // GetPrimeBaseHash() spans nVersion..nBits in memory which includes
    // hashMerkleRoot, so base_hash changes on every KEEPALIVE even though
    // the proof-hash space (hashPrevBlock + nHeight) has not moved.
    //
    // The same-base predicate must use hashPrevBlock + nHeight only, NOT
    // base_hash, so that KEEPALIVE republishes preserve the cooperative
    // cursor instead of triggering a full allocator reset.
    auto feed = std::make_shared<WorkerTemplateFeed>();
    PrimeMiningEngine engine{default_cfg(), feed};

    // Helper that creates a WorkPackage WITHOUT a precomputed base hash so
    // the consumer computes it via Block_data::GetPrimeBaseHash().  The
    // Merkle root is varied to simulate KEEPALIVE coinbase rotation.
    auto make_wp_no_precompute = [](std::uint32_t nbits,
                                    std::uint32_t height,
                                    std::uint64_t merkle_discriminator)
    {
        ::LLP::CBlock block;
        block.nVersion = 4;
        block.nChannel = 2;
        block.nHeight  = height;
        block.nBits    = nbits;
        block.nNonce   = 0;
        block.hashMerkleRoot = uint512_t{merkle_discriminator};
        // set_prime_base_hash() intentionally NOT called → consumer computes it.
        return std::make_shared<WorkPackage>(block, nbits);
    };

    constexpr std::uint32_t kHeight = 1'000'000u;
    constexpr std::uint32_t kNbits  = 0x1c00ffffu;

    feed->publish(make_epoch(make_wp_no_precompute(kNbits, kHeight, 1)));
    auto p1 = engine.wait_for_sessions_published_after(0, 2s);
    print_result("KEEPALIVE test: first publish produces 1 session", p1 == 1);
    print_result("KEEPALIVE test: first publish triggers allocator reset",
                 engine.allocator_resets() == 1);

    // Advance cursor a couple of steps so preservation is observable.
    auto& alloc = engine.segment_allocator();
    (void)alloc.next_segment_start();
    (void)alloc.next_segment_start();
    const auto cursor_before = alloc.current();
    print_result("KEEPALIVE test: cursor advanced 2*S",
                 cursor_before == 2 * kSegmentSize);

    // Republish with SAME height / hashPrevBlock but DIFFERENT hashMerkleRoot.
    // This is exactly what LLL-TAO does on every KEEPALIVE.  The base_hash
    // will differ (Merkle is in the ProofHash span) but the proof-hash space
    // (hashPrevBlock + nHeight) has not changed — the short-circuit must fire.
    feed->publish(make_epoch(make_wp_no_precompute(kNbits, kHeight, 2)));
    auto p2 = engine.wait_for_sessions_published_after(p1, 2s);
    print_result("KEEPALIVE test: second publish produces 2 sessions total", p2 == 2);

    print_result("KEEPALIVE test: same_base_short_circuits() == 1",
                 engine.same_base_short_circuits() == 1);
    print_result("KEEPALIVE test: allocator_resets() unchanged (cursor preserved)",
                 engine.allocator_resets() == 1);
    print_result("KEEPALIVE test: cursor preserved across Merkle-only rotation",
                 alloc.current() == cursor_before);
}

} // namespace

int main()
{
    if (!spdlog::get("logger")) {
        spdlog::create<spdlog::sinks::null_sink_mt>("logger");
    }

    std::cout << "Running cpu::PrimeMiningEngine tests...\n";

    test_lifecycle_no_publish();
    test_first_publish_builds_session();
    test_same_base_hash_short_circuit();
    test_different_base_hash_resets_cursor();
    test_same_height_different_merkle_short_circuit();
    test_null_payload_publish_is_noop();
    test_shutdown_wakes_parked_consumer();
    test_register_worker_accepts_null();
    test_construction_rejects_null_feed();
    test_construction_rejects_zero_segment_size();

    std::cout << "Running cpu::Shared_segment_allocator tests...\n";
    test_shared_allocator_single_threaded();
    test_shared_allocator_concurrent_no_dupes_no_gaps();

    std::cout << "\nResult: " << (tests_run - tests_failed) << "/" << tests_run << " passed\n";
    return tests_failed == 0 ? 0 : 1;
}
