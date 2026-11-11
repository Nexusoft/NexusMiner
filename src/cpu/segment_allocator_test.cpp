// Tests for cpu::Segment_allocator — Stone 3 pluggable per-segment cursor.

#include "cpu/prime/segment_allocator.hpp"

#include <spdlog/sinks/null_sink.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
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
    if (!passed)
        ++tests_failed;
}

void test_per_worker_reset_then_three_advances()
{
    constexpr std::uint64_t S = 1024;
    constexpr std::uint64_t N = 12345;
    nexusminer::cpu::Per_worker_segment_allocator alloc{S};
    alloc.reset(N);

    const auto a = alloc.next_segment_start();
    const auto b = alloc.next_segment_start();
    const auto c = alloc.next_segment_start();

    print_result("Per_worker reset(N) → next_segment_start() returns N",   a == N);
    print_result("Per_worker second next_segment_start() returns N + S",   b == N + S);
    print_result("Per_worker third  next_segment_start() returns N + 2S",  c == N + 2 * S);
}

void test_per_worker_current_is_pre_advance_cursor()
{
    constexpr std::uint64_t S = 256;
    nexusminer::cpu::Per_worker_segment_allocator alloc{S};
    alloc.reset(0);

    const auto current_before = alloc.current();
    const auto handed_out = alloc.next_segment_start();
    const auto current_after = alloc.current();

    print_result("current() returns the cursor BEFORE the next next_segment_start() advances it",
                 current_before == handed_out);
    print_result("current() advances by segment_size after next_segment_start()",
                 current_after == current_before + S);
}

void verify_factory_yields_per_worker_semantics(const std::string& engine_mode)
{
    constexpr std::uint64_t S = 4096;
    auto alloc = nexusminer::cpu::make_segment_allocator_for_engine_mode(engine_mode, S);

    const std::string label = "make_segment_allocator_for_engine_mode(\"" + engine_mode + "\", S)";

    print_result((label + " returns non-null").c_str(), alloc != nullptr);
    if (!alloc)
        return;

    alloc->reset(0);
    const auto a = alloc->next_segment_start();
    const auto b = alloc->next_segment_start();
    const auto c = alloc->next_segment_start();

    print_result((label + " yields per-worker semantics: 0, S, 2S").c_str(),
                 a == 0 && b == S && c == 2 * S);
}

void test_factory_workers_mode()  { verify_factory_yields_per_worker_semantics("workers"); }
void test_factory_engine_mode()   { verify_factory_yields_per_worker_semantics("engine");  }
void test_factory_garbage_mode()  { verify_factory_yields_per_worker_semantics("garbage"); }
void test_factory_empty_mode()    { verify_factory_yields_per_worker_semantics("");        }

void test_shared_smoke_reset_then_three_advances()
{
    constexpr std::uint64_t S = 2048;
    constexpr std::uint64_t N = 99'000;
    nexusminer::cpu::Shared_segment_allocator alloc{S};
    alloc.reset(N);

    const auto a = alloc.next_segment_start();
    const auto b = alloc.next_segment_start();
    const auto c = alloc.next_segment_start();

    print_result("Shared reset(N) → next_segment_start() returns N",  a == N);
    print_result("Shared second next_segment_start() returns N + S",  b == N + S);
    print_result("Shared third  next_segment_start() returns N + 2S", c == N + 2 * S);
    print_result("Shared current() points at next pending offset",
                 alloc.current() == N + 3 * S);
}

// Stone 6.5 — concurrent next_segment_chunk() partition.  Many threads each
// pull many chunks; the union of (base, base + chunk*S) intervals must form
// a non-overlapping, contiguous, monotonically-increasing partition of
// [0, total) — same wait-free correctness property as next_segment_start().
void test_shared_chunk_concurrent_partition()
{
    constexpr std::uint64_t S = 4096;
    constexpr std::uint64_t CHUNK_SEGMENTS = 64;
    constexpr int N_THREADS = 8;
    constexpr int CHUNKS_PER_THREAD = 100;

    nexusminer::cpu::Shared_segment_allocator alloc{S};
    alloc.reset(0);

    std::vector<std::vector<std::uint64_t>> per_thread_bases(N_THREADS);
    std::vector<std::thread> threads;
    threads.reserve(N_THREADS);
    for (int t = 0; t < N_THREADS; ++t)
    {
        threads.emplace_back([&, t] {
            per_thread_bases[t].reserve(CHUNKS_PER_THREAD);
            for (int i = 0; i < CHUNKS_PER_THREAD; ++i)
            {
                per_thread_bases[t].push_back(alloc.next_segment_chunk(CHUNK_SEGMENTS));
            }
        });
    }
    for (auto& th : threads) th.join();

    // Collect all bases, sort, and assert they form a contiguous strictly-
    // increasing sequence at stride CHUNK_SEGMENTS * S starting at 0.
    std::vector<std::uint64_t> all_bases;
    all_bases.reserve(N_THREADS * CHUNKS_PER_THREAD);
    for (const auto& v : per_thread_bases)
    {
        all_bases.insert(all_bases.end(), v.begin(), v.end());
    }
    std::sort(all_bases.begin(), all_bases.end());

    constexpr std::uint64_t stride = CHUNK_SEGMENTS * S;
    const std::size_t expected_count = N_THREADS * CHUNKS_PER_THREAD;

    bool count_ok = all_bases.size() == expected_count;
    bool partition_ok = true;
    for (std::size_t i = 0; i < all_bases.size(); ++i)
    {
        if (all_bases[i] != static_cast<std::uint64_t>(i) * stride)
        {
            partition_ok = false;
            break;
        }
    }
    bool cursor_ok =
        alloc.current() == static_cast<std::uint64_t>(expected_count) * stride;

    print_result("Shared next_segment_chunk: all chunk bases collected",
                 count_ok);
    print_result("Shared next_segment_chunk: bases form a non-overlapping "
                 "contiguous monotonic partition of cursor space",
                 partition_ok);
    print_result("Shared next_segment_chunk: cursor advanced by total "
                 "chunks * chunk_segments * segment_size",
                 cursor_ok);
}

void test_shared_chunk_smoke()
{
    constexpr std::uint64_t S = 1024;
    constexpr std::uint64_t C = 8;
    nexusminer::cpu::Shared_segment_allocator alloc{S};
    alloc.reset(0);
    const auto a = alloc.next_segment_chunk(C);
    const auto b = alloc.next_segment_chunk(C);
    print_result("next_segment_chunk first call returns 0", a == 0);
    print_result("next_segment_chunk second call returns C*S",
                 b == C * S);
    print_result("next_segment_chunk advances cursor by C*S per call",
                 alloc.current() == 2 * C * S);
}
} // namespace

int main()
{
    // The factory may emit warn-level logs for non-canonical engine modes; the
    // logger lookup tolerates a missing "logger" but we register a null sink
    // so the warn path is exercised cleanly under test.
    if (!spdlog::get("logger"))
    {
        spdlog::create<spdlog::sinks::null_sink_mt>("logger");
    }

    std::cout << "Running cpu::Segment_allocator tests...\n";
    test_per_worker_reset_then_three_advances();
    test_per_worker_current_is_pre_advance_cursor();
    test_factory_workers_mode();
    test_factory_engine_mode();
    test_factory_garbage_mode();
    test_factory_empty_mode();
    test_shared_smoke_reset_then_three_advances();
    test_shared_chunk_smoke();
    test_shared_chunk_concurrent_partition();

    std::cout << "\nResult: " << (tests_run - tests_failed) << "/" << tests_run << " passed\n";
    return tests_failed == 0 ? 0 : 1;
}
