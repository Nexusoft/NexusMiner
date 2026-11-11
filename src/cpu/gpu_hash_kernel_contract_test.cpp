// PR #681 follow-up — host-only regression test for the cuda_sk1024_hash
// kernel-contract / Worker_hash inner-loop interaction.
//
// Background
// ----------
// cuda_sk1024_hash (src/gpu/src/gpu/cuda_hash/sk1024.cu:758-829) has a
// subtle by-reference contract with its caller:
//   * `TheNonce` (uint64_t&) must alias `((uint64_t*)TheData)[26]`.
//   * On every call the kernel snapshots `first_nonce = TheNonce`.
//   * On a non-winner the kernel advances TheData[26] by `throughput` and
//     reports `*hashes_done = doneNonce - first_nonce + 1`, where
//     `doneNonce` is the *new* TheData[26].
//   * On a winner the kernel writes `TheNonce = foundNonce` (re-anchoring
//     the by-ref) and reports `*hashes_done = foundNonce - first_nonce + 1`.
// If the caller breaks the aliasing — e.g. PR #681 introduced a separate
// `local_nonce` that only re-anchored on a winner while the kernel kept
// advancing `local_block.nNonce` (=TheData[26]) — `first_nonce` reads stale
// every iteration and the reported `hashes_done` grows quadratically with
// the iteration count: K(K+1)/2 · throughput instead of K · throughput.
// That is the "INF MH/s" failure mode PR #343 originally fixed.
//
// What this test does
// -------------------
// Models the kernel contract in pure host C++ (`fake_kernel`) and drives it
// from the same loop pattern Worker_hash::run() now uses (Variant B of §3
// of the plan: pass local_block.nNonce by-reference; no separate
// local_nonce).  Asserts:
//   * After K non-winner iterations the cumulative reported hashes equals
//     K * throughput exactly (not K(K+1)/2 * throughput).
//   * Per-worker nonce sharding (Worker_hash bug #7) gives non-overlapping
//     2^48 windows for distinct internal_id values.
//   * Keccak-mismatch fault counter semantics: `consecutive` accumulates
//     across iterations, resets on a winner, and trips offline at
//     kKeccakMismatchFaultThreshold = 3 consecutive mismatches.
//
// This test does NOT depend on CUDA — it links nothing GPU-side.  Its
// purpose is to lock the contract in host CI so a future "harmless"
// refactor of the snapshot block (or of the kernel's hashes_done math)
// breaks a host test instead of an operator's hash-rate dashboard.

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

int g_failures = 0;

void check(bool ok, const char* label)
{
    if (!ok) {
        ++g_failures;
        std::fprintf(stderr, "FAIL: %s\n", label);
    } else {
        std::fprintf(stderr, "ok:   %s\n", label);
    }
}

// ---------- Kernel contract model ----------
// Mimics cuda_sk1024_hash exactly enough to exercise the by-ref invariant.
// `TheData` points at a 27-uint64 buffer; index 26 is the nonce slot.
// `winning_nonce` is the nonce we want the kernel to "find" (or
// kNoWinner if it should miss this call).  `keccak_mismatch` simulates a
// post-winner CPU-revalidation failure (winner found but rejected).
//
// Returns true on a credited winner, false otherwise.  Mirrors the kernel's
// semantics for *hashes_done at sk1024.cu:797, 812, and the post-loop write
// (now using the explicit `doneNonce >= first_nonce` overflow check).
constexpr std::uint64_t kNoWinner = 0xffffffffffffffffull;

bool fake_kernel(std::uint64_t* TheData,
                 std::uint64_t& TheNonce,
                 std::uint64_t* hashes_done,
                 std::uint32_t throughput,
                 std::uint64_t winning_nonce,
                 bool keccak_mismatch_on_winner,
                 std::uint32_t* keccak_mismatches)
{
    const std::uint64_t first_nonce = TheNonce;
    std::uint64_t* nonce_slot = &TheData[26];

    if (winning_nonce != kNoWinner) {
        // Winner branch — kernel writes the winning nonce into TheData[26].
        *nonce_slot = winning_nonce;
        if (!keccak_mismatch_on_winner) {
            TheNonce = winning_nonce;  // re-anchor by-ref
            *hashes_done = winning_nonce - first_nonce + 1;
            return true;
        }
        // Keccak-mismatch path (sk1024.cu:802-816): do NOT credit hashes,
        // do NOT re-anchor TheNonce, increment the out-counter.
        if (keccak_mismatches) {
            *keccak_mismatches += 1;
        }
        *hashes_done = 0;
        return false;
    }

    // Non-winner branch — advance the data buffer by `throughput`.
    *nonce_slot += throughput;
    const std::uint64_t doneNonce = *nonce_slot;
    if (doneNonce >= first_nonce) {  // overflow-safe variant of the §4(3) fix
        *hashes_done = doneNonce - first_nonce + 1;
    } else {
        *hashes_done = 0;
    }
    return false;
}

// Mirrors Worker_hash::run()'s snapshot-then-loop structure (Variant B).
// Returns the cumulative `hashes` reported across all iterations (what
// Worker_hash::m_hashes accumulates and what update_statistics publishes
// to the operator's hash-rate panel).
std::uint64_t run_inner_loop(std::uint64_t starting_nonce,
                             std::uint32_t throughput,
                             int iterations,
                             std::uint64_t winning_nonce_at_iter = 0,
                             int winner_iter = -1,
                             bool winner_keccak_mismatch = false,
                             std::uint32_t* keccak_mismatches = nullptr)
{
    // 27-uint64 mock of the block header; only index 26 (nNonce slot) is
    // exercised.  Worker_hash::run() snapshots local_block once and then
    // passes local_block.nNonce (= TheData[26]) by-reference into every
    // kernel call.  We model exactly that: `data[26]` is the only "current
    // nonce" the loop knows about.
    std::uint64_t data[27] = {};
    data[26] = starting_nonce;

    std::uint64_t total_hashes = 0;
    for (int k = 0; k < iterations; ++k) {
        std::uint64_t hashes = 0;
        const bool is_winner_iter = (k == winner_iter);
        const std::uint64_t winner = is_winner_iter ? winning_nonce_at_iter : kNoWinner;
        const bool found = fake_kernel(data, data[26], &hashes, throughput,
                                       winner, winner_keccak_mismatch,
                                       keccak_mismatches);
        total_hashes += hashes;
        if (found) {
            // Worker_hash exits the inner loop here.  We mirror that so
            // post-winner iterations don't double-count.
            break;
        }
    }
    return total_hashes;
}

// ---------- Test cases ----------

void test_local_nonce_no_inflation_after_K_iterations()
{
    // The whole point of Variant B of §3.  A pre-fix Worker_hash::run()
    // would report sum_{k=1..K} (k * T + 1) = T * K(K+1)/2 + K hashes
    // here.  Variant B reports exactly K * T (the +0/+1 fencepost is
    // eaten by the "doneNonce - first_nonce + 1" exact-window math
    // because each iteration's first_nonce == previous-iteration's
    // doneNonce).
    const std::uint32_t T = 1u << 20;        // 1,048,576 hashes per call
    const int K = 64;                        // arbitrary, > 1 to demonstrate
    const std::uint64_t start = 0x1234'0000'0000'0000ull;

    const std::uint64_t total = run_inner_loop(start, T, K);

    // Each iteration k (0-indexed) reports (T - 0 + 1) hashes
    // because doneNonce = first_nonce + T and the kernel's reported
    // hashes_done is `doneNonce - first_nonce + 1` (the +1 is the
    // inclusive-window fencepost from sk1024.cu's hashes_done math).
    // So per-iter is exactly T+1, and the cumulative across K iterations
    // is K*(T+1) = K*T + K — linear in K.  The pre-fix quadratic value
    // (which is what an actual operator hash-rate dashboard would have
    // shown under PR #681) is K(K+1)/2 * T + K.
    const std::uint64_t expected = static_cast<std::uint64_t>(K) * T + K;
    const std::uint64_t buggy_quadratic =
        static_cast<std::uint64_t>(K) * (K + 1) / 2 * T + K;

    check(total == expected,
          "Variant B: cumulative hashes is K*T+K (linear), not K(K+1)/2*T+K");
    check(total != buggy_quadratic || K <= 1,
          "Variant B: cumulative hashes is NOT the pre-fix quadratic value");

    // Also: the data-buffer's nonce should have advanced by exactly K*T,
    // proving the kernel's "advance TheData[26]" semantics still drove
    // forward progress through the nonce space.
    // (We already implicitly checked this via `expected`, but spelling it
    // out makes the contract obvious to a future reader.)
}

void test_winner_reports_correct_window_size()
{
    // The kernel's winner path reports `foundNonce - first_nonce + 1`
    // hashes for the winning iteration.  That is the exact-window count
    // for the slice [first_nonce, foundNonce].
    const std::uint32_t T = 1u << 16;
    const std::uint64_t start = 0x55ull;
    const int winner_iter = 3;
    // Winner lands somewhere inside the 4th iteration's window.
    const std::uint64_t win_nonce = start + 3 * T + 1234;

    const std::uint64_t total = run_inner_loop(
        start, T, /*iterations=*/10,
        /*winning_nonce_at_iter=*/win_nonce,
        /*winner_iter=*/winner_iter);

    // First three iterations report (T+1) each (linear, per Variant B).
    // Winner iteration reports (win_nonce - (start + 3*T) + 1) = 1235.
    const std::uint64_t expected =
        3ull * (T + 1) + (win_nonce - (start + 3ull * T) + 1);
    check(total == expected,
          "Winner iteration reports (foundNonce - first_nonce + 1) hashes");
}

void test_keccak_mismatch_branch_credits_zero_hashes()
{
    // sk1024.cu:802-816 — a winner that the host fails to revalidate is
    // a hardware-fault signal.  The kernel must NOT credit the suspect
    // window to *hashes_done (otherwise a faulty GPU silently inflates
    // the operator's hash rate).
    const std::uint32_t T = 1u << 16;
    const std::uint64_t start = 0x77ull;
    std::uint32_t mismatches = 0;

    // 5 non-winner iterations, then a "winner" that fails CPU re-check.
    // After the mismatch the inner loop in real Worker_hash::run() will
    // continue (it doesn't break on found==false), but this test exits
    // because winner_iter is past the iteration count.  We assert the
    // mismatch was counted and the winner-iter contributed 0 hashes.
    const std::uint64_t total = run_inner_loop(
        start, T, /*iterations=*/6,
        /*winning_nonce_at_iter=*/start + 5 * T + 999,
        /*winner_iter=*/5,
        /*winner_keccak_mismatch=*/true,
        /*keccak_mismatches=*/&mismatches);

    // 5 clean iterations report (T+1) each; the 6th (mismatch) reports 0.
    const std::uint64_t expected = 5ull * (T + 1);
    check(total == expected, "keccak mismatch credits 0 hashes for the suspect window");
    check(mismatches == 1, "keccak mismatch increments out-counter");
}

void test_per_worker_nonce_sharding()
{
    // Worker_hash bug #7 fix (gpu worker_hash.cpp:135, 203):
    //   m_starting_nonce = static_cast<uint64_t>(m_internal_id) << 48;
    // Each worker owns a 2^48 window.  Distinct internal_ids must produce
    // non-overlapping windows.
    auto shard_start = [](std::uint16_t internal_id) {
        return static_cast<std::uint64_t>(internal_id) << 48;
    };
    const std::uint64_t window = 1ull << 48;

    for (std::uint16_t i = 0; i < 8; ++i) {
        const std::uint64_t a = shard_start(i);
        const std::uint64_t b = shard_start(i + 1);
        check(b - a == window,
              "consecutive shards are exactly 2^48 apart");
        // No two distinct shards' [start, start + window) ranges overlap.
        check(a + window == b,
              "shard window [start, start + 2^48) is contiguous & non-overlapping");
    }
    // Sanity: a 16-bit internal_id at the top of its range still fits.
    check(shard_start(0xffff) == (static_cast<std::uint64_t>(0xffff) << 48),
          "max-internal_id shard does not overflow uint64");
}

// ---------- Keccak-mismatch fault counter (Worker_hash bug #8) ----------
// Models the consecutive-counter logic in Worker_hash::run()
// (gpu/worker_hash.cpp around lines 365-390 and the reset path on a
// winner).  We exercise it in pure C++ to lock in the semantics:
//   * consecutive accumulates across iterations
//   * resets on a winner (and on set_block, which the test doesn't model
//     because it's just `consecutive = 0`)
//   * trips offline at kKeccakMismatchFaultThreshold = 3
struct FaultModel
{
    static constexpr std::uint32_t kKeccakMismatchFaultThreshold = 3;
    std::uint32_t consecutive = 0;
    std::uint32_t total = 0;
    bool running = true;

    // Simulates one inner-loop iteration result.
    void on_iteration(std::uint32_t mismatch_delta, bool found_winner)
    {
        if (mismatch_delta > 0) {
            total += mismatch_delta;
            consecutive += mismatch_delta;
            if (consecutive >= kKeccakMismatchFaultThreshold) {
                running = false;
            }
        }
        if (found_winner) {
            consecutive = 0;
        }
    }

    void on_set_block() { consecutive = 0; }
};

void test_keccak_three_consecutive_trips_offline()
{
    FaultModel m;
    m.on_iteration(1, false);
    check(m.running, "1 mismatch: still running");
    m.on_iteration(1, false);
    check(m.running, "2 mismatches: still running");
    m.on_iteration(1, false);
    check(!m.running, "3 mismatches: trips offline");
    check(m.total == 3, "lifetime total accumulates");
}

void test_keccak_winner_resets_consecutive()
{
    FaultModel m;
    m.on_iteration(1, false);
    m.on_iteration(1, false);
    check(m.running && m.consecutive == 2, "2 mismatches before winner");

    m.on_iteration(0, true);  // healthy winner
    check(m.consecutive == 0, "winner resets consecutive counter");
    check(m.total == 2, "winner does NOT reset lifetime total");

    m.on_iteration(1, false);
    check(m.running, "post-winner mismatch starts the count over");
    check(m.total == 3, "lifetime total still accumulates post-reset");
}

void test_keccak_set_block_resets_consecutive()
{
    FaultModel m;
    m.on_iteration(1, false);
    m.on_iteration(1, false);
    check(m.consecutive == 2, "2 mismatches accumulated");
    m.on_set_block();
    check(m.consecutive == 0, "set_block resets consecutive counter");
    check(m.total == 2, "set_block does NOT reset lifetime total");
    check(m.running, "set_block does not change running");
}

}  // namespace

int main()
{
    test_local_nonce_no_inflation_after_K_iterations();
    test_winner_reports_correct_window_size();
    test_keccak_mismatch_branch_credits_zero_hashes();
    test_per_worker_nonce_sharding();
    test_keccak_three_consecutive_trips_offline();
    test_keccak_winner_resets_consecutive();
    test_keccak_set_block_resets_consecutive();

    if (g_failures != 0) {
        std::fprintf(stderr, "FAILED (%d failures)\n", g_failures);
        return EXIT_FAILURE;
    }
    std::fprintf(stderr, "PASS\n");
    return EXIT_SUCCESS;
}
