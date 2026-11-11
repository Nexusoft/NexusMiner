// Stone — GPU host-side unit tests.
//
// Exercises the host-side `gpu::Chain` class introduced/upgraded in this PR:
//   * The kHopeKeepaliveThreshold (=4) policy added to is_there_still_hope
//     to mirror the CUDA cuda_chain.cu reference (and the CPU
//     chain_sieve.cpp Chain::is_there_still_hope).
//   * The fake-pass walk via get_best_fermat_chain that catches the
//     "totals are enough but max contiguous run is not" case — the
//     regression class that caused the length-T histogram bucket to read 0
//     even at length-T target on the CPU side and was latent on the GPU
//     host-fallback path.
//   * Per-instance T plumbing (m_min_chain_length, m_min_chain_report_length)
//     that replaced the previous static constexpr defaults so the host
//     fallback path tracks the same per-session T as the CUDA pipeline.
//
// This test compiles `gpu/prime/chain.cpp` standalone — it has no CUDA
// dependencies (chain.hpp only includes prime_common.hpp + std headers), so
// it runs in any host-only CI without a GPU toolchain.  The test deliberately
// does NOT exercise any kernel or device-side code; that is covered by the
// existing on-device validation runs.
//
// Note re. close_chain_min slack: as of the PR #678 follow-up that landed on
// main, mining/prime_thresholds.hpp::close_chain_min(T) returns T+1 (an
// empirical quality gate), while popcount_window_floor(T) stays at the
// strict lower bound T.  These tests exercise the host-side gpu::Chain
// hope/walk logic, which keys off the per-instance m_min_chain_length
// (== T) — not the close_chain_min slack — so the +1 does not affect
// what is asserted here.

#include "gpu/prime/chain.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

int g_failures = 0;

void check(bool ok, const char* label)
{
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", label);
    if (!ok) ++g_failures;
}

using nexusminer::gpu::Chain;
using nexusminer::gpu::Fermat_test_status;

// Helper: build a chain at base_offset with the given offsets-from-base, with
// each slot's Fermat status taken from the parallel statuses vector.  The
// first offset must be 0 (the base offset slot).
Chain make_chain(std::uint64_t base_offset,
                 const std::vector<int>& offsets,
                 const std::vector<Fermat_test_status>& statuses)
{
    Chain c(base_offset);
    // Chain ctor with base creates the offset-0 slot already.  Push back the
    // remaining offsets.
    for (std::size_t i = 1; i < offsets.size(); ++i)
        c.push_back(offsets[i]);
    // Now apply the per-slot Fermat statuses.
    c.m_prime_count = 0;
    c.m_untested_count = 0;
    for (std::size_t i = 0; i < c.m_offsets.size() && i < statuses.size(); ++i)
    {
        c.m_offsets[i].m_fermat_test_status = statuses[i];
        if (statuses[i] == Fermat_test_status::pass)
            ++c.m_prime_count;
        else if (statuses[i] == Fermat_test_status::untested)
            ++c.m_untested_count;
    }
    return c;
}

void test_default_target_fields()
{
    std::printf("test_default_target_fields\n");
    Chain c;
    // The host-side Chain still uses the legacy pre-session defaults; the
    // production path overrides them per-chain via Sieve::get_chains.  Lock
    // these defaults down so any change is intentional.
    check(c.m_min_chain_length == 8,
          "default m_min_chain_length == 8 (mining::MIN_CHAIN_LENGTH)");
    check(c.m_min_chain_report_length == 5,
          "default m_min_chain_report_length == 5 (legacy host fallback)");
    check(Chain::kHopeKeepaliveThreshold == 4,
          "kHopeKeepaliveThreshold == 4 (matches CUDA + CPU reference)");
}

void test_hope_empty_and_zero_untested()
{
    std::printf("test_hope_empty_and_zero_untested\n");
    // is_there_still_hope MUST return false if no untested offsets remain
    // (early-out, lifted directly from the CUDA reference).
    Chain c(1000);
    c.m_prime_count = 0;
    c.m_untested_count = 0;
    c.m_offsets[0].m_fermat_test_status = Fermat_test_status::fail;
    check(!c.is_there_still_hope(),
          "no untested offsets → no hope (regardless of prime_count)");

    // Even if prime_count >= keepalive, untested == 0 means we cannot do any
    // more work, so the early-out wins over the keepalive (mirrors CUDA).
    Chain c2 = make_chain(2000, {0, 2, 4, 6}, {
        Fermat_test_status::pass,
        Fermat_test_status::pass,
        Fermat_test_status::pass,
        Fermat_test_status::pass,
    });
    check(!c2.is_there_still_hope(),
          "untested==0 → false even when prime_count >= keepalive");
}

void test_hope_keepalive_threshold()
{
    std::printf("test_hope_keepalive_threshold\n");
    // 4 proven primes and at least one untested → keepalive returns true
    // unconditionally, even if the totals/walk would otherwise prune.
    Chain c = make_chain(3000, {0, 2, 4, 6, 16}, {
        Fermat_test_status::pass,
        Fermat_test_status::pass,
        Fermat_test_status::pass,
        Fermat_test_status::pass,
        Fermat_test_status::untested,  // far slot (gap=10 from prev slot)
    });
    // Force a target bigger than what the totals could ever reach (4+1 < 99)
    c.m_min_chain_length = 99;
    check(c.is_there_still_hope(),
          "prime_count >= 4 + at-least-one-untested → keepalive returns true");

    // 3 primes (just below keepalive) + 1 untested + huge target → no hope.
    Chain c2 = make_chain(4000, {0, 2, 4, 16}, {
        Fermat_test_status::pass,
        Fermat_test_status::pass,
        Fermat_test_status::pass,
        Fermat_test_status::untested,
    });
    c2.m_min_chain_length = 99;
    check(!c2.is_there_still_hope(),
          "prime_count < keepalive + totals < min → no hope");
}

void test_hope_totals_prune()
{
    std::printf("test_hope_totals_prune\n");
    // prime + untested < min → totals prune fires.
    Chain c = make_chain(5000, {0, 2}, {
        Fermat_test_status::pass,
        Fermat_test_status::untested,
    });
    c.m_min_chain_length = 8;  // 1 + 1 = 2 < 8 → prune
    check(!c.is_there_still_hope(),
          "prime + untested < m_min_chain_length → totals prune returns false");
}

void test_hope_contiguous_walk_catches_gap_split()
{
    std::printf("test_hope_contiguous_walk_catches_gap_split\n");
    // The headline regression class.  totals say we have enough material
    // (prime+untested >= min) but the offsets are split into two clusters
    // separated by more than maxGap, so no contiguous run can ever reach the
    // target length.  The naive totals-only test would say "still hope";
    // the upgraded walk catches it and returns false.
    //
    // Cluster 1: offsets {0, 2, 4, 6} all pass (4-chain)
    // BIG gap of 100 (>> maxGap=12)
    // Cluster 2: offsets {106, 108, 110, 112} untested
    //
    // Total prime+untested = 8.  With m_min_chain_length=8 the totals test
    // returns true; the contiguous-walk test returns false because no run
    // crossing the 100-unit gap is possible.
    Chain c = make_chain(6000,
        {0, 2, 4, 6, 106, 108, 110, 112},
        {
            Fermat_test_status::pass,
            Fermat_test_status::pass,
            Fermat_test_status::pass,
            Fermat_test_status::pass,
            Fermat_test_status::untested,
            Fermat_test_status::untested,
            Fermat_test_status::untested,
            Fermat_test_status::untested,
        });
    c.m_min_chain_length = 8;
    // prime_count = 4 → would hit the keepalive branch.  Force keepalive off
    // by lowering prime_count so we exercise the walk branch directly.
    c.m_prime_count = 3;  // bump 4→3 (still below keepalive)
    c.m_offsets[0].m_fermat_test_status = Fermat_test_status::untested;
    c.m_untested_count = 5;  // recompute: 4 originally untested + 1 demoted
    check(!c.is_there_still_hope(),
          "split clusters across maxGap → walk returns false even though "
          "totals are sufficient (the regression-class case)");
}

void test_hope_contiguous_walk_passes_when_run_possible()
{
    std::printf("test_hope_contiguous_walk_passes_when_run_possible\n");
    // Same shape, but the offsets are all within maxGap of their neighbours,
    // so an optimistic walk reaches m_min_chain_length and we keep the chain.
    Chain c = make_chain(7000,
        {0, 2, 4, 6, 8, 10, 12, 14},
        {
            Fermat_test_status::untested,
            Fermat_test_status::untested,
            Fermat_test_status::untested,
            Fermat_test_status::untested,
            Fermat_test_status::untested,
            Fermat_test_status::untested,
            Fermat_test_status::untested,
            Fermat_test_status::untested,
        });
    c.m_min_chain_length = 8;
    check(c.is_there_still_hope(),
          "tight cluster (all gaps <= maxGap) → walk returns true");
}

void test_per_instance_target_propagation()
{
    std::printf("test_per_instance_target_propagation\n");
    // Exercises the per-instance min_chain_length plumbing — a new chain can
    // be told its session T independently of any other chain.  This mirrors
    // what gpu::Sieve::get_chains/get_long_chains now do (they propagate T
    // from CudaChain into the host Chain so the host fallback path sees the
    // same session T as the CUDA pipeline).
    Chain c1; c1.m_min_chain_length = 7; c1.m_min_chain_report_length = 7;
    Chain c2; c2.m_min_chain_length = 9; c2.m_min_chain_report_length = 9;
    check(c1.m_min_chain_length == 7
          && c1.m_min_chain_report_length == 7
          && c2.m_min_chain_length == 9
          && c2.m_min_chain_report_length == 9,
          "per-instance m_min_chain_length / m_min_chain_report_length "
          "are independent (matches CudaChain instance-field SSOT)");
}

}  // namespace

int main()
{
    std::printf("== gpu_chain_test ==\n");
    test_default_target_fields();
    test_hope_empty_and_zero_untested();
    test_hope_keepalive_threshold();
    test_hope_totals_prune();
    test_hope_contiguous_walk_catches_gap_split();
    test_hope_contiguous_walk_passes_when_run_possible();
    test_per_instance_target_propagation();
    std::printf("== gpu_chain_test summary: %d failure(s) ==\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
