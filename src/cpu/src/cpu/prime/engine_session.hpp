#ifndef NEXUSMINER_CPU_PRIME_ENGINE_SESSION_HPP
#define NEXUSMINER_CPU_PRIME_ENGINE_SESSION_HPP

#include "block.hpp"        // Block_data
#include "worker.hpp"       // Worker::Block_found_handler

#include <boost/multiprecision/cpp_int.hpp>

#include <atomic>
#include <cstdint>

namespace nexusminer {
namespace cpu {

// ─────────────────────────────────────────────────────────────────────────────
// Stone 5 — Immutable per-template "session" owned by PrimeMiningEngine.
// ─────────────────────────────────────────────────────────────────────────────
// One EngineSession describes everything the engine pool needs to mine a
// single template:
//   * which template (epoch_id, base_hash, block_data, nbits)
//   * what nonce space to start from (starting_nonce)
//   * where to send a found block (on_found + internal_id_for_solution)
//
// Sessions are immutable once published.  The engine swaps them atomically
// (atomic shared_ptr operations) when WorkerTemplateFeed
// advances; pool threads load a session at the top of every segment and rebind
// when its epoch_id differs from the one they're currently mining.
//
// `sieve_start` is intentionally NOT stored here in Stone 5: the per-pool-
// thread Sieve owns its own rounded sieve_start as today (`Sieve::prepare`
// returns the rounded value for the caller to keep).  Stone 6, when it adds
// the pool, will compute sieve_start once on the leader thread and either
// thread it through here or keep it per-pool-thread — the choice depends on
// whether all pool threads share one Sieve or one per thread.  Either way,
// nothing in Stone 5 fixes that decision.
// ─────────────────────────────────────────────────────────────────────────────
struct EngineSession
{
    using uint1k = boost::multiprecision::uint1024_t;

    // Mirrors WorkerTemplateFeed::TemplateEpoch::epoch_id so pool threads
    // can do a single relaxed compare to detect "is this still my session?".
    std::uint64_t epoch_id{0};

    // Snapshot of the block payload taken once on the engine consumer
    // thread.  Pool threads copy this when constructing the per-found-block
    // Block_data they hand to on_found.
    Block_data block_data;

    // Prime base hash for this template (extracted from
    // WorkPackage::get_prime_base_hash() if precomputed, otherwise derived
    // from block_data.GetPrimeBaseHash()).  All pool threads use the same
    // base_hash; only their nonce offset differs.
    uint1k base_hash{};

    // Channel-level starting nonce.  Today (Worker_prime) this is
    // `internal_id << 48` per worker.  Under the engine the channel uses one
    // representative starting_nonce (typically the lowest registered
    // worker's nonce seed) so the cooperative segment allocator hands out a
    // single contiguous space.
    std::uint64_t starting_nonce{0};

    // Difficulty for this template; copied so on_found callbacks can
    // serialise the right nBits without touching the live work_package.
    std::uint32_t nbits{0};

    // Found-block callback shared across all pool threads consuming this
    // session.  Captured once per template (was per worker).
    Worker::Block_found_handler on_found;

    // Which Worker_prime should be credited with a solution found in this
    // session.  Documented choice: the channel's representative worker
    // (lowest-numbered registered Worker_prime) — per-thread attribution
    // loses meaning when N threads cooperate on one nonce space.
    std::uint32_t internal_id_for_solution{0};

    // Stone 6 — found-block consumed signal, mirroring TemplateEpoch::consumed.
    //
    // The first pool thread to dispatch a found block from this session calls
    // mark_consumed() *before* asio::post() so every other pool thread, on its
    // next session re-check, observes is_consumed() == true and idles instead
    // of grinding the spent template.  This is the single-found-block-wins
    // contract: one solution drains the entire pool for this session.
    //
    // mutable + atomic for the same reason as TemplateEpoch::consumed: the
    // EngineSession is published as shared_ptr<const EngineSession> so the
    // immutable work payload is read wait-free, but the consumed transition
    // (false → true, one-way) is allowed to race past const-correctness.
    mutable std::atomic<bool> consumed{false};

    bool is_consumed() const noexcept
    {
        return consumed.load(std::memory_order_acquire);
    }

    // Idempotent — calling more than once is harmless.  Returns the previous
    // value so callers can implement "first observer" semantics if desired.
    bool mark_consumed() const noexcept
    {
        return consumed.exchange(true, std::memory_order_acq_rel);
    }
};

} // namespace cpu
} // namespace nexusminer

#endif // NEXUSMINER_CPU_PRIME_ENGINE_SESSION_HPP
