#ifndef NEXUSMINER_CPU_PRIME_PRIME_MINING_ENGINE_HPP
#define NEXUSMINER_CPU_PRIME_PRIME_MINING_ENGINE_HPP

#include "cpu/prime/engine_session.hpp"
#include "cpu/prime/segment_allocator.hpp"
#include "stats/prime_stats_snapshot.hpp"

#include <spdlog/spdlog.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

// Forward-declare asio::io_context to keep <asio.hpp> out of this header
// (compile-time win — pulled in by the .cpp where it's actually used).
namespace asio { class io_context; }

namespace nexusminer {

class WorkerTemplateFeed;  // worker/template_feed.hpp

namespace cpu {

class Worker_prime;  // forward decl — registered for stats fan-in (Stone 7).
class Sieve;         // forward decl — per-pool-thread sieve owner (Stone 6.8 fan-in).

// ─────────────────────────────────────────────────────────────────────────────
// Stone 5 — PrimeMiningEngine: session creation
// ─────────────────────────────────────────────────────────────────────────────
// One instance per CPU prime channel.  In Stone 5 the engine implements just
// the session-creation slice of the Option-3 plan:
//
//   * Subscribes to WorkerTemplateFeed via a single consumer thread.
//   * On every new TemplateEpoch, builds an immutable EngineSession and
//     publishes it through atomic shared_ptr operations.
//   * Owns one Shared_segment_allocator (the cooperative cursor) that Stone 6
//     pool threads will pull segment offsets from.  reset() runs on the
//     consumer thread before each new session is published; pool threads only
//     ever call next_segment_start() concurrently.
//   * Implements the same-base-hash short-circuit (preserve search progress
//     when the new template targets the exact same prime proof-hash space)
//     so the allocator is NOT reset on a duplicate base_hash — matching the
//     existing per-worker `same_active_search` optimisation in Worker_prime.
//
// Pool threads, found-block submission, and Worker_prime adapter migration
// are explicitly out of scope for Stone 5 — Stones 6 and 7 build on this
// session-creation primitive without changing its public contract.  The
// engine therefore does not yet drive any sieving work; it only tracks the
// currently-active session and exposes it for tests and future pool threads.
// ─────────────────────────────────────────────────────────────────────────────
struct Engine_config
{
    // Cooperative segment length handed out by Shared_segment_allocator —
    // typically Sieve::get_segment_size().  Constant for the engine's
    // lifetime.
    std::uint64_t segment_size{0};

    // Channel-level starting nonce baseline.  Copied into every published
    // EngineSession::starting_nonce so pool threads compute their per-segment
    // sieve start from a deterministic point.  Stone 6: the cooperative
    // cursor is RELATIVE — it always begins at 0 on a new template; pool
    // threads add `session->starting_nonce` themselves at use time.
    std::uint64_t channel_starting_nonce{0};

    // Which Worker_prime gets credited with a found block from this engine.
    // Stone 7 will normally set this to the lowest registered worker's
    // m_internal_id; tests can pass any sentinel value.
    std::uint32_t internal_id_for_solution{0};

    // Stone 6 — pool sieve thread count.
    //   * 0 (default): auto-derive from std::thread::hardware_concurrency(),
    //     capped at PrimeMiningEngine::pool_threads_max_auto_cap to avoid
    //     pathological stack/sieve memory blow-up on very large machines.
    //   * non-zero: honored as-is; values exceeding hardware concurrency
    //     produce a warning at construction.
    std::uint32_t pool_threads{0};

    // Stone 6 — io_context for found-block dispatch.  Pool threads MUST NEVER
    // call on_found directly; instead they asio::post(*io_context, ...) so
    // the callback runs on the io_context thread and the sieve pipeline
    // never blocks on network/submission I/O.  When this is null the engine
    // runs in **session-only mode**: pool threads are NOT spawned (regardless
    // of `pool_threads`), only the consumer thread runs, and all pool-thread
    // diagnostic counters stay at zero.  Stored as shared_ptr so the engine
    // keeps the io_context alive for the duration of any in-flight posted
    // callbacks.
    std::shared_ptr<asio::io_context> io_context;

    // Stone 6.5 — number of contiguous segments drawn atomically from the
    // cooperative cursor per pull.  Chunking amortises the expensive
    // Sieve::calculate_starting_multiples() call over `pool_chunk_segments`
    // segments instead of paying it once per segment, which is what tanked
    // Stone 6 throughput in production.  Default is 64 (≈64× amortisation
    // while keeping cooperative-cursor latency response under a few hundred
    // milliseconds — one chunk worth of work).  Validated to [1, 1024] in
    // the engine constructor; a value of 1 reproduces the (broken) per-
    // segment behaviour and exists ONLY as a regression-test seam.  This is
    // NOT a TOML knob and NOT an operator-facing setting.
    std::uint32_t pool_chunk_segments{64};

    // ── Test seams (defaults match production behaviour) ────────────────────
    //
    // Bypass the real Sieve construction and pipeline in pool threads.  The
    // pool loop still draws segments and exercises the full session-rebind /
    // epoch-recheck / dispatch logic; it just doesn't run the (heavy) prime
    // sieve itself.  Used by prime_mining_engine_pool_test to keep tests
    // deterministic and fast.  Production callers must leave this false.
    bool test_skip_sieve{false};

    // When true (and only meaningful with test_skip_sieve), every pool-thread
    // segment synthesises one fake chain candidate that passes through the
    // full asio::post + mark_consumed dispatch path.  Lets the candidates-
    // dispatched test assert exact counts without depending on the actual
    // prime distribution of an arbitrary base hash.
    bool test_force_candidate_per_segment{false};

    // Test seam: artificially widen the race window between drawing a
    // segment from the cooperative cursor and the post-segment session
    // re-check.  Pool threads sleep for this duration immediately AFTER
    // sieve_segment() / find_chains() / test_chains() but BEFORE the
    // re-load of current_session().  Defaults to zero (no latency).
    //
    // Used by prime_mining_engine_pool_test::test_heavy_churn_drives_discards
    // to make the discard race deterministic on fast hosts: without it, a
    // pool thread can complete the (no-op test) sieve faster than the
    // publisher can rotate base_hash, producing zero discards and a flaky
    // failure.  Production callers must leave this at zero.
    std::chrono::microseconds simulated_segment_latency{0};
};

class PrimeMiningEngine
{
public:
    // Auto-derive cap for Engine_config::pool_threads == 0.  Public so tests
    // can assert against it without duplicating the magic number.
    static constexpr std::uint32_t pool_threads_max_auto_cap = 32;

    // Construct and start the consumer thread.  The feed must outlive the
    // engine: Worker_manager owns one feed per worker batch and (per Stone 4)
    // resets it only after every worker is destroyed; engine destruction is
    // sequenced before that reset.
    PrimeMiningEngine(Engine_config cfg,
                      std::shared_ptr<WorkerTemplateFeed> feed);

    // Joins the consumer thread cleanly.  Sets shutdown, wakes the feed CV,
    // joins.  Safe to call on a never-started engine (constructor failed).
    ~PrimeMiningEngine();

    PrimeMiningEngine(const PrimeMiningEngine&) = delete;
    PrimeMiningEngine& operator=(const PrimeMiningEngine&) = delete;
    PrimeMiningEngine(PrimeMiningEngine&&) = delete;
    PrimeMiningEngine& operator=(PrimeMiningEngine&&) = delete;

    // Register a Worker_prime so Stone 7 stats fan-in can locate it from the
    // engine.  In Stone 5 the engine only stores the shared_ptr; it does not
    // call into the worker.  Safe to call from Worker_manager during
    // create_workers_locked() (no thread is reading m_registered yet).
    void register_worker(std::shared_ptr<Worker_prime> worker);

    // ── Read-only accessors ──────────────────────────────────────────────

    // Latest published session, or nullptr if no template has arrived yet.
    // Pool threads (Stone 6) call this at the top of every segment.
    std::shared_ptr<const EngineSession> current_session() const
    {
        return std::atomic_load_explicit(&m_session, std::memory_order_acquire);
    }

    // The shared cursor.  Owned by the engine; Stone 6 pool threads call
    // next_segment_start() on it directly.  Returned by reference because the
    // engine outlives the pool threads.
    Segment_allocator& segment_allocator() { return *m_segment_allocator; }

    // Diagnostic counters.  Read-only from outside the engine; useful for
    // tests and for the eventual stats-printer "engine churn" rows.
    std::uint64_t sessions_published() const
    {
        return m_sessions_published.load(std::memory_order_relaxed);
    }
    std::uint64_t same_base_short_circuits() const
    {
        return m_same_base_short_circuits.load(std::memory_order_relaxed);
    }
    std::uint64_t allocator_resets() const
    {
        return m_allocator_resets.load(std::memory_order_relaxed);
    }

    // ── Stone 6 — pool-thread diagnostic counters ──────────────────────────
    std::uint64_t segments_processed() const
    {
        return m_segments_processed.load(std::memory_order_relaxed);
    }
    std::uint64_t segments_discarded_epoch_changed() const
    {
        return m_segments_discarded_epoch_changed.load(std::memory_order_relaxed);
    }
    std::uint64_t segments_skipped_consumed() const
    {
        return m_segments_skipped_consumed.load(std::memory_order_relaxed);
    }
    std::uint64_t candidates_dispatched() const
    {
        return m_candidates_dispatched.load(std::memory_order_relaxed);
    }
    std::uint64_t pool_threads_running() const
    {
        return m_pool_threads_running.load(std::memory_order_relaxed);
    }
    std::uint64_t pool_threads_crashed() const
    {
        return m_pool_threads_crashed.load(std::memory_order_relaxed);
    }
    std::uint32_t pool_thread_count() const { return m_pool_thread_count; }

    // Stone 6.5 — chunk-amortisation diagnostic counters.  Operators can
    // confirm the fix is live by checking that
    //   starting_multiples_calls ≈ chunks_drawn          (NOT segments_processed)
    // Pre-fix the ratio was ~1:1 with segments_processed; post-fix it is
    // ~1:pool_chunk_segments.
    std::uint64_t chunks_drawn() const
    {
        return m_chunks_drawn.load(std::memory_order_relaxed);
    }
    std::uint64_t starting_multiples_calls() const
    {
        return m_starting_multiples_calls.load(std::memory_order_relaxed);
    }
    std::uint32_t pool_chunk_segments() const { return m_pool_chunk_segments; }

    // ── Stone 7 — stats fan-in snapshot ────────────────────────────────────
    //
    // Aggregated counters published once per snapshot, designed to be split
    // across registered Worker_prime instances using a floor-plus-remainder
    // partition.  Worker_prime's update_statistics() (under engine mode)
    // calls this and computes its own share — no per-thread attribution is
    // attempted (cooperative pool sieving makes per-thread credit
    // meaningless, see EngineSession comments).
    struct Engine_stats_snapshot
    {
        std::uint64_t segments_processed{0};
        std::uint64_t candidates_dispatched{0};
        std::uint64_t segment_size{0};      // copied from Engine_config
        std::uint64_t sessions_published{0};
        std::uint32_t nbits{0};             // current session nBits, 0 if none
        std::uint32_t pool_thread_count{0};
        // Stone 6.5 — chunk-amortisation diagnostics.
        std::uint64_t chunks_drawn{0};
        std::uint64_t starting_multiples_calls{0};
        std::uint64_t segments_discarded_epoch_changed{0};
        std::uint64_t segments_skipped_consumed{0};
        std::uint64_t same_base_short_circuits{0};
        std::uint64_t allocator_resets{0};
        std::uint64_t pool_threads_running{0};
        std::uint64_t pool_threads_crashed{0};
        std::uint32_t pool_chunk_segments{0};

        // Stone 6.8 — chain-length histogram aggregated bucket-wise across
        // every live pool thread's Sieve::m_chain_histogram.  Buckets are
        // saturating-summed into stats::Prime_histogram so the stats printer
        // can render the same chain row in engine mode that it does in legacy
        // workers mode.  Histograms are diagnostic and intentionally NOT
        // partitioned across registered Worker_prime instances — every worker
        // sees the same aggregated view.
        stats::Prime_histogram chain_histogram{};

        // Stone 6.9 — "attempted" companion to chain_histogram, fanned-in
        // from each pool sieve's m_chain_histogram_attempted.  Bucket k =
        // count of chain candidates that had >= k sieve-survivor slots
        // available to Fermat-test (i.e. survived close_chain()'s slot
        // filter).  Combined with chain_histogram (best Fermat run achieved)
        // operators can distinguish "no length-7 candidates produced by the
        // sieve" from "many length-7 candidates, all aborted by
        // is_there_still_hope() before reaching length 7".
        stats::Prime_histogram chain_histogram_attempted{};

        // Stone 6.9 — find→test→dispatch funnel snapshot.
        std::uint64_t chains_found_by_sieve{0};
        std::uint64_t chains_pushed_long{0};
        // Stone 6.9.2 — popcount-stage funnel counter.  Always >=
        // chains_found_by_sieve in a healthy run; a sudden drop in the
        // ratio chains_found_by_sieve / popcount_windows_passed is the
        // smoking gun for a future close_chain regression.
        std::uint64_t popcount_windows_passed{0};
        std::uint64_t validate_attempts{0};
        std::uint64_t validate_rejected_base_not_prime{0};
        std::uint64_t validate_rejected_below_diff{0};
        std::uint64_t validate_rejected_malformed{0};

        // Stone 6.8 — channel-wide best chain difficulty observed since the
        // engine started.  Updated by pool threads via a CAS-loop max
        // whenever ValidatePrimeCandidate flags a candidate as meeting the
        // network's required difficulty (the same condition that triggers
        // m_segmented_sieve->m_best_chain updates in legacy workers mode).
        // Like chain_histogram, this is intentionally NOT partitioned across
        // registered Worker_prime instances — pool threads cooperate on a
        // single cursor, so "best" is a channel-wide truth and every worker
        // reports the same value (which matches operator expectations: a
        // freshly-accepted block lifts everyone's "Best" line at once).
        double best_difficulty{0.0};
    };

    Engine_stats_snapshot snapshot_stats() const;

    // Number of currently-registered Worker_prime instances (live weak_ptrs
    // are counted; expired ones are not).  Used by the stats fan-in path
    // when the manager needs the live denominator without locking.
    std::size_t registered_worker_count() const;

    // Accessor for the channel-level representative worker id (== lowest
    // registered Worker_prime's m_internal_id at construction time).
    std::uint32_t internal_id_for_solution() const { return m_cfg.internal_id_for_solution; }

    // Block until the consumer thread has processed at least one publish
    // event whose epoch_id is greater than `last_seen`.  Returns the latest
    // sessions_published() count once the wait completes.  Used by tests to
    // synchronise with the consumer without busy-spinning; production
    // callers do not need this.
    std::uint64_t wait_for_sessions_published_after(std::uint64_t last_seen,
                                                    std::chrono::milliseconds timeout);

private:
    void run_consumer();
    void run_pool_thread(std::uint32_t pool_index);
    static std::uint32_t resolve_pool_thread_count(std::uint32_t configured,
                                                   std::shared_ptr<spdlog::logger>& logger);

    Engine_config                          m_cfg;
    std::shared_ptr<WorkerTemplateFeed>    m_feed;
    std::shared_ptr<spdlog::logger>        m_logger;

    // The cooperative cursor.  Created once at construction; reset() is only
    // called on the consumer thread, next_segment_start() only by pool
    // threads (Stone 6).  Held by unique_ptr so Segment_allocator stays
    // polymorphic for tests that want to inject a fake.
    std::unique_ptr<Segment_allocator>     m_segment_allocator;

    // Use shared_ptr atomic operations for standard libraries without the
    // C++20 atomic<shared_ptr> specialization.
    std::shared_ptr<const EngineSession> m_session;

    // Worker_prime instances registered with the engine.  Read-only after
    // create_workers_locked() returns; Stone 7 uses these for stats fan-in.
    // Stored under a small mutex to support tests that register workers from
    // multiple threads, and to keep the door open for future hot-add.
    mutable std::mutex                     m_registered_mtx;
    std::vector<std::weak_ptr<Worker_prime>> m_registered;

    // Diagnostic counters.  All updates are on the consumer thread.
    std::atomic<std::uint64_t>             m_sessions_published{0};
    std::atomic<std::uint64_t>             m_same_base_short_circuits{0};
    std::atomic<std::uint64_t>             m_allocator_resets{0};

    // Stone 6 — pool-thread diagnostic counters (relaxed; updated on pool threads).
    std::atomic<std::uint64_t>             m_segments_processed{0};
    std::atomic<std::uint64_t>             m_segments_discarded_epoch_changed{0};
    std::atomic<std::uint64_t>             m_segments_skipped_consumed{0};
    std::atomic<std::uint64_t>             m_candidates_dispatched{0};
    std::atomic<std::uint64_t>             m_pool_threads_running{0};
    std::atomic<std::uint64_t>             m_pool_threads_crashed{0};

    // Stone 6.9 — find→test→dispatch funnel diagnostics.  Together with the
    // per-Sieve histograms these answer "of N chains the sieve produced,
    // M survived Fermat, K reached the dispatch gate, and L of K were
    // rejected for reason R".  All updated on pool threads with relaxed
    // ordering; read by the stats logger / Engine_stats_snapshot.
    std::atomic<std::uint64_t>             m_chains_found_by_sieve{0};
    std::atomic<std::uint64_t>             m_chains_pushed_long{0};
    // Stone 6.9.2 — popcount-stage funnel counter (see Engine_stats_snapshot).
    std::atomic<std::uint64_t>             m_popcount_windows_passed{0};
    std::atomic<std::uint64_t>             m_validate_attempts{0};
    std::atomic<std::uint64_t>             m_validate_rejected_base_not_prime{0};
    std::atomic<std::uint64_t>             m_validate_rejected_below_diff{0};
    std::atomic<std::uint64_t>             m_validate_rejected_malformed{0};

    // Stone 6.5 — chunk-amortisation diagnostic counters.
    std::atomic<std::uint64_t>             m_chunks_drawn{0};
    std::atomic<std::uint64_t>             m_starting_multiples_calls{0};

    // Stone 6.8 — channel-wide best chain difficulty observed since engine
    // startup.  std::atomic<double> in C++20 supports load/store/compare-
    // exchange but no fetch_max, so pool threads update via a CAS loop
    // (see prime_mining_engine.cpp run_pool_thread, after is_valid).
    std::atomic<double>                    m_best_difficulty{0.0};

    // Consumer-thread shutdown flag + condvar for the wait-for-publish helper.
    std::atomic<bool>                      m_shutdown{false};
    mutable std::mutex                     m_publish_mtx;
    std::condition_variable                m_publish_cv;

    // Stone 6 — pool-thread idle/wakeup CV.  Pool threads park here while the
    // engine has no usable session (null or consumed); the consumer notifies
    // after every publish and the destructor notifies on shutdown.  The hot
    // path (segment loop with a usable session) does NOT touch this mutex.
    mutable std::mutex                     m_pool_mtx;
    std::condition_variable                m_pool_cv;

    // Resolved pool-thread count (after auto-derive / cap).  Stable for the
    // engine's lifetime; exposed via pool_thread_count() for tests.
    std::uint32_t                          m_pool_thread_count{0};
    std::vector<std::thread>               m_pool_threads;

    // Stone 6.5 — resolved chunk size (validated to [1, 1024] in ctor).
    std::uint32_t                          m_pool_chunk_segments{0};

    // Option 2 (Sieve histogram concurrency) — per-pool-thread published
    // snapshot of the Sieve's chain-length histograms.  Pool threads call
    // publish_histogram_snapshot() at chunk boundaries from the OWNING
    // thread; the stats path reads only these snapshots and never touches
    // the live Sieve*.  Restores the documented invariant in
    // prime_stats_snapshot.hpp that diagnostic counters are published from
    // the owning thread.
    //
    // Holds (best, attempted) pairs as a single shared_ptr swap so the two
    // arrays are observed atomically (a half-updated pair would skew the
    // operator-facing "survival probability" view).
    struct Pool_histogram_snapshot
    {
        stats::Prime_histogram best{};
        stats::Prime_histogram attempted{};
    };
    std::unique_ptr<stats::Atomic_snapshot<Pool_histogram_snapshot>[]>
        m_pool_histogram_snapshots;

    // Stone 6.5 — periodic engine-stats info-level log line (every 30s).
    void                                   run_stats_logger();
    std::thread                            m_stats_logger;

    // Consumer thread.  Must be the LAST member so it is destroyed first
    // (and joined by the destructor body, not by the implicit member dtor).
    std::thread                            m_consumer;
};

} // namespace cpu
} // namespace nexusminer

#endif // NEXUSMINER_CPU_PRIME_PRIME_MINING_ENGINE_HPP
