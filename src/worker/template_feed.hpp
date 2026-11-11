#ifndef NEXUSMINER_WORKER_TEMPLATE_FEED_HPP
#define NEXUSMINER_WORKER_TEMPLATE_FEED_HPP

#include "worker/worker.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>

namespace nexusminer {

// ─────────────────────────────────────────────────────────────────────────────
// Stone 4: Shared TemplateEpoch instead of N×set_block fanout
// ─────────────────────────────────────────────────────────────────────────────
// Each new template is published once into a shared, atomically-readable slot
// instead of being pushed into N individual worker mutex/CV chains.  Workers
// observe new work by loading the latest TemplateEpoch from this feed at the
// natural rebind point in their mining loop (top of segment, end of nonce
// batch, etc.).  This decouples template-feed latency from worker count and
// removes the parent m_worker_mutex hold across N CV notifications, which is
// the same parent mutex the stats path holds.
//
// Publish-side semantics:
//   * Single-publisher (Worker_manager template handler).
//   * publish() advances a monotonic epoch_id, atomically swaps the slot, and
//     notify_all() on the shared CV so cold-start (waiting) workers wake up.
//
// Subscribe-side semantics:
//   * Many-reader. load() acquires a coherent latest-wins snapshot.
//   * wait_for_epoch_after(last_seen, predicate) blocks until either a newer
//     epoch is published or predicate() returns true (used for shutdown
//     wake-up by the worker).
//
// This feed is *infrastructure* for Stones 5-7.  In this PR Worker_manager
// publishes for every template, but the existing per-worker
// set_block(WorkPackage, handler) shim is retained so unmigrated workers keep
// working unchanged.  The Worker_manager fanout that calls those shims now
// runs *outside* m_worker_mutex (snapshot-and-release), which already removes
// the stats-path contention called out in the Stone 4 brief.  Per-worker
// migrations to consume directly from this feed land in follow-up stones.
// ─────────────────────────────────────────────────────────────────────────────

struct TemplateEpoch
{
    // Monotonically increasing per-feed publish counter.  0 means "no epoch
    // has been published yet"; first real publish carries epoch_id == 1.
    std::uint64_t epoch_id{0};

    // Shared immutable work package (built once by Worker_manager).
    std::shared_ptr<WorkPackage> work_package;

    // Found-block callback shared across all workers consuming this epoch.
    // Captured once per template instead of re-captured per worker.
    Worker::Block_found_handler on_found;

    // ── Stone "Option D" — explicit epoch-consumed signal ────────────────────
    // Set true the moment any worker (legacy set_block shim path or future
    // feed-consumer path) dispatches a found-block for this epoch.  Flipped
    // by Worker_manager's on_found handler via mark_consumed() so that all
    // observers — including this same epoch still sitting in the feed slot —
    // see the transition atomically.
    //
    // Future feed-consumer workers consult is_consumed() (or
    // WorkerTemplateFeed::latest_unconsumed()) at the top of their mining
    // loop and idle until a fresh epoch is published, instead of grinding
    // the spent template waiting for the next BLOCK_DATA push.  Legacy
    // workers self-track an equivalent per-worker flag in their own loops;
    // this one is the cross-worker / cross-epoch source of truth.
    //
    // mutable + atomic: the slot in WorkerTemplateFeed is held as
    // shared_ptr<const TemplateEpoch> for atomic reader access, but the
    // consumed transition is one-way (false → true) and may be observed
    // racily by readers without breaking immutability of the work payload.
    mutable std::atomic<bool> consumed{false};

    bool is_consumed() const noexcept
    {
        return consumed.load(std::memory_order_acquire);
    }

    // Idempotent — calling more than once is harmless.  Returns the previous
    // value so callers can log "first observer" semantics if desired.
    bool mark_consumed() const noexcept
    {
        return consumed.exchange(true, std::memory_order_acq_rel);
    }
};

class WorkerTemplateFeed
{
public:
    WorkerTemplateFeed() = default;

    WorkerTemplateFeed(const WorkerTemplateFeed&) = delete;
    WorkerTemplateFeed& operator=(const WorkerTemplateFeed&) = delete;
    WorkerTemplateFeed(WorkerTemplateFeed&&) = delete;
    WorkerTemplateFeed& operator=(WorkerTemplateFeed&&) = delete;

    // Publish a new TemplateEpoch.  The epoch_id field of `epoch` is
    // overwritten with the next monotonic value, so callers may pass in a
    // freshly-constructed shared_ptr without pre-filling epoch_id.
    //
    // Single-publisher only.  Returns the assigned epoch_id.
    std::uint64_t publish(std::shared_ptr<TemplateEpoch> epoch)
    {
        if (!epoch)
        {
            // No state mutation on the null path — relaxed read is sufficient
            // and matches the relaxed load below that seeds `next_id`.
            return m_latest_epoch_id.load(std::memory_order_relaxed);
        }

        const std::uint64_t next_id = m_latest_epoch_id.load(std::memory_order_relaxed) + 1;
        epoch->epoch_id = next_id;

        store_epoch(std::shared_ptr<const TemplateEpoch>(std::move(epoch)));
        m_latest_epoch_id.store(next_id, std::memory_order_release);

        // Wake any worker waiting in wait_for_epoch_after().
        //
        // The empty critical section is intentional: the predicate
        // (m_latest_epoch_id) is updated above WITHOUT holding m_wake_mtx,
        // so a waiter that has just observed the old predicate value but has
        // not yet parked on m_wake_cv could otherwise miss the notify_all()
        // (classic lost-wakeup race).  Briefly acquiring m_wake_mtx here
        // serialises with the waiter's `cv.wait(lock, predicate)` block: by
        // the time we own the lock, the waiter is either (a) still in its
        // predicate re-evaluation under the same lock and will observe the
        // new id when we release, or (b) fully parked on the CV and will
        // be reached by the notify_all() below.
        {
            std::lock_guard<std::mutex> lock(m_wake_mtx);
        }
        m_wake_cv.notify_all();

        return next_id;
    }

    // Atomic read of the latest published epoch. Returns nullptr when
    // nothing has been published yet.
    std::shared_ptr<const TemplateEpoch> load() const
    {
#if defined(__cpp_lib_atomic_shared_ptr) && __cpp_lib_atomic_shared_ptr >= 201711L
        return m_slot.load(std::memory_order_acquire);
#else
        std::scoped_lock lock(m_slot_mtx);
        return m_slot;
#endif
    }

    // Atomic read of the latest published epoch IFF it has not yet been
    // marked consumed via TemplateEpoch::mark_consumed().  Returns nullptr
    // when nothing has been published yet OR when the latest epoch's
    // found-block handler has already fired.
    //
    // Intended for Option D feed-consumer workers: they call this at the
    // top of each mining-loop iteration and idle (via
    // wait_for_epoch_after) instead of hashing the spent template.  Legacy
    // workers that go through the per-worker set_block shim use their own
    // per-worker consumed flag and need not call this.
    std::shared_ptr<const TemplateEpoch> latest_unconsumed() const
    {
        auto epoch = load();
        if (!epoch || epoch->is_consumed()) {
            return nullptr;
        }
        return epoch;
    }

    // Latest epoch_id without dereferencing the slot (useful for fast-path
    // staleness checks in worker loops).  Returns 0 before any publish.
    std::uint64_t latest_epoch_id() const
    {
        return m_latest_epoch_id.load(std::memory_order_acquire);
    }

    // Block the caller until either:
    //   * a newer epoch than `last_seen` is published, OR
    //   * `wake_predicate()` returns true (e.g. worker shutdown signalled).
    //
    // Returns the latest epoch_id observed when the wait completes.
    //
    // Lock-order note: `wake_predicate` is invoked while m_wake_mtx is held
    // (std::condition_variable::wait re-checks the predicate under the lock).
    // The intended use case is a trivial atomic load such as
    // `[&]{ return m_shutdown.load(); }` which takes no further lock; if a
    // future caller passes a predicate that acquires another mutex, that
    // mutex must be ordered AFTER m_wake_mtx everywhere else it is held to
    // avoid an inversion deadlock.
    template <typename Predicate>
    std::uint64_t wait_for_epoch_after(std::uint64_t last_seen, Predicate wake_predicate)
    {
        std::unique_lock<std::mutex> lock(m_wake_mtx);
        m_wake_cv.wait(lock, [&]() {
            return m_latest_epoch_id.load(std::memory_order_acquire) > last_seen
                || wake_predicate();
        });
        return m_latest_epoch_id.load(std::memory_order_acquire);
    }

    // Total number of successful publishes since construction (or since the
    // last reset_for_new_batch()).  Currently identical to latest_epoch_id()
    // because epoch_id is advanced exactly once per non-null publish, but
    // exposed as a separate API for stats / diagnostic correlation (e.g.
    // template-churn-vs-stats-output dashboards) so callers don't have to
    // assume the two will stay coupled forever.
    std::uint64_t publish_count() const
    {
        return m_latest_epoch_id.load(std::memory_order_acquire);
    }

    // Drop the currently-published epoch and reset the monotonic counter to 0,
    // making the feed indistinguishable from a freshly-constructed instance
    // for the next worker batch.  Used when the feed is reused across batches
    // (today Worker_manager destroys and recreates the feed in
    // stop_all_workers, which is equivalent; this method makes the intent
    // explicit for future single-feed-across-batches code paths).
    //
    // Must be called only when no worker is consuming from the feed (i.e. all
    // readers have been joined or have observed the wake_predicate).  The
    // wake mutex is taken to serialise with any straggling waiter still
    // inside wait_for_epoch_after, and notify_all() lets that waiter re-
    // evaluate its wake_predicate (typically `m_shutdown`) cleanly.
    void reset_for_new_batch()
    {
        {
            std::lock_guard<std::mutex> lock(m_wake_mtx);
            store_epoch({});
            m_latest_epoch_id.store(0, std::memory_order_release);
        }
        m_wake_cv.notify_all();
    }

    // Wake any waiters without publishing a new epoch.  Used by Worker_manager
    // during shutdown so workers blocked in wait_for_epoch_after() can re-
    // evaluate their wake_predicate (typically an m_shutdown flag).
    //
    // The empty critical section is intentional for the same reason as in
    // publish() — the wake_predicate is flipped by the caller WITHOUT holding
    // m_wake_mtx, so we briefly take the lock here to close the gap between
    // a waiter's predicate check and its park-on-CV (lost-wakeup race).
    void notify_wake()
    {
        {
            std::lock_guard<std::mutex> lock(m_wake_mtx);
        }
        m_wake_cv.notify_all();
    }

private:
    void store_epoch(std::shared_ptr<const TemplateEpoch> epoch)
    {
#if defined(__cpp_lib_atomic_shared_ptr) && __cpp_lib_atomic_shared_ptr >= 201711L
        m_slot.store(std::move(epoch), std::memory_order_release);
#else
        std::scoped_lock lock(m_slot_mtx);
        m_slot.swap(epoch);
#endif
    }

    // libc++ versions without atomic<shared_ptr> use the same mutex fallback
    // as stats::Atomic_snapshot.
#if defined(__cpp_lib_atomic_shared_ptr) && __cpp_lib_atomic_shared_ptr >= 201711L
    std::atomic<std::shared_ptr<const TemplateEpoch>> m_slot;
#else
    std::shared_ptr<const TemplateEpoch> m_slot;
    mutable std::mutex m_slot_mtx;
#endif

    // Mirrors m_slot->epoch_id for fast staleness checks without a shared_ptr
    // load.  Updated after the slot is stored so a reader that sees a newer
    // id is guaranteed to also see the corresponding slot.
    std::atomic<std::uint64_t> m_latest_epoch_id{0};

    // Cold-start / shutdown CV.  Hot-path readers do NOT touch this mutex —
    // they call load() / latest_epoch_id() instead.
    mutable std::mutex m_wake_mtx;
    std::condition_variable m_wake_cv;
};

}  // namespace nexusminer

#endif  // NEXUSMINER_WORKER_TEMPLATE_FEED_HPP
