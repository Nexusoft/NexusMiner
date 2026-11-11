#ifndef NEXUSMINER_DEBUG_LOCK_AUDIT_H
#define NEXUSMINER_DEBUG_LOCK_AUDIT_H

#include <mutex>

#ifndef NDEBUG
#include <cassert>
#include <cstdio>
#endif

namespace nexusminer {
namespace util {

/**
 * @brief Debug-only lock-hierarchy audit.
 *
 * Context: destroying a Worker (shared_ptr reset) or a PrimeMiningEngine
 * joins a std::thread in its destructor.  If that join ever happened while
 * a mutex such as Worker_manager::m_worker_mutex were still held, and any
 * code reachable from the joined thread ever needed that same mutex
 * (directly or transitively), the calling thread would deadlock
 * permanently.  On the io_context thread this manifests as exactly the
 * "silent freeze" failure mode described in the deadlock audit: CPU mining
 * threads keep running, but every io_context-driven feature (stats
 * printer, GET_ROUND poll, GET_BLOCK/push) goes dead simultaneously because
 * the io_context thread never returns from the handler.
 *
 * stop_all_workers() was refactored to snapshot-then-release: it moves the
 * worker vector / engine / feed out of member state while the lock is
 * held, releases the lock, and only then destroys (joins) the snapshots.
 * That fix is structural, but nothing enforces it stays that way as the
 * code evolves.  Debug_tracked_mutex + assert_no_tracked_locks_held() is a
 * cheap regression guard for exactly that invariant: it fails fast, in
 * debug builds only, the moment a future change reintroduces a
 * join-while-locked pattern on a tracked mutex.
 *
 * Usage:
 *   - Declare the mutex that guards a resource whose teardown can join a
 *     thread as `nexusminer::util::Debug_tracked_mutex` instead of
 *     `std::mutex`.  It satisfies the same BasicLockable interface, so
 *     existing std::lock_guard / std::unique_lock call sites are unchanged.
 *   - Immediately before any call that can block on another thread (a
 *     std::thread::join(), or a shared_ptr::reset() whose destructor
 *     joins), call assert_no_tracked_locks_held() to assert that no
 *     Debug_tracked_mutex is currently held by the calling thread.
 *
 * In release builds (NDEBUG defined) tracking is compiled out entirely:
 * Debug_tracked_mutex degrades to a plain std::mutex with no counter
 * overhead, and assert_no_tracked_locks_held() is a no-op.
 */

#ifndef NDEBUG

namespace detail {
// One counter per thread: how many Debug_tracked_mutex instances the
// calling thread currently holds locked.  Deliberately a simple depth
// counter (not a per-mutex set) — the only question callers need
// answered is "am I safe to block on another thread right now?".
inline int& tracked_lock_depth()
{
    thread_local int depth = 0;
    return depth;
}
} // namespace detail

class Debug_tracked_mutex
{
public:
    void lock()
    {
        m_mutex.lock();
        ++detail::tracked_lock_depth();
    }

    bool try_lock()
    {
        if (m_mutex.try_lock())
        {
            ++detail::tracked_lock_depth();
            return true;
        }
        return false;
    }

    void unlock()
    {
        assert(detail::tracked_lock_depth() > 0 &&
               "Debug_tracked_mutex::unlock() called without a matching lock");
        --detail::tracked_lock_depth();
        m_mutex.unlock();
    }

private:
    std::mutex m_mutex;
};

/// Returns how many Debug_tracked_mutex locks the calling thread currently
/// holds. Exposed mainly for testability of the counting logic itself.
inline int tracked_locks_held_count()
{
    return detail::tracked_lock_depth();
}

/// Asserts (debug builds only) that the calling thread holds no
/// Debug_tracked_mutex before performing a blocking call such as
/// std::thread::join(). `context` is included in the failure message to
/// identify which call site regressed.
inline void assert_no_tracked_locks_held(const char* context)
{
    if (detail::tracked_lock_depth() != 0)
    {
        std::fprintf(stderr,
                     "[Debug_lock_audit] FATAL: %s is about to block on another "
                     "thread while %d tracked lock(s) are still held on this "
                     "thread — this is the join-while-locked deadlock pattern.\n",
                     context ? context : "unknown call site",
                     detail::tracked_lock_depth());
        assert(false && "join-while-locked: see stderr message above");
    }
}

#else // NDEBUG

using Debug_tracked_mutex = std::mutex;

inline int tracked_locks_held_count() { return 0; }

inline void assert_no_tracked_locks_held(const char* /*context*/) {}

#endif // NDEBUG

} // namespace util
} // namespace nexusminer

#endif // NEXUSMINER_DEBUG_LOCK_AUDIT_H
