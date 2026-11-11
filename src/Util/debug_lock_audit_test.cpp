// Debug_tracked_mutex / assert_no_tracked_locks_held unit tests.
//
// These pin the counting behaviour that Worker_manager::stop_all_workers()
// (and its sibling teardown path in Worker_manager::stop()) rely on to catch
// a regression back to the "join-while-locked" deadlock pattern: joining a
// std::thread (via shared_ptr::reset() on a Worker or PrimeMiningEngine)
// while still holding m_worker_mutex.  See Util/include/debug_lock_audit.h
// for the full rationale.
//
// Debug_tracked_mutex only tracks in non-NDEBUG builds; in release builds it
// degrades to std::mutex with the counter always reporting 0. These tests
// therefore only assert the depth-counter behaviour when tracking is active,
// and otherwise confirm the release no-op degrades safely.

#include <cassert>
#include <iostream>
#include <mutex>

#include "Util/include/debug_lock_audit.h"

using nexusminer::util::Debug_tracked_mutex;
using nexusminer::util::assert_no_tracked_locks_held;
using nexusminer::util::tracked_locks_held_count;

namespace {

bool expect(bool condition, const char* message)
{
    if (!condition)
    {
        std::cerr << "  [FAIL] " << message << std::endl;
        return false;
    }
    std::cout << "  [PASS] " << message << std::endl;
    return true;
}

} // namespace

int main()
{
    bool ok = true;

    std::cout << "========================================\n";
    std::cout << "Debug_lock_audit Tests\n";
    std::cout << "========================================\n\n";

    {
        std::cout << "Test 1: No locks held initially\n";
        ok &= expect(tracked_locks_held_count() == 0, "Depth is 0 with no locks taken");
        std::cout << '\n';
    }

    {
        std::cout << "Test 2: lock_guard increments/decrements depth (CTAD, like production call sites)\n";
        Debug_tracked_mutex m;
        {
            std::lock_guard lock(m);
#ifndef NDEBUG
            ok &= expect(tracked_locks_held_count() == 1, "Depth is 1 while lock_guard holds the mutex");
#else
            ok &= expect(tracked_locks_held_count() == 0, "Release build: depth always reports 0");
#endif
        }
        ok &= expect(tracked_locks_held_count() == 0, "Depth returns to 0 after lock_guard destructs");
        std::cout << '\n';
    }

    {
        std::cout << "Test 3: Nested Debug_tracked_mutex instances stack correctly\n";
        Debug_tracked_mutex a;
        Debug_tracked_mutex b;
        {
            std::lock_guard lock_a(a);
            {
                std::lock_guard lock_b(b);
#ifndef NDEBUG
                ok &= expect(tracked_locks_held_count() == 2, "Depth is 2 with two nested locks held");
#endif
            }
#ifndef NDEBUG
            ok &= expect(tracked_locks_held_count() == 1, "Depth is 1 after inner lock releases");
#endif
        }
        ok &= expect(tracked_locks_held_count() == 0, "Depth is 0 after both locks release");
        std::cout << '\n';
    }

    {
        std::cout << "Test 4: try_lock() participates in tracking like lock()\n";
        Debug_tracked_mutex m;
        bool acquired = m.try_lock();
        ok &= expect(acquired, "try_lock() succeeds on an unheld mutex");
#ifndef NDEBUG
        ok &= expect(tracked_locks_held_count() == 1, "Depth is 1 after a successful try_lock()");
#endif
        m.unlock();
        ok &= expect(tracked_locks_held_count() == 0, "Depth is 0 after unlock()");
        std::cout << '\n';
    }

    {
        std::cout << "Test 5: assert_no_tracked_locks_held() is a silent no-op when depth is 0\n";
        // This is exactly the call stop_all_workers() makes right after
        // releasing m_worker_mutex and before joining worker/engine threads;
        // it must not abort when no tracked lock is held.
        assert_no_tracked_locks_held("debug_lock_audit_test (depth 0)");
        ok &= expect(true, "assert_no_tracked_locks_held() returned without aborting");
        std::cout << '\n';
    }

    std::cout << "========================================\n";
    if (ok)
    {
        std::cout << "ALL TESTS PASSED\n";
    }
    else
    {
        std::cout << "SOME TESTS FAILED\n";
    }
    std::cout << "========================================\n";

    return ok ? 0 : 1;
}
