// Smoke test for LivenessWatchdog: verifies (1) it does NOT force-exit while
// the io_context is healthy and pumping, and (2) it DOES force-exit when the
// io_context thread is deliberately wedged past the stall timeout.
//
// The force-exit path is verified out-of-process (a child process is spawned
// that wedges its own io_context) since LivenessWatchdog's whole point is to
// terminate the process — that behavior cannot be observed in-process.
#include "liveness_watchdog.hpp"
#include "chrono/timer_factory.hpp"

#include <asio.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <thread>

#ifndef _WIN32
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

int g_failures = 0;

void expect(bool condition, const char* description)
{
    if (condition) {
        std::cout << "[PASS] " << description << "\n";
    } else {
        std::cout << "[FAIL] " << description << "\n";
        ++g_failures;
    }
}

// Runs the io_context normally for `run_duration` while the watchdog is
// armed with a stall_timeout comfortably longer than that. The watchdog
// must not force-exit — if it did, this whole test process would vanish
// with exit code 42 instead of completing normally.
void test_healthy_io_context_does_not_trip_watchdog()
{
    auto io = std::make_shared<asio::io_context>();
    auto logger = spdlog::stdout_color_mt("healthy_test_logger");
    auto timer_factory = std::make_shared<nexusminer::chrono::Timer_factory>(io);

    nexusminer::LivenessWatchdog watchdog(
        timer_factory, logger, std::chrono::seconds{1}, std::chrono::seconds{5});
    watchdog.start();

    auto work_guard = asio::make_work_guard(*io);
    std::thread runner([&io] { io->run_for(std::chrono::seconds{3}); });
    runner.join();

    watchdog.stop();
    spdlog::drop("healthy_test_logger");

    expect(true, "healthy io_context ran for 3s without the watchdog force-exiting the process");
}

// stop()/start() must be idempotent and safe to call multiple times without
// crashing, double-joining, or leaking the watchdog thread.
void test_start_stop_idempotent()
{
    auto io = std::make_shared<asio::io_context>();
    auto logger = spdlog::stdout_color_mt("idempotent_test_logger");
    auto timer_factory = std::make_shared<nexusminer::chrono::Timer_factory>(io);

    nexusminer::LivenessWatchdog watchdog(
        timer_factory, logger, std::chrono::seconds{1}, std::chrono::seconds{30});

    watchdog.start();
    watchdog.start();  // no-op, must not double-start the thread
    watchdog.stop();
    watchdog.stop();   // no-op, must not double-join
    watchdog.start();  // restart after stop must work cleanly
    watchdog.stop();

    spdlog::drop("idempotent_test_logger");
    expect(true, "start()/stop() sequence completed without crashing or hanging");
}

#ifndef _WIN32
// Spawns a child process whose io_context thread wedges forever (simulating
// a deadlock in the synchronous handler chain). The watchdog in the child
// must detect the stall and force-exit with code 42 within a bounded time.
void test_wedged_io_context_triggers_forced_exit()
{
    pid_t pid = fork();
    if (pid < 0) {
        expect(false, "fork() for wedge-detection child process");
        return;
    }

    if (pid == 0) {
        // ── Child process ──
        auto io = std::make_shared<asio::io_context>();
        auto logger = spdlog::stdout_color_mt("wedge_test_logger");
        auto timer_factory = std::make_shared<nexusminer::chrono::Timer_factory>(io);

        nexusminer::LivenessWatchdog watchdog(
            timer_factory, logger, std::chrono::seconds{1}, std::chrono::seconds{2});
        watchdog.start();

        // Permanently wedge the io_context thread with a blocking mutex wait
        // that is never released — simulating a deadlocked handler chain.
        // No further io_context work (including the heartbeat timer) can run
        // once this executes on the single io_context thread.
        std::mutex deadlock_mtx;
        deadlock_mtx.lock();
        asio::post(*io, [&deadlock_mtx] {
            deadlock_mtx.lock();  // never returns — the thread is now wedged
        });

        // Give the io_context something to run so the posted handler actually
        // executes and wedges.
        io->run();

        // This point should be unreachable in a correctly-behaving watchdog:
        // io_context->run() only returns once its work queue is drained, but
        // the posted handler above never returns (permanent deadlock), so
        // run() blocks forever *unless* the watchdog's independent thread
        // detects the stall first and calls std::_Exit(42) itself. Reaching
        // this line at all means the watchdog failed to fire, so exit with a
        // distinct failure code (1) that the parent's assertion below will
        // reject (it only accepts exit code 42).
        std::_Exit(1);
    }

    // ── Parent process ── wait for the child, bounded so the test itself
    // cannot hang if something regresses.
    int status = 0;
    bool reaped = false;
    for (int i = 0; i < 100; ++i) {  // up to ~10s
        pid_t result = waitpid(pid, &status, WNOHANG);
        if (result == pid) {
            reaped = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds{100});
    }

    if (!reaped) {
        // Child never exited within the bounded wait — terminate and reap it
        // so the test suite cannot leak a wedged process.
        kill(pid, SIGKILL);
        waitpid(pid, &status, 0);
    }

    bool exited = WIFEXITED(status);
    int exit_code = exited ? WEXITSTATUS(status) : -1;
    expect(reaped && exited && exit_code == 42,
           "wedged io_context causes LivenessWatchdog to force-exit the process with code 42");
}
#endif

} // namespace

int main()
{
    std::cout << "=== LivenessWatchdog tests ===\n";
    test_healthy_io_context_does_not_trip_watchdog();
    test_start_stop_idempotent();
#ifndef _WIN32
    test_wedged_io_context_triggers_forced_exit();
#endif

    if (g_failures == 0) {
        std::cout << "\nAll LivenessWatchdog tests passed.\n";
        return 0;
    }
    std::cout << "\n" << g_failures << " LivenessWatchdog test(s) FAILED.\n";
    return 1;
}
