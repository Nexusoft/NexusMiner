#include "protocol/epoch_coordinator.hpp"
#include "protocol/session_manager.hpp"
#include <cassert>
#include <iostream>
#include <thread>
#include <atomic>
#include <vector>

using namespace nexusminer::protocol;

static int g_pass = 0;
static int g_fail = 0;

static void print_result(const char* name, bool ok) {
    if (ok) {
        ++g_pass;
        std::cout << "  [PASS] " << name << "\n";
    } else {
        ++g_fail;
        std::cout << "  [FAIL] " << name << "\n";
    }
}

void test_initial_state() {
    std::cout << "\nTest: initial state\n";
    EpochCoordinator ec;
    print_result("session_epoch starts at 0", ec.session_epoch().get() == 0);
    print_result("recovery_epoch starts at 0", ec.recovery_epoch() == 0);
    print_result("global_epoch starts at 0", ec.global_epoch() == 0);
    auto snap = ec.snapshot();
    print_result("snapshot session=0", snap.session_epoch.get() == 0);
    print_result("snapshot recovery=0", snap.recovery_epoch == 0);
    print_result("snapshot global=0", snap.global_epoch == 0);
}

void test_session_epoch_monotonic() {
    std::cout << "\nTest: session_epoch monotonic\n";
    EpochCoordinator ec;
    auto v1 = ec.advance_session_epoch("test_a");
    print_result("first advance returns 1", v1.get() == 1);
    print_result("session_epoch() == 1", ec.session_epoch().get() == 1);
    auto v2 = ec.advance_session_epoch("test_b");
    print_result("second advance returns 2", v2.get() == 2);
    print_result("never decreases", ec.session_epoch().get() >= v1.get());
}

void test_recovery_epoch_monotonic() {
    std::cout << "\nTest: recovery_epoch monotonic\n";
    EpochCoordinator ec;
    auto v1 = ec.advance_recovery_epoch("recovery_a");
    print_result("first recovery advance returns 1", v1 == 1);
    print_result("recovery_epoch() == 1", ec.recovery_epoch() == 1);
    ec.advance_recovery_epoch("recovery_b");
    print_result("second recovery advance returns 2", ec.recovery_epoch() == 2);
}

void test_global_epoch() {
    std::cout << "\nTest: global_epoch = max(session, recovery)\n";
    EpochCoordinator ec;
    ec.advance_session_epoch("s1");
    ec.advance_session_epoch("s2");
    print_result("global after 2 session advances == 2", ec.global_epoch() == 2u);
    ec.advance_recovery_epoch("r1");
    print_result("global after 1 recovery advance == 2 (max)", ec.global_epoch() == 2u);
    ec.advance_recovery_epoch("r2");
    ec.advance_recovery_epoch("r3");
    print_result("global after 3 recovery advances == 3", ec.global_epoch() == 3);
}

void test_observer_notification() {
    std::cout << "\nTest: observer notification\n";
    EpochCoordinator ec;
    std::string last_domain;
    uint64_t last_old{999};
    uint64_t last_new{999};
    ec.add_observer([&](const char* domain, uint64_t old_val, uint64_t new_val) {
        last_domain = domain;
        last_old = old_val;
        last_new = new_val;
    });
    ec.advance_session_epoch("obs_test");
    print_result("observer called for session domain", last_domain == "session");
    print_result("observer old_val == 0", last_old == 0);
    print_result("observer new_val == 1", last_new == 1);
    ec.advance_recovery_epoch("obs_recovery");
    print_result("observer called for recovery domain", last_domain == "recovery");
    print_result("recovery observer new_val == 1", last_new == 1);
}

void test_thread_safety() {
    std::cout << "\nTest: thread safety\n";
    EpochCoordinator ec;
    constexpr int N_THREADS = 8;
    constexpr int ADVANCES_PER_THREAD = 100;
    std::vector<std::thread> threads;
    for (int i = 0; i < N_THREADS; ++i) {
        threads.emplace_back([&ec]() {
            for (int j = 0; j < ADVANCES_PER_THREAD; ++j) {
                ec.advance_session_epoch("thread");
            }
        });
    }
    for (auto& t : threads) t.join();
    print_result("session_epoch == N_THREADS * ADVANCES_PER_THREAD after concurrent advances",
                 ec.session_epoch().get() == static_cast<uint64_t>(N_THREADS * ADVANCES_PER_THREAD));
}

void test_no_reset_after_session_clear() {
    std::cout << "\nTest: session epoch preserved after simulated clear\n";
    EpochCoordinator ec;
    ec.advance_session_epoch("auth1");
    ec.advance_session_epoch("auth2");
    uint64_t epoch_before = ec.session_epoch().get();
    // Simulate SessionManager calling session_epoch() after clear_runtime_session_locked()
    uint64_t restored = ec.session_epoch().get();
    print_result("epoch preserved (not reset to 0)", restored == epoch_before);
    print_result("epoch is 2 after 2 advances", epoch_before == 2);
    // Advance again — must go to 3, not back to 1
    ec.advance_session_epoch("auth3");
    print_result("epoch is 3 after third advance", ec.session_epoch().get() == 3);
}

// Regression test: wiring the coordinator AFTER local epoch increments must
// never decrease m_session.session_epoch (monotonic invariant).
void test_set_epoch_coordinator_no_regression() {
    std::cout << "\nTest: set_epoch_coordinator() must not regress session_epoch\n";

    // Step 1-2: Create SessionManager without a coordinator; authenticate once
    // via the fallback path so local epoch becomes 1.
    auto sm = std::make_shared<SessionManager>(24, nullptr);
    sm->start_session(SessionId(1001u));
    uint64_t epoch_after_auth = sm->get_session_epoch().get();
    print_result("epoch is 1 after first auth (no coordinator)", epoch_after_auth == 1);

    // Step 3: Create a fresh EpochCoordinator — it starts at epoch 0.
    auto coordinator = std::make_shared<EpochCoordinator>();
    print_result("coordinator starts at epoch 0", coordinator->session_epoch().get() == 0);

    // Step 4: Wire the coordinator AFTER the local increment.
    sm->set_epoch_coordinator(coordinator);

    // Step 5: Epoch must not have gone backwards.
    uint64_t epoch_after_wire = sm->get_session_epoch().get();
    print_result("epoch did not regress after late coordinator wire",
                 epoch_after_wire >= epoch_after_auth);
    print_result("epoch is still >= 1 after set_epoch_coordinator",
                 epoch_after_wire >= 1);
}

// Regression test: clear_for_disconnect() (which calls clear_runtime_session_locked())
// must not decrease the epoch below the pre-clear value.
void test_clear_for_disconnect_epoch_preserved() {
    std::cout << "\nTest: clear_for_disconnect() must preserve session_epoch\n";

    // Wire the coordinator first so the normal authenticated path uses it.
    auto coordinator = std::make_shared<EpochCoordinator>();
    auto sm = std::make_shared<SessionManager>(24, nullptr);
    sm->set_epoch_coordinator(coordinator);

    // Authenticate — advances coordinator epoch to 1.
    sm->start_session(SessionId(2002u));
    uint64_t epoch_before_clear = sm->get_session_epoch().get();
    print_result("epoch is 1 after auth with coordinator", epoch_before_clear == 1);

    // Disconnect — calls clear_runtime_session_locked().
    sm->clear_for_disconnect();

    uint64_t epoch_after_clear = sm->get_session_epoch().get();
    print_result("epoch did not regress after clear_for_disconnect",
                 epoch_after_clear >= epoch_before_clear);
}

int main() {
    std::cout << "=== EpochCoordinator Unit Tests ===\n";
    test_initial_state();
    test_session_epoch_monotonic();
    test_recovery_epoch_monotonic();
    test_global_epoch();
    test_observer_notification();
    test_thread_safety();
    test_no_reset_after_session_clear();
    test_set_epoch_coordinator_no_regression();
    test_clear_for_disconnect_epoch_preserved();
    std::cout << "\n=== Results: " << g_pass << " passed, " << g_fail << " failed ===\n";
    return g_fail == 0 ? 0 : 1;
}
