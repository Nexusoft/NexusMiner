#include "worker/node_shutdown_policy.hpp"

#include <iostream>

namespace {

int tests_run = 0;
int tests_failed = 0;

void check(bool condition, const char* label)
{
    ++tests_run;
    if (condition) {
        std::cout << "  [PASS] " << label << '\n';
    } else {
        ++tests_failed;
        std::cout << "  [FAIL] " << label << '\n';
    }
}

void test_no_failover_retries_primary()
{
    check(nexusminer::decide_node_shutdown_action(false) ==
              nexusminer::NodeShutdownAction::RETRY_PRIMARY_NODE,
          "NODE_SHUTDOWN with no failover retries the primary");
}

void test_failover_switches_to_standby()
{
    check(nexusminer::decide_node_shutdown_action(true) ==
              nexusminer::NodeShutdownAction::SWITCH_TO_STANDBY_NODE,
          "NODE_SHUTDOWN with failover selects standby-node switch");
}

void test_shutdown_invalidates_current_work()
{
    constexpr auto invalidation = nexusminer::node_shutdown_work_invalidation();
    check(invalidation.stop_workers,
          "NODE_SHUTDOWN invalidation stops workers");
    check(invalidation.discard_template,
          "NODE_SHUTDOWN invalidation discards the current template");
    check(invalidation.reset_session,
          "NODE_SHUTDOWN invalidation resets the current node session");
    check(invalidation.quarantine_current_generation,
          "NODE_SHUTDOWN invalidation quarantines current-generation templates");
}

} // namespace

int main()
{
    std::cout << "node_shutdown_policy_test\n";

    test_no_failover_retries_primary();
    test_failover_switches_to_standby();
    test_shutdown_invalidates_current_work();

    if (tests_failed != 0) {
        std::cout << "\nnode_shutdown_policy_test: " << tests_failed
                  << " of " << tests_run << " test(s) failed.\n";
        return 1;
    }

    std::cout << "\nnode_shutdown_policy_test: all " << tests_run
              << " test(s) passed.\n";
    return 0;
}
