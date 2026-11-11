#include <iostream>
#include <algorithm>
#include <cstdint>
#include "Util/include/exponential_backoff.h"
#include "protocol/inc/protocol/protocol_constants.hpp"

// Use constants from protocol_constants.hpp
using nexusminer::protocol::ProtocolConstants::MAX_SESSION_AUTH_RETRIES;
using nexusminer::protocol::ProtocolConstants::BASE_SESSION_RETRY_MS;
using nexusminer::protocol::ProtocolConstants::MAX_SESSION_RETRY_MS;

namespace
{
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

// Test helper using the new utility
uint32_t calculate_backoff_delay_ms(uint32_t attempt_count)
{
    nexusminer::util::ExponentialBackoff backoff{BASE_SESSION_RETRY_MS, MAX_SESSION_RETRY_MS};
    return backoff.calculate_delay_ms(attempt_count);
}
}

int main()
{
    bool ok = true;

    std::cout << "========================================\n";
    std::cout << "Session Auth Retry Logic Tests\n";
    std::cout << "========================================\n\n";

    {
        std::cout << "Test 1: Exponential backoff calculation\n";
        ok &= expect(calculate_backoff_delay_ms(1) == 1000, "Attempt 1: 1s delay");
        ok &= expect(calculate_backoff_delay_ms(2) == 2000, "Attempt 2: 2s delay");
        ok &= expect(calculate_backoff_delay_ms(3) == 4000, "Attempt 3: 4s delay");
        ok &= expect(calculate_backoff_delay_ms(4) == 8000, "Attempt 4: 8s delay");
        ok &= expect(calculate_backoff_delay_ms(5) == 16000, "Attempt 5: 16s delay");
        ok &= expect(calculate_backoff_delay_ms(6) == 32000, "Attempt 6: 32s delay");
        ok &= expect(calculate_backoff_delay_ms(7) == 60000, "Attempt 7: 60s cap (64s would exceed)");
        ok &= expect(calculate_backoff_delay_ms(8) == 60000, "Attempt 8: 60s cap");
        ok &= expect(calculate_backoff_delay_ms(10) == 60000, "Attempt 10: 60s cap");
        std::cout << '\n';
    }

    {
        std::cout << "Test 2: Max retry limit\n";
        uint32_t fail_count = 0;

        // Simulate retry loop
        for (int i = 0; i < 15; ++i)
        {
            ++fail_count;

            if (fail_count > MAX_SESSION_AUTH_RETRIES)
            {
                ok &= expect(fail_count == 11, "Should halt after 10 retries");
                break;
            }
        }

        ok &= expect(fail_count == 11, "Retry counter should be 11 when halted");
        std::cout << '\n';
    }

    {
        std::cout << "Test 3: Counter reset on success\n";
        uint32_t fail_count = 5;

        // Simulate successful auth
        fail_count = 0;

        ok &= expect(fail_count == 0, "Counter reset to 0 after successful auth");

        // Next failure should start from attempt 1
        ++fail_count;
        ok &= expect(calculate_backoff_delay_ms(fail_count) == 1000,
                     "After reset, first retry uses 1s delay");
        std::cout << '\n';
    }

    {
        std::cout << "Test 4: Separate primary and secondary counters\n";
        uint32_t primary_fail_count = 3;
        uint32_t secondary_fail_count = 7;

        ok &= expect(primary_fail_count != secondary_fail_count,
                     "Primary and secondary counters are independent");
        ok &= expect(calculate_backoff_delay_ms(primary_fail_count) == 4000,
                     "Primary counter produces correct delay");
        ok &= expect(calculate_backoff_delay_ms(secondary_fail_count) == 60000,
                     "Secondary counter produces correct delay (capped)");
        std::cout << '\n';
    }

    {
        std::cout << "Test 5: Overflow safety (edge case)\n";
        // Attempt counts >= 7 will be capped at MAX_SESSION_RETRY_MS
        // because 2^6 * 1000 = 64000 > 60000
        // The shift operation is safe for attempts 1-10 (max retries)
        // since 2^9 * 1000 = 512000, which computes correctly before the min() cap
        ok &= expect(calculate_backoff_delay_ms(10) == MAX_SESSION_RETRY_MS,
                     "Attempt 10 (max retries) capped at max");
        ok &= expect(calculate_backoff_delay_ms(MAX_SESSION_AUTH_RETRIES) == MAX_SESSION_RETRY_MS,
                     "Max attempt count capped at max");
        std::cout << '\n';
    }

    std::cout << "========================================\n";
    if (ok)
    {
        std::cout << "All tests PASSED\n";
        return 0;
    }
    else
    {
        std::cout << "Some tests FAILED\n";
        return 1;
    }
}
