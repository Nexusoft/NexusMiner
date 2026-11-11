#ifndef NEXUSMINER_EXPONENTIAL_BACKOFF_H
#define NEXUSMINER_EXPONENTIAL_BACKOFF_H

#include <cstdint>
#include <algorithm>

namespace nexusminer {
namespace util {

/**
 * @brief Exponential Backoff Utility
 *
 * Provides a reusable exponential backoff calculator for retry logic.
 * Used across connection retries, session authentication retries, and other
 * retry mechanisms to avoid code duplication.
 *
 * Formula: delay = min(base * 2^(attempt-1), cap)
 * Example with base=1000ms, cap=60000ms:
 *   Attempt 1: 1s
 *   Attempt 2: 2s
 *   Attempt 3: 4s
 *   Attempt 4: 8s
 *   Attempt 5: 16s
 *   Attempt 6: 32s
 *   Attempt 7+: 60s (capped)
 */
struct ExponentialBackoff {
    uint32_t base_ms;  // Base delay in milliseconds
    uint32_t cap_ms;   // Maximum delay cap in milliseconds

    /**
     * Calculate exponential backoff delay for a given attempt count
     * @param attempt_count The attempt number (1-based, i.e., first attempt = 1)
     * @return Delay in milliseconds, capped at cap_ms
     */
    inline uint32_t calculate_delay_ms(uint32_t attempt_count) const {
        if (attempt_count == 0) {
            return base_ms;  // Defensive: treat 0 as first attempt
        }
        // Calculate 2^(attempt_count-1), with overflow protection
        // For attempt_count >= 32, the shift would overflow, but cap_ms handles it
        uint32_t multiplier = (attempt_count >= 32) ? UINT32_MAX : (1u << (attempt_count - 1));

        // Prevent overflow in multiplication
        if (multiplier > cap_ms / base_ms) {
            return cap_ms;
        }

        return std::min(base_ms * multiplier, cap_ms);
    }

    /**
     * Calculate exponential backoff delay in seconds (convenience method)
     * @param attempt_count The attempt number (1-based)
     * @return Delay in seconds, rounded down
     */
    inline uint32_t calculate_delay_seconds(uint32_t attempt_count) const {
        return calculate_delay_ms(attempt_count) / 1000;
    }
};

/**
 * @brief Exponential Backoff with State
 *
 * Maintains internal state for current delay and attempt count,
 * useful for connection retry scenarios where delay doubles on each failure.
 */
struct ExponentialBackoffWithState {
    uint32_t base;          // Base delay (in seconds or milliseconds, depending on use)
    uint32_t cap;           // Maximum delay cap (same units as base)
    uint32_t current_delay; // Current delay value
    uint32_t attempt_count; // Number of retry attempts

    /**
     * Initialize with base and cap values
     * @param base_value Base delay value
     * @param cap_value Maximum delay cap value
     */
    ExponentialBackoffWithState(uint32_t base_value, uint32_t cap_value)
        : base(base_value), cap(cap_value), current_delay(0), attempt_count(0) {}

    /**
     * Calculate next delay: doubles current delay or uses base on first attempt
     * @return Next delay value
     */
    inline uint32_t next_delay() {
        ++attempt_count;
        if (current_delay == 0) {
            current_delay = base;
        } else {
            current_delay = std::min(current_delay * 2, cap);
        }
        return current_delay;
    }

    /**
     * Reset to initial state (used after successful connection)
     */
    inline void reset() {
        current_delay = 0;
        attempt_count = 0;
    }

    /**
     * Override current delay (useful for special cases like degraded mode or node shutdown)
     * @param delay_value New delay value to set
     */
    inline void set_delay(uint32_t delay_value) {
        current_delay = std::min(delay_value, cap);
    }

    /**
     * Get current delay without advancing
     * @return Current delay value
     */
    inline uint32_t get_current_delay() const {
        return current_delay;
    }

    /**
     * Get attempt count
     * @return Number of attempts
     */
    inline uint32_t get_attempt_count() const {
        return attempt_count;
    }
};

} // namespace util
} // namespace nexusminer

#endif // NEXUSMINER_EXPONENTIAL_BACKOFF_H
