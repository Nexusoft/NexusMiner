#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <optional>
#include <vector>

namespace nexusminer::protocol {

enum class GetBlockControlReason : uint8_t {
    RATE_LIMIT_EXCEEDED = 1,
    SESSION_INVALID = 2,
    UNAUTHENTICATED = 3,
    TEMPLATE_NOT_READY = 4,
    INTERNAL_RETRY = 5,
    TEMPLATE_STALE = 6,
    TEMPLATE_REBUILD_IN_PROGRESS = 7,
    TEMPLATE_SOURCE_UNAVAILABLE = 8,
    CHANNEL_NOT_SET = 9
};

struct GetBlockControl {
    GetBlockControlReason reason;
    uint32_t retry_after_ms;

    static std::optional<GetBlockControl> parse(const std::vector<uint8_t>& data)
    {
        if (data.size() != 8 || data[0] != 1 || data[1] < 1 || data[1] > 9 ||
            data[2] != 0 || data[3] != 0) {
            return std::nullopt;
        }
        return GetBlockControl{
            static_cast<GetBlockControlReason>(data[1]),
            (uint32_t(data[4]) << 24) | (uint32_t(data[5]) << 16) |
            (uint32_t(data[6]) << 8) | uint32_t(data[7])
        };
    }

    bool retryable() const
    {
        return reason == GetBlockControlReason::RATE_LIMIT_EXCEEDED ||
               reason == GetBlockControlReason::TEMPLATE_NOT_READY ||
               reason == GetBlockControlReason::INTERNAL_RETRY ||
               reason == GetBlockControlReason::TEMPLATE_STALE ||
               reason == GetBlockControlReason::TEMPLATE_REBUILD_IN_PROGRESS ||
               reason == GetBlockControlReason::TEMPLATE_SOURCE_UNAVAILABLE;
    }

    const char* reason_name() const
    {
        switch (reason) {
        case GetBlockControlReason::RATE_LIMIT_EXCEEDED: return "RATE_LIMIT_EXCEEDED";
        case GetBlockControlReason::SESSION_INVALID: return "SESSION_INVALID";
        case GetBlockControlReason::UNAUTHENTICATED: return "UNAUTHENTICATED";
        case GetBlockControlReason::TEMPLATE_NOT_READY: return "TEMPLATE_NOT_READY";
        case GetBlockControlReason::INTERNAL_RETRY: return "INTERNAL_RETRY";
        case GetBlockControlReason::TEMPLATE_STALE: return "TEMPLATE_STALE";
        case GetBlockControlReason::TEMPLATE_REBUILD_IN_PROGRESS: return "TEMPLATE_REBUILD_IN_PROGRESS";
        case GetBlockControlReason::TEMPLATE_SOURCE_UNAVAILABLE: return "TEMPLATE_SOURCE_UNAVAILABLE";
        case GetBlockControlReason::CHANNEL_NOT_SET: return "CHANNEL_NOT_SET";
        }
        return "UNKNOWN";
    }

    std::chrono::milliseconds retry_delay() const
    {
        // Stay below the node's rolling 25-requests-per-minute limit.
        return std::chrono::milliseconds(std::max<uint32_t>(retry_after_ms, 5000));
    }
};

}
