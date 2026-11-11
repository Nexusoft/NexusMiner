#pragma once

#include <cstdint>

namespace nexusminer {
namespace qtv {

enum class QTVBackendKind : std::uint8_t {
    Auto = 0,
    Cpp = 1,
    Julia = 2,
    Null = 3,
};

constexpr const char* to_string(QTVBackendKind kind) noexcept
{
    switch (kind) {
    case QTVBackendKind::Auto:
        return "auto";
    case QTVBackendKind::Cpp:
        return "cpp";
    case QTVBackendKind::Julia:
        return "julia";
    case QTVBackendKind::Null:
        return "null";
    }

    return "unknown";
}

} // namespace qtv
} // namespace nexusminer
