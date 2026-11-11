#pragma once

#include "qtv/QTVBackendKind.hpp"

namespace nexusminer {
namespace qtv {

struct QTVCapabilities
{
    bool cpp_backend_available{true};
    bool julia_bridge_available{false};
    bool julia_fixture_available{false};
    bool julia_parity_available{false};
    bool production_cpp_authority{true};

    static constexpr QTVCapabilities cpp_only() noexcept
    {
        return {};
    }

    static constexpr QTVCapabilities with_julia(bool fixture_available, bool parity_available) noexcept
    {
        return {true, fixture_available && parity_available, fixture_available, parity_available, true};
    }

    constexpr bool julia_backend_available() const noexcept
    {
        return julia_bridge_available && julia_fixture_available && julia_parity_available;
    }

    constexpr bool supports(QTVBackendKind requested_backend) const noexcept
    {
        switch (requested_backend) {
        case QTVBackendKind::Auto:
            return select_backend(requested_backend) != QTVBackendKind::Null;
        case QTVBackendKind::Cpp:
            return cpp_backend_available;
        case QTVBackendKind::Julia:
            return julia_backend_available();
        case QTVBackendKind::Null:
            return true;
        }

        return false;
    }

    constexpr QTVBackendKind select_backend(QTVBackendKind requested_backend) const noexcept
    {
        if (requested_backend == QTVBackendKind::Null)
            return QTVBackendKind::Null;

        if (requested_backend == QTVBackendKind::Julia && julia_backend_available())
            return QTVBackendKind::Julia;

        if ((requested_backend == QTVBackendKind::Auto || requested_backend == QTVBackendKind::Cpp) &&
            cpp_backend_available)
            return QTVBackendKind::Cpp;

        if (requested_backend == QTVBackendKind::Auto && julia_backend_available())
            return QTVBackendKind::Julia;

        if (requested_backend == QTVBackendKind::Julia && cpp_backend_available)
            return QTVBackendKind::Cpp;

        return QTVBackendKind::Null;
    }
};

} // namespace qtv
} // namespace nexusminer
