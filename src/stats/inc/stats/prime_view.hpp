#ifndef NEXUSMINER_STATS_PRIME_VIEW_HPP
#define NEXUSMINER_STATS_PRIME_VIEW_HPP

#include "stats/types.hpp"
#include <algorithm>

// Issue 5A — single source of truth for "presentation" derivations of a
// prime worker's snapshot.  The console printer, the file printer, and
// any future consumer (JSON exporter, monitoring API, …) all funnel
// through compute_prime_view() so the GISPS / difficulty / CPU-load
// math is defined in exactly one place and the per-publish-vs-per-print
// semantics are explicit.

namespace nexusminer {
namespace stats
{

struct Prime_view
{
    double gisps{0.0};        // billion-integers-per-second sieved over interval_s
    double difficulty{0.0};   // chain difficulty in human units (raw / 1e7)
    double cpu_load{0.0};     // [0.0, 1.0]
};

// Compute the derived "view" values for a prime worker over the interval
// since its previous snapshot.
//
// `current`     — most recent published prime snapshot from the worker.
// `previous`    — the snapshot the consumer saw on its previous emit
//                 (used for cumulative -> rate conversions).
// `interval_s`  — wall-clock seconds since the consumer's previous emit.
//                 Callers should clamp this to a sensible minimum (the
//                 helper additionally guards against <= 0 internally).
// `degraded`    — when true (mining stopped), rates are reported as 0.
//                 Difficulty / cpu_load still reflect the snapshot so the
//                 operator can see why the mode entered degraded.
inline Prime_view compute_prime_view(Prime const& current,
                                     Prime const& previous,
                                     double interval_s,
                                     bool degraded) noexcept
{
    Prime_view view{};
    view.difficulty = current.m_difficulty / 10000000.0;
    view.cpu_load   = std::max(0.0, std::min(1.0, current.m_cpu_load));

    if (degraded || interval_s <= 0.0) {
        view.gisps = 0.0;
        return view;
    }

    // Saturating-subtract: guards against worker resets where the new
    // cumulative range_searched is smaller than the previously-seen one
    // (e.g. fresh start, reset_stats(), or counter wraparound).
    auto const range_delta = current.m_range_searched >= previous.m_range_searched
        ? (current.m_range_searched - previous.m_range_searched)
        : current.m_range_searched;

    view.gisps = static_cast<double>(range_delta) / (1.0e9 * interval_s);
    return view;
}

}
}
#endif
