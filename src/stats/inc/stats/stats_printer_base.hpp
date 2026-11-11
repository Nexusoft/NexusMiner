#ifndef NEXUSMINER_STATS_PRINTER_BASE_HPP
#define NEXUSMINER_STATS_PRINTER_BASE_HPP

#include "stats/stats_printer.hpp"
#include "stats/stats_collector.hpp"
#include "stats/prime_view.hpp"
#include "config/worker_config.hpp"
#include <spdlog/spdlog.h>
#include <cassert>
#include <chrono>
#include <iomanip>
#include <memory>
#include <sstream>
#include <vector>

namespace nexusminer {
namespace stats
{

// Shared implementation for Printer_console and Printer_file. The two printers
// only differ in (a) which spdlog logger they own and (b) whether they flush
// after each emit, so the entire formatting and worker-iteration body lives
// here in one place.
//
// Issue 3A: this used to iterate over std::vector<std::variant<Hash, Prime>>
// and dispatch with std::get<>. It now downcasts the (mode-tagged) base
// Collector to its typed Worker_stats_collector<T> and iterates over a
// concrete std::vector<T>. The downcast is safe because the mode is fixed at
// Collector construction (make_collector(config)).
//
// Issue 5A: the prime-mode formatter funnels GISPS/difficulty/cpu_load math
// through stats::compute_prime_view() so future consumers (JSON, monitoring)
// share the exact same derivation.
template<typename PrinterType>
class Printer_base : public Printer {
public:
    Printer_base(config::Mining_mode mining_mode,
                 std::vector<config::Worker_config> const& worker_config,
                 Collector& stats_collector,
                 std::shared_ptr<spdlog::logger> logger,
                 bool flush_after_emit)
        : m_mining_mode{ mining_mode }
        , m_worker_config{ worker_config }
        , m_stats_collector{ stats_collector }
        , m_logger{ std::move(logger) }
        , m_last_print_time{ std::chrono::steady_clock::now() }
        , m_previous_prime_stats(worker_config.size())
        , m_flush_after_emit{ flush_after_emit }
    {
        m_logger->set_pattern("[%D %H:%M:%S.%e][%^%n%$] %v");
        // Defensive: the printer's runtime mode must match the collector
        // it was constructed against. Worker_manager wires these together
        // in lockstep; the assert catches future misconfiguration early.
        assert(m_mining_mode == m_stats_collector.get_mining_mode());
    }

    void print() override
    {
        auto const now = std::chrono::steady_clock::now();
        double interval_s = std::chrono::duration<double>(now - m_last_print_time).count();
        if (interval_s < 1.0) interval_s = 1.0;

        auto const global_stats = m_stats_collector.get_global_stats();

        if (global_stats.m_degraded_mode) {
            m_logger->warn("╔═══════════════════════════════════════════════════════╗");
            m_logger->warn("║ ⚠️  MINING STOPPED - WAITING FOR VALID TEMPLATE     ║");
            m_logger->warn("╚═══════════════════════════════════════════════════════╝");
        }

        auto const globals_string = PrinterType::print_global(m_stats_collector);

        std::stringstream ss;
        ss << globals_string;

        if (m_mining_mode == config::Mining_mode::HASH) {
            format_hash_workers(ss, global_stats, interval_s);
        } else {
            format_prime_workers(ss, global_stats, interval_s);
        }

        m_logger->info(ss.str());
        if (m_flush_after_emit) {
            m_logger->flush();
        }
        m_last_print_time = now;
    }

private:
    void format_hash_workers(std::stringstream& ss,
                             Global const& global_stats,
                             double interval_s)
    {
        // Issue 3A: typed downcast — no more std::variant / std::get<>.
        auto& typed = as_typed<Hash>(m_stats_collector);
        auto const workers = typed.get_workers_stats();

        for (std::size_t i = 0; i < workers.size(); ++i) {
            ss << "Worker " << m_worker_config[i].m_id << " stats: ";
            format_one_hash_worker(ss, workers[i], i, global_stats, interval_s);
            if (i + 1 < workers.size()) {
                ss << std::endl;
            }
        }
    }

    void format_one_hash_worker(std::stringstream& ss,
                                Hash const& hash_stats,
                                std::size_t worker_config_index,
                                Global const& global_stats,
                                double interval_s)
    {
        double hashrate = 0.0;
        if (!global_stats.m_degraded_mode) {
            hashrate = (hash_stats.m_hash_count / interval_s) / 1.0e6;
        }

        ss << std::setprecision(2) << std::fixed << hashrate << "MH/s";
        if (global_stats.m_degraded_mode) {
            ss << " (idle)";
        }
        ss << ". ";
        ss << (m_worker_config[worker_config_index].m_mode == config::Worker_mode::FPGA
                   ? hash_stats.m_nonce_candidates_recieved
                   : hash_stats.m_met_difficulty_count)
           << " candidates found. Most difficult: " << hash_stats.m_best_leading_zeros;
        if (m_worker_config[worker_config_index].m_mode == config::Worker_mode::FPGA) {
            ss << " Hash Errors: " << hash_stats.m_hash_error_count;
        }
    }

    void format_prime_workers(std::stringstream& ss,
                              Global const& global_stats,
                              double interval_s)
    {
        auto& typed = as_typed<Prime>(m_stats_collector);
        auto const workers = typed.get_workers_stats();

        for (std::size_t i = 0; i < workers.size(); ++i) {
            ss << "Worker " << m_worker_config[i].m_id << " stats: ";
            format_one_prime_worker(ss, workers[i], i, global_stats, interval_s);
            if (i + 1 < workers.size()) {
                ss << std::endl;
            }
        }
    }

    void format_one_prime_worker(std::stringstream& ss,
                                 Prime const& prime_stats,
                                 std::size_t worker_config_index,
                                 Global const& global_stats,
                                 double interval_s)
    {
        // Issue 5A: single source of truth for derived prime view values.
        Prime_view const view = compute_prime_view(
            prime_stats,
            m_previous_prime_stats[worker_config_index],
            interval_s,
            global_stats.m_degraded_mode);

        ss << std::setprecision(2) << std::fixed;
        ss << view.gisps << " GISPS";
        if (global_stats.m_degraded_mode) {
            ss << " (idle)";
        }
        ss << " Chain Count: ";
        for (std::size_t i = 5; i < prime_stats.m_chain_histogram.size(); i++)
        {
            ss << i << ":" << prime_stats.m_chain_histogram[i] << " ";
        }
        ss << " Best " << prime_stats.m_most_difficult_chain;
        ss << " Current Difficulty " << view.difficulty;

        m_previous_prime_stats[worker_config_index] = prime_stats;
    }

    config::Mining_mode m_mining_mode;
    std::vector<config::Worker_config> const& m_worker_config;
    Collector& m_stats_collector;
    std::shared_ptr<spdlog::logger> m_logger;
    std::chrono::steady_clock::time_point m_last_print_time;
    std::vector<Prime> m_previous_prime_stats;
    bool m_flush_after_emit;
};

}
}
#endif
