#ifndef NEXUSMINER_STATS_TYPES_HPP
#define NEXUSMINER_STATS_TYPES_HPP

#include <array>
#include <variant>
#include <chrono>
#include "stats/prime_stats_snapshot.hpp"

namespace nexusminer {
namespace stats
{

struct Global_delta
{
    std::uint32_t m_accepted_blocks{ 0 };
    std::uint32_t m_rejected_blocks{ 0 };
    std::uint32_t m_accepted_shares{ 0 };
    std::uint32_t m_rejected_shares{ 0 };
    std::uint32_t m_connection_retries{ 0 };
    // A block that a worker solved but Solo::submit_block() refused to
    // transmit at all (e.g. stale session epoch, submit-height mismatch,
    // missing template, encryption failure). Distinct from m_rejected_blocks,
    // which counts blocks the NODE rejected after SUBMIT_BLOCK was sent.
    // See Solo::submit_block() discard paths in src/protocol/src/protocol/solo.cpp.
    std::uint32_t m_discarded_blocks{ 0 };

    Global_delta& operator+=(Global_delta const& other)
    {
        m_accepted_blocks += other.m_accepted_blocks;
        m_rejected_blocks += other.m_rejected_blocks;
        m_accepted_shares += other.m_accepted_shares;
        m_rejected_shares += other.m_rejected_shares;
        m_connection_retries += other.m_connection_retries;
        m_discarded_blocks += other.m_discarded_blocks;

        return *this;
    }
};

struct Global_state
{
    bool m_degraded_mode{ false };  // Mining stopped due to invalid template
};

struct Global
{
    std::uint32_t m_accepted_blocks{ 0 };
    std::uint32_t m_rejected_blocks{ 0 };
    std::uint32_t m_accepted_shares{ 0 };
    std::uint32_t m_rejected_shares{ 0 };
    std::uint32_t m_connection_retries{ 0 };
    // See Global_delta::m_discarded_blocks.
    std::uint32_t m_discarded_blocks{ 0 };
    bool m_degraded_mode{ false };  // Mining stopped due to invalid template
};

struct Hash
{
    std::uint64_t m_hash_count{0};
    int m_best_leading_zeros{0};
    int m_met_difficulty_count{0};
    int m_nonce_candidates_recieved{0};
    int m_hash_error_count{0};

    Hash() = default;

    Hash(Hash const& other)
    {
        m_hash_count = other.m_hash_count;
        m_best_leading_zeros = other.m_best_leading_zeros;
        m_met_difficulty_count = other.m_met_difficulty_count;
        m_nonce_candidates_recieved = other.m_nonce_candidates_recieved;
        m_hash_error_count = other.m_hash_error_count;
    }

    Hash& operator+=(Hash const& other)
    {
        m_hash_count += other.m_hash_count;
        m_best_leading_zeros += std::max(m_best_leading_zeros, other.m_best_leading_zeros);
        m_met_difficulty_count += other.m_met_difficulty_count;
        m_nonce_candidates_recieved += other.m_nonce_candidates_recieved;
        m_hash_error_count += other.m_hash_error_count;
        return *this;
    }
};

struct Prime
{
    std::uint32_t m_primes{ 0 };
    std::uint32_t m_chains{ 0 };
    std::uint32_t m_difficulty{ 0 };
    std::uint64_t m_range_searched { 0 };
    double m_most_difficult_chain{ 0.0 };
    double m_cpu_load{ 0.0 };  // estimated CPU load in [0.0, 1.0]
    Prime_histogram m_chain_histogram{};
    Prime_sieve_diag m_sieve_diag{};

    Prime& operator+=(Prime const& other)
    {
        return *this;
    }
};

}
}
#endif
