#ifndef NEXUSMINER_STATS_TYPES_HPP
#define NEXUSMINER_STATS_TYPES_HPP

#include <memory>
#include <vector>
#include <variant>
#include <chrono>
#include <mutex>

namespace nexusminer {
namespace stats
{

struct Global
{
    std::uint32_t m_accepted_blocks{ 0 };
    std::uint32_t m_rejected_blocks{ 0 };
    std::uint32_t m_accepted_shares{ 0 };
    std::uint32_t m_rejected_shares{ 0 };
    std::uint32_t m_connection_retries{ 0 };

    Global& operator+=(Global const& other)
    {
        m_accepted_blocks += other.m_accepted_blocks;
        m_rejected_blocks += other.m_rejected_blocks;
        m_accepted_shares += other.m_accepted_shares;
        m_rejected_shares += other.m_rejected_shares;
        m_connection_retries += other.m_connection_retries;

        return *this;
    }
};

struct Hash
{
    std::uint64_t m_hash_count{0};
    int m_best_leading_zeros{0};
    int m_met_difficulty_count{0};
    int m_nonce_candidates_recieved{0};

    Hash() = default;

    Hash(Hash const& other)
    {
        m_hash_count = other.m_hash_count;
        m_best_leading_zeros = other.m_best_leading_zeros;
        m_met_difficulty_count = other.m_met_difficulty_count;
        m_nonce_candidates_recieved = other.m_nonce_candidates_recieved;
    }

    Hash& operator+=(Hash const& other)
    {
        m_hash_count += other.m_hash_count;
        m_best_leading_zeros += std::max(m_best_leading_zeros, other.m_best_leading_zeros);
        m_met_difficulty_count += other.m_met_difficulty_count;
        m_nonce_candidates_recieved += other.m_nonce_candidates_recieved;

        return *this;
    }
};

struct Prime
{
    Prime& operator+=(Prime const& other)
    {
        return *this;
    }
};

}
}
#endif
