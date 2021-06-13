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

struct Hash
{
    std::uint64_t m_hash_count{0};
    int m_best_leading_zeros{0};
    int m_met_difficulty_count{0};
    int m_nonce_candidates_recieved{0};

    Hash& operator+=(Hash const& other)
    {
        m_hash_count += other.m_hash_count;
        m_best_leading_zeros += other.m_best_leading_zeros;
        m_met_difficulty_count += other.m_met_difficulty_count;

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
