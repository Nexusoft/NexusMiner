#include "stats/stats_collector.hpp"
#include "config/config.hpp"
#include "config/types.hpp"

namespace nexusminer
{
namespace stats
{

Collector::Collector(Config& config)
: m_config{config}
, m_start_time{std::chrono::steady_clock::now()}
, m_accepted_blocks{0}
, m_rejected_blocks{0}
, m_connection_retries{0}
{
    for(auto& worker_config : m_config.get_worker_config())
    {
        if(m_config.get_mining_mode() == config::Mining_mode::HASH)
        {
            m_workers.push_back(Hash{});
        }
        else
        {
            m_workers.push_back(Prime{});
        }
    }
}

void Collector::update_worker_stats(std::uint16_t internal_worker_id, Hash const& stats)
{
    assert(m_config.get_mining_mode() == config::Mining_mode::HASH);

    std::scoped_lock lock(m_worker_mutex);

    auto& hash_stats = std::get<Hash>(m_workers[internal_worker_id]);
    hash_stats += stats;
}

void Collector::update_worker_stats(std::uint16_t internal_worker_id, Prime const& stats)
{
    assert(m_config.get_mining_mode() == config::Mining_mode::PRIME);

    std::scoped_lock lock(m_worker_mutex);
    
    auto& prime_stats = std::get<Prime>(m_workers[internal_worker_id]);
    prime_stats += stats;
}

}
}