#ifndef NEXUSMINER_GPU_WORKER_PRIME_HPP
#define NEXUSMINER_GPU_WORKER_PRIME_HPP

#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include "worker.hpp"
#include "hash/nexus_skein.hpp"
#include "hash/nexus_keccak.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <spdlog/spdlog.h>
#include "LLC/types/bignum.h"

namespace asio { class io_context; }

namespace nexusminer {
namespace config { class Worker_config; }
namespace stats { class Collector; }

namespace gpu
{
    using uint1k = boost::multiprecision::uint1024_t;
    class Prime;
    class Sieve;

class Worker_prime : public Worker, public std::enable_shared_from_this<Worker_prime>
{
public:

    Worker_prime(std::shared_ptr<asio::io_context> io_context, config::Worker_config& config);
    ~Worker_prime() noexcept override;

    void set_block(LLP::CBlock block, std::uint32_t nbits, Worker::Block_found_handler result) override;
    void update_statistics(stats::Collector& stats_collector) override;

private:

    void run();
    double getDifficulty(uint1k p);
    double getNetworkDifficulty();
    bool difficulty_check(uint1k p);
   
    std::shared_ptr<asio::io_context> m_io_context;
    std::shared_ptr<spdlog::logger> m_logger;
    config::Worker_config& m_config;
    std::unique_ptr<Prime> m_prime_helper;
    std::atomic<bool> m_stop;
    std::thread m_run_thread;
    Worker::Block_found_handler m_found_nonce_callback;
    std::unique_ptr<Sieve> m_segmented_sieve;

    Block_data m_block;
    std::mutex m_mtx;
    std::uint64_t m_starting_nonce = 0;
    std::string m_log_leader;

    std::uint32_t m_primes{ 0 };
    std::uint32_t m_chains{ 0 };
    std::uint32_t m_difficulty{ 0 };

    std::uint32_t m_pool_nbits;

    std::uint64_t m_nonce = 0;
    uint1k m_base_hash;
    static LLC::CBigNum boost_uint1024_t_to_CBignum(uint1k);

    //stats
    uint64_t m_range_searched = 0;

};
}

}


#endif