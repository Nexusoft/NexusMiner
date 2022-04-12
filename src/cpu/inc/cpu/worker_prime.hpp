#ifndef NEXUSMINER_CPU_WORKER_PRIME_HPP
#define NEXUSMINER_CPU_WORKER_PRIME_HPP
//a software prime channel miner for demo and test purposes

#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include "worker.hpp"
#include "hash/nexus_skein.hpp"
#include "hash/nexus_keccak.hpp"
#include <spdlog/spdlog.h>
#include "LLC/types/bignum.h"
#include "ump.hpp"

namespace asio { class io_context; }

namespace nexusminer {
namespace config { class Worker_config; }
namespace stats { class Collector; }

namespace cpu
{
    using uint1k = ump::uint1024_t;
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
    //std::uint64_t leading_zero_mask();
    bool isPrime(uint1k p);
    void fermat_performance_test();

    //Poor man's difficulty.  Report any nonces with at least this many leading zeros. Let the software perform additional filtering. 
    //static constexpr int leading_zeros_required = 20;    //set lower to find more nonce candidates

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

    void reset_statistics();
    std::uint32_t m_primes{ 0 };
    std::uint32_t m_chains{ 0 };
    std::uint32_t m_difficulty{ 0 };

    std::uint32_t m_pool_nbits;

    std::uint64_t m_nonce = 0;
    uint1k m_base_hash;


    void generate_seive(uint1k);
    void analyze_chains();
    void mine_region(uint1k);
    static LLC::CBigNum ump_uint1024_t_to_CBignum(uint1k);


    std::vector<bool>m_sieve;
    static constexpr int m_primorialEndPrime = 3000000;
    static constexpr int m_minChainLength = 8;  //min chain length
    static constexpr int m_maxGap = 12 / 2;  //the largest allowable prime gap.Divide by two because the sieve excludes all even numbers.

    //we have finite memory so we have to limit the sieve to some reasonable size
    static constexpr int m_maxSieveLength = 30000000;
    static constexpr int m_sieveLength = m_maxSieveLength;
    static constexpr int m_sieveRange = m_sieveLength * 2;

    //Vectors containing information about chains we find
    //int chainListLength = std::min(sieveLength, 10000000);
    std::vector<uint64_t> m_chainStartPosArray;
    std::vector<int> m_chainLengthArray;
    std::vector<std::vector<int>> m_chainOffsets;
    int m_chainCount = 0;  //number of chains found
    int m_candidateCount = 0;  //number of possible primes in the sieve

    //stats
    std::vector<std::uint32_t> m_chain_histogram;
    uint64_t m_range_searched = 0;

};
}

}


#endif