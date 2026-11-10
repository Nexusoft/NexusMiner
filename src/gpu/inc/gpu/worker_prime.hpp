#ifndef NEXUSMINER_GPU_WORKER_PRIME_HPP
#define NEXUSMINER_GPU_WORKER_PRIME_HPP

#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include "worker.hpp"
#include "hash/nexus_skein.hpp"
#include "hash/nexus_keccak.hpp"
#include <boost/multiprecision/cpp_int.hpp>
#include <spdlog/spdlog.h>
#include "LLC/types/bignum.h"
#include "stats/types.hpp"
#include "stats/prime_stats_snapshot.hpp"

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

    void set_block(::LLP::CBlock block, std::uint32_t nbits, Worker::Block_found_handler result) override;

    // Optimized version: accepts shared WorkPackage to eliminate repeated block data construction
    void set_block(std::shared_ptr<WorkPackage> work_package, Worker::Block_found_handler result) override;

    bool is_running() const override { return m_running.load(); }
    void update_statistics(stats::Collector& stats_collector) override;

private:

    void run();
    double getDifficulty(const uint1k& p);
    double getNetworkDifficulty();
    bool difficulty_check(const uint1k& p);
    void publish_statistics_snapshot();

    // Stone — single source of truth for the "use pool nbits if set, else
    // block nbits" selection (mirrors the same idiom on CPU worker_hash and
    // CPU worker_prime).  Callers MUST hold m_mtx when invoking — both
    // m_pool_nbits and m_block.nBits are written under m_mtx by set_block().
    std::uint32_t effective_nbits_locked() const noexcept
    {
        return m_pool_nbits != 0 ? m_pool_nbits : m_block.nBits;
    }
   
    std::shared_ptr<asio::io_context> m_io_context;
    std::shared_ptr<spdlog::logger> m_logger;
    config::Worker_config& m_config;
    std::unique_ptr<Prime> m_prime_helper;
    std::atomic<bool> m_stop;
    std::atomic<bool> m_running{false};  // true once set_block() has been called with valid work
    std::thread m_run_thread;
    Worker::Block_found_handler m_found_nonce_callback;
    std::unique_ptr<Sieve> m_segmented_sieve;
    bool m_gpu_initialized = false;
    Block_data m_block;
    std::mutex m_mtx;
    std::condition_variable m_cv;  // For persistent thread wake-up
    bool m_new_work = false;       // Flag to indicate new work is available
    bool m_shutdown = false;       // Flag to indicate worker should shut down
    // Option D, matching cpu::Worker_prime.  Set after a valid submit so the
    // mining loop stops grinding the same template until set_block delivers
    // a new one.  Read and written under m_mtx.
    bool m_template_consumed = false;
    std::uint64_t m_starting_nonce = 0;
    std::string m_log_leader;

    std::uint32_t m_primes{ 0 };
    std::uint32_t m_chains{ 0 };
    // Stone — m_difficulty is read on the worker thread (getNetworkDifficulty,
    // publish_statistics_snapshot) without holding m_mtx, but written under
    // m_mtx by set_block().  Make it std::atomic to close the documented
    // data race.  Tearing was benign on x86 for uint32_t but we want this
    // race-free under TSAN and on weakly-ordered architectures.  All reads
    // use memory_order_relaxed since this value is purely informational and
    // any specific session-vs-snapshot ordering is established by the
    // surrounding scoped_lock around the m_block snapshot.
    std::atomic<std::uint32_t> m_difficulty{ 0 };

    std::uint32_t m_pool_nbits;

    std::uint64_t m_nonce = 0;
    uint1k m_base_hash;
    static LLC::CBigNum boost_uint1024_t_to_CBignum(const uint1k&);
    static uint1024_t boost_uint1024_t_to_uint1024_t(const uint1k&);

    //stats
    uint64_t m_range_searched = 0;
    nexusminer::stats::Atomic_snapshot<nexusminer::stats::Prime> m_published_stats;

};
}

}


#endif
