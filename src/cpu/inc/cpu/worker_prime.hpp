#ifndef NEXUSMINER_CPU_WORKER_PRIME_HPP
#define NEXUSMINER_CPU_WORKER_PRIME_HPP
//a software prime channel miner for demo and test purposes

#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include "worker.hpp"
#include <spdlog/spdlog.h>

namespace asio { class io_context; }

namespace nexusminer {
namespace config { class Worker_config; }
namespace stats { class Collector; }

namespace cpu
{
class Worker_prime : public Worker, public std::enable_shared_from_this<Worker_prime>
{
public:

    Worker_prime(std::shared_ptr<asio::io_context> io_context, config::Worker_config& config);
    ~Worker_prime() noexcept override;

    void set_block(LLP::CBlock block, std::uint32_t nbits, Worker::Block_found_handler result) override;
    void update_statistics(stats::Collector& stats_collector) override;

private:

    void run();
    bool difficulty_check();
    std::uint64_t leading_zero_mask();


    //Poor man's difficulty.  Report any nonces with at least this many leading zeros. Let the software perform additional filtering. 
    static constexpr int leading_zeros_required = 20;    //set lower to find more nonce candidates

    std::shared_ptr<asio::io_context> m_io_context;
    std::shared_ptr<spdlog::logger> m_logger;
    config::Worker_config& m_config;
    std::atomic<bool> m_stop;
    std::thread m_run_thread;
    Worker::Block_found_handler m_found_nonce_callback;

    Block_data m_block;
    std::mutex m_mtx;
    uint64_t m_starting_nonce = 0;
    std::string m_log_leader;

    void reset_statistics();
    std::uint64_t m_hash_count;
    int m_best_leading_zeros;
    int m_met_difficulty_count;

    std::uint32_t m_pool_nbits;

};
}

}


#endif