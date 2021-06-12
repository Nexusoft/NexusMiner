#ifndef NEXUSMINER_WORKER_SOFTWARE_HASH_HPP
#define NEXUSMINER_WORKER_SOFTWARE_HASH_HPP
//a software hash channel miner for demo and test purposes

#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include "worker.hpp"
#include "nexus_skein.hpp"
#include "nexus_keccak.hpp"
#include <spdlog/spdlog.h>

namespace asio { class io_context; }

namespace nexusminer {
namespace config{ class Worker_config; }

class Stats_collector;


class Worker_software_hash : public Worker, public std::enable_shared_from_this<Worker_software_hash>
{
public:

    using Worker_config = config::Worker_config;

    Worker_software_hash(std::shared_ptr<asio::io_context> io_context, Worker_config& config);
    ~Worker_software_hash();

    // Sets a new block (nexus data type) for the miner worker. The miner worker must reset the current work.
    // When  the worker finds a new block, the BlockFoundHandler has to be called with the found BlockData
    void set_block(const LLP::CBlock& block, Worker::Block_found_handler result) override;
    void update_statistics(Stats_collector& stats_collector) override;

private:

    void run();
    bool difficulty_check();
    std::uint64_t leading_zero_mask();  
 

    //Poor man's difficulty.  Report any nonces with at least this many leading zeros. Let the software perform additional filtering. 
    static constexpr int leading_zeros_required = 20;    //set lower to find more nonce candidates

    std::shared_ptr<asio::io_context> m_io_context;
    std::shared_ptr<spdlog::logger> m_logger;
    Worker_config& m_config;
    std::atomic<bool> m_stop;
    std::thread m_run_thread;
    Worker::Block_found_handler m_found_nonce_callback;
    NexusSkein m_skein;
    Block_data m_block;
    std::mutex m_mtx;
    uint64_t m_starting_nonce = 0;
    std::string m_log_leader;

    //stats
    //TODO move to statistics class
 
    void reset_statistics();
    std::uint64_t m_hash_count;
    int m_best_leading_zeros;
    int m_met_difficulty_count;


};

}


#endif