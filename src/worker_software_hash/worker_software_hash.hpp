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
#include <asio.hpp>
#include <spdlog/spdlog.h>

namespace nexusminer {

class Statistics;

class Worker_software_hash : public Worker, public std::enable_shared_from_this<Worker_software_hash>
{
public:

    Worker_software_hash(std::shared_ptr<asio::io_context> io_context);
    ~Worker_software_hash();

    // Sets a new block (nexus data type) for the miner worker. The miner worker must reset the current work.
    // When  the worker finds a new block, the BlockFoundHandler has to be called with the found BlockData
    void set_block(const LLP::CBlock& block, Worker::Block_found_handler result) override;
    void print_statistics() override;

    Block_data get_block_data() const;


private:

    void run();
    bool difficultyCheck();
    uint64_t leadingZeroMask();  

    std::atomic<bool> stop;
    std::thread runThread;
    Worker::Block_found_handler foundNonceCallback;
    std::unique_ptr<Statistics> m_statistics;

    int leadingZerosRequired;  //Poor man's difficulty.  Report any nonces with at least this many leading zeros. Let the software perform additional filtering. 
    
    NexusSkein skein;
    Block_data block_;
    std::mutex mtx;
    //std::condition_variable cv;
    //std::atomic<bool> mine = false;

    std::shared_ptr<asio::io_context> m_io_context;
    std::shared_ptr<spdlog::logger> m_logger;


};

}


#endif