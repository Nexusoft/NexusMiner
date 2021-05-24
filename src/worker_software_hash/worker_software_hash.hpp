#ifndef NEXUSMINER_WORKER_SOFTWARE_HASH_HPP
#define NEXUSMINER_WORKER_SOFTWARE_HASH_HPP
//a software hash channel miner for demo and test purposes

#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include "worker.hpp"
#include "nexus_skein.hpp"
#include "nexus_keccak.hpp"
#include <asio.hpp>

namespace nexusminer {

class Statistics;

class Worker_software_hash : public Worker
{
public:

    Worker_software_hash( );

    // Sets a new block (nexus data type) for the miner worker. The miner worker must reset the current work.
    // When  the worker finds a new block, the BlockFoundHandler has to be called with the found BlockData
    void set_block(const LLP::CBlock& block, Worker::Block_found_handler result) override;
    void print_statistics() override;
    ~Worker_software_hash() { stop = true;  runThread.join(); }

private:
    std::atomic<bool> stop;
    std::thread runThread;
    void run();
    Worker::Block_found_handler foundNonceCallback;
    bool difficultyCheck();
    std::unique_ptr<Statistics> m_statistics;

    static constexpr int leadingZeros = 20;  //Poor man's difficulty.  Report any nonces with at least this many leading zeros. Let the software perform additional filtering. 
    static constexpr uint64_t leadingZeroMask = ((1ull << leadingZeros) - 1) << (64 - leadingZeros);
    NexusSkein skein;
    Block_data block_;
    std::mutex mtx;
    std::condition_variable cv;
    bool mine = false;
    std::shared_ptr<asio::io_context> m_io_context_;


};

}


#endif