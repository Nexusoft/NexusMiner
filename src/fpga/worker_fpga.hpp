#ifndef NEXUSMINER_WORKER_FPGA_HPP
#define NEXUSMINER_WORKER_FPGA_HPP

#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>
#include "worker.hpp"
#include "nexus_skein.hpp"
#include "nexus_keccak.hpp"
#include "nexus_hash_utils.hpp"
#include <asio.hpp>
#include <spdlog/spdlog.h>

namespace nexusminer {

class Statistics;

class Worker_fpga : public Worker, public std::enable_shared_from_this<Worker_fpga>
{
public:

    Worker_fpga(std::shared_ptr<asio::io_context> io_context, int workerID, std::string serialPort);
    ~Worker_fpga();

    // Sets a new block (nexus data type) for the miner worker. The miner worker must reset the current work.
    // When  the worker finds a new block, the BlockFoundHandler has to be called with the found BlockData
    void set_block(const LLP::CBlock& block, Worker::Block_found_handler result) override;
    void print_statistics() override;

    Block_data get_block_data() const;
    void SetTestBlock();


private:

    void run();
    bool difficultyCheck();

    int baud = 230400;
    std::string serialPortStr;
    asio::serial_port serial;
    std::atomic<bool> stop;
    std::thread runThread;
    Worker::Block_found_handler foundNonceCallback;
    std::unique_ptr<Statistics> m_statistics;
    NexusSkein skein;
    static constexpr int workPackageLength = 224; //bytes
    static constexpr int responseLength = 8; //bytes
    uint64_t startingNonce = 0;


    Block_data block_;
    std::mutex mtx;

    std::shared_ptr<asio::io_context> m_io_context;
    std::shared_ptr<spdlog::logger> m_logger;
    std::string log_leader;



};

}


#endif