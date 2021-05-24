#ifndef NEXUSMINER_WORKER_FPGA_HPP
#define NEXUSMINER_WORKER_FPGA_HPP

#include <memory>
#include "worker.hpp"

namespace nexusminer {

class Statistics;

class Worker_fpga : public Worker
{
public:

    Worker_fpga(/* fpga config */ );

    // Sets a new block (nexus data type) for the miner worker. The miner worker must reset the current work.
    // When  the worker finds a new block, the BlockFoundHandler has to be called with the found BlockData
    void set_block(const LLP::CBlock& block, Worker::Block_found_handler result) override;

    void print_statistics() override;

private:

    std::unique_ptr<Statistics> m_statistics;

};

}


#endif