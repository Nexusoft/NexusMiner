#include "worker_fpga.hpp"
#include "statistics.hpp"


namespace nexusminer
{

Worker_fpga::Worker_fpga()
{}

void Worker_fpga::set_block(const LLP::CBlock& block, Worker::Block_found_handler result)
{
    
}

void Worker_fpga::print_statistics()
{
    m_statistics->print();
}

}