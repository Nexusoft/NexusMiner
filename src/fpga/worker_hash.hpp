#ifndef NEXUSMINER_FPGA_WORKER_HASH_HPP
#define NEXUSMINER_FPGA_WORKER_HASH_HPP

#include <memory>
#include <mutex>
#include "worker.hpp"
#include "nexus_skein.hpp"
#include "nexus_keccak.hpp"
#include "nexus_hash_utils.hpp"
#include <spdlog/spdlog.h>
#include <asio.hpp>

namespace nexusminer {
namespace config { class Worker_config; }
namespace stats { class Collector; }
namespace fpga 
{

class Worker_hash : public Worker, public std::enable_shared_from_this<Worker_hash>
{
public:

    using Worker_config = config::Worker_config;

    Worker_hash(std::shared_ptr<asio::io_context> io_context, Worker_config& config);
    ~Worker_hash();

    // Sets a new block (nexus data type) for the miner worker. The miner worker must reset the current work.
    // When  the worker finds a new block, the BlockFoundHandler has to be called with the found BlockData
    void set_block(LLP::CBlock block, std::uint32_t nbits, Worker::Block_found_handler result) override;
    void update_statistics(stats::Collector& stats_collector) override;
    void set_test_block();

private:

    void start_read();
    void handle_read(const asio::error_code& error, std::size_t bytes_transferred);
    bool difficulty_check();

    static constexpr int baud = 230400;
    static constexpr int workPackageLength = 224; //bytes
    static constexpr int responseLength = 8; //bytes

    std::shared_ptr<asio::io_context> m_io_context;
    std::shared_ptr<spdlog::logger> m_logger;
    std::vector<unsigned char> m_receive_nonce_buffer;
    Worker_config& m_config;
    asio::serial_port m_serial;
    std::string m_serial_port_path;
    Worker::Block_found_handler m_found_nonce_callback;
    NexusSkein m_skein;
    uint64_t m_starting_nonce = 0;
    Block_data m_block;
    std::mutex m_mtx;
    std::string m_log_leader;


    static constexpr uint64_t nonce_difficulty_filter = 1ULL << 32;  //the fixed difficulty check inside the FPGA
    void reset_statistics();
    int m_nonce_candidates_recieved;
    int m_best_leading_zeros;
    int m_met_difficulty_count;

    std::uint32_t m_pool_nbits;
};

}
}


#endif