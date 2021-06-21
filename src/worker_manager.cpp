#include "worker_manager.hpp"
#include "cpu/worker_software_hash.hpp"
#include "fpga/worker_fpga.hpp"
#include "packet.hpp"
#include "config/config.hpp"
#include "config/types.hpp"
#include "LLP/block.hpp"
#include "stats/stats_printer_console.hpp"
#include "stats/stats_printer_file.hpp"
#include "stats/stats_collector.hpp"
#include "protocol/solo.hpp"
#include "protocol/pool.hpp"
#include <variant>

namespace nexusminer
{
Worker_manager::Worker_manager(std::shared_ptr<asio::io_context> io_context, Config& config, 
    chrono::Timer_factory::Sptr timer_factory, network::Socket::Sptr socket)
: m_io_context{std::move(io_context)}
, m_config{config}
, m_socket{std::move(socket)}
, m_logger{spdlog::get("logger")}
, m_stats_collector{std::make_shared<stats::Collector>(m_config)}
, m_timer_manager{std::move(timer_factory)}
{
    if(m_config.get_use_bool())
    {
        m_miner_protocol = std::make_shared<protocol::Pool>();
    }
    else
    {
      m_miner_protocol = std::make_shared<protocol::Solo>(m_config.get_mining_mode() == config::Mining_mode::PRIME ? 1U : 2U); 
    } 
  
    create_stats_printers();
    create_workers();
}

void Worker_manager::create_stats_printers()
{
    bool printer_console_created = false;
    bool printer_file_created = false;
    for(auto& stats_printer_config : m_config.get_stats_printer_config())
    {
        switch(stats_printer_config.m_mode)
        {
            case config::Stats_printer_mode::FILE:
            {
                if(!printer_file_created)
                {
                    printer_file_created = true;
                    auto& stats_printer_config_file = std::get<config::Stats_printer_config_file>(stats_printer_config.m_printer_mode);
                    m_stats_printers.push_back(std::make_shared<stats::Printer_file>(stats_printer_config_file.file_name, 
                        m_config.get_mining_mode(), m_config.get_worker_config(), *m_stats_collector));
                }

                break;
            }
            case config::Stats_printer_mode::CONSOLE:    // falltrough
            default:
            {
                if(!printer_console_created)
                {
                    printer_console_created = true;
                    m_stats_printers.push_back(std::make_shared<stats::Printer_console>(m_config.get_mining_mode(), 
                        m_config.get_worker_config(), *m_stats_collector));
                }

                break;
            }
        }
    }

    if(m_stats_printers.empty())
    {
        m_logger->warn("No stats printer configured.");
    }
}

void Worker_manager::create_workers()
{
    auto internal_id = 0U;
    for(auto& worker_config : m_config.get_worker_config())
    {
        worker_config.m_internal_id = internal_id;
        switch(worker_config.m_mode)
        {
            case config::Worker_mode::FPGA:
            {
                m_workers.push_back(std::make_shared<Worker_fpga>(m_io_context, worker_config));
                break;
            }
            case config::Worker_mode::GPU:
            {
                break;
            }
            case config::Worker_mode::CPU:    // falltrough
            default:
            {
                m_workers.push_back(std::make_shared<Worker_software_hash>(m_io_context, worker_config));
                break;
            }
        }
        internal_id++;
    }
}

void Worker_manager::stop()
{
    m_timer_manager.stop();

    // close connection
    m_connection.reset();

    // destroy workers
    for(auto& worker : m_workers)
    {
        worker.reset();
    }
}

void Worker_manager::retry_connect(network::Endpoint const& wallet_endpoint)
{           
    m_connection = nullptr;		// close connection (socket etc)
    m_miner_protocol->reset();
    m_stats_collector->connection_retry_attempt();

    // retry connect
    auto const connection_retry_interval = m_config.get_connection_retry_interval();
    m_logger->info("Connection retry {} seconds", connection_retry_interval);
    m_timer_manager.start_connection_retry_timer(connection_retry_interval, shared_from_this(), wallet_endpoint);
}

bool Worker_manager::connect(network::Endpoint const& wallet_endpoint)
{
    std::weak_ptr<Worker_manager> weak_self = shared_from_this();
    auto connection = m_socket->connect(wallet_endpoint, [weak_self, wallet_endpoint](auto result, auto receive_buffer)
    {
        auto self = weak_self.lock();
        if(self)
        {
            if (result == network::Result::connection_declined ||
                result == network::Result::connection_aborted ||
                result == network::Result::connection_closed ||
                result == network::Result::connection_error)
            {
                self->m_logger->error("Connection to wallet not sucessful. Result: {}", result);
                self->retry_connect(wallet_endpoint);
            }
            else if (result == network::Result::connection_ok)
            {
                self->m_logger->info("Connection to wallet established");

                // login
                self->m_connection->transmit(self->m_miner_protocol->login(self->m_config.get_pool_config().m_username,
                [self, wallet_endpoint](bool login_result)
                {
                    if(!login_result)
                    {
                        self->retry_connect(wallet_endpoint);
                        return;
                    }

                    auto const print_statistics_interval = self->m_config.get_print_statistics_interval();
                    self->m_timer_manager.start_stats_collector_timer(print_statistics_interval, self->m_workers, self->m_stats_collector);
                    self->m_timer_manager.start_stats_printer_timer(print_statistics_interval, self->m_stats_printers);
                    // only solo miner uses GET_HEIGHT message
                    if(!self->m_config.get_use_bool())
                    {
                        auto const get_height_interval = self->m_config.get_height_interval();
                        self->m_timer_manager.start_get_height_timer(get_height_interval, self->m_connection);
                    }

                    self->m_miner_protocol->set_block_handler([self](auto block, auto nBits)
                    {
                        for(auto& worker : self->m_workers)
                        {
                            worker->set_block(block, nBits, [self](auto id, auto block_data)
                            {
                                if (self->m_connection)
                                    self->m_connection->transmit(self->m_miner_protocol->submit_block(
                                        block_data->merkle_root.GetBytes(), uint2bytes64(block_data->nNonce)));
                                else
                                    self->m_logger->error("No connection. Can't submit block.");
                            });
                        }
                    });
                }));
            }
            else
            {	// data received
                self->process_data(std::move(receive_buffer));
            }
        }
    });

    if(!connection)
    {
        return false;
    }

    m_connection = std::move(connection);
    return true;
}

void Worker_manager::process_data(network::Shared_payload&& receive_buffer)
{
    Packet packet{ std::move(receive_buffer) };
    if (!packet.is_valid())
    {
        // log invalid packet
        m_logger->error("Received packet is invalid. Header: {0}", packet.m_header);
        return;
    }

    if (packet.m_header == Packet::PING)
    {
        Packet response;
        response = response.get_packet(Packet::PING);
        m_connection->transmit(response.get_bytes());
    }
    else
    {
        // solo/pool specific messages
        m_miner_protocol->process_messages(std::move(packet), m_connection);
    }
}

}