#include "worker_manager.hpp"
#include "cpu/worker_software_hash.hpp"
#include "fpga/worker_fpga.hpp"
#include "packet.hpp"
#include "config.hpp"
#include "LLP/block.hpp"

namespace nexusminer
{
Worker_manager::Worker_manager(std::shared_ptr<asio::io_context> io_context, Config& config, 
    chrono::Timer_factory::Sptr timer_factory, network::Socket::Sptr socket)
: m_io_context{std::move(io_context)}
, m_config{config}
, m_socket{std::move(socket)}
, m_logger{spdlog::get("logger")}
, m_stats_collector{m_config}
, m_timer_manager{m_config, m_stats_collector, std::move(timer_factory)}
, m_current_height{0}
{
    create_workers();
}

void Worker_manager::create_workers()
{
    auto internal_id = 0U;
    for(auto& worker_config : m_config.get_worker_config())
    {
        worker_config.m_internal_id = internal_id;
        switch(worker_config.m_mode)
        {
            case Worker_config::FPGA:
            {
                m_workers.push_back(std::make_shared<Worker_fpga>(m_io_context, worker_config));
                break;
            }
            case Worker_config::GPU:
            {
                break;
            }
            case Worker_config::CPU:    // falltrough
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
                self->m_connection = nullptr;		// close connection (socket etc)
                self->m_current_height = 0;     // reset height
                self->m_stats_collector.connection_retry_attempt();

                // retry connect
                auto const connection_retry_interval = self->m_config.get_connection_retry_interval();
                self->m_logger->info("Connection retry {} seconds", connection_retry_interval);
                self->m_timer_manager.start_connection_retry_timer(self, wallet_endpoint);
            }
            else if (result == network::Result::connection_ok)
            {
                self->m_logger->info("Connection to wallet established");

                // set channel
                std::uint32_t channel = (self->m_config.get_mining_mode() == Config::PRIME) ? 1U : 2U;
                Packet packet_set_channel;
                packet_set_channel.m_header = Packet::SET_CHANNEL;
                packet_set_channel.m_length = 4;
                packet_set_channel.m_data = std::make_shared<std::vector<std::uint8_t>>(uint2bytes(channel));
                self->m_connection->transmit(packet_set_channel.get_bytes());

                self->m_current_height = 0;     // reset height
                self->m_timer_manager.start_get_height_timer(self->m_connection, self->m_workers);
                self->m_timer_manager.start_stats_printer_timer();
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
        else if (packet.m_header == Packet::BLOCK_HEIGHT)
		{
			auto const height = bytes2uint(*packet.m_data);
			//m_logger->debug("Height Received {}", height);

			if (height > m_current_height)
			{
				m_current_height = height;

                get_block();             
			}			
		}
        // Block from wallet received
        else if(packet.m_header == Packet::BLOCK_DATA)
        {
            auto block = deserialize_block(packet.m_data);
			if (block.nHeight == m_current_height)
			{
	            for(auto& worker : m_workers)
                {
                    worker->set_block(block, [self = shared_from_this()](auto id, auto block_data)
                    {
                        // create block and submit
                        self->m_logger->info("Submitting Block...");

                        Packet PACKET;
                        Packet submit_block;
		                submit_block.m_header = Packet::SUBMIT_BLOCK;
			
			            submit_block.m_data = std::make_shared<std::vector<uint8_t>>(block_data->merkle_root.GetBytes());
			            std::vector<std::uint8_t> nonce  = uint2bytes64(block_data->nNonce);

                        submit_block.m_data->insert(submit_block.m_data->end(), nonce.begin(), nonce.end());
			            submit_block.m_length = 72;

                        if (self->m_connection)
                            self->m_connection->transmit(submit_block.get_bytes());  
                        else
                            self->m_logger->error("No connection. Can't submit block.");
                    });
                }
			}
			else
			{
				m_logger->warn("Block Obsolete Height = {}, Skipping over.", block.nHeight);
			}
        }
        else if(packet.m_header == Packet::ACCEPT)
        {
            m_logger->info("Block Accepted By Nexus Network.");
            m_stats_collector.block_accepted();
        }
        else if(packet.m_header == Packet::REJECT)
        {
            m_logger->warn("Block Rejected by Nexus Network.");
            get_block();
            m_stats_collector.block_rejected();
        }
        else
        {
            m_logger->error("Invalid header received.");
        }
}

void Worker_manager::get_block()
{
    m_logger->info("Nexus Network: New Block [Height] {}", m_current_height);

    // get new block from wallet
    Packet packet_get_block;
    packet_get_block.m_header = Packet::GET_BLOCK;

    m_connection->transmit(packet_get_block.get_bytes());         
}

/** Convert the Header of a Block into a Byte Stream for Reading and Writing Across Sockets. **/
LLP::CBlock Worker_manager::deserialize_block(network::Shared_payload data)
{
    LLP::CBlock block;
    block.nVersion = bytes2uint(std::vector<uint8_t>(data->begin(), data->begin() + 4));

    block.hashPrevBlock.SetBytes(std::vector<uint8_t>(data->begin() + 4, data->begin() + 132));
    block.hashMerkleRoot.SetBytes(std::vector<uint8_t>(data->begin() + 132, data->end() - 20));

    block.nChannel = bytes2uint(std::vector<uint8_t>(data->end() - 20, data->end() - 16));
    block.nHeight = bytes2uint(std::vector<uint8_t>(data->end() - 16, data->end() - 12));
    block.nBits = bytes2uint(std::vector<uint8_t>(data->end() - 12, data->end() - 8));
    block.nNonce = bytes2uint64(std::vector<uint8_t>(data->end() - 8, data->end()));

    return block;
}

}