#include "worker_manager.hpp"
#include "worker_software_hash.hpp"
#include "fpga/worker_fpga.hpp"
#include "packet.hpp"
#include "config.hpp"
#include "LLP/block.hpp"

namespace nexusminer
{
Worker_manager::Worker_manager(Config& config, chrono::Timer_factory::Sptr timer_factory, network::Socket::Sptr socket)
: m_config{config}
, m_socket{std::move(socket)}
, m_logger{spdlog::get("logger")}
, m_timer_factory{std::move(timer_factory)}
, m_current_height{0}
{
    m_connection_retry_timer = m_timer_factory->create_timer();
    m_statistics_timer = m_timer_factory->create_timer();
    m_get_height_timer = m_timer_factory->create_timer();
}

void Worker_manager::stop()
{
    m_connection_retry_timer->cancel();
    m_statistics_timer->cancel();
    m_get_height_timer->cancel();

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

                // retry connect
                auto const connection_retry_interval = self->m_config.get_connection_retry_interval();
                self->m_logger->info("Connection retry {} seconds", connection_retry_interval);
                self->m_connection_retry_timer->start(chrono::Seconds(connection_retry_interval), 
                    self->connection_retry_handler(wallet_endpoint));
            }
            else if (result == network::Result::connection_ok)
            {
                self->m_logger->info("Connection to wallet established");

                auto const print_statistics_interval = self->m_config.get_print_statistics_interval();
                self->m_statistics_timer->start(chrono::Seconds(print_statistics_interval), self->statistics_handler(print_statistics_interval));

                // set channel
                std::uint32_t channel = (self->m_config.get_mining_mode() == Config::PRIME) ? 1U : 2U;
                Packet packet_set_channel;
                packet_set_channel.m_header = Packet::SET_CHANNEL;
                packet_set_channel.m_length = 4;
                packet_set_channel.m_data = std::make_shared<std::vector<std::uint8_t>>(uint2bytes(channel));
                self->m_connection->transmit(packet_set_channel.get_bytes());

                self->m_current_height = 0;     // reset height
                auto const get_height_interval = self->m_config.get_height_interval();
                self->m_get_height_timer->start(chrono::Seconds(get_height_interval), self->get_height_handler(get_height_interval));
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

chrono::Timer::Handler Worker_manager::connection_retry_handler(network::Endpoint const& wallet_endpoint)
{
    std::weak_ptr<Worker_manager> weak_self = shared_from_this();
    return[weak_self, wallet_endpoint](bool canceled)
    {
        if (canceled)	// don't do anything if the timer has been canceled
        {
            return;
        }

        auto self = weak_self.lock();
        if(self)
        {
            self->connect(wallet_endpoint);
        }
    }; 
}

chrono::Timer::Handler Worker_manager::statistics_handler(std::uint16_t print_statistics_interval)
{
    std::weak_ptr<Worker_manager> weak_self = shared_from_this();
    return[weak_self, print_statistics_interval](bool canceled)
    {
        if (canceled)	// don't do anything if the timer has been canceled
        {
            return;
        }

        auto self = weak_self.lock();
        if(self)
        {
            // print statistics
             for(auto& worker : self->m_workers)
             {
                 worker->print_statistics();
             }

            // restart timer
            self->m_statistics_timer->start(chrono::Seconds(print_statistics_interval), 
                self->statistics_handler(print_statistics_interval));
        }
    }; 
}

chrono::Timer::Handler Worker_manager::get_height_handler(std::uint16_t get_height_interval)
{
    std::weak_ptr<Worker_manager> weak_self = shared_from_this();
    return[weak_self, get_height_interval](bool canceled)
    {
        if (canceled)	// don't do anything if the timer has been canceled
        {
            return;
        }

        auto self = weak_self.lock();
        if(self)
        {
            Packet packet_get_height;
            packet_get_height.m_header = Packet::NEW_BLOCK;
            self->m_connection->transmit(packet_get_height.get_bytes());
            // restart timer
            self->m_statistics_timer->start(chrono::Seconds(get_height_interval), 
                self->get_height_handler(get_height_interval));
        }
    }; 
}

void Worker_manager::add_worker(std::shared_ptr<Worker> worker)
{
    m_workers.push_back(std::move(worker));
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
			m_logger->debug("Height Received {}", height);

			if (height > m_current_height)
			{
				m_current_height = height;

                get_block();             
			}			
		}
        // Block from wallet received
        else if(packet.m_header == Packet::BLOCK_DATA)
        {
            m_logger->debug("Block data received");
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
            // TODO add to statistics

        }
        else if(packet.m_header == Packet::REJECT)
        {
            m_logger->warn("Block Rejected by Nexus Network.");
            get_block();
            // TODO add to statistics
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