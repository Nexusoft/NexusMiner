#include "worker_manager.hpp"
#include "packet.hpp"
#include "config.hpp"
#include "fpga/worker_fpga.hpp"
#include "block.hpp"

namespace nexusminer
{
Worker_manager::Worker_manager(Config& config, network::Socket::Sptr socket)
: m_config{config}
, m_socket{std::move(socket)}
, m_logger{spdlog::get("logger")}
, m_current_height{0}
{
}

bool Worker_manager::connect(network::Endpoint const& wallet_endpoint)
{
    auto connection = m_socket->connect(wallet_endpoint, [self = shared_from_this()](auto result, auto receive_buffer)
    {
        if (result == network::Result::connection_declined ||
            result == network::Result::connection_aborted ||
            result == network::Result::connection_closed ||
            result == network::Result::connection_error)
        {
            self->m_logger->error("Connection to wallet not sucessful. Result: {}", result);
            self->m_connection = nullptr;		// close connection (socket etc)
            // retry connect
        }
        else if (result == network::Result::connection_ok)
        {
            self->m_logger->info("Connection to wallet established");
        //	self->m_maintenance_timer->start(chrono::Seconds(60), self->maintenance_timer_handler());
          //  self->m_block_timer->start(chrono::Milliseconds(50), self->block_timer_handler());
         //   self->m_orphan_check_timer->start(chrono::Seconds(20), self->orphan_check_timer_handler());			
           // self->m_get_height_timer->start(chrono::Seconds(2), self->get_height_timer_handler());

            // set channel
       

            std::uint32_t channel = (self->m_config.get_mining_mode() == Config::PRIME) ? 1U : 2U;
            Packet packet;
		    packet.m_header = Packet::SET_CHANNEL;
		    packet.m_length = 4;
		    packet.m_data = std::make_shared<std::vector<std::uint8_t>>(uint2bytes(channel));

            self->m_connection->transmit(packet.get_bytes());

        }
        else
        {	// data received
            self->process_data(std::move(receive_buffer));
        }

    });

    if(!connection)
    {
        return false;
    }

    m_connection = std::move(connection);
    return true;
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
				m_logger->info("Nexus Network: New Block [Height] {}", height);
				m_current_height = height;

                // get new block from wallet
                Packet packet_get_block;
		        packet_get_block.m_header = Packet::GET_BLOCK;

                m_connection->transmit(packet.get_bytes());                
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
                    worker->set_block(block, [self = shared_from_this()](auto block_data)
                    {
                        // create block and submit
                    });
                }
			}
			else
			{
				m_logger->info("Block Obsolete Height = {}, Skipping over.", block.nHeight);
			}
        }
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