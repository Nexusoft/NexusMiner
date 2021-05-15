#include "worker_manager.hpp"
#include "packet.hpp"
#include "config.hpp"
#include "fpga/worker_fpga.hpp"

namespace nexusminer
{
Worker_manager::Worker_manager(Config& config, network::Socket::Sptr socket)
: m_config{config}
, m_socket{std::move(socket)}
, m_logger{spdlog::get("logger")}
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

void Worker_manager::process_data(network::Shared_payload&& receive_buffer)
{


if( response == Packet::ACCEPT)
{

}
}

}