#ifndef NEXUSMINER_NETWORK_SOCKET_FACTORY_IMPL_HPP
#define NEXUSMINER_NETWORK_SOCKET_FACTORY_IMPL_HPP

#include "asio/io_service.hpp"
#include "socket_factory.hpp"
#include "tcp/socket_impl.hpp"
#include "tcp/protocol_description.hpp"

namespace nexusminer {
namespace network {

class Socket_factory_impl : public Socket_factory 
{
public:
    Socket_factory_impl(std::shared_ptr<::asio::io_context> io_context)
        : m_io_context{std::move(io_context)}
    {
    }

private:
    std::shared_ptr<asio::io_context> m_io_context;

    Socket::Sptr create_socket_impl(Endpoint local_endpoint) override
    {
        Socket::Sptr result{};
        if (local_endpoint.transport_protocol() == Transport_protocol::udp)
		{
			result = nullptr;
        }
        else if (local_endpoint.transport_protocol() == Transport_protocol::tcp)
		{
            result = std::make_shared<tcp::Socket_impl<tcp::Protocol_description>>(
                    m_io_context, std::move(local_endpoint));
        }

        return result;
    }
};

} 
} 

#endif 