#ifndef NEXUSMINER_NETWORK_TCP_PROTOCOL_DESCRIPTION_HPP
#define NEXUSMINER_NETWORK_TCP_PROTOCOL_DESCRIPTION_HPP

#include "asio/ip/tcp.hpp"
#include "../endpoint.hpp"

#include <string>

namespace nexusminer {
namespace network {
namespace tcp {


class Protocol_description {
public:
    using Endpoint = Endpoint_tcp;
    using Acceptor = ::asio::ip::tcp::acceptor;
    using Socket = ::asio::ip::tcp::socket;

    static void close_acceptor(Acceptor& acceptor)
    {
        if (acceptor.is_open())
		{
            ::asio::error_code error;
            acceptor.close(error);
        }
    }

    static Result::Code bind_acceptor(Acceptor& acceptor, network::Endpoint const& local_endpoint)
    {
        ::asio::error_code error;
        acceptor.bind(get_endpoint_base<Endpoint>(local_endpoint), error);
        if (error) 
		{
            close_acceptor(acceptor);
            return Result::socket_error;
        }

        return Result::ok;
    }

    static void update_port(Endpoint const& source, network::Endpoint& destination)
    {
        destination.port(source.port());
    }
};

}
}
} 

#endif