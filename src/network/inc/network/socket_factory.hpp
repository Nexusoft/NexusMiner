#ifndef NEXUSMINER_NETWORK_SOCKET_FACTORY_HPP
#define NEXUSMINER_NETWORK_SOCKET_FACTORY_HPP

#include "network/endpoint.hpp"
#include "network/socket.hpp"

#include <memory>
#include <cassert>

namespace nexusminer {
namespace network {

// Provides socket factory functions
// All sockets created from the Socket_factory are independent in
// life-time from the Socket_factory itself. Thus, destroying the Socket_factory, does not affect existence and proper operation of Sockets
// created from the Socket_factory.
class Socket_factory {
public:

    using Sptr = std::shared_ptr<Socket_factory>;
    using Wptr = std::weak_ptr<Socket_factory>;

    // Creates a socket object
	//
    //   Creates and returns a socket object configured to the local_endpoint properties. The construction of the object does not yet
    //   perform any operation on the underlying layer but just instantiates classes and saves the passed information.
    //   Consequently there will be no error based on I/O operation during construction.
    Socket::Sptr create_socket(Endpoint local_endpoint);

private:
    virtual Socket::Sptr create_socket_impl(Endpoint local_endpoint) = 0;
};


inline Socket::Sptr Socket_factory::create_socket(Endpoint local_endpoint)
{
    assert(local_endpoint.is_valid());
    assert(!local_endpoint.is_multicast());

    // implements
    return create_socket_impl(std::move(local_endpoint));
}

}
} 

#endif
