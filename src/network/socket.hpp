#ifndef NEXUSMINER_NETWORK_SOCKET_HPP
#define NEXUSMINER_NETWORK_SOCKET_HPP

#include "connection.hpp"

#include <functional>
#include <memory>

namespace nexusminer {
namespace network {

// This interface represents a socket. A socket is a communication
// resource bound to a local endpoint, providing (active and passive) connection setup functionality.
// All connections created from a Socket are independent in
// life-time from the Socket itself. Thus, destroying the Socket, does not affect existence and proper operation of Connections
// created from the Socket.
//
// Once a transport-protocol-socket is opened, it is closed only if the socket object is destroyed.
// Ortherwise the transport-protocol-socket resource is kept (including the port number obtained).
// Consequently, once an ephemeral port is selected, it belongs to the socket until the socket is destroyed.
class Socket {
public:

	virtual ~Socket() = default;

    using Sptr = std::shared_ptr<Socket>;

    // A call to the Connect_handler informs the user about a new connection object (not necessarily an established connection).
    // The user finally provides a handler of type Connection::Handler (connection_handler) as return value.
    // Events on the connection (Result::Categories connection, receive, transmit) are indicated to the connection_handler.
    //
    // The user may decline the connection by providing an invalid connection_handler, and finally delete the Connection object.
    // The user may accept but ignore the connection by providing an empty (no code containing) connection_handler, and keeping the Connection object.
    using Connect_handler = std::function<Connection::Handler(Connection::Sptr&& connection)>;

    // Start listening for connections
    //
    // Update the port information of local_endpoint (if not already opened) in order to get the actually used port number.
	//
    // Configure the transport-protocol-socket to accept new incoming connections (protocol specific).
    // (For connection-less protocols a connection is established on the first data exchange with a particular remote endpoint.)
	// If any transport-protocol-socket operation fails, close the transport-protocol-socket and return Result::socket_error.
    // Return Result::socket_ok if no error occured.
    // If a new connection is detected, call 'handler' in order to provide the connection object (asynchronous operation).
    virtual Result::Code listen(Connect_handler handler) = 0;

    // Stop listening for connections
    // Already established connections are not affected.
    virtual void stop_listen() = 0;

    // Returns the local endpoint all this socket is uses for communication (receives data at, sends data from).
    virtual Endpoint const& local_endpoint() const = 0;

    // Connect to a remote endpoint
    // If any transport-protocol-socket operation fails, close the transport-protocol-socket and return nullptr.
    //
    // If a connection to the given remote_endpoint already exist, return nullptr.
    // Duplicate connections on the same [socket,remote_endpoint] tuple are not possible.
    //
    // In case the connection setup happens immediately within the function call, 'handler' will be called within the context of
    // connect().
    virtual Connection::Sptr connect(Endpoint remote_endpoint, Connection::Handler handler) = 0;
};

} 
} 

#endif 