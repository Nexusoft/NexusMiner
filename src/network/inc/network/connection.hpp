#ifndef NEXUSMINER_NETWORK_CONNECTION_HPP
#define NEXUSMINER_NETWORK_CONNECTION_HPP

#include "network/endpoint.hpp"
#include "network/types.hpp"
#include "protocol_lane.hpp"

#include <functional>
#include <memory>

namespace nexusminer {
namespace network {

//  Provides connection control, data transfer and information functions
//
// This interface represents a generic Connection. A connection is a communication
// relationship between two endpoints. The connection interface provides functions for
// transfering data and information retrival.
// If a connection is established, call 'handler' with Result::Code::connection_ok.
//
class Connection {
public:

    using Sptr = std::shared_ptr<Connection>;

	virtual ~Connection() = default;

    // Type of handler called when events for a connection occur
    using Handler = std::function<void(Result::Code result, Shared_payload&& receive_buffer)>;

    // Returns the remote endpoint of the connection
    virtual Endpoint const& remote_endpoint() const = 0;

    //  Returns the local endpoint of the connection
    virtual Endpoint const& local_endpoint() const = 0;

    //  Transmit payload on the connection.
    //  Returns true if the payload was accepted into the connection's TX path,
    //  false if it was rejected locally (e.g. null/empty payload, closed
    //  connection, missing socket/handler, or TX queue overflow).
    virtual bool transmit(Shared_payload tx_buffer) = 0;

    // Closes the connection
    virtual void close() = 0;
    
    // Get protocol lane for this connection
    virtual ProtocolLane get_protocol_lane() const = 0;
};


}
}

#endif 
