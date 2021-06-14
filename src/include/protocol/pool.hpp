#ifndef NEXUSMINER_PROTOCOL_POOL_HPP
#define NEXUSMINER_PROTOCOL_POOL_HPP

#include "protocol.hpp"

namespace nexusminer {
namespace protocol
{

class Pool : public Protocol {
public:

     network::Shared_payload login() { return network::Shared_payload{ }; }
     network::Shared_payload get_work() { return network::Shared_payload{ }; }

private:

};

}
}
#endif