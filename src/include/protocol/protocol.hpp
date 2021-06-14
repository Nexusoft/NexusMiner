#ifndef NEXUSMINER_PROTOCOL_PROTOCOL_HPP
#define NEXUSMINER_PROTOCOL_PROTOCOL_HPP

#include "../network/types.hpp"

namespace nexusminer {
namespace protocol
{

class Protocol {
public:

    virtual ~Protocol() = default;

    virtual network::Shared_payload login() = 0;
    virtual network::Shared_payload get_work() = 0;
};

}
}
#endif