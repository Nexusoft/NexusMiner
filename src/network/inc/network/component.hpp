#ifndef NEXUSMINER_NETWORK_COMPONENT_HPP
#define NEXUSMINER_NETWORK_COMPONENT_HPP

#include "network/socket_factory.hpp"

#include <memory>

namespace nexusminer {
namespace network {

class Component {
public:
    /// \brief Pointer to component
    using Uptr = std::unique_ptr<Component>;

    virtual ~Component() = default;

    // Get the Socket_factory interface
    virtual Socket_factory::Sptr get_socket_factory() = 0;
};


} 
} 

#endif 
