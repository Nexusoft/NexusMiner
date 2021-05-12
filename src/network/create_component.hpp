#ifndef NEXUSMINER_NETWORK_CREATE_COMPONENT_HPP
#define NEXUSMINER_NETWORK_CREATE_COMPONENT_HPP

#include "component.hpp"
#include "asio/io_service.hpp"

#include <memory>

namespace nexusminer {
namespace network {

// Component factory

Component::Uptr create_component(std::shared_ptr<::asio::io_context> io_context);

}
} 

#endif 
