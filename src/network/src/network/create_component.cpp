#include "network/create_component.hpp"
#include "network/component_impl.hpp"

#include <asio/io_service.hpp>
#include <memory>
#include <cassert>

namespace nexusminer {
namespace network {

Component::Uptr create_component(std::shared_ptr<asio::io_context> io_context)
{
    assert(io_context);
    return std::make_unique<Component_impl>(std::move(io_context));
}

} 
} 
