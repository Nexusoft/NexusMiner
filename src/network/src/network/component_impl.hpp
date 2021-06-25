#ifndef NEXUSMINER_NETWORK_COMPONENT_IMPL_HPP
#define NEXUSMINER_NETWORK_COMPONENT_IMPL_HPP

#include "network/create_component.hpp"
#include "network/socket_factory_impl.hpp"

#include <memory>
#include "asio/io_service.hpp"

namespace nexusminer {
namespace network {

class Component_impl : public Component 
{
public:
    Component_impl(std::shared_ptr<::asio::io_context> io_context)
        : m_socket_factory{std::make_shared<Socket_factory_impl>(std::move(io_context))}
	{
    }

    Socket_factory::Sptr get_socket_factory() override { return m_socket_factory; }

private:
    Socket_factory::Sptr m_socket_factory;
};

}
}

#endif
