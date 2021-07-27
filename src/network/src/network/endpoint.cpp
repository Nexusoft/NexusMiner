#include "network/endpoint.hpp"

namespace nexusminer {
namespace network {

namespace internal {

::asio::ip::address_v4::bytes_type get_ipv4_bytes(::asio::ip::address const& address)
{
    return address.to_v4().to_bytes();
}

::asio::ip::address_v6::bytes_type get_ipv6_bytes(::asio::ip::address const& address)
{
    return address.to_v6().to_bytes();
}

} // namespace internal


constexpr Scope_id Endpoint::m_default_scope;
constexpr Scope_id Endpoint::m_ephemeral_port;

}
} 
