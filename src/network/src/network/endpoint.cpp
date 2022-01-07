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

static std::string port_to_string(std::uint16_t port)
{
    return std::to_string(static_cast<std::uint32_t>(port));
}

std::string Endpoint::to_string() const
{
    // Note: address().to_string() -> for ipv6 asio already adds %scope_id if scope_id != 0 (check
    // asio code 'inet_ntop') only call this for tcp or udp because for uds the is_v6() method would
    // assert
    std::string scope_id_string{};
    if ((m_protocol == Transport_protocol::udp) || (m_protocol == Transport_protocol::tcp)) {
        scope_id_string = ((scope_id() != 0) && is_v6()) ? "%" + std::to_string(scope_id()) : "";
    }

    std::string result{};
    if (m_protocol == Transport_protocol::udp) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-union-access)
        result += "udp://" + m_endpoint.udp.address().to_string() + scope_id_string + "#" +
            port_to_string(port());
    }
    else if (m_protocol == Transport_protocol::tcp) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-union-access)
        result += "tcp://" + m_endpoint.tcp.address().to_string() + scope_id_string + "#" +
            port_to_string(port());
    }
    else {
        // If the Endpoint is not valid
    }
    return result;
}


constexpr Scope_id Endpoint::m_default_scope;
constexpr Scope_id Endpoint::m_ephemeral_port;

}
} 
