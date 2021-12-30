#ifndef NEXUSMINER_NETWORK_ENDPOINT_HPP
#define NEXUSMINER_NETWORK_ENDPOINT_HPP

#include "network/types.hpp"
#include "network/utility.hpp"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <string>
#include <type_traits>
#include <utility>
#include <cassert>

#include "asio/ip/address.hpp"
#include "asio/ip/address_v4.hpp"
#include "asio/ip/address_v6.hpp"
#include "asio/ip/tcp.hpp"
#include "asio/ip/udp.hpp"
#include "asio/local/stream_protocol.hpp"

namespace nexusminer {
namespace network {

using Endpoint_udp = ::asio::ip::udp::endpoint;
using Endpoint_tcp = ::asio::ip::tcp::endpoint;

namespace internal {

::asio::ip::address_v4::bytes_type get_ipv4_bytes(::asio::ip::address const& address);
::asio::ip::address_v6::bytes_type get_ipv6_bytes(::asio::ip::address const& address);

} // namespace internal


class Endpoint final {
public:
    using Transport_protocol = ::nexusminer::network::Transport_protocol;
    using Internet_protocol = ::nexusminer::network::Internet_protocol;

    Endpoint();

    // Construct endpoint from underlying representation
    explicit Endpoint(Endpoint_udp base_endpoint, Scope_id scope = m_default_scope);
    explicit Endpoint(Endpoint_tcp base_endpoint, Scope_id scope = m_default_scope);

    // Construct endpoint from string ip-Address (ip version auto detect)
    Endpoint(Transport_protocol protocol, std::string const& address,
             std::uint16_t port = m_ephemeral_port, Scope_id scope = m_default_scope);

    // Constructs an UDP/TCP endpoint from iterators 'begin' and 'end' and parameters 'port' and 'scope'
    template<typename Iterator>
    Endpoint(Transport_protocol protocol, Iterator const& begin, Iterator const& end,
             std::uint16_t port = m_ephemeral_port, Scope_id scope = m_default_scope);

    // Constructs an UDP/TCP endpoint with IPV4/IPV6 address from iterator 'begin' and parameters 'port' and 'scope'
    template<typename Iterator>
    Endpoint(Internet_protocol::Ipv4_tag tag, Transport_protocol protocol, Iterator const& begin,
             std::uint16_t port = m_ephemeral_port, Scope_id scope = m_default_scope);

    template<typename Iterator>
    Endpoint(Internet_protocol::Ipv6_tag tag, Transport_protocol protocol, Iterator const& begin,
             std::uint16_t port = m_ephemeral_port, Scope_id scope = m_default_scope);


    Endpoint(Endpoint const& other);
    Endpoint& operator=(Endpoint const& other);
    Endpoint(Endpoint&& other) noexcept;
    Endpoint& operator=(Endpoint&& other) noexcept;
    ~Endpoint();

    // Returns the binary (IPV4/IPV6) address of the endpoint in network byte order
    template<typename InternetProtocol, typename OutputIt>
    void address(InternetProtocol tag, OutputIt out) const;

	void address(std::string& out) const;

    std::string to_string() const;

    // Returns the port number
    std::uint16_t port() const;

    // Sets the port number
    void port(std::uint16_t port);

    // Returns the scope-id
    Scope_id scope_id() const;

    // Sets the scope-id
    void scope_id(Scope_id scope_id);

    // Returns whether this endpoint has an IPV6 address or not
    bool is_v6() const;

    // Returns whether this endpoint has an IPV4 address or not
    bool is_v4() const;

    // Returns whether this is an UDP endpoint or not
    bool is_udp() const;

    // Returns whether this is a TCP endpoint or not
    bool is_tcp() const;

    //  Returns whether this endpoint has a multicast address or not
    bool is_multicast() const;

    // Returns whether this endpoint is valid or not
    bool is_valid() const;

    // Return the endpoint's transport protocol
    Transport_protocol transport_protocol() const;

    explicit operator bool() const;
    bool operator==(Endpoint const& other) const;
    bool operator<(Endpoint const& other) const;

    // Get endpoint in underlying technology format (asio udp)
    Endpoint_udp const& get_udp_base() const;
    // Get endpoint in underlying technology format (asio tcp)
    Endpoint_tcp const& get_tcp_base() const;

private:
    Transport_protocol m_protocol{Transport_protocol::none};
    union Endpoint_base {
        Endpoint_udp udp;
        Endpoint_tcp tcp;
        explicit Endpoint_base(Endpoint_udp&& ep) : udp{std::move(ep)} {}
        explicit Endpoint_base(Endpoint_tcp&& ep) : tcp{std::move(ep)} {}
        explicit Endpoint_base() {}
    };
    Endpoint_base m_endpoint{};

    static constexpr Scope_id m_default_scope = 0U;
    static constexpr Scope_id m_ephemeral_port = 0U;

    void init_ip_endpoint(asio::ip::address const& ip_address, std::uint16_t port);
    template<typename IpAddressType>
    void init_ip_endpoint(IpAddressType ip_address, std::uint16_t port);
    template<typename Iterator>
    void init_ip_endpoint(Iterator const& begin, Iterator const& end, std::uint16_t port,
                          Scope_id scope, std::random_access_iterator_tag category);
    void copy_endpoint_base(Endpoint const& other);
    void move_endpoint_base(Endpoint&& other);
    void destruct_endpoint_base();

    template<typename F>
    auto call_on_base_address(F const& function) const
    {
		if (m_protocol == Transport_protocol::udp) {
			return function(m_endpoint.udp.address());
		}
		else {
            return function(m_endpoint.tcp.address());
        }
    }

    ::asio::ip::address_v4::bytes_type get_address_bytes(Internet_protocol::Ipv4_tag tag) const;
    ::asio::ip::address_v6::bytes_type get_address_bytes(Internet_protocol::Ipv6_tag tag) const;
};

inline Endpoint::Endpoint() = default;


inline Endpoint::Endpoint(Endpoint_udp base_endpoint, Scope_id scope)
    : m_protocol{Transport_protocol::udp}
    , m_endpoint{((scope != m_default_scope) && base_endpoint.address().is_v6())
                     ? Endpoint_udp{asio::ip::address{asio::ip::address_v6{
                                        base_endpoint.address().to_v6().to_bytes(), scope}},
                                    base_endpoint.port()}
                     : std::move(base_endpoint)}
{
}

inline Endpoint::Endpoint(Endpoint_tcp base_endpoint, Scope_id scope)
    : m_protocol{Transport_protocol::tcp}
    , m_endpoint{((scope != m_default_scope) && base_endpoint.address().is_v6())
                     ? Endpoint_tcp{asio::ip::address{asio::ip::address_v6{
                                        base_endpoint.address().to_v6().to_bytes(), scope}},
                                    base_endpoint.port()}
                     : std::move(base_endpoint)}
{
}


inline Endpoint::Endpoint(Transport_protocol protocol, std::string const& address,
                          std::uint16_t port, Scope_id scope)
    : m_protocol{protocol}, m_endpoint{}
{
    ::asio::error_code ec;
    auto ip_address = ::asio::ip::make_address(address, ec);

    if (!ec) 
    {
        if ((scope != m_default_scope) && ip_address.is_v6()) 
        {
            auto ipv6_address = ip_address.to_v6();
            ipv6_address.scope_id(scope);
            ip_address = ipv6_address;
        }
    }
    else 
    {
        m_protocol = Transport_protocol::none;
    }
    init_ip_endpoint(ip_address, port);
}

template<typename Iterator>
inline Endpoint::Endpoint(Transport_protocol protocol, Iterator const& begin, Iterator const& end,
                          std::uint16_t port, Scope_id scope)
    : m_protocol{protocol}, m_endpoint{}
{
    assert(is_valid());
    assert(is_udp() || is_tcp());
    using category = typename std::iterator_traits<Iterator>::iterator_category;
    init_ip_endpoint(begin, end, port, scope, category());
}


template<typename Iterator>
inline Endpoint::Endpoint(Internet_protocol::Ipv4_tag /* tag */, Transport_protocol protocol,
                          Iterator const& begin, std::uint16_t port, Scope_id scope)
    : Endpoint(protocol, begin, begin + std::tuple_size<asio::ip::address_v4::bytes_type>::value,
               port, scope)
{
}

template<typename Iterator>
inline Endpoint::Endpoint(Internet_protocol::Ipv6_tag /* tag */, Transport_protocol protocol,
                          Iterator const& begin, std::uint16_t port, Scope_id scope)
    : Endpoint(protocol, begin, begin + std::tuple_size<asio::ip::address_v6::bytes_type>::value,
               port, scope)
{
}

inline Endpoint::Endpoint(Endpoint const& other) : m_protocol{other.m_protocol}
{
    copy_endpoint_base(other);
}

inline Endpoint& Endpoint::operator=(Endpoint const& other)
{
    m_protocol = other.m_protocol;
    copy_endpoint_base(other);
    return *this;
}

inline Endpoint::Endpoint(Endpoint&& other) noexcept : m_protocol{other.m_protocol}
{
    move_endpoint_base(std::move(other));
}

inline Endpoint& Endpoint::operator=(Endpoint&& other) noexcept
{
    m_protocol = other.m_protocol;
    move_endpoint_base(std::move(other));
    return *this;
}

inline Endpoint::~Endpoint()
{
    destruct_endpoint_base();
}

template<typename InternetProtocol, typename OutputIt>
inline void Endpoint::address(InternetProtocol tag, OutputIt out) const
{
    assert(is_valid());
    assert(is_udp() || is_tcp());

    auto const address_bytes = get_address_bytes(tag);
    std::copy(address_bytes.begin(), address_bytes.end(), out);
}

inline void Endpoint::address(std::string& out) const
{
	assert(is_valid());
	assert(is_udp() || is_tcp());

	::asio::ip::address asio_address;

	if (m_protocol == Transport_protocol::tcp)
	{
		asio_address = m_endpoint.tcp.address();
	}
	else
	{
		asio_address = m_endpoint.udp.address();
	}

	out = asio_address.to_string();
}


inline std::uint16_t Endpoint::port() const
{
    assert(is_valid());
	assert(is_udp() || is_tcp());
    return ((m_protocol == Transport_protocol::udp)
                ? m_endpoint.udp.port()
                : m_endpoint.tcp.port());
}


inline void Endpoint::port(std::uint16_t port)
{
	assert(is_valid());
	assert(is_udp() || is_tcp());
    ((m_protocol == Transport_protocol::udp)
         ? m_endpoint.udp.port(port)
         : m_endpoint.tcp.port(port));
}

inline Scope_id Endpoint::scope_id() const
{
    assert(is_valid());
	assert(is_udp() || is_tcp());
    if (is_v6()) {
        return call_on_base_address([](auto const& address) { return address.to_v6().scope_id(); });
    }
    return m_default_scope;
}

inline void Endpoint::scope_id(Scope_id scope_id)
{
	assert(is_valid());
	assert(is_udp() || is_tcp());
    if (is_v6() && (this->scope_id() != scope_id)) {
        init_ip_endpoint(call_on_base_address([scope_id](auto const& address) {
                             return asio::ip::address{
                                 asio::ip::address_v6{address.to_v6().to_bytes(), scope_id}};
                         }),
                         port());
    }
}

inline bool Endpoint::is_v6() const
{
	assert(is_valid());
	assert(is_udp() || is_tcp());
    return call_on_base_address([](auto const& address) { return address.is_v6(); });
}

inline bool Endpoint::is_v4() const
{
	assert(is_valid());
	assert(is_udp() || is_tcp());
    return call_on_base_address([](auto const& address) { return address.is_v4(); });
}
inline bool Endpoint::is_udp() const
{
    return (m_protocol == Transport_protocol::udp);
}

inline bool Endpoint::is_tcp() const
{
    return (m_protocol == Transport_protocol::tcp);
}

inline bool Endpoint::is_multicast() const
{
	assert(is_valid());
    return call_on_base_address([](auto const& address) { return address.is_multicast(); });
}

inline bool Endpoint::is_valid() const
{
    return (m_protocol != Transport_protocol::none);
}

inline Endpoint::operator bool() const
{
    return is_valid();
}

inline Transport_protocol Endpoint::transport_protocol() const
{
    return m_protocol;
}

inline bool Endpoint::operator==(Endpoint const& other) const
{
    bool result;
    if (other.m_protocol == m_protocol) {
        switch (m_protocol) {
        case Transport_protocol::udp: {
            result = (m_endpoint.udp == other.m_endpoint.udp);
            break;
        }
        case Transport_protocol::tcp: {
            result = (m_endpoint.tcp == other.m_endpoint.tcp);
            break;
        }
        default:
            // Transport_protocol::none
            result = false;
            break;
        }
    }
    else {
        result = false;
    }

    return result;
}

inline bool Endpoint::operator<(Endpoint const& other) const
{
    bool result;
    if (other.m_protocol == m_protocol) {
        switch (m_protocol) {
        case Transport_protocol::udp: {
            result = (m_endpoint.udp < other.m_endpoint.udp);
            break;
        }
        case Transport_protocol::tcp: {
            result = (m_endpoint.tcp < other.m_endpoint.tcp);
            break;
        }
        default:
            // Transport_protocol::none
            result = false;
            break;
        }
    }
    else {
        result = (m_protocol < other.m_protocol);
    }
    return result;
}

inline Endpoint_udp const& Endpoint::get_udp_base() const
{
	assert(is_valid());
	assert(is_udp());
    return m_endpoint.udp;
}

inline Endpoint_tcp const& Endpoint::get_tcp_base() const
{
	assert(is_valid());
	assert(is_tcp());
    return m_endpoint.tcp;
}

// Private API implementation
inline void Endpoint::init_ip_endpoint(asio::ip::address const& ip_address, std::uint16_t port)
{
    if (m_protocol == Transport_protocol::udp) {
        m_endpoint.udp = Endpoint_udp(ip_address, port);
    }
    else if (m_protocol == Transport_protocol::tcp) {
        // check if address is a multicast address
        if (ip_address.is_multicast()) {
            m_protocol = Transport_protocol::none; // invalid endpoint
        }
        else {
            m_endpoint.tcp = Endpoint_tcp(ip_address, port);
        }
    }
    else {
    }
}

template<typename Iterator>
inline void Endpoint::init_ip_endpoint(Iterator const& begin, Iterator const& end,
                                       std::uint16_t port, Scope_id scope,
                                       std::random_access_iterator_tag /* tag */)
{
    const auto size = std::distance(begin, end);

    if (size == std::tuple_size<asio::ip::address_v4::bytes_type>::value) {
        init_ip_endpoint(
            asio::ip::address_v4{
                *reinterpret_cast<asio::ip::address_v4::bytes_type const*>(&(*begin))},
            port);
    }
    else if (size == std::tuple_size<asio::ip::address_v6::bytes_type>::value) {
        init_ip_endpoint(
            asio::ip::address_v6{
                *reinterpret_cast<asio::ip::address_v6::bytes_type const*>(&(*begin)), scope},
            port);
    }
    else {
        m_protocol = Transport_protocol::none;
    }
}

template<typename IpAddressType>
inline void Endpoint::init_ip_endpoint(IpAddressType ip_address, std::uint16_t port)
{
    init_ip_endpoint(asio::ip::address{std::move(ip_address)}, port);
}

inline void Endpoint::copy_endpoint_base(Endpoint const& other)
{
    if (m_protocol == Transport_protocol::udp) {
        m_endpoint.udp = other.m_endpoint.udp;
    }
    else if (m_protocol == Transport_protocol::tcp) {
        m_endpoint.tcp = other.m_endpoint.tcp;
    }
}

inline void Endpoint::move_endpoint_base(Endpoint&& other)
{
    if (m_protocol == Transport_protocol::udp) {
        m_endpoint.udp = std::move(other.m_endpoint.udp);
    }
    else if (m_protocol == Transport_protocol::tcp) {
        m_endpoint.tcp = std::move(other.m_endpoint.tcp);
    }
}

inline void Endpoint::destruct_endpoint_base()
{
    if (m_protocol == Transport_protocol::udp) {
        m_endpoint.udp.~Endpoint_udp();
    }
    else if (m_protocol == Transport_protocol::tcp) {
        m_endpoint.tcp.~Endpoint_tcp();
    }
}

inline asio::ip::address_v4::bytes_type
    Endpoint::get_address_bytes(Internet_protocol::Ipv4_tag /* tag */) const
{
    return call_on_base_address(
        [](auto const& address) { return internal::get_ipv4_bytes(address); });
}

inline asio::ip::address_v6::bytes_type
    Endpoint::get_address_bytes(Internet_protocol::Ipv6_tag /* tag */) const
{
    return call_on_base_address(
        [](auto const& address) { return internal::get_ipv6_bytes(address); });
}


template<typename T>
T const& get_endpoint_base(Endpoint const&);

template<>
inline Endpoint_udp const& get_endpoint_base<Endpoint_udp>(Endpoint const& endpoint)
{
    return endpoint.get_udp_base();
}
template<>
inline Endpoint_tcp const& get_endpoint_base<Endpoint_tcp>(Endpoint const& endpoint)
{
    return endpoint.get_tcp_base();
}


}
}

#endif 