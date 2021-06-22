#ifndef NEXUSMINER_NETWORK_TYPES_HPP
#define NEXUSMINER_NETWORK_TYPES_HPP

// include winsock2 header to prevent issues with wrong order of inclusion in ASIO headers
#if _WIN32_WINNT
#include <winsock2.h>
#endif

#include <memory>
#include <vector>

namespace nexusminer {
namespace network {

namespace Result {

static constexpr unsigned int category_mask = 0xF0U;
static constexpr unsigned int code_mask = 0x0FU;

enum Category { general = 0x00, socket = 0x10, connection = 0x20, receive = 0x30, transmit = 0x40 };

enum Code {
    ok = Category::general,
    error,

    socket_ok = Category::socket,
    socket_error,

    connection_ok = Category::connection,
    connection_error,
    connection_closed,
    connection_aborted,
    connection_declined,

    receive_ok = Category::receive,

    transmit_ok = Category::transmit,
};

inline Category category(Code code)
{
    auto const value = static_cast<unsigned int>(code);
    auto const category_value = (value & category_mask);
    return static_cast<Category>(category_value);
}

inline bool is_ok(Code code)
{
    auto const value = static_cast<unsigned int>(code);
    auto const code_value = (value & code_mask);
    return (code_value == 0U);
}

inline bool is_error(Code code)
{
    return !is_ok(code);
}

} // namespace Result

using Payload = std::vector<std::uint8_t>;
using Shared_payload = std::shared_ptr<Payload>;

enum class Transport_protocol { tcp = 0, udp = 1, none = 3 };

struct Internet_protocol {
    struct Ipv4_tag {
    };

    struct Ipv6_tag {
    };
    static constexpr Ipv4_tag ipv4 = Ipv4_tag{};
    static constexpr Ipv6_tag ipv6 = Ipv6_tag{};
};

} // namespace network
} // namespace nexusminer

#endif // NEXUSMINER_NETWORK_TYPES_HPP