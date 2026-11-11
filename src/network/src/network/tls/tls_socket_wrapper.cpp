#include "network/tls/tls_socket_wrapper.hpp"

namespace nexusminer {
namespace network {
namespace tls {

TlsSocketWrapper::TlsSocketWrapper(asio::io_context& io_context, TlsContext& tls_context)
    : m_stream(io_context, tls_context.get_context())
{
}

TlsSocketWrapper::TlsSocketWrapper(asio::ip::tcp::socket socket, TlsContext& tls_context)
    : m_stream(std::move(socket), tls_context.get_context())
{
}

} // namespace tls
} // namespace network
} // namespace nexusminer
