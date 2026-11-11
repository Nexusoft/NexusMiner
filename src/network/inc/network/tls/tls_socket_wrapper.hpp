#ifndef NEXUSMINER_NETWORK_TLS_SOCKET_WRAPPER_HPP
#define NEXUSMINER_NETWORK_TLS_SOCKET_WRAPPER_HPP

#include "asio/ssl.hpp"
#include "asio/ip/tcp.hpp"
#include "network/tls/tls_context.hpp"
#include <memory>

namespace nexusminer {
namespace network {
namespace tls {

/**
 * @brief TLS Socket Wrapper
 * 
 * Wraps an ASIO SSL stream to provide TLS/SSL encrypted TCP connections.
 * This wrapper is compatible with the existing TCP socket interface.
 */
class TlsSocketWrapper {
public:
    
    using StreamType = asio::ssl::stream<asio::ip::tcp::socket>;
    
    /**
     * @brief Constructor for client connections
     * @param io_context ASIO io_context
     * @param tls_context TLS context for SSL settings
     */
    TlsSocketWrapper(asio::io_context& io_context, TlsContext& tls_context);
    
    /**
     * @brief Constructor from existing socket (for server accept)
     * @param socket Existing TCP socket
     * @param tls_context TLS context for SSL settings
     */
    TlsSocketWrapper(asio::ip::tcp::socket socket, TlsContext& tls_context);
    
    /**
     * @brief Get the underlying SSL stream
     * @return Reference to SSL stream
     */
    StreamType& get_stream() { return m_stream; }
    const StreamType& get_stream() const { return m_stream; }
    
    /**
     * @brief Get the underlying TCP socket (lowest layer)
     * @return Reference to lowest layer socket
     */
    auto& get_socket() { return m_stream.lowest_layer(); }
    const auto& get_socket() const { return m_stream.lowest_layer(); }
    
    /**
     * @brief Perform TLS handshake (client mode)
     * @param handler Callback for handshake completion
     */
    template<typename HandshakeHandler>
    void async_handshake_client(HandshakeHandler&& handler) {
        m_stream.async_handshake(asio::ssl::stream_base::client, 
                                std::forward<HandshakeHandler>(handler));
    }
    
    /**
     * @brief Perform TLS handshake (server mode)
     * @param handler Callback for handshake completion
     */
    template<typename HandshakeHandler>
    void async_handshake_server(HandshakeHandler&& handler) {
        m_stream.async_handshake(asio::ssl::stream_base::server,
                                std::forward<HandshakeHandler>(handler));
    }
    
    /**
     * @brief Asynchronous read operation
     */
    template<typename MutableBufferSequence, typename ReadHandler>
    void async_read_some(const MutableBufferSequence& buffers, ReadHandler&& handler) {
        m_stream.async_read_some(buffers, std::forward<ReadHandler>(handler));
    }
    
    /**
     * @brief Asynchronous write operation
     */
    template<typename ConstBufferSequence, typename WriteHandler>
    void async_write_some(const ConstBufferSequence& buffers, WriteHandler&& handler) {
        m_stream.async_write_some(buffers, std::forward<WriteHandler>(handler));
    }
    
    /**
     * @brief Shutdown TLS connection
     */
    void shutdown() {
        asio::error_code ec;
        m_stream.shutdown(ec);
        // Ignore errors during shutdown (common in TLS)
    }
    
    /**
     * @brief Close the socket
     */
    void close() {
        shutdown();
        asio::error_code ec;
        get_socket().close(ec);
    }

private:
    
    StreamType m_stream;
};

} // namespace tls
} // namespace network
} // namespace nexusminer

#endif // NEXUSMINER_NETWORK_TLS_SOCKET_WRAPPER_HPP
