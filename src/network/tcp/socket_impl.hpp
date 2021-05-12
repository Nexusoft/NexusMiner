#ifndef NEXUSMINER_NETWORK_TCP_SOCKET_IMPL_HPP
#define NEXUSMINER_NETWORK_TCP_SOCKET_IMPL_HPP

#include "asio/io_service.hpp"
#include "../socket.hpp"
#include "connection_impl.hpp"

#include <memory>

namespace nexusminer {
namespace network {
namespace tcp {

template<typename ProtocolDescriptionType>
class Socket_impl : public Socket, public std::enable_shared_from_this<Socket_impl<ProtocolDescriptionType>> 
{
public:
    Socket_impl(std::shared_ptr<asio::io_context> io_context, Endpoint local_endpoint);

    Connection::Sptr connect(Endpoint destination, Connection::Handler handler) override;
    Result::Code listen(Connect_handler handler) override;
    void stop_listen() override;
    Endpoint const& local_endpoint() const override { return m_local_endpoint; }

protected:
    std::shared_ptr<::asio::io_context> m_io_context;
    Endpoint m_local_endpoint;
    typename ProtocolDescriptionType::Acceptor m_acceptor;

    void accept(Connect_handler handler);
};

template<typename ProtocolDescriptionType>
inline Socket_impl<ProtocolDescriptionType>::Socket_impl(
    std::shared_ptr<asio::io_context> io_context, Endpoint local_endpoint)
    : m_io_context{std::move(io_context)}
    , m_local_endpoint{std::move(local_endpoint)}
    , m_acceptor{*m_io_context }
{
}


template<typename ProtocolDescriptionType>
inline Connection::Sptr Socket_impl<ProtocolDescriptionType>::connect(Endpoint destination, Connection::Handler handler)
{
    assert(handler);
    assert(destination.is_valid());

	auto connection =
		std::make_shared<Connection_impl<ProtocolDescriptionType>>(
			m_io_context, std::move(destination), m_local_endpoint, std::move(handler));

	if (connection->connect() == Result::ok) 
	{
		ProtocolDescriptionType::update_port(get_endpoint_base<typename ProtocolDescriptionType::Endpoint>(
				connection->local_endpoint()), m_local_endpoint);

		return Connection::Sptr{connection};
    }

    return Connection::Sptr{nullptr};
}

template<typename ProtocolDescriptionType>
inline void Socket_impl<ProtocolDescriptionType>::stop_listen()
{
	ProtocolDescriptionType::close_acceptor(m_acceptor);
}

template<typename ProtocolDescriptionType>
inline void Socket_impl<ProtocolDescriptionType>::accept(Connect_handler handler)
{
    auto new_connection_socket = std::make_shared<typename ProtocolDescriptionType::Socket>(*m_io_context);
    std::weak_ptr<Socket_impl> weak_self = this->shared_from_this();
    m_acceptor.async_accept(*new_connection_socket, [weak_self, handler = std::move(handler),
                                                     new_connection_socket](::asio::error_code const& error) mutable 
	{
        auto self = weak_self.lock();
        if (self && !error) 
		{
            Endpoint remote_ep = Endpoint(std::move(new_connection_socket->remote_endpoint()));
    
			auto connection = std::make_shared<Connection_impl<ProtocolDescriptionType>>(
				self->m_io_context, std::move(new_connection_socket), std::move(remote_ep));

			Connection::Handler connection_handler = handler(connection);
			connection->handle_accept(std::move(connection_handler));

			self->accept(handler);
        }       
    } );
}

template<typename ProtocolDescriptionType>
inline Result::Code Socket_impl<ProtocolDescriptionType>::listen(Connect_handler handler)
{
    assert(handler);

    if (m_acceptor.is_open()) 
	{
        return Result::socket_error;
    }

    asio::error_code error;

    m_acceptor.open(get_endpoint_base<typename ProtocolDescriptionType::Endpoint>(m_local_endpoint).protocol(), error);
    if (error)
	{
        return Result::socket_error;
    }

    if (ProtocolDescriptionType::bind_acceptor(m_acceptor, m_local_endpoint) != Result::ok)
	{
        ProtocolDescriptionType::close_acceptor(m_acceptor);
        return Result::socket_error;
    }

    ProtocolDescriptionType::update_port(m_acceptor.local_endpoint(), m_local_endpoint);

    m_acceptor.listen(asio::socket_base::max_listen_connections, error);
    if (error)
	{
        ProtocolDescriptionType::close_acceptor(m_acceptor);
        return Result::socket_error;
    }

    accept(std::move(handler));
    return Result::socket_ok;
}

}
} 
}

#endif