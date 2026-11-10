#ifndef NEXUSMINER_NETWORK_TCP_CONNECTION_IMPL_HPP
#define NEXUSMINER_NETWORK_TCP_CONNECTION_IMPL_HPP

#include "asio/io_service.hpp"
#include "asio/write.hpp"
#include "network/connection.hpp"
#include "network/tcp/protocol_description.hpp"
#include "LLP/llp_logging.hpp"
#include "protocol_lane.hpp"
#include <spdlog/spdlog.h>
#include <queue>
#include <memory>

namespace nexusminer {
namespace network {
namespace tcp {

template<typename ProtocolDescriptionType>
class Connection_impl
    : public Connection
    , public std::enable_shared_from_this<Connection_impl<ProtocolDescriptionType>>
{
    // protocol specific types
    using Protocol_description = ProtocolDescriptionType;
    using Protocol_socket = typename Protocol_description::Socket;
    using Protocol_endpoint = typename Protocol_description::Endpoint;

public:
    Connection_impl(std::shared_ptr<::asio::io_context> io_context,
                    Endpoint remote_endpoint, Endpoint local_endpoint, Connection::Handler handler);
    Connection_impl(std::shared_ptr<::asio::io_context> io_context,
                    std::shared_ptr<Protocol_socket> asio_socket, Endpoint remote_endpoint);

    // no copies
    Connection_impl(const Connection_impl&) = delete;
    Connection_impl& operator=(const Connection_impl&) = delete;
    // no moves
    Connection_impl(Connection_impl&&) = delete;
    Connection_impl& operator=(Connection_impl&&) = delete;

    // Connection interface
    Endpoint const& remote_endpoint() const override { return m_remote_endpoint; }
    Endpoint const& local_endpoint() const override { return m_local_endpoint; }
    bool transmit(Shared_payload tx_buffer) override;
    void close() override;
    ProtocolLane get_protocol_lane() const override { return m_protocol_lane; }

    // interface towards socket
    Result::Code connect();
    void handle_accept(Connection::Handler connection_handler);

private:
    enum class EmptyReadAction { Closed, Rearm, Abort, DataAppeared };

    std::weak_ptr<Connection_impl<ProtocolDescriptionType>> get_weak_self();
    Result::Code initialise_socket();
    void transmit_trigger();
    void receive();
    EmptyReadAction classify_empty_read();
    void change(Result::Code code);
    void close_internal(Result::Code code);

    // Maximum TX queue depth. During burst blocks rapid transmit() calls can
    // outpace async_write completions; when full, drop the newest payload.
    static constexpr std::size_t MAX_TX_QUEUE_SIZE = 64;

    std::shared_ptr<::asio::io_context> m_io_context;
    std::shared_ptr<Protocol_socket> m_asio_socket;
    Endpoint m_remote_endpoint;
    Endpoint m_local_endpoint;
    std::queue<Shared_payload> m_tx_queue;
    Connection::Handler m_connection_handler;
    std::shared_ptr<spdlog::logger> m_logger;
    ProtocolLane m_protocol_lane;
};


template<typename ProtocolDescriptionType>
inline Connection_impl<ProtocolDescriptionType>::Connection_impl(
    std::shared_ptr<::asio::io_context> io_context, Endpoint remote_endpoint,
    Endpoint local_endpoint, Connection::Handler handler)
    : m_io_context{std::move(io_context)}
    , m_asio_socket{std::make_shared<Protocol_socket>(*m_io_context)}
    , m_remote_endpoint{std::move(remote_endpoint)}
    , m_local_endpoint{std::move(local_endpoint)}
    , m_tx_queue{}
    , m_connection_handler{std::move(handler)}
    , m_logger{spdlog::get("logger")}
    , m_protocol_lane{determine_lane_from_port(m_remote_endpoint.port())}
{
}

template<typename ProtocolDescriptionType>
inline Connection_impl<ProtocolDescriptionType>::Connection_impl(
    std::shared_ptr<::asio::io_context> io_context,
    std::shared_ptr<Protocol_socket> asio_socket, Endpoint remote_endpoint)
    : m_io_context{std::move(io_context)}
    , m_asio_socket{std::move(asio_socket)}
    , m_remote_endpoint{std::move(remote_endpoint)}
    , m_local_endpoint{}     // will be set later, this constructor is called in accept/listen case
    , m_tx_queue{}
	, m_connection_handler{} // will be set later, this constructor is called in accept/listen case
    , m_logger{spdlog::get("logger")}
    , m_protocol_lane{determine_lane_from_port(m_remote_endpoint.port())}
{
}

template<typename ProtocolDescriptionType>
inline std::weak_ptr<Connection_impl<ProtocolDescriptionType>>
Connection_impl<ProtocolDescriptionType>::get_weak_self()
{
    return this->shared_from_this();
}

template<typename ProtocolDescriptionType>
inline Result::Code
Connection_impl<ProtocolDescriptionType>::initialise_socket()
{
    asio::error_code error;
    this->m_asio_socket->open(get_endpoint_base<Protocol_endpoint>(m_local_endpoint).protocol(), error);
    if (error) 
	{
        return Result::error;
    }

    // Disable Nagle's algorithm — mining is latency-sensitive and LLP
    // packets are small.  Nagle can add up to 40ms per small packet.
    this->m_asio_socket->set_option(::asio::ip::tcp::no_delay(true), error);
    if (error && m_logger)
    {
        m_logger->warn("[LLP] Failed to set TCP_NODELAY: {}", error.message());
    }

    this->m_asio_socket->bind(get_endpoint_base<Protocol_endpoint>(m_local_endpoint), error);
    if (error)
	{
        m_asio_socket->close(error);
        return Result::error;
    }

    Protocol_description::update_port(m_asio_socket->local_endpoint(), m_local_endpoint);

    return Result::ok;
}


template<typename ProtocolDescriptionType>
inline Result::Code Connection_impl<ProtocolDescriptionType>::connect()
{
    if (initialise_socket() != Result::ok)
	{
        return Result::error;
    }

    std::weak_ptr<Connection_impl<ProtocolDescriptionType>> weak_self = this->shared_from_this();
    this->m_asio_socket->async_connect(get_endpoint_base<Protocol_endpoint>(m_remote_endpoint),
                                       [weak_self](::asio::error_code const& error)
	{
		auto self = weak_self.lock();
		if (self && self->m_connection_handler)
		{
            if (!error) 
            {
                self->change(Result::Code::connection_ok);
            }
            else 
            {
                self->change(Result::Code::connection_declined);
            }
		}
	});

    return Result::ok;
}

template<typename ProtocolDescriptionType>
inline void Connection_impl<ProtocolDescriptionType>::receive()
{
    m_asio_socket->async_receive(asio::null_buffers(), [weak_self = get_weak_self()](auto error, auto) 
	{        
        auto self = weak_self.lock();
        if (self && self->m_connection_handler) 
		{
            if (!error) 
            {
                // read length of received message;
                ::asio::error_code available_error;
                auto length = self->m_asio_socket->available(available_error);
                if (available_error)
                {
                    if (self->m_logger)
                    {
                        self->m_logger->error("[LLP RECV] available() failed: {}", available_error.message());
                    }
                    self->change(Result::Code::connection_aborted);
                    return;
                }
                if (length == 0)
                {
                    // null_buffers readiness with available()==0 is not proof of
                    // EOF. A spurious wakeup must re-arm; only a non-blocking
                    // peek that returns 0/eof/reset is a real interrupt.
                    switch (self->classify_empty_read())
                    {
                    case EmptyReadAction::Rearm:
                        self->receive();
                        return;
                    case EmptyReadAction::DataAppeared:
                        length = self->m_asio_socket->available(available_error);
                        if (available_error || length == 0)
                        {
                            self->receive();
                            return;
                        }
                        break;
                    case EmptyReadAction::Closed:
                        if (self->m_logger)
                        {
                            self->m_logger->warn("[LLP RECV] Connection closed by remote (EOF, no data available)");
                        }
                        self->change(Result::Code::connection_closed);
                        return;
                    case EmptyReadAction::Abort:
                        self->change(Result::Code::connection_aborted);
                        return;
                    }
                }

                Shared_payload receive_buffer = std::make_shared<std::vector<std::uint8_t>>(length);
                const auto bytes_read = self->m_asio_socket->receive(
                    asio::buffer(*receive_buffer, receive_buffer->size()), 0, error);
                if (!error && bytes_read > 0)
                {
                    // available() can over-report across a connection interrupt.
                    // Passing the unread tail would inject zero bytes into the
                    // RX accumulator and look like a malformed frame.
                    receive_buffer->resize(bytes_read);
                    // Log received LLP packets using lane-aware parsing
                    if (self->m_logger && receive_buffer && !receive_buffer->empty())
                    {
                        // Parse packet(s) from buffer using lane-aware framing
                        std::size_t offset = 0;
                        
                        while (offset < receive_buffer->size())
                        {
                            bool is_stateless = (self->m_protocol_lane == ProtocolLane::STATELESS);
                            std::size_t header_size = is_stateless ? 2 : 1;
                            
                            // Check if we have enough bytes for header
                            if (offset + header_size > receive_buffer->size()) break;
                            
                            // Parse header based on lane
                            uint16_t header = 0;
                            if (is_stateless) {
                                // 16-bit header, big-endian
                                header = (static_cast<uint16_t>((*receive_buffer)[offset]) << 8) |
                                        static_cast<uint16_t>((*receive_buffer)[offset + 1]);
                            } else {
                                // 8-bit header
                                header = (*receive_buffer)[offset];
                            }
                            
                            // Check if we have length field
                            std::uint32_t pkt_length = 0;
                            std::size_t length_offset = offset + header_size;
                            
                            if (length_offset + 4 <= receive_buffer->size())
                            {
                                // Parse length (4 bytes, big-endian)
                                pkt_length = ((*receive_buffer)[length_offset] << 24) + 
                                           ((*receive_buffer)[length_offset + 1] << 16) + 
                                           ((*receive_buffer)[length_offset + 2] << 8) + 
                                           (*receive_buffer)[length_offset + 3];

                                // Bounds-check: reject obviously oversized packets
                                // (max reasonable LLP payload is ~4 MB for block data)
                                static constexpr std::uint32_t MAX_LLP_PACKET_SIZE = 4u * 1024u * 1024u;
                                if (pkt_length > MAX_LLP_PACKET_SIZE)
                                {
                                    self->m_logger->warn("[LLP RECV] Oversized packet length {} (max {}), skipping parse",
                                        pkt_length, MAX_LLP_PACKET_SIZE);
                                    break;
                                }
                            }
                            
                            // Create data payload for hex preview
                            network::Shared_payload data_payload;
                            std::size_t data_offset = length_offset + 4;
                            if (pkt_length > 0 && data_offset < receive_buffer->size())
                            {
                                std::size_t data_end = std::min(data_offset + pkt_length, receive_buffer->size());
                                if (data_end > data_offset) {
                                    data_payload = std::make_shared<network::Payload>(
                                        receive_buffer->begin() + data_offset, 
                                        receive_buffer->begin() + data_end);
                                }
                            }
                            
                            // Log with appropriate format (debug level — payload hex
                            // can be noisy and may expose sensitive data at info).
                            std::string hex_preview = format_llp_payload_hex(data_payload, 16);
                            if (is_stateless) {
                                if (!hex_preview.empty()) {
                                    self->m_logger->debug("[LLP RECV] header=0x{:04x} {} length={} payload=[{}]", 
                                        header, get_llp_header_name(header), pkt_length, hex_preview);
                                } else {
                                    self->m_logger->debug("[LLP RECV] header=0x{:04x} {} length={}", 
                                        header, get_llp_header_name(header), pkt_length);
                                }
                            } else {
                                if (!hex_preview.empty()) {
                                    self->m_logger->debug("[LLP RECV] header=0x{:02x} {} length={} payload=[{}]", 
                                        static_cast<uint8_t>(header), get_llp_header_name(static_cast<uint8_t>(header)), 
                                        pkt_length, hex_preview);
                                } else {
                                    self->m_logger->debug("[LLP RECV] header=0x{:02x} {} length={}", 
                                        static_cast<uint8_t>(header), get_llp_header_name(static_cast<uint8_t>(header)), 
                                        pkt_length);
                                }
                            }
                            
                            // Move to next packet
                            if (pkt_length > 0 && data_offset + pkt_length <= receive_buffer->size())
                            {
                                offset = data_offset + pkt_length;
                            }
                            else if (pkt_length == 0 && length_offset + 4 <= receive_buffer->size())
                            {
                                // Header-only packet with length field = 0
                                offset = length_offset + 4;
                            }
                            else if (length_offset > receive_buffer->size())
                            {
                                // Header-only packet without length field
                                offset += header_size;
                            }
                            else
                            {
                                // Incomplete packet, stop logging
                                break;
                            }
                        }
                    }
                    
                    try
                    {
                        self->m_connection_handler(Result::receive_ok, std::move(receive_buffer));
                    }
                    catch (const std::exception& ex)
                    {
                        if (self->m_logger)
                        {
                            self->m_logger->error("[LLP RECV] Exception in connection handler: {} — closing connection", ex.what());
                        }
                        self->close_internal(Result::Code::connection_closed);
                        return;
                    }
                    catch (...)
                    {
                        if (self->m_logger)
                        {
                            self->m_logger->error("[LLP RECV] Unknown exception in connection handler — closing connection");
                        }
                        self->close_internal(Result::Code::connection_closed);
                        return;
                    }
                    self->receive();
                }
                else if (!error || error == ::asio::error::eof || error == ::asio::error::connection_reset)
                {
                    if (self->m_logger)
                    {
                        self->m_logger->warn("[LLP RECV] Connection closed by remote during read: {}",
                                             error ? error.message() : "short read returned 0");
                    }
                    self->change(Result::Code::connection_closed);
                }
                else if (error == ::asio::error::would_block || error == ::asio::error::try_again)
                {
                    self->receive();
                }
                else
                {
                    // established connection fails for any other reason
                    if (self->m_logger)
                    {
                        self->m_logger->error("[LLP RECV] Socket receive error: {}", error.message());
                    }
                    self->change(Result::Code::connection_aborted);
                }
            }
            else if ((error == ::asio::error::eof) || (error == ::asio::error::connection_reset))
            {
                // established connection closed by remote
                if (self->m_logger)
                {
                    self->m_logger->warn("[LLP RECV] Connection closed by remote: {}", error.message());
                }
                self->change(Result::Code::connection_closed);
            }
            else if (error == ::asio::error::operation_aborted)
            {
                // Deliberate socket close (e.g. shutdown); not a real error
                if (self->m_logger)
                {
                    self->m_logger->debug("[LLP RECV] Receive cancelled (operation_aborted)");
                }
                self->change(Result::Code::connection_aborted);
            }
            else
            {
                // established connection fails for any other reason
                if (self->m_logger)
                {
                    self->m_logger->error("[LLP RECV] Connection error: {}", error.message());
                }
                self->change(Result::Code::connection_aborted);
            }
        }
    });
}

template<typename ProtocolDescriptionType>
inline typename Connection_impl<ProtocolDescriptionType>::EmptyReadAction
Connection_impl<ProtocolDescriptionType>::classify_empty_read()
{
    if (!m_asio_socket)
    {
        return EmptyReadAction::Abort;
    }

    ::asio::error_code mode_error;
    const bool was_non_blocking = m_asio_socket->non_blocking();
    m_asio_socket->non_blocking(true, mode_error);
    if (mode_error)
    {
        return EmptyReadAction::Abort;
    }

    std::uint8_t peek_byte = 0;
    ::asio::error_code peek_error;
    const auto peeked = m_asio_socket->receive(
        ::asio::buffer(&peek_byte, 1),
        ::asio::socket_base::message_peek,
        peek_error);

    ::asio::error_code restore_error;
    m_asio_socket->non_blocking(was_non_blocking, restore_error);

    if (!peek_error && peeked > 0)
    {
        return EmptyReadAction::DataAppeared;
    }
    if (peek_error == ::asio::error::would_block || peek_error == ::asio::error::try_again)
    {
        return EmptyReadAction::Rearm;
    }
    if (!peek_error || peeked == 0 ||
        peek_error == ::asio::error::eof ||
        peek_error == ::asio::error::connection_reset)
    {
        return EmptyReadAction::Closed;
    }
    return EmptyReadAction::Abort;
}

template<typename ProtocolDescriptionType>
inline void Connection_impl<ProtocolDescriptionType>::change(Result::Code code)
{
    if (code == Result::Code::connection_ok) 
    {
        try
        {
            m_connection_handler(code, Shared_payload{});
        }
        catch (const std::exception& ex)
        {
            if (m_logger)
            {
                m_logger->error("[LLP] Exception in connection handler (change): {} — closing", ex.what());
            }
            close_internal(Result::Code::connection_closed);
            return;
        }
        catch (...)
        {
            if (m_logger)
            {
                m_logger->error("[LLP] Unknown exception in connection handler (change) — closing");
            }
            close_internal(Result::Code::connection_closed);
            return;
        }
        receive();
    }
    else 
    {
        close_internal(code);
    }
}

template<typename ProtocolDescriptionType>
inline void Connection_impl<ProtocolDescriptionType>::handle_accept(Connection::Handler connection_handler)
{
    assert(connection_handler);
    m_connection_handler = std::move(connection_handler);
    m_local_endpoint = Endpoint(m_asio_socket->local_endpoint());
    change(Result::Code::connection_ok);
}


template<typename ProtocolDescriptionType>
bool Connection_impl<ProtocolDescriptionType>::transmit(Shared_payload tx_buffer)
{
    // Early-return if connection handler is null (connection already closed/uninitialised)
    if (!m_connection_handler) 
    {
        if (m_logger)
        {
            m_logger->warn("[LLP SEND] Cannot transmit - connection handler is null (connection closed/uninitialised)");
        }
        return false;
    }
    
    // Early-return if socket is null
    if (!m_asio_socket)
    {
        if (m_logger)
        {
            m_logger->error("[LLP SEND] Cannot transmit - socket is null");
        }
        return false;
    }
    
    // Early-return if buffer is null or empty (no header/payload to send)
    if (!tx_buffer || tx_buffer->empty())
    {
        if (m_logger)
        {
            m_logger->error("[LLP SEND] Cannot transmit - payload is null or empty");
        }
        return false;
    }
    
    // TX queue overflow protection: during burst blocks rapid transmit() calls
    // can outpace async_write completions.  We cannot safely pop the front of
    // the queue because it is the in-flight payload whose async_write completion
    // handler will pop it; removing it here would cause the handler to pop the
    // NEXT payload instead, silently losing an unsent message.  Drop the newest
    // payload (this one) instead.
    if (m_tx_queue.size() >= MAX_TX_QUEUE_SIZE)
    {
        if (m_logger)
        {
            // Log the dropped payload size for shadow-ban diagnostics: if
            // recovery logic floods the queue, GET_ROUND/SUBMIT_BLOCK packets
            // are silently lost (PUSH is unaffected — uses node's TX queue).
            m_logger->error("[LLP SEND] TX QUEUE OVERFLOW ({}/{}) — DROPPING outgoing payload "
                "({} bytes). Recovery flooding may cause silent shadow ban!",
                m_tx_queue.size(), MAX_TX_QUEUE_SIZE,
                tx_buffer ? tx_buffer->size() : 0);
        }
        return false;
    }

    // Enqueue the payload and trigger transmission if queue was previously empty
    m_tx_queue.emplace(tx_buffer);

    if (m_tx_queue.size() == 1) 
    {
        transmit_trigger();
    }

    return true;
}

template<typename ProtocolDescriptionType>
void Connection_impl<ProtocolDescriptionType>::transmit_trigger()
{
    // Check socket is non-null
    if (!m_asio_socket)
    {
        if (m_logger)
        {
            m_logger->error("[LLP SEND] transmit_trigger: socket is null");
        }
        // Drop the front of queue if any and return
        if (!m_tx_queue.empty())
        {
            m_tx_queue.pop();
        }
        return;
    }
    
    // Check queue is non-empty
    if (m_tx_queue.empty())
    {
        if (m_logger)
        {
            m_logger->warn("[LLP SEND] transmit_trigger: queue is empty");
        }
        return;
    }
    
    auto const payload = m_tx_queue.front();
    
    // Check payload is non-null and non-empty
    if (!payload || payload->empty())
    {
        if (m_logger)
        {
            m_logger->error("[LLP SEND] transmit_trigger: payload is null or empty, dropping from queue");
        }
        m_tx_queue.pop();
        // Recursively call if more queued payloads
        if (!m_tx_queue.empty())
        {
            transmit_trigger();
        }
        return;
    }
    
    // Log LLP packet send using lane-aware parsing
    if (m_logger)
    {
        bool is_stateless = (m_protocol_lane == ProtocolLane::STATELESS);
        std::size_t header_size = is_stateless ? 2 : 1;
        
        // Check if we have enough bytes for header
        if (payload->size() >= header_size)
        {
            // Parse header based on lane
            uint16_t header = 0;
            if (is_stateless) {
                // 16-bit header, big-endian
                header = (static_cast<uint16_t>((*payload)[0]) << 8) |
                        static_cast<uint16_t>((*payload)[1]);
            } else {
                // 8-bit header
                header = (*payload)[0];
            }
            
            // Check if we have length field
            std::uint32_t length = 0;
            std::size_t length_offset = header_size;
            
            if (payload->size() == header_size)
            {
                // Header-only packet
                if (is_stateless) {
                    m_logger->debug("[LLP SEND] header=0x{:04x} {} length=0 (header-only)", 
                        header, get_llp_header_name(header));
                } else {
                    m_logger->debug("[LLP SEND] header=0x{:02x} {} length=0 (header-only)", 
                        static_cast<uint8_t>(header), get_llp_header_name(static_cast<uint8_t>(header)));
                }
            }
            else if (payload->size() >= header_size + 4)
            {
                // Parse length (4 bytes, big-endian)
                length = (static_cast<std::uint32_t>((*payload)[length_offset]) << 24) |
                         (static_cast<std::uint32_t>((*payload)[length_offset + 1]) << 16) |
                         (static_cast<std::uint32_t>((*payload)[length_offset + 2]) << 8) |
                         static_cast<std::uint32_t>((*payload)[length_offset + 3]);
                
                // Create a shared pointer to the data portion for hex formatting
                network::Shared_payload data_payload;
                std::size_t data_offset = length_offset + 4;
                if (payload->size() > data_offset)
                {
                    std::size_t data_end = std::min(data_offset + length, payload->size());
                    if (data_end > data_offset)
                    {
                        data_payload = std::make_shared<network::Payload>(
                            payload->begin() + data_offset, 
                            payload->begin() + data_end);
                    }
                }
                
                std::string hex_preview = format_llp_payload_hex(data_payload, 16);
                if (is_stateless) {
                    if (!hex_preview.empty()) {
                        m_logger->debug("[LLP SEND] header=0x{:04x} {} length={} payload=[{}]", 
                            header, get_llp_header_name(header), length, hex_preview);
                    } else {
                        m_logger->debug("[LLP SEND] header=0x{:04x} {} length={}", 
                            header, get_llp_header_name(header), length);
                    }
                } else {
                    if (!hex_preview.empty()) {
                        m_logger->debug("[LLP SEND] header=0x{:02x} {} length={} payload=[{}]", 
                            static_cast<uint8_t>(header), get_llp_header_name(static_cast<uint8_t>(header)), 
                            length, hex_preview);
                    } else {
                        m_logger->debug("[LLP SEND] header=0x{:02x} {} length={}", 
                            static_cast<uint8_t>(header), get_llp_header_name(static_cast<uint8_t>(header)), 
                            length);
                    }
                }
            }
            else
            {
                // Malformed packet
                m_logger->error("[LLP SEND] Malformed LLP payload: size={} (expected {} for header-only or >={} for data packet)", 
                    payload->size(), header_size, header_size + 4);
            }
        }
    }
    
    ::asio::async_write(*m_asio_socket, ::asio::buffer(*payload, payload->size()),
        // don't forget to keep the payload until transmission has been completed!!!
        [weak_self = get_weak_self(), payload](const ::asio::error_code& error, std::size_t /*bytes_transferred*/) 
        {
            auto self = weak_self.lock();
            if (!self)
                return;

            // ── CRITICAL: check async_write error ──────────────────────────
            if (error)
            {
                if (error != ::asio::error::operation_aborted)
                {
                    if (self->m_logger)
                    {
                        self->m_logger->error("[LLP SEND] async_write failed: {} — draining TX queue ({} pending) and closing connection",
                            error.message(), self->m_tx_queue.size());
                    }
                }
                // Drain the entire queue — all pending payloads are stale once
                // the underlying socket has failed.
                while (!self->m_tx_queue.empty())
                    self->m_tx_queue.pop();

                self->close_internal(Result::Code::connection_aborted);
                return;
            }

            if (!self->m_connection_handler)
            {
                return;
            }

            // Safely pop from queue
            if (!self->m_tx_queue.empty())
            {
                self->m_tx_queue.pop();
            }
            
            // Tail-recurse if more queued payloads
            if (!self->m_tx_queue.empty()) 
            {
                self->transmit_trigger();
            }
        });
}

template<typename ProtocolDescriptionType>
inline void Connection_impl<ProtocolDescriptionType>::close()
{
    close_internal(Result::Code::connection_closed);
}

template<typename ProtocolDescriptionType>
inline void Connection_impl<ProtocolDescriptionType>::close_internal(Result::Code code)
{
    if (m_connection_handler) 
    {
        if (m_asio_socket->is_open()) 
        {
            ::asio::error_code error;
            (void)m_asio_socket->shutdown(::asio::socket_base::shutdown_both, error);
            (void)m_asio_socket->close(error);
        }

        auto const connection_handler = std::move(m_connection_handler);
        m_connection_handler = nullptr;
        connection_handler(code, Shared_payload{});
    }
}

}
}
}

#endif
