#ifndef NEXUSMINER_PROTOCOL_PACKET_ROUTER_HPP
#define NEXUSMINER_PROTOCOL_PACKET_ROUTER_HPP

#include "packet.hpp"
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace nexusminer {
namespace network { class Connection; }
namespace protocol {

/// Table-driven packet dispatcher.
///
/// Replaces the 320-line if/else-if chain in Solo::process_messages() with a
/// compact lookup table keyed on canonical (legacy 8-bit) opcode.  Each
/// handler is registered once at construction time and is called with the
/// original Packet and the Connection that delivered it.
///
/// Design notes:
///   * Opcode matching follows the same rules as Solo::matches_opcode():
///     if the packet carries a uint16_t opcode the router translates it to
///     its legacy mirror (via LLP::UnmirrorOpcode) before table lookup.
///   * A handler may be registered for multiple opcodes (e.g. ACCEPT +
///     GOOD_BLOCK both map to the "block accepted" handler).
///   * The router does NOT own pre-dispatch guards (lane validation, packet
///     validity, session preflight) — those remain in Solo::process_messages()
///     which calls router.dispatch() after the guards pass.
class PacketRouter {
public:
    /// Packet handler signature — same as Solo's on_* methods.
    using Handler = std::function<void(Packet const&, std::shared_ptr<network::Connection>)>;

    /// Register a handler for a legacy 8-bit opcode.
    /// If the same opcode is registered twice the later registration wins.
    void register_handler(uint16_t legacy_opcode, Handler handler);

    /// Register a handler for a raw uint16_t opcode that has no legacy mirror
    /// (e.g. SESSION_STATUS_ACK 0xD0DC).  The handler is keyed on the exact
    /// uint16_t value in the Packet::m_header field.
    void register_raw_handler(uint16_t raw_opcode, Handler handler);

    /// Attempt to dispatch the packet to a registered handler.
    /// Returns true if a handler was found and invoked, false otherwise.
    bool dispatch(Packet const& packet, const std::shared_ptr<network::Connection>& connection) const;

    /// Returns the number of registered handlers (for diagnostics / tests).
    std::size_t handler_count() const;

private:
    /// Handlers keyed on canonical (legacy 8-bit) opcode.
    std::unordered_map<uint16_t, Handler> m_handlers;

    /// Handlers keyed on raw uint16_t opcode (for unmirror-able opcodes).
    std::unordered_map<uint16_t, Handler> m_raw_handlers;
};

}  // namespace protocol
}  // namespace nexusminer

#endif  // NEXUSMINER_PROTOCOL_PACKET_ROUTER_HPP
