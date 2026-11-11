#ifndef NEXUSMINER_PROTOCOL_SESSION_INGRESS_GATE_HPP
#define NEXUSMINER_PROTOCOL_SESSION_INGRESS_GATE_HPP

#include "protocol/packet_ingress_preflight.hpp"

namespace nexusminer {
namespace protocol {

using SessionIngressDecision = PacketIngressDecision;

class SessionIngressGate
{
public:
    using Input = PacketIngressPreflight::Input;

    static SessionIngressDecision preflight(const Input& input)
    {
        return PacketIngressPreflight::evaluate(input);
    }
};

} // namespace protocol
} // namespace nexusminer

#endif // NEXUSMINER_PROTOCOL_SESSION_INGRESS_GATE_HPP
