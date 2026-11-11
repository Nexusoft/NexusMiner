#ifndef NEXUSMINER_PROTOCOL_PACKET_INGRESS_PREFLIGHT_HPP
#define NEXUSMINER_PROTOCOL_PACKET_INGRESS_PREFLIGHT_HPP

#include "protocol/session_semantic_types.hpp"
#include "protocol_lane.hpp"
#include <string>
#include <cstdint>

namespace nexusminer {
namespace protocol {

struct SessionOwnershipStamp
{
    SessionId session_id{};
    SessionEpoch session_epoch{};

    bool valid() const
    {
        return !session_id.is_default() && !session_epoch.is_default();
    }

    void clear()
    {
        session_id.clear();
        session_epoch.clear();
    }
};

enum class PacketStaleReason
{
    NONE,
    SESSION_ID_MISMATCH,
    OWNERSHIP_EPOCH_MISMATCH,
    OWNERSHIP_SESSION_ID_MISMATCH
};

struct PacketIngressDecision
{
    bool allow_processing{false};
    bool force_reauth{false};
    bool drop_as_stale{false};
    bool mark_degraded{false};
    PacketStaleReason stale_reason{PacketStaleReason::NONE};
    std::string reason;
};

class PacketIngressPreflight
{
public:
    struct Input
    {
        bool has_authoritative_session{false};
        bool authoritative_authenticated{false};
        SessionId authoritative_session_id{};
        SessionEpoch authoritative_session_epoch{};
        ProtocolLane authoritative_lane{ProtocolLane::UNKNOWN};
        ProtocolLane packet_lane{ProtocolLane::UNKNOWN};
        bool validate_lane{false};         // Lane Health Monitor: reject cross-lane packets
        bool allow_without_active_session{false};
        SessionId packet_session_id{};   // default = no session ID in packet
        SessionEpoch owner_epoch{};      // default = no ownership stamp
        SessionId owner_session_id{};    // default = no ownership stamp
    };

    static PacketIngressDecision evaluate(const Input& input)
    {
        PacketIngressDecision decision;

        if (!input.has_authoritative_session) {
            decision.reason = "no authoritative session container";
            return decision;
        }

        if (input.validate_lane &&
            input.authoritative_lane != ProtocolLane::UNKNOWN &&
            input.packet_lane != ProtocolLane::UNKNOWN &&
            input.authoritative_lane != input.packet_lane) {
            decision.reason = "packet lane mismatched authoritative session lane";
            decision.mark_degraded = true;
            return decision;
        }

        if (!input.allow_without_active_session && !input.authoritative_authenticated) {
            decision.reason = "authoritative session is not authenticated";
            decision.force_reauth = true;
            decision.mark_degraded = true;
            return decision;
        }

        if (!input.packet_session_id.is_default() &&
            input.packet_session_id != input.authoritative_session_id) {
            decision.reason = "packet session id mismatched authoritative session";
            decision.drop_as_stale = true;
            decision.mark_degraded = true;
            decision.stale_reason = PacketStaleReason::SESSION_ID_MISMATCH;
            return decision;
        }

        if (!input.owner_epoch.is_default()) {
            if (input.owner_epoch != input.authoritative_session_epoch) {
                decision.reason = "packet ownership epoch mismatched authoritative session";
                decision.drop_as_stale = true;
                decision.mark_degraded = true;
                decision.stale_reason = PacketStaleReason::OWNERSHIP_EPOCH_MISMATCH;
                return decision;
            }

            if (!input.owner_session_id.is_default() &&
                input.owner_session_id != input.authoritative_session_id) {
                decision.reason = "packet ownership session id mismatched authoritative session";
                decision.drop_as_stale = true;
                decision.mark_degraded = true;
                decision.stale_reason = PacketStaleReason::OWNERSHIP_SESSION_ID_MISMATCH;
                return decision;
            }
        }

        decision.allow_processing = true;
        decision.reason = "authoritative session preflight passed";
        return decision;
    }
};

} // namespace protocol
} // namespace nexusminer

#endif // NEXUSMINER_PROTOCOL_PACKET_INGRESS_PREFLIGHT_HPP
