#ifndef NEXUSMINER_PROTOCOL_SESSION_BINDING_HPP
#define NEXUSMINER_PROTOCOL_SESSION_BINDING_HPP

#include "protocol/session_identity.hpp"
#include <vector>
#include <string>

namespace nexusminer {
namespace protocol {

/**
 * @brief Canonical batch value for session, crypto, reward, and submit ownership.
 *
 * SessionBinding packages the authoritative session-container fields that must
 * travel together through reconnect, reward, submit, template, and diagnostics
 * flows so they cannot silently diverge.
 */
struct SessionBinding
{
    SessionId session_id{};
    SessionEpoch session_epoch{};
    SessionGenesisHash session_genesis{};
    FalconHashKeyId falcon_key_id{};
    std::vector<uint8_t> chacha20_session_key;
    SessionFingerprint chacha20_key_fingerprint{};
    ProtocolLane active_lane{ProtocolLane::UNKNOWN};
    bool authenticated{false};
    bool chacha20_ready{false};

    std::string reward_address;
    RewardHash reward_hash{};
    bool reward_bound{false};

    uint32_t channel{0};
    bool ready_for_submit{false};
    bool ready_for_get_block{false};
    bool full_recovery_required{true};
    bool work_request_allowed{false};
    bool mining_ready{false};

    SessionIdentity identity{};

    bool has_session() const
    {
        return !session_id.is_default() && !session_epoch.is_default();
    }

    bool session_requires_full_recovery() const
    {
        return full_recovery_required || !authenticated || !has_session();
    }

    bool may_request_work() const
    {
        return work_request_allowed && !session_requires_full_recovery();
    }

    bool is_fully_mining_ready() const
    {
        return mining_ready && !session_requires_full_recovery();
    }

    bool can_submit_work() const
    {
        return is_fully_mining_ready();
    }

    bool can_request_get_block() const
    {
        return may_request_work();
    }

    bool has_crypto_context() const
    {
        return !chacha20_session_key.empty() && chacha20_ready;
    }

    bool identity_matches_session() const
    {
        return identity.is_empty() ||
               (identity.session_id() == session_id &&
                identity.session_epoch() == session_epoch &&
                identity.lane() == active_lane);
    }
};

} // namespace protocol
} // namespace nexusminer

#endif // NEXUSMINER_PROTOCOL_SESSION_BINDING_HPP
