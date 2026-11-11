#ifndef NEXUSMINER_PROTOCOL_SUBMIT_CONTEXT_HPP
#define NEXUSMINER_PROTOCOL_SUBMIT_CONTEXT_HPP

#include "protocol/session_semantic_types.hpp"
#include "protocol/session_identity.hpp"
#include <cstdint>

namespace nexusminer {
namespace protocol {

struct SubmitContext
{
    SessionId session_id{};
    SessionEpoch session_epoch{};
    uint32_t template_height{0};
    uint32_t chain_height{0};

    /// Canonical identity bundle captured at submit time.
    /// Binds session_id, epoch, crypto context, and miner identity together.
    SessionIdentity identity{};

    bool has_authoritative_height() const
    {
        return template_height != 0;
    }

    bool matches_submit_height(uint32_t submit_height) const
    {
        return !has_authoritative_height() || submit_height == template_height;
    }
};

} // namespace protocol
} // namespace nexusminer

#endif // NEXUSMINER_PROTOCOL_SUBMIT_CONTEXT_HPP
