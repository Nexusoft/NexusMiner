#ifndef NEXUSMINER_PROTOCOL_SESSION_IDENTITY_HPP
#define NEXUSMINER_PROTOCOL_SESSION_IDENTITY_HPP

#include "protocol/session_semantic_types.hpp"
#include "protocol_lane.hpp"
#include <cstdint>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace nexusminer {
namespace protocol {

/**
 * @brief SessionIdentity — Readonly value object capturing the complete
 *        identity state of an authenticated mining session at a point in time.
 *
 * This is the canonical session identity bundle.  It binds together the
 * node-assigned session_id, session epoch, Falcon hashKeyID, ChaCha20
 * fingerprint, and protocol lane so they cannot diverge.
 *
 * SessionIdentity is constructed at authentication time and frozen — individual
 * fields cannot be mutated independently.  A new identity is created on each
 * authentication cycle.
 *
 * @invariant Once constructed, a valid SessionIdentity has:
 *   - A non-zero session_id
 *   - A non-zero session_epoch
 *   - A non-empty falcon hashKeyID or pubkey hash
 *   - session_id, epoch, ChaCha20 fingerprint, Falcon identity, and lane are all bound together
 *
 * Identity hashing:  The Falcon public key is canonically hashed using
 * LLC::SK256(vPubKey) → 256-bit Skein-Keccak hash.  This matches the NODE-side
 * hashKeyID used for miner identity in session recovery and cross-comparison.
 * The caller (SessionManager) pre-computes the SK256 hash and passes the
 * 32-byte result to the constructor.
 *
 * Thread safety: SessionIdentity is a value type.  It can be copied freely and
 * shared between threads without synchronization.
 */
class SessionIdentity
{
public:
    // ── Construction ──────────────────────────────────────────────────────

    /// Default-constructed identity is invalid (empty/zero).
    SessionIdentity() = default;

    /**
     * @brief Construct a complete session identity from individual fields.
     *
     * Typically called from SessionManager::transition_to_authenticated_locked()
     * at the moment authentication succeeds.
     *
     * @param session_id            Node-assigned wire protocol session ID (non-zero)
     * @param session_epoch         Monotonic session generation counter
     * @param genesis_hash          Tritium genesis hash (32 bytes, used for ChaCha20 KDF)
     * @param chacha20_key          Derived ChaCha20 session key (32 bytes)
     * @param falcon_pubkey_hash    Pre-computed SK256 hash of Falcon public key (32 bytes).
     *                              Must be computed by the caller via LLC::SK256(vPubKey).GetBytes()
     *                              to match the NODE-side hashKeyID used for miner identity.
     * @param lane                  Active protocol lane (LEGACY or STATELESS)
     * @param falcon_key_id         Canonical hex hashKeyID for the Falcon public key
     * @param chacha20_fingerprint  Diagnostic/session fingerprint for the active ChaCha20 key
     */
    SessionIdentity(SessionId session_id,
                    SessionEpoch session_epoch,
                    std::vector<uint8_t> genesis_hash,
                    std::vector<uint8_t> chacha20_key,
                    std::vector<uint8_t> falcon_pubkey_hash,
                    ProtocolLane lane,
                    FalconHashKeyId falcon_key_id = {},
                    SessionFingerprint chacha20_fingerprint = {})
        : m_session_id(session_id)
        , m_session_epoch(session_epoch)
        , m_genesis_hash(std::move(genesis_hash))
        , m_chacha20_key(std::move(chacha20_key))
        , m_falcon_pubkey_hash(std::move(falcon_pubkey_hash))
        , m_lane(lane)
        , m_falcon_key_id(std::move(falcon_key_id))
        , m_chacha20_fingerprint(std::move(chacha20_fingerprint))
    {
    }

    // ── Accessors ─────────────────────────────────────────────────────────

    /// Node-assigned wire protocol session ID.
    SessionId session_id() const { return m_session_id; }

    /// Monotonic session generation counter (never resets to 0).
    SessionEpoch session_epoch() const { return m_session_epoch; }

    /// Tritium genesis hash used for ChaCha20 key derivation (32 bytes).
    const std::vector<uint8_t>& genesis_hash() const { return m_genesis_hash; }

    /// Derived ChaCha20 session key (32 bytes).
    const std::vector<uint8_t>& chacha20_key() const { return m_chacha20_key; }

    /// SK256 hash of the Falcon public key (32 bytes, matches NODE-side hashKeyID).
    /// Used for cross-miner identity comparison and session recovery handshake.
    const std::vector<uint8_t>& falcon_pubkey_hash() const { return m_falcon_pubkey_hash; }

    /// Canonical Falcon hashKeyID string for the authenticated miner identity.
    FalconHashKeyId falcon_key_id() const { return m_falcon_key_id; }

    /// Fingerprint of the active ChaCha20 session key.
    SessionFingerprint chacha20_fingerprint() const { return m_chacha20_fingerprint; }

    /// Active protocol lane at authentication time.
    ProtocolLane lane() const { return m_lane; }

    // ── Validation ────────────────────────────────────────────────────────

    /**
     * @brief Check if this identity represents a valid authenticated session.
     *
     * A valid identity has:
     *  - Non-zero session_id
     *  - Non-zero session_epoch
     *
     * Note: chacha20_key and genesis_hash may be empty in legacy mode (no
     * encryption), so they are not required for validity.
     *
     * @return true if the identity is valid
     */
    bool is_valid() const
    {
        return !m_session_id.is_default() && !m_session_epoch.is_default();
    }

    /**
     * @brief Check if this identity is empty (default-constructed or cleared).
     * @return true if session_id is zero
     */
    bool is_empty() const
    {
        return m_session_id.is_default() && m_session_epoch.is_default();
    }

    /**
     * @brief Check if the ChaCha20 crypto context is ready (key + genesis present).
     * @return true if both chacha20_key and genesis_hash are non-empty
     */
    bool has_crypto_context() const
    {
        return !m_chacha20_key.empty() && !m_genesis_hash.empty();
    }

    // ── Comparison ────────────────────────────────────────────────────────

    /**
     * @brief Check if another identity refers to the same authenticated session.
     *
     * Two identities match if they have the same session_id AND session_epoch.
     * This is the primary check for "is this template/submit still valid?"
     *
     * @param other The identity to compare against
     * @return true if both refer to the same session
     */
    bool matches(const SessionIdentity& other) const
    {
        return m_session_id == other.m_session_id &&
               m_session_epoch == other.m_session_epoch;
    }

    /**
     * @brief Check if another identity was issued by the same miner identity.
     *
     * Two identities share a miner if they have the same falcon_pubkey_hash.
     * This detects cross-miner credential leakage.
     *
     * @param other The identity to compare against
     * @return true if both belong to the same Falcon identity
     */
    bool same_miner(const SessionIdentity& other) const
    {
        if (!m_falcon_key_id.is_default() && !other.m_falcon_key_id.is_default()) {
            return m_falcon_key_id == other.m_falcon_key_id;
        }
        return !m_falcon_pubkey_hash.empty() &&
               m_falcon_pubkey_hash == other.m_falcon_pubkey_hash;
    }

    /**
     * @brief Full cryptographic identity match — session, miner, and crypto context.
     *
     * Strongest check: session matches, same miner, and same ChaCha20 key.
     * Used for hardened submit validation.
     *
     * @param other The identity to compare against
     * @return true if all identity fields match
     */
    bool full_match(const SessionIdentity& other) const
    {
        bool fingerprint_matches = false;
        if (!m_chacha20_fingerprint.is_default() && !other.m_chacha20_fingerprint.is_default()) {
            fingerprint_matches = (m_chacha20_fingerprint == other.m_chacha20_fingerprint);
        } else if (m_chacha20_fingerprint.is_default() && other.m_chacha20_fingerprint.is_default()) {
            fingerprint_matches = (m_chacha20_key == other.m_chacha20_key);
        }
        return matches(other) &&
               same_miner(other) &&
               fingerprint_matches &&
               m_lane == other.m_lane;
    }

    // ── Diagnostics ───────────────────────────────────────────────────────

    /**
     * @brief Short human-readable fingerprint for log messages.
     *
     * Format: "sid=0x{session_id:08x}/e{epoch}"
     * Example: "sid=0x548c90d3/e2"
     *
     * @return Fingerprint string
     */
    std::string fingerprint() const
    {
        std::ostringstream oss;
        oss << "sid=0x" << std::hex << std::setw(8) << std::setfill('0')
            << m_session_id.get() << "/e" << std::dec << m_session_epoch.get();
        return oss.str();
    }

    /**
     * @brief Extended diagnostic string with all identity fields.
     *
     * Includes session_id, epoch, lane, genesis fingerprint, key fingerprint,
     * and pubkey hash.  Suitable for error logs and diagnostics.
     *
     * @return Full diagnostic string
     */
    std::string diagnostics() const
    {
        std::ostringstream oss;
        oss << "SessionIdentity{"
            << "sid=0x" << std::hex << std::setw(8) << std::setfill('0') << m_session_id.get()
            << " epoch=" << std::dec << m_session_epoch.get()
            << " lane=" << get_lane_name(m_lane)
            << " genesis=" << hex_prefix(m_genesis_hash, 4)
            << " key=" << hex_prefix(m_chacha20_key, 4)
            << " chacha_fp=" << printable_semantic(m_chacha20_fingerprint)
            << " hashkeyid=" << printable_semantic(m_falcon_key_id)
            << " pubkey_hash=" << hex_prefix(m_falcon_pubkey_hash, 4)
            << " valid=" << (is_valid() ? "yes" : "no")
            << "}";
        return oss.str();
    }

    // ── Equality ──────────────────────────────────────────────────────────

    friend bool operator==(const SessionIdentity& lhs, const SessionIdentity& rhs)
    {
        return lhs.m_session_id == rhs.m_session_id &&
               lhs.m_session_epoch == rhs.m_session_epoch &&
               lhs.m_genesis_hash == rhs.m_genesis_hash &&
               lhs.m_chacha20_key == rhs.m_chacha20_key &&
               lhs.m_falcon_pubkey_hash == rhs.m_falcon_pubkey_hash &&
               lhs.m_lane == rhs.m_lane &&
               lhs.m_falcon_key_id == rhs.m_falcon_key_id &&
               lhs.m_chacha20_fingerprint == rhs.m_chacha20_fingerprint;
    }

    friend bool operator!=(const SessionIdentity& lhs, const SessionIdentity& rhs)
    {
        return !(lhs == rhs);
    }

private:
    /// Format first N bytes of a vector as hex for diagnostics.
    static std::string hex_prefix(const std::vector<uint8_t>& data, size_t n)
    {
        if (data.empty()) return "(empty)";
        std::ostringstream oss;
        size_t len = std::min(n, data.size());
        for (size_t i = 0; i < len; ++i) {
            oss << std::hex << std::setw(2) << std::setfill('0')
                << static_cast<unsigned>(data[i]);
        }
        if (data.size() > n) oss << "...";
        return oss.str();
    }

    template <typename SemanticT>
    static std::string printable_semantic(const SemanticT& value)
    {
        return value.is_default() ? "(empty)" : value.get();
    }

    SessionId m_session_id{};
    SessionEpoch m_session_epoch{};
    std::vector<uint8_t> m_genesis_hash;
    std::vector<uint8_t> m_chacha20_key;
    std::vector<uint8_t> m_falcon_pubkey_hash;  // SK256 hash of Falcon pubkey (32 bytes)
    ProtocolLane m_lane{ProtocolLane::UNKNOWN};
    FalconHashKeyId m_falcon_key_id{};
    SessionFingerprint m_chacha20_fingerprint{};
};

} // namespace protocol
} // namespace nexusminer

#endif // NEXUSMINER_PROTOCOL_SESSION_IDENTITY_HPP
