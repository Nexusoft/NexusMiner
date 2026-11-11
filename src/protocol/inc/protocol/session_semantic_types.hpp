#ifndef NEXUSMINER_PROTOCOL_SESSION_SEMANTIC_TYPES_HPP
#define NEXUSMINER_PROTOCOL_SESSION_SEMANTIC_TYPES_HPP

#include <cstdint>
#include <functional>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace nexusminer {
namespace protocol {

template <typename T, typename Tag>
class SemanticValue
{
public:
    using value_type = T;

    SemanticValue() = default;

    template <typename U,
              typename = typename std::enable_if<std::is_constructible<T, U&&>::value>::type>
    explicit SemanticValue(U&& value)
        : m_value(std::forward<U>(value))
    {
    }

    const T& get() const
    {
        return m_value;
    }

    T& get()
    {
        return m_value;
    }

    bool is_default() const
    {
        return m_value == T{};
    }

    void clear()
    {
        m_value = T{};
    }

    friend bool operator==(const SemanticValue& lhs, const SemanticValue& rhs)
    {
        return lhs.m_value == rhs.m_value;
    }

    friend bool operator!=(const SemanticValue& lhs, const SemanticValue& rhs)
    {
        return !(lhs == rhs);
    }

    friend bool operator<(const SemanticValue& lhs, const SemanticValue& rhs)
    {
        return lhs.m_value < rhs.m_value;
    }

    friend std::ostream& operator<<(std::ostream& os, const SemanticValue& sv)
    {
        os << sv.m_value;
        return os;
    }

    /// Return a new SemanticValue with the underlying value incremented by 1.
    /// Useful for monotonic counters like SessionEpoch.
    SemanticValue next() const
    {
        return SemanticValue(m_value + 1);
    }

private:
    T m_value{};
};

struct SessionIdTag {};
struct SessionEpochTag {};
struct SessionGenesisHashTag {};
struct RewardHashTag {};
struct FalconHashKeyIdTag {};
struct SessionFingerprintTag {};

using SessionId = SemanticValue<uint32_t, SessionIdTag>;
using SessionEpoch = SemanticValue<uint64_t, SessionEpochTag>;
using SessionGenesisHash = SemanticValue<std::vector<uint8_t>, SessionGenesisHashTag>;
using RewardHash = SemanticValue<std::vector<uint8_t>, RewardHashTag>;
using FalconHashKeyId = SemanticValue<std::string, FalconHashKeyIdTag>;
using SessionFingerprint = SemanticValue<std::string, SessionFingerprintTag>;

} // namespace protocol
} // namespace nexusminer

// std::hash specializations for use in unordered containers.
namespace std {

template <typename T, typename Tag>
struct hash<nexusminer::protocol::SemanticValue<T, Tag>>
{
    std::size_t operator()(const nexusminer::protocol::SemanticValue<T, Tag>& v) const noexcept
    {
        return std::hash<T>{}(v.get());
    }
};

} // namespace std

#endif // NEXUSMINER_PROTOCOL_SESSION_SEMANTIC_TYPES_HPP
