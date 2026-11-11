#pragma once

#ifndef NEXUSMINER_PROTOCOL_HEX_PREFIX_UTILS_HPP
#define NEXUSMINER_PROTOCOL_HEX_PREFIX_UTILS_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace nexusminer {
namespace protocol {

// Callers outside nexusminer::protocol should qualify this helper as
// nexusminer::protocol::format_hex_prefix (or add a local using declaration
// in the implementation file that includes this header).
inline std::string format_hex_prefix(const uint8_t* bytes, std::size_t size, std::size_t prefix_bytes)
{
    const std::size_t prefix_size = std::min(size, prefix_bytes);
    static const char* const HEX = "0123456789abcdef";

    std::string out;
    out.reserve(prefix_size * 2);

    for (std::size_t i = 0; i < prefix_size; ++i) {
        const uint8_t byte = bytes[i];
        out.push_back(HEX[(byte >> 4) & 0x0F]);
        out.push_back(HEX[byte & 0x0F]);
    }

    return out;
}

inline std::string format_hex_prefix(const std::vector<uint8_t>& bytes, std::size_t prefix_bytes)
{
    return format_hex_prefix(bytes.data(), bytes.size(), prefix_bytes);
}

template <std::size_t N>
inline std::string format_hex_prefix(const std::array<uint8_t, N>& bytes, std::size_t prefix_bytes)
{
    return format_hex_prefix(bytes.data(), bytes.size(), prefix_bytes);
}

} // namespace protocol
} // namespace nexusminer

#endif // NEXUSMINER_PROTOCOL_HEX_PREFIX_UTILS_HPP
