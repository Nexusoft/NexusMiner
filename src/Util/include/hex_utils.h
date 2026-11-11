#ifndef NEXUSMINER_HEX_UTILS_H
#define NEXUSMINER_HEX_UTILS_H

#include <string>
#include <vector>
#include <algorithm>
#include <limits>
#include <cstdint>

namespace HexUtils
{
    /** Format byte vector as hex string (matches node-side FormatHexDump)
     *  Uses lookup table for 4x faster performance than snprintf
     *  @param data The byte vector to format
     *  @param maxBytes Maximum number of bytes to format
     *  @return Hex string with spaces between bytes (e.g., "a3 7f 2c 9b")
     */
    inline std::string FormatHexDump(const std::vector<uint8_t>& data, size_t maxBytes = std::numeric_limits<size_t>::max())
    {
        static const char hex_chars[] = "0123456789abcdef";
        
        size_t count = std::min(data.size(), maxBytes);
        if(count == 0)
            return "";
        
        std::string result;
        result.reserve(count * 3);  // 2 hex chars + 1 space per byte
        
        for(size_t i = 0; i < count; ++i)
        {
            uint8_t byte = data[i];
            result += hex_chars[(byte >> 4) & 0x0F];  // High nibble
            result += hex_chars[byte & 0x0F];         // Low nibble
            result += ' ';
        }
        
        // Remove trailing space
        if(!result.empty() && result.back() == ' ')
            result.pop_back();
        
        return result;
    }

    /** Split hex dump into multiple lines for readability
     *  @param hexDump The hex dump string to split
     *  @param bytesPerLine Number of bytes per line (default 32)
     *  @return Vector of lines
     */
    inline std::vector<std::string> SplitHexDump(const std::string& hexDump, size_t bytesPerLine = 32)
    {
        std::vector<std::string> lines;
        if(hexDump.empty())
            return lines;
        
        // Clamp to reasonable limits
        bytesPerLine = std::max(size_t(1), std::min(bytesPerLine, size_t(256)));
        
        size_t pos = 0;
        while(pos < hexDump.length())
        {
            size_t remaining = hexDump.length() - pos;
            size_t charsThisLine = std::min(remaining, bytesPerLine * 3);
            
            lines.push_back(hexDump.substr(pos, charsThisLine));
            pos += charsThisLine;
            
            // Skip spaces at boundary
            while(pos < hexDump.length() && hexDump[pos] == ' ')
                pos++;
        }
        
        return lines;
    }

    /** Check if data looks like plaintext (heuristic for debugging)
     *  @param data The data to check
     *  @param sampleSize Number of bytes to check (default 32)
     *  @return true if data appears to be plaintext
     */
    inline bool LooksLikePlaintext(const std::vector<uint8_t>& data, size_t sampleSize = 32)
    {
        if(data.empty())
            return true;
        
        size_t checkSize = std::min(data.size(), sampleSize);
        size_t zeroCount = 0;
        
        // Plaintext blocks often have many zero bytes in specific positions
        for(size_t i = 0; i < checkSize; ++i)
        {
            if(data[i] == 0x00)
                zeroCount++;
        }
        
        // If more than 30% are zeros, likely plaintext (heuristic)
        return (zeroCount * 100 / checkSize) > 30;
    }
}

#endif // NEXUSMINER_HEX_UTILS_H
