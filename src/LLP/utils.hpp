#ifndef NEXUSMINER_UTILS_H
#define NEXUSMINER_UTILS_H

#include <string>
#include <vector>
#include <cstdlib>

namespace nexusminer
{

// ============================================================================
//  Endian-safe byte-to-integer helpers
//  ------------------------------------
//  These explicitly cast each uint8_t to the target unsigned type BEFORE
//  shifting, avoiding the C++ integer-promotion trap where a uint8_t is
//  promoted to *signed* int and a left-shift into the sign bit triggers
//  undefined behaviour (UB) in C++17 and earlier.
// ============================================================================

/// Read a big-endian uint32_t from a raw byte pointer.
/// Caller must guarantee that at least 4 bytes are readable at @p p.
inline uint32_t read_be32(const uint8_t* p) noexcept
{
	return (static_cast<uint32_t>(p[0]) << 24)
	     | (static_cast<uint32_t>(p[1]) << 16)
	     | (static_cast<uint32_t>(p[2]) <<  8)
	     |  static_cast<uint32_t>(p[3]);
}

/// Read a big-endian uint16_t from a raw byte pointer.
/// Caller must guarantee that at least 2 bytes are readable at @p p.
inline uint16_t read_be16(const uint8_t* p) noexcept
{
	return static_cast<uint16_t>(
		(static_cast<uint16_t>(p[0]) << 8) | static_cast<uint16_t>(p[1]));
}

/** Determines the Decimal of nBits per Channel for a decent "Frame of Reference".
Has no functionality in Network Operation. **/
inline double get_difficulty(uint32_t nBits, int nChannel)
{
	/** Prime Channel is just Decimal Held in Integer
		Multiplied and Divided by Significant Digits. **/
	if (nChannel == 1)
		return nBits / 10000000.0;

	/** Get the Proportion of the Bits First. **/
	double dDiff = (double)0x0000ffff / (double)(nBits & 0x00ffffff);

	/** Calculate where on Compact Scale Difficulty is. **/
	int nShift = nBits >> 24;

	/** Shift down if Position on Compact Scale is above 124. **/
	while (nShift > 124)
	{
		dDiff = dDiff / 256.0;
		nShift--;
	}

	/** Shift up if Position on Compact Scale is below 124. **/
	while (nShift < 124)
	{
		dDiff = dDiff * 256.0;
		nShift++;
	}

	/** Offset the number by 64 to give larger starting reference. **/
	return dDiff * ((nChannel == 2) ? 64 : 1024 * 1024 * 256);
}

/** Convert a 32 bit Unsigned Integer to Byte Vector using Bitwise Shifts. **/
inline std::vector<uint8_t> uint2bytes(uint32_t UINT)
{
	std::vector<uint8_t> BYTES(4, 0);
	BYTES[0] = UINT >> 24;
	BYTES[1] = UINT >> 16;
	BYTES[2] = UINT >> 8;
	BYTES[3] = UINT;

	return BYTES;
}


/** Convert a byte stream into unsigned integer 32 bit. **/
inline uint32_t bytes2uint(std::vector<uint8_t> const& BYTES, int nOffset = 0)
{
	if (BYTES.size() < nOffset + 4)
		return 0;

	return read_be32(BYTES.data() + nOffset);
}

/** Convert a 64 bit Unsigned Integer to Byte Vector using Bitwise Shifts. **/
inline std::vector<uint8_t> uint2bytes64(uint64_t UINT)
{
	std::vector<uint8_t> INTS[2];
	INTS[0] = uint2bytes(static_cast<uint32_t>(UINT));
	INTS[1] = uint2bytes(static_cast<uint32_t>(UINT >> 32));

	std::vector<uint8_t> BYTES;
	BYTES.insert(BYTES.end(), INTS[0].begin(), INTS[0].end());
	BYTES.insert(BYTES.end(), INTS[1].begin(), INTS[1].end());

	return BYTES;
}


/** Convert a byte Vector into unsigned integer 64 bit. **/
inline uint64_t bytes2uint64(std::vector<uint8_t> const& BYTES, int nOffset = 0) 
{
	return (bytes2uint(BYTES, nOffset) | (static_cast<uint64_t>(bytes2uint(BYTES, nOffset + 4)) << 32));
}

/** Convert Standard String into Byte Vector. **/
inline std::vector<uint8_t> string2bytes(std::string const& STRING)
{
	std::vector<uint8_t> BYTES(STRING.begin(), STRING.end());
	return BYTES;
}


/** Convert Byte Vector into Standard String. **/
inline std::string bytes2string(std::vector<uint8_t> const& BYTES, int nOffset = 0)
{
	std::string STRING(BYTES.begin() + nOffset, BYTES.end());
	return STRING;
}

/** Convert double into Byte Vector **/
inline std::vector<uint8_t> double2bytes(double DOUBLE)
{
	union {
		double DOUBLE;
		uint64_t UINT64;
	} u;
	u.DOUBLE = DOUBLE;

	return uint2bytes64(u.UINT64);
}

/** Convert Byte Vector into double **/
inline double bytes2double(std::vector<uint8_t> const& BYTES)
{
	uint64_t n64 = bytes2uint64(BYTES);
	union {
		double DOUBLE;
		uint64_t UINT64;
	} u;
	u.UINT64 = n64;
	return u.DOUBLE;
}

//inline const std::string time2datetimestring(time_t nTime)
//{
//	struct tm  tstruct;
//	char       buf[80];
//	tstruct = *gmtime(&nTime);
//	strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);
//
//	return buf;
//}
//
//// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
//inline const std::string currentDateTime()
//{
//	return time2datetimestring(time(0));
//}

/**
 * @brief Decode a Base58-encoded string to raw bytes
 * 
 * Used for decoding NXS account addresses to their 32-byte register representation.
 * 
 * @param str Base58-encoded string
 * @return Decoded bytes, or empty vector if decoding fails
 */
inline std::vector<uint8_t> decode_base58(const std::string& str)
{
    // Reverse lookup table for base58 characters (Bitcoin/NXS alphabet)
    // Maps ASCII character to base58 digit value (0-57), -1 for invalid
    static const int8_t base58_map[256] = {
        -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
        -1, 0, 1, 2, 3, 4, 5, 6,  7, 8,-1,-1,-1,-1,-1,-1,  // 0-9
        -1, 9,10,11,12,13,14,15, 16,-1,17,18,19,20,21,-1,  // A-O
        22,23,24,25,26,27,28,29, 30,31,32,-1,-1,-1,-1,-1,  // P-Z
        -1,33,34,35,36,37,38,39, 40,41,42,43,-1,44,45,46,  // a-n
        47,48,49,50,51,52,53,54, 55,56,57,-1,-1,-1,-1,-1,  // o-z
        -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
    };
    
    std::vector<uint8_t> result;
    result.reserve(str.length());
    
    // Count leading '1' characters (will become leading 0x00 bytes)
    size_t leading_zeros = 0;
    for (size_t i = 0; i < str.length() && str[i] == '1'; ++i) {
        ++leading_zeros;
    }
    
    // Allocate enough space (base58 encoding increases size by ~37%)
    std::vector<uint8_t> b256(str.length() * 733 / 1000 + 1, 0);
    
    // Process each character
    for (size_t i = 0; i < str.length(); ++i) {
        int8_t carry = base58_map[static_cast<uint8_t>(str[i])];
        if (carry == -1) {
            // Invalid character
            return std::vector<uint8_t>();
        }
        
        // Multiply by 58 and add carry
        for (int j = static_cast<int>(b256.size()) - 1; j >= 0; --j) {
            int temp = static_cast<int>(b256[j]) * 58 + carry;
            b256[j] = static_cast<uint8_t>(temp % 256);
            carry = static_cast<int8_t>(temp / 256);
        }
    }
    
    // Skip leading zeros in b256
    auto it = b256.begin();
    while (it != b256.end() && *it == 0) {
        ++it;
    }
    
    // Build result with leading zeros + decoded data
    result.assign(leading_zeros, 0x00);
    result.insert(result.end(), it, b256.end());
    
    return result;
}


}

#endif
