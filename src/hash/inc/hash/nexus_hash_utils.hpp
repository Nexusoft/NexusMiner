#ifndef NEXUS_HASH_UTILS_HPP
#define NEXUS_HASH_UTILS_HPP

//#include <nexus_skein.hpp>
//#include <nexus_keccak.hpp>

template <typename T>
static int findMSB(T v)
//finds the position of the most significant 1.  returns zero if zero.
{
    unsigned r = 0;
    //shift right until all the ones are gone
    while (v >>= 1) {
        r++;
    }
    return r;
}


static void decodeBits(uint32_t bits, int& leadingZerosRequired, uint64_t& difficulty64)
//takes the "bits" field as input and outputs the required leading zeros for the given difficulty and a difficulty number for comparison with the hash.
//Note that we take a shortcut and only compare the upper 64 bits instead of all 1024. 
{
	uint32_t nCompact = bits;
	//the upper 8 bits are the exponent
	int nSize = nCompact >> 24;
	//the lower 24 bits are the mantissa
	uint32_t nWord = nCompact & 0x007fffff;
	int matissaLeadingZeros = 31 - findMSB(nWord);
	int exponentLeadingZeros = 0;
	if (nSize >= 3)
	{
		exponentLeadingZeros = 1024 - 8 * (nSize - 3) - 32;
	}
	//leading zeros in bits required of the hash for it to pass the current difficulty.
	leadingZerosRequired = exponentLeadingZeros + matissaLeadingZeros;

	if (exponentLeadingZeros <= 32)
	{
		difficulty64 = static_cast<uint64_t>(nWord) << (32 - exponentLeadingZeros);
	}
	else
	{
		difficulty64 = static_cast<uint64_t>(nWord) >> (exponentLeadingZeros - 32);
	}
}


#endif