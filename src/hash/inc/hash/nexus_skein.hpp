// skein 1024 implementation stripped down for NXS

// ACH Skein optimization notes
// The config stage has fixed inputs and is precomputed.  
// There are two calls to the UBI function. One for the message, one for the output. We will call these UBI1 and UBI2
// There are three 128 bit tweaks resulting in nine 64 bit subtweaks.Two tweaks for UBI1 and one for UBI2. The tweaks are fixed and can be precomputed.
// The key schedule for UBI1 is fixed but the subkeys for UBI2 must be calculated using the output of UBI1.
// UBI1 calls threefish twice. UBI2 calls threefish once for three total calls to threefish. Each threefish call is 80 rounds for a total of 240 rounds.
// For Nexus there are three threefish calls. We will call them threefish1, threefish2, and threefish3. Each has some unique properties that can be precomputed.
// Each threefish takes a 1024 bit key and a 1024 bit message as input. These are stored as arrays of 16 64 bit words.
// The key and message for threefish2 are key2 and message2.
// Threefish1 processes the first 1024 bits of the header. For a given block, this is fixed. The key for threefish1 is also fixed and can be precomputed.
// Threefish1 is therefore fixed for a given block and can be computed once per block in software.
// Threefish2 takes the output of threefish1 to make it's key.  Key2 is therefore also fixed per block and can be computed once for each new block in software.
// The message for threefish2 (aka message2) is the last half of the nexus header padded with zeros. The nonce is included in this message.
// Most of message2 is fixed per block.The only part that changes is the 64 bit nonce section.
// Message3 is fixed at all zeros. Key3 is a function of threefish2 and message2.
// We will send the hardware accelerator these values per block as inputs :
// Key2 as 17 unsigned 64 bit numbers
// Message2 as 11 unsigned 64 bit numbers (The 11th word is the starting nonce).
// 

#ifndef NEXUS_SKEIN_HPP
#define NEXUS_SKEIN_HPP

#include "int_array.hpp"
#include <cstdint>

class NexusSkein
{
public:
    NexusSkein();
    NexusSkein(const std::vector<unsigned char>& m);
private:
    // special key constant in threefish. This is the original constant. It was changed in skein version 1.3    
    static constexpr uint64_t C240 = 0x5555555555555555;
    static constexpr int numRounds = 80;  //rounds within threefish
    static constexpr int numWords = 16;  //number of 64 bit words (1024 bits / 64)
    static constexpr int subkeyCount = numRounds / 4 + 1;  //21 subkeys
    static constexpr int headerLength = 216;  //bytes (hashing mode)
    static constexpr int headerLengthPrime = 208; 
        
     
public:
    using stateType = Int_array<uint64_t, numWords>;
    using keyType = Int_array<uint64_t, numWords + 1>;

private:
    using tweakType = std::array<uint64_t, 3>;
    using subkeyType = std::array<Int_array<uint64_t, numWords>, subkeyCount>;
    //word permutation constants
    static constexpr int permuteIndices[numWords] = { 0, 9, 2, 13, 6, 11, 4, 15, 10, 7, 12, 3, 14, 5, 8, 1 };
    //The original mix function rotation constants.These changed between version 1.1 and 1.2 of Skein
    static constexpr int R[8][8] = { {55, 43, 37, 40, 16, 22, 38, 12},{25, 25, 46, 13, 14, 13, 52, 57},
        {33, 8, 18, 57, 21, 12, 32, 54},{34, 43, 25, 60, 44, 9, 59, 34},
        {28, 7, 47, 48, 51, 9, 35, 41},{17, 6, 18, 25, 43, 42, 40, 15},
        {58, 7, 32, 45, 19, 18, 2, 56},{47, 49, 27, 58, 37, 48, 53, 56} };

    //This is the precomputed config key for the first threefish call.
    const std::string hashInitStr = "56210962be52435aca01f0721a8b6e5f26cea2a19cfecbffca8b036796c3236c6ceb34cefc8b3a583e6aa4d411fbdb3f980930a8fcac0433d20f7fa15f67f6b26babf70e7399259de4a9fe3d0da21409d3db94a4af9c1acc8c38a6a00d032898dce3deaa5d9d330d86a0e2c435de46fcd1a6192ef5e4d653dd1d5d712f956356";
        
    //precomputed tweaks
    static constexpr tweakType t1 { 0x00000000000080, 0x7000000000000000, 0x7000000000000080 };
    static constexpr tweakType t2 { 0x000000000000D8, 0xB000000000000000, 0xB0000000000000D8 };
    static constexpr tweakType t3 { 0x00000000000008, 0xFF00000000000000, 0xFF00000000000008 };

    //for prime the message length is shorter so we need a different tweak
    static constexpr tweakType t2_prime{ 0x000000000000D0, 0xB000000000000000, 0xB0000000000000D0 };


    // Rotate left : 0b1001 -- > 0b0011
    inline uint64_t rol(uint64_t val, int r_bits)
    {
        return (val << r_bits) | (val >> (64 - r_bits));
    }

    inline uint64_t mix1(uint64_t x0, uint64_t x1)
    {
        //first part of the mix function of threefish
        //x0 and x1 are the two 64 bit inputs
        //just add them.  overflow is ok here.
        uint64_t y0 = x0 + x1;
        return y0;
    }

    inline uint64_t mix2(uint64_t x1, uint64_t y0, int d, int j)
    {
        //second part of the mix function of threefish
        //x1 is a two 64 bit inputs
        //y0 is the result of mix 1
        //d is the threefish round(0 to Nr = 80)
        //j is the column(0 to 7)
        //the output is the xor of y0 with a rotated version of x1
        //the rotation constants repeat every 8 rows hence the d % 8
        //bitwise rotation by the rotation constant and bitwise xor with y0
        uint64_t y1 = rol(x1, R[d % 8][j]) ^ y0;
        return y1;
    }

    stateType permute(const stateType& s);
    stateType threefish1024(stateType p, const subkeyType& subkey);
    void makeSubkeys(const stateType& key, const tweakType& tweak, subkeyType& subkey);
    keyType makeKeyFromState(const stateType& state);
        
    //key2 is a midstate value that is fixed per block (not dependent on the nonce).  
    keyType key2;
    //message2 is the second portion of the block header that includes the nonce. 
    stateType message2;
    //message1 is the first part of the block header.  This is independent of the nonce.
    stateType message1;
    //the state after the first round of threefish.  This is independent of the nonce.
    stateType threefish1Out;
    //the set of subkeys used as input to round two. This is independent of the nonce.
    subkeyType subkey2;
    //the fixed initial state for Nexus
    stateType hashInitState;
    //the final output of the skein hash function after three rounds of threefish
    stateType hash;
    bool primeMode = false;

public:
    void setMessage(std::vector<unsigned char>);
    void calculateKey2();
    keyType getKey2();
    stateType getMessage1();
    stateType getMessage2();
    void calculateHash();
    stateType getHash();
    void setNonce(uint64_t nonce);
    uint64_t getNonce();
};


#endif
