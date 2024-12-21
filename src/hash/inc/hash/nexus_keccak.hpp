// keccak 1024 implementation stripped down for NXS
#ifndef NEXUS_KECCAK_HPP
#define NEXUS_KECCAK_HPP

#include "int_array.hpp"
#include <cstdint>

class NexusKeccak
{
public:
    using k_1024 = Int_array<uint64_t, 16>;

private:
    static constexpr int numPlane = 5;
    static constexpr int numSheet = 5;
    static constexpr int messageLength = 9;
    using k_plane = Int_array<uint64_t, numSheet>;
    using k_message = Int_array<uint64_t, messageLength>;
    using k_state = std::array<k_plane, numPlane>;
    static constexpr int numRounds = 24;
    static constexpr uint64_t NXS_SUFFIX_1 = 0x5;
    static constexpr uint64_t NXS_SUFFIX_2 = 0x8000000000000000;
    static constexpr uint64_t round_const[numRounds] = 
       {0x0000000000000001, 0x0000000000008082, 0x800000000000808A,0x8000000080008000,
        0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
        0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
        0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
        0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
        0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008 };

    //rotation constants
    static constexpr int r[numSheet][numPlane] =
        { {0, 36, 3, 41, 18},
        {1, 44, 10, 45, 2},
        {62, 6, 43, 15, 61},
        {28, 55, 25, 21, 56},
        {27, 20, 39, 8, 14} };
    
    // Rotate left : 0b1001 -- > 0b0011
    inline uint64_t rol(uint64_t val, int r_bits)
    {
        return (r_bits == 0) ? val : (val << r_bits) | (val >> (64 - r_bits));
    }

    k_state keccak_round(const k_state& state_in, int round);
    k_state messageToState(const k_message& m);
    k_state state_XOR(const k_state& s, const k_message& m);
    k_state theta(const k_state& s);

    k_message message1, message2;
    k_state hash1, hash2;

public:
    NexusKeccak();
    NexusKeccak(const Int_array<uint64_t, 16>& m);
    void calculateHash();
    uint64_t getResult();
    k_1024 getHashResult();
    void setMessage(const Int_array<uint64_t, 16>& m);



};
#endif
