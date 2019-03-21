#ifndef NEXUS_LLP_MINER_H
#define NEXUS_LLP_MINER_H

#include "outbound.h"
#include "block.h"
#include <thread>
#include <stdint.h>

namespace LLP
{
    class Miner : public Outbound
    {
    public:
        Miner(std::string ip, std::string port);

        enum
        {
          /** DATA PACKETS **/
            BLOCK_DATA   = 0,
            SUBMIT_BLOCK = 1,
            BLOCK_HEIGHT = 2,
            SET_CHANNEL  = 3,

            /** REQUEST PACKETS **/
            GET_BLOCK    = 129,
            GET_HEIGHT   = 130,

            /** RESPONSE PACKETS **/
            GOOD     = 200,
            FAIL     = 201,

            /** GENERIC **/
            PING     = 253,
            CLOSE    = 254
        };

        void SetChannel(uint32_t nChannel);

        bool GetBlock(CBlock &block, uint8_t nTimeout = 30);

        uint32_t GetHeight(uint8_t nTimeout = 30);

        uint8_t SubmitBlock(uint512 hashMerkleRoot, uint64 nNonce, uint8_t nTimeout = 30);

    private:
        void DeserializeBlock(CBlock &BLOCK, std::vector<uint8_t> DATA);

        /** Convert a 32 bit Unsigned Integer to Byte Vector using Bitwise Shifts. **/
        std::vector<uint8_t> uint2bytes(uint32_t UINT);

        uint32_t bytes2uint(std::vector<uint8_t> BYTES, uint32_t nOffset = 0);

        /** Convert a 64 bit Unsigned Integer to Byte Vector using Bitwise Shifts. **/
        std::vector<uint8_t> uint2bytes64(uint64 UINT);

        uint64 bytes2uint64(std::vector<uint8_t> BYTES);
    };
}

namespace Core
{
    /** Class to hold the basic data a Miner will use to build a Block.
    Used to allow one Connection for any amount of threads. **/
    class MinerThreadGPU
    {
    public:
        LLP::CBlock BLOCK;
        std::thread THREAD;
        uint8_t threadIndex;
        uint8_t threadAffinity;
        bool fBlockFound;
        bool fNewBlock;
        bool fReady;

        MinerThreadGPU(uint8_t tid, uint8_t affinity);
        ~MinerThreadGPU();

        void PrimeMiner();

    };

    class MinerThreadCPU
    {
    public:
        std::thread THREAD;
        uint8_t threadIndex;
        uint8_t threadAffinity;

        MinerThreadCPU(uint8_t tid, uint8_t affinity);
        ~MinerThreadCPU();

        void PrimeMiner();
    };

}

#endif
