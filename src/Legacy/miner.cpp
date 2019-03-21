/*******************************************************************************************

      Hash(BEGIN(Satoshi[2010]), END(Sunny[2012])) == Videlicet[2014] ++

 [Learn, Create, but do not Forge] Viz. http://www.opensource.org/licenses/mit-license.php
 [Scale Indefinitely]        BlackJack.

*******************************************************************************************/
#include "miner.h"
#include "prime.h"
#include "sleep.h"

#if defined(_MSC_VER)
  typedef int pid_t;
#else
  #include <sys/types.h>
  #include <sys/time.h>
  #include <sys/resource.h>
#endif

#include <functional>
#include <queue>
#include <atomic>

volatile uint32_t nDifficulty = 0;
extern std::atomic<bool> quit;

namespace LLP
{
    Miner::Miner(std::string ip, std::string port) : Outbound(ip, port){}

    void Miner::SetChannel(uint32_t nChannel)
    {
        Packet packet;
        packet.HEADER = SET_CHANNEL;
        packet.LENGTH = 4;
        packet.DATA   = uint2bytes(nChannel);

        this -> WritePacket(packet);
    }

    bool Miner::GetBlock(CBlock &block, uint8_t nTimeout)
    {
        Packet packet;
        packet.HEADER = GET_BLOCK;
        this -> WritePacket(packet);

        Packet RESPONSE = ReadNextPacket(nTimeout);

        if(RESPONSE.IsNull())
            return false;

      DeserializeBlock(block, RESPONSE.DATA);
        ResetPacket();

        return true;
    }

    uint32_t Miner::GetHeight(uint8_t nTimeout)
    {
        Packet packet;
        packet.HEADER = GET_HEIGHT;
        this -> WritePacket(packet);

        Packet RESPONSE = ReadNextPacket(nTimeout);

        if(RESPONSE.IsNull())
            return 0;

        uint32_t nHeight = bytes2uint(RESPONSE.DATA);
        ResetPacket();

        return nHeight;
    }

    uint8_t Miner::SubmitBlock(uint512 hashMerkleRoot, uint64 nNonce, uint8_t nTimeout)
    {
        Packet PACKET;
        PACKET.HEADER = SUBMIT_BLOCK;

        PACKET.DATA = hashMerkleRoot.GetBytes();
        std::vector<uint8_t> NONCE  = uint2bytes64(nNonce);

        PACKET.DATA.insert(PACKET.DATA.end(), NONCE.begin(), NONCE.end());
        PACKET.LENGTH = 72;

        this->WritePacket(PACKET);
        Packet RESPONSE = ReadNextPacket(nTimeout);
        if(RESPONSE.IsNull())
            return 0;

        ResetPacket();

        return RESPONSE.HEADER;
    }

    void Miner::DeserializeBlock(CBlock &BLOCK, std::vector<uint8_t> DATA)
    {
        BLOCK.nVersion      = bytes2uint(std::vector<uint8_t>(DATA.begin(), DATA.begin() + 4));

        BLOCK.hashPrevBlock.SetBytes (std::vector<uint8_t>(DATA.begin() + 4, DATA.begin() + 132));
        BLOCK.hashMerkleRoot.SetBytes(std::vector<uint8_t>(DATA.begin() + 132, DATA.end() - 20));

        BLOCK.nChannel      = bytes2uint(std::vector<uint8_t>(DATA.end() - 20, DATA.end() - 16));
        BLOCK.nHeight       = bytes2uint(std::vector<uint8_t>(DATA.end() - 16, DATA.end() - 12));
        BLOCK.nBits         = bytes2uint(std::vector<uint8_t>(DATA.end() - 12,  DATA.end() - 8));
        BLOCK.nNonce        = bytes2uint64(std::vector<uint8_t>(DATA.end() - 8,  DATA.end()));
    }

    /** Convert a 32 bit Unsigned Integer to Byte Vector using Bitwise Shifts. **/
    std::vector<uint8_t> Miner::uint2bytes(uint32_t UINT)
    {
      std::vector<uint8_t> BYTES(4, 0);
        BYTES[0] = UINT >> 24;
        BYTES[1] = UINT >> 16;
        BYTES[2] = UINT >> 8;
        BYTES[3] = UINT;

        return BYTES;
    }

    uint32_t Miner::bytes2uint(std::vector<uint8_t> BYTES, uint32_t nOffset)
    {
        return (BYTES[0 + nOffset] << 24) + (BYTES[1 + nOffset] << 16) +
               (BYTES[2 + nOffset] << 8)  +  BYTES[3 + nOffset];
    }

    /** Convert a 64 bit Unsigned Integer to Byte Vector using Bitwise Shifts. **/
    std::vector<uint8_t> Miner::uint2bytes64(uint64 UINT)
    {
        std::vector<uint8_t> INTS[2];
        INTS[0] = uint2bytes((uint32_t) UINT);
        INTS[1] = uint2bytes((uint32_t) (UINT >> 32));

        std::vector<uint8_t> BYTES;
        BYTES.insert(BYTES.end(), INTS[0].begin(), INTS[0].end());
        BYTES.insert(BYTES.end(), INTS[1].begin(), INTS[1].end());

        return BYTES;
    }

    uint64 Miner::bytes2uint64(std::vector<uint8_t> BYTES)
    {
        return (bytes2uint(BYTES) | ((uint64)bytes2uint(BYTES, 4) << 32));
    }
}

namespace Core
{
    static void affine_to_cpu(uint8_t id, uint8_t cpu)
    {
    #if defined(_MSC_VER)
        DWORD mask = 1 << cpu;
        SetThreadAffinityMask(GetCurrentThread(), mask);
    #else
        cpu_set_t set;

        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        sched_setaffinity(0, sizeof(&set), &set);
    #endif
    }

  /** Class to hold the basic data a Miner will use to build a Block.
    Used to allow one Connection for any amount of threads. **/
    MinerThreadGPU::MinerThreadGPU(uint8_t tid, uint8_t affinity)
        : THREAD(std::bind(&MinerThreadGPU::PrimeMiner, this))
        , threadIndex(tid)
        , threadAffinity(affinity)
        , fBlockFound(false)
        , fNewBlock(true)
        , fReady(false) { }

    MinerThreadGPU::~MinerThreadGPU()
    {
        THREAD.join();
    }


    /** Main Miner Thread. Bound to the class. **/
    void MinerThreadGPU::PrimeMiner()
    {
        /* all CUDA threads run on CPU core 0 + threadIndex */
        affine_to_cpu(threadIndex, threadAffinity);

        PrimeInit(threadIndex);
        fReady = true;

        while (true)
        {
            /* Keep thread at idle CPU usage if waiting to submit or recieve block. **/
            sleep_milliseconds(1);

            if (!(fNewBlock || fBlockFound))
            {
                nDifficulty = BLOCK.nBits;

                PrimeSieve(threadIndex,
                           BLOCK.GetPrimeOrigin(),
                           BLOCK.nBits,
                           BLOCK.nHeight,
                           BLOCK.hashMerkleRoot);

                fNewBlock = true;
            }

            if (quit.load())
                break;
        }
        PrimeFree(threadIndex);
    }

    MinerThreadCPU::MinerThreadCPU(uint8_t tid, uint8_t affinity)
        : THREAD(std::bind(&MinerThreadCPU::PrimeMiner, this))
        , threadIndex(tid)
        , threadAffinity(affinity) { }

    MinerThreadCPU::~MinerThreadCPU()
    {
        THREAD.join();
    }

    /** Main Miner Thread. Bound to the class. **/
    void MinerThreadCPU::PrimeMiner()
    {
        affine_to_cpu(threadIndex, threadAffinity);

        while (true)
        {
            if (!PrimeQuery())
                sleep_milliseconds(100);


            if (quit.load())
                break;
        }
    }
}
