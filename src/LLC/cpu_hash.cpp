/*__________________________________________________________________________________________

            (c) Hash(BEGIN(Satoshi[2010]), END(Sunny[2012])) == Videlicet[2014] ++

            (c) Copyright The Nexus Developers 2014 - 2019

            Distributed under the MIT software license, see the accompanying
            file COPYING or http://www.opensource.org/licenses/mit-license.php.

            "ad vocem populi" - To the Voice of the People

____________________________________________________________________________________________*/

#include <LLC/include/global.h>
#include <LLC/types/bignum.h>
#include <LLC/types/cpu_hash.h>

#include <TAO/Ledger/types/block.h>
#include <TAO/Ledger/include/difficulty.h>

#include <Util/include/runtime.h>
#include <Util/include/args.h>
#include <Util/include/debug.h>

namespace LLC
{

    HashCPU::HashCPU(uint8_t id, TAO::Ledger::Block *block)
    : Proof(id, block)
    , nTarget()
    , nIntensity(0)
    {
    }

    HashCPU::~HashCPU()
    {
    }

    /* The main proof of work function. */
    bool HashCPU::Work()
    {
        /* Check for early out. */
		if (fReset.load())
            return false;

        /* Do hashing on CPU. */
		uint64_t hashes = 0;
        uint64_t throughput = 1 << 13;
        bool fFound = false;
        for(uint1024_t i =0; i < throughput; ++i)
        {
            if(fReset.load())
                return false;

            if(pBlock->ProofHash() < nTarget)
            {
                fFound = true;
                ++hashes;
                break;
            }
            ++pBlock->nNonce;
            ++hashes;
        }

        /* Increment number of hashes for this round. */
		LLC::nHashes += hashes;

        /* If a nonce with the right diffulty was found, return true and submit block. */
		if(fFound)
        {
            /* Calculate the number of leading zero-bits and display. */
            uint32_t nBits = pBlock->ProofHash().BitCount();
            uint32_t nLeadingZeroes = 1024 - nBits;
            debug::log(0, "[MASTER] Found Hash Block with ",
                nLeadingZeroes, " Leading Zero-Bits");

            fReset = true;
            return true;
        }

        return false;
    }

    void HashCPU::Init()
    {
        debug::log(3, FUNCTION, "HashCPU", static_cast<uint32_t>(nID));
        fReset = false;

        /* Get the target difficulty. */
		CBigNum target;
		target.SetCompact(pBlock->nBits);
        nTarget = target.getuint1024();

        debug::log(1, "Target ", nTarget.ToString().substr(0, 20));

    }

    void HashCPU::Load()
    {
        debug::log(3, FUNCTION, "HashCPU", static_cast<uint32_t>(nID));
    }

    void HashCPU::Shutdown()
    {
        debug::log(3, FUNCTION, "HashCPU", static_cast<uint32_t>(nID));
        fReset = true;
    }


}
