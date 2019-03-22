/*__________________________________________________________________________________________

            (c) Hash(BEGIN(Satoshi[2010]), END(Sunny[2012])) == Videlicet[2014] ++

            (c) Copyright The Nexus Developers 2014 - 2019

            Distributed under the MIT software license, see the accompanying
            file COPYING or http://www.opensource.org/licenses/mit-license.php.

            "ad vocem populi" - To the Voice of the People

____________________________________________________________________________________________*/

#include <CUDA/include/util.h>
//#include <CUDA/include/frame_resources.h>
#include <CUDA/include/sk1024.h>

#include <LLC/include/global.h>
#include <LLC/types/cuda_hash.h>
#include <LLC/types/bignum.h>


#include <TAO/Ledger/types/block.h>
#include <TAO/Ledger/include/difficulty.h>

#include <Util/include/runtime.h>
#include <Util/include/args.h>
#include <Util/include/debug.h>


namespace LLC
{

    HashCUDA::HashCUDA(uint8_t id, TAO::Ledger::Block *block)
    : Proof(id, block)
    , nTarget()
    , nIntensity(0)
    {
    }

    HashCUDA::~HashCUDA()
    {
    }

    /* The main proof of work function. */
    bool HashCUDA::Work()
    {
        /* Check for early out. */
		if (fReset.load())
            return false;

		uint64_t hashes = 0;


        /* Get the data for this block. */
		uint32_t *pData = reinterpret_cast<uint32_t *>(&pBlock->nVersion);

        /* Calculate threads per block. */
        uint32_t nThreadsPerBlock = 896;

        /* Calcluate the throughput for the cuda hash mining. */
        uint32_t nThroughput = 256 * nThreadsPerBlock * nIntensity;


        /* Do hashing on a CUDA device. */
		bool fFound = cuda_sk1024_hash(
            nID,
            pData,
            nTarget,
            pBlock->nNonce,
            &hashes,
            nThroughput,
            nThreadsPerBlock,
            pBlock->nHeight);

        /* Increment number of hashes for this round. */
        if (hashes < 0x0000FFFFFFFFFFFF)
			LLC::nHashes += hashes;

        /* If a nonce with the right diffulty was found, return true and submit block. */
		if(fFound)
        {
            /* Calculate the number of leading zero-bits and display. */
            uint32_t nBits = pBlock->ProofHash().BitCount();
            uint32_t nLeadingZeroes = 1024 - nBits - 1;
            debug::log(0, "[MASTER] Found Hash Block with ",
                nLeadingZeroes, " Leading Zero-Bits");

            fReset = true;
            return true;
        }

        return false;
    }

    void HashCUDA::Init()
    {
        debug::log(2, FUNCTION, "HashCUDA", static_cast<uint32_t>(nID));
        fReset = false;

        /* Set the block for this device */
        cuda_sk1024_setBlock(&pBlock->nVersion, pBlock->nHeight);

        /* Get the target difficulty. */
		CBigNum target;
		target.SetCompact(pBlock->nBits);
        nTarget = target.getuint1024();

        /* Set the target hash on this device for the difficulty. */
        cuda_sk1024_set_Target((uint64_t *)nTarget.data());

        uint32_t nHeight = pBlock->nHeight;


    }

    void HashCUDA::Load()
    {
        debug::log(2, FUNCTION, "HashCUDA", static_cast<uint32_t>(nID));

        /* Initialize the cuda device associated with this ID. */
        cuda_init(nID);

        /* Allocate memory associated with Device Hashing. */
        cuda_sk1024_init(nID);

        /* Compute the intensity by determining number of multiprocessors. */
        nIntensity = 2 * cuda_device_multiprocessors(nID);
        debug::log(0, cuda_devicename(nID), " intensity set to ", nIntensity);
    }

    void HashCUDA::Shutdown()
    {
        debug::log(2, FUNCTION, "HashCUDA", static_cast<uint32_t>(nID));
        fReset = true;

        /* Free the GPU device memory associated with hashing. */
        cuda_sk1024_free(nID);

        /* Free the GPU device memory and reset them. */
        cuda_free(nID);
    }


}
