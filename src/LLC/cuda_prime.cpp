/*__________________________________________________________________________________________

            (c) Hash(BEGIN(Satoshi[2010]), END(Sunny[2012])) == Videlicet[2014] ++

            (c) Copyright The Nexus Developers 2014 - 2019

            Distributed under the MIT software license, see the accompanying
            file COPYING or http://www.opensource.org/licenses/mit-license.php.

            "ad vocem populi" - To the Voice of the People

____________________________________________________________________________________________*/

#include <CUDA/include/sieve.h>
#include <CUDA/include/test.h>
#include <CUDA/include/util.h>
#include <CUDA/include/frame_resources.h>

#include <LLC/include/global.h>
#include <LLC/prime/prime.h>
#include <LLC/types/cuda_prime.h>

#include <TAO/Ledger/types/block.h>

#include <Util/include/runtime.h>
#include <Util/include/args.h>
#include <Util/include/debug.h>
#include <Util/include/prime_config.h>

#include <cmath>
#include <bitset>

/*
namespace
{
    uint64_t range = 0;
    uint64_t gpu_offset = 0;

}
*/

namespace LLC
{

    std::vector<uint32_t> get_limbs(const mpz_t &big_int)
    {
      std::vector<uint32_t> limbs(WORD_MAX);
      mpz_export(&limbs[0], 0, -1, sizeof(uint32_t), 0, 0, big_int);
      return limbs;
    }

    PrimeCUDA::PrimeCUDA(uint8_t id, TAO::Ledger::Block *block)
    : Proof(id, block)
    , zPrimeOrigin()
    , zFirstSieveElement()
    , zPrimorialMod()
    , zTempVar()
    , nCount(0)
    , nSieveIndex(0)
    , nTestIndex(0)
    , nIterations(0)
    , nSievePrimes(0)
    , nSieveBits(0)
    , nTestLevel(0)
    {
        for(uint8_t i = 0; i < 16; ++i)
        {
            nPrimesChecked[i] = 0;
            nPrimesFound[i] = 0;
        }
    }

    PrimeCUDA::~PrimeCUDA()
    {
    }

    /* The main proof of work function. */
    bool PrimeCUDA::Work()
    {

        /* Get thread local nonce offsets and meta. */
        uint64_t *nonce_offsets = g_nonce_offsets[nID];
        uint64_t *nonce_meta = g_nonce_meta[nID];

        /* Get difficulty. */
        uint32_t nDifficulty = pBlock->nBits;


        /* Compute non-colliding origins for each GPU within 2^64 search space */
        //uint64_t range = ~(0) / nPrimorial / GPU_MAX;
        //gpu_offset = base_offset;// + range * nPrimorial * nID;

        /* Check for early out. */
        if(fReset.load())
        {
            cuda_set_quit(1);
            return false;
        }

        /* Sieve bit array and compact test candidate nonces */
        if(cuda_primesieve(nID, base_offset, nPrimorial,
                        nPrimorialEndPrime, primeLimitA, nSievePrimes,
                        nSieveBits, nDifficulty, nSieveIndex, nTestIndex))
        {
            /* Check for early out. */
            if(fReset.load())
            {
                cuda_set_quit(1);
                return false;
            }

            /* After the number of iterations have been satisfied,
             * start filling next queue */
            if (nSieveIndex % nIterations == 0 && nSieveIndex > 0)
            {
                /* Test results. */
                cuda_fermat(nID, nSieveIndex, nTestIndex, nTestLevel);
                ++nTestIndex;
            }

            ++nSieveIndex;
            SievedBits += nSieveBits;
        }

        /* Check for early out. */
        if(fReset.load())
        {
            cuda_set_quit(1);
            return false;
        }

        /* Obtain the final results and push them onto the queue */
        if(nTestIndex)
        {
            cuda_results(nID, nTestIndex-1, nonce_offsets, nonce_meta,
                &nCount, nPrimesChecked, nPrimesFound);
        }


        /* Total up global stats from each device. */
        for(uint8_t i = 0; i < 16; ++i)
        {
            PrimesChecked[i] += nPrimesChecked[i];
            Tests_GPU += nPrimesChecked[i];
            PrimesFound[i] += nPrimesFound[i];
        }

        /* Check for early out. */
        if(fReset.load())
        {
            cuda_set_quit(1);
            return false;
        }


        /* Add GPU sieve results to work queue */
        SendResults(nCount);


        /* Change frequency of looping for better GPU utilization, can lead to
        lower latency than from a calling thread waking a blocking-sync thread */
        runtime::sleep(1);

        return false;
    }

    void PrimeCUDA::Init()
    {
        debug::log(3, FUNCTION, "PrimeCUDA", static_cast<uint32_t>(nID));

        /* Atomic set reset flag to false. */
        fReset = false;

        /* Clear the work queue for this round. */
        {
            std::unique_lock<std::mutex> lk(g_work_mutex);
            g_work_queue.clear();
        }

        /* Initialize the stats counts for this GPU. */
        cuda_init_counts(nID);

        /* Initialize the stats for this CPU. */
        nCount = 0;
        nSieveIndex = 0;
        nTestIndex = 0;
        for(uint8_t i = 0; i < 16; ++i)
        {
            nPrimesChecked[i] = 0;
            nPrimesFound[i] = 0;
        }

        /* Compute non-colliding origins for each GPU within 2^64 search space */
        //uint64_t range = ~(0) / nPrimorial / GPU_MAX;
        //uint64_t gpu_offset = base_offset + range * nPrimorial * nID;

        /* Set the prime origin from the block hash. */
        mpz_import(zPrimeOrigin, 32, -1, sizeof(uint32_t), 0, 0, pBlock->ProofHash().data());


        /* Compute the primorial mod from the origin. */
        mpz_mod(zPrimorialMod, zPrimeOrigin, zPrimorial);
        mpz_sub(zPrimorialMod, zPrimorial, zPrimorialMod);
        mpz_add(zTempVar, zPrimeOrigin, zPrimorialMod);

        /* Compute base remainders. */
        cuda_set_zTempVar(nID, (const uint64_t*)zTempVar[0]._mp_d);
        cuda_base_remainders(nID, nSievePrimes);


        /* Compute first sieving element. */
        mpz_add_ui(zFirstSieveElement, zTempVar, base_offset);


        /* Convert the first sieve element and set it on GPU. */
        std::vector<uint32_t> limbs = get_limbs(zFirstSieveElement);
        cuda_set_FirstSieveElement(nID, &limbs[0]);


        /* Set the sieving primes for the first bucket (rest computed on the fly) */
        //cuda_set_sieve(nID, base_offset, nPrimorial,
        //               primeLimitA, prime_limit, sieve_bits_log2);


        /* Set the GPU quit flag to false. */
        cuda_set_quit(0);

    }

    void PrimeCUDA::Load()
    {
        debug::log(3, FUNCTION, "PrimeCUDA", static_cast<uint32_t>(nID));

        /* Initialize the cuda device associated with this ID. */
        cuda_init(nID);

        /* Get config parameters for this device index */
        nIterations = 1 << nSieveIterationsLog2[nID];
        nSievePrimes = 1 << nSievePrimesLog2[nID];
        nSieveBits =  1 << nSieveBitsLog2[nID];
        nTestLevel = nTestLevels[nID];

        /* Load the primes lists on the GPU device. */
        cuda_init_primes(nID, primes, primesInverseInvk, nSievePrimes,
        nSieveBits, 32,
        nPrimorialEndPrime, primeLimitA);

        /* Set the primorial for this GPU device. */
        cuda_set_primorial(nID, nPrimorial);

        /* Load the sieve offsets configuration on the GPU device. */
        cuda_set_offset_patterns(nID, vOffsets, vOffsetsA, vOffsetsB, vOffsetsT);

        /* Allocate memory for offsets testing meta and bit array sieve. */
        g_nonce_offsets[nID] =   (uint64_t*)malloc(OFFSETS_MAX * sizeof(uint64_t));
        g_nonce_meta[nID] =      (uint64_t*)malloc(OFFSETS_MAX * sizeof(uint64_t));
        g_bit_array_sieve[nID] = (uint32_t *)malloc(16 * (nSieveBits >> 5) * sizeof(uint32_t));

        /* Initialize the GMP objects. */
        mpz_init(zPrimeOrigin);
        mpz_init(zFirstSieveElement);
        mpz_init(zPrimorialMod);
        mpz_init(zTempVar);

    }

    void PrimeCUDA::Shutdown()
    {
        debug::log(3, FUNCTION, "PrimeCUDA", static_cast<uint32_t>(nID));

        /* Set the GPU quit flag to true. */
        cuda_set_quit(1);

        /* Atomic set reset flag to false. */
        fReset = true;

        /* Clear the work queue for this round. */
        {
            std::unique_lock<std::mutex> lk(g_work_mutex);
            g_work_queue.clear();
        }

        /* Free the global nonce and sieve memory. */
        free(g_nonce_offsets[nID]);
        free(g_nonce_meta[nID]);
        free(g_bit_array_sieve[nID]);

        /* Free the GPU device memory and reset them. */
        cuda_free_primes(nID);
        cuda_free(nID);

        /* Free the GMP object memory. */
        mpz_clear(zPrimeOrigin);
        mpz_clear(zFirstSieveElement);
        mpz_clear(zPrimorialMod);
        mpz_clear(zTempVar);
    }


    void PrimeCUDA::SendResults(uint32_t &count)
    {
        if (count)
        {
            /* Get thread local nonce offsets and meta. */
            uint64_t *nonce_offsets = g_nonce_offsets[nID];
            uint64_t *nonce_meta = g_nonce_meta[nID];

            std::map<uint64_t, uint64_t> nonces;
            std::vector<std::pair<uint64_t, uint64_t> > dups;

            for(uint32_t i = 0; i < count; ++i)
            {
                auto it = nonces.find(nonce_offsets[i]);
                if(it != nonces.end())
                {
                    uint64_t meta = it->second;
                    // Extract combo bits and chain info.
                    std::bitset<32> combo(meta >> 32);
                    uint32_t chain_offset_beg = (meta >> 24) & 0xFF;
                    uint32_t chain_offset_end = (meta >> 16) & 0xFF;
                    uint32_t chain_length = meta & 0xFF;

                    uint64_t meta2 = nonce_meta[i];
                    // Extract combo bits and chain info.
                    std::bitset<32> combo2(meta2 >> 32);
                    uint32_t chain_offset_beg2 = (meta2 >> 24) & 0xFF;
                    uint32_t chain_offset_end2 = (meta2 >> 16) & 0xFF;
                    uint32_t chain_length2 = meta2 & 0xFF;

                    debug::log(3, FUNCTION, std::hex, nonce_offsets[i], " existing:  ",
                        " combo: ", combo,
                        " beg: ", std::dec, chain_offset_beg,
                        " end: ", chain_offset_end,
                        " len: ", chain_length,
                        " | duplicate: ", std::hex,
                            " combo: ", combo2,
                            " beg: ", std::dec, chain_offset_beg2,
                            " end: ", chain_offset_end2,
                            " len: ", chain_length2);



                    dups.push_back(std::pair<uint64_t, uint64_t>(nonce_offsets[i], nonce_meta[i]));


                }
                else
                nonces[nonce_offsets[i]] = nonce_meta[i];
            }


            std::vector<uint64_t> work_offsets;
            std::vector<uint64_t> work_meta;


            debug::log(3, FUNCTION, cuda_devicename(nID), "[", (uint32_t)nID, "] ",  dups.size(), "/", count, "(", (double)dups.size()/count * 100.0,"%) duplicates");

            /*for(uint32_t i = 0; i < dups.size(); ++i)
            {
                uint64_t meta = dups[i].second;
                // Extract combo bits and chain info.
                uint32_t combo = meta >> 32;
                uint32_t chain_offset_beg = (meta >> 24) & 0xFF;
                uint32_t chain_offset_end = (meta >> 16) & 0xFF;
                uint32_t chain_length = meta & 0xFF;

                debug::log(3, FUNCTION, "nonce: ", std::hex, dups[i].first, std::dec,
                    " combo: ", combo,
                    " beg: ", chain_offset_beg,
                    " end: ", chain_offset_end,
                    " len: ", chain_length);
            } */


            for(auto it = nonces.begin(); it != nonces.end(); ++it)
            {
                work_offsets.push_back(it->first);
                work_meta.push_back(it->second);
            }

            {
                /* Atomic add nonces to work queue for testing. */
                std::unique_lock<std::mutex> lk(g_work_mutex);
                g_work_queue.emplace_back(work_info(work_offsets, work_meta, pBlock, nID));
            }

            count = 0;
        }
    }




}
