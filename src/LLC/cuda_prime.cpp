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

    PrimeCUDA::PrimeCUDA(uint32_t id)
    : Proof(id)
    , vWorkOrigins()
    , zPrimeOrigin()
    , zPrimorialMod()
    , zTempVar()
    , nCount(0)
    , nSieveIndex(0)
    , nTestIndex(0)
    , nBitArrayIndex(0)
    , nIterations(0)
    , nSievePrimes(0)
    , nSieveBits(0)
    , nMaxCandidates(0)
    , nDeviceThreads(0)
    , nTestLevel(0)
    {
        for(uint8_t i = 0; i < OFFSETS_MAX; ++i)
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
        uint32_t *nonce_meta = g_nonce_meta[nID];

        /* Get difficulty. */
        uint32_t nDifficulty = block.nBits;
        uint32_t nOrigins = vWorkOrigins.size();


        /* Check for early out. */
        if(fReset.load())
            return false;


        bool fSynchronize = false;


        /* Sieve bit array and compact test candidate nonces */
        if(cuda_primesieve(nID, nPrimorial, nPrimorialEndPrime, primeLimitA, primeLimitB, nSievePrimes, nSieveBits, nDifficulty,
                           nSieveIndex, nTestIndex, nOrigins, nMaxCandidates))
        {
            uint32_t nOriginIndex = nSieveIndex % nOrigins;

            bool fNewSieve = nOriginIndex == 0;
            bool fLastSieve = nBitArrayIndex == nSievesPerOrigin[nID] - 1;

            /* Determine if we should synchronize early due to running out of work. */
            fSynchronize = fNewSieve && fLastSieve;


            /* Determine if there should be a new test round. */
            bool fTestRound = nSieveIndex && nSieveIndex % nIterations == 0;

            /* After the number of iterations have been satisfied, start filling next queue */
            if (fTestRound || fSynchronize)
            {
                /* Test results. */
                cuda_fermat(nID, nSieveIndex, nTestIndex, nTestLevel, nMaxCandidates);
                ++nTestIndex;
            }

            /* Shift the working origin over by an entire sieve range. */
            vWorkOrigins[nOriginIndex] += nPrimorial * nSieveBits;

            ++nSieveIndex;
            SievedBits += nSieveBits;

            if(nSieveIndex % nOrigins == 0)
            {
                ++nBitArrayIndex;

                /* Compute prime remainders for each origin. */
                cuda_set_origins(nID, primeLimitA, vWorkOrigins.data(), vWorkOrigins.size());

                //debug::log(0, "cuda_set_origins ", nID, " ", nBitArrayIndex, " ", nSieveIndex);
            }
        }


        /* Obtain the final results and push them onto the queue */
        if(nTestIndex)
            cuda_results(nID, nTestIndex - 1, nonce_offsets, nonce_meta, &nCount, nPrimesChecked, nPrimesFound, fSynchronize);

        /* Add GPU sieve results to work queue */
        SendResults(nCount);

        /* Change frequency of looping for better GPU utilization, can lead to
        lower latency than from a calling thread waking a blocking-sync thread */
        runtime::sleep(1);

        /* Tell worker we want to request a new block, the prime origins have been exhausted. */
        if(fSynchronize)
        {
            debug::log(0, FUNCTION, (uint32_t)nID, " - Requesting more work");
            fReset = true;
        }

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

        /* Set GPU quit flag to true and ensure GPU is synchronized. */
        cuda_set_quit(1);
        cuda_device_synchronize();

        /* Initialize the stats counts for this GPU. */
        cuda_init_counts(nID);

        /* Set the GPU quit flag to false. */
        cuda_set_quit(0);

        /* Initialize the stats for this CPU. */
        nCount = 0;
        nSieveIndex = 0;
        nTestIndex = 0;
        nBitArrayIndex = 0;
        vWorkOrigins = vOrigins;

        for(uint8_t i = 0; i < OFFSETS_MAX; ++i)
        {
            nPrimesChecked[i] = 0;
            nPrimesFound[i] = 0;
        }

        /* Set the prime origin from the block hash. */
        mpz_import(zPrimeOrigin, 32, -1, sizeof(uint32_t), 0, 0, block.ProofHash().begin());

        /* Compute the primorial mod from the origin. */
        mpz_mod(zPrimorialMod, zPrimeOrigin, zPrimorial);
        mpz_sub(zPrimorialMod, zPrimorial, zPrimorialMod);
        mpz_add(zTempVar, zPrimeOrigin, zPrimorialMod);

        /* Compute base remainders and base origin. */
        std::vector<uint32_t> limbs = get_limbs(zTempVar);
        cuda_set_zTempVar(nID, (const uint64_t*)zTempVar[0]._mp_d);
        cuda_set_BaseOrigin(nID, &limbs[0]);
        cuda_base_remainders(nID, nSievePrimes);

        /* Compute prime remainders for each origin. */
        cuda_set_origins(nID, primeLimitA, vWorkOrigins.data(), vWorkOrigins.size());
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
        nMaxCandidates = 1 << nMaxCandidatesLog2[nID];
        nTestLevel = nTestLevels[nID];

        nDeviceThreads = cuda_device_threads(nID);

        debug::log(0, FUNCTION, nDeviceThreads, " CUDA Cores");
        debug::log(0, FUNCTION, "nIterations ", nIterations, " nSievePrimes ", nSievePrimes, " nSieveBits ", nSieveBits, " nMaxCandidates ", nMaxCandidates);

        /* Load the primes lists on the GPU device. */
        cuda_init_primes(nID, vOrigins.data(), primes, primesInverseInvk, nSievePrimes, nSieveBits, 32, nPrimorialEndPrime,
                         primeLimitA, vOrigins.size(), nMaxCandidates);


        /* Find prime limit B which is the first prime larger than bit array size (index 0 reserved for size). */
        primeLimitB = 0;
        for(int i = 1; i < nSievePrimes; ++i)
        {
            if(primes[i] > nSieveBits)
            {
                debug::log(0, i, ": First prime found larger than ", nSieveBits, " = ", primes[i]);
                primeLimitB = i;
                break;
            }
        }

        /* Set the primorial for this GPU device. */
        cuda_set_primorial(nID, nPrimorial);

        /* Load the sieve offsets configuration on the GPU device. */
        cuda_set_offset_patterns(nID, vOffsets, vOffsetsA, vOffsetsB, vOffsetsT);

        /* Allocate memory for offsets testing meta and bit array sieve. */
        g_nonce_offsets[nID] =   (uint64_t*)malloc(nMaxCandidates * sizeof(uint64_t));
        g_nonce_meta[nID] =      (uint32_t*)malloc(nMaxCandidates * sizeof(uint32_t));
        //g_bit_array_sieve[nID] = (uint32_t *)malloc(16 * (nSieveBits >> 5) * sizeof(uint32_t));


        /* Initialize the GMP objects. */
        mpz_init(zPrimeOrigin);
        mpz_init(zPrimorialMod);
        mpz_init(zTempVar);
    }

    void PrimeCUDA::Shutdown()
    {
        debug::log(3, FUNCTION, "PrimeCUDA", static_cast<uint32_t>(nID));

        /* Set GPU quit flag to true and ensure GPU is synchronized. */
        cuda_set_quit(1);
        cuda_device_synchronize();

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
        //free(g_bit_array_sieve[nID]);

        /* Free the GPU device memory and reset them. */
        cuda_free_primes(nID);
        cuda_free(nID);

        /* Free the GMP object memory. */
        mpz_clear(zPrimeOrigin);
        mpz_clear(zPrimorialMod);
        mpz_clear(zTempVar);
    }


    void PrimeCUDA::SendResults(uint32_t &count)
    {
        if (count)
        {
            /* Get thread local nonce offsets and meta. */
            uint64_t *nonce_offsets = g_nonce_offsets[nID];
            uint32_t *nonce_meta = g_nonce_meta[nID];

            /* Total up global stats from each device. */
            for(uint8_t i = 0; i < OFFSETS_MAX; ++i)
            {
                PrimesChecked[i] += nPrimesChecked[i];
                Tests_GPU += nPrimesChecked[i];
                PrimesFound[i] += nPrimesFound[i];
            }

            std::map<uint64_t, uint32_t> nonces;
            std::vector<std::pair<uint64_t, uint32_t> > dups;

            for(uint32_t i = 0; i < count; ++i)
            {
                auto it = nonces.find(nonce_offsets[i]);
                if(it != nonces.end())
                {
                    // Extract combo bits and chain info.
                    std::bitset<32> combo(it->second);
                    std::bitset<32> combo2(nonce_meta[i]);


                    debug::log(3, FUNCTION, std::hex, nonce_offsets[i], " existing:  ",
                        combo, " | duplicate: ", combo2);


                    dups.push_back(std::pair<uint64_t, uint32_t>(nonce_offsets[i], nonce_meta[i]));


                }
                else
                    nonces[nonce_offsets[i]] = nonce_meta[i];
            }


            std::vector<uint64_t> work_offsets;
            std::vector<uint32_t> work_meta;

            debug::log(3, FUNCTION, cuda_devicename(nID), "[", (uint32_t)nID, "] ",  dups.size(), "/", count, "(", (double)dups.size()/count * 100.0,"%) duplicates");

            for(auto it = nonces.begin(); it != nonces.end(); ++it)
            {
                work_offsets.push_back(it->first);
                work_meta.push_back(it->second);
            }

            {
                /* Atomic add nonces to work queue for testing. */
                std::unique_lock<std::mutex> lk(g_work_mutex);
                g_work_queue.emplace_back(work_info(work_offsets, work_meta, block, nID));
            }

            count = 0;
        }
    }




}
