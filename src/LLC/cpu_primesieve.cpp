/*__________________________________________________________________________________________

            (c) Hash(BEGIN(Satoshi[2010]), END(Sunny[2012])) == Videlicet[2014] ++

            (c) Copyright The Nexus Developers 2014 - 2019

            Distributed under the MIT software license, see the accompanying
            file COPYING or http://www.opensource.org/licenses/mit-license.php.

            "ad vocem populi" - To the Voice of the People

____________________________________________________________________________________________*/

#include <LLC/include/global.h>
#include <LLC/types/cpu_primesieve.h>

#include <TAO/Ledger/types/block.h>

#include <Util/include/runtime.h>
#include <Util/include/debug.h>
#include <Util/include/print_colors.h>
#include <Util/include/prime_config.h>

#include <iomanip>

namespace LLC
{
    PrimeSieveCPU::PrimeSieveCPU(uint8_t id, TAO::Ledger::Block *block)
    : Proof(id, block)
    , vBaseRemainders()
    , nBitArraySize(1 << 23)
    , pBitArraySieve(nullptr)
    , nSieveIndex(0)
    , zPrimeOrigin()
    , zPrimorialMod()
    , zTempVar()
    {
    }

    PrimeSieveCPU::~PrimeSieveCPU()
    {
    }

    bool PrimeSieveCPU::Work()
    {
        std::vector<uint64_t> vNonces;
        std::vector<uint64_t> vMeta;

        /* Clear the bit array. */
        memset(pBitArraySieve, 0x00, nBitArraySize>>5);

        uint64_t primorial_start = (uint64_t)nBitArraySize * (uint64_t)nSieveIndex;
        uint64_t base_offsetted = (uint64_t)base_offset + (uint64_t)nPrimorial * primorial_start;

        /* Loop through each sieving prime and sieve. */
        for(uint32_t i = nPrimorialEndPrime; i < nSievePrimeLimit && !fReset.load(); ++i)
        {
            /* Get the global prime and inverse. */
            uint64_t p =   primesInverseInvk[i * 4 + 0];
            uint32_t inv = primesInverseInvk[i * 4 + 1];

            /* Sieve with the offsets. */
            for(uint8_t j = 0; j < 6; ++j)
                sieve_offset(base_offsetted, p, inv, vBaseRemainders[i], offsetsA[j]);

        }

        /* Add nonce offsets to queue. */
        for(uint32_t i = 0; i < nBitArraySize; ++i)
        {
            /* Make sure this offset survived the sieve. */
            if(pBitArraySieve[i >> 5] & (1 << (i & 31)))
                continue;

            /* Compute the global nonce index. */
            vNonces.push_back((uint64_t)i + primorial_start);
            vMeta.push_back(0);
        }


        {
            /* Atomic add nonces to work queue for testing. */
            std::unique_lock<std::mutex> lk(g_work_mutex);
            g_work_queue.emplace_back(work_info(vNonces, vMeta, pBlock, nID));
        }

        /* Increment the sieve index. */
        ++nSieveIndex;
        SievedBits += nBitArraySize;

        return false;
    }

    void PrimeSieveCPU::Load()
    {
        debug::log(2, FUNCTION, "PrimeSieveCPU", static_cast<uint32_t>(nID));

        /* Initialize the GMP objects. */
        mpz_init(zPrimeOrigin);
        mpz_init(zPrimorialMod);
        mpz_init(zTempVar);

        /* Create the bit array sieve. */
        pBitArraySieve = (uint32_t *)malloc((nBitArraySize >> 5) * sizeof(uint32_t));
    }

    void PrimeSieveCPU::Init()
    {
        debug::log(2, FUNCTION, "PrimeSieveCPU", static_cast<uint32_t>(nID));

        /* Atomic set reset flag to false. */
        fReset = false;

        nSieveIndex = 0;

        /* Clear the work queue for this round. */
        {
            std::unique_lock<std::mutex> lk(g_work_mutex);
            g_work_queue.clear();
        }

        /* Set the prime origin from the block hash. */
        mpz_import(zPrimeOrigin, 32, -1, sizeof(uint32_t), 0, 0, pBlock->ProofHash().data());


        /* Compute the primorial mod from the origin. */
        mpz_mod(zPrimorialMod, zPrimeOrigin, zPrimorial);
        mpz_sub(zPrimorialMod, zPrimorial, zPrimorialMod);
        mpz_add(zTempVar, zPrimeOrigin, zPrimorialMod);


        /* Compute the base remainders. */
        vBaseRemainders.clear();
        for(uint32_t i = 0; i < nSievePrimeLimit; ++i)
        {
            /* Get the global prime and inverse. */
            uint64_t p =   primesInverseInvk[i * 4 + 0];
            uint32_t inv = primesInverseInvk[i * 4 + 1];

            uint32_t base_remainder = mpz_tdiv_ui(zTempVar, p);

            /* Compute the base remainder. */
            vBaseRemainders.emplace_back(base_remainder);
        }
    }

    void PrimeSieveCPU::Shutdown()
    {
        debug::log(2, FUNCTION, "PrimeSieveCPU", static_cast<uint32_t>(nID));

        /* Atomic set reset flag to true. */
        fReset = true;

        /* Clear the work queue for this round. */
        {
            std::unique_lock<std::mutex> lk(g_work_mutex);
            g_work_queue.clear();
        }

        /* Free the GMP object memory. */
        mpz_clear(zPrimeOrigin);
        mpz_clear(zPrimorialMod);
        mpz_clear(zTempVar);

        /* Free the bit array sieve memory. */
        free(pBitArraySieve);
    }


    void PrimeSieveCPU::sieve_offset(uint64_t origin, uint64_t p, uint32_t inv, uint32_t base_rem, uint32_t o)
    {
        uint64_t rem = origin + base_rem + o;

        if(p < rem)
            rem -= p;

        uint64_t r = (p - rem) * inv;
        uint32_t index = r % p;

        while (index < nBitArraySize)
        {
            pBitArraySieve[index>>5] |= (1<<(index & 31));
            index += p;
        }
    }

}
