/*__________________________________________________________________________________________

            (c) Hash(BEGIN(Satoshi[2010]), END(Sunny[2012])) == Videlicet[2014] ++

            (c) Copyright The Nexus Developers 2014 - 2019

            Distributed under the MIT software license, see the accompanying
            file COPYING or http://www.opensource.org/licenses/mit-license.php.

            "ad vocem populi" - To the Voice of the People

____________________________________________________________________________________________*/

#pragma once
#ifndef NEXUS_LLC_TYPES_CPU_PRIMESIEVE_H
#define NEXUS_LLC_TYPES_CPU_PRIMESIEVE_H

#include <LLC/types/proof.h>
#include <LLC/include/global.h>
#include <LLC/prime/prime.h>
#include <LLC/prime/prime2.h>

#include <vector>
#include <cstdint>

#if defined(_MSC_VER)
#include <mpir.h>
#else
#include <gmp.h>
#endif

/* Forward declared. */
namespace TAO
{
    namespace Ledger
    {
        class Block;
    }
}


namespace LLC
{
    /** CUDAPrime
     *
     *  Proof of Work Class for Prime Mining.
     *
     **/
    class PrimeSieveCPU : public Proof
    {
    public:
        PrimeSieveCPU(uint8_t id, TAO::Ledger::Block *block);
        virtual ~PrimeSieveCPU();

        /** Channel
         *
         *
         *
         **/
        static uint32_t Channel() { return 1; }

        /** Work
         *
         *
         *
         **/
        virtual bool Work() override;


        /** Load
         *
         *
         *
         **/
        virtual void Load() override;


        /** Init
         *
         *
         *
         **/
        virtual void Init() override;


        /** Shutdown
         *
         *
         *
         **/
        virtual void Shutdown() override;


    private:

        void sieve_offset(uint64_t base_offsetted, uint32_t i, uint32_t o);


    private:

        std::vector<uint32_t> vBaseRemainders;
        uint32_t nBitArraySize;
        uint32_t *pBitArraySieve;
        uint32_t nSieveIndex;
        mpz_t zPrimeOrigin;
        mpz_t zPrimorialMod;
        mpz_t zTempVar;

    };
}

#endif
