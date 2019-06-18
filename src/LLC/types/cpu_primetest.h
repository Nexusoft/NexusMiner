/*__________________________________________________________________________________________

            (c) Hash(BEGIN(Satoshi[2010]), END(Sunny[2012])) == Videlicet[2014] ++

            (c) Copyright The Nexus Developers 2014 - 2019

            Distributed under the MIT software license, see the accompanying
            file COPYING or http://www.opensource.org/licenses/mit-license.php.

            "ad vocem populi" - To the Voice of the People

____________________________________________________________________________________________*/

#pragma once
#ifndef NEXUS_LLC_TYPES_CPU_PRIMETEST_H
#define NEXUS_LLC_TYPES_CPU_PRIMETEST_H

#include <LLC/types/proof.h>
#include <LLC/include/global.h>
#include <LLC/prime/prime.h>
#include <LLC/prime/prime2.h>

#include <cstdint>
#include <map>

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
    class PrimeTestCPU : public Proof
    {
    public:

        PrimeTestCPU(uint32_t id);
        virtual ~PrimeTestCPU();

        /** Channel
         *
         *
         *
         **/
        virtual uint32_t Channel() override { return 1; }


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

        mpz_t zTempVar;
        mpz_t zN;
        mpz_t zBaseOffsetted;
        mpz_t zBaseOrigin;
        mpz_t zPrimeOrigin;
        mpz_t zResidue;
        mpz_t zPrimorialMod;

        work_info work;

        std::map<uint32_t, uint32_t> mapTest;
        uint32_t gpu_begin;
        uint32_t gpu_end;

    };
}

#endif
