/*__________________________________________________________________________________________

            (c) Hash(BEGIN(Satoshi[2010]), END(Sunny[2012])) == Videlicet[2014] ++

            (c) Copyright The Nexus Developers 2014 - 2019

            Distributed under the MIT software license, see the accompanying
            file COPYING or http://www.opensource.org/licenses/mit-license.php.

            "ad vocem populi" - To the Voice of the People

____________________________________________________________________________________________*/

#pragma once
#ifndef NEXUS_LLC_TYPES_CUDA_PRIME_H
#define NEXUS_LLC_TYPES_CUDA_PRIME_H

#include <LLC/types/proof.h>
#include <CUDA/include/macro.h>

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


    /** PrimeCUDA
     *
     *  Proof of Work Class for Prime Mining.
     *
     **/
    class PrimeCUDA : public Proof
    {
    public:

        PrimeCUDA(uint32_t id);
        virtual ~PrimeCUDA();


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

        /** SendResults
         *
         *
         *
         **/
        void SendResults(uint32_t &nCount);


    private:

        mpz_t zPrimeOrigin;
        mpz_t zPrimorialMod;
        mpz_t zTempVar;

        uint32_t nCount;
        uint32_t nPrimesChecked[OFFSETS_MAX];
        uint32_t nPrimesFound[OFFSETS_MAX];
        uint32_t nSieveIndex;
        uint32_t nTestIndex;

        uint32_t nIterations;
        uint32_t nSievePrimes;
        uint32_t nSieveBits;
        uint8_t nTestLevel;


    };

}

#endif
