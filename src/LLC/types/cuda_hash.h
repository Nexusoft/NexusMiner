/*__________________________________________________________________________________________

            (c) Hash(BEGIN(Satoshi[2010]), END(Sunny[2012])) == Videlicet[2014] ++

            (c) Copyright The Nexus Developers 2014 - 2019

            Distributed under the MIT software license, see the accompanying
            file COPYING or http://www.opensource.org/licenses/mit-license.php.

            "ad vocem populi" - To the Voice of the People

____________________________________________________________________________________________*/

#pragma once
#ifndef NEXUS_LLC_TYPES_CUDA_HASH_H
#define NEXUS_LLC_TYPES_CUDA_HASH_H

#include <LLC/types/uint1024.h>
#include <LLC/types/proof.h>
#include <cstdint>

#if defined(_MSC_VER)
#include <mpir.h>
#else
#include <gmp.h>
#endif


namespace LLC
{


    /** HashCUDA
     *
     *  Proof of Work Class for Prime Mining.
     *
     **/
    class HashCUDA : public Proof
    {
    public:

        HashCUDA(uint32_t id);
        virtual ~HashCUDA();


        /** Channel
         *
         *
         *
         **/
        virtual uint32_t Channel() override { return 2; }


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

        uint1024_t nTarget;
        uint64_t nHashes;
        uint32_t nIntensity;
        uint32_t nThroughput;
        uint32_t nThreadsPerBlock;



    };

}

#endif
