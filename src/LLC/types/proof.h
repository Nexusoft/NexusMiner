/*__________________________________________________________________________________________

            (c) Hash(BEGIN(Satoshi[2010]), END(Sunny[2012])) == Videlicet[2014] ++

            (c) Copyright The Nexus Developers 2014 - 2019

            Distributed under the MIT software license, see the accompanying
            file COPYING or http://www.opensource.org/licenses/mit-license.php.

            "ad vocem populi" - To the Voice of the People

____________________________________________________________________________________________*/

#pragma once
#ifndef NEXUS_LLP_TEMPLATES_PROOF_H
#define NEXUS_LLP_TEMPLATES_PROOF_H

#include <cstdint>
#include <atomic>
#include <string>

namespace TAO
{
    namespace Ledger
    {
        class Block;
    }
}


namespace LLC
{

    /** Proof
     *
     *  Abstract Virtual Class for Proof of Work.
     *
     **/
    class Proof
    {
    public:
        Proof(uint8_t id, TAO::Ledger::Block *block)
        : pBlock(block)
        , nID(id)
        , fReset(false)
        {
        }

        virtual ~Proof()
        {
        }


        /** Work
         *
         *
         *
         **/
        virtual bool Work() = 0;


        /** Load
         *
         *
         *
         **/
        virtual void Load() = 0;


        /** Init
         *
         *
         *
         **/
        virtual void Init() = 0;


        /** Shutdown
         *
         *
         *
         **/
        virtual void Shutdown() = 0;


        /** Reset
         *
         *
         *
         **/
         virtual void Reset()
         {
             fReset = true;
         }


    protected:
        TAO::Ledger::Block *pBlock;
        uint8_t nID;
        std::atomic<bool> fReset;
    };


}

#endif
