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

#include <TAO/Ledger/types/block.h>

#include <cstdint>
#include <atomic>
#include <string>
#include <mutex>

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

        Proof(uint32_t id)
        : MUTEX()
        , block()
        , nID(id)
        , fReset(false)
        {
        }


        virtual ~Proof()
        {
        }


        /** Channel
         *
         *
         *
         **/
        virtual uint32_t Channel() = 0;


        /** SetBlock
         *
         *  Sets the block for this proof.
         *
         **/
        void SetBlock(const TAO::Ledger::Block &block_)
        {
            std::unique_lock<std::mutex> lk(MUTEX);
            block = block_;

            /* If the block is set to null, update the reset flag. */
            if(block.IsNull())
                Reset();
        }


        /** GetBlock
         *
         *  Gets the block for this proof.
         *
         **/
        TAO::Ledger::Block GetBlock()
        {
            std::unique_lock<std::mutex> lk(MUTEX);
            return block;
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


         /** IsReset
          *
          *
          *
          **/
         bool IsReset() const { return fReset.load(); }


    protected:
        std::mutex MUTEX;
        TAO::Ledger::Block block;
        uint32_t nID;
        std::atomic<bool> fReset;
    };


}

#endif
