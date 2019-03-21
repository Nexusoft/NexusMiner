/*__________________________________________________________________________________________

            (c) Hash(BEGIN(Satoshi[2010]), END(Sunny[2012])) == Videlicet[2014] ++

            (c) Copyright The Nexus Developers 2014 - 2019

            Distributed under the MIT software license, see the accompanying
            file COPYING or http://www.opensource.org/licenses/mit-license.php.

            "ad vocem populi" - To the Voice of the People

____________________________________________________________________________________________*/

#pragma once
#ifndef NEXUS_LLP_TEMPLATES_WORKER_H
#define NEXUS_LLP_TEMPLATES_WORKER_H

#include <thread>
#include <atomic>
#include <cstdint>

namespace TAO
{
    namespace Ledger
    {
        class Block;
    }
}

namespace LLC
{
    class Proof;
}


namespace LLP
{
    /* Forward Declarations. */
    class Miner;


    /** Worker
     *
     *  The main worker class for working on a proof of work.
     *
     **/
    class Worker
    {
    public:

        /** Default constructor **/
        Worker(uint8_t threadID, uint8_t blockID, Miner *miner, LLC::Proof *proof);


        ~Worker();


        /** Channel
         *
         *  Get the channel index this worker is on.
         *
         **/
         uint32_t Channel();


        /** Thread
         *
         * The main thread for this worker.
         *
         **/
        void Thread();


        /** Start
         *
         *  Starts the proof with the block.
         *
         **/
         void Start();


         /** Stop
          *
          *  Stop this worker thread by stopping the proof of work.
          *
          **/
         void Stop();


         /** Pause
          *
          *  Pause this worker thread momentarily from doing work.
          *
          **/
         void Pause();


         /** Reset
          *
          *  Reset this worker thread by stopping the proof of work.
          *
          **/
         void Reset();


    private:
        Miner *pMiner;
        LLC::Proof *pProof;

        std::thread workerThread;
        std::atomic<bool> fReset;
        std::atomic<bool> fStop;
        std::atomic<bool> fPause;
        uint8_t nID;
        uint8_t nBlockID;
    };
}

#endif
