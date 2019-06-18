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

#include <condition_variable>
#include <thread>
#include <mutex>
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

        /** Default Constructor. **/
        Worker(uint32_t threadID, Miner *miner, LLC::Proof *proof, bool fSubscribe_ = true);


        /** Default Destructor. **/
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


        /** SetBlock
         *
         *  Set the block for this worker's proof
         *
         **/
        void SetBlock(const TAO::Ledger::Block &block);


    private:

        /** Wait
         *
         *  Wait for reset (and block to be ready).
         *
         **/
        void Wait();


        Miner *pMiner;
        LLC::Proof *pProof;

        std::condition_variable condition;
        std::mutex mut;
        std::thread workerThread;
        uint32_t nID;
        bool fSubscribe;
        std::atomic<bool> fReset;
        std::atomic<bool> fStop;
        std::atomic<bool> fPause;

    };
}

#endif
