/*__________________________________________________________________________________________

            (c) Hash(BEGIN(Satoshi[2010]), END(Sunny[2012])) == Videlicet[2014] ++

            (c) Copyright The Nexus Developers 2014 - 2019

            Distributed under the MIT software license, see the accompanying
            file COPYING or http://www.opensource.org/licenses/mit-license.php.

            "ad vocem populi" - To the Voice of the People

____________________________________________________________________________________________*/

#include <LLC/types/proof.h>
#include <TAO/Ledger/types/block.h>
#include <LLP/templates/worker.h>
#include <LLP/templates/miner.h>

#include <Util/include/debug.h>

#include <functional>

namespace LLP
{
    Worker::Worker(uint8_t threadID, uint8_t blockID, Miner *miner, LLC::Proof *proof)
    : pMiner(miner)
    , pProof(proof)
    , workerThread()
    , fReset(true)
    , fStop(false)
    , fPause(true)
    , nID(threadID)
    , nBlockID(blockID)
    {
        /*Bind the worker thread. */
        workerThread = std::thread(std::bind(&Worker::Thread, this));
    }


    Worker::~Worker()
    {
        Stop();

        //if(pProof)
        //    delete pProof;

        /* Join the worker thread. */
        if(workerThread.joinable())
            workerThread.join();
    }


    void Worker::Thread()
    {
        /* Load the proof of work. */
        pProof->Load();

        /* Let miner know this worker is ready. */
        pMiner->Notify();

        /* Keep doing rounds of work until it is time to shutdown. */
        while (!fStop.load())
        {
            /* Initialize the proof of work. */
            if(!fReset.load() && !fPause.load())
                pProof->Init();

            /* Do work if there is no reset. */
            while(!fReset.load() && !fPause.load() && !fStop.load())
            {
                if(pProof->Work())
                {
                    if(!fReset.load())
                        pMiner->SubmitBlock(nBlockID);
                }

            }

            if(fStop.load())
                break;

            if(!fPause.load())
                fReset = false;

            /* Sleep for a small amount of time to avoid burning CPU cycles. */
            runtime::sleep(100);
        }

        /* Shutdown the worker. */
        pProof->Shutdown();

        /* Let miner know we are finished working. */
        pMiner->Notify();
    }


    void Worker::Start()
    {
        fReset = false;
        fStop = false;
        fPause = false;
    }

    void Worker::Pause()
    {
        fPause = true;
        pProof->Reset();
    }

    void Worker::Stop()
    {
        fStop = true;
    }

    void Worker::Reset()
    {
        fReset = true;
        fPause = false;
        pProof->Reset();
    }

}
