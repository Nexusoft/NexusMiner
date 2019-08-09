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

    /** Default Constructor. **/
    Worker::Worker(uint32_t threadID, Miner *miner, LLC::Proof *proof, bool fSubscribe_)
    : pMiner(miner)
    , pProof(proof)
    , workerThread()
    , nID(threadID)
    , fSubscribe(fSubscribe_)
    , fReset(false)
    , fStop(false)
    , fPause(true)
    {
        /*Bind the worker thread. */
        workerThread = std::thread(std::bind(&Worker::Thread, this));
    }


    /** Default Destructor. **/
    Worker::~Worker()
    {
        Stop();

        /* Join the worker thread. */
        if(workerThread.joinable())
            workerThread.join();
    }


    uint32_t Worker::Channel()
    {
        return pProof->Channel();
    }


    void Worker::Thread()
    {
        /* Load the proof of work. */
        pProof->Load();

        /* Let miner know this worker is ready. */
        pMiner->Notify();

        Wait();

        /* Keep doing rounds of work until it is time to shutdown. */
        while (!fStop.load())
        {
            /* Wait for the block to be ready. */
            if(fSubscribe && !fPause.load())
                pProof->SetBlock(pMiner->GetBlock(Channel()));

            /* If not paused, disable reset flag. */
            if(!fPause.load())
                fReset = false;

            /* Initialize the proof of work. */
            if(!fReset.load())
                pProof->Init();

            /* Do work if there is no reset. */
            while(!fReset.load() && !fPause.load() && !fStop.load())
            {
                if(pProof->Work())
                {
                    if(!fReset.load())
                        pMiner->SubmitBlock(pProof->GetBlock());
                }

                /* If the proof is reset, get more work. */
                if(pProof->IsReset())
                    break;

            }

            if(fStop.load())
                break;

            /* Sleep for a small amount of time to avoid burning CPU cycles. */
            runtime::sleep(100);
        }

        /* Shutdown the worker. */
        pProof->Shutdown();

        /* Let miner know we are finished working. */
        pMiner->Notify();
    }


    /* Set the block for this worker's proof. */
    void Worker::SetBlock(const TAO::Ledger::Block &block)
    {
        if(fSubscribe)
        {
            pProof->SetBlock(block);
            pProof->Init();
        }
    }


    void Worker::Start()
    {
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
        condition.notify_one();
    }


    /* Wait for reset (and block to be ready). */
    void Worker::Wait()
    {
        std::unique_lock<std::mutex> lk(mut);
        condition.wait(lk, [this] {return fReset.load() && !fPause.load();});

        if(!fPause.load())
            fReset = false;
    }

}
