/*__________________________________________________________________________________________

            (c) Hash(BEGIN(Satoshi[2010]), END(Sunny[2012])) == Videlicet[2014] ++

            (c) Copyright The Nexus Developers 2014 - 2019

            Distributed under the MIT software license, see the accompanying
            file COPYING or http://www.opensource.org/licenses/mit-license.php.

            "ad vocem populi" - To the Voice of the People

____________________________________________________________________________________________*/

#pragma once
#ifndef NEXUS_LLP_TEMPLATES_MINER_H
#define NEXUS_LLP_TEMPLATES_MINER_H

#include <LLC/types/uint1024.h>

#include <LLP/include/base_address.h>
#include <LLP/templates/outbound.h>
#include <LLP/templates/worker.h>

#include <TAO/Ledger/types/block.h>

#include <Util/include/runtime.h>
#include <Util/include/debug.h>

#include <string>
#include <vector>
#include <map>
#include <queue>
#include <thread>
#include <condition_variable>

namespace LLP
{
    enum
    {
        /** DATA PACKETS **/
        BLOCK_DATA   = 0,
        SUBMIT_BLOCK = 1,
        BLOCK_HEIGHT = 2,
        SET_CHANNEL  = 3,
        BLOCK_REWARD = 4,
        SET_COINBASE = 5,
        GOOD_BLOCK   = 6,
        ORPHAN_BLOCK = 7,


        /** DATA REQUESTS **/
        CHECK_BLOCK  = 64,
        SUBSCRIBE    = 65,


        /** REQUEST PACKETS **/
        GET_BLOCK    = 129,
        GET_HEIGHT   = 130,
        GET_REWARD   = 131,


        /** SERVER COMMANDS **/
        CLEAR_MAP    = 132,
        GET_ROUND    = 133,


        /** RESPONSE PACKETS **/
        BLOCK_ACCEPTED       = 200,
        BLOCK_REJECTED       = 201,


        /** VALIDATION RESPONSES **/
        COINBASE_SET  = 202,
        COINBASE_FAIL = 203,

        /** ROUND VALIDATIONS. **/
        NEW_ROUND     = 204,
        OLD_ROUND     = 205,

        /** GENERIC **/
        PING     = 253,
        CLOSE    = 254
    };

    class Miner : public Outbound
    {
    public:
        Miner(const std::string &ip, uint16_t port, uint16_t timeout);
        virtual ~Miner();

        void Thread();
        void Notify();
        void Wait();


        /** AddWorker
         *
         *
         *
         **/
        template <class ProofType>
        void AddWorker(uint8_t threadID, bool fSubscribeBlock = true)
        {
            /* Create a new proof for this worker. */
            ProofType *pProof = new ProofType(threadID);

            /* Set the channel flags for the miner. */
            nChannels |= pProof->Channel();

            /* Create a new worker with associated miner and proof. */
            Worker *pWorker = new Worker(threadID, this, pProof, fSubscribeBlock);

            /* Add a new worker to the list of workers. */
            vWorkers.push_back(pWorker);

            if(fSubscribeBlock)
                vSubscribed.push_back(pWorker);
        }

        void Start();
        void Stop();

        void SubmitBlock(const TAO::Ledger::Block &block);
        TAO::Ledger::Block GetBlock(uint32_t nChannel);

    private:

        void CheckSubmit();

        void Pause();
        void SetChannel(uint32_t nChannel);


        bool GetBlocks();
        uint32_t GetHeight();
        void Reset();
        void PrintStats();

    private:

        TAO::Ledger::Block get_block(uint32_t nChannel);

        std::vector<Worker *> vWorkers;
        std::vector<Worker *> vSubscribed;
        std::queue<TAO::Ledger::Block> qSubmit;

        std::thread minerThread;
        std::condition_variable condition;
        std::mutex mut;

        runtime::timer minerTimer;
        runtime::timer startTimer;
        runtime::timer wpsTimer;

        std::atomic<uint32_t> nBestHeight;
        std::atomic<uint32_t> nAccepted[2];
        std::atomic<uint32_t> nRejected[2];

        double nHashDifficulty;
        double nPrimeDifficulty;

        std::atomic<uint8_t> nReady;
        std::atomic<bool> fReset;
        std::atomic<bool> fStop;
        std::atomic<bool> fPause;

        uint32_t nChannels;

    };
}
#endif
