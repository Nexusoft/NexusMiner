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

    namespace POOL
    {
        enum
        {
            /** DATA PACKETS **/
            LOGIN            = 0,
            BLOCK_DATA       = 1,
            SUBMIT_SHARE     = 2,
            ACCOUNT_BALANCE  = 3,
            PENDING_PAYOUT   = 4,
            SUBMIT_STATS     = 5,

            /** REQUEST PACKETS **/
            GET_BLOCK    = 129,
            NEW_BLOCK    = 130,
            GET_BALANCE  = 131,
            GET_PAYOUT   = 132,


            /** RESPONSE PACKETS **/
            ACCEPT     = 200,
            REJECT     = 201,
            BLOCK      = 202,
            STALE      = 203,
        };
    }


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
        BLOCK_ACCEPTED  = 200,
        BLOCK_REJECTED  = 201,


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


    /** Miner
     *
     *
     *
     **/
    class Miner : public Outbound
    {
    public:

        /** Default constructor. **/
        Miner(const std::string &ip, uint16_t port, uint16_t timeout, double devfee = 0.0f);


        /** Default destructor. **/
        virtual ~Miner();


        /** Thread
         *
         *
         *
         **/
        void Thread();


        /** Notify
         *
         *
         *
         **/
        void Notify();


        /** Wait
         *
         *
         *
         **/
        void Wait();


        /** AddWorker
         *
         *
         *
         **/
        template <class ProofType>
        void AddWorker(uint32_t threadID, bool fSubscribeBlock = true)
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


        /** Start
         *
         *
         *
         **/
        void Start();


        /** Stop
         *
         *
         *
         **/
        void Stop();


        /** SubmitBlock
         *
         *
         *
         **/
        void SubmitBlock(const TAO::Ledger::Block &block);


        /** GetBlock
         *
         *
         *
         **/
        TAO::Ledger::Block GetBlock(uint32_t nChannel);


        /** GetBalance
         *
         *  Get your current balance in NXS that has not been included in a payout.
         *
         **/
        void GetBalance();


        /** GetPayouts
         *
         *  Get the current pending payouts for the next coinbase tx.
         *
         **/
        void GetPayouts();


        /** Ping
         *
         *  Ping the Pool Server to let it know connection is still alive.
         *
         **/
        void Ping();


        /** SubmitPPS
         *
         *  Send current PPS / WPS data to the pool
         *
         **/
        void SubmitPPS(double PPS, double WPS);


        /** Login
         *
         *  Send your address for Pool Login.
         *
         **/
        void Login(const std::string &addr);


        /** SubmitShare
         *
         *  Submit a Share to the Pool Server.
         *
         **/
        void SubmitShare(const uint1024_t& nPrimeOrigin, uint64_t nNonce);


        /** Subscribe
         *
         *  Tell the mining pool how many blocks to subscribe to.
         *
         **/
        void Subscribe(uint32_t nBlocks);


    private:

        /** CheckSubmit
         *
         *
         *
         **/
        void CheckSubmit();


        /** Pause
         *
         *
         *
         **/
        void Pause();


        /** SetChannel
         *
         *
         *
         **/
        void SetChannel(uint32_t nChannel);


        /** GetHeight
         *
         *
         *
         **/
        uint32_t GetHeight();


        /** Reset
         *
         *
         *
         **/
        void Reset();


        /** PrintStats
         *
         *
         *
         **/
        void PrintStats();

    private:

        /** get_block
         *
         *
         *
         **/
        TAO::Ledger::Block get_block(uint32_t nChannel);


        /** set_coinbase
         *
         *
         **/
        void set_coinbase();


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
        double nDevFee;

        std::atomic<uint8_t> nReady;
        std::atomic<bool> fReset;
        std::atomic<bool> fStop;
        std::atomic<bool> fPause;

        uint32_t nChannels;



    };
}
#endif
