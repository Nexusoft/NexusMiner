/*__________________________________________________________________________________________

            (c) Hash(BEGIN(Satoshi[2010]), END(Sunny[2012])) == Videlicet[2014] ++

            (c) Copyright The Nexus Developers 2014 - 2019

            Distributed under the MIT software license, see the accompanying
            file COPYING or http://www.opensource.org/licenses/mit-license.php.

            "ad vocem populi" - To the Voice of the People

____________________________________________________________________________________________*/

#include <LLC/include/global.h>
#include <LLP/templates/miner.h>
#include <TAO/Ledger/include/difficulty.h>
#include <Util/include/convert.h>
#include <Util/include/print_colors.h>
#include <functional>
#include <numeric>
#include <iomanip>

const char *ChannelName[3] =
{
    "Stake",
    "Prime",
    "Hash"
};

namespace LLP
{

    Miner::Miner(const std::string &ip, uint16_t port, uint16_t timeout)
    : Outbound(ip, port, timeout)
    , vWorkers()
    , mapBlocks()
    , qSubmit()
    , minerThread()
    , condition()
    , mut()
    , minerTimer()
    , startTimer()
    , wpsTimer()
    , nBestHeight(0)
    , nHashDifficulty(0.0)
    , nPrimeDifficulty(0.0)
    , nAccepted(0)
    , nRejected(0)
    , nWorkers(0)
    , nReady(0)
    , fReset(true)
    , fStop(true)
    , fPause(true)
    , nChannels(0)
    {
    }


    Miner::~Miner()
    {
        /* Stop the miner thread. */
        Stop();

        /* Free memory for workers. */
        for(auto it = vWorkers.begin(); it != vWorkers.end(); ++it)
            delete *it;

        /* Free memory for blocks. */
        for(auto it = mapBlocks.begin(); it != mapBlocks.end(); ++it)
            delete it->second;

        /* Set workers to zero. */
        mapBlocks.clear();
        vWorkers.clear();
        nWorkers = 0;
        nReady = 0;
    }


    /* Notify the miner that a worker is ready. */
    void Miner::Notify()
    {
        ++nReady;
        condition.notify_one();
    }

    void Miner::Wait()
    {
        std::unique_lock<std::mutex> lk(mut);
        condition.wait(lk, [this] {return nReady.load() >= nWorkers.load();});
        lk.unlock();
    }


    /* The main thread for the miner connection. */
    void Miner::Thread()
    {

        /* Wait for all workers to load. */
        Wait();

        debug::log(0, "Workers Loaded. ");

        minerTimer.Start();
        wpsTimer.Start();

        while(!fStop.load())
        {
            /* Run this thread at 1 cycle per second. */
            runtime::sleep(1000);

            /** Attempt with best efforts to keep the Connection Alive. **/
            if (!Connected())
            {
                if (!Connect())
                    continue;

                /* After connection has been reestablished, reset the miner. */
                Reset();
            }

            /* Check if shutdown occurred after sleep cycle. */
            if(fStop.load())
                break;

            /** Check the Block Height. **/
            uint32_t nHeight = GetHeight();
            if (nHeight == 0)
            {
                debug::error("Failed to Update Height...");
                Pause();
                continue;
            }

            /** If there is a new block, Flag the Threads to Stop Mining. **/
            if (nHeight != nBestHeight.load())
            {
                nBestHeight = nHeight;
                debug::log(0, "[MASTER] Nexus Network: New Block ", nHeight);

                /* Reset the workers so they can recieve new blocks. */
                Reset();
            }

            /* WPS Meter for Prime Minining. */
            if (wpsTimer.Elapsed() >= 1)
            {
                uint32_t nElapsed = wpsTimer.Elapsed();

                double WPS = LLC::nWeight.load() / (double)(nElapsed * 10000000);

                if (LLC::vWPSValues.size() >= 300)
                    LLC::vWPSValues.pop_front();

                LLC::vWPSValues.push_back(WPS);

                LLC::nWeight.store(0);

                wpsTimer.Reset();
            }

            /* Rudimentary meter. */
            if(minerTimer.Elapsed() >= 10)
            {
                PrintStats();
                minerTimer.Reset();
            }

            /* Get blocks for workers if reset. */
            if(fReset.load() && !fStop.load())
            {
                if(!GetBlocks())
                  continue;

                fReset = false;
            }

            /* Check a submit queue to see if there have been solutions submitted. */
            CheckSubmit();
        }
    }

    void Miner::SetChannel(uint32_t nChannel)
    {
        Packet REQUEST;

        REQUEST.HEADER = SET_CHANNEL;
        REQUEST.LENGTH = 4;
        REQUEST.DATA   = convert::uint2bytes(nChannel);

        WritePacket(REQUEST);
    }


    bool Miner::GetBlocks()
    {
        TAO::Ledger::Block block;
        Packet REQUEST;
        Packet RESPONSE;

        REQUEST.HEADER = GET_BLOCK;

        uint32_t count = static_cast<uint32_t>(mapBlocks.size());

        debug::log(1, FUNCTION, "Requesting ", count, " new Block", count > 1 ? "s." : ".");

        /* Lock all the blocks so miners can't submit stale blocks. */
        std::unique_lock<std::mutex> lk(mut);

        for(auto it = mapBlocks.begin(); it != mapBlocks.end(); ++it)
        {
            TAO::Ledger::Block *pBlock = it->second;

            /* Set the channel of the worker channel. */
            SetChannel(pBlock->nChannel);
            WritePacket(REQUEST);
            ReadNextPacket(RESPONSE);

            /* Check for null packet. */
            if(RESPONSE.IsNull())
                return debug::error(FUNCTION, " invalid block response.");

            /* Decode the response data into a block. */
            block.Deserialize(RESPONSE.DATA);

            /*Assign the block to the one we just recieved. */
            *pBlock = block;

            debug::log(1, FUNCTION, "Recieved new Block ",
                pBlock->ProofHash().ToString().substr(0, 20), " on channel ", pBlock->nChannel);

                /* Set the global difficulty. */
                if(pBlock->nChannel == 1)
                    nPrimeDifficulty = TAO::Ledger::GetDifficulty(pBlock->nBits, 1);
                else if(pBlock->nChannel == 2)
                    nHashDifficulty = TAO::Ledger::GetDifficulty(pBlock->nBits, 2);
                else
                {
                    nPrimeDifficulty = 0;
                    nHashDifficulty = 0;
                }
        }

        /* Allow worker to continue work with new block. */
        for(uint8_t i = 0; i < nWorkers.load(); ++i)
            vWorkers[i]->Reset();


        return true;
    }


    uint32_t Miner::GetHeight()
    {
        uint32_t nHeight = 0;
        Packet REQUEST;
        Packet RESPONSE;

        REQUEST.HEADER = GET_HEIGHT;

        {
            std::unique_lock<std::mutex> lk(mut);
            WritePacket(REQUEST);
            ReadNextPacket(RESPONSE);
        }

        if(!RESPONSE.IsNull())
            nHeight = convert::bytes2uint(RESPONSE.DATA);

        debug::log(3, FUNCTION, nHeight);

        return nHeight;
    }

    /* Check if there are any blocks to submit. */
    void Miner::CheckSubmit()
    {
        uint8_t blockID = 0;
        bool have_submit = false;

        /* Attempt to get a work result from queue. */
        {
            std::unique_lock<std::mutex> lk(mut);
            if(!qSubmit.empty())
            {
                blockID = qSubmit.front();
                qSubmit.pop();
                have_submit = true;
            }
        }

        /* Make sure there is work to submit. */
        if(have_submit == false || fReset.load())
            return;

        debug::log(0, "");
        debug::log(0, "[MASTER] Submitting ", ChannelName[mapBlocks[blockID]->nChannel], " Block...");


        Packet REQUEST;
        Packet RESPONSE;

        /* Make sure that the block to submit didn't come from a previous round. */
        if(mapBlocks[blockID]->nHeight != nBestHeight.load() && nBestHeight.load() != 1)
        {
            debug::log(0, "[MASTER] ", KLYEL, "ORPHANED (Stale)", KNRM, mapBlocks[blockID]->nHeight, " ", nBestHeight.load());
            return;
        }

        /* Build a Submit block packet request. */
        REQUEST.HEADER = SUBMIT_BLOCK;

        /* Submit the merkle root and nonce as requirements for Mining LLP server. */
        std::vector<uint8_t> vData = mapBlocks[blockID]->hashMerkleRoot.GetBytes();
        std::vector<uint8_t> vNonce = convert::uint2bytes64(mapBlocks[blockID]->nNonce);
        vData.insert(vData.end(), vNonce.begin(), vNonce.end());

        /* Set the packet data and length. */
        REQUEST.DATA = vData;
        REQUEST.LENGTH = vData.size();

        {
            std::unique_lock<std::mutex> lk(mut);
            WritePacket(REQUEST);
            ReadNextPacket(RESPONSE);
        }

        /* If the block was a valid block, send another request to make
           sure block made it into main chain or is an orphan. */
        if(RESPONSE.HEADER == BLOCK_ACCEPTED)
        {

            REQUEST.HEADER = CHECK_BLOCK;
            vData = mapBlocks[blockID]->GetHash().GetBytes();

            REQUEST.DATA = vData;
            REQUEST.LENGTH = vData.size();


            {
                std::unique_lock<std::mutex> lk(mut);
                WritePacket(REQUEST);
                ReadNextPacket(RESPONSE);
            }


            if(RESPONSE.HEADER == GOOD_BLOCK)
            {
                debug::log(0, "[MASTER] ", KLGRN, "ACCEPTED", KNRM);
                ++nAccepted;
                Reset();
            }
            else if(RESPONSE.HEADER == ORPHAN_BLOCK)
            {
                debug::log(0, "[MASTER] ", KLYEL, "ORPHANED", KNRM);
                ++nRejected;
            }
            /* If there was an error disconnect and try and reestablish connection. */
            else
            {
                debug::log(0, "[MASTER] Failure to Submit Block. Reconnecting...");
                Disconnect();
            }

        }

        /* If the block was outright rejected, increment rejected and continue mining. */
        else if(RESPONSE.HEADER == BLOCK_REJECTED)
        {
            debug::log(0, "[MASTER] ", KRED, "REJECTED", KNRM);
            ++nRejected;
        }

        /* If there was an error disconnect and try and reestablish connection. */
        else
        {
            debug::log(0, "[MASTER] Failure to Submit Block. Reconnecting...");
            Disconnect();
        }

        /* Newline. */
        debug::log(0, "");
    }


    void Miner::SubmitBlock(uint8_t blockID)
    {
        std::unique_lock<std::mutex> lk(mut);
        qSubmit.push(blockID);
    }


    void Miner::Reset()
    {

        fReset = true;

        if(fPause.load())
        {
            debug::log(0, "Resuming Miner ", addrOut.ToString());
            fPause = false;
        }

        /* Clear the submit queue. */
        std::unique_lock<std::mutex> lk(mut);
        qSubmit = std::queue<uint8_t>();
    }


    void Miner::Start()
    {

        /* If we are already started, don't start again. */
        if(!fStop.load())
            return;

        fStop = false;
        fPause = false;

        debug::log(0, "Starting Miner ", addrOut.ToString());

        startTimer.Start();
        minerTimer.Start();

        /* Start the workers. */
        for(uint8_t i = 0; i < nWorkers.load(); ++i)
            vWorkers[i]->Start();

        /* Bind the miner thread. */
        if(!minerThread.joinable())
        {
            debug::log(0, "Initializing Miner ", addrOut.ToString(),
                " Workers = ", static_cast<uint32_t>(nWorkers.load()),
                " Timeout = ", static_cast<uint32_t>(nTimeout));

            debug::log(0, "");

            minerThread = std::thread(std::bind(&Miner::Thread, this));
        }

    }

    void Miner::Pause()
    {
        /* If we are already paused, don't pause again. */
        if(!fPause.load())
        {

            debug::log(0, "Pausing Miner ", addrOut.ToString());

            fPause = true;

            for(uint8_t i = 0; i < nWorkers.load(); ++i)
                vWorkers[i]->Pause();

        }
    }


    void Miner::Stop()
    {
        /* If we are already. stopped, don't call again. */
        if(fStop.load())
            return;

        debug::log(0, "Stopping Miner...");


        /* No workers are ready. */
        nReady = 0;

        /* Stop the worker threads. */
        for(uint8_t i = 0; i < nWorkers.load(); ++i)
            vWorkers[i]->Stop();

        /* Wait for worker signals. */
        Wait();

        /* Workers stopped, tell miner we are stopped. */
        fStop = true;

        minerTimer.Stop();
        startTimer.Stop();

        /* Join the miner thread. */
        if(minerThread.joinable())
            minerThread.join();

    }


    void Miner::PrintStats()
    {
        uint32_t SecondsElapsed = startTimer.Elapsed();
        uint32_t nElapsed = minerTimer.Elapsed();
        uint64_t nElapsedMS = minerTimer.ElapsedMilliseconds();





        /* Print Hash Channel Stats. */
        if(nChannels & 2)
        {
            /* Calculate the Megahashes per second. */
            double nKH = static_cast<double>(LLC::nHashes.load()) / 1000.0;
            double nMHPerSecond = nKH / nElapsedMS;

            LLC::nHashes = 0;

            debug::log(0, "[HASHES] ", nMHPerSecond, " MH/s ",
            " | Diff = ", nHashDifficulty,
            " | Block(s) A=", nAccepted.load(), " R=", nRejected.load(),
            " | "); //TODO: format time here);


        }

        /* Print Prime Channel Stats. */
        if(nChannels & 1)
        {
            uint64_t gibps = LLC::SievedBits.load() / nElapsed;
            LLC::SievedBits = 0;

            uint64_t tests_cpu = LLC::Tests_CPU.load();
            uint64_t tests_gpu = LLC::Tests_GPU.load();

            uint64_t tps_cpu = tests_cpu / nElapsed;
            uint64_t tps_gpu = tests_gpu / nElapsed;

            LLC::Tests_CPU = 0;
            LLC::Tests_GPU = 0;

            uint64_t checked = LLC::PrimesChecked.load();
            uint64_t found = LLC::PrimesFound.load();

            double pratio = 0.0;

            if (checked)
                pratio = (double)(100 * found) / checked;

            double WPS = 1.0 * std::accumulate(LLC::vWPSValues.begin(), LLC::vWPSValues.end(), 0.0) / LLC::vWPSValues.size();

            uint32_t maxChToPrint = 9;

            /*
            printf("\n[PRIMES] %-5.02f WPS | Largest %f | Diff = %f | Block(s) A=%u R=%u | %02dd-%02d:%02d:%02d\n",
                WPS,
                LLC::nLargest.load() / 10000000.0,
                nPrimeDifficulty,
                nAccepted.load(),
                nRejected.load(),
                SecondsElapsed / 86400,     //days
                (SecondsElapsed / 3600) % 24, //hours
                (SecondsElapsed / 60) % 60,   //minutes
                (SecondsElapsed) % 60);     //seconds
            */
            debug::log(0, "[PRIMES] ", std::left, std::setw(8), std::setprecision(7), WPS, " WPS",
            " | Diff = ", nPrimeDifficulty,
            " | Block(s) A=", nAccepted.load(), " R=", nRejected.load(),
            " | ", std::setfill('0'), std::setprecision(2),
            SecondsElapsed / 86400, "d-",
            (SecondsElapsed / 3600) % 24, ":",
            (SecondsElapsed / 60) % 60, ":",
            (SecondsElapsed) % 60);

            printf("\n-----------------------------------------------------------------------------------------------\nch  \t| ");
            for (uint32_t i = 3; i <= maxChToPrint; i++)
                printf("%-7d  |  ", i);
            printf("\n---------------------------------------------------------------------------------------------\ncount\t| ");
            for (uint32_t i = 3; i <= maxChToPrint; i++)
                printf("%-7d  |  ", LLC::nChainCounts[i].load());
            printf("\n---------------------------------------------------------------------------------------------\nch/m\t| ");
            for (uint32_t i = 3; i <= maxChToPrint; i++)
            {
                double sharePerHour = ((double)LLC::nChainCounts[i].load() / SecondsElapsed) * 60.0;
                printf("%-7.02f  |  ", sharePerHour);
            }
            printf("\n---------------------------------------------------------------------------------------------\nratio\t| ");

            for (uint32_t i = 3; i <= maxChToPrint; i++)
            {
                double chRatio = 0;

                uint32_t c = LLC::nChainCounts[i].load();
                uint32_t c2 = LLC::nChainCounts[i - 1].load();

                if (c != 0)
                    chRatio = ((double)c2 / (double)c);
                printf("%-7.03f  |  ", chRatio);
            }
            printf("\n---------------------------------------------------------------------------------------------\n");
            printf("Sieved %5.2f GiB/s | Tested %lu T/s GPU, %lu T/s CPU | Ratio: %.3f %%\n\n",
                (double)gibps / (1 << 30),
                tps_gpu,
                tps_cpu,
                pratio);

            /* TODO: stick this at the end: LLC::nLargest.load() / 10000000.0, */

        }

        /* Print the total stats from each worker. */
        //debug::log(0, statsTotal.ToString());
    }

}
