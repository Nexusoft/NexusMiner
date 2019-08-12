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
#include <Util/include/prime_config.h>
#include <functional>
#include <numeric>
#include <iomanip>
#include <cmath>

namespace
{
    const char *ChannelName[3] =
    {
        "Stake",
        "Prime",
        "Hash"
    };


    uint32_t nPrevChannel = 0;

}



namespace LLP
{

    Miner::Miner(const std::string &ip, uint16_t port, uint16_t timeout)
    : Outbound(ip, port, timeout)
    , vWorkers()
    , vSubscribed()
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
    , nReady(0)
    , fReset(true)
    , fStop(true)
    , fPause(true)
    , nChannels(0)
    {
        nAccepted[0] = 0;
        nAccepted[1] = 0;
        nRejected[0] = 0;
        nRejected[1] = 0;
    }


    Miner::~Miner()
    {
        /* Stop the miner thread. */
        Stop();

        /* Free memory for workers. */
        for(auto it = vWorkers.begin(); it != vWorkers.end(); ++it)
            delete *it;

        /* Clear the workers. */
        vWorkers.clear();
        nReady = 0;
    }


    /* Notify the miner that a worker is ready. */
    void Miner::Notify()
    {
        ++nReady;
        condition.notify_one();
    }


    /* Wait for workers to be ready. */
    void Miner::Wait()
    {
        std::unique_lock<std::mutex> lk(mut);
        condition.wait(lk, [this] {return nReady.load() >= vWorkers.size();});
    }


    /* The main thread for the miner connection. */
    void Miner::Thread()
    {

        /* Wait for all workers to load. */
        Wait();

        debug::log(0, "Workers Loaded. ");

        minerTimer.Start();
        wpsTimer.Start();

        uint32_t nCounter = 0;

        while(!fStop.load())
        {
            /* Run this thread at 10 cycles per second. */
            runtime::sleep(100);

            /* Check if shutdown occurred after sleep cycle. */
            if(fStop.load())
                break;

            /** Attempt with best efforts to keep the Connection Alive. **/
            if (!fStop.load() && !Connected())
            {
                if (!Connect())
                {
                    /* Sleep an additional 5 seconds if reconnect attempt fails. */
                    runtime::sleep(5000);
                    continue;
                }
            }

            if(++nCounter >= 5)
            {
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

                    /* Set the coinbase reward for this round. */
                    {
                        std::unique_lock<std::mutex> lk(mut);
                        set_coinbase();
                    }


                    /* Reset the workers so they can recieve new blocks. */
                    Reset();
                }

                nCounter = 0;
            }


            /* WPS Meter for Prime Mining. */
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
                if(!fStop.load() && !fPause.load())
                    PrintStats();

                minerTimer.Reset();
            }


            /* Get blocks for workers if reset. */
            if(fReset.load() && !fStop.load())
            {
                /* Tell the workers to restart it's work. */
                for(const auto& worker : vWorkers)
                    worker->Reset();

                fReset = false;
            }

            /* Check a submit queue to see if there have been solutions submitted. */
            CheckSubmit();
        }
    }

    void Miner::SetChannel(uint32_t nChannel)
    {

        /* Make sure we don't attempt to set an invalid channel. */
        if(nChannel == 0 || nChannel > 2)
        {
            debug::error(FUNCTION, "Invalid Channel ", nChannel);
            return;
        }

        //if(nPrevChannel != nChannel)
        //{
        //    nPrevChannel = nChannel;

            Packet REQUEST;

            REQUEST.HEADER = SET_CHANNEL;
            REQUEST.LENGTH = 4;
            REQUEST.DATA   = convert::uint2bytes(nChannel);

            WritePacket(REQUEST);
        //}
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
        TAO::Ledger::Block block;
        bool have_submit = false;

        /* Get the best height. */
        uint32_t best_height = nBestHeight.load();

        /* Attempt to get a work result from queue. */
        {
            std::unique_lock<std::mutex> lk(mut);
            while(!qSubmit.empty())
            {
                block = qSubmit.front();
                qSubmit.pop();

                if(block.nHeight == best_height && best_height)
                {
                    have_submit = true;
                    break;
                }
            }
        }

        uint32_t nChannel = block.nChannel;

        /* Make sure there is work to submit. */
        if(have_submit == false || fReset.load())
            return;

        debug::log(0, "");
        debug::log(0, "[MASTER] Submitting ", ChannelName[nChannel], " Block ", block.ProofHash().SubString());


        Packet REQUEST;
        Packet RESPONSE;

        /* Build a Submit block packet request. */
        REQUEST.HEADER = SUBMIT_BLOCK;

        /* Submit the merkle root and nonce as requirements for Mining LLP server. */
        std::vector<uint8_t> vData = block.hashMerkleRoot.GetBytes();
        std::vector<uint8_t> vNonce = convert::uint2bytes64(block.nNonce);
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
            vData = block.GetHash().GetBytes();

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
                if(nChannel == 1 || nChannel == 2)
                    ++nAccepted[nChannel - 1];

                Reset();
            }
            else if(RESPONSE.HEADER == ORPHAN_BLOCK)
            {
                debug::log(0, "[MASTER] ", KLYEL, "ORPHANED", KNRM);
                if(nChannel == 1 || nChannel == 2)
                    ++nRejected[nChannel - 1];
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
            if(nChannel == 1 || nChannel == 2)
                ++nRejected[nChannel - 1];
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


    void Miner::SubmitBlock(const TAO::Ledger::Block &block)
    {
        std::unique_lock<std::mutex> lk(mut);

        /* Get the block from the map of blocks. */
        debug::log(2, FUNCTION, block.ProofHash().SubString());

        /* Push the block onto the queue. */
        qSubmit.push(block);
    }


    TAO::Ledger::Block Miner::GetBlock(uint32_t nChannel)
    {
        std::unique_lock<std::mutex> lk(mut);

        TAO::Ledger::Block block;
        block.SetNull();

        /* Send LLP messages to obtain a new block. */
        if(Connected())
            block = get_block(nChannel);

        /* Check to see if the block was recieved properly. */
        if(block.IsNull())
        {
            Pause();
            return block;
        }

        /* Print a debug log message for this block. */
        uint1024_t hashProof = block.ProofHash();
        debug::log(2, FUNCTION, hashProof.BitCount(), "-Bit ", std::setw(5), ChannelName[block.nChannel], " Block ",
            hashProof.SubString());

        return block;
    }


    /* Get your current balance in NXS that has not been included in a payout. */
    void Miner::GetBalance()
    {
        Packet REQUEST;
        REQUEST.HEADER = POOL::GET_BALANCE;
        WritePacket(REQUEST);
    }


    /* Get the Current Pending Payouts for the Next Coinbase Tx. */
    void Miner::GetPayouts()
    {
        Packet REQUEST;
        REQUEST.HEADER = POOL::GET_PAYOUT;
        WritePacket(REQUEST);
    }


    /* Ping the Pool Server to let it know Connection is Still Alive. */
    void Miner::Ping()
    {
        Packet REQUEST;
        REQUEST.HEADER = PING;
        WritePacket(REQUEST);
    }


    /* Send current PPS / WPS data to the pool */
    void Miner::SubmitPPS(double PPS, double WPS)
    {
        Packet REQUEST;
        REQUEST.HEADER = POOL::SUBMIT_STATS;

        std::vector<uint8_t> vPPSBytes = convert::uint2bytes64(static_cast<uint64_t>(PPS));
        std::vector<uint8_t> vWPSBytes = convert::uint2bytes64(static_cast<uint64_t>(WPS));

        /* Add PPS */
        REQUEST.DATA.insert(REQUEST.DATA.end(), vPPSBytes.begin(), vPPSBytes.end());

        /* Add WPS */
        REQUEST.DATA.insert(REQUEST.DATA.end(), vWPSBytes.begin(), vWPSBytes.end());

        /* Set the length. */
        REQUEST.LENGTH = REQUEST.DATA.size();

        WritePacket(REQUEST);
    }


    /* Send your address for Pool Login. */
    void Miner::Login(const std::string &addr)
    {
        Packet REQUEST;
        REQUEST.HEADER = POOL::LOGIN;
        REQUEST.DATA   = std::vector<uint8_t>(addr.begin(), addr.end());
        REQUEST.LENGTH = REQUEST.DATA.size();

        debug::log(0, "[MASTER] Logged in With Address: ", addr);
        WritePacket(REQUEST);
    }


    /* Submit a Share to the Pool Server. */
    void Miner::SubmitShare(const uint1024_t& nPrimeOrigin, uint64_t nNonce)
    {
        Packet REQUEST;
        REQUEST.HEADER = POOL::SUBMIT_SHARE;
        REQUEST.DATA = nPrimeOrigin.GetBytes();
        std::vector<uint8_t> NONCE  = convert::uint2bytes64(nNonce);

        REQUEST.DATA.insert(REQUEST.DATA.end(), NONCE.begin(), NONCE.end());
        REQUEST.LENGTH = 136;

        WritePacket(REQUEST);
    }


    /* Tell the mining pool how many blocks to subscribe to. */
    void Miner::Subscribe(uint32_t nBlocks)
    {
        Packet REQUEST;
        REQUEST.HEADER = SUBSCRIBE;
        REQUEST.DATA = convert::uint2bytes(nBlocks);
        REQUEST.LENGTH = 4;

        WritePacket(REQUEST);
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
        qSubmit = std::queue<TAO::Ledger::Block>();
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
        for(const auto& worker : vWorkers)
            worker->Start();

        /* Bind the miner thread. */
        if(!minerThread.joinable())
        {
            debug::log(0, "Initializing Miner ", addrOut.ToString(),
                " Workers = ", static_cast<uint32_t>(vWorkers.size()),
                " Timeout = ", static_cast<uint32_t>(nTimeout));

            debug::log(0, "");

            minerThread = std::thread(std::bind(&Miner::Thread, this));
        }

    }

    void Miner::Pause()
    {
        /* If we are already paused, don't pause again. */
        if(!fPause.load() && !fStop.load())
        {
            debug::log(0, "Pausing Miner ", addrOut.ToString());

            /* Set the miner pause flag. */
            fPause = true;

            nBestHeight = 0;

            /* Pause the workers. */
            for(const auto& worker : vWorkers)
                worker->Pause();
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
        for(const auto& worker : vWorkers)
        {
            worker->Reset();
            worker->Stop();
        }


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

        std::string strTime = debug::safe_printstr(
            std::setfill('0'), std::setw(2), SecondsElapsed / 86400, "d-",
            std::setfill('0'), std::setw(2), (SecondsElapsed / 3600) % 24, ":",
            std::setfill('0'), std::setw(2), (SecondsElapsed / 60) % 60, ":",
            std::setfill('0'), std::setw(2), (SecondsElapsed) % 60);


        debug::log(0, "");

        /* Print Hash Channel Stats. */
        if(nChannels & 2)
        {
            /* Calculate the Megahashes per second. */
            double nKH = static_cast<double>(LLC::nHashes.load()) / 1000.0;
            double nMHPerSecond = nKH / nElapsedMS;

            LLC::nHashes = 0;

            debug::log(0, "[HASHES] ", std::setw(9), std::left, std::fixed, std::setprecision(3), nMHPerSecond, " MH/s",
            " | Diff = ", std::setw(9), nHashDifficulty,
            " | Blocks A=", std::setw(2), nAccepted[1].load(), " R=", std::setw(2), nRejected[1].load(),
            " | ", strTime);
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

            uint64_t checked = 0;
            uint64_t found = 0;

            for(uint8_t i = 0; i < vOffsets.size(); ++i)
            {
                checked += LLC::PrimesChecked[i].load();
                found   += LLC::PrimesFound[i].load();
            }

            double ratio = 0.0;

            if (checked)
             ratio = (double)(100 * found) / checked;

            double WPS = 1.0 * std::accumulate(LLC::vWPSValues.begin(), LLC::vWPSValues.end(), 0.0) / LLC::vWPSValues.size();


            debug::log(0, "[PRIMES] Largest ", std::fixed, std::setprecision(7), (double)LLC::nLargest.load() / 1e7,
            " | Diff = ", std::setw(9), std::setprecision(7), nPrimeDifficulty,
            " | Blocks A=", std::setw(2), nAccepted[0].load(), " R=", std::setw(2), nRejected[0].load(),
            " | ", strTime);

            std::string stats;
            uint32_t maxChToPrint = 9;

            debug::log(0, "-----------------------------------------------------------------------------------------------");
            stats = debug::safe_printstr(std::setw(9), std::left, "| ch") + " | ";
            for(uint32_t i = 3; i <= maxChToPrint; ++i)
                stats += debug::safe_printstr(std::setw(9), std::left, i) + " | ";
            debug::log(0, stats);
            debug::log(0, "-----------------------------------------------------------------------------------------------");
            stats = debug::safe_printstr(std::setw(9), std::left, "| count") + " | ";
            for(uint32_t i = 3; i <= maxChToPrint; ++i)
                stats += debug::safe_printstr(std::fixed, std::setprecision(2), std::setw(9), std::left, LLC::nChainCounts[i].load()) + " | ";
            debug::log(0, stats);
            debug::log(0, "-----------------------------------------------------------------------------------------------");
            stats = debug::safe_printstr(std::setw(9), std::left, "| ch/m") + " | ";
            for(uint32_t i = 3; i <= maxChToPrint; ++i)
            {
                double sharePerHour = ((double)LLC::nChainCounts[i].load() / SecondsElapsed) * 60.0;
                stats += debug::safe_printstr(std::fixed, std::setprecision(2), std::setw(9), std::left, sharePerHour) + " | ";
            }

            debug::log(1, stats);
            debug::log(1, "-----------------------------------------------------------------------------------------------");
            stats = debug::safe_printstr(std::setw(9), std::left, "| ratio") + " | ";
            for(uint32_t i = 3; i <= maxChToPrint; ++i)
            {
                double chRatio = 0;

                uint32_t c = LLC::nChainCounts[i].load();
                uint32_t c2 = LLC::nChainCounts[i - 1].load();

                if (c != 0)
                    chRatio = ((double)c2 / (double)c);
                stats += debug::safe_printstr(std::fixed, std::setprecision(2), std::setw(9), std::left, chRatio) + " | ";
            }
            debug::log(1, stats);
            debug::log(1, "-----------------------------------------------------------------------------------------------");

            debug::log(0, "[PRIMES] ", std::setw(9), std::left, std::fixed, std::setprecision(3), WPS, " WP/s",
            " | ", std::fixed, std::setprecision(2), (double)gibps / (1 << 30), " GiB/s",
            " | ", tps_gpu, " T/s GPU, ", tps_cpu, " T/s CPU | Ratio: ", std::setprecision(3), ratio, " %");
            debug::log(0, "");



            /* Calculate and print the prime pattern offset ratios. */
            //debug::log(1, "[PRIMES] Offset Ratios: ");
            //for(uint16_t i = 0; i < vOffsets.size(); ++i)
            //{
            //    found = LLC::PrimesFound[i].load();
            //    checked = LLC::PrimesChecked[i].load();

                /* Check for divide by zero. */
            //    if(checked)
            //    {
            //        ratio = (double)(100 * found) / checked;

            //        LLC::minRatios[i] = std::min(LLC::minRatios[i], ratio);
            //        LLC::maxRatios[i] = std::max(LLC::maxRatios[i], ratio);
            //    }

            //    debug::log(1, std::setw(2), std::right, i, ": ",
            //                  std::setw(2), std::right, vOffsets[i], " = ",
            //                  std::setprecision(3), std::fixed, "[", LLC::minRatios[i], "-", LLC::maxRatios[i],  "]", "%  ");
            //}

        }
    }


    TAO::Ledger::Block Miner::get_block(uint32_t nChannel)
    {
        TAO::Ledger::Block block;
        Packet REQUEST;
        Packet RESPONSE;

        REQUEST.HEADER = GET_BLOCK;

        /* Set the channel of the worker channel. */
        SetChannel(nChannel);
        WritePacket(REQUEST);
        ReadNextPacket(RESPONSE);

        /* Check for null packet. */
        if(RESPONSE.IsNull())
        {
            debug::error(FUNCTION, " invalid block response.");
            block.SetNull();
            return block;
        }


        /* Decode the response data into a block. */
        block.Deserialize(RESPONSE.DATA);

        /* Make sure the channel from the block matches what was requested. */
        if(block.nChannel != nChannel)
        {
            debug::error("Recieved block channel: ", ChannelName[block.nChannel],
                " does not match channel: ", ChannelName[nChannel], " as requested");

            block.SetNull();
        }

        /* Set the global difficulty. */
        switch (nChannel)
        {
            case 1:
            {
                nPrimeDifficulty = TAO::Ledger::GetDifficulty(block.nBits, 1);
                break;
            }
            case 2:
            {
                nHashDifficulty = TAO::Ledger::GetDifficulty(block.nBits, 2);
                break;
            }
            default:
            {
                nPrimeDifficulty = 0;
                nHashDifficulty = 0;
                break;
            }
        }

        /* Return the newly created block. */
        return block;
    }

    void Miner::set_coinbase()
    {
        Packet REQUEST;
        Packet RESPONSE;

        /* Setup a reward request and wait for its arrival. */
        REQUEST.HEADER = GET_REWARD;
        WritePacket(REQUEST);
        ReadNextPacket(RESPONSE);

        /* Check if the reward was recieved. */
        if(RESPONSE.IsNull() || RESPONSE.HEADER != BLOCK_REWARD)
        {
            debug::error(FUNCTION, "invalid reward response.");
            return;
        }

        /* Get the maximum reward. */
        const uint64_t nMaxReward = convert::bytes2uint64(RESPONSE.DATA);

        /* Get the dev fee. */
        uint64_t nFee = static_cast<uint64_t>(static_cast<double>(nMaxReward) * 0.01);

        /* Get the reward minus fees. */
        uint64_t nReward = nMaxReward - nFee;

        std::vector<uint8_t> FEE = convert::uint2bytes64(nFee);
        std::vector<uint8_t> REWARD = convert::uint2bytes64(nReward);

        /* Setup a set coinbase request. */
        REQUEST.HEADER = SET_COINBASE;

        /* Add number of coinbase recipients. */
        uint8_t nSize = 1;
        REQUEST.DATA.push_back(nSize);

        /* Add the wallet reward. */
        REQUEST.DATA.insert(REQUEST.DATA.end(), REWARD.begin(), REWARD.end());

        /* Set the dev address. */
        std::string strAddress = "2SXUYZMAThtEz1aeVmGtJBLLgnmxrendSTj5A4qej771K2WZd9T";
        std::vector<uint8_t> ADDRESS = convert::string2bytes(strAddress);

        /* Set the length. */
        uint8_t nLength = ADDRESS.size();
        REQUEST.DATA.push_back(nLength);

        /* Set the string address. */
        REQUEST.DATA.insert(REQUEST.DATA.end(), ADDRESS.begin(), ADDRESS.end());

        /* Set the value for the coinbase output. */
        REQUEST.DATA.insert(REQUEST.DATA.end(), FEE.begin(), FEE.end());

        /* Set the length to the size of the packet. */
        REQUEST.LENGTH = REQUEST.DATA.size();

        WritePacket(REQUEST);
        ReadNextPacket(RESPONSE);

        /* Check if the set coinbase responded properly. */
        if(RESPONSE.IsNull())
        {
            debug::error(FUNCTION, "invalid set coinbase response.");
            return;
        }

        /* Print a coinbase fail message. */
        if(RESPONSE.HEADER == COINBASE_FAIL)
            debug::error(FUNCTION, "failed to set the coinbase.");

        /* Print a coinbase set message. */
        //if(RESPONSE.HEADER == COINBASE_SET)
        //    debug::log(0, FUNCTION, "coinbase set");

    }

}
