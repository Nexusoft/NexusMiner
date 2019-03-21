#include "serverconnection.h"
#include "miner.h"
#include "work_info.h"
#include "sleep.h"
#include <stdint.h>
#include <functional>
#include <numeric>
#include <atomic>
#include <deque>
#include <queue>
#include <mutex>

volatile uint32_t nBlocksAccepted = 0;
volatile uint32_t nBlocksRejected = 0;
volatile uint32_t nBestHeight = 0;
volatile uint32_t nLargest = 0;
extern volatile uint32_t nDifficulty;

extern std::atomic<bool> quit;
extern std::atomic<uint32_t> chain_counter[14];
std::atomic<uint64_t> SievedBits;
std::atomic<uint64_t> Tests_CPU;
std::atomic<uint64_t> Tests_GPU;
std::atomic<uint64_t> PrimesFound;
std::atomic<uint64_t> PrimesChecked;
std::atomic<uint64_t> nWeight;
std::deque<double> vWPSValues;

uint32_t nStartTimer = 0;

namespace Core
{
    extern std::queue<work_info> result_queue;
    extern std::mutex work_mutex;

    ServerConnection::ServerConnection(std::string ip, std::string port,
        uint8_t nMaxThreadsGPU, uint8_t nMaxThreadsCPU, uint8_t nMaxTimeout)
        : IP(ip)
        , PORT(port)
        , TIMER()
        , nThreadsGPU(nMaxThreadsGPU)
        , nThreadsCPU(nMaxThreadsCPU)
        , nTimeout(nMaxTimeout)
        , THREAD(std::bind(&ServerConnection::ServerThread, this))
        , fNewBlock(true)
    {

        uint8_t affinity = 0;
        uint8_t nthr = std::thread::hardware_concurrency();
        uint8_t i = 0;

        for (i = 0; i < nThreadsGPU; ++i)
            THREADS_GPU.push_back(new MinerThreadGPU(i, (affinity++) % nthr));

        for (i = 0; i < nThreadsCPU; ++i)
            THREADS_CPU.push_back(new MinerThreadCPU(i, (affinity++) % nthr));

        nStartTimer = (uint32_t)time(0);
    }

    ServerConnection::~ServerConnection()
    {
        THREAD.join();
        uint8_t i = 0;

        for(i = 0; i < nThreadsGPU; ++i)
            delete THREADS_GPU[i];

        for(i = 0; i < nThreadsCPU; ++i)
            delete THREADS_CPU[i];

        THREADS_GPU.clear();
        THREADS_CPU.clear();
    }

    /** Reset the block on each of the Threads. **/
    void ServerConnection::ResetThreads()
    {
        uint8_t i = 0;
        /** Reset each individual flag to tell threads to stop mining. **/
        for (i = 0; i < nThreadsGPU; ++i)
            THREADS_GPU[i]->fNewBlock = true;

        fNewBlock = true;
    }

    /** Main Connection Thread. Handles all the networking to allow
    Mining threads the most performance. **/
    void ServerConnection::ServerThread()
    {
        /** Don't begin until all mining threads are Created. **/
        while (THREADS_GPU.size() != nThreadsGPU)
            sleep_milliseconds(1);

        uint32_t ready = 0;
        while (ready != nThreadsGPU)
        {
            ready = 0;
            for (uint32_t i = 0; i < nThreadsGPU; ++i)
            {
                if (THREADS_GPU[i]->fReady)
                    ++ready;
            }
            sleep_milliseconds(1);
        }

        /** Initialize the Server Connection. **/
        CLIENT = new LLP::Miner(IP, PORT);


        /** Initialize a Timer for the Hash Meter. **/
        TIMER.Start();
        PrimeTimer.Start();

        while (true)
        {
            /** Run this thread at 1 Cycle per Second. **/
            sleep_milliseconds(1000);

            if (quit.load())
                break;


            /** Attempt with best efforts to keep the Connection Alive. **/
            if (!CLIENT->Connected() || CLIENT->Errors())
            {
                ResetThreads();

                if (!CLIENT->Connect())
                    continue;
                else
                    CLIENT->SetChannel(1);
            }


            /** Check the Block Height. **/
            uint32_t nHeight = CLIENT->GetHeight(nTimeout);
            if (nHeight == 0)
            {
                printf("Failed to Update Height...\n");
                CLIENT->Disconnect();
                continue;
            }

            /** If there is a new block, Flag the Threads to Stop Mining. **/
            if (nHeight != nBestHeight)
            {
                nBestHeight = nHeight;
                printf("\n[MASTER] Nexus Network: New Block %u\n", nHeight);

                ResetThreads();
            }

            if (PrimeTimer.Elapsed() >= 1)
            {
                uint32_t nElapsed = PrimeTimer.Elapsed();

                double WPS = nWeight.load() / (double)(nElapsed * 10000000);

                if (vWPSValues.size() >= 300)
                    vWPSValues.pop_front();

                vWPSValues.push_back(WPS);

                nWeight.store(0);

                PrimeTimer.Reset();
            }

            /** Rudimentary Meter **/
            if (TIMER.Elapsed() >= 15)
            {
                time_t now = time(0);
                uint32_t SecondsElapsed = (uint32_t)now - nStartTimer;
                uint32_t nElapsed = TIMER.Elapsed();

                uint64_t gibps = SievedBits.load() / nElapsed;
                SievedBits = 0;

                uint64_t tests_cpu = Tests_CPU.load();
                uint64_t tests_gpu = Tests_GPU.load();

                uint64_t tps_cpu = tests_cpu / nElapsed;
                uint64_t tps_gpu = tests_gpu / nElapsed;

                Tests_CPU = 0;
                Tests_GPU = 0;

                uint64_t checked = PrimesChecked.load();
                uint64_t found = PrimesFound.load();

                double pratio = 0.0;

                if (checked)
                    pratio = (double)(100 * found) / checked;

                double WPS = 1.0 * std::accumulate(vWPSValues.begin(), vWPSValues.end(), 0.0) / vWPSValues.size();

                uint32_t maxChToPrint = 9;

                printf("\n[METERS] %-5.02f WPS | Largest %f | Diff = %f | Block(s) A=%u R=%u | %02dd-%02d:%02d:%02d\n",
                    WPS,
                    nLargest / 10000000.0,
                    (double)nDifficulty / 10000000.0,
                    nBlocksAccepted,
                    nBlocksRejected,
                    SecondsElapsed / 86400,     //days
                    (SecondsElapsed / 3600) % 24, //hours
                    (SecondsElapsed / 60) % 60,   //minutes
                    (SecondsElapsed) % 60);     //seconds

                printf("\n-----------------------------------------------------------------------------------------------\nch  \t| ");
                for (uint32_t i = 3; i <= maxChToPrint; i++)
                    printf("%-7d  |  ", i);
                printf("\n---------------------------------------------------------------------------------------------\ncount\t| ");
                for (uint32_t i = 3; i <= maxChToPrint; i++)
                    printf("%-7d  |  ", chain_counter[i].load());
                printf("\n---------------------------------------------------------------------------------------------\nch/m\t| ");
                for (uint32_t i = 3; i <= maxChToPrint; i++)
                {
                    double sharePerHour = ((double)chain_counter[i].load() / SecondsElapsed) * 60.0;
                    printf("%-7.02f  |  ", sharePerHour);
                }
                printf("\n---------------------------------------------------------------------------------------------\nratio\t| ");

                for (uint32_t i = 3; i <= maxChToPrint; i++)
                {
                    double chRatio = 0;

                    uint32_t c = chain_counter[i].load();
                    uint32_t c2 = chain_counter[i - 1].load();

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

                    TIMER.Reset();
            }


            /** Attempt to get a new block from the Server if Thread needs One. **/
            if (fNewBlock)
            {
                /** Retrieve new block from Server.
                  If the Block didn't come in properly, Reconnect to the Server. **/
                LLP::CBlock BLOCK;

                if (!CLIENT->GetBlock(BLOCK, nTimeout))
                    CLIENT->Disconnect();

                /** If the Block isn't less than 1024-bits request a new one **/
                while (BLOCK.GetHash().high_bits(0x80000000))
                {
                    if (!CLIENT->GetBlock(BLOCK, nTimeout))
                        CLIENT->Disconnect();
                }

                size_t i;
                for (i = 0; i < THREADS_GPU.size(); ++i)
                {
                    /** If the block is good, tell the Mining Thread its okay to Mine. **/
                    THREADS_GPU[i]->BLOCK = BLOCK;
                    THREADS_GPU[i]->fBlockFound = false;
                    THREADS_GPU[i]->fNewBlock = false;
                }

                fNewBlock = false;
            }

            {
                std::unique_lock<std::mutex> lk(work_mutex);
                while (result_queue.empty() == false)
                {
                    printf("\nSubmitting Block...\n");
                    work_info work = result_queue.front();
                    result_queue.pop();

                    double difficulty = (double)work.nNonceDifficulty / 10000000.0;

                    printf("\n[MASTER] Prime Cluster of Difficulty %f Found\n", difficulty);

                    /** Attempt to Submit the Block to Network. **/
                    unsigned char RESPONSE = CLIENT->SubmitBlock(work.merkleRoot, work.nNonce, nTimeout);

                    /** Check the Response from the Server.**/
                    if (RESPONSE == 200)
                    {
                        printf("\n[MASTER] Block Accepted By Nexus Network.\n");

                        ResetThreads();
                        ++nBlocksAccepted;
                    }
                    else if (RESPONSE == 201)
                    {
                        printf("\n[MASTER] Block Rejected by Nexus Network.\n");
                        ++nBlocksRejected;
                    }

                    /** If the Response was Bad, Reconnect to Server. **/
                    else
                    {
                        printf("\n[MASTER] Failure to Submit Block. Reconnecting...\n");
                        CLIENT->Disconnect();
                        break;
                    }
                }
            }
        }
    }

}
