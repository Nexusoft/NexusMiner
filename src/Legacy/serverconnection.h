#ifndef NEXUS_CORE_SERVERCONNECTION_H
#define NEXUS_CORE_SERVERCONNECTION_H

#include "timer.h"
#include <thread>
#include <string>
#include <vector>

namespace LLP
{
    class Miner;
}


namespace Core
{
    class MinerThreadGPU;
    class MinerThreadCPU;

    /** Class to handle all the Connections via Mining LLP.
    Independent of Mining Threads for Higher Efficiency. **/
    class ServerConnection
    {
    public:
        LLP::Miner* CLIENT;
        uint8_t nThreadsGPU;
        uint8_t nThreadsCPU;
        uint8_t nTimeout;
        std::vector<MinerThreadGPU *> THREADS_GPU;
        std::vector<MinerThreadCPU *> THREADS_CPU;
        std::thread THREAD;
        LLP::Timer    TIMER;
        LLP::Timer PrimeTimer;
        std::string   IP, PORT;
        bool fNewBlock;

        ServerConnection(std::string ip, std::string port, uint8_t nMaxThreadsGPU,
            uint8_t nMaxThreadsCPU, uint8_t nMaxTimeout);

        ~ServerConnection();

        /** Reset the block on each of the Threads. **/
        void ResetThreads();

        /** Main Connection Thread. Handles all the networking to allow
        Mining threads the most performance. **/
        void ServerThread();
    };
}

#endif
