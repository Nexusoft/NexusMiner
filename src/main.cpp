/*__________________________________________________________________________________________

            (c) Hash(BEGIN(Satoshi[2010]), END(Sunny[2012])) == Videlicet[2014] ++

            (c) Copyright The Nexus Developers 2014 - 2019

            Distributed under the MIT software license, see the accompanying
            file COPYING or http://www.opensource.org/licenses/mit-license.php.

            "ad vocem populi" - To the Voice of the People

____________________________________________________________________________________________*/

#include <CUDA/include/util.h>

#include <LLC/include/global.h>
#include <LLC/types/cuda_prime.h>
#include <LLC/types/cuda_hash.h>
#include <LLC/types/cpu_primetest.h>
#include <LLC/types/cpu_primesieve.h>

#include <LLP/templates/miner.h>

#include <Util/include/debug.h>
#include <Util/include/signals.h>
#include <Util/include/prime_config.h>

#include <vector>
#include <thread>

int main(int argc, char **argv)
{
    /* Setup the signal handler. */
    signals::Setup();

    /* Open the debug log file. */
    debug::init("debug.log");

    /* Parse the command line parameters. */
    config::ParseParameters(argc, argv);

    /* Once we have read in the CLI paramters and config file, cache the args into global variables*/
    config::CacheArgs();

    /* Display the driver version. */
    int major = 0;
    int minor = 0;
    cuda_driver_version(major, minor);
    debug::log(0, "CUDA Driver Version ", major, ".", minor);


    std::string ip = config::GetArg(std::string("-ip"), "127.0.0.1");
    uint16_t port = config::GetArg(std::string("-port"), config::fTestNet ? 8325 : 9325);
    uint16_t nTimeout = config::GetArg(std::string("-timeout"), 10);


    uint8_t nThreadsGPU = config::GetArg(std::string("-devices"), cuda_num_devices());
    uint8_t nThreadsCPU = config::GetArg(std::string("-threads"), std::thread::hardware_concurrency());

    /*

    if (argc > 3)
    {
        uint8_t num_processors = nThreadsGPU;
        char * pch = strtok(argv[3], ",");
        nThreadsGPU = 0;
        while (pch != NULL)
        {
            if (pch[0] >= '0' && pch[0] <= '9' && pch[1] == '\0')
            {
                if (atoi(pch) < num_processors)
                    device_map[nThreadsGPU++] = atoi(pch);
                else
                {
                    fprintf(stderr, "Non-existant CUDA device #%d specified\n", atoi(pch));
                    exit(1);
                }
            }
            else
            {
                fprintf(stderr, "Non-existant CUDA device '%s' specified\n", pch);
                exit(1);
            }
            pch = strtok(NULL, ",");
        }
    }

    */

    if(config::GetBoolArg(std::string("-prime")))
    {
        /* Load in GPU config files for prime mining. */
        prime::load_config(nThreadsGPU);
        prime::load_offsets();

        /* Initialize primes used for prime mining. */
        LLC::InitializePrimes();
    }

    /* Initialize the Prime Miner. */
    LLP::Miner Miner(ip, port, nTimeout);

    /* Add workers to miner */
    if(config::GetBoolArg(std::string("-prime")))
    {

        if(config::GetBoolArg(std::string("-cpu"), false))
        {
            /* Sieve */
            for(uint8_t tid = 0; tid < nThreadsCPU; ++tid)
                Miner.AddWorker<LLC::PrimeSieveCPU>(tid, tid);

            /* Test */
            for(uint8_t tid = 0; tid < nThreadsCPU; ++tid)
                Miner.AddWorker<LLC::PrimeTestCPU>(tid, tid);
        }
        else
        {
            /* Sieve */
            for(uint8_t tid = 0; tid < nThreadsGPU; ++tid)
                Miner.AddWorker<LLC::PrimeCUDA>(tid, tid);

            /* Test */
            for(uint8_t tid = 0; tid < nThreadsGPU; ++tid)
                Miner.AddWorker<LLC::PrimeTestCPU>(tid, tid);
        }

    }
    else if(config::GetBoolArg(std::string("-hash")))
    {
        for(uint8_t tid = 0; tid < nThreadsGPU; ++tid)
            Miner.AddWorker<LLC::HashCUDA>(tid, tid);
    }


    /* Start the miner and workers. */
    Miner.Start();


    /* GDB mode waits for keyboard input to initiate clean shutdown. */
    if(config::GetBoolArg(std::string("-gdb")))
    {
        getchar();
        config::fShutdown = true;
    }
    else
    {
        /* Wait for the shutdown signal sequence. */
        signals::Wait();
    }

    debug::log(0, "Shutting down...");


    /* Stop the miner and workers. */
    Miner.Stop();


    if(config::GetBoolArg(std::string("-prime")))
    {
        /* Free the primes that were used for mining. */
        LLC::FreePrimes();
    }

    /* Shutdown CUDA internal components. */
    cuda_shutdown();

    /* Close the debug log file. */
    debug::shutdown();

    return 0;
}
