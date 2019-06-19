/*__________________________________________________________________________________________

            (c) Hash(BEGIN(Satoshi[2010]), END(Sunny[2012])) == Videlicet[2014] ++

            (c) Copyright The Nexus Developers 2014 - 2019

            Distributed under the MIT software license, see the accompanying
            file COPYING or http://www.opensource.org/licenses/mit-license.php.

            "ad vocem populi" - To the Voice of the People

____________________________________________________________________________________________*/

#include <CUDA/include/util.h>

#include <LLC/include/global.h>
#include <LLC/prime/origins.h>
#include <LLC/types/cuda_prime.h>
#include <LLC/types/cuda_hash.h>
#include <LLC/types/cpu_hash.h>
#include <LLC/types/cpu_primetest.h>
#include <LLC/types/cpu_primesieve.h>

#include <LLP/templates/miner.h>

#include <Util/include/debug.h>
#include <Util/include/signals.h>
#include <Util/include/prime_config.h>
#include <Util/include/csv.h>

#include <vector>
#include <thread>

int main(int argc, char **argv)
{
    /* Setup the signal handler. */
    signals::Setup();

    /* Parse the command line parameters. */
    config::ParseParameters(argc, argv);

    /* Open the debug log file. */
    debug::Initialize();

    /* Once we have read in the CLI paramters and config file, cache the args into global variables*/
    config::CacheArgs();

    /* Display the CUDA runtime and driver version. */
    int runtime_major = 0;
    int runtime_minor = 0;
    int driver_major = 0;
    int driver_minor = 0;
    cuda_runtime_version(runtime_major, runtime_minor);
    cuda_driver_version(driver_major, driver_minor);
    debug::log(0, "CUDA Runtime/Driver Version ", "[", runtime_major, ".", runtime_minor, " / ", driver_major, ".", driver_minor, "]");

    /* Get the IP address, port number, and timeout used to initialize miner with. */
    std::string ip = config::GetArg(std::string("-ip"), "127.0.0.1");
    uint16_t port = config::GetArg(std::string("-port"), config::fTestNet ? 8325 : 9325);
    uint16_t nTimeout = config::GetArg(std::string("-timeout"), 10);

    /* Get the number of CUDA Devices. */
    uint32_t nDevices = cuda_num_devices();

    /* Get the number of CPU threads allowed. */
    uint32_t nThreads = config::GetArg(std::string("-threads"), std::thread::hardware_concurrency());

    /* The comma seperated index list passed in from command line. */
    std::string strPrimeIndices = config::GetArg(std::string("-prime"), "");
    std::string strHashIndices = config::GetArg(std::string("-hash"), "");

    /* List of indices for prime and hash mining. */
    std::vector<uint32_t> primeIndices;
    std::vector<uint32_t> hashIndices;

    /* Get comma seperated values from string passed in via command line. */
    config::CommaSeperatedValues(primeIndices, strPrimeIndices);
    config::CommaSeperatedValues(hashIndices, strHashIndices);

    /* Get the number of GPUs allocated for prime and/or hash mining. */
    uint32_t nPrimeGPU = primeIndices.size();
    uint32_t nHashGPU = hashIndices.size();

    /* Get the number of CPU cores alloacted for prime/and or hash mining. */
    uint32_t nPrimeCPU = config::GetArg(std::string("-cpuprime"), 0);
    uint32_t nHashCPU = config::GetArg(std::string("-cpuhash"), 0);

    /* If prime origins is specified, generate a prime origin list. */
    if(config::GetBoolArg(std::string("-primeorigins")))
    {
        if(!prime::load_offsets())
            return 0;

        LLC::InitializePrimes();
        LLC::ComputeOrigins(base_offset, vOffsets, 8, 12);
        LLC::FreePrimes();

        return 0;
    }

    /* If there are any prime workers at all, load primes. */
    if(nPrimeGPU || nPrimeCPU)
    {
        /* Load in GPU config files for prime mining. */
        prime::load_config(nPrimeGPU);

        if(!prime::load_offsets())
            return 0;

        if(!prime::load_origins())
            return 0;

        /* Initialize primes used for prime mining. */
        LLC::InitializePrimes();
    }

    /* Initialize the miner. */
    LLP::Miner Miner(ip, port, nTimeout);

    /* Add GPU prime sieve/test workers to the miner. */
    for(uint32_t tid = 0; tid < nPrimeGPU; ++tid)
        Miner.AddWorker<LLC::PrimeCUDA>(primeIndices[tid]);

    /* Add CPU prime sieve workers to the miner. */
    for(uint32_t tid = 0; tid < nPrimeCPU; ++tid)
        Miner.AddWorker<LLC::PrimeSieveCPU>(tid);

    /* Add CPU prime test workers to the miner. */
    if(nPrimeGPU || nPrimeCPU)
    {
        for(uint32_t tid = 0; tid < nThreads; ++tid)
            Miner.AddWorker<LLC::PrimeTestCPU>(tid, false);
    }


    /* Add CPU hash workers to the miner. */
    for(uint32_t tid = 0; tid < nHashCPU; ++tid)
        Miner.AddWorker<LLC::HashCPU>(tid);

    /* Add GPU hash workers to the miner. */
    for(uint32_t tid = 0; tid < nHashGPU; ++tid)
        Miner.AddWorker<LLC::HashCUDA>(hashIndices[tid]);

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

    /* Output shutdown message. */
    debug::log(0, "Shutting down...");

    /* Stop the miner and workers. */
    Miner.Stop();

    /* Free the primes that were used for mining. */
    if(nPrimeGPU || nPrimeCPU)
        LLC::FreePrimes();

    /* Shutdown CUDA internal components. */
    cuda_shutdown();

    /* Close the debug log file. */
    debug::Shutdown();

    return 0;
}
