#include "prime.h"
#include "config.h"
#include "serverconnection.h"
#include "../cuda/util.h"
#include "signals.h"
#include <vector>
#include <atomic>
#include <mutex>
#include <condition_variable>


extern std::atomic<bool> quit;
std::mutex m;
std::condition_variable cv;

void HandleSIGTERM(int signum)
{
    if(signum != SIGPIPE)
    {
        quit.store(true);
        cv.notify_one();
    }
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        printf("Too Few Arguments. The Required Arguments are Ip and Port\n");
        printf("Default Arguments are Total Threads = nVidia GPUs and Connection Timeout = 10 Seconds\n");
        printf("Format for Arguments is 'IP PORT DEVICELIST CPUTHREADS TIMEOUT'\n");
        return 0;
    }

    std::string IP = argv[1];
    std::string PORT = argv[2];
    uint8_t nThreadsGPU = cuda_num_devices();
    uint8_t nThreadsCPU = std::thread::hardware_concurrency();
    uint32_t nTimeout = 10;

    for (uint8_t i = 0; i < GPU_MAX; ++i) //initialize device map
        device_map[i] = i;

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

    if (argc > 4)
    nTimeout = static_cast<uint32_t>(atoi(argv[4]));

    load_config(nThreadsGPU);
    load_offsets();

    /* setup the signal interrupt callback */
    SetupSignals();
    quit.store(false);

    Core::InitializePrimes();

    printf("Initializing Miner %s:%s ThreadsGPU = %i, ThreadsCPU = %i, Timeout = %i\n",
        IP.c_str(), PORT.c_str(), nThreadsGPU, nThreadsCPU, nTimeout);

    Core::ServerConnection MINERS(IP, PORT, nThreadsGPU, nThreadsCPU, nTimeout);

    /* wait for callback before shutting down */
    std::unique_lock<std::mutex> lk(m);
    cv.wait(lk, []{return quit.load();});

    Core::FreePrimes();

    cuda_free(nThreadsGPU);

    return 0;
}
