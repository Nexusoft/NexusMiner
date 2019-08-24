/*__________________________________________________________________________________________

            (c) Hash(BEGIN(Satoshi[2010]), END(Sunny[2012])) == Videlicet[2014] ++

            (c) Copyright The Nexus Developers 2014 - 2019

            Distributed under the MIT software license, see the accompanying
            file COPYING or http://www.opensource.org/licenses/mit-license.php.

            "ad vocem populi" - To the Voice of the People

____________________________________________________________________________________________*/

#include <CUDA/include/util.h>
#include <LLC/include/global.h>
#include <Util/include/debug.h>
#include <Util/include/prime_list.h>
#include <Util/include/prime_config.h>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace LLC
{

    uint32_t *primes;
    uint32_t *primesInverseInvk;

    uint64_t nPrimorial;
    mpz_t zPrimorial;
    mpz_t zTwo;

    uint64_t *g_nonce_offsets[GPU_MAX] = {0};
    uint32_t *g_nonce_meta[GPU_MAX] = {0};
    uint32_t *g_bit_array_sieve[GPU_MAX] = {0};

    uint64_t nBitArray_Stride;
    uint64_t nBitArray_StartIndex[GPU_MAX] = {0};

    uint16_t primeLimitA = 4096;
    uint32_t primeLimitB = 564164;

    std::atomic<uint32_t> nChainCounts[14];


    std::mutex g_work_mutex;
    std::deque<work_info> g_work_queue;

    std::atomic<uint32_t> nLargest;
    std::atomic<uint32_t> nBestHeight;

    std::atomic<uint64_t> SievedBits;
    std::atomic<uint64_t> Tests_CPU;
    std::atomic<uint64_t> Tests_GPU;
    std::atomic<uint64_t> PrimesFound[OFFSETS_MAX];
    std::atomic<uint64_t> PrimesChecked[OFFSETS_MAX];
    double minRatios[OFFSETS_MAX];
    double maxRatios[OFFSETS_MAX];

    std::atomic<uint64_t> nWeight;
    std::deque<double> vWPSValues;

    std::atomic<uint64_t> nHashes;


    void InitializePrimes()
    {
        debug::log(0, "Generating primes...");
        primes = (uint32_t *)malloc((nSievePrimeLimit + 1) * sizeof(uint32_t));
        primesInverseInvk = (uint32_t*)malloc(sizeof(uint32_t) * 4 * nSievePrimeLimit);

        {
            std::vector<uint32_t> primevec;
            prime::generate_n_primes(nSievePrimeLimit, primevec);
            primes[0] = nSievePrimeLimit;
            debug::log(0, nSievePrimeLimit, " primes generated");
            memcpy(&primes[1], &primevec[0], nSievePrimeLimit * sizeof(uint32_t));
        }

        for (uint32_t i = 0; i < nSievePrimeLimit; ++i)
            memcpy(&primesInverseInvk[i * 4], &primes[i], sizeof(uint32_t));

        //calculate primorial and sieving stats
        mpz_init_set_ui(zPrimorial, 1);
        mpz_init_set_ui(zTwo, 2);

        /* Get the primorial. */
        for (uint32_t i = 1; i < nPrimorialEndPrime; ++i)
            mpz_mul_ui(zPrimorial, zPrimorial, primes[i]);
        nPrimorial = mpz_get_ui(zPrimorial);

        double max_sieve = std::pow(2.0, 64) / nPrimorial;

        debug::log(0, "");
        debug::log(0, "Primorial: ", nPrimorial);
        debug::log(0, "Last Primorial Prime = ", primes[nPrimorialEndPrime - 1]);
        debug::log(0, "First Sieving Prime = ", primes[nPrimorialEndPrime]);

        int nSize = (int)mpz_sizeinbase(zPrimorial, 2);
        debug::log(0, "Primorial Size = ", nSize, "-bit");
        debug::log(0, "Max. sieve size: ", (uint64_t)max_sieve, " bits");
        debug::log(0, "");

        mpz_t zPrime, zInverse, zResult, n1, n2;
        mpz_init(zPrime);
        mpz_init(zInverse);
        mpz_init(zResult);
        mpz_init(n1);
        mpz_init(n2);
        mpz_set_ui(n1, 2);
        mpz_pow_ui(n1, n1, 64);

        debug::log(0, "Generating inverses...");
        for (uint32_t i = nPrimorialEndPrime; i < nSievePrimeLimit; ++i)
        {
            mpz_set_ui(zPrime, primes[i]);

            int  inv = mpz_invert(zResult, zPrimorial, zPrime);
            if (inv <= 0)
            {
                debug::error("No Inverse for prime ", primes[i], " at position ", i);
                exit(0);
            }
            else
                primesInverseInvk[i * 4 + 1] = (uint32_t)mpz_get_ui(zResult);
        }
        debug::log(0, nSievePrimeLimit - nPrimorialEndPrime + 1, " inverses generated");
        debug::log(0, "");

        debug::log(0, "Generating invK...");
        for (uint32_t i = nPrimorialEndPrime; i < nSievePrimeLimit; ++i)
        {
            mpz_div_ui(n2, n1, primes[i]);
            uint64_t invK = mpz_get_ui(n2);
            memcpy(&primesInverseInvk[i * 4 + 2], &invK, sizeof(uint64_t));
        }
        debug::log(0, "invK generated");
        debug::log(0, "");

        mpz_clear(n1);
        mpz_clear(n2);
        mpz_clear(zPrime);
        mpz_clear(zInverse);
        mpz_clear(zResult);

        for(uint32_t i = 0; i < OFFSETS_MAX; ++i)
        {
            minRatios[i] = 100.0;
            maxRatios[i] = 0.0;
        }
    }


    void FreePrimes()
    {
        if (primes)
        {
            free(primes);
            primes = 0;
        }

        if (primesInverseInvk)
        {
            free(primesInverseInvk);
            primesInverseInvk = 0;
        }

        mpz_clear(zPrimorial);
        mpz_clear(zTwo);
    }
}
