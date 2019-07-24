/*__________________________________________________________________________________________

            (c) Hash(BEGIN(Satoshi[2010]), END(Sunny[2012])) == Videlicet[2014] ++

            (c) Copyright The Nexus Developers 2014 - 2019

            Distributed under the MIT software license, see the accompanying
            file COPYING or http://www.opensource.org/licenses/mit-license.php.

            "ad vocem populi" - To The Voice of The People

____________________________________________________________________________________________*/

#include <LLC/prime/origins.h>
#include <LLC/include/global.h>
#include <Util/include/debug.h>
#include <Util/include/args.h>
#include <Util/include/bitmanip.h>
#include <algorithm>
#include <limits>
#include <iomanip>
#include <fstream>
#include <omp.h>

#define MAX_OFFSETS 24
#define MAX_THREADS 8

namespace LLC
{

    void compute_primorial(mpz_t &primorial, uint32_t nPrimorialEndPrime)
    {
        for (uint32_t i = 1; i < nPrimorialEndPrime; ++i)
            mpz_mul_ui(primorial, primorial, primes[i]);
    }

    void ComputeOrigins(uint32_t nBaseOffset,
                         const std::vector<uint32_t> &vOffsets,
                         uint32_t nPrimorialEndPrimeSmall,
                         uint32_t nPrimorialendPrimeLarge)
    {

        mpz_t zPrimorialSmall;
        mpz_t zPrimorialLarge;
        mpz_t zFirstElement;

        mpz_t zTemp[MAX_THREADS];
        mpz_t zResidue[MAX_THREADS];
        mpz_t zN[MAX_THREADS];
        mpz_t zElement[MAX_THREADS];
        mpz_t zOrigin[MAX_THREADS];



        uint32_t nOffsets = vOffsets.size();
        uint64_t nPrimorialSmall;
        uint64_t nPrimorialLarge;
        uint64_t nOrigins;
        uint64_t nMaxSieveBits;
        uint32_t nThreads;
        uint32_t nTestCandidates;
        std::vector<uint64_t> vOrigins;
        std::multimap<uint32_t, uint64_t> mapOrigins;

        /* Initialize GMP variables. */
        mpz_init_set_ui(zPrimorialSmall, 1);
        mpz_init_set_ui(zPrimorialLarge, 1);
        mpz_init(zFirstElement);

        for(int i = 0; i < MAX_THREADS; ++i)
        {
            mpz_init(zTemp[i]);
            mpz_init(zResidue[i]);
            mpz_init(zN[i]);
            mpz_init(zElement[i]);
            mpz_init(zOrigin[i]);
        }


        /* Compute the small and large primorials. */
        compute_primorial(zPrimorialSmall, nPrimorialEndPrimeSmall);
        compute_primorial(zPrimorialLarge, nPrimorialendPrimeLarge);
        nPrimorialSmall = mpz_get_ui(zPrimorialSmall);
        nPrimorialLarge = mpz_get_ui(zPrimorialLarge);

        /* Compute the number of origins to test. */
        nOrigins = nPrimorialLarge / nPrimorialSmall;
        nTestCandidates = 128;

        /* Compute the max sieve size in number of bits. */
        nMaxSieveBits = std::numeric_limits<uint64_t>::max();
        nMaxSieveBits = nMaxSieveBits / nPrimorialLarge;

        debug::log(0, FUNCTION, "Base Offset     = ", nBaseOffset);
        debug::log(0, FUNCTION, "PrimorialSmall  = ", nPrimorialSmall);
        debug::log(0, FUNCTION, "PrimorialLarge  = ", nPrimorialLarge);
        debug::log(0, FUNCTION, "Test Origins    = ", nOrigins);
        debug::log(0, FUNCTION, "Test Candidates = ", nTestCandidates);
        debug::log(0, FUNCTION, "Max Sieve Bits  = ", nMaxSieveBits);
        debug::log(0, FUNCTION, "Bit Array Size  = ", nMaxSieveBits >> 13, " KB");


        /* Make a really large first element */
        uint32_t nBits = 192;
        mpz_set(zFirstElement, zPrimorialLarge);
        uint32_t nSize = mpz_sizeinbase(zFirstElement, 2);
        mpz_mul_2exp(zFirstElement, zFirstElement, nBits - nSize);
        nSize = mpz_sizeinbase(zFirstElement, 2);
        debug::log(0, FUNCTION, "First Element Size: ", nSize, "-Bit");

        /* Add the base_offset. */
        mpz_add_ui(zFirstElement, zFirstElement, nBaseOffset);



        /* Request as many threads as processors. */
        nThreads = omp_get_num_procs();
        debug::log(0, FUNCTION, nThreads, " Threads");

        omp_lock_t lk;
        omp_init_lock(&lk);

        /* Loop through each possible origin. */
        uint32_t i;
        uint32_t nProcessed = 0;


        /* Disable dynamic adjustment of number of threads. */
        omp_set_dynamic(0);
        omp_set_num_threads(nThreads);

        #pragma omp parallel for
        for(i = 0; i < nOrigins; ++i)
        {
            /* Get the thread id. */
            uint32_t idx = omp_get_thread_num();

            /* Calculate candidate prime origin. */
            mpz_mul_ui(zOrigin[idx], zPrimorialSmall, i);
            mpz_add(zElement[idx], zFirstElement, zOrigin[idx]);

            /* Since we added base offset to first element, instead of origin
               to save an extra calculation, add it back in. */
            mpz_add_ui(zOrigin[idx], zOrigin[idx], nBaseOffset);

            /* Set the origin. */
            uint64_t nOrigin = mpz_get_ui(zOrigin[idx]);

            /* Set the bitmask. */
            uint32_t nMask = 0;


            for(uint32_t j = 0; j < nTestCandidates; ++j)
            {
                /* Check if kill switch is engaged. */
                if(config::fShutdown.load())
                    break;

                /* Compute the next element. */
                mpz_add_ui(zElement[idx], zElement[idx], nPrimorialLarge);

                /* Add each offset to the element and test. */
                for(uint32_t k = 0; k < nOffsets; ++k)
                {
                    /* Make sure offsets that pass aren't checked further. */
                    if((nMask & (1 << k)) == 0)
                    {
                        /* Add the test offset. */
                        mpz_add_ui(zTemp[idx], zElement[idx], vOffsets[k]);

                        /* Check for Fermat test. */
                        mpz_sub_ui(zN[idx], zTemp[idx], 1);
                        mpz_powm(zResidue[idx], zTwo, zN[idx], zTemp[idx]);

                        if (mpz_cmp_ui(zResidue[idx], 1) == 0)
                            nMask |= (1 << k);
                    }
                }

                if(convert::popc(nMask) == nOffsets)
                {


                    omp_set_lock(&lk);
                    //vOrigins.push_back(nOrigin);
                    mapOrigins.insert(std::pair<uint32_t, uint64_t>(j, nOrigin));

                    /* Print info and add it to the list of origins. */
                    debug::log(0, FUNCTION, std::setw(6), std::left, i, " : Found good origin: ",
                        std::setw(12), std::left, nOrigin, " in ",
                        std::setw(3), std::left, j, " tests (", mapOrigins.size(), ")");

                    omp_unset_lock(&lk);
                    break;
                }
            }


            omp_set_lock(&lk);
            ++nProcessed;
            omp_unset_lock(&lk);

            if(nProcessed > 0 && (nProcessed % 1000) == 0)
                debug::log(0, FUNCTION,  "Processed ", nProcessed, " origins. ",
                    std::fixed, std::setprecision(3),
                    (double)(100 * nProcessed) / nOrigins, "%");

        }

        //debug::log(0, FUNCTION, "Generated ", vOrigins.size(), " origins");
        debug::log(0, FUNCTION, "Generated ", mapOrigins.size(), " origins");

        /* Sort any out of place origins from parallel testing. */
        //std::sort(vOrigins.begin(), vOrigins.end());

        ///* Print the offset counts and ratios. */
        //for(uint32_t i = 0; i < vOrigins.size(); ++i)
        //    debug::log(0, i, ": ", vOrigins[i]);
        i = 0;
        for(auto it = mapOrigins.begin(); it != mapOrigins.end(); ++it, ++i)
            debug::log(0, i, ": ", std::setw(3), std::left, it->first, " - ", std::setw(12), std::left, it->second);


        /* Store origins in a file named origins.ini. */
        std::ofstream fout("origins.ini");
        //for(uint32_t i = 0; i < vOrigins.size(); ++i)
        //    fout << vOrigins[i] << "\n";
        for(auto it = mapOrigins.begin(); it != mapOrigins.end(); ++it)
            fout << it->second << "\n";
        fout.close();

        omp_destroy_lock(&lk);


        /* Free GMP variables. */
        mpz_clear(zPrimorialSmall);
        mpz_clear(zPrimorialLarge);
        mpz_clear(zFirstElement);

        for(int i = 0; i < MAX_THREADS; ++i)
        {
            mpz_clear(zTemp[i]);
            mpz_clear(zResidue[i]);
            mpz_clear(zN[i]);
            mpz_clear(zElement[i]);
            mpz_clear(zOrigin[i]);
        }


    }


}
