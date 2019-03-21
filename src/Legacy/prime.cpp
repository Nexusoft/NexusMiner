/*******************************************************************************************

      Hash(BEGIN(Satoshi[2010]), END(Sunny[2012])) == Videlicet[2014] ++

 [Learn, Create, but do not Forge] Viz. http://www.opensource.org/licenses/mit-license.php
 [Scale Indefinitely]        BlackJack.

*******************************************************************************************/

#include "prime.h"
#include "../cuda/sieve.h"
#include "../cuda/fermat.h"
#include "../cuda/util.h"
#include "../cuda/frame_resources.h"
#include "work_info.h"
#include "config.h"
#include "sleep.h"
#include "prime_list.h"
#include "print_colors.h"

#include <stdint.h>
#include <queue>
#include <atomic>
#include <algorithm>
#include <cmath>
#include <thread>
#include <mutex>

#if defined(_MSC_VER)
#include <mpir.h>
#else
#include <gmp.h>
#endif

#include <map>

uint32_t *primes;
uint32_t *primesInverseInvk;
mpz_t  zPrimorial;
uint64_t primorial;

static uint64_t *g_nonce_offsets[GPU_MAX] =   { 0 };
static uint64_t *g_nonce_meta[GPU_MAX] =      { 0 };
static uint32_t *g_bit_array_sieve[GPU_MAX] = { 0 };

const uint8_t nPrimorialEndPrime = 8;

uint64_t nBitArray_Stride;
uint64_t nBitArray_StartIndex[GPU_MAX] = { 0 };

extern std::atomic<uint32_t> nLargest;
extern std::atomic<uint32_t> nBestHeight;
extern std::atomic<uint64_t> SievedBits;
extern std::atomic<uint64_t> Tests_CPU;
extern std::atomic<uint64_t> Tests_GPU;
extern std::atomic<uint64_t> PrimesFound;
extern std::atomic<uint64_t> PrimesChecked;
extern std::atomic<uint64_t> nWeight;

uint16_t primeLimitA = 512;

std::atomic<bool> quit;
extern std::atomic<uint32_t> chain_counter[14];


std::vector<uint32_t> get_limbs(const mpz_t &big_int)
{
  std::vector<uint32_t> limbs(WORD_MAX);
  mpz_export(&limbs[0], 0, -1, sizeof(uint32_t), 0, 0, big_int);
  return limbs;
}

std::vector<uint8_t> get_bytes(const mpz_t &big_int)
{
    std::vector<uint8_t> bytes(128);
    mpz_export(&bytes[0], 0, -1, sizeof(uint8_t), 0, 0, big_int);
}


namespace Core
{
    std::mutex work_mutex;
    std::deque<work_info> work_queue;
    std::queue<work_info> result_queue;

    void InitializePrimes()
    {
        printf("\nGenerating primes...\n");
        primes = (uint32_t *)malloc((nSievePrimeLimit + 1) * sizeof(uint32_t));
        primesInverseInvk = (uint32_t*)malloc(sizeof(uint32_t) * 4 * nSievePrimeLimit);

        {
            std::vector<uint32_t> primevec;
            generate_n_primes(nSievePrimeLimit, primevec);
            primes[0] = nSievePrimeLimit;
            printf("%d primes generated\n", nSievePrimeLimit);
            memcpy(&primes[1], &primevec[0], nSievePrimeLimit * sizeof(uint32_t));
        }

        for (uint32_t i = 0; i < nSievePrimeLimit; ++i)
            memcpy(&primesInverseInvk[i * 4], &primes[i], sizeof(uint32_t));

        //calculate primorial and sieving stats
        mpz_init(zPrimorial);
        mpz_set_ui(zPrimorial, 1);
        double max_sieve = std::pow(2.0, 64);
        for (uint8_t i = 1; i < nPrimorialEndPrime; ++i)
        {
            mpz_mul_ui(zPrimorial, zPrimorial, primes[i]);
            max_sieve /= primes[i];
        }

        primorial = mpz_get_ui(zPrimorial);
        printf("\nPrimorial: %lu\n", primorial);
        printf("Last Primorial Prime = %u\n", primes[nPrimorialEndPrime - 1]);
        printf("First Sieving Prime = %u\n", primes[nPrimorialEndPrime]);

        int nSize = (int)mpz_sizeinbase(zPrimorial, 2);
        printf("Primorial Size = %d-bit\n", nSize);
        printf("Max. sieve size: %lu bits\n", (uint64_t)max_sieve);

        mpz_t zPrime, zInverse, zResult, n1, n2;
        mpz_init(zPrime);
        mpz_init(zInverse);
        mpz_init(zResult);
        mpz_init(n1);
        mpz_init(n2);
        mpz_set_ui(n1, 2);
        mpz_pow_ui(n1, n1, 64);

        printf("\nGenerating inverses...\n");
        for (uint32_t i = nPrimorialEndPrime; i <= nSievePrimeLimit; ++i)
        {
            mpz_set_ui(zPrime, primes[i]);

            int  inv = mpz_invert(zResult, zPrimorial, zPrime);
            if (inv <= 0)
            {
                printf("\nNo Inverse for prime %u at position %u\n\n", primes[i], i);
                exit(0);
            }
            else
                primesInverseInvk[i * 4 + 1] = (uint32_t)mpz_get_ui(zResult);
        }
        printf("%d inverses generated\n\n", nSievePrimeLimit - nPrimorialEndPrime + 1);

        printf("\nGenerating invK...\n");
        for (uint32_t i = nPrimorialEndPrime; i <= nSievePrimeLimit; ++i)
        {
            mpz_div_ui(n2, n1, primes[i]);
            uint64_t invK = mpz_get_ui(n2);
            memcpy(&primesInverseInvk[i * 4 + 2], &invK, sizeof(uint64_t));
        }

        mpz_clear(n1);
        mpz_clear(n2);
        mpz_clear(zPrime);
        mpz_clear(zInverse);
        mpz_clear(zResult);
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
    }

    void PrimeInit(uint8_t tid)
    {
        printf("Thread %d starting up...\n", tid);

        uint32_t bit_array_size = 1 << nSieveBitsLog2[tid];
        uint32_t prime_limit = 1 << nSievePrimesLog2[tid];

        cuda_init(tid);

        cuda_init_primes(tid, primes, primesInverseInvk, prime_limit,
        bit_array_size, 32,
        nPrimorialEndPrime, primeLimitA);

        cuda_set_sieve_offsets(tid, &offsetsA[0], (uint8_t)offsetsA.size(),
            &offsetsB[0], (uint8_t)offsetsB.size());

        cuda_set_test_offsets(tid, &offsetsTest[0], (uint32_t)offsetsTest.size());

        g_nonce_offsets[tid] = (uint64_t*)malloc(OFFSETS_MAX * sizeof(uint64_t));
        g_nonce_meta[tid] = (uint64_t*)malloc(OFFSETS_MAX * sizeof(uint64_t));
        g_bit_array_sieve[tid] = (uint32_t *)malloc(16 * (bit_array_size >> 5) * sizeof(uint32_t));
    }

    void PrimeFree(uint8_t tid)
    {
        delete[] g_nonce_offsets[tid];
        delete[] g_nonce_meta[tid];
        delete[] g_bit_array_sieve[tid];

        cuda_free_primes(tid);
    }

    /** Simple Modular Exponential Equation a^(n - 1) % n == 1 or notated in
        Modular Arithmetic a^(n - 1) = 1 [mod n]. **/
    uint1024 FermatTest(uint1024 n)
    {
        uint1024 r;
        mpz_t zR, zE, zN, zA;
        mpz_init(zR);
        mpz_init(zE);
        mpz_init(zN);
        mpz_init_set_ui(zA, 2);

        mpz_import(zN, 32, -1, sizeof(uint32_t), 0, 0, n.data());

        mpz_sub_ui(zE, zN, 1);
        mpz_powm(zR, zA, zE, zN);

        mpz_export(r.data(), 0, -1, sizeof(uint32_t), 0, 0, zR);

        mpz_clear(zR);
        mpz_clear(zE);
        mpz_clear(zN);
        mpz_clear(zA);

        return r;
    }


    double GetPrimeDifficulty(uint1024 next, uint32_t clusterSize)
    {
        /** Calulate the rarety of cluster from proportion of fermat remainder
            of last prime + 2. Keep fractional remainder in bounds of [0, 1] **/
        double fractionalRemainder = 1000000.0 / GetFractionalDifficulty(next);

        if (fractionalRemainder > 1.0 || fractionalRemainder < 0.0)
            fractionalRemainder = 0.0;

        return (clusterSize + fractionalRemainder);
    }

    /** Breaks the remainder of last composite in Prime Cluster into an integer.
        Larger numbers are more rare to find, so a proportion can be determined
        to give decimal difficulty between whole number increases. **/
    uint32_t GetFractionalDifficulty(uint1024 composite)
    {
        /** Break the remainder of Fermat test to calculate fractional difficulty [Thanks Sunny] **/
        mpz_t zA, zB, zC, zN;
        mpz_init(zA);
        mpz_init(zB);
        mpz_init(zC);
        mpz_init(zN);

        mpz_import(zB, 32, -1, sizeof(uint32_t), 0, 0, FermatTest(composite).data());
        mpz_import(zC, 32, -1, sizeof(uint32_t), 0, 0, composite.data());
        mpz_sub(zA, zC, zB);
        mpz_mul_2exp(zA, zA, 24);

        mpz_tdiv_q(zN, zA, zC);

        uint32_t diff = mpz_get_ui(zN);

        mpz_clear(zA);
        mpz_clear(zB);
        mpz_clear(zC);
        mpz_clear(zN);

        return diff;
    }


    /** Convert Double to unsigned int Representative. Used for
        encoding / decoding prime difficulty from nBits. **/
    uint32_t SetBits(double nDiff)
    {
        uint32_t nBits = 10000000;
        nBits = (uint32_t)(nBits * nDiff);

        return nBits;
    }

    void check_print(uint8_t thr_id, uint64_t nonce, uint32_t sieve_difficulty,
    const char *color, uint8_t chain_length, uint8_t chain_target)
    {
        if (chain_length == chain_target)
        {

            #if defined(_MSC_VER)
            printf("[METERS] %d-Chain Found: %f  Nonce: %016llX  %s[%d]\n",
                (int)chain_length,
                (double)sieve_difficulty / 1e7,
                nonce,
                cuda_devicename(thr_id),
                thr_id);
            #else
            printf("[METERS] %s%d-Chain Found: %f%s  Nonce: %016lX  %s[%d]\n",
                color,
                (int)chain_length,
                (double)sieve_difficulty / 1e7,
                KNRM,
                nonce,
                cuda_devicename(thr_id),
                thr_id);
            #endif
        }
    }


    bool PrimeQuery()
    {
        work_info work;
        bool have_work = false;
        {
            std::unique_lock<std::mutex> lk(work_mutex);
            if (!work_queue.empty())
            {
                work = work_queue.front();
                work_queue.pop_front();

                lk.unlock();
                have_work = true;
            }
        }

        if (have_work)
        {
            work.nNonce = 0;
            work.nNonceDifficulty = 0;

            mpz_t zTempVar, zN, zBaseOffsetted, zPrimeOrigin, zResidue, zTwo;
            mpz_init(zTempVar);
            mpz_init(zBaseOffsetted);
            mpz_init(zN);
            //mpz_init_set(zFirstSieveElement, work.zFirstSieveElement.__get_mp());
            mpz_init(zPrimeOrigin);

            mpz_import(zPrimeOrigin, 32, -1, sizeof(uint32_t), 0, 0, work.BaseHash.data());

            mpz_init(zResidue);
            mpz_init_set_ui(zTwo, 2);

            uint64_t nNonce = 0;
            uint32_t s = (uint32_t)work.nonce_offsets.size();
            uint8_t &thr_id = work.gpu_thread;

            for(uint32_t i = 0; i < s; ++i)
            {
                uint32_t nSieveDifficulty = 0;


                uint64_t &offset = work.nonce_offsets[i];
                uint64_t &meta = work.nonce_meta[i];

                uint32_t combo = meta >> 32;
                uint32_t chain_offset_beg = (meta >> 24) & 0xFF;
                uint32_t chain_offset_end = (meta >> 16) & 0xFF;
                uint32_t chain_length = meta & 0xFF;

                //printf("beg: %u end %u len %u\n", chain_offset_beg, chain_offset_end, chain_length);


                if (work.nHeight != nBestHeight || nBestHeight == 0 || quit.load())
                    break;

                /* compute the base offset of the nonce */
                mpz_mul_ui(zTempVar, zPrimorial, offset);
                mpz_add(zBaseOffsetted, work.zFirstSieveElement, zTempVar);



                chain_offset_beg = offsetsTest[chain_offset_beg];
                uint8_t nPrimeGap = 0;
                /* uint8_t next = 0;


                /* Get the next offset in the combo and test it */
                /* while(combo && nPrimeGap <= 12)
                {
                    next = __builtin_clz(combo);
                    combo ^= 0x80000000 >> next;

                    if(next >= offsetsTest.size())
                        break;

                    mpz_add_ui(zTempVar, zBaseOffsetted, offsetsTest[next]);
                    mpz_sub_ui(zN, zTempVar, 1);
                    mpz_powm(zResidue, zTwo, zN, zTempVar);
                    if (mpz_cmp_ui(zResidue, 1) == 0)
                    {
                        ++PrimesFound;
                        ++chain_length;

                        nPrimeGap = 0;
                    }
                    ++PrimesChecked;
                    ++Tests_CPU;

                    nPrimeGap += offsetsTest[next] - offsetsTest[chain_offset_end];

                    chain_offset_end = next;
                } */


                chain_offset_end = offsetsTest[chain_offset_end] + 2;

                mpz_add_ui(zTempVar, zBaseOffsetted, chain_offset_end);

                //uint16_t nStart = 0;
                //uint16_t nStop = 0;


                //search for primes after small cluster
                while (nPrimeGap <= 12)
                {
                    mpz_sub_ui(zN, zTempVar, 1);
                    mpz_powm(zResidue, zTwo, zN, zTempVar);
                    if (mpz_cmp_ui(zResidue, 1) == 0)
                    {
                        ++PrimesFound;
                        ++chain_length;

                        nPrimeGap = 0;
                    }
                    ++PrimesChecked;
                    ++Tests_CPU;

                    mpz_add_ui(zTempVar, zTempVar, 2);
                    chain_offset_end += 2;
                    nPrimeGap += 2;
                }


                nPrimeGap = 0;

                uint32_t begin_offset = 0;
                uint32_t begin_next = 2;

                //search for primes before small cluster
                mpz_add_ui(zTempVar, zBaseOffsetted, chain_offset_beg);
                mpz_sub_ui(zTempVar, zTempVar, begin_next);

                while (nPrimeGap <= 12)
                {
                    mpz_sub_ui(zN, zTempVar, 1);
                    mpz_powm(zResidue, zTwo, zN, zTempVar);
                    if (mpz_cmp_ui(zResidue, 1) == 0)
                    {
                        ++PrimesFound;
                        ++chain_length;
                        nPrimeGap = 0;
                        begin_offset = begin_next;
                    }
                    ++PrimesChecked;
                    ++Tests_CPU;

                    mpz_sub_ui(zTempVar, zTempVar, 2);
                    begin_next += 2;
                    nPrimeGap += 2;
                }


                /* Translate nonce offset of begin prime to global offset */
                mpz_add_ui(zTempVar, zBaseOffsetted, chain_offset_beg);
                mpz_sub_ui(zTempVar, zTempVar, begin_offset);
                mpz_sub(zTempVar, zTempVar, zPrimeOrigin);
                nNonce = mpz_get_ui(zTempVar);


                if (chain_length >= 3)
                {
                    nSieveDifficulty = SetBits(GetPrimeDifficulty(
                        work.BaseHash + nNonce + chain_offset_end, chain_length));

                    nWeight += nSieveDifficulty * 50;
                }

                if (nSieveDifficulty > nLargest)
                    nLargest = nSieveDifficulty;

                ++chain_counter[chain_length];

                check_print(thr_id, nNonce, nSieveDifficulty, KLGRN, chain_length, 2);
                check_print(thr_id, nNonce, nSieveDifficulty, KLGRN, chain_length, 3);
                check_print(thr_id, nNonce, nSieveDifficulty, KLGRN, chain_length, 4);
                check_print(thr_id, nNonce, nSieveDifficulty, KLGRN, chain_length, 5);
                check_print(thr_id, nNonce, nSieveDifficulty, KLCYN, chain_length, 6);
                check_print(thr_id, nNonce, nSieveDifficulty, KLMAG, chain_length, 7);
                check_print(thr_id, nNonce, nSieveDifficulty, KLYEL, chain_length, 8);
                check_print(thr_id, nNonce, nSieveDifficulty, KLYEL, chain_length, 9);

                if (nSieveDifficulty >= work.nDifficulty)
                {
                    work.nNonce = nNonce;
                    work.nNonceDifficulty = nSieveDifficulty;

                    std::unique_lock<std::mutex> lk(work_mutex);
                    result_queue.emplace(work);
                    break;
                }

            }

            mpz_clear(zPrimeOrigin);
            //mpz_clear(zFirstSieveElement);
            mpz_clear(zResidue);
            mpz_clear(zTwo);
            mpz_clear(zN);
            mpz_clear(zTempVar);
        }

        return have_work;
    }

    void PrimeSieve(uint8_t tid, uint1024 BaseHash,
                    uint32_t nDifficulty, uint32_t nHeight, uint512 merkleRoot)
    {
        mpz_t zPrimeOrigin;
        mpz_t zFirstSieveElement;
        mpz_t zPrimorialMod;
        mpz_t zTempVar;

        uint64_t *nonce_offsets = g_nonce_offsets[tid];
        uint64_t *nonce_meta = g_nonce_meta[tid];

        mpz_init(zFirstSieveElement);
        mpz_init(zPrimorialMod);
        mpz_init(zTempVar);
        mpz_init(zPrimeOrigin);

        mpz_import(zPrimeOrigin, 32, -1, sizeof(uint32_t), 0, 0, BaseHash.data());

        cuda_init_counts(tid);

        uint32_t prime_limit = (1 << nSievePrimesLog2[tid]);
        uint8_t sieve_bits_log2 = nSieveBitsLog2[tid];
        uint8_t test_levels = nTestLevels[tid];

        mpz_mod(zPrimorialMod, zPrimeOrigin, zPrimorial);
        mpz_sub(zPrimorialMod, zPrimorial, zPrimorialMod);
        mpz_add(zTempVar, zPrimeOrigin, zPrimorialMod);

        //compute base remainders
        cuda_set_zTempVar(tid, (const uint64_t*)zTempVar[0]._mp_d);
        cuda_base_remainders(tid, nPrimorialEndPrime, prime_limit);

        //compute non-colliding origins for each GPU within 2^64 search space
        uint64_t range = ~(0) / primorial / GPU_MAX;
        uint64_t gpu_offset = base_offset + range * primorial * tid;
        //uint64_t cpu_offset = base_offset + range * primorial * (tid * 2 + 1);
        //uint64_t index_range = (gpu_range >> sieve_bits_log2) / FRAME_COUNT;

        //compute first sieving element, and set on GPU
        mpz_add_ui(zFirstSieveElement, zTempVar, gpu_offset);

        //mpz_add_ui(zFirstSieveElement_CPU, zTempVar, cpu_offset);

        {
            std::vector<uint32_t> limbs = get_limbs(zFirstSieveElement);
            cuda_set_FirstSieveElement(tid, &limbs[0]);
        }

        //set the sieving primes for the first bucket (rest computed on the fly)
        //cuda_set_sieve(tid, gpu_offset, primorial,
        //               primeLimitA, prime_limit, sieve_bits_log2);

        cuda_set_quit(0);

        uint32_t count = 0;
        uint32_t primes_checked = 0;
        uint32_t primes_found = 0;
        uint32_t sieve_index = 0;
        uint32_t test_index = 0;
        uint32_t nIterations = 1 << nSieveIterationsLog2[tid];

        uint32_t sieve_index_cpu = 0;
        std::vector<uint64_t> work_offsets_cpu;
        std::vector<uint64_t> work_meta_cpu;

        while (nHeight && nHeight == nBestHeight && quit.load() == false)
        {
            //sieve bit array and compact test candidate nonces
            if(cuda_primesieve(tid, gpu_offset, primorial,
                            nPrimorialEndPrime, primeLimitA, prime_limit,
                            sieve_bits_log2, nDifficulty, sieve_index, test_index))
            {
                //after the number of iterations have been satisfied, start filling next queue
                if (sieve_index % nIterations == 0 && sieve_index > 0)
                {
                    //test results
                    cuda_fermat(tid, sieve_index, test_index, primorial, test_levels);
                }

                ++sieve_index;
                SievedBits += (uint64_t)1 << sieve_bits_log2;
            }

            /*
                uint64_t nBitArray_Size = (uint64_t)1 << sieve_bits_log2;
                uint64_t primorial_start = nBitArray_Size * (uint64_t)sieve_index_cpu;
                uint64_t base_offsetted = cpu_offset + primorial * primorial_start;

                uint32_t *bit_array_sieve = g_bit_array_sieve[tid];

                uint8_t s = offsetsA.size();

                /*clear the sieving array
                memset(g_bit_array_sieve[tid], 0, 16 * (nBitArray_Size >> 5));

                /*sieve with the given offsets
                for(uint32_t o = 0; o < s; ++o)
                {
                    for(uint32_t i = nPrimorialEndPrime; i < prime_limit; ++i)
                    {
                        uint32_t p = primesInverseInvk[i * 4 + 0];
                        uint32_t inv = primesInverseInvk[i * 4 + 1];

                        uint32_t base_remainder = mpz_tdiv_ui(zTempVar, p);

                        uint64_t r = base_offsetted + base_remainder + offsetsA[o];

                        r = ((uint64_t)p-r)*inv;

                        uint64_t index = r % p;

                        while(index < nBitArray_Size)
                        {
                            bit_array_sieve[index>>5] |= (1 << (index & 31));
                            index += p;
                        }
                    }
                    bit_array_sieve += nBitArray_Size >> 5;
                }
                bit_array_sieve = g_bit_array_sieve[tid];

                /* compact candidates from sieve
                uint8_t chain_threshold = 8;


                for(uint32_t i = 0; i < nBitArray_Size; ++i)
                {
                    uint32_t combo = 0;


                    uint8_t c = 0;
                    for(uint8_t o = 0; o < s; ++o)
                    {
                        uint32_t idx = o * nBitArray_Size + i;
                        uint16_t bit = (bit_array_sieve[idx >> 5] & (1 << (idx & 31))) == 0 ? 1 : 0;

                        c += bit;
                        combo |= bit << o;
                    }

                    if(c >= chain_threshold)
                    {
                        uint8_t max_gap = 0;

                        combo <<= (32 - s); //align to most significant bits

                        for(uint8_t o = 0; o < s; ++o)
                        {
                            max_gap = std::max(max_gap, (uint8_t)__builtin_clz(combo << o));
                        }

                        /* We want to make sure that the combination follows prime gap rule
                           as close as possible
                        if(max_gap < 2)
                        {
                            uint64_t nonce_offset = primorial_start + i;

                            uint8_t next = __builtin_clz(combo);
                            combo ^= 0x80000000 >> next;

                            //encode meta_data
                            uint64_t nonce_meta = 0;
                            nonce_meta |= ((uint64_t)combo << 32);
                            nonce_meta |= ((uint64_t)next << 24);
                            nonce_meta |= ((uint64_t)next << 16);

                            work_offsets_cpu.push_back(nonce_offset);
                            work_meta_cpu.push_back(nonce_meta);
                        }
                    }
                }

                printf("work_offsets_cpu.size() = %u\n", work_offsets_cpu.size());
                ++sieve_index_cpu;
                */

            //obtain the final results and push them onto the queue
            cuda_results(tid, test_index, nonce_offsets, nonce_meta,
                &count, &primes_checked, &primes_found);

            PrimesChecked += primes_checked;
            Tests_GPU += primes_checked;
            PrimesFound += primes_found;

            if (nHeight != nBestHeight || quit.load())
                break;

            /* add GPU sieve results to work queue */
            if (count)
            {
                std::map<uint64_t, uint64_t> nonces;

                for(uint32_t i = 0; i < count; ++i)
                    nonces[nonce_offsets[i]] = nonce_meta[i];

                std::vector<uint64_t> work_offsets;
                std::vector<uint64_t> work_meta;

                bool debug = false;
                if(debug)
                {
                    uint32_t dups = count - nonces.size();
                    if(dups > 0)
                        printf("[WARNING] GPU[%d] detected %3d duplicate nonces.\n", tid, dups);
                }


                for(auto it = nonces.begin(); it != nonces.end(); ++it)
                {
                    work_offsets.push_back(it->first);
                    work_meta.push_back(it->second);
                }

                {
                    std::unique_lock<std::mutex> lk(work_mutex);

                    work_queue.emplace_back(
                        work_info(BaseHash,
                                  nDifficulty,
                                  nHeight,
                                  work_offsets,
                                  work_meta,
                                  zFirstSieveElement,
                                  merkleRoot,
                                  tid));
                }
                count = 0;
                ++test_index;
            }

            /*add CPU sieve results to work queue */
            /* if(work_offsets_cpu.size())
            {
                std::unique_lock<std::mutex> lk(work_mutex);

                work_queue.emplace_back(
                    work_info(BaseHash,
                              nDifficulty,
                              nHeight,
                              work_offsets_cpu,
                              work_meta_cpu,
                              zFirstSieveElement_CPU,
                              merkleRoot,
                              tid));

                work_offsets_cpu.clear();
                work_meta_cpu.clear();
            } */

            /*change frequency of looping for better GPU utilization, can lead to
            lower latency than from a calling thread waking a blocking-sync thread */
            sleep_milliseconds(1);
            //sleep_microseconds(800);
        }

        cuda_set_quit(1);

        mpz_clear(zPrimeOrigin);
        mpz_clear(zFirstSieveElement);
        //mpz_clear(zFirstSieveElement_CPU);
        mpz_clear(zPrimorialMod);
        mpz_clear(zTempVar);
    }
}
