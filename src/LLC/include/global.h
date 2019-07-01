/*__________________________________________________________________________________________

			(c) Hash(BEGIN(Satoshi[2010]), END(Sunny[2012])) == Videlicet[2014] ++

			(c) Copyright The Nexus Developers 2014 - 2019

			Distributed under the MIT software license, see the accompanying
			file COPYING or http://www.opensource.org/licenses/mit-license.php.

			"ad vocem populi" - To the Voice of the People

____________________________________________________________________________________________*/

#pragma once
#ifndef NEXUS_LLC_INCLUDE_GLOBAL_H
#define NEXUS_LLC_INCLUDE_GLOBAL_H

#include <LLC/include/work_info.h>
#include <CUDA/include/util.h>
#include <CUDA/include/macro.h>

#include <cstdint>
#include <deque>
#include <mutex>
#include <atomic>

#if defined(_MSC_VER)
#include <mpir.h>
#else
#include <gmp.h>
#endif

namespace LLC
{
    extern uint32_t *primes;
    extern uint32_t *primesInverseInvk;
    extern uint64_t nPrimorial;
    extern mpz_t zPrimorial;
    extern mpz_t zTwo;

    extern uint64_t *g_nonce_offsets[GPU_MAX];
    extern uint32_t *g_nonce_meta[GPU_MAX];
    extern uint32_t *g_bit_array_sieve[GPU_MAX];

    extern uint16_t primeLimitA;
    extern uint32_t primeLimitB;

    extern std::mutex g_work_mutex;
    extern std::deque<work_info> g_work_queue;

    extern std::atomic<uint32_t> nLargest;
    extern std::atomic<uint32_t> nBestHeight;

    extern std::atomic<uint64_t> SievedBits;
    extern std::atomic<uint64_t> Tests_CPU;
    extern std::atomic<uint64_t> Tests_GPU;
    extern std::atomic<uint64_t> PrimesFound[OFFSETS_MAX];
    extern std::atomic<uint64_t> PrimesChecked[OFFSETS_MAX];
    extern double minRatios[OFFSETS_MAX];
    extern double maxRatios[OFFSETS_MAX];

    extern std::atomic<uint64_t> nWeight;
    extern std::deque<double> vWPSValues;

    extern std::atomic<uint64_t> nHashes;

    #ifndef MAX_CHAIN_LENGTH
    #define MAX_CHAIN_LENGTH 14
    #endif

    extern std::atomic<uint32_t> nChainCounts[MAX_CHAIN_LENGTH];

    /* Global initialization. */
    void InitializePrimes();
    void FreePrimes();
}

#endif
