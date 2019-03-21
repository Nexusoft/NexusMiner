#ifndef NEXUS_CORE_PRIME_H
#define NEXUS_CORE_PRIME_H

#include "../hash/uint1024.h"

namespace Core
{
    void InitializePrimes();
    void FreePrimes();

    void PrimeInit(uint8_t threadIndex);
    void PrimeFree(uint8_t threadIndex);

    uint1024 FermatTest(uint1024 n);
    double GetPrimeDifficulty(uint1024 next, uint32_t clusterSize);
    uint32_t GetFractionalDifficulty(uint1024 composite);
    uint32_t SetBits(double nDiff);


    void PrimeSieve(uint8_t threadIndex,
                    uint1024 BaseHash,
                    uint32_t nDifficulty,
                    uint32_t nHeight,
                    uint512 merkleRoot);

    bool PrimeQuery();
}

#endif
