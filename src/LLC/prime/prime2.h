/*__________________________________________________________________________________________

			(c) Hash(BEGIN(Satoshi[2010]), END(Sunny[2012])) == Videlicet[2014] ++

			(c) Copyright The Nexus Developers 2014 - 2019

			Distributed under the MIT software license, see the accompanying
			file COPYING or http://www.opensource.org/licenses/mit-license.php.

			"ad vocem populi" - To The Voice of The People

____________________________________________________________________________________________*/

#pragma once
#ifndef NEXUS_LLC_INCLUDE_PRIME_TWO_H
#define NEXUS_LLC_INCLUDE_PRIME_TWO_H

#include <LLC/types/uint1024.h>
#include <cstdint>



namespace LLC
{

    bool PrimeCheck(const uint1024_t &n);

    bool SmallDivisor(const uint1024_t &n);
    bool Miller_Rabin(const uint1024_t& n, uint32_t nChecks);
    uint1024_t FermatTest(const uint1024_t &n);
    double GetPrimeDifficulty(const uint1024_t &next, uint32_t clusterSize);
    uint32_t GetFractionalDifficulty(const uint1024_t &composite);
}

#endif
