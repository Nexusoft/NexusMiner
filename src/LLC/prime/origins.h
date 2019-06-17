/*__________________________________________________________________________________________

			(c) Hash(BEGIN(Satoshi[2010]), END(Sunny[2012])) == Videlicet[2014] ++

			(c) Copyright The Nexus Developers 2014 - 2019

			Distributed under the MIT software license, see the accompanying
			file COPYING or http://www.opensource.org/licenses/mit-license.php.

			"ad vocem populi" - To The Voice of The People

____________________________________________________________________________________________*/

#pragma once
#ifndef NEXUS_LLC_INCLUDE_ORIGINS_H
#define NEXUS_LLC_INCLUDE_ORIGINS_H


#if defined(_MSC_VER)
#include <mpir.h>
#else
#include <gmp.h>
#endif

#include <vector>
#include <cstdint>
namespace LLC
{
    void ComputeOrigins(uint32_t base_offset,
                        const std::vector<uint32_t> &offsets,
                        uint32_t nPrimorialEndPrimeSmall,
                        uint32_t nPrimorialendPrimeLarge);
}

#endif
