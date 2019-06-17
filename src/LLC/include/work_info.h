/*__________________________________________________________________________________________

			(c) Hash(BEGIN(Satoshi[2010]), END(Sunny[2012])) == Videlicet[2014] ++

			(c) Copyright The Nexus Developers 2014 - 2019

			Distributed under the MIT software license, see the accompanying
			file COPYING or http://www.opensource.org/licenses/mit-license.php.

			"ad vocem populi" - To the Voice of the People

____________________________________________________________________________________________*/

#pragma once
#ifndef NEXUS_LLC_INCLUDE_WORK_INFO_H
#define NEXUS_LLC_INCLUDE_WORK_INFO_H

#include <cstdint>
#include <vector>

namespace TAO
{
    namespace Ledger
    {
        class Block;
    }
}

namespace LLC
{
    class work_info
    {
    public:
        work_info() {}
        work_info(const std::vector<uint64_t> &nOffsets,
                  const std::vector<uint32_t> &nMeta,
                  TAO::Ledger::Block *block,
                  uint32_t tid)
        : nonce_offsets(nOffsets.begin(), nOffsets.end())
        , nonce_meta(nMeta.begin(), nMeta.end())
        , pBlock(block)
        , thr_id(tid)
        {
        }

        ~work_info()
        {
            //mpz_clear(zFirstSieveElement);
        }

        /* GPU intermediate results */
        std::vector<uint64_t> nonce_offsets;
        std::vector<uint32_t> nonce_meta;
        TAO::Ledger::Block *pBlock;
        uint32_t thr_id;
    };

}

#endif
