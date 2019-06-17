/*__________________________________________________________________________________________

            (c) Hash(BEGIN(Satoshi[2010]), END(Sunny[2012])) == Videlicet[2014] ++

            (c) Copyright The Nexus Developers 2014 - 2019

            Distributed under the MIT software license, see the accompanying
            file COPYING or http://www.opensource.org/licenses/mit-license.php.

            "ad vocem populi" - To the Voice of the People

____________________________________________________________________________________________*/

#pragma once
#ifndef NEXUS_UTIL_INCLUDE_BITMANIP_H
#define NEXUS_UTIL_INCLUDE_BITMANIP_H

#ifdef WIN32
#include <intrin.h>
#endif

namespace convert
{

    /** popc
     *
     *  Counts the number of bits set in a word.
     *
     *  @param[in] nWord The 32-bit word to count from.
     *
     *  @return Returns the number of bits set in a word.
     *
     **/
    inline uint32_t popc(uint32_t nWord)
    {
    #ifdef WIN32
        return __popcnt(nWord);
    #else
        return __builtin_popcount(nWord);
    #endif
    }


    /** clz
     *
     *  Counts the number of leading zeroes in a word (also used to get the
     *  index of the most significant bit).
     *
     *  @param[in] nWord The 32-bit word to count from.
     *
     *  @return Returns the count of leading zeroes.
     *
     **/
    inline uint32_t clz(uint32_t nWord)
    {
    #ifdef WIN32
            return __popcnt(nWord);
    #else
            return __builtin_clz(nWord);
    #endif
    }


    /** ctz
     *
     *  Counts the number of trailing zeroes in a word (also used to get the
     *  index of the first least significant bit).
     *
     *  @param[in] nWord The 32-bit word to count from.
     *
     *  @return Returns the count of trailing zeroes.
     *
     **/
    inline uint32_t ctz(uint32_t nWord)
    {
    #ifdef WIN32
        uint32_t i = 0;
        _BitScanForward(i, nWord);
        return i;
    #else
        return __builtin_ctz(nWord);
    #endif
    }

}

#endif
