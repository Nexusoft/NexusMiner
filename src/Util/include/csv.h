/*__________________________________________________________________________________________

            (c) Hash(BEGIN(Satoshi[2010]), END(Sunny[2012])) == Videlicet[2014] ++

            (c) Copyright The Nexus Developers 2014 - 2019

            Distributed under the MIT software license, see the accompanying
            file COPYING or http://www.opensource.org/licenses/mit-license.php.

            "ad vocem populi" - To the Voice of the People

____________________________________________________________________________________________*/

#pragma once
#ifndef NEXUS_UTIL_INCLUDE_CSV_H
#define NEXUS_UTIL_INCLUDE_CSV_H

#include <cstdint>
#include <vector>
#include <string>
#include <sstream>


namespace config
{
    /** CommaSeperatedValues
     *
     *  Get the comma seperated values from a string.
     *
     *  @param[out] values The values to return.
     *  @param[in] strCSV The comma seperated value string to parse for values.
     *
     **/
    void CommaSeperatedValues(std::vector<uint32_t> &values, std::string &strCSV);
    
}

#endif
