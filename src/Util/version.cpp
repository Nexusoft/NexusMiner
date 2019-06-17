/*__________________________________________________________________________________________

            (c) Hash(BEGIN(Satoshi[2010]), END(Sunny[2012])) == Videlicet[2014] ++

            (c) Copyright The Nexus Developers 2014 - 2019

            Distributed under the MIT software license, see the accompanying
            file COPYING or http://www.opensource.org/licenses/mit-license.php.

            "ad vocem populi" - To the Voice of the People

____________________________________________________________________________________________*/

#include <Util/include/version.h>
#include <Util/include/debug.h>

namespace version
{

    /* The version of the actual wallet client. */
    const int CLIENT_VERSION =
                    1000000 * CLIENT_MAJOR
                  +   10000 * CLIENT_MINOR
                  +     100 * CLIENT_PATCH
                  +       1 * CLIENT_BUILD;


    /* Client Version Outputs. */
    const std::string CLIENT_NAME("Nexus Miner");
    const std::string CLIENT_TYPE("Beta");
    const std::string CLIENT_DATE(__DATE__ " " __TIME__);

    /* The version number */
    const std::string CLIENT_VERSION_STRING = debug::safe_printstr(CLIENT_MAJOR, ".", CLIENT_MINOR, ".", CLIENT_PATCH);


    /* The Architecture (32-Bit or 64-Bit) */
    #if defined x64
        const std::string BUILD_ARCH = "[x64]";
    #else
        const std::string BUILD_ARCH = "[x86]";
    #endif

    const std::string CLIENT_VERSION_BUILD_STRING(CLIENT_VERSION_STRING + " " + CLIENT_NAME  + " " +  BUILD_ARCH + " " + CLIENT_TYPE);
}
