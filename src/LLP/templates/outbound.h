/*__________________________________________________________________________________________

            (c) Hash(BEGIN(Satoshi[2010]), END(Sunny[2012])) == Videlicet[2014] ++

            (c) Copyright The Nexus Developers 2014 - 2019

            Distributed under the MIT software license, see the accompanying
            file COPYING or http://www.opensource.org/licenses/mit-license.php.

            "ad vocem populi" - To the Voice of the People

____________________________________________________________________________________________*/

#pragma once
#ifndef NEXUS_LLP_TEMPLATES_OUTBOUND_H
#define NEXUS_LLP_TEMPLATES_OUTBOUND_H

#include <LLP/templates/connection.h>
#include <LLP/include/base_address.h>

#include <string>
#include <cstdint>


namespace LLP
{

    /* Forward declared. */
    class Packet;

    class Outbound : public Connection
    {

    public:
        Outbound(const std::string &ip, uint16_t port, uint16_t timeout = 10);
        virtual ~Outbound();

        bool Connect();

        void ReadNextPacket(Packet &PACKET);

    protected:
        BaseAddress addrOut;
        uint16_t nTimeout;
    };
}

#endif
