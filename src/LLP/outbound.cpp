/*__________________________________________________________________________________________

            (c) Hash(BEGIN(Satoshi[2010]), END(Sunny[2012])) == Videlicet[2014] ++

            (c) Copyright The Nexus Developers 2014 - 2019

            Distributed under the MIT software license, see the accompanying
            file COPYING or http://www.opensource.org/licenses/mit-license.php.

            "ad vocem populi" - To the Voice of the People

____________________________________________________________________________________________*/

#include <LLP/templates/outbound.h>
#include <LLP/packets/packet.h>
#include <Util/include/debug.h>

namespace LLP
{
    Outbound::Outbound(const std::string &ip, uint16_t port, uint16_t timeout)
    : Connection()
    , addrOut(ip, port, true)
    , nTimeout(timeout)
    {
    }

    Outbound::~Outbound()
    {
    }

    bool Outbound::Connect()
    {
        if(!((BaseConnection<Packet> *)this)->Connect(addrOut) || Timeout(nTimeout))
        {
            Disconnect();
            return debug::error("Failed to Connect to LLP Server");
        }

        debug::log(0, "Connected to ", addrOut.ToString());
        return true;
    }

    void Outbound::ReadNextPacket(Packet &PACKET)
    {
        PACKET.SetNull();

        while(!PacketComplete())
        {
            if(Errors() || Timeout(nTimeout))
            {
                Disconnect();
                return;
            }


            ReadPacket();

            runtime::sleep(1);
        }

        PACKET = INCOMING;

        ResetPacket();
    }

}
