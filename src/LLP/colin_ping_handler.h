/*__________________________________________________________________________________________

            Hash(BEGIN(Satoshi[2010]), END(Sunny[2012])) == Videlicet[2014]++

            (c) Copyright The Nexus Developers 2014 - 2025

            Distributed under the MIT software license, see the accompanying
            file COPYING or http://www.opensource.org/licenses/mit-license.php.

            "ad vocem populi" - To the Voice of the People

____________________________________________________________________________________________*/

#pragma once
#ifndef NEXUSMINER_LLP_COLIN_PING_HANDLER_H
#define NEXUSMINER_LLP_COLIN_PING_HANDLER_H

#include "LLP/include/colin_ping_protocol.h"
#include <spdlog/spdlog.h>
#include <atomic>
#include <cstdint>
#include <memory>

namespace LLP
{

    /** ColinPingHandler
     *
     *  Miner-side handler for the Colin AI Diagnostic PING/PONG protocol.
     *
     *  Responsibilities:
     *  1. Parse incoming 64-byte PingFrame from node (PING_DIAG opcode)
     *  2. Record receive timestamp for RTT echo
     *  3. Collect live miner telemetry (hash rate, temp, threads, queue)
     *  4. Build 64-byte PongFrame response
     *  5. Log received node health flags to Colin AI Miner terminal report
     *
     *  Usage (from connection thread):
     *    ColinPingHandler handler;
     *    handler.set_logger(my_logger);
     *    handler.set_telemetry_source(&myTelemetry);
     *
     *    // On PING_DIAG packet receipt:
     *    auto pong_data = handler.HandlePing(packet.DATA, is_stateless_lane);
     *    connection.Send(pong_data);
     *
     **/
    class ColinPingHandler
    {
    public:

        /** Telemetry source — implemented by the connection/mining layer **/
        struct ITelemetry
        {
            virtual uint32_t GetHashRateKHs()   const = 0;   // kH/s or chains/s
            virtual uint32_t GetTempCdeg()       const = 0;   // Celsius * 10 (0=unavailable)
            virtual uint32_t GetThreadCount()    const = 0;   // Active mining threads
            virtual uint32_t GetQueueDepth()     const = 0;   // Work queue depth
            virtual uint8_t  GetHealthFlags()    const = 0;   // PongFrame::MFLAG_* bits
            virtual ~ITelemetry() = default;
        };


        ColinPingHandler() = default;

        /** set_logger — bind spdlog logger for health-flag output.
         *  If not set, health-flag logging is skipped.
         **/
        void set_logger(std::shared_ptr<spdlog::logger> logger) { m_logger = std::move(logger); }

        /** set_telemetry_source — bind telemetry provider.
         *  Must be called before HandlePing() if telemetry is desired.
         **/
        void set_telemetry_source(ITelemetry* src) { m_telemetry = src; }

        /** set_channel — set miner channel (1=Prime, 2=Hash) for drought detection **/
        void set_channel(uint8_t channel) { m_channel = channel; }


        /** HandlePing
         *
         *  Full PING handling pipeline:
         *    1. Parse PingFrame from raw payload
         *    2. Stamp recv timestamp
         *    3. Collect telemetry
         *    4. Build PongFrame
         *    5. Log node health flags
         *
         *  @param[in] payload       Raw 64-byte wire data from PING_DIAG packet
         *  @param[in] stateless     True if stateless lane (16-bit opcode), false for legacy
         *
         *  @return Serialized 64-byte PongFrame ready for wire transmission.
         *          Empty vector on parse failure.
         *
         **/
        std::vector<uint8_t> HandlePing(const std::vector<uint8_t>& payload,
                                        bool stateless);


        /** last_received_ping — access the last successfully parsed PingFrame
         *  for use by the Colin AI Miner terminal report.
         **/
        const ReceivedPingFrame& last_received_ping() const { return m_last_ping; }

        /** ping_count — number of pings handled this session **/
        uint64_t ping_count() const { return m_ping_count.load(); }

        /** last_rtt_us — one-way latency of last received ping (node send → miner recv) in µs **/
        uint64_t last_rtt_us() const { return m_last_rtt_us.load(); }

    private:

        ITelemetry*                       m_telemetry{nullptr};
        ReceivedPingFrame                 m_last_ping;
        std::atomic<uint64_t>             m_ping_count{0};
        std::atomic<uint64_t>             m_last_rtt_us{0};
        std::shared_ptr<spdlog::logger>   m_logger;
        uint8_t                           m_channel{2};              // 1=Prime, 2=Hash
        uint32_t                          m_prev_hash_rate{0};       // for drop detection

        /** get_time_us — current time in microseconds since epoch **/
        static uint64_t get_time_us();

        /** log_node_health — render node health_flags to debug output **/
        void log_node_health(const ReceivedPingFrame& ping) const;
    };

} // namespace LLP

#endif
