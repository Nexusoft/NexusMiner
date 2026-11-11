/*__________________________________________________________________________________________

            Hash(BEGIN(Satoshi[2010]), END(Sunny[2012])) == Videlicet[2014]++

            (c) Copyright The Nexus Developers 2014 - 2025

            Distributed under the MIT software license, see the accompanying
            file COPYING or http://www.opensource.org/licenses/mit-license.php.

            "ad vocem populi" - To the Voice of the People

____________________________________________________________________________________________*/

#include "LLP/colin_ping_handler.h"
#include <chrono>

namespace LLP
{

    uint64_t ColinPingHandler::get_time_us()
    {
        using namespace std::chrono;
        return static_cast<uint64_t>(
            duration_cast<microseconds>(
                system_clock::now().time_since_epoch()).count());
    }


    std::vector<uint8_t> ColinPingHandler::HandlePing(
        const std::vector<uint8_t>& payload,
        bool stateless)
    {
        /* 1. Stamp receive time first — before any processing for accuracy */
        uint64_t recv_us = get_time_us();

        /* 2. Parse PingFrame */
        ReceivedPingFrame ping;
        if(!ping.Parse(payload))
        {
            if(m_logger)
                m_logger->warn("[Colin PING] Malformed PingFrame received — discarding");
            return {};
        }

        m_last_ping = ping;
        ++m_ping_count;

        /* 3. Record one-way latency (node send → miner receive) */
        if(recv_us > ping.timestamp_send_us)
            m_last_rtt_us.store(recv_us - ping.timestamp_send_us);

        /* 4. Log node health flags if any warnings are set */
        log_node_health(ping);

        /* 5. Build PongFrame echo */
        PongFrame pong = PongFrame::BuildFromPing(ping, recv_us, stateless);

        /* 7. Fill telemetry (safe defaults if no source registered) */
        if(m_telemetry)
        {
            uint32_t hr      = m_telemetry->GetHashRateKHs();
            uint32_t temp    = m_telemetry->GetTempCdeg();
            uint32_t threads = m_telemetry->GetThreadCount();
            uint32_t queue   = m_telemetry->GetQueueDepth();

            pong.miner_hash_rate   = hr;
            pong.miner_temp_cdeg   = temp;
            pong.miner_threads     = threads;
            pong.miner_queue_depth = queue;

            /* Compute miner_health_flags from live telemetry */
            static constexpr uint32_t OVERHEAT_THRESHOLD_CDEG    = 850;  // 85 °C × 10
            static constexpr uint32_t QUEUE_FULL_THRESHOLD        = 8;
            static constexpr uint32_t HASH_RATE_DROP_THRESHOLD_PCT = 80; // flag if hr < 80% of prev
            static constexpr uint32_t ACCEPT_RATE_THRESHOLD_PCT   = 80;  // flag if accept < 80%

            uint8_t mflags = 0;

            /* MFLAG_IDLE: hash rate zero or no active threads */
            if(hr == 0 || threads == 0)
                mflags |= PongFrame::MFLAG_IDLE;

            /* MFLAG_OVERHEATING: temperature > 85 °C (stored as °C * 10) */
            if(temp > OVERHEAT_THRESHOLD_CDEG)
                mflags |= PongFrame::MFLAG_OVERHEATING;

            /* MFLAG_QUEUE_FULL: work queue deeper than threshold */
            if(queue > QUEUE_FULL_THRESHOLD)
                mflags |= PongFrame::MFLAG_QUEUE_FULL;

            /* MFLAG_HASH_RATE_DROP: hash rate dropped > 20% vs previous ping */
            if(m_prev_hash_rate > 0 && hr < m_prev_hash_rate * HASH_RATE_DROP_THRESHOLD_PCT / 100)
                mflags |= PongFrame::MFLAG_HASH_RATE_DROP;
            m_prev_hash_rate = hr;

            /* MFLAG_STALE_WORK: left unset by miner side (node sets NFLAG_STALE_TEMPLATE if it detects staleness) */

            /* MFLAG_RECONNECT_RECOVERY: first PONG of this session */
            if(m_ping_count.load() == 1)
                mflags |= PongFrame::MFLAG_RECONNECT_RECOVERY;

            /* MFLAG_WORK_REJECTED_LOCAL: accept rate < 80% of submissions */
            if(ping.blocks_submitted > 0 &&
               ping.blocks_accepted < ping.blocks_submitted * ACCEPT_RATE_THRESHOLD_PCT / 100u)
                mflags |= PongFrame::MFLAG_WORK_REJECTED_LOCAL;

            pong.miner_health_flags = mflags;
        }

        if(m_logger)
            m_logger->debug("[Colin PING] Seq #{} NodeHeight={} RTT-echo: stamp sent",
                ping.sequence, ping.unified_height);

        /* 8. Serialize and return */
        return pong.Serialize();
    }


    void ColinPingHandler::log_node_health(const ReceivedPingFrame& ping) const
    {
        if(!m_logger) return;
        uint8_t f = ping.health_flags;
        if(f == 0) return;

        m_logger->warn("[Colin PING] Node health flags for ping #{}:", ping.sequence);

        if(f & ReceivedPingFrame::NFLAG_STALE_TEMPLATE)
            m_logger->warn("  [Colin PING] NODE: STALE_TEMPLATE_DETECTED");
        if(f & ReceivedPingFrame::NFLAG_RATE_LIMITED)
            m_logger->warn("  [Colin PING] NODE: RATE_LIMITED");
        if(f & ReceivedPingFrame::NFLAG_SIM_LINK_ACTIVE)
            m_logger->info("  [Colin PING] NODE: SIM_LINK_ACTIVE");
        if(f & ReceivedPingFrame::NFLAG_DEDUP_HIT)
            m_logger->warn("  [Colin PING] NODE: DEDUP_HIT (duplicate submission detected)");
        if(f & ReceivedPingFrame::NFLAG_CHANNEL_MISMATCH)
            m_logger->warn("  [Colin PING] NODE: CHANNEL_MISMATCH");
        if(f & ReceivedPingFrame::NFLAG_HIGH_REJECT_RATE)
            m_logger->warn("  [Colin PING] NODE: HIGH_REJECTION_RATE (>20%)");
        if(f & ReceivedPingFrame::NFLAG_FIRST_CONNECT)
            m_logger->info("  [Colin PING] NODE: FIRST_CONNECT (new session)");
        if(f & ReceivedPingFrame::NFLAG_NODE_SYNCING)
            m_logger->warn("  [Colin PING] NODE: NODE_SYNCING (work may be stale)");
    }


} // namespace LLP
