# Worker-Feed Dedup Guard + MinedBlockCache — Architecture Diagrams

This document contains five Mermaid diagrams describing the defense layers introduced by
PR #263 ("Add worker-feed dedup guard + mutex-based recovery gate as defense-in-depth").

---

## Diagram 1 — Worker-Feed Dedup Flowchart

Decision path from template notification arrival to worker restart.

```mermaid
flowchart TD
    A([SendChannelNotification / set_block_handler callback]) --> B[Extract block.nHeight\nand block.hashPrevBlock]
    B --> C[Build dedup key:\nheight == m_last_worker_feed_height\nAND hashPrevBlock == m_last_worker_feed_prev_hash]
    C --> D{same_template?}
    D -- No --> G[Update m_last_worker_feed_tp\nm_last_worker_feed_height\nm_last_worker_feed_prev_hash]
    D -- Yes --> E[Compute ms_since_last\n= now − m_last_worker_feed_tp]
    E --> F{ms_since_last\n< DEBOUNCE_MS\n(2 000 ms)?}
    F -- Yes --> H([⏱ Early-return — duplicate suppressed\nWorkers NOT restarted])
    F -- No --> G
    G --> I[update_confirmations on MinedBlockCache]
    I --> J[Distribute template to all workers]
    J --> K([Workers restart sieve / GPU loop])
```

---

## Diagram 2 — MinedBlockCache Three-Tier State Machine

State transitions for `record_accepted_block()` and `update_confirmations()`.

```mermaid
stateDiagram-v2
    [*] --> Tier1 : record_accepted_block()

    state Tier1 {
        [*] --> Hot
        Hot --> Hot : push_front new record
        note right of Hot
            Max 5 records (TIER1_MAX)
            Confirmation tracking active
            Newest record is at front
        end note
    }

    Tier1 --> Tier2 : Tier 1 full — oldest evicted\n(push_front to Tier 2)
    Tier1 --> Tier2 : update_confirmations():\nconfirmations >= CONFIRMATION_THRESHOLD (5)\nAND record is not the sole front

    state Tier2 {
        [*] --> Warm
        Warm --> Warm : accumulate
        note right of Warm
            Max 100 records (TIER2_MAX)
            No active confirmation tracking
        end note
    }

    Tier2 --> Tier3 : Tier 2 full — oldest overflow\n(push_front to Tier 3)

    state Tier3 {
        [*] --> Archive
        Archive --> Archive : accumulate (unbounded)
        note right of Archive
            No size limit
            Permanent audit trail
        end note
    }
```

---

## Diagram 3 — Colin Diagnostic Report Pipeline

How the MinedBlockCache feeds the Colin 60-second diagnostic report.

```mermaid
sequenceDiagram
    participant Timer as asio timer (60 s)
    participant Colin as ColinAgent::emit_report()
    participant Source as m_mined_block_cache_source\n(lambda in Worker_manager)
    participant Cache as MinedBlockCache::tier1()
    participant Log as spdlog logger

    Timer->>Colin: fires every 60 s
    Colin->>Source: m_mined_block_cache_source()
    Source->>Cache: wm->m_mined_block_cache.tier1()
    Cache-->>Source: deque<MinedBlockRecord> (≤ 5)
    Source-->>Colin: vector<MinedBlockSnapshot>\n(height, channel, confirmations,\nhash_prev_block_hex, status_emoji)
    Colin->>Log: info "── Mined Block History (Top 5) ─"
    loop for each snapshot i
        Colin->>Log: info "{emoji} #{i} height={h} channel={ch}\nconfirmations={n} prev={hex}..."
    end
```

---

## Diagram 4 — BLOCK_ACCEPTED / GOOD_BLOCK → Cache Record Flow

End-to-end path from node acknowledgement to MinedBlockCache.

```mermaid
sequenceDiagram
    participant Node as Nexus Node
    participant Solo as Solo::process_messages()
    participant Handler as m_block_accepted_handler\n(Worker_manager::on_block_accepted)
    participant Cache as MinedBlockCache

    Node->>Solo: BLOCK_ACCEPTED opcode\n(or GOOD_BLOCK)\npayload: height, hash_prev_block, channel, nonce
    Solo->>Solo: Decode block-accepted payload
    Solo->>Handler: m_block_accepted_handler(\n  height, hash_prev_block,\n  channel, nonce)
    Handler->>Cache: m_mined_block_cache\n.record_accepted_block(\n  height, hash_prev_block,\n  channel, nonce)
    Cache-->>Cache: push_front to Tier 1\n(evict oldest to Tier 2 if > TIER1_MAX)
    Cache-->>Handler: done
    Handler-->>Solo: done
```

---

## Diagram 5 — End-to-End Defense-in-Depth Stack

Layered architecture showing all four defense layers stacked vertically.

```mermaid
flowchart TB
    subgraph L4["Layer 4 — Historical / compatibility dual-feed race source"]
        N1[SendChannelNotification — primary lane]
        N2[GET_BLOCK response — optional secondary compatibility path]
    end

    subgraph L3["Layer 3 — Worker-feed dedup guard (miner side)"]
        D[Dedup check: same height + hashPrevBlock\nwithin 2 000 ms debounce window]
        D -- duplicate --> DROP([⏱ Suppressed — workers NOT restarted])
        D -- unique / fork / expired --> PASS([✔ Passed through to workers])
    end

    subgraph L2["Layer 2 — Mutex-based recovery gate (m_recovery_mutex)"]
        M[std::mutex m_recovery_mutex\nSerialises create_workers + stop_all_workers]
        M -- blocked --> WAIT([⏳ Second thread waits])
        M -- acquired --> EXEC([✔ Single recovery executes])
    end

    subgraph L1["Layer 1 — MinedBlockCache (post-acceptance audit)"]
        C[record_accepted_block() → Tier 1 hot\nupdate_confirmations() on each template\nEviction: Tier 1 → Tier 2 → Tier 3]
    end

    subgraph L0["Layer 0 — Colin diagnostic report (operator visibility)"]
        R[emit_report() every 60 s\nLogs Mined Block History Top 5\nwith height, channel, confirmations, emoji]
    end

    N1 --> D
    N2 --> D
    PASS --> M
    EXEC --> C
    C --> R
```
