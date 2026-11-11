# NexusMiner Push Notification Architecture

## Universal Push Flow (PR #164+, unified-height-driven model)

The node emits `PRIME_BLOCK_AVAILABLE` / `HASH_BLOCK_AVAILABLE` whenever **any**
channel advances the unified tip — not only when the miner's own channel mines.
Every PUSH means the unified tip moved, which changes `hashPrevBlock`, so the
miner **always** requests a fresh template.

```mermaid
graph TB
    subgraph "Node (LLL-TAO)"
        NP[Prime block accepted] --> EP[Emit PRIME_BLOCK_AVAILABLE]
        NH[Hash block accepted]  --> EH[Emit HASH_BLOCK_AVAILABLE]
        NS[Stake block accepted] --> EP
        NS --> EH
    end

    subgraph "Legacy Lane (Port 8323)"
        L1[PRIME_BLOCK_AVAILABLE<br/>0xD9] --> UH
        L2[HASH_BLOCK_AVAILABLE<br/>0xDA] --> UH
    end

    subgraph "Stateless Lane (Port 9323)"
        S1[STATELESS_PRIME_BLOCK_AVAILABLE<br/>0xD0D9] --> UH
        S2[STATELESS_HASH_BLOCK_AVAILABLE<br/>0xD0DA] --> UH
    end

    EP --> L1
    EH --> L2
    EP --> S1
    EH --> S2

    UH[PushNotificationHandler<br/>Unified Logic] --> CC{Cross-channel?}
    CC -->|Yes| XC[unified_height advanced?]
    XC -->|Yes| RX[Request Fresh Template<br/>hashPrevBlock changed]
    XC -->|No| LV[Liveness only]
    CC -->|No| VP[Validate Payload<br/>12/140/148 bytes]
    VP --> HT[Update HeightTracker<br/>unified_height, channel_height, nBits]
    HT --> CS{Channel stale?<br/>channel_height ≥ channel_target}
    CS -->|Yes| AT[AdvanceChannelTarget<br/>doom-loop prevention]
    CS -->|No| SH{Same-height hash<br/>mismatch?}
    SH -->|Yes| DT[Discard Template<br/>same-height reorg]
    SH -->|No| OK[No action needed]
    AT --> RW[ALWAYS Request Fresh Template<br/>PUSH = unified tip moved = hashPrevBlock changed]
    DT --> RW
    OK --> RW
```

### Key Principle: Unified-Height-Driven Model

Every PUSH from the node signifies a unified tip advance.  Every unified height
movement changes `hashPrevBlock`, so the mining template **must** be refreshed.
Channel heights are tracked informationally (doom-loop prevention, diagnostics)
but do **not** drive the template refresh decision.

Same-height dedup is unified-height-only via `GetBlockDedupGuard`.  The 100ms
rapid-burst guard prevents two identical pushes from racing.

## Payload Format

**12-byte compact (legacy):**
```
[0–3]   unified_height  — best-tip height across ALL channels
[4–7]   channel_height  — miner's own channel height
[8–11]  nBits           — compact difficulty target
```

**148-byte v2 (full-picture, preferred):**
```
[0–3]   unified_height
[4–7]   channel_height (own channel)
[8–11]  nBits (difficulty)
[12–15] other_channel_height (other PoW channel)
[16–19] stake_height
[20–147] hashBestChain (uint1024_t, 128 bytes LE)
```

See [unified-tip-vs-channel-height.md](../current/mining/unified-tip-vs-channel-height.md)
for full definitions of `unified_height`, `channel_height`, `channel_target`.
