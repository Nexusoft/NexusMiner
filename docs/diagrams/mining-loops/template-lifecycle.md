# Template Lifecycle

From `GET_BLOCK` request to block submission and acceptance.

```mermaid
sequenceDiagram
    participant Miner
    participant Node
    participant Worker

    Miner->>Node: MINER_READY (216)
    Note over Node: Node queues miner for push notifications<br/>(push sent on ANY channel block — universal PoW tip push)

    Node->>Miner: PRIME_BLOCK_AVAILABLE (217)<br/>or HASH_BLOCK_AVAILABLE (218)
    Miner->>Node: GET_BLOCK (129 / 0xD081)
    Node->>Miner: BLOCK_DATA (0) — 228 bytes

    Note over Miner: Parse template:<br/>[0-3] nUnifiedHeight (uint32 BE)<br/>[4-7] nChannelHeight (uint32 BE)<br/>[8-11] nBits (uint32 BE)<br/>[12-227] Serialized block (216B)

    Note over Miner: Two reasons to request new template:<br/>1. channel_advanced: channel_height ≥ channel_target<br/>2. tip_moved: unified_height > template_unified_height<br/>(another channel found a block — hashPrevBlock stale)

    Note over Miner: GET_BLOCK pending lifecycle (PR #605 / #606):<br/>mark_pending() sets m_pending_get_block = true (atomic)<br/>Cleared unconditionally at top of on_block_data() / on_stateless_get_block()<br/>Auto-clears with logged timeout if BLOCK_DATA not received within timeout window<br/>Dedup guard is NOT reset on every template receive — only on fork detection

    Miner->>Worker: Distribute template to worker threads

    loop Mining
        Worker->>Worker: Search for valid nonce
        Worker->>Miner: Check stale status
    end

    Worker->>Miner: Found valid nonce
    Miner->>Node: SUBMIT_BLOCK (1 / 0xD001) — 216B solved block

    alt Accepted
        Node->>Miner: BLOCK_ACCEPTED (200)
        Note over Miner: Log reward, wait for next push
    else Rejected
        Node->>Miner: BLOCK_REJECTED (201)
        Note over Miner: Reason: STALE / INVALID_POW /<br/>INVALID_SIG / DUPLICATE / FORK
        Miner->>Worker: Resume mining current template
    end
```

## Template Validation

```mermaid
flowchart TD
    A[Receive 228B Payload] --> B{Format Detection}
    B -- "216B block" --> C[TRITIUM Format]
    B -- "≥220B block" --> D[LEGACY Format]
    B -- "Other" --> E[COMPACT Format]
    C --> F[Parse Metadata Fields]
    F --> G{merkle_valid?}
    G -- No --> H[Reject Template]
    G -- Yes --> I{height_valid?}
    I -- No --> H
    I -- Yes --> J{bits_valid?}
    J -- No --> H
    J -- Yes --> K{channel_valid?}
    K -- No --> H
    K -- Yes --> P{hashPrevBlock in CheckpointGuard OR mismatch <= MAX?}
    P -- Yes --> L[Template Ready for Mining]
    P -- No  --> Q[Warn: sustained flux — accept anyway]
    Q --> L
    L --> M[Feed to Worker Threads]
```

## Key Fields

| Offset | Size | Field | Encoding |
|--------|------|-------|----------|
| 0–3 | 4 bytes | nUnifiedHeight | uint32 big-endian |
| 4–7 | 4 bytes | nChannelHeight | uint32 big-endian |
| 8–11 | 4 bytes | nBits (difficulty) | uint32 big-endian |
| 12–227 | 216 bytes | Serialized block | Tritium format |
