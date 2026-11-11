# Hash Mining Flow

State machine for the hash channel mining loop (SK-1024 proof-of-work).

```mermaid
stateDiagram-v2
    [*] --> Connecting
    Connecting --> Authenticating: TCP established
    Authenticating --> SettingChannel: Falcon OK
    SettingChannel --> Ready: SET_CHANNEL(2)
    Ready --> WaitingPush: MINER_READY sent
    WaitingPush --> ReceiveTemplate: HASH_BLOCK_AVAILABLE
    ReceiveTemplate --> Hashing: Parse 228B template
    Hashing --> CheckStale: Increment nonce
    CheckStale --> Hashing: Still valid
    CheckStale --> ReceiveTemplate: New block pushed
    Hashing --> Submitting: Hash meets difficulty
    Submitting --> WaitingPush: BLOCK_ACCEPTED (200)
    Submitting --> Hashing: BLOCK_REJECTED (201)
    WaitingPush --> [*]: Connection lost
```

## Hash Mining Inner Loop

```mermaid
flowchart TD
    A[Receive Block Template] --> B[Extract nBits / Difficulty Target]
    B --> C[Initialize Nonce = 0]
    C --> D[Compute SK-1024 Hash]
    D --> E{Hash < Target?}
    E -- No --> F[Increment Nonce]
    F --> G{Stale Check}
    G -- Valid --> D
    G -- Stale --> H[Request New Template]
    H --> A
    E -- Yes --> I[SUBMIT_BLOCK with Solved Block]
    I --> J{Response}
    J -- BLOCK_ACCEPTED --> K[Wait for Next Push]
    J -- BLOCK_REJECTED --> F
```

## Key Details

- **Channel:** Hash (channel 2)
- **Push notification:** `HASH_BLOCK_AVAILABLE` (218) — sent on **any** channel block (universal PoW tip push)
- **Hashing algorithm:** SK-1024
- **GPU acceleration:** Supported via CUDA workers
- **Template format:** Same 228-byte structure as prime channel
- **Stale detection:** Two reasons trigger a template refresh:
  - `channel_advanced`: `channel_height >= channel_target` (own channel found block)
  - `tip_moved`: `unified_height > template_unified_height` (other channel found block — `hashPrevBlock` stale)
  - See [unified-tip-vs-channel-height.md](../../current/mining/unified-tip-vs-channel-height.md)
