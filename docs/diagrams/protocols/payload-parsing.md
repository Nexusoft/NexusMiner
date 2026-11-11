# Payload Parsing

Big-endian parsing visualization for NexusMiner protocol payloads.

## Block Template Parsing (228 bytes)

```mermaid
flowchart TD
    A["Raw Payload (228 bytes)"] --> B["Bytes 0-3: nUnifiedHeight"]
    A --> C["Bytes 4-7: nChannelHeight"]
    A --> D["Bytes 8-11: nBits (difficulty)"]
    A --> E["Bytes 12-227: Serialized Block (216B)"]

    B --> B1["Big-endian uint32\n0x00 0x01 0xA4 0x10 → 107,536"]
    C --> C1["Big-endian uint32\n0x00 0x00 0xD2 0xF0 → 54,000"]
    D --> D1["Big-endian uint32\nDifficulty target bits"]
    E --> E1["Tritium Block Format\nHeader + Transactions"]
```

## Big-Endian Byte Order

```mermaid
flowchart LR
    subgraph "Network Byte Order (Big-Endian)"
        NB0["Byte 0\nMSB"] --> NB1["Byte 1"] --> NB2["Byte 2"] --> NB3["Byte 3\nLSB"]
    end
    subgraph "Example: Value 0x0001A410"
        V0["0x00"] --> V1["0x01"] --> V2["0xA4"] --> V3["0x10"]
    end
    subgraph "Host Value"
        H["107,536 (decimal)"]
    end
    V3 --> H
```

## Authentication Payload Parsing

### MINER_AUTH_INIT (207) — Miner → Node

```mermaid
flowchart TD
    P["AUTH_INIT Payload"] --> PK["Public Key\n(897 bytes for Falcon-512,\n1793 bytes for Falcon-1024)"]
    P --> MID["Miner ID\n(variable length string)"]
```

### MINER_AUTH_CHALLENGE (208) — Node → Miner

```mermaid
flowchart TD
    P["AUTH_CHALLENGE Payload"] --> N["Challenge Nonce\n(32 bytes, random)"]
```

### MINER_AUTH_RESPONSE (209) — Miner → Node

```mermaid
flowchart TD
    P["AUTH_RESPONSE Payload"] --> SIG["Falcon Signature\n(~666 bytes for Falcon-512,\n~1280 bytes for Falcon-1024)"]
```

### MINER_AUTH_RESULT (210) — Node → Miner

```mermaid
flowchart TD
    P["AUTH_RESULT Payload"] --> S["Success Flag\n(1 byte: 0x00=fail, 0x01=pass)"]
    P --> SID["Session ID\n(variable length, bound to template)"]
```

## Stateless Opcode Header Parsing

```
Received header bytes: [0xD0] [0x81]

Step 1: Combine to uint16 big-endian → 0xD081
Step 2: Check prefix: 0xD081 & 0xF000 == 0xD000 → Stateless protocol
Step 3: Extract opcode: 0xD081 & 0x0FFF == 0x0081 == 129 → GET_BLOCK
```

## Reward Binding Payload (ChaCha20 Encrypted)

```mermaid
flowchart TD
    P["MINER_SET_REWARD (213)"] --> ENC["ChaCha20 Encrypted Payload"]
    ENC --> DEC["Decrypt with session key\nDerived from SHA256(KDF_DOMAIN || genesis_hash)"]
    DEC --> ADDR["Reward Address\n(Nexus address for block rewards)"]
    DEC --> AAD["Authenticated via per-message AAD"]
```
