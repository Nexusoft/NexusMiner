# Falcon Handshake Detail

Complete Falcon-512/1024 authentication handshake between miner and node.

## Full Handshake Sequence

```mermaid
sequenceDiagram
    participant Miner
    participant Node

    Note over Miner: Load Falcon-512 keypair<br/>(pubkey: 897B, privkey: 1281B)

    rect rgb(230, 240, 255)
    Note over Miner,Node: Phase 1: AUTH_INIT (Opcode 207 / 0xD0CF)
    Miner->>Node: MINER_AUTH_INIT
    Note right of Miner: Packet payload:<br/>[genesis_hash (32B)]<br/>[pubkey_len (2B, BE)]<br/>[pubkey (897B or ChaCha20-wrapped 925B)]<br/>[miner_id_len (2B, BE)]<br/>[miner_id (variable)]
    end

    rect rgb(255, 240, 230)
    Note over Miner,Node: Phase 2: AUTH_CHALLENGE (Opcode 208 / 0xD0D0)
    Node->>Miner: MINER_AUTH_CHALLENGE
    Note left of Node: Packet payload:<br/>[nonce_len (2B, BE)]<br/>[nonce (32B, random)]
    end

    rect rgb(230, 255, 230)
    Note over Miner,Node: Phase 3: AUTH_RESPONSE (Opcode 209 / 0xD0D1)
    Note over Miner: Sign nonce with Falcon-512<br/>private key (~690B signature)
    Miner->>Node: MINER_AUTH_RESPONSE
    Note right of Miner: Packet payload:<br/>[sig_len (2B, LE)]<br/>[signature (~690B)]
    end

    rect rgb(255, 230, 255)
    Note over Miner,Node: Phase 4: AUTH_RESULT (Opcode 210 / 0xD0D2)
    Node->>Miner: MINER_AUTH_RESULT
    Note left of Node: Packet payload:<br/>[status (1B: 0x00=fail, 0x01=pass)]<br/>[session_id (4B, LE, if success)]
    end

    alt Authentication Success
        Note over Miner: Store session_id<br/>Proceed to SET_CHANNEL
    else Authentication Failure
        Note over Miner: Retry with backoff
    end
```

## Handshake State Machine

```mermaid
stateDiagram-v2
    [*] --> Disconnected
    Disconnected --> SendingAuthInit: TCP connected
    SendingAuthInit --> WaitingChallenge: AUTH_INIT sent (207)
    WaitingChallenge --> SigningChallenge: AUTH_CHALLENGE received (208)
    SigningChallenge --> WaitingResult: AUTH_RESPONSE sent (209)
    WaitingResult --> Authenticated: AUTH_RESULT success (210)
    WaitingResult --> Disconnected: AUTH_RESULT failure
    WaitingChallenge --> Disconnected: Timeout
    Authenticated --> SettingChannel: Bind session_id
    SettingChannel --> Ready: CHANNEL_ACK
```

## Genesis-First Public Key Protection

When a genesis hash is available, the public key is wrapped with ChaCha20 before transmission.

```mermaid
flowchart TD
    A[Miner has genesis_hash?] --> B{Yes}
    A --> C{No}

    B --> D["Derive ChaCha20 key:<br/>SHA256(KDF_DOMAIN || genesis_hash)"]
    D --> E["Generate random nonce (12B)"]
    E --> F["Encrypt pubkey with ChaCha20-Poly1305<br/>AAD = 'FALCON_PUBKEY'"]
    F --> G["Wrapped pubkey (925B):<br/>[nonce(12)] [ciphertext(897)] [tag(16)]"]
    G --> H[Send wrapped pubkey in AUTH_INIT]

    C --> I[Send raw pubkey in AUTH_INIT]
    I --> J["Node receives raw pubkey (897B)"]
    H --> K["Node derives same key from genesis_hash"]
    K --> L["Decrypt and verify Poly1305 tag"]
    L --> M{Tag valid?}
    M -- Yes --> N[Accept pubkey]
    M -- No --> O[Reject connection]
```

## Falcon Signature Details

```mermaid
flowchart TD
    subgraph KeyGeneration["Key Generation (one-time)"]
        KG1["falcon_keygen(FALCON-512)"] --> KG2["Public key: 897 bytes"]
        KG1 --> KG3["Private key: 1281 bytes"]
    end

    subgraph ChallengeResponse["Challenge-Response Signing"]
        CR1["Receive 32-byte nonce from node"] --> CR2["sign_payload(nonce, private_key)"]
        CR2 --> CR3["Falcon-512 signature (~690 bytes)"]
        CR3 --> CR4["Encode: [sig_len(2B, LE)][signature]"]
    end

    subgraph Verification["Node-Side Verification"]
        V1["Extract signature from AUTH_RESPONSE"] --> V2["falcon_verify(nonce, signature, pubkey)"]
        V2 --> V3{Valid?}
        V3 -- Yes --> V4["Generate session_id (4B, LE)"]
        V3 -- No --> V5["Send AUTH_RESULT failure"]
    end
```

## Key Sizes Reference

| Component | Falcon-512 | Falcon-1024 |
|-----------|-----------|------------|
| Public key | 897 bytes | 1793 bytes |
| Private key | 1281 bytes | 2305 bytes |
| Signature | ~690 bytes | ~1280 bytes |
| Wrapped pubkey | 925 bytes | 1821 bytes |
| Security level | NIST Level I | NIST Level V |
