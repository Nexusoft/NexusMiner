# ChaCha20 Encryption Flow

ChaCha20-Poly1305 authenticated encryption used in NexusMiner for protecting sensitive protocol data.

## Key Derivation

```mermaid
flowchart TD
    A["Genesis Hash (32 bytes)"] --> B["Prepend KDF domain string"]
    B --> C["preimage = 'nexus-mining-chacha20-v1' || genesis_hash"]
    C --> D["SHA256(preimage)"]
    D --> E["Session Key (32 bytes)"]

    style E fill:#4CAF50,color:#fff
    
    F["Both miner and node<br/>derive the SAME key<br/>from shared genesis_hash"] -.-> E
```

## Encryption Flow (Encrypt)

```mermaid
flowchart TD
    subgraph Inputs
        P["Plaintext<br/>(e.g., pubkey 897B, reward address)"]
        K["Session Key (32B)<br/>from SHA256(KDF || genesis)"]
        N["Random Nonce (12B)<br/>EVP_RAND_bytes()"]
        AAD["AAD Domain Tag<br/>(e.g., 'FALCON_PUBKEY',<br/>'REWARD_ADDRESS')"]
    end

    subgraph "ChaCha20-Poly1305 AEAD"
        E1["Initialize cipher context"]
        E2["Set key + nonce"]
        E3["Feed AAD for authentication"]
        E4["Encrypt plaintext → ciphertext"]
        E5["Finalize → Poly1305 tag (16B)"]
    end

    subgraph Output
        O["Wrapped payload:<br/>[nonce (12B)][ciphertext][tag (16B)]"]
    end

    P --> E1
    K --> E2
    N --> E2
    AAD --> E3
    E1 --> E2 --> E3 --> E4 --> E5
    E5 --> O
```

## Decryption Flow (Decrypt)

```mermaid
flowchart TD
    subgraph Input
        W["Wrapped payload:<br/>[nonce (12B)][ciphertext][tag (16B)]"]
        K["Session Key (32B)<br/>(derived from same genesis)"]
        AAD["Expected AAD Domain Tag"]
    end

    subgraph Parse
        PA1["Extract nonce (first 12B)"]
        PA2["Extract ciphertext (middle)"]
        PA3["Extract Poly1305 tag (last 16B)"]
    end

    subgraph "ChaCha20-Poly1305 Verify & Decrypt"
        D1["Initialize cipher context"]
        D2["Set key + extracted nonce"]
        D3["Feed AAD"]
        D4["Set expected tag"]
        D5["Decrypt ciphertext → plaintext"]
        D6["Verify Poly1305 tag"]
    end

    W --> PA1 --> D2
    W --> PA2 --> D5
    W --> PA3 --> D4
    K --> D2
    AAD --> D3
    D1 --> D2 --> D3 --> D4 --> D5 --> D6

    D6 --> V{Tag valid?}
    V -- Yes --> OK["Accept plaintext"]
    V -- No --> FAIL["Reject: tampered or wrong key"]
```

## Where ChaCha20 Is Used

```mermaid
flowchart TD
    subgraph "Protected Protocol Data"
        U1["AUTH_INIT (207)<br/>Public key wrapping"]
        U2["MINER_SET_REWARD (213)<br/>Reward address encryption"]
        U3["MINER_REWARD_RESULT (214)<br/>Reward confirmation encryption"]
    end

    subgraph "AAD Domain Separation"
        A1["'FALCON_PUBKEY'"]
        A2["'REWARD_ADDRESS'"]
        A3["'REWARD_RESULT'"]
    end

    subgraph "Not Encrypted"
        N1["Block templates (GET_BLOCK)"]
        N2["Block submissions (SUBMIT_BLOCK)"]
        N3["Push notifications"]
        N4["Height/reward queries"]
    end

    U1 --- A1
    U2 --- A2
    U3 --- A3
```

## Wrapped Payload Size Calculations

| Data | Plaintext | + Nonce (12B) | + Tag (16B) | Wrapped Total |
|------|-----------|---------------|-------------|---------------|
| Falcon-512 pubkey | 897 bytes | 909 bytes | 925 bytes | **925 bytes** |
| Falcon-1024 pubkey | 1793 bytes | 1805 bytes | 1821 bytes | **1821 bytes** |
| Reward address | variable | +12 bytes | +16 bytes | **+28 bytes** |

## Security Properties

- **RFC 8439 compliant:** ChaCha20-Poly1305 AEAD construction
- **Fresh nonce per operation:** 12-byte random nonce via `EVP_RAND_bytes()` prevents ciphertext reuse
- **Deterministic key derivation:** Both parties derive the same key from shared genesis hash
- **Domain separation via AAD:** Different AAD tags prevent cross-context decryption
- **Integrity protection:** Poly1305 tag detects any tampering with ciphertext or AAD
