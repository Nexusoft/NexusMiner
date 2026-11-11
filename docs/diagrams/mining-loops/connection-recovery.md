# Connection Recovery

Reconnection and resume logic when the network connection is lost.

```mermaid
stateDiagram-v2
    [*] --> Connected
    Connected --> Disconnected: TCP error / timeout
    Disconnected --> Backoff: Start retry timer
    Backoff --> Reconnecting: Timer expired
    Reconnecting --> TCPEstablished: Socket connected
    Reconnecting --> Backoff: Connection refused
    TCPEstablished --> Authenticating: Send MINER_AUTH_INIT
    Authenticating --> Authenticated: MINER_AUTH_RESULT OK
    Authenticating --> Backoff: Auth failed
    Authenticated --> RestoringSession: Rebind session_id
    RestoringSession --> SettingChannel: SET_CHANNEL
    SettingChannel --> Ready: CHANNEL_ACK
    Ready --> Connected: MINER_READY sent
```

## Reconnection Sequence

```mermaid
sequenceDiagram
    participant Miner
    participant Node

    Note over Miner: Connection lost detected

    loop Retry with Backoff
        Miner->>Node: TCP SYN
        alt Connection Refused
            Note over Miner: Wait (backoff interval)
        else Connected
            Miner->>Node: MINER_AUTH_INIT (207)<br/>pubkey + miner_id
            Node->>Miner: MINER_AUTH_CHALLENGE (208)<br/>nonce
            Miner->>Node: MINER_AUTH_RESPONSE (209)<br/>Falcon-512 signature
            Node->>Miner: MINER_AUTH_RESULT (210)<br/>session_id + success
        end
    end

    Note over Miner: Genesis preserved — no reconfiguration needed

    Miner->>Node: SET_CHANNEL (channel 1 or 2)
    Node->>Miner: CHANNEL_ACK
    Miner->>Node: MINER_READY (216)
    Node->>Miner: BLOCK_AVAILABLE (217/218)

    Note over Miner: Mining resumes
```

## Session Management Details

- **Keepalive interval:** Configurable (default ~30 seconds)
- **Session binding:** `session_id` ties miner to template interface
- **Genesis preservation:** Reconnection reuses genesis config (no reconfig needed)
- **State machine:** `DISCONNECTED → AUTHENTICATED` managed by `session_manager`
- **Encryption:** ChaCha20 derived from `SHA256(KDF_DOMAIN || genesis_hash)`
