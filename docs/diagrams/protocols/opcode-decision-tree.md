# Opcode Decision Tree

Which opcode to use in each situation.

```mermaid
flowchart TD
    Start[What do you need to do?] --> A{Authentication?}
    A -- Yes --> A1{Step?}
    A1 -- "1. Initiate" --> AUTH_INIT["MINER_AUTH_INIT (207)\nSend pubkey + miner_id"]
    A1 -- "2. Respond to challenge" --> AUTH_RESP["MINER_AUTH_RESPONSE (209)\nSign nonce with Falcon-512"]
    A -- No --> B{Mining Setup?}

    B -- Yes --> B1{Action?}
    B1 -- "Set channel" --> SET_CH["SET_CHANNEL\nchannel 1 (prime) or 2 (hash)"]
    B1 -- "Signal ready" --> READY["MINER_READY (216)\nTell node you're ready for work"]
    B1 -- "Set reward address" --> REWARD["MINER_SET_REWARD (213)\nChaCha20 encrypted"]
    B -- No --> C{Mining Operations?}

    C -- Yes --> C1{Action?}
    C1 -- "Request block" --> GET_BLK["GET_BLOCK (129 / 0xD081)\nRequest new template"]
    C1 -- "Submit solution" --> SUB_BLK["SUBMIT_BLOCK (1 / 0xD001)\nSend 216B solved block"]
    C1 -- "Query height" --> GET_H["GET_HEIGHT (130 / 0xD082)\nCurrent blockchain height"]
    C1 -- "Query reward" --> GET_R["GET_REWARD (131)\nCurrent block reward"]
    C -- No --> D{Push Notifications?}

    D -- Yes --> D1{Direction?}
    D1 -- "From node" --> D2{Channel?}
    D2 -- "Prime" --> PRIME_AVAIL["PRIME_BLOCK_AVAILABLE (217)\nNew prime work available"]
    D2 -- "Hash" --> HASH_AVAIL["HASH_BLOCK_AVAILABLE (218)\nNew hash work available"]
    D -- No --> E{Response Handling?}

    E -- Yes --> E1{Result?}
    E1 -- "Block accepted" --> ACCEPTED["BLOCK_ACCEPTED (200)\nSolution was valid"]
    E1 -- "Block rejected" --> REJECTED["BLOCK_REJECTED (201)\nReasons: STALE, INVALID_POW,\nINVALID_SIG, DUPLICATE, FORK"]
```

## Opcode Reference Table

| Opcode | Value | Direction | Description |
|--------|-------|-----------|-------------|
| BLOCK_DATA | 0 | Node→Miner | Block template payload |
| SUBMIT_BLOCK | 1 | Miner→Node | Submit solved block |
| SET_CHANNEL | — | Miner→Node | Select prime (1) or hash (2) |
| GET_BLOCK | 129 | Miner→Node | Request block template |
| GET_HEIGHT | 130 | Miner→Node | Query blockchain height |
| GET_REWARD | 131 | Miner→Node | Query block reward |
| GET_ROUND | 133 | Miner→Node | Query current round |
| BLOCK_ACCEPTED | 200 | Node→Miner | Submission accepted |
| BLOCK_REJECTED | 201 | Node→Miner | Submission rejected |
| MINER_AUTH_INIT | 207 | Miner→Node | Start authentication |
| MINER_AUTH_CHALLENGE | 208 | Node→Miner | Auth challenge nonce |
| MINER_AUTH_RESPONSE | 209 | Miner→Node | Signed challenge |
| MINER_AUTH_RESULT | 210 | Node→Miner | Auth result + session_id |
| MINER_SET_REWARD | 213 | Miner→Node | Set reward address |
| MINER_REWARD_RESULT | 214 | Node→Miner | Reward binding result |
| MINER_READY | 216 | Miner→Node | Ready for push notifications |
| PRIME_BLOCK_AVAILABLE | 217 | Node→Miner | New prime block available |
| HASH_BLOCK_AVAILABLE | 218 | Node→Miner | New hash block available |
