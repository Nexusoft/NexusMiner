# HeightTracker State Machine — Canonical vs Diagnostic

Data-flow diagrams for the `CanonicalChainState` / `DiagnosticObserverState`
architectural separation introduced to eliminate keepalive-driven FUBAR states.

---

## 1. Input Routing Overview

Every input to `HeightTracker` is now routed to exactly one lane:

```mermaid
flowchart TD
    subgraph INPUTS["Input Sources"]
        BD["BLOCK_DATA / STATELESS_GET_BLOCK\n(node's authoritative template response)"]
        PUSH["PRIME/HASH_BLOCK_AVAILABLE\n(12/140/148-byte push payload)"]
        GR["GET_ROUND / NEW_ROUND\n(polling response)"]
        KA["SESSION_KEEPALIVE ACK\n(32-byte: heights + fork_score + hash_tip_lo32)"]
    end

    subgraph CANONICAL["CanonicalChainState  (m_canonical)"]
        direction TB
        CU["canonical_unified_height\n(monotonically advancing)"]
        CC["canonical_channel_height\n(monotonically advancing)"]
        CT["canonical_channel_target\n(channel_height + 1)"]
        CN["canonical_difficulty_nbits"]
        CH["canonical_hash_prev_block\n(hashPrevBlock anchor)"]
        CR["canonical_received_at"]
    end

    subgraph DIAGNOSTIC["DiagnosticObserverState  (m_diagnostic)"]
        direction TB
        PU["push_unified_height\npush_channel_height\npush_difficulty_nbits\npush_prime_height\npush_hash_height\npush_stake_height"]
        RU["round_unified_height\nround_channel_height\nround_difficulty_nbits"]
        KU["keepalive_unified_height\nkeepalive_prime/hash/stake_height\nkeepalive_hash_tip_lo32\nkeepalive_fork_score\nkeepalive_peak_fork_score"]
    end

    BD -->|"OnBlockDataReceived()\nOnly canonical update path"| CANONICAL
    PUSH -->|"OnPushNotification()\npush_* fields only"| PU
    GR -->|"OnGetRound()\nround_* fields only"| RU
    KA -->|"OnKeepaliveResponse()\nkeepalive_* fields only\n⚠ NO sync_channel_height_locked()"| KU

    style CANONICAL fill:#1a3a1a,stroke:#4caf50,color:#e8f5e9
    style DIAGNOSTIC fill:#1a1a3a,stroke:#7986cb,color:#e8eaf6
    style BD fill:#2d4a1a,stroke:#8bc34a,color:#f0f4e8
    style KA fill:#3a1a2a,stroke:#e91e63,color:#fce4ec
```

---

## 2. `GetSnapshot()` Composition

`GetSnapshot()` composes the backward-compatible `Snapshot` struct from both state
objects.  The `max(canonical, push)` rule for heights is critical: it preserves push-driven
staleness detection while preventing keepalive regressions.

```mermaid
flowchart LR
    subgraph CAN["CanonicalChainState"]
        C1["canonical_unified_height"]
        C2["canonical_channel_height"]
        C3["canonical_channel_target"]
        C4["canonical_difficulty_nbits"]
        C5["canonical_hash_prev_block"]
        C6["canonical_received_at"]
    end

    subgraph DIAG["DiagnosticObserverState"]
        D1["push_unified_height"]
        D2["push_channel_height"]
        D3["push_difficulty_nbits"]
        D4["keepalive_prime/hash/stake_height"]
        D5["keepalive_hash_tip_lo32"]
        D6["keepalive_fork_score"]
        D7["keepalive_peak_fork_score"]
        D8["keepalive_ack_at"]
        D9["last_push_at"]
    end

    subgraph SNAP["Snapshot (backward-compat)"]
        S1["unified_height"]
        S2["channel_height"]
        S3["difficulty_nbits"]
        S4["channel_target"]
        S5["hash_prev_block"]
        S6["prime/hash/stake_height"]
        S7["hash_tip_lo32"]
        S8["fork_score / peak_fork_score"]
        S9["last_keepalive_ack_at"]
        S10["last_template_update"]
        S11["last_height_update"]
    end

    C1 --> |"max(canonical, push)"| S1
    D1 --> |"max(canonical, push)"| S1
    C2 --> |"max(canonical, push)"| S2
    D2 --> |"max(canonical, push)"| S2
    C4 --> |"if initialized"| S3
    D3 --> |"fallback"| S3
    C3 --> S4
    C5 --> S5
    D4 --> S6
    D5 --> S7
    D6 --> S8
    D7 --> S8
    D8 --> S9
    C6 --> S10
    C6 --> |"max"| S11
    D9 --> |"max"| S11
```

---

## 3. Keepalive Isolation — The Fix in Detail

This sequence shows why the old `sync_channel_height_locked()` call was dangerous
and how isolation eliminates the bug:

```mermaid
sequenceDiagram
    participant Node
    participant HT as HeightTracker
    participant WM as Worker_manager

    Note over HT: Canonical state: channel=2332107, target=2332108

    Node->>HT: OnPushNotification(unified=6611228, channel=2332107, nbits=…)
    Note over HT: push_channel_height=2332107<br/>Snapshot channel_height=max(2332107,2332107)=2332107

    Node->>HT: BLOCK_DATA received → OnBlockDataReceived(6611228, 2332107, nbits, hashPrev)
    Note over HT: canonical_channel_height=2332107<br/>canonical_channel_target=2332108<br/>canonical_hash_prev_block=0xABCD...

    Note over Node,HT: 45 seconds pass…

    Node->>HT: OnKeepaliveResponse(unified=6611150, prime=2332050, hash=9000, …)
    Note over HT: ✅ keepalive_* fields updated ONLY<br/>canonical_channel_height still 2332107<br/>channel_height in snapshot still 2332107

    WM->>HT: GetSnapshot()
    Note over HT: Snapshot: channel_height=max(2332107, 2332107)=2332107 ✅<br/>fork_score from diagnostic (informational)
    HT-->>WM: Snapshot{channel_height=2332107, channel_target=2332108}
    Note over WM: is_template_stale()=false ✅<br/>Workers continue mining 🎉

    Note over WM: ❌ OLD BEHAVIOUR (removed):<br/>sync_channel_height_locked() set<br/>channel_height=keepalive_hash_height=9000 ???<br/>→ FUBAR: channel_height=9000 << channel_target=2332108<br/>→ is_template_stale()=false (wrong — hides real staleness)<br/>OR fork_score triggered false FORK DETECTED stop
```

---

## 4. `hashPrevBlock` Canonical Protection

```mermaid
flowchart TD
    A["BLOCK_DATA packet received"] --> B["OnBlockDataReceived(\n  unified, channel, nbits, {}\n)"]
    B --> C["Read template bytes\nread_template(block_serial, endpoint)"]
    C --> D{"Validation\nsucceeded?"}
    D -->|No| E["Log error, request retry"]
    D -->|Yes| F["UpdateWithHashPrevBlock(\n  tmpl->block.hashPrevBlock\n)"]
    F --> G["m_canonical.canonical_hash_prev_block\n= hashPrevBlock\n(protected from keepalive)"]

    subgraph PROTECTION["Why it can't regress"]
        H["OnKeepaliveResponse() called\n45 s later"]
        H --> I["writes ONLY to\nm_diagnostic.keepalive_*"]
        I --> J["canonical_hash_prev_block\nunchanged ✅"]
    end

    G --> PROTECTION

    style G fill:#1a3a1a,stroke:#4caf50,color:#e8f5e9
    style J fill:#1a3a1a,stroke:#4caf50,color:#e8f5e9
```

---

## 5. Mining Decision Path (Workers use Canonical only)

```mermaid
flowchart TD
    START["Worker_manager::check_template_health()"]

    START --> SNAP["GetSnapshot()\n(backward-compat; channel_height = max canonical+push)"]
    SNAP --> ST{is_template_stale?\nchannel_height ≥ channel_target}
    ST -->|"Yes + pre-push"| STOP["Stop workers\nRetry GET_BLOCK"]
    ST -->|"No"| DRIFT{HEIGHT_DRIFT?\nunified > template+2}
    DRIFT -->|Yes| STOP
    DRIFT -->|No| FORKCHECK["get_diagnostic_snapshot()\n⚠ WARN only — no stop"]
    FORKCHECK --> AGE{Template age\n> emergency timeout?}
    AGE -->|Yes| STOP
    AGE -->|No| MINING["Continue mining ✅"]

    style FORKCHECK fill:#1a1a3a,stroke:#7986cb,color:#e8eaf6
    style MINING fill:#1a3a1a,stroke:#4caf50,color:#e8f5e9
    style STOP fill:#3a1a1a,stroke:#f44336,color:#ffebee
```

---

## 6. Colin Diagnostics vs Worker Mining — Read Path

```mermaid
flowchart LR
    HT["HeightTracker"]

    HT -->|"GetCanonicalSnapshot()"| CAN_R["CanonicalChainState\n(read-only snapshot)"]
    HT -->|"GetDiagnosticSnapshot()"| DIAG_R["DiagnosticObserverState\n(read-only snapshot)"]
    HT -->|"GetSnapshot()"| SNAP_R["Snapshot\n(composed, backward-compat)"]

    CAN_R --> WORKERS["Workers\n✅ Template staleness\n✅ Fork anchor check\n✅ Height drift\n✅ Worker dispatch"]
    DIAG_R --> COLIN["Colin Agent\n📊 Fork canary display\n📊 Live chain telemetry\n📊 Keepalive health\n📊 TipSync mismatch"]
    SNAP_R --> LEGACY["Legacy call sites\n(unchanged callers)"]

    style WORKERS fill:#1a3a1a,stroke:#4caf50,color:#e8f5e9
    style COLIN fill:#1a1a3a,stroke:#7986cb,color:#e8eaf6
    style LEGACY fill:#2a2a2a,stroke:#888,color:#eee
```

---

## Related Documents

- [height-tracker-canonical-state.md](../current/mining/height-tracker-canonical-state.md) — full architecture reference
- [unified-push-architecture.md](unified-push-architecture.md) — push notification flow
- [mining-loops/template-lifecycle.md](mining-loops/template-lifecycle.md) — template lifecycle
- `src/protocol/inc/protocol/height_tracker.hpp` — struct definitions
- `src/protocol/height_tracker_test.cpp` — Tests 27–31 (canonical isolation)
