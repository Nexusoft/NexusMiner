# Colin Agent Hook Flow — Mermaid Diagrams

This document contains Mermaid diagrams showing how the three new Colin agent hooks
(`check_canonical_drift`, `check_diagnostic_initialized`, `check_diagnostic_freshness`)
plug into the diagnostic pipeline.

---

## Diagram 1: Colin Hook Input Routing

```mermaid
graph TD
    subgraph DataSources["HeightTracker Data Sources"]
        BD[BLOCK_DATA / STATELESS_GET_BLOCK]
        PUSH[Push Notification<br/>BLOCK_AVAILABLE]
        KA[Keepalive ACK<br/>SESSION_KEEPALIVE]
        GR[GET_ROUND / NEW_ROUND]
    end

    subgraph Canonical["CanonicalChainState (mining truth)"]
        C1[canonical_unified_height]
        C2[canonical_channel_height]
        C3[canonical_channel_target]
        C4[canonical_hash_prev_block]
        C5[canonical_received_at]
    end

    subgraph Diagnostic["DiagnosticObserverState (Colin telemetry)"]
        D1[push_unified_height / last_push_at]
        D2[round_unified_height / last_round_at]
        D3[keepalive_unified_height / fork_score / last_keepalive_ack_at]
    end

    subgraph ColinHooks["Colin Agent Hooks (run_diagnostics)"]
        H1["check_canonical_drift(drift)<br/>← height_drift_from_canonical()"]
        H2["check_diagnostic_initialized(is_init, elapsed)<br/>← is_initialized()"]
        H3["check_diagnostic_freshness(age, is_init)<br/>← latest_received_at()"]
    end

    BD -->|OnBlockDataReceived| Canonical
    PUSH -->|OnPushNotification| D1
    GR -->|OnGetRound| D2
    KA -->|OnKeepaliveResponse| D3

    Canonical -->|GetCanonicalSnapshot| H1
    Diagnostic -->|GetDiagnosticSnapshot is_initialized| H2
    Diagnostic -->|GetDiagnosticSnapshot latest_received_at| H3

    H1 -->|warning if drift > 500| W[Colin Warnings List]
    H2 -->|warning if uninitialized > 30s| W
    H3 -->|warning if silent > 180s| W

    W --> R[Colin Diagnostic Report]
```

---

## Diagram 2: `check_canonical_drift()` Decision Flow

```mermaid
flowchart TD
    A[run_diagnostics called] --> B{canonical.is_initialized?}
    B -- No --> C[Skip — no canonical data yet]
    B -- Yes --> D[drift = height_drift_from_canonical]
    D --> E{drift == 0?}
    E -- Yes --> F["Log: drift=0 ✓ (unified == channel_target)"]
    E -- No --> G{|drift| > 500?}
    G -- No --> H["Log: drift=N (inter-channel skew, normal)"]
    G -- Yes --> I["⚠ LOG WARN: drift=N outside expected range"]
    I --> J["Add recommendation: check BLOCK_DATA feed"]
```

---

## Diagram 3: `check_diagnostic_initialized()` Decision Flow

```mermaid
flowchart TD
    A[run_diagnostics called] --> B[diag = GetDiagnosticSnapshot]
    B --> C[elapsed = now - m_start_time]
    C --> D{diag.is_initialized?}
    D -- Yes --> E[No warning — at least one source received]
    D -- No --> F{elapsed >= 30s?}
    F -- No --> G[Within grace period — skip]
    F -- Yes --> H["⚠ LOG WARN: DiagnosticObserver uninitialized after Ns"]
    H --> I["Add recommendation: check push/keepalive"]

    style G fill:#d4edda,color:#000
    style E fill:#d4edda,color:#000
    style H fill:#f8d7da,color:#000
```

---

## Diagram 4: `check_diagnostic_freshness()` Decision Flow

```mermaid
flowchart TD
    A[run_diagnostics called] --> B[diag = GetDiagnosticSnapshot]
    B --> C{diag.is_initialized?}
    C -- No --> D[Skip — separate check covers this]
    C -- Yes --> E["latest = diag.latest_received_at()"]
    E --> F["age = now - latest"]
    F --> G{age > 180s?}
    G -- No --> H["Log: latest_received_at=Ns ago ✓"]
    G -- Yes --> I["⚠ LOG WARN: observer silent for Ns"]
    I --> J["Add recommendation: check node block propagation"]

    style H fill:#d4edda,color:#000
    style D fill:#d4edda,color:#000
    style I fill:#f8d7da,color:#000
```

---

## Diagram 5: Colin Report Section — Canonical + Diagnostic

```mermaid
sequenceDiagram
    participant Colin as ColinAgent
    participant HT as HeightTracker
    participant Log as Logger

    Note over Colin: emit_report() called

    Colin->>HT: GetCanonicalSnapshot()
    HT-->>Colin: CanonicalChainState

    alt canonical is_initialized
        Colin->>Log: [Colin] Canonical │ unified=N channel=M target=T nbits=...
        Colin->>Log: [Colin] Canonical │ height_drift_from_canonical=D (skew or 0)
    else not initialized
        Colin->>Log: [Colin] Canonical │ not yet initialized (no BLOCK_DATA)
    end

    Colin->>HT: GetDiagnosticSnapshot()
    HT-->>Colin: DiagnosticObserverState

    alt diag.is_initialized
        Colin->>Log: [Colin] Diagnostic │ latest_received_at=Ns ago ✓/⚠ push=... round=... ka=...
        opt age > 180s
            Colin->>Log: [Colin] Diagnostic │ ⚠ observer silent for Ns
        end
    else not initialized
        opt elapsed >= 30s
            Colin->>Log: [Colin] Diagnostic │ ⚠ uninitialized after Ns (no push/GET_ROUND/keepalive)
        end
    end
```

---

## Diagram 6: Full Colin Report Section Order

```mermaid
graph LR
    A["[Colin] DIAGNOSTIC REPORT"] --> B[Lane Health]
    B --> C[Block Stats]
    C --> D["Height Tracker (GetSnapshot)"]
    D --> E[Keepalive ACK Age]
    E --> F["► Canonical Chain State (NEW)<br/>height_drift_from_canonical"]
    F --> G["► Diagnostic Observer State (NEW)<br/>is_initialized + latest_received_at"]
    G --> H[TipSync Cross-Check]
    H --> I[SESSION_STATUS_ACK]
    I --> J[Warnings + Recommendations]
    J --> K[Node Diagnostics PING_DIAG]
    K --> L[Active Template]
    L --> M[Miner Telemetry PONG]

    style F fill:#cce5ff,color:#000
    style G fill:#cce5ff,color:#000
```
