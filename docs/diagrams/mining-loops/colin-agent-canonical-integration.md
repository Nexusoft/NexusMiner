# Colin Agent — Canonical vs Diagnostic Integration

> **Reference diagram** for how `ColinAgent` uses the two-lane HeightTracker
> state (canonical + diagnostic) for intelligent diagnostics via
> `height_drift_from_canonical()`, `DiagnosticObserverState::is_initialized()`,
> and `DiagnosticObserverState::latest_received_at()`.

---

## 1. Colin Agent Data Sources

```mermaid
graph TB
    subgraph "HeightTracker"
        CS["CanonicalChainState<br/>(BLOCK_DATA only)"]
        DS["DiagnosticObserverState<br/>(push / round / keepalive)"]
        SNAP["GetSnapshot()<br/>Composed: max(canonical, push, round)"]
    end

    subgraph "ColinAgent Hooks"
        DRIFT["height_drift_from_canonical()<br/>Snapshot method"]
        DINIT["is_initialized()<br/>DiagnosticObserverState method"]
        DLAT["latest_received_at()<br/>DiagnosticObserverState method"]
        CINIT["is_initialized()<br/>CanonicalChainState method"]
        CRA["canonical_received_at<br/>CanonicalChainState field"]
    end

    subgraph "Colin Diagnostic Report"
        CD["Canonical vs Diagnostic State"]
        HD["HeightDrift section"]
        DGS["DiagStale section"]
    end

    CS -->|"GetCanonicalSnapshot()"| CINIT
    CS -->|"GetCanonicalSnapshot()"| CRA
    DS -->|"GetDiagnosticSnapshot()"| DINIT
    DS -->|"GetDiagnosticSnapshot()"| DLAT
    SNAP -->|"GetSnapshot()"| DRIFT

    CINIT --> CD
    CRA --> CD
    DINIT --> CD
    DLAT --> CD
    DRIFT --> HD
    DLAT --> DGS

    style CS fill:#2d6a4f,color:#fff
    style DS fill:#d4a373,color:#000
    style SNAP fill:#264653,color:#fff
```

---

## 2. Three Colin Diagnostic Hooks

### `height_drift_from_canonical()` — Snapshot Method

Measures how far the composed snapshot heights (max of canonical, push, round)
have drifted from the canonical BLOCK_DATA path.

```
drift = unified_height − canonical_unified_height
```

| Drift | Meaning | Colin Action |
|-------|---------|--------------|
| 0 | Canonical caught up with push/round | ✅ Log info |
| +1 to +3 | Push/round slightly ahead — normal during BLOCK_DATA latency | ✅ Log info |
| > +3 | Push/round significantly ahead — BLOCK_DATA may be delayed | ⚠️ Warning + recommendation |
| < 0 | Anomaly — canonical ahead of composed snapshot | ⚠️ Warning (should not happen) |

### `DiagnosticObserverState::is_initialized()` — Struct Method

Returns `true` when any diagnostic data has been received (push, round, or keepalive).

```cpp
bool is_initialized() const {
    return push_unified_height > 0 || round_unified_height > 0 || keepalive_unified_height > 0;
}
```

Colin checks this to determine if diagnostic sources are feeding data:
- **Not initialized** → warn: "no push/round/keepalive data yet"
- **Initialized** → log diagnostic heights and latest timestamp

### `DiagnosticObserverState::latest_received_at()` — Struct Method

Returns the most recent timestamp across all diagnostic sources.
Diagnostic equivalent of `CanonicalChainState::canonical_received_at`.

```cpp
std::chrono::steady_clock::time_point latest_received_at() const {
    return std::max({last_push_at, last_round_at, last_keepalive_ack_at});
}
```

Colin checks this to detect all-source staleness:
- **< 120s** → fresh, no warning
- **≥ 120s** → all diagnostic sources have gone silent → warning

---

## 3. Colin Report Flow

```mermaid
flowchart TD
    START["ColinAgent::emit_report()"]

    START --> CS_CHECK{"canonical.is_initialized()?"}
    CS_CHECK -->|Yes| CS_LOG["Log canonical state<br/>unified, channel, target, age"]
    CS_CHECK -->|No| CS_WARN["⚠ NOT initialized<br/>(no BLOCK_DATA received)"]

    CS_LOG --> DS_CHECK{"diag.is_initialized()?"}
    CS_WARN --> DS_CHECK

    DS_CHECK -->|Yes| DS_LOG["Log diagnostic state<br/>push, round, keepalive heights + age"]
    DS_CHECK -->|No| DS_WARN["⚠ NOT initialized<br/>(no push/round/keepalive data)"]

    DS_LOG --> DRIFT_CHECK{"canonical.is_initialized()?"}
    DS_WARN --> DRIFT_CHECK

    DRIFT_CHECK -->|Yes| DRIFT["Compute height_drift_from_canonical()"]
    DRIFT_CHECK -->|No| STALE_CHECK

    DRIFT --> DRIFT_OK{"drift ≤ 3?"}
    DRIFT_OK -->|Yes| DRIFT_LOG["✅ Log drift (normal)"]
    DRIFT_OK -->|No| DRIFT_WARN["⚠ HeightDrift warning"]

    DRIFT_LOG --> STALE_CHECK
    DRIFT_WARN --> STALE_CHECK

    STALE_CHECK{"diag.is_initialized()?"}
    STALE_CHECK -->|Yes| STALE_AGE["Compute latest_received_at() age"]
    STALE_CHECK -->|No| DONE

    STALE_AGE --> STALE_OK{"age < 120s?"}
    STALE_OK -->|Yes| DONE["Continue to next section"]
    STALE_OK -->|No| STALE_WARN["⚠ DiagStale warning"]
    STALE_WARN --> DONE

    style CS_LOG fill:#2d6a4f,color:#fff
    style DS_LOG fill:#d4a373,color:#000
    style DRIFT_LOG fill:#2d6a4f,color:#fff
    style CS_WARN fill:#e9c46a,color:#000
    style DS_WARN fill:#e9c46a,color:#000
    style DRIFT_WARN fill:#e9c46a,color:#000
    style STALE_WARN fill:#e9c46a,color:#000
```

---

## 4. Warning Catalog: New Static Methods

| Method | Threshold | Trigger |
|--------|-----------|---------|
| `check_canonical_drift(drift)` | drift > 3 | Canonical BLOCK_DATA path lagging behind push/round |
| `check_diagnostic_staleness(age_s)` | age ≥ 120s | All diagnostic sources (push/round/keepalive) have gone silent |

Both return empty string when healthy, non-empty warning string when triggered.

---

## 5. Timestamp Coverage

```mermaid
flowchart LR
    subgraph "CanonicalChainState"
        CRA2["canonical_received_at<br/>Set by OnBlockDataReceived()"]
        CINIT2["is_initialized()<br/>canonical_unified_height > 0"]
    end

    subgraph "DiagnosticObserverState"
        LPA["last_push_at<br/>Set by OnPushNotification()"]
        LRA["last_round_at<br/>Set by OnGetRound()"]
        LKA["last_keepalive_ack_at<br/>Set by OnKeepaliveResponse()"]
        LRAT["latest_received_at()<br/>max(push, round, keepalive)"]
        DINIT2["is_initialized()<br/>any unified_height > 0"]
    end

    LPA --> LRAT
    LRA --> LRAT
    LKA --> LRAT

    style CRA2 fill:#2d6a4f,color:#fff
    style CINIT2 fill:#2d6a4f,color:#fff
    style LPA fill:#d4a373,color:#000
    style LRA fill:#d4a373,color:#000
    style LKA fill:#d4a373,color:#000
    style LRAT fill:#264653,color:#fff
    style DINIT2 fill:#264653,color:#fff
```

---

## 6. Related Documents

- [canonical-height-architecture.md](canonical-height-architecture.md) — Two-lane state isolation architecture
- [canonical-height-isolation-pr.md](../../archive/pr-summaries/canonical-height-isolation-pr.md) — Historical PR summary
- `src/colin_agent.hpp` — ColinAgent class definition
- `src/colin_agent.cpp` — ColinAgent implementation with diagnostic hooks
- `src/protocol/inc/protocol/height_tracker.hpp` — HeightTracker, CanonicalChainState, DiagnosticObserverState
