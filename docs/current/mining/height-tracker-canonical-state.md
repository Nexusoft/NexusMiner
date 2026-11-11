# HeightTracker: Canonical State Architecture

> **Developer reference** for the `CanonicalChainState` / `DiagnosticObserverState`
> separation introduced in the *HeightTracker: Separate canonical mining state from
> diagnostic telemetry* PR.  
> Applies to `src/protocol/inc/protocol/height_tracker.hpp` and
> `src/protocol/src/protocol/height_tracker.cpp`.

---

## 1. The Problem This Solves

Before this change, `HeightTracker` maintained a **single flat `Snapshot`** that mixed
two fundamentally different categories of data:

| Input | Frequency | Old Behaviour |
|-------|-----------|---------------|
| `OnKeepaliveResponse()` — SESSION_KEEPALIVE ACK | Every ~45 s | **Overwrote** `unified_height` + `channel_height` via `sync_channel_height_locked()` |
| `OnPushNotification()` — BLOCK_AVAILABLE payload | Every ~18 s | Overwrote the same fields |
| `OnTemplateMetadata()` / `OnBlockDataReceived()` — BLOCK_DATA | On template receipt | Overwrote the same fields |

Because the keepalive ACK arrives ~45 seconds after the last push, it could **regress
`channel_height` to a stale value**, making `is_template_stale()` return `false` when the
chain had already advanced.  Simultaneously, `fork_score` and `hash_tip_lo32` from the
keepalive polluted `check_template_health()`, producing:

- **False `FORK DETECTED`** — miner's `hashPrevBlock` compared against a 45 s-stale tip
- **`peak_fork_score=1` doom loop** — workers stopped permanently despite a valid template

---

## 2. Architectural Separation

`HeightTracker` now maintains **two completely separate state objects**:

```
┌──────────────────────────────────────────────────────────────┐
│                      HeightTracker                          │
│                                                              │
│  ┌─────────────────────────────────┐                         │
│  │      CanonicalChainState        │  ← Mining decisions     │
│  │  (m_canonical)                  │    template staleness   │
│  │                                 │    worker dispatch      │
│  │  canonical_unified_height       │    hashPrevBlock anchor │
│  │  canonical_channel_height  ───  │  ← monotonically        │
│  │  canonical_difficulty_nbits     │    advancing ONLY       │
│  │  canonical_channel_target       │                         │
│  │  canonical_hash_prev_block      │                         │
│  │  canonical_received_at          │                         │
│  └─────────────────────────────────┘                         │
│          ↑                                                    │
│          │ Updated ONLY by OnBlockDataReceived()              │
│          │                                                    │
│  ┌─────────────────────────────────┐                         │
│  │    DiagnosticObserverState      │  ← Colin diagnostics    │
│  │  (m_diagnostic)                 │    logging only         │
│  │                                 │    NEVER drives mining  │
│  │  push_unified_height            │                         │
│  │  push_channel_height  ──────── ─│─ OnPushNotification()  │
│  │  push_difficulty_nbits          │                         │
│  │                                 │                         │
│  │  round_unified_height           │                         │
│  │  round_channel_height  ─────── ─│─ OnGetRound()          │
│  │  round_difficulty_nbits         │                         │
│  │                                 │                         │
│  │  keepalive_unified_height       │                         │
│  │  keepalive_prime_height   ───── │─ OnKeepaliveResponse() │
│  │  keepalive_hash_height          │                         │
│  │  keepalive_stake_height         │                         │
│  │  keepalive_hash_tip_lo32        │  ← fork canary          │
│  │  keepalive_fork_score           │  ← diagnostic only      │
│  │  keepalive_peak_fork_score      │                         │
│  └─────────────────────────────────┘                         │
└──────────────────────────────────────────────────────────────┘
```

### The Core Invariant

> **Only `OnBlockDataReceived()` may update `CanonicalChainState`.**  
> Push notifications, GET_ROUND responses, and keepalive ACKs write to
> `DiagnosticObserverState` exclusively.  Canonical heights never regress.

---

## 3. Data Flow by Input Source

| Method | Writes To | Used For |
|--------|-----------|----------|
| `OnBlockDataReceived()` | `m_canonical` | All mining decisions |
| `OnTemplateMetadata()` | `m_canonical` (via `OnBlockDataReceived`) | Backward-compat wrapper |
| `OnTemplateReceived()` | `m_canonical.canonical_channel_target` | channel_target advance |
| `AdvanceChannelTarget()` | `m_canonical.canonical_channel_target` | Push staleness prevention |
| `UpdateWithHashPrevBlock()` | `m_canonical.canonical_hash_prev_block` | hashPrevBlock anchor |
| `OnPushNotification()` | `m_diagnostic.push_*` | Colin height display |
| `OnGetRound()` | `m_diagnostic.round_*` | Colin round display (TIP heights — no normalization) |
| `OnKeepaliveResponse()` | `m_diagnostic.keepalive_*` | Colin telemetry display |

---

## 4. Snapshot Composition (`GetSnapshot()`)

The backward-compatible `GetSnapshot()` composes its fields from both state objects.
The key rule: **canonical wins for mining-critical fields; diagnostic provides telemetry**.

| `Snapshot` field | Source | Rationale |
|-----------------|--------|-----------|
| `unified_height` | `max(canonical, push, round)` | Push/round-driven staleness + canonical protection |
| `channel_height` | `max(canonical, push, round)` | Push/round-driven staleness + canonical protection |
| `difficulty_nbits` | canonical if initialized, else push | Template difficulty is authoritative |
| `channel_target` | canonical only | Never corrupted by keepalive |
| `hash_prev_block` | canonical only | Fork detection anchor |
| `prime_height` | `keepalive_prime_height` | Diagnostic display |
| `hash_height` | `keepalive_hash_height` | Diagnostic display |
| `stake_height` | `keepalive_stake_height` | Diagnostic display |
| `hash_tip_lo32` | `keepalive_hash_tip_lo32` | Diagnostic canary |
| `fork_score` | `keepalive_fork_score` | Diagnostic canary |
| `peak_fork_score` | `keepalive_peak_fork_score` | Diagnostic canary |
| `last_keepalive_ack_at` | `keepalive_ack_at` | Session health |
| `last_template_update` | `canonical_received_at` | Post-push guard |
| `last_height_update` | `max(canonical_received_at, last_push_at)` | Post-push guard |

The `max(canonical, push, round)` composition for `channel_height` preserves all push- and
round-driven staleness detection (`is_template_stale()`, `is_tip_moved()`) while ensuring the keepalive
ACK can never regress the value below what a push, GET_ROUND, or BLOCK_DATA has already established.

---

## 5. The `hashPrevBlock` Canonical Anchor

`canonical_hash_prev_block` is set by `UpdateWithHashPrevBlock()` — called in both BLOCK_DATA
handlers immediately after `read_template()` succeeds.  It represents the chain tip at the
moment the current template was received.

This field is **the primary fork-detection anchor**: if the node's `hashBestChain` no longer
matches this value, the template is building on a stale tip.  Because it lives in
`CanonicalChainState`, it is never clobbered by a keepalive ACK.

```
BLOCK_DATA received
      │
      ├─ OnBlockDataReceived(unified, channel, nbits, {})   ← canonical metadata
      │
      ├─ read_template() → parse block header
      │
      └─ UpdateWithHashPrevBlock(tmpl->block.hashPrevBlock) ← canonical anchor
```

---

## 6. Fork Detection: Diagnostic Only

Prior to this change, `check_template_health()` used keepalive `fork_score` and
`hash_tip_lo32` to hard-stop workers:

```cpp
// OLD (removed): false positives during normal block advancement
if (ht_snap.fork_score > 0 && miner_lo32 != ht_snap.hash_tip_lo32) {
    stop_all_workers();  // ← fired on every keepalive if chain moved at all
}
```

After this change, keepalive fork data is **logged as a diagnostic canary only**:

```cpp
// NEW: informational only — workers never stopped by keepalive data
auto diag = solo_protocol->get_diagnostic_snapshot();
if (diag.keepalive_peak_fork_score > 0) {
    m_logger->warn("[Worker_manager] [Colin] FORK CANARY: peak_fork_score={} ...",
        diag.keepalive_peak_fork_score, ...);
}
// Do NOT stop workers here based on fork_score.
```

Real fork recovery is handled by the `TEMPLATE_ANCHOR` debounce in `solo.cpp`, which
compares successive `hashPrevBlock` values from BLOCK_DATA receipts.

---

## 7. `CanonicalChainState` Helper Methods

```cpp
/// True once canonical state has been set from at least one BLOCK_DATA receipt
bool is_initialized() const { return canonical_unified_height > 0; }

/// True when the chain's channel height has reached our target (template stale)
bool is_canonically_stale() const {
    return (canonical_channel_height > 0 && canonical_channel_target > 0 &&
            canonical_channel_height >= canonical_channel_target);
}

/// Difference between canonical unified height and canonical channel target.
/// Useful for Colin HEIGHT_DRIFT diagnostics.
int32_t height_drift_from_canonical() const;
```

---

## 8. `DiagnosticObserverState` Helper Methods

```cpp
/// True when peak_fork_score > 0 (any fork ever reported by node)
bool is_fork_canary_active() const { return keepalive_peak_fork_score > 0; }

/// True when node's reported best-chain tip differs from template's hashPrevBlock
/// (informational — fires for 0-4 s after every new block; NOT a hard-stop trigger)
bool is_tip_sync_mismatch(uint32_t canonical_hash_prev_lo32) const;

/// True when at least one diagnostic source (push, GET_ROUND, or keepalive) has provided data.
/// Diagnostic equivalent of CanonicalChainState::is_initialized().
bool is_initialized() const;

/// Most recent update time across push, GET_ROUND, and keepalive sources.
/// Diagnostic equivalent of CanonicalChainState::canonical_received_at.
/// Returns epoch time_point when no source has been received yet.
std::chrono::steady_clock::time_point latest_received_at() const;
```

---

## 9. New Public API

```cpp
// ── Primary canonical access ────────────────────────────────────────────────
void OnBlockDataReceived(uint32_t block_unified_height,
                         uint32_t metadata_channel_height,
                         uint32_t metadata_nbits,
                         const uint1024_t& hash_prev_block);

CanonicalChainState   GetCanonicalSnapshot()   const;  // for mining decisions
DiagnosticObserverState GetDiagnosticSnapshot() const;  // for Colin only

// ── Backward-compatible (unchanged call sites continue to work) ──────────────
Snapshot  GetSnapshot() const;     // composes from canonical + diagnostic
```

In `solo.hpp`:

```cpp
HeightTracker::CanonicalChainState    get_canonical_snapshot()   const;
HeightTracker::DiagnosticObserverState get_diagnostic_snapshot() const;
HeightTracker::Snapshot               get_height_tracker_snapshot() const;
```

---

## 10. Monotonic Guarantee

`OnBlockDataReceived()` only **advances** canonical heights — it never regresses them:

```cpp
if (block_unified_height > m_canonical.canonical_unified_height)
    m_canonical.canonical_unified_height = block_unified_height;

if (metadata_channel_height > m_canonical.canonical_channel_height) {
    m_canonical.canonical_channel_height = metadata_channel_height;
    // channel_target advances in lockstep
    uint32_t new_target = metadata_channel_height + 1;
    if (new_target > m_canonical.canonical_channel_target)
        m_canonical.canonical_channel_target = new_target;
}
```

A stale GET_BLOCK response (e.g. arriving after several pushes have already advanced the
tracker) cannot cause canonical heights to retreat.

---

## 11. Test Coverage

`src/protocol/height_tracker_test.cpp`:

| Test | What it verifies |
|------|-----------------|
| 27 | `OnPushNotification()`, `OnGetRound()`, `OnKeepaliveResponse()` do NOT initialize or update canonical state |
| 28 | A stale keepalive ACK after canonical initialization does NOT regress `canonical_unified_height`, `canonical_channel_height`, or `canonical_channel_target` |
| 29 | `OnBlockDataReceived()` is monotonic — stale BLOCK_DATA with lower heights is silently ignored |
| 30 | `fork_score` lives exclusively in `DiagnosticObserverState`; `GetCanonicalSnapshot()` has no fork field |
| 31 | `height_drift_from_canonical()` returns 0 when unified equals channel_target; non-zero otherwise |
| 32 | `DiagnosticObserverState::is_initialized()` — false before any data, true after any of push/GET_ROUND/keepalive; false when only BLOCK_DATA received |
| 33 | `DiagnosticObserverState::latest_received_at()` — epoch before any data; advances with each source; equals `last_keepalive_ack_at` after keepalive |

---

## 12. Related Documents

- [height-tracker-block-data-feed.md](height-tracker-block-data-feed.md) — two-step BLOCK_DATA feed sequence
- [unified-tip-vs-channel-height.md](unified-tip-vs-channel-height.md) — `is_template_stale()` and `is_tip_moved()` semantics
- [../mining-protocols/height-tracking.md](../mining-protocols/height-tracking.md) — GET_ROUND / 12-byte format
- `src/protocol/inc/protocol/height_tracker.hpp` — struct definitions
- `src/protocol/src/protocol/height_tracker.cpp` — implementation
- `src/protocol/height_tracker_test.cpp` — unit tests (Tests 27–31)
