# Colin Agent — Diagnostic Hooks (Canonical/Diagnostic Split)

**Context-Oriented LLP Intelligence Node**

This document describes the three new diagnostic hooks added to `ColinAgent` that expose
`height_drift_from_canonical()`, `DiagnosticObserverState::is_initialized()`, and
`DiagnosticObserverState::latest_received_at()` in the structured diagnostic report.

---

## 1. Why These Hooks Exist

Prior to the canonical/diagnostic split, `HeightTracker` held a single flat `Snapshot` that
mixed keepalive-sourced data with push-sourced data. This made it impossible for Colin to
distinguish between:

- **Canonical problems** (the BLOCK_DATA feed is broken, `canonical_channel_height` is wrong)
- **Diagnostic problems** (push notifications or keepalive ACKs have stopped arriving)

The three new hooks give Colin precise visibility into each category:

| Hook | What it answers |
|------|----------------|
| `check_canonical_drift()` | Is the BLOCK_DATA canonical state internally consistent? |
| `check_diagnostic_initialized()` | Has the miner received _any_ push/GET_ROUND/keepalive data yet? |
| `check_diagnostic_freshness()` | Has the miner received _recent_ push/GET_ROUND/keepalive data? |

---

## 2. Hook Catalog

### 2.1 `check_canonical_drift(int32_t drift)`

```cpp
static std::string ColinAgent::check_canonical_drift(int32_t drift);
```

Wraps `Snapshot::height_drift_from_canonical()` — **not** `CanonicalChainState::height_drift_from_canonical()`.

**Important distinction:**
- `Snapshot::height_drift_from_canonical()` = `unified_height − canonical_unified_height`
  (measures how far push/round heights are ahead of BLOCK_DATA canonical — a legitimate health signal)
- `CanonicalChainState::height_drift_from_canonical()` = `canonical_unified_height − canonical_channel_target`
  (measures a cross-dimension structural difference; on a 3-channel chain this is always ~4M and
  is NOT a health signal — it is logged as pure info in the Canonical Chain State section)

**When it fires:** `|drift| > WARN_CANONICAL_DRIFT_THRESHOLD` (500 blocks), applied **only** to
`Snapshot::height_drift_from_canonical()` (push vs canonical). A push running 500+ blocks ahead
of BLOCK_DATA indicates a real BLOCK_DATA delivery failure.

**Warning example (Snapshot drift):**
```
Canonical height drift=720 (|drift|>500) — unified vs channel_target skew outside expected range
```

**Recommendation added to report:**
```
Verify node is on the expected channel; check BLOCK_DATA feed
```

**Normal Canonical Chain State output (always info, never warning):**
```
[Colin]    Canonical │ height_drift_from_canonical=4279910 (inter-channel structural skew — normal on multi-channel chain)
```

---

### 2.2 `check_diagnostic_initialized(bool is_initialized, uint64_t elapsed_seconds)`

```cpp
static std::string ColinAgent::check_diagnostic_initialized(bool is_initialized,
                                                             uint64_t elapsed_seconds);
```

Wraps `DiagnosticObserverState::is_initialized()`.

`is_initialized()` returns `true` as soon as _any_ of push notifications, GET_ROUND responses,
or keepalive ACKs has been received. It returns `false` when only `OnBlockDataReceived()` has
been called (canonical data is isolated from diagnostic initialisation by design).

A **grace period** of `WARN_DIAGNOSTIC_INIT_GRACE_SECONDS` (30 s) is applied: the miner is
expected to receive its first push or keepalive ACK within the first few seconds of a session,
so Colin does not warn before the grace period expires.

**When it fires:** `!is_initialized && elapsed_seconds >= 30`

**Warning example:**
```
DiagnosticObserver uninitialized after 45s — no push notification, GET_ROUND, or keepalive ACK
received yet; check node connectivity
```

**Recommendation added to report:**
```
Ensure node is sending push notifications and keepalive ACKs
```

---

### 2.3 `check_diagnostic_freshness(uint64_t latest_age_seconds, bool is_initialized)`

```cpp
static std::string ColinAgent::check_diagnostic_freshness(uint64_t latest_age_seconds,
                                                           bool is_initialized);
```

Wraps `DiagnosticObserverState::latest_received_at()`.

`latest_received_at()` returns `max(last_push_at, last_round_at, last_keepalive_ack_at)` —
the most recent timestamp across _all_ diagnostic sources. If this timestamp ages beyond
`WARN_DIAGNOSTIC_STALE_SECONDS` (180 s), it means push notifications, GET_ROUND, and
keepalive ACKs have all stopped simultaneously — a strong indicator of a silent-death scenario.

The check skips itself when `is_initialized` is false (that case is covered by
`check_diagnostic_initialized` above).

**When it fires:** `is_initialized && latest_age_seconds > 180`

**Warning example:**
```
Diagnostic observer silent for 245s (no push/GET_ROUND/keepalive) — node may have dropped
the session or push notifications have stopped
```

**Recommendation added to report:**
```
Check node block propagation; verify keepalive interval is ≤60s
```

---

## 3. Emit-Report Section

All three hooks feed into a new `── Canonical Chain State ──` / `── Diagnostic Observer State ──`
section in the Colin diagnostic report, emitted each report interval.

**Sample output (healthy):**
```
[Colin]  ── Canonical Chain State ────────────────────────
[Colin]    Canonical │ unified=6628914 channel=2337423 channel_target=2337424 nbits=0x1d00ffff
[Colin]    Canonical │ height_drift_from_canonical=4291491 (inter-channel skew, normal)
[Colin]  ── Diagnostic Observer State ─────────────────────
[Colin]    Diagnostic │ latest_received_at=12s ago ✓ push_unified=6628914 round_unified=0 keepalive_unified=6628900
```

**Sample output (diagnostic silent):**
```
[Colin]  ── Canonical Chain State ────────────────────────
[Colin]    Canonical │ unified=6628914 channel=2337423 channel_target=2337424 nbits=0x1d00ffff
[Colin]    Canonical │ height_drift_from_canonical=4291491 (inter-channel skew, normal)
[Colin]  ── Diagnostic Observer State ─────────────────────
[Colin]    Diagnostic │ latest_received_at=210s ago ⚠ push_unified=6628200 round_unified=0 keepalive_unified=6628100
[Colin]    Diagnostic │ ⚠ observer silent for 210s — no push/GET_ROUND/keepalive
```

---

## 4. Wiring in `run_diagnostics()`

The hooks are called from `ColinAgent::run_diagnostics()` after the existing keepalive and
fork-score checks. They populate the `warnings` and `recommendations` vectors, which are
then logged in `emit_report()`.

```
run_diagnostics()
├── GetSnapshot()               → fork_score, hash_tip_lo32, keepalive age  (existing)
├── GetCanonicalSnapshot()      → logged as pure info in emit_report()        (structural skew)
└── GetDiagnosticSnapshot()     → is_initialized(), latest_received_at()     (NEW)
        ├── check_diagnostic_initialized(is_init, elapsed_s)
        └── check_diagnostic_freshness(age_s, is_init)
```

---

## 5. Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `WARN_CANONICAL_DRIFT_THRESHOLD` | 500 | Max expected `|Snapshot::height_drift_from_canonical()|` (push vs canonical) before warning. Not applicable to `CanonicalChainState::height_drift_from_canonical()` (cross-dimension skew, always large on multi-channel chain). |
| `WARN_DIAGNOSTIC_STALE_SECONDS` | 180 | Max age of `latest_received_at()` before "silent observer" warning |
| `WARN_DIAGNOSTIC_INIT_GRACE_SECONDS` | 30 | Grace period before "not yet initialized" warning fires |

---

## 6. Architectural Invariants Enforced

These hooks enforce the canonical/diagnostic split from the _operator-visibility_ layer:

- `check_canonical_drift()` reads **only** from `GetCanonicalSnapshot()` — it is never
  confused by stale keepalive heights.
- `check_diagnostic_initialized()` and `check_diagnostic_freshness()` read **only** from
  `GetDiagnosticSnapshot()` — `OnBlockDataReceived()` cannot mask a diagnostic silence.

Colin therefore provides independent visibility into both data lanes, making it possible
to distinguish a broken BLOCK_DATA feed from a broken push/keepalive feed.

---

## 7. Related Documents

- [`height-tracker-canonical-state.md`](height-tracker-canonical-state.md) — canonical/diagnostic split architecture
- [`height-tracker-block-data-feed.md`](height-tracker-block-data-feed.md) — BLOCK_DATA source hierarchy
- [`../../../docs/diagrams/height-tracker-state-machine.md`](../../../docs/diagrams/height-tracker-state-machine.md) — Mermaid diagrams
- [`../../../docs/diagrams/colin-agent-hooks.md`](../../../docs/diagrams/colin-agent-hooks.md) — Colin hook flow diagrams
