# HeightTracker BLOCK_DATA Feed Sequence

> **Developer reference** for the two-step HeightTracker feed introduced in
> PR #237 ("Feed HeightTracker critical fields from node BLOCK_DATA metadata").
> Applies to both the legacy lane (`src/protocol/src/protocol/solo.cpp` —
> `BLOCK_DATA` handler) and the stateless lane (`STATELESS_GET_BLOCK` handler).

---

## 1. The Two-Step Feed Sequence

Every time a `BLOCK_DATA` packet is received (on either lane),
`Solo::process_messages()` executes the following two-step sequence **before**
forwarding the block to `MiningTemplateInterface`:

```
Step 1/2 — Height + difficulty feed
────────────────────────────────────
update_height_state(unified_height, channel_height, nBits, PUSH)
  └─ calls HeightTracker::OnPushNotification(unified_height, channel_height, nBits)
  └─ calls ClientChannelManager::UpdateFromGetRound(unified_height, channel_height)

Step 2/2 — Channel target registration
───────────────────────────────────────
if (channel_height > 0)
    m_height_tracker.OnTemplateReceived(m_channel, channel_height + 1)
    // records channel_target = channel_height + 1
```

After Step 2, `UpdateWithHashPrevBlock(tmpl->block.hashPrevBlock)` anchors the
tip hash once the block header has been parsed from the payload.

### Why Three Separate Calls?

| Call | What it writes | Used by |
|------|----------------|---------|
| `OnPushNotification` | `unified_height`, `channel_height`, `difficulty_nbits`, `last_update_source=PUSH` | `is_template_stale()`, `is_tip_moved()`, worker difficulty |
| `OnTemplateReceived` | `channel_target`, `template_unified_height` | `is_template_stale()` single source of truth |
| `UpdateWithHashPrevBlock` | `hash_prev_block` | Fork-canary cross-check |

---

## 2. Why `block.nHeight` Must NOT Be Used as the Channel Target

`block.nHeight` (in the block header returned with the template) is the
**unified blockchain height** — it counts blocks across *all* channels and is
used as the input to the ProofHash computation.

The **channel target** is the *channel-specific* block height the miner is
trying to produce (`channel_height + 1`).  Using `block.nHeight` as the
channel target would cause `is_template_stale()` to compare the wrong values
and either never signal staleness (if unified > channel) or false-positive at
startup.

```
✗ Wrong:  channel_target = block.nHeight          ← unified height, not channel height
✓ Correct: channel_target = channel_height + 1    ← from BLOCK_DATA metadata bytes [4–7]
```

The authoritative source for `channel_height` is the 12-byte BLOCK_DATA
metadata prefix, **not** any field in the block header.

### TIP vs TARGET in BLOCK_DATA

The 228-byte BLOCK_DATA response contains two different height semantics:

| Component | Field | Semantics | Example (chain at height 6652588) |
|-----------|-------|-----------|-----------------------------------|
| 12-byte metadata prefix | `nUnifiedHeight` [0–3] | **TIP** | 6652588 |
| 12-byte metadata prefix | `nChannelHeight` [4–7] | **TIP** | 2348878 |
| 216-byte block body | `block.nHeight` [200–203] | **TARGET** | 6652589 (= TIP + 1) |

NexusMiner uses metadata TIP heights for `HeightTracker` (staleness detection,
template validation) and `block.nHeight` for ProofHash computation.  The
`channel_target` is derived as `metadata_channel_height + 1`, which matches
`block.nHeight`'s TARGET semantics for the channel dimension.

> **Node source**: See `NamecoinGithub/LLL-TAO:src/LLP/miner.cpp`
> `handle_get_round_stateless()` and `SharedGetBlockHandler()` for the
> authoritative metadata construction.

---

## 3. Authoritative Source Hierarchy (Post-Canonical Refactor)

After the `CanonicalChainState` / `DiagnosticObserverState` separation, the source
hierarchy no longer uses a single priority ordering — sources are now strictly separated
by purpose:

| Source | Writes To | Purpose |
|--------|-----------|---------|
| Node BLOCK_DATA metadata prefix | `CanonicalChainState` | All mining decisions (the only canonical update path) |
| Push notification (BLOCK_AVAILABLE) | `DiagnosticObserverState.push_*` | Push-driven staleness via `max(canonical, push)` composition |
| GET_ROUND / NEW_ROUND response | `DiagnosticObserverState.round_*` | Colin diagnostic display only |
| Keepalive ACK | `DiagnosticObserverState.keepalive_*` | Colin telemetry display only |

The key invariant: **only `OnBlockDataReceived()` updates `CanonicalChainState`**.
Keepalive and GET_ROUND data can never regress canonical heights.

See [height-tracker-canonical-state.md](height-tracker-canonical-state.md) for the full
architectural reference and the `GetSnapshot()` composition rules.

---

## 4. Staleness Detection — Single Source of Truth

`Solo::validate_current_template()` uses **only** `HeightTracker` to determine
whether the current template is stale.  It never reads from
`ClientChannelManager` for staleness decisions.

```cpp
// In validate_current_template():
auto snap = m_height_tracker.GetSnapshot();
if (snap.is_template_stale()) { ... }   // channel_height >= channel_target
if (snap.is_tip_moved())      { ... }   // unified_height > template_unified_height
```

`ClientChannelManager` holds a copy of the same data for legacy display/stats
purposes, but is not consulted for staleness.

---

## 5. Genesis Guard

Both lanes skip the `OnTemplateReceived` call when `channel_height == 0`:

```cpp
if (channel_height > 0) {
    m_height_tracker.OnTemplateReceived(m_channel, channel_height + 1);
}
```

This prevents a false-positive `is_template_stale()` at startup, where
`channel_target` would be set to 1 before any real height data is available.

---

## 6. Log Signatures

After a successful two-step feed, both lanes emit a single consolidated info line:

**Legacy lane:**
```
[Solo BLOCK_DATA] HeightTracker fed: unified={} channel={} nBits=0x{:08x} → channel_target={}
```

**Stateless lane:**
```
[Solo Stateless] HeightTracker fed: unified={} channel={} nBits=0x{:08x} → channel_target={}
```

If `channel_height == 0` (genesis guard), the `channel_target={}` part is omitted and no
`OnTemplateReceived` call is made.

---

## 7. Test Coverage

`src/protocol/phase2b_update_height_test.cpp` contains:

| Test | What it verifies |
|------|-----------------|
| Test 6 | `update_height_state() → OnTemplateReceived()` correctly sets `channel_target = channel_height + 1` and `template_unified_height` |
| Test 7 | Legacy and stateless lanes produce identical `HeightTracker` state after the same two-step feed |

---

## 8. Related Documents

- [canonical-height-architecture.md](../../diagrams/mining-loops/canonical-height-architecture.md) — Canonical vs Diagnostic state isolation architecture and diagrams
- [unified-tip-vs-channel-height.md](unified-tip-vs-channel-height.md) — `is_template_stale()` and `is_tip_moved()` semantics
- [height-tracking.md](../mining-protocols/height-tracking.md) — GET_ROUND / 12-byte format
- [stateless-mining.md](../mining-protocols/stateless-mining.md) — stateless lane overview
- `src/protocol/inc/protocol/height_tracker.hpp` — `HeightTracker` class (CanonicalChainState / DiagnosticObserverState)
- `src/protocol/src/protocol/solo.cpp` — `process_messages()` BLOCK_DATA and STATELESS_GET_BLOCK handlers
- [height-tracker-canonical-state.md](height-tracker-canonical-state.md) — `CanonicalChainState` / `DiagnosticObserverState` architecture (current authoritative reference)
- [unified-tip-vs-channel-height.md](unified-tip-vs-channel-height.md) — `is_template_stale()` and `is_tip_moved()` semantics
- [../mining-protocols/height-tracking.md](../mining-protocols/height-tracking.md) — GET_ROUND / 12-byte format
- [../mining-protocols/stateless-mining.md](../mining-protocols/stateless-mining.md) — stateless lane overview
- `src/protocol/inc/protocol/height_tracker.hpp` — `CanonicalChainState`, `DiagnosticObserverState`, `Snapshot` structs
- `src/protocol/src/protocol/height_tracker.cpp` — implementation
- `src/protocol/height_tracker_test.cpp` — unit tests (Tests 1–26 existing; Tests 27–31 canonical isolation)
