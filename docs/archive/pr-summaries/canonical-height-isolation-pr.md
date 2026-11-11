# PR Summary: Canonical Height Isolation — Preventing False Fork Detection

## Problem

`OnKeepaliveResponse()` unconditionally overwrote `unified_height` and
`channel_height` via `sync_channel_height_locked()`.  Because keepalive ACKs
arrive on a 45-second cycle, they could carry heights that are **stale** relative
to push notifications that already advanced the tracker.  This caused:

1. **False `FORK DETECTED` hard stops** — the height tracker appeared to regress,
   and `check_template_health()` interpreted the difference as a chain fork.
2. **Permanent `peak_fork_score=1` doom loops** — once set, the fork canary was
   never cleared because every keepalive renewed the stale height, triggering the
   fork detection logic again.

### Root Cause

```
T0:  Push notification advances unified_height to 6533549
T45: Keepalive ACK arrives with stale unified_height=6533548
     → HeightTracker regresses unified_height from 6533549 → 6533548
     → check_template_health() sees height mismatch → "FORK DETECTED"
     → Workers hard-stopped, doom loop begins
```

## Solution: Two-Lane State Isolation

Split `HeightTracker` internal state into two strictly separated lanes:

### CanonicalChainState (The Authoritative Truth)

Updated **exclusively** by `OnBlockDataReceived()` and `UpdateWithHashPrevBlock()`.
Anchored to the decoded **216-byte Tritium block** from `BLOCK_DATA` /
`STATELESS_GET_BLOCK`.  Monotonically advancing — stale responses cannot regress
heights.

Fields (all use `canonical_*` prefix):
- `canonical_unified_height` — `block.nHeight` from BLOCK_DATA
- `canonical_channel_height` — `nChannelHeight` from metadata prefix
- `canonical_difficulty_nbits` — `nBits` from metadata prefix
- `canonical_channel_target` — `channel_height + 1` (block being mined)
- `canonical_hash_prev_block` — `hashPrevBlock` from decoded Tritium block (uint1024_t fork anchor)
- `canonical_received_at` — timestamp when canonical state was set
- `is_initialized()` — true when set at least once

### DiagnosticObserverState (Telemetry Only)

Updated by push notifications, keepalive ACKs, and GET_ROUND responses.
**Never drives mining decisions.** Read-only for diagnostics.

Notable field:
- `hash_tip_lo32` — lo32 of node's `hashBestChain` from keepalive ACK.  This is
  the diagnostic equivalent of `canonical_hash_prev_block` — a lightweight fork
  cross-check peer.

### Snapshot Composition

`GetSnapshot()` composes backward-compatible output:
```
unified_height = max(canonical, push, round)
channel_height = max(canonical, push, round)
```
Keepalive heights are **excluded from the composition entirely**.

## Data Flow

```
OnPushNotification()      → m_diagnostic.push_*
OnGetRound()              → m_diagnostic.round_*
OnKeepaliveResponse()     → m_diagnostic.keepalive_*  (hash_tip_lo32 = diagnostic peer)
OnBlockDataReceived()     → m_canonical                ← sole canonical writer
UpdateWithHashPrevBlock() → m_canonical.canonical_hash_prev_block
```

## Fork Detection Changes

`check_template_health()` fork/tip-mismatch detection was demoted to
**diagnostic-only** — it logs a warning with `keepalive_peak_fork_score` but
does NOT stop workers or discard templates.  Only canonical `hashPrevBlock`
changes (detected when the next `BLOCK_DATA` is decoded) trigger real fork
recovery.

## Files Changed

| File | Change |
|------|--------|
| `height_tracker.hpp` | Add `CanonicalChainState`, `DiagnosticObserverState`, `OnBlockDataReceived()`, `GetCanonicalSnapshot()`, `GetDiagnosticSnapshot()`, `height_drift_from_canonical()`. Snapshot gains `canonical_hash_prev_block` field. |
| `height_tracker.cpp` | Route each update method to correct lane. `OnTemplateMetadata()` delegates to `OnBlockDataReceived()`. `UpdateWithHashPrevBlock()` writes to `m_canonical.canonical_hash_prev_block`. |
| `solo.hpp` | Expose `get_canonical_snapshot()` / `get_diagnostic_snapshot()`. |
| `worker_manager.cpp` | Fork detection demoted to diagnostic-only warn. |
| `height_tracker_test.cpp` | Update tests 14, 15, 18, 19. Add tests 27 (canonical isolation), 30 (monotonicity), 31 (drift), 32 (hash_prev_block anchoring). |

## Invariant

> Only `OnBlockDataReceived()` and `UpdateWithHashPrevBlock()` may update
> canonical chain state.  Push, GET_ROUND, and keepalive ACKs update
> `DiagnosticObserverState` only.

## Test Coverage

| Suite | Tests |
|-------|-------|
| `height_tracker_test` | 165 pass |
| `phase2b_update_height_test` | 33 pass |
| `keepalive_v2_test` | 22 pass |
| **Total** | **220 pass** |

## Related Documents

- [Canonical Height Architecture Diagram](../../diagrams/mining-loops/canonical-height-architecture.md)
- [HeightTracker BLOCK_DATA Feed](../../current/mining/height-tracker-block-data-feed.md)
- [Unified Tip vs Channel Height](../../current/mining/unified-tip-vs-channel-height.md)
- [Push Notification Architecture](../../diagrams/unified-push-architecture.md)
