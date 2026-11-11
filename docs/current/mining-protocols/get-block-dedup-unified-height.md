# GET_BLOCK Deduplication: Unified Height Only

## Overview

The GET_BLOCK deduplication logic is centralized in `GetBlockDedupGuard` (header-only
class in `get_block_dedup_guard.hpp`).  The guard suppresses redundant template requests
using a two-tier system:

1. **Rapid-burst guard (100ms)**: Prevents two code paths racing on the same event.
2. **Height-based guard**: Prevents redundant GET_BLOCK when unified height hasn't
   changed and a valid template already exists.

The guard compares **only `unified_height`** — channel height is intentionally excluded
because `hashPrevBlock` changes on every unified-height advance regardless of which
channel (Hash, Stake, Prime) mined the block.

## GetBlockReason Enum

Each `GET_BLOCK` request carries a `GetBlockReason` that expresses the *intent* of
the request.  The dedup policy is derived from the reason rather than hardcoded at
each call site.

```
Tier 1: bypass_all    → RECOVERY_FORCED, RECOVERY_TIMER, HEALTH_NO_TEMPLATE
         Skip all guards; degraded-mode retries and no-template health checks
         must always make progress.

Tier 2: bypass_height → PUSH_TIP_MOVED, PUSH_SAME_HEIGHT_TIP,
                         TEMPLATE_AGE_*, VALIDATION_FAILURE,
                         BLOCK_REJECTED, GET_ROUND_*, SESSION_REAUTH,
                         HEIGHT_DRIFT, HEALTH_CHANNEL_ADVANCE, etc.
         Skip height guard, keep rapid-burst guard.

Tier 3: full dedup    → INITIAL_REQUEST, HEALTH_STALE_SUPPRESSED
         Both guards active.
```

See `src/protocol/inc/protocol/get_block_reason.hpp` for the complete enum and
`should_bypass_all_dedup()` / `should_bypass_height_dedup()` policy functions.

---

## The False Assumption (Old Behavior)

The original guard compared the **(unified_height, channel_height)** pair:

```cpp
// OLD — INCORRECT
if (m_last_get_block_unified_height > 0 &&
    cur_unified == m_last_get_block_unified_height &&
    cur_channel == m_last_get_block_channel_height &&  // ← bug
    have_valid_template)
{
    // suppress GET_BLOCK
}
```

This assumed: *"If our channel height hasn't changed, we don't need a new template."*

That assumption is **wrong** because:

1. `hashPrevBlock` changes on **every** unified height advance, regardless of which
   channel (Hash, Stake, or Prime) mined the block.
2. A Prime miner's block must chain from the **current unified tip**.
3. When a Hash or Stake block advances the unified chain, Prime channel height stays
   the same — but the canonical `hashPrevBlock` is now different.

---

## The Failure Scenario

```
State before:
  unified_height  = 6650428
  channel_height  = 2347879  (Prime)
  template valid  = true
  last_get_block  = (unified=6650428, channel=2347879)

Event: Hash block found on network
  → unified_height  = 6650429   (+1)
  → channel_height  = 2347879   (unchanged — Hash block, not Prime)
  → hashPrevBlock   changed!    (new canonical tip)

GET_BLOCK dedup guard (old):
  cur_unified (6650429) != last_unified (6650428) → guard should PASS
  BUT: the template for unified=6650429 may have already been fetched
       at the same channel=2347879, so if the miner received a PUSH and
       issued GET_BLOCK for this unified height already, then:
  cur_unified == last_unified AND cur_channel == last_channel → SUPPRESSED ❌
```

From the logs showing the bug:
```
[Solo GET_ROUND] NEW_ROUND received but channel height unchanged; treating as OLD_ROUND/backoff
[Solo] GET_BLOCK height-based dedup: suppressing request
       (unified=6650429 channel=2347879 unchanged since last GET_BLOCK, template valid)
```

Even after `STATELESS_Prime_BLOCK_AVAILABLE` confirmed the chain advanced, the
GET_BLOCK was suppressed — the miner continued mining on a stale `hashPrevBlock`.

---

## The Fix (New Behavior)

The guard now compares **only `unified_height`**:

```cpp
// NEW — CORRECT
if (m_last_get_block_unified_height > 0 &&
    cur_unified == m_last_get_block_unified_height &&
    have_valid_template)
{
    // suppress GET_BLOCK — unified height unchanged, template is current
}
```

When `unified_height` advances (any block on any channel), the dedup guard passes
and a fresh GET_BLOCK is issued, fetching the new `hashPrevBlock`.

The `m_last_get_block_channel_height` member variable has been removed entirely.

---

## Flow Diagram

```
PUSH notification arrives
         │
         ▼
  PushNotificationHandler (unified-height-driven model)
  ├── Cross-channel? → unified advanced? → request_work_fn() ✅
  │
  └── Same-channel:
      1. Channel stale? → AdvanceChannelTarget (informational)
      2. Same-height hash mismatch? → discard_template
      3. NODE auto-sends BLOCK_DATA after PUSH — no GET_BLOCK needed
                │
                ▼
         Heights/state updated; node sends fresh template automatically
         GET_ROUND is the backup for tip changes.

GET_ROUND response (NEW_ROUND)
         │
         ▼
  unified_height changed? → get_work(GET_ROUND_STALE) → guard allows ✅
  unchanged?              → get_work(GET_ROUND_HEIGHT_PARITY) → bypass_height ✅
```

---

## Related: Channel Height in HeightTracker

Channel height is still tracked in `HeightTracker` and used for:
- `OnTemplateReceived()` — feeding the channel target for worker threads
- `GetSnapshot().channel_height` — diagnostic logging, doom-loop prevention
- `is_template_stale()` — channel-level staleness check (informational in push handler)

Channel height is **not** used as a dedup key for GET_BLOCK requests.

---

## Relation to Node-side nSequence Fix (PRs #479–#481)

The NODE-side PRs fixed the `hashPrevTx` race condition in `CreateTransaction()`
and `AcceptMinedBlock()`.  This NexusMiner fix is complementary: it ensures the
miner always requests a fresh template when the unified chain tip advances, so
the submitted block's `hashPrevBlock` is always correct regardless of which
channel mined the most recent block.

---

## Files Changed (Current State)

| File | Description |
|------|-------------|
| `src/protocol/inc/protocol/get_block_reason.hpp` | `GetBlockReason` enum — 22 named reasons replacing `bool bForce/bypass_dedup` |
| `src/protocol/inc/protocol/get_block_dedup_guard.hpp` | `GetBlockDedupGuard` — centralized two-tier dedup state machine (header-only) |
| `src/protocol/inc/protocol/push_notification_handler.hpp` | Simplified to 7 params (removed `recovery_initiated_fn`, `reset_dedup_fn`) |
| `src/protocol/src/protocol/push_notification_handler.cpp` | Unified-height-driven model: every PUSH always requests work |
| `src/protocol/inc/protocol/solo.hpp` | `get_work(GetBlockReason)` replaces `get_work(bool)`; `m_dedup_guard` replaces raw fields |
| `src/protocol/src/protocol/solo.cpp` | Dedup delegated to `m_dedup_guard`; `reset_get_block_dedup_state()` delegates to `m_dedup_guard.reset()` |
| `src/worker_manager.cpp` | All `retry_template_request()` calls use `GetBlockReason` |
| `src/protocol/get_block_dedup_recovery_test.cpp` | 71 tests including GetBlockReason dedup policy (Test 17, 29 assertions) |
| `src/protocol/push_notification_lane_test.cpp` | 109 tests covering unified-height-driven push handler |
