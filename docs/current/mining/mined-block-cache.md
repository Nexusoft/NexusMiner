# MinedBlockCache — Three-Tier Block Confirmation Cache

`MinedBlockCache` (`src/stats/inc/stats/mined_block_cache.hpp`) is a lightweight, in-memory
cache that records every block accepted by the Nexus network and tracks its confirmation
progress. It provides the data for the **Mined Block History** section of Colin's 60-second
diagnostic report.

---

## Overview

When the node sends a `BLOCK_ACCEPTED` or `GOOD_BLOCK` message, `Solo::process_messages()`
fires `m_block_accepted_handler`, which calls `Worker_manager::on_block_accepted()`, which
calls `MinedBlockCache::record_accepted_block()`. From that moment the block is tracked through
three tiers as the chain advances.

```
record_accepted_block()
        │
        ▼
┌───────────────────────────────────┐
│  Tier 1 — Hot  (max 5 blocks)    │  ← confirmation tracking active
│  Newest block at front            │
└───────────────────────────────────┘
        │ oldest evicted when Tier 1 full, OR
        │ confirmations >= CONFIRMATION_THRESHOLD
        ▼
┌───────────────────────────────────┐
│  Tier 2 — Warm  (max 100 blocks)  │  ← no active tracking
└───────────────────────────────────┘
        │ oldest evicted when Tier 2 full
        ▼
┌───────────────────────────────────┐
│  Tier 3 — Archive  (unbounded)    │  ← permanent record
└───────────────────────────────────┘
```

---

## Tier 1 — Hot (max 5 records)

**Purpose:** Active confirmation tracking for the most recently mined blocks.

**Fields stored per record (`MinedBlockRecord`):**

| Field | Type | Description |
|-------|------|-------------|
| `height` | `uint32_t` | Block height at which the block was accepted |
| `hash_prev_block` | `uint1024_t` | Full 128-byte hashPrevBlock |
| `channel` | `uint32_t` | 1 = Prime, 2 = Hash |
| `nonce` | `uint64_t` | Winning nonce |
| `accepted_at` | `steady_clock::time_point` | Wall time of acceptance |
| `confirmations` | `uint32_t` | Number of blocks on top of this block |

**Eviction policy:**
- New block pushed to the front (`push_front`).
- If `size > TIER1_MAX (5)`, the oldest record (back) is moved to Tier 2.
- A record whose `confirmations >= CONFIRMATION_THRESHOLD (5)` is also promoted to Tier 2
  during `update_confirmations()`, provided it is not the sole/front record (the front block
  always stays in Tier 1 until a newer block displaces it).

---

## Tier 2 — Warm (max 100 records)

**Purpose:** Historical record of confirmed mined blocks without active tracking overhead.

Tier 2 holds up to `TIER2_MAX = 100` records. New arrivals from Tier 1 go to the front; when
Tier 2 itself overflows, the oldest record is moved to Tier 3.

No `update_confirmations()` processing runs on Tier 2. The confirmation count is frozen at
whatever value the record had when it was promoted from Tier 1.

---

## Tier 3 — Archive (unbounded)

**Purpose:** Permanent audit trail of all mined blocks for the lifetime of the miner process.

Tier 3 has no size limit. Records pushed here are never deleted. This tier allows post-session
analysis and is intended for future log-export features.

---

## Confirmation Threshold

| Constant | Value | Meaning |
|----------|-------|---------|
| `CONFIRMATION_THRESHOLD` | 5 | Confirmations needed to change emoji and trigger Tier 1 → Tier 2 promotion |

- **Below threshold (< 5):** `status_emoji()` returns `"⛏"` (freshly mined)
- **At or above threshold (≥ 5):** `status_emoji()` returns `"✅"` (confirmed)

`update_confirmations(current_chain_height)` sets `confirmations = current_chain_height - height + 1`
for every Tier 1 record where `current_chain_height >= record.height`. Call this on every new
template arrival so the confirmation counts stay current.

---

## Colin Integration

Colin receives Tier 1 data through a callback registered via `set_mined_block_cache_source()`.

### Wiring (Worker_manager)

```cpp
m_colin_agent->set_mined_block_cache_source(
    [weak_wm]() -> std::vector<ColinAgent::MinedBlockSnapshot> {
        auto wm = weak_wm.lock();
        if (!wm) return {};
        const auto& tier1 = wm->m_mined_block_cache.tier1();
        // Convert each MinedBlockRecord to MinedBlockSnapshot ...
        return result;
    });
```

Each `MinedBlockSnapshot` carries: `height`, `channel`, `confirmations`,
`hash_prev_block_hex` (first 32 hex chars), `status_emoji`, `channel_name`.

### Report Output

The Colin report includes a "Mined Block History (Top 5)" section every 60 seconds:

```
[Colin]  ── Mined Block History (Top 5) ─────────────────
[Colin]    ✅ #1 height=4821043 channel=Prime confirmations=7 prev=0a3f9b2c1e7d4a56...
[Colin]    ⛏ #2 height=4821036 channel=Hash  confirmations=2 prev=f9a3120c456bde78...
[Colin]    ✅ #3 height=4821001 channel=Prime confirmations=8 prev=1c2d3e4f5a6b7c8d...
```

If no blocks have been mined yet the section prints `(no blocks mined yet)`.

---

## Thread Safety

> **Important:** All `MinedBlockCache` methods must be called on the **asio I/O thread**.
> `Worker_manager` serialises all callbacks on this thread, so no additional locking is required
> inside `MinedBlockCache` itself. Do not call these methods from worker threads or other
> concurrent contexts.

---

## Related

| Topic | File |
|-------|------|
| Cache header | `src/stats/inc/stats/mined_block_cache.hpp` |
| Worker_manager wiring | `src/worker_manager.cpp` |
| Colin report | `src/colin_agent.cpp` (`emit_report()`) |
| Solo handler | `src/protocol/src/protocol/solo.cpp` (`set_block_accepted_handler`) |
| Unit tests | `src/stats/mined_block_cache_test.cpp` |
| Architecture diagrams | `docs/diagrams/worker-feed-dedup-and-mined-block-cache.md` |
