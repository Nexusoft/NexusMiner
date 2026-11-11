# HashCheckpointGuard

> Added in NexusMiner PR #607

## Purpose

`HashCheckpointGuard` is a fixed-capacity ring buffer that stores the last N
canonical tip hashes (`hashPrevBlock` values) seen in incoming BLOCK_DATA /
STATELESS_GET_BLOCK responses.  It is used by `validate_current_template()` in
`solo.cpp` to classify an incoming template's `hashPrevBlock` as either:

- **Known-canonical** — the hash appears in the checkpoint ring → no mismatch
  logged; template accepted as normal.
- **Recently-seen** — same as `m_last_known_hash_prev_block` → tip unchanged;
  debug log only.
- **Novel** — hash not in the ring and differs from last seen → tiered advisory
  log emitted; template still accepted (never discarded).

## Why hashPrevBlock discard was wrong

The prior 3-strike `discard_template()` path assumed that 3 consecutive
`hashPrevBlock` mismatches indicated a miner–node desync worth recovering from
by throwing away all templates.  In practice, on a multi-channel Nexus network
(Prime + Hash + Stake all contributing to `unified_height`), rapid tip changes
can produce 3+ mismatches in a single GET_BLOCK polling window during normal
network operation.  Discarding templates in that scenario caused a **doom loop**:
no valid template → workers stall → no blocks mined.

`HashCheckpointGuard` replaces this with a truth-based query: "Is this
`hashPrevBlock` a hash I have actually received from the canonical node?"  If
yes, the template is canonical regardless of how many mismatches have accumulated.

## Implementation

- **File**: `src/protocol/inc/protocol/hash_checkpoint_guard.hpp`
- **Capacity**: `HASH_CHECKPOINT_CAPACITY = 10` (last 10 canonical hashes)
- **Integration**: `Solo` class holds `m_hash_checkpoint_guard`; populated on
  every STATELESS_GET_BLOCK / BLOCK_DATA delivery via `record_checkpoint()` in
  `finalize_and_feed_current_template()`; queried in
  `validate_current_template()`.

## API

```cpp
class HashCheckpointGuard {
public:
    explicit HashCheckpointGuard(size_t capacity = 10);

    // Record a newly-observed canonical hashPrevBlock
    void add(const uint256_t& hash);

    // Returns true if `hash` is in the recent-canonical ring
    bool contains(const uint256_t& hash) const;

    // How many hashes are currently stored
    size_t size() const;
};
```

## Log output

When `validate_current_template()` consults the guard:

```
[ValidateTemplate] hashPrevBlock differs from canonical — node may be processing reorg
    (mismatch count = 1, advisory only)

[ValidateTemplate] Consecutive hashPrevBlock drift — chain tip churning
    (mismatch count <= MAX_CONSECUTIVE_HASHPREV_MISMATCHES, advisory only)

[ValidateTemplate] Sustained chain flux — N consecutive mismatches — node authoritative, accepting
    (mismatch count > MAX_CONSECUTIVE_HASHPREV_MISMATCHES, advisory only)
```

⚠️ In all cases the template is **accepted**. `discard_template()` and `return false`
paths for hashPrevBlock mismatch were removed in PR #607.

## Ring buffer diagram

```
HashCheckpointGuard (ring buffer, capacity = 10)
  ┌──────────────────────────────────────────────────────┐
  │ Stores last N canonical tip hashes (hashPrevBlock)   │
  │ populated from every BLOCK_DATA / STATELESS_GET_BLOCK│
  │ Queried by validate_current_template() before any    │
  │ mismatch counter logic                               │
  └──────────────────────────────────────────────────────┘
  contains(h) → true if h is a recently-seen canonical hash
  add(h)      → appends; evicts oldest when full
```

## Related documentation

- [template-age-policy.md](../../diagrams/protocols/template-age-policy.md) — Diagram 6
- [template-read-feed.md](template-read-feed.md) — rejection scenarios § hashPrevBlock
- [height-tracking.md](height-tracking.md) — unified_height as primary staleness gate
