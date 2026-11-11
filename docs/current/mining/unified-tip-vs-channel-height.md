# Unified Tip Anchoring and Template Refresh in NexusMiner

> **Canonical reference** for Nexus multi-channel mining architecture (PR #164+).
> All other docs should be consistent with the definitions here.

---

## 1. Key Concepts

### 1.1 Unified Best Tip (`hashBestChain`)

The Nexus blockchain maintains a **single unified best tip** — the most recently
accepted block across *all* channels (Stake, Prime, Hash).  The corresponding
block hash is `hashBestChain` on the node side.

Miners work with heights rather than hashes, but the same principle applies:

| Term | Description |
|------|-------------|
| `unified_height` | Height of the best block across **all** channels (bytes [0–3] of every push payload) |
| `channel_height` | Height of the most recent block on the **miner's subscribed channel** (bytes [4–7] of push payload) |
| `channel_target` | The block height the current template is **building toward** (`channel_height + 1` at template-receipt time) |

> **Critical insight**: A Prime block, a Hash block, or a Stake block all advance
> the unified tip.  Any advance means the `hashPrevBlock` field in an existing
> template is no longer the chain tip, making that template cryptographically
> stale regardless of whether the miner's own channel height changed.

### 1.2 `hashPrevBlock` vs `hashBestChain`

```
hashBestChain  → the hash of the current best block (any channel)
hashPrevBlock  → the hash stored in a template's block header, pointing to the
                 block the template extends
```

A template is **valid for submission** only when `hashPrevBlock == hashBestChain`.
If another channel finds a block after a template was issued, `hashBestChain`
advances but `hashPrevBlock` in the old template does not — the template is stale.

Miners never receive raw hashes over the wire; they only see the encoded heights
(`unified_height`, `channel_height`).  The two-reason refresh model (§ 2) maps
height changes back to these hash-level staleness conditions.

### 1.3 StakeMinter Analogy

The node's `StakeMinter` tracks `hashLastBlock` to detect when the unified tip
moves so it can rebuild its stake template.  NexusMiner performs the equivalent
check using `HeightTracker::Snapshot::is_tip_moved()`, which returns `true`
when `unified_height > template_unified_height` (the unified height recorded
when the current template was received).

---

## 2. Two Reasons to Refresh a Template

Every push notification may trigger one of two refresh reasons:

### Reason: `hashPrevBlock_mismatch_reorg`

**When**: The canonical `hash_prev_block` field in `HeightTracker` (set from the last
adopted `BLOCK_DATA` response) is non-zero and differs from the live template's
`hashPrevBlock`.

**Meaning**: The node has moved to a new chain tip at the **same unified height** via a
reorganization.  Height-based staleness (`channel_height >= channel_target`) and
`is_tip_moved()` both fail to detect this because the heights don't change — only the
tip hash changes.  Any block found on this template would be unconditionally rejected by
the node's Guard 2 check (`hashPrevBlock != hashBestChain`).

**Action**: Discard current template immediately, return `false` from
`validate_current_template()` so the caller (Worker_manager) triggers a `GET_BLOCK`
refresh.  This discard path is guarded by a consecutive-mismatch counter
(`m_hashprev_mismatch_consecutive`): after `MAX_CONSECUTIVE_HASHPREV_MISMATCHES` (3)
consecutive mismatches the template is **accepted** instead of discarded to prevent the
doom loop described in the Chain-in-Flux section below.

```
m_template_interface->discard_template("hashPrevBlock_mismatch_reorg")
```

Log signature:
```
[ValidateTemplate] ⚡ Unified Tip-Anchor Changed — hashPrevBlock mismatch (canonical=<hex>, template=<hex>) — discarding stale template [consecutive mismatch #N/3]
```

Chain-in-flux (accept) log signature:
```
[ValidateTemplate] ⚠️  Chain in flux: N consecutive hashPrevBlock mismatches (canonical=<hex>, template=<hex>) — accepting template to avoid doom loop (chain may be under attack / reorg storm)
```

> **Note:** `push_hash_prev_block` is **not** used as a discard trigger in
> `validate_current_template()`.  Using it would re-introduce the infinite
> soft-refresh loop: a push sets `push_hash_prev_block = H_new`, but the node's
> `BLOCK_DATA` response may legitimately return `hashPrevBlock = H_old` when the push
> was premature.  Only the canonical `hash_prev_block` (from the last adopted
> `BLOCK_DATA`) is authoritative.

### Chain-in-Flux Guard (DDoS / orphan-limit defense)

When a miner node is under a DDoS block-flood attack (e.g. a peer exceeding 500
`ACTION::GET::BLOCK` requests/60 s), the node may hit its orphan limit, drop
connections, and return inconsistent `BLOCK_DATA` responses under load.  This causes
the canonical `hash_prev_block` to disagree with successive `BLOCK_DATA` responses,
triggering repeated `hashPrevBlock_mismatch_reorg` discards in a tight loop and putting
workers into the `NO VALID TEMPLATE` state indefinitely.

The guard works in two layers:

1. **`Solo::validate_current_template()` consecutive-mismatch counter** — after
   `MAX_CONSECUTIVE_HASHPREV_MISMATCHES` (3) consecutive discards the next mismatch is
   treated as chain-in-flux and the template is accepted.  The counter resets on every
   successful template adoption.

2. **`Worker_manager` exponential GET_BLOCK backoff** — each consecutive mismatch
   increases the inter-request delay (2 s → 4 s → … up to 30 s), reducing load on an
   already-stressed node.  The backoff resets when the miner transitions to `HEALTHY`
   (template successfully adopted).

---

### Unified-Height-Driven Refresh (Current Model)

> **Note:** The old two-reason model (`channel_advanced` vs `tip_moved`) has been
> replaced by the unified-height-driven model.  Every PUSH means unified height
> moved, so the mining template is **always** refreshed.

**Channel staleness** (`channel_height >= channel_target`) is now **informational
only**.  `AdvanceChannelTarget()` has been removed (see Bug #6 fix in
`height_tracker.hpp`).  Post-refactor, `canonical_channel_target` is set ONLY by
`OnBlockDataReceived()` and `OnTemplateReceived()`, so doom-loop prevention is
structural rather than push-driven.

**Same-height tip replacement** (hash mismatch at same channel height) is still
detected and triggers a template `discard_template()` to force replacement.

**Every PUSH always requests fresh work** — the 100ms rapid-burst guard in
`GetBlockDedupGuard` prevents two identical pushes from racing.

### Decision Summary (Unified Model)

```
On every same-channel push notification:
  1. Update HeightTracker (unified_height, channel_height, difficulty)
  2. If extended push carries hashPrevBlock, store as tip anchor hint
  3. Channel staleness (informational — logged, no action):
       if channel_height >= channel_target → log warning (AdvanceChannelTarget removed)
  4. Same-height tip replacement (only when NOT channel-stale):
       if hash mismatch at same channel height → discard_template("same_height_tip_update")
  5. ALWAYS → request_work_fn()
       Every PUSH = unified tip moved = hashPrevBlock changed

On cross-channel push (different channel than miner's subscription):
  1. Record push liveness
  2. If notification_unified_height > snap.unified_height → request_work_fn()

Before any freshly validated template is fed live (validate_current_template):
  A. finalize channel target metadata
  B. UpdateWithHashPrevBlock(template.hashPrevBlock)  ← MUST come first
       [advances canonical anchor to the new template BEFORE the mismatch guard runs;
        prevents the doom loop where validate fires against a stale anchor and
        UpdateWithHashPrevBlock is never reached]
  C. ClearPushTipAnchor()
       [prevents spurious "push tip-anchor differs" advisory log from repeating]
  D. if canonical hash_prev_block != 0 AND template.hashPrevBlock != canonical hash_prev_block
       → discard template [reason: hashPrevBlock_mismatch_reorg]
       → return false (triggers GET_BLOCK refresh)
       [guards against templates buffered before the anchor advanced — i.e. a template
        that arrived during a reorg but was not yet discarded]
  E. if snap.has_same_height_push_tip_replacement(template.hashPrevBlock, template.nChannelHeight)
       → informational log only (advisory); do NOT discard (prevents infinite loop)
  F. otherwise feed workers
```

---

## 3. Push Payload Semantics

Every `PRIME_BLOCK_AVAILABLE` / `HASH_BLOCK_AVAILABLE` carries either:

- **12 bytes** on the compact/legacy form, or
- **140 bytes** on the extended/stateless form (`12-byte metadata + 128-byte hashPrevBlock (1024-bit hash)`)

Common metadata fields are big-endian:

```
Byte offset  Field               Type
──────────── ──────────────────  ─────────────────────
[0 – 3]      unified_height      uint32 BE
[4 – 7]      channel_height      uint32 BE   (Prime or Hash — miner's channel only)
[8 – 11]     nBits (difficulty)  uint32 BE   (compact target format)
[12 – 139]   hashPrevBlock       uint1024 LE (extended/stateless only)
```

The node sends this payload:
- **Immediately** after the miner subscribes (`MINER_READY`)
- **On every block accepted on any channel** (universal PoW tip push — see § 4)

The extended `hashPrevBlock` is authoritative for the node's newly announced tip
anchor, but it is still only a push-side hint inside the miner until a
replacement template is received and cross-checked. The miner must not use that
push hash alone as the final trigger for hard degraded-mode decisions; it first
enters the lighter template-swap path and escalates only if replacement stalls.

---

## 4. Universal PoW Tip Push

The node emits `PRIME_BLOCK_AVAILABLE` / `HASH_BLOCK_AVAILABLE` to subscribed
miners whenever **any** channel advances the unified tip:

| Event on node | Who receives a push |
|---------------|---------------------|
| Prime block accepted | All Prime-subscribed miners |
| Hash block accepted  | All Hash-subscribed miners  |
| Stake block accepted | All Prime miners **and** all Hash miners |

This ensures miners always refresh their template when `hashPrevBlock` changes.

---

## 5. PUSH is Primary; GET_ROUND is Backup

| Mechanism | Role | Latency |
|-----------|------|---------|
| `PRIME/HASH_BLOCK_AVAILABLE` push | **Primary** — event-driven, < 10 ms | Lowest |
| `GET_ROUND` / `NEW_ROUND` poll | **Backup** — timer-driven, every 5 s | Up to 5 s |

A miner that has successfully subscribed via `MINER_READY` should receive pushes
for every chain event.  `GET_ROUND` acts as a safety net to recover from missed
pushes or dropped connections.

Both sources update `HeightTracker` through `Solo::update_height_state()`.

---

## 6. Height Terminology Reference

Use these terms consistently across all documentation:

| Preferred term | Meaning | Old / avoid |
|----------------|---------|-------------|
| `unified_height` | Best-tip height across all channels | "height", "blockchain height" (ambiguous) |
| `channel_height` | Last accepted block on the miner's channel | "height" (ambiguous) |
| `channel_target` | Block height the template is mining toward | "template height" |
| `[reason: channel_advanced]` | channel_height reached channel_target | "stale" (unqualified) |
| `[reason: tip_moved]` | unified_height passed template_unified_height | "stale" (unqualified) |

### 6.1 Wire Protocol Height Semantics (TIP vs TARGET)

All height values in the mining protocol use one of two semantics:

| Source | Format | Bytes | Semantics | Value |
|--------|--------|-------|-----------|-------|
| BLOCK_DATA metadata | 12-byte prefix | [0–3] unified | **TIP** | `tStateBest.nHeight` |
| BLOCK_DATA metadata | 12-byte prefix | [4–7] channel | **TIP** | `stateChannel.nChannelHeight` |
| BLOCK_DATA metadata | 12-byte prefix | [8–11] nBits | Difficulty | compact target |
| Block header | 216-byte block | [200–203] nHeight | **TARGET** | `tStateBest.nHeight + 1` |
| GET_ROUND / NEW_ROUND | 16-byte response | [0–3] unified | **TIP** | `tStateBest.nHeight` |
| GET_ROUND / NEW_ROUND | 16-byte response | [4–7] prime | **TIP** | `stateChannel.nChannelHeight` (ch 1) |
| GET_ROUND / NEW_ROUND | 16-byte response | [8–11] hash | **TIP** | `stateChannel.nChannelHeight` (ch 2) |
| GET_ROUND / NEW_ROUND | 16-byte response | [12–15] stake | **TIP** | `stateChannel.nChannelHeight` (ch 0) |
| GET_HEIGHT response | 4-byte | [0–3] | **TARGET** | `nBestHeight + 1` |
| PUSH notification | 12-byte payload | [0–3] unified | **TIP** | `tStateBest.nHeight` |
| PUSH notification | 12-byte payload | [4–7] channel | **TIP** | `stateChannel.nChannelHeight` |

> **Key rule**: GET_ROUND sends TIP heights.  GET_HEIGHT sends TARGET (TIP + 1).
> They are inconsistent — GET_ROUND does **not** follow GET_HEIGHT's convention.
> NexusMiner applies **no normalization** to GET_ROUND values.

> **Node compatibility**: The 16-byte GET_ROUND format is specific to the
> [NamecoinGithub/LLL-TAO](https://github.com/NamecoinGithub/LLL-TAO) fork
> (branch `NODE`).  Standard Nexusoft/LLL-TAO sends NEW_ROUND with 0-byte payload.

---

## 7. HeightTracker Snapshot Fields

`HeightTracker::Snapshot` (see `src/protocol/inc/protocol/height_tracker.hpp`):

| Field | Set by | Used for |
|-------|--------|---------|
| `unified_height` | Push / GET_ROUND | `is_tip_moved()` comparison |
| `channel_height` | Push / GET_ROUND | `is_template_stale()` comparison |
| `difficulty_nbits` | Push / GET_ROUND | Worker difficulty target |
| `channel_target` | Template received | `is_template_stale()` comparison |
| `template_unified_height` | Template received | `is_tip_moved()` comparison |
| `hash_prev_block` | `OnBlockDataReceived` / `UpdateWithHashPrevBlock` | `hashPrevBlock_mismatch_reorg` discard check |
| `push_hash_prev_block` | Extended push notification | Advisory logging only (not a discard trigger) |

`is_template_stale()` → `channel_height >= channel_target` (both non-zero)  
`is_tip_moved()`      → `unified_height > template_unified_height` (both non-zero)

---

## 8. Debugging / Troubleshooting

### 8.1 Expected Log Patterns

| Scenario | Log line to look for |
|----------|----------------------|
| Channel stale (informational) | `Channel N block(s) behind … — advancing target` |
| Same-height reorg (discard) | `Same-height tip update — discarding template for replacement` |
| Normal PUSH refresh | `Requesting fresh {channel} template (PUSH → unified tip moved → hashPrevBlock changed)` |
| Cross-channel tip advance | `Cross-channel tip advance: unified X → Y — requesting fresh template` |
| hashPrevBlock reorg discard | `hashPrevBlock mismatch (canonical=..., template=...) — discarding stale template` |
| No template yet | `No template — requesting initial {channel} template` |
| Dedup suppressed | `[DedupGuard] height-match: suppressing` or `[DedupGuard] rapid-burst: suppressing` |
| Dedup bypass | `[DedupGuard] bypass_height — reason: push_stale` |

### 8.2 Diagnostic Checklist

| Symptom | Likely cause | Check |
|---------|-------------|-------|
| Template refreshed on every PUSH | Correct — unified model | Every PUSH requests fresh work |
| Cross-channel push requests fresh template | Correct — `hashPrevBlock` changed | Look for `Cross-channel tip advance` in logs |
| Template **not** refreshed after push | Bug or dedup suppression | Check `[DedupGuard]` logs for suppression |
| Template submitted and rejected as STALE | Push missed; submitted on old `hashPrevBlock` | Check for missed push notifications |
| Template discarded with `hashPrevBlock_mismatch_reorg` | Same-height chain reorg detected | Correct — canonical tip moved without height change; miner requests fresh work |
| Rapid GET_BLOCK suppressed | 100ms rapid-burst guard | Expected — prevents push races; check `[DedupGuard] rapid-burst` |

### 8.3 Verifying Push Subscription

After `MINER_READY`, look for:
```
[Solo Push] ✓ Subscribed to push notifications
[Solo Push]   Node will send immediate PRIME_BLOCK_AVAILABLE / HASH_BLOCK_AVAILABLE
```

If the subscription is dropped, `GET_ROUND` polling recovers state within 5 seconds
(check for `[GET_ROUND]` lines without corresponding push lines).

---

## 9. Related Documents

- [PROTOCOL_LANES.md](../../PROTOCOL_LANES.md) — wire format and lane architecture
- [push-notifications.md](../mining-protocols/push-notifications.md) — push opcode details
- [height-tracking.md](../mining-protocols/height-tracking.md) — GET_ROUND / 12-byte format
- [template-lifecycle.md](../../diagrams/mining-loops/template-lifecycle.md) — template lifecycle diagram
- [unified-push-architecture.md](../../diagrams/unified-push-architecture.md) — push flow diagram
