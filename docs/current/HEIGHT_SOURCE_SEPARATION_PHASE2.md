# Height Source Separation — Phase 2: Research, Bug Analysis & TIP/TARGET Taxonomy

**Status:** ✅ IMPLEMENTED — All 6 bugs fixed, Colin diagnostics updated  
**Predecessor:** `HEIGHT_SOURCE_SEPARATION_PLAN.md` (Phase 1 — architecture plan)

---

## Table of Contents

1. [Clarifying Question Answers (Applied)](#1-clarifying-question-answers-applied)
2. [TIP vs TARGET Taxonomy — Complete Field Classification](#2-tip-vs-target-taxonomy)
3. [Bug Analysis — 6 Identified Issues](#3-bug-analysis)
4. [AdvanceChannelTarget() — Complete Teardown & Removal Assessment](#4-advancechanneltarget-teardown)
5. [ClientChannelManager — Complete Teardown & Removal Assessment](#5-clientchannelmanager-teardown)
6. [Optimization Paths](#6-optimization-paths)
7. [New Clarifying Questions for Decision](#7-new-clarifying-questions)

---

## 1. Clarifying Question Answers (Applied)

| # | Question | Answer | Impact on Plan |
|---|----------|--------|----------------|
| 1 | GET_ROUND always trigger GET_BLOCK? | **Only on HEIGHT CHANGE** | No change needed — current behavior is already "trigger on height change." The plan's trigger_* fields handle this correctly. |
| 2 | GET_ROUND heights for template discard? | **Only if GET_ROUND is the trigger source for GET_BLOCK, same as PUSH** | GET_ROUND-triggered discards happen the same way as PUSH-triggered discards: via trigger heights, not canonical. |
| 3 | ClientChannelManager kept or eliminated? | **Fold relevant parts into HeightTracker if sure. Leave alone if unsure.** | See Section 5 — detailed assessment of what's redundant vs unique. |
| 4 | AdvanceChannelTarget() preserved? | **Remove. Canonical should be the Channel Target Source.** | See Section 4 — removal plan with safety analysis. |
| 5 | External consumers of composite? | **Colin Diagnostics only. Keep per-category breakdowns.** | Composite max() removed from mining decisions. Colin keeps access to per-source heights. |

---

## 2. TIP vs TARGET Taxonomy

### 2.1 Definitions

| Term | Meaning | Example | Convention |
|------|---------|---------|------------|
| **TIP** | The height of the **last confirmed block** on the chain. The current chain state. | "The unified chain is at block 5000" | All internal tracking uses TIP semantics |
| **TARGET** | The height of the **next block we're trying to mine**. Always `TIP + 1`. | "We're mining block 5001" | Templates carry TARGET in `nChannelHeight` and `block.nHeight` |

**The Golden Rule:** `TARGET = TIP + 1`

### 2.2 Wire Format Semantics

| Wire Source | Field | Semantics on Wire | NexusMiner Normalizes To |
|---|---|---|---|
| BLOCK_DATA metadata [0-3] | unified_height | **TIP** | Stored as TIP ✓ |
| BLOCK_DATA metadata [4-7] | channel_height | **TIP** | Stored as TIP ✓ |
| BLOCK_DATA metadata [8-11] | difficulty | N/A | Stored as-is ✓ |
| BLOCK_DATA body [212-215] | block.nHeight | **TARGET** (tip + 1) | Used as TARGET ✓ |
| Template body | nChannelHeight | **TARGET** (channel_tip + 1) | Used as TARGET ✓ |
| PUSH notification [0-3] | unified_height | **TIP** | Stored as TIP ✓ |
| PUSH notification [4-7] | channel_height | **TIP** | Stored as TIP ✓ |
| GET_ROUND response [0-3] | unified_height | **TARGET** (tip + 1) | **Normalized to TIP** (−1 applied in solo.cpp:2346-2358) |
| GET_ROUND response [4-7] | prime_height | **TARGET** (tip + 1) | **Normalized to TIP** (−1 applied) |
| GET_ROUND response [8-11] | hash_height | **TARGET** (tip + 1) | **Normalized to TIP** (−1 applied) |
| GET_ROUND response [12-15] | stake_height | **TARGET** (tip + 1) | **Normalized to TIP** (−1 applied) |

### 2.3 Complete Field Classification

#### HeightTracker — CanonicalChainState

| Field | Classification | Source | Naming Correct? |
|-------|---|---|---|
| `canonical_unified_height` | **TIP** ✓ | BLOCK_DATA metadata [0-3] | ✅ Correct — "height" = TIP |
| `canonical_channel_height` | **TIP** ✓ | BLOCK_DATA metadata [4-7] | ✅ Correct |
| `canonical_channel_target` | **TARGET** ✓ | Computed: `channel_height + 1` | ✅ Correct — "target" = TARGET |
| `canonical_prime_height` | **TIP** ✓ | BLOCK_DATA (when channel=1) | ✅ Correct |
| `canonical_hash_height` | **TIP** ✓ | BLOCK_DATA (when channel=2) | ✅ Correct |
| `canonical_difficulty_nbits` | N/A | BLOCK_DATA metadata [8-11] | ✅ |
| `canonical_hash_prev_block` | N/A | Block body hashPrevBlock | ✅ |

#### HeightTracker — DiagnosticObserverState

| Field | Classification | Source | Naming Correct? |
|-------|---|---|---|
| `push_unified_height` | **TIP** ✓ | PUSH [0-3] | ✅ |
| `push_channel_height` | **TIP** ✓ | PUSH [4-7] | ✅ |
| `round_unified_height` | **TIP** ✓ | GET_ROUND (after −1 normalization) | ✅ |
| `round_channel_height` | **TIP** ✓ | GET_ROUND (after −1, channel-selected) | ✅ |
| `round_prime_height` | **TIP** ✓ | GET_ROUND (after −1) | ✅ |
| `round_hash_height` | **TIP** ✓ | GET_ROUND (after −1) | ✅ |
| `round_stake_height` | **TIP** ✓ | GET_ROUND (after −1) | ✅ |
| `keepalive_unified_height` | **TIP** ✓ | Keepalive ACK | ✅ |
| `keepalive_{prime,hash,stake}_height` | **TIP** ✓ | Keepalive ACK | ✅ |

#### HeightTracker — Snapshot (Current — Before Refactor)

| Field | Classification | Source | Naming Correct? |
|-------|---|---|---|
| `unified_height` | **TIP** (but COMPOSITE: `max(canonical, push, round)`) | Three TIP sources | ⚠️ Composite of TIPs — technically still a TIP, but from mixed authorities |
| `channel_height` | **TIP** (but COMPOSITE: `max(canonical, push, round)`) | Three TIP sources | ⚠️ Same problem |
| `channel_target` | **TARGET** (COMPOSITE: `max(m_channel_target, canonical)`) | Two TARGET sources | ⚠️ Dual-write target mixing |
| `template_unified_height` | **TIP** (captured at template receipt from COMPOSITE) | Snapshot at capture time | ⚠️ Captured from composite, should be from canonical |
| `push_channel_height` | **TIP** ✓ | Raw push value | ✅ |
| `prime_height`, `hash_height`, `stake_height` | **TIP** (composite of keepalive/round/push) | Diagnostic max() | ✅ Labeled diagnostic |
| `canonical_unified_height` | **TIP** ✓ | Copy of canonical | ✅ |
| `canonical_channel_height` | **TIP** ✓ | Copy of canonical | ✅ |

#### ClientChannelManager

| Field | Classification | Source | Naming Correct? |
|-------|---|---|---|
| `m_nNodeUnifiedHeight` | **TIP** ✓ | GET_ROUND (after normalization) | ✅ |
| `m_nNodeChannelHeight` | **TIP** ✓ | GET_ROUND (after normalization) | ✅ |
| `m_nPrevUnifiedHeight` | **TIP** ✓ | Previous GET_ROUND | ✅ |
| `m_nPrevChannelHeight` | **TIP** ✓ | Previous GET_ROUND | ✅ |

#### Template Fields (ClientBlockState)

| Field | Classification | Source | Naming Correct? |
|-------|---|---|---|
| `block.nHeight` | **TARGET** | Block body [212-215] | ⚠️ Named "Height" but is TARGET. Convention: this is the block's own number = unified TIP + 1 |
| `nChannelHeight` | **TARGET** | Set by `set_channel_height(channel_tip + 1)` | ⚠️ Named "Height" but is TARGET |

### 2.4 Naming Convention Going Forward

**Proposal:** All fields should use one of these suffixes:

| Suffix | Meaning | Example |
|--------|---------|---------|
| `_tip` | Current chain state (last confirmed block) | `canonical_unified_tip`, `push_channel_tip` |
| `_target` | Next block to mine (tip + 1) | `canonical_channel_target`, `template_unified_target` |

**For backward compatibility**, the existing `_height` suffix can remain, with the understanding that:
- `_height` without `_target` = **TIP** (the default meaning)
- `_target` = **TARGET** (explicitly labeled)

This is already mostly true in the codebase. The key fields that need attention are in the Snapshot struct where COMPOSITE heights blur the TIP/TARGET distinction.

---

## 3. Bug Analysis

### Bug #1: Channel Target Dual-Write Inconsistency ⭐ HIGH

**Severity:** HIGH  
**Type:** State Corruption Risk

**The Problem:** Three independent writers to channel_target fields, writing to DIFFERENT fields:

| Writer | Writes `m_channel_target` | Writes `canonical_channel_target` |
|--------|---|---|
| `OnTemplateReceived()` | ✅ Yes | ❌ No |
| `OnBlockDataReceived()` | ❌ No | ✅ Yes |
| `AdvanceChannelTarget()` | ✅ Yes | ✅ Yes |

`GetSnapshot()` returns `max(m_channel_target, canonical_channel_target)` — masking the asymmetry.

**Risk:** The two fields can diverge when OnTemplateReceived() advances the legacy field but not canonical, or when OnBlockDataReceived() advances canonical but not legacy. The max() in GetSnapshot() hides this divergence, but downstream code reading `GetCanonicalSnapshot().canonical_channel_target` gets a different value than `GetSnapshot().channel_target`.

**Fix (aligned with user's answer #4 — remove AdvanceChannelTarget):**
- Eliminate `m_channel_target` entirely
- `OnTemplateReceived()` should write to `canonical_channel_target`
- `GetSnapshot().channel_target` = `canonical_channel_target` (no more max())
- Remove `AdvanceChannelTarget()` (per user decision)

---

### Bug #2: Template Unified Height Captured from Composite ⭐ MEDIUM-HIGH

**Severity:** MEDIUM-HIGH  
**Type:** Semantic Error

**The Problem:** In `OnTemplateReceived()` (height_tracker.cpp:173-175):
```cpp
auto snap = build_snapshot_locked();
m_template_unified_height = snap.unified_height;  // ← COMPOSITE: max(canonical, push, round)
```

`is_tip_moved()` later compares this composite-captured value against the current composite. If a stale push had inflated the composite at capture time, subsequent fresh data may appear as "tip hasn't moved" (false negative).

**Fix:**
```cpp
m_template_unified_height = m_canonical.canonical_unified_height;  // ← CANONICAL ONLY
```

---

### Bug #3: Dedup Guard Uses Composite Height ⭐ MEDIUM

**Severity:** MEDIUM  
**Type:** Incorrect Guard Logic

**The Problem:** In `get_work()` (solo.cpp ~1137):
```cpp
auto snap = m_height_tracker.GetSnapshot();
uint32_t cur_unified = snap.unified_height;  // ← COMPOSITE
auto verdict = m_dedup_guard.check(reason, cur_unified, have_valid_template);
```

The dedup guard should use **trigger heights** (push/round) to determine "has something new arrived?", not composite heights that include canonical (which arrives later via BLOCK_DATA).

**Fix:** After the snapshot refactor, use `snap.trigger_unified_height` instead.

---

### Bug #4: validate_current_template() False Rejection ⭐ MEDIUM

**Severity:** MEDIUM  
**Type:** Timing-Dependent False Stale

**The Problem:** At solo.cpp:4640:
```cpp
if (tmpl->nChannelHeight <= snap.channel_height) {
    discard_template("Channel height stale");
}
```

`snap.channel_height` is `max(canonical, push, round)`. If PUSH advances it beyond what BLOCK_DATA reported, a valid BLOCK_DATA template gets falsely rejected.

**Example:** PUSH says channel=150, BLOCK_DATA arrives with template targeting 146 (valid), but `snap.channel_height = max(canonical=140, push=150) = 150`. Template 146 ≤ 150 → **FALSE REJECTION**.

**Fix:** After the snapshot refactor, `snap.channel_height` = canonical only. The comparison becomes `146 ≤ 140` → false → template accepted ✓.

---

### Bug #5: effective_channel_height Override ⭐ MEDIUM

**Severity:** MEDIUM  
**Type:** Wrong Source Selection

**The Problem:** In on_block_data() (~1898) and on_stateless_get_block() (~3711):
```cpp
if (ht_snap.channel_height > effectiveChannelHeight) {
    effectiveChannelHeight = ht_snap.channel_height;  // Overrides BLOCK_DATA with composite!
}
```

BLOCK_DATA metadata is the authoritative source. Overriding it with a composite that includes stale push data corrupts the canonical chain view.

**Fix:** Remove the override. BLOCK_DATA metadata is authoritative. Add diagnostic log if trigger heights are ahead (informational, not actionable).

---

### Bug #6: AdvanceChannelTarget() Corrupts Canonical Invariant ⭐ MEDIUM

**Severity:** MEDIUM  
**Type:** Invariant Violation

**The Problem:** `AdvanceChannelTarget()` writes to `canonical_channel_target` based on PUSH data. This creates:
```
canonical_channel_target = 161  (from push-derived advance)
canonical_channel_height = 155  (from actual BLOCK_DATA)
```

The invariant `canonical_channel_target = canonical_channel_height + 1` is violated. Subsequent staleness checks using `is_canonically_stale()` compare these misaligned values.

**Fix:** Remove `AdvanceChannelTarget()` entirely (per user decision). Canonical target is set only by `OnBlockDataReceived()`.

---

## 4. AdvanceChannelTarget() — Complete Teardown & Removal Assessment

### 4.1 Current Purpose

`AdvanceChannelTarget()` exists for **doom-loop prevention**: when a PUSH notification arrives and `is_template_stale()` is true (because push_channel_height reached or passed the template target), the method advances the target so subsequent PUSHes at the same channel height don't re-trigger recovery logic.

### 4.2 Call Site Analysis

There is **exactly 1 production call site**: `push_notification_handler.cpp:272`

```cpp
if (channel_stale && height_tracker) {
    height_tracker->AdvanceChannelTarget(snap.channel_height + 1);
}
```

### 4.3 Why It Can Be Safely Removed

After the Height Source Separation refactor:

1. **`is_template_stale()` will use canonical heights only** — PUSH data no longer inflates `snap.channel_height`. So the "doom loop" scenario (push advances composite → triggers stale → triggers recovery → stale again) **cannot occur** because push doesn't affect the channel_height used in `is_template_stale()`.

2. **Canonical target is set authoritatively by `OnBlockDataReceived()`** — When BLOCK_DATA arrives with the real channel height, `canonical_channel_target` is set to `channel_height + 1`. This is the correct target.

3. **PUSH's role becomes trigger-only** — PUSH says "something changed, go fetch a new block." It doesn't need to modify canonical state.

### 4.4 Removal Steps

1. **Delete** `AdvanceChannelTarget()` method from `height_tracker.hpp` and `height_tracker.cpp`
2. **Delete** `m_channel_target` legacy field from `HeightTracker` (fold into canonical only)
3. **Update** `push_notification_handler.cpp:272` — remove the call. The doom-loop prevention becomes unnecessary because `is_template_stale()` uses canonical heights.
4. **Update** `GetSnapshot()` — `channel_target` = `m_canonical.canonical_channel_target` directly (no more max())
5. **Update** `OnTemplateReceived()` — write to `m_canonical.canonical_channel_target` instead of legacy `m_channel_target`
6. **Update tests** — `height_tracker_test.cpp` tests 21 and 34, `degraded_recovery_test.cpp` test 2

### 4.5 Safety Assessment

| Concern | Assessment |
|---------|------------|
| Doom-loop without advance | ✅ Safe — doom loop can't occur when is_template_stale() uses canonical only |
| Push-driven staleness detection | ✅ Safe — PUSH triggers GET_BLOCK via trigger heights, not by modifying canonical |
| Template target regression | ✅ Safe — OnBlockDataReceived() is monotonic |
| Test coverage | ⚠️ Need to update 4 test cases |

**Verdict: SAFE TO REMOVE** (post-refactor, when canonical-only heights are in effect)

---

## 5. ClientChannelManager — Complete Teardown & Removal Assessment

### 5.1 Current Functionality

ClientChannelManager provides **three capabilities**:

| Capability | How It Works | Unique to CCM? |
|---|---|---|
| **Fork Detection** (2+ block rollback) | Compares current GET_ROUND unified height vs previous | ⚠️ Could be done in HeightTracker |
| **Phantom Stake Detection** (1-block rollback) | Same comparison, but 1-block threshold | ⚠️ Could be done in HeightTracker |
| **Template Validation** (ValidateTemplate) | Checks nHeight == unified+1 and nChannelHeight > channel | ❌ NOT USED in production (Solo uses validate_current_template() instead) |

### 5.2 Usage in Production

| Method | Call Count | Purpose | Could HeightTracker Do It? |
|---|---|---|---|
| `UpdateFromGetRound(u, c)` | 2 (via apply_channel_manager_update) | Store GET_ROUND heights, set fork flags | ✅ Yes — HeightTracker.OnGetRound already stores these |
| `IsForkDetected()` | 3 (apply_channel_manager_update, sync_template_state, handle_fork_detected) | Check 2+ block regression | ✅ Yes — compare round_unified_height vs previous |
| `IsPhantomStakeRegression()` | 1 (apply_channel_manager_update) | Check 1-block regression | ✅ Yes — same comparison with 1-block threshold |
| `ClearForkFlag()` | 2 (apply_channel_manager_update, handle_fork_detected) | Reset detection state | ✅ Yes — HeightTracker could maintain these flags |
| `GetPreviousHeights()` | 2 (apply_channel_manager_update, handle_fork_detected) | Log rollback delta | ✅ Yes — HeightTracker could store previous GET_ROUND heights |
| `GetChannelName()` | 3 (sync_template_state, handle_fork_detected) | "PRIME" or "HASH" string | ✅ Yes — trivial |
| `ValidateTemplate()` | **0** (NOT USED in production) | — | N/A |
| `SetCurrentTemplate()` | **0** (NOT USED in production) | — | N/A |
| `GetCurrentTemplate()` | **0** (NOT USED in production) | — | N/A |

### 5.3 What's Redundant vs What's Unique

**Fully Redundant (HeightTracker already does it):**
- Height storage from GET_ROUND — `HeightTracker.OnGetRound()` already stores these
- Channel identification — HeightTracker already knows the channel

**Unique (HeightTracker does NOT currently do it):**
- **Fork detection (2+ block regression)** — HeightTracker has monotonic guards that PREVENT regression, so it cannot detect regression. It would need to store previous heights to detect rollback.
- **Phantom Stake detection (1-block regression)** — Same: needs previous height comparison.
- **Previous height storage** — HeightTracker doesn't store "the previous GET_ROUND height" — it only stores the monotonically-advanced current.

### 5.4 Removal Assessment

**To remove ClientChannelManager, HeightTracker would need:**
1. `uint32_t m_prev_round_unified_height` — previous GET_ROUND unified height
2. `uint32_t m_prev_round_channel_height` — previous GET_ROUND channel height
3. `bool m_fork_detected` — set when unified rollback > 1 block
4. `bool m_phantom_stake_detected` — set when unified rollback == 1 block
5. A `ClearForkFlags()` method
6. Logic in `OnGetRound()` to compare incoming vs previous and set flags

**Risk Assessment:**

| Concern | Risk Level | Mitigation |
|---------|------------|------------|
| Missing something subtle in CCM | ⚠️ Medium | CCM's production interface is well-defined (6 methods). All can be mapped. |
| Test breakage | ⚠️ Medium | `client_channel_manager_test.cpp` tests the standalone class. Would need adaptation. |
| Architecture regression | Low | CCM is already a thin wrapper around 4 atomics and 2 bools |
| Two-class channel management (Prime/Hash) | Low | HeightTracker already handles channel via `m_channel` member |

### 5.5 Recommendation

**Per user guidance: "If unsure just leave it alone. Only remove if you're sure of its parts being redundant."**

**My assessment:** ClientChannelManager's production-facing functionality is **simple enough to fold into HeightTracker** (6 methods, 6 atomic fields, 2 boolean flags). However, the removal requires:
1. Adding previous-height tracking to HeightTracker
2. Adding fork/phantom-stake detection to HeightTracker
3. Updating all call sites in solo.cpp
4. Adapting tests

**Recommendation: LEAVE ALONE for now.** The refactor is mechanical but non-trivial, and the risk of missing an edge case is not zero. The bigger win (eliminating composite heights) can be done independently. ClientChannelManager removal can be a separate PR after the composite elimination is proven stable.

If you decide to proceed with removal in a future phase, the path is clearly mapped above.

---

## 6. Optimization Paths

### 6.1 Eliminate Legacy `m_channel_target` Field

**Impact:** Removes Bug #1 entirely  
**When:** Part of AdvanceChannelTarget removal (Section 4)

After removal:
- `OnTemplateReceived()` writes to `m_canonical.canonical_channel_target`
- `GetSnapshot().channel_target` reads from `m_canonical.canonical_channel_target` directly
- No more max() composition for targets

### 6.2 Add Canonical Regression Detection

**Currently missing:** HeightTracker's `OnBlockDataReceived()` has monotonic guards (`if (new > old) advance`), but does NOT log when `new < old` (a regression). This means blockchain reorgs are silently ignored at the canonical level.

**Proposal:** Add regression logging (not blocking) in `OnBlockDataReceived()`:
```cpp
if (block_unified_height < m_canonical.canonical_unified_height) {
    m_logger->warn("[HeightTracker] ⚠️ CANONICAL REGRESSION: unified {} → {} (delta: {})",
                   m_canonical.canonical_unified_height, block_unified_height,
                   m_canonical.canonical_unified_height - block_unified_height);
}
```

### 6.3 Type-Safe TIP/TARGET Wrappers

The existing `UnifiedHeight` and `ChannelHeight` wrapper types could be extended to carry TIP/TARGET semantics:

```cpp
struct UnifiedTip   { uint32_t value{0}; };
struct UnifiedTarget { uint32_t value{0}; };
struct ChannelTip   { uint32_t value{0}; };
struct ChannelTarget { uint32_t value{0}; };
```

This would make TIP→TARGET conversions explicit and compiler-enforced. Low priority but high long-term value.

### 6.4 Per-Source Diagnostic Dashboard

After composite elimination, Colin agent should show:

```
┌─────────────── Height Sources ──────────────────┐
│ CANONICAL  unified_tip=5000  channel_tip=2300   │
│ PUSH       unified_tip=5001  channel_tip=2301   │
│ GET_ROUND  unified_tip=5000  channel_tip=2300   │
│ KEEPALIVE  unified_tip=4998  channel_tip=2299   │
│                                                  │
│ TRIGGER    unified_tip=5001  channel_tip=2301   │
│ DRIFT      trigger ahead by 1 block             │
└──────────────────────────────────────────────────┘
```

Each source visible independently — no more "what height is this?" confusion.

---

## 7. New Clarifying Questions for Decision

### Q1: OnTemplateReceived() — Should It Write to Canonical Channel Target?

**Current:** `OnTemplateReceived()` writes to the legacy `m_channel_target` field only.  
**After AdvanceChannelTarget removal:** `m_channel_target` is eliminated.

**Question:** Should `OnTemplateReceived()` write to `canonical_channel_target`?

- **Option A: YES** — The template we received has a channel target. Store it in canonical.
  - Pro: `canonical_channel_target` stays in sync with the template we're mining.
  - Con: Template channel target comes from `effective_channel_height + 1`, which currently includes the composite override (Bug #5). After Bug #5 fix, it comes from BLOCK_DATA metadata only, which is correct.
  
- **Option B: NO** — Only `OnBlockDataReceived()` writes canonical target.
  - Pro: Strict canonical purity — only raw BLOCK_DATA updates canonical.
  - Con: `OnTemplateReceived()` updates `m_channel` and `m_template_unified_height` in canonical context. It's weird to update some fields but not target.

**My recommendation:** Option A, but only AFTER Bug #5 is fixed (so the effective_channel_height is truly from BLOCK_DATA, not composite).

### Q2: Push Doom-Loop Prevention — Still Needed Post-Refactor?

**Current:** `AdvanceChannelTarget()` prevents push-triggered doom loops.  
**After refactor:** `is_template_stale()` uses canonical only, so PUSH can't trigger false staleness.

**But:** Could there be a scenario where:
1. PUSH arrives with channel_tip = 150
2. Canonical channel_tip is already 150 (from prior BLOCK_DATA)
3. Canonical channel_target = 151
4. `is_template_stale()` = (150 >= 151)? NO — not stale
5. Template is targeting 151, chain is at 150 — correct, not stale

This is fine. But what if:
1. PUSH arrives with channel_tip = 151 (new block found)
2. Canonical channel_tip is still 150 (BLOCK_DATA hasn't arrived yet)
3. `is_template_stale()` = (150 >= 151)? NO — not stale (correct! canonical hasn't confirmed)
4. PUSH triggers GET_BLOCK (via trigger heights, not staleness)
5. BLOCK_DATA arrives with channel_tip = 151 → canonical updates → `is_template_stale()` fires

**This is correct behavior.** No doom loop possible because:
- PUSH triggers GET_BLOCK via trigger pathway (separate from staleness)
- Staleness only fires after BLOCK_DATA confirms the new height
- One GET_BLOCK per BLOCK_DATA confirmation — no loop

**Answer: No, doom-loop prevention is NOT needed post-refactor.** Confirming safe to remove `AdvanceChannelTarget()`.

### Q3: Template Discard on GET_ROUND — Fast-Path Details

Per answer #2: "Only if GET_ROUND is the source Trigger for GET_BLOCK, same as PUSH."

**Current behavior:**
- GET_ROUND triggers GET_BLOCK when heights change (via `get_work()`)
- GET_ROUND ALSO discards templates when Stake/cross-channel advance detected (lines 2460-2468, 2715-2723)

**Question:** In the new architecture, should the Stake/cross-channel template discard:
- **(A)** Compare trigger heights (GET_ROUND) vs template heights to decide discard?
- **(B)** Compare trigger heights vs CANONICAL to decide discard?
- **(C)** Not discard at all — just trigger GET_BLOCK and let `validate_current_template()` handle discard when BLOCK_DATA arrives?

**My recommendation:** Option C for simplicity. The purpose of the refactor is to make canonical the sole discard authority. GET_ROUND's role is trigger-only.

### Q4: MAX_TEMPLATE_HEIGHT_LAG_BLOCKS Grace Buffer

**RESOLVED: REMOVED.** This constant was a workaround for composite height inflation.
With `snap.unified_height` now canonical-only, the template comes FROM canonical (BLOCK_DATA)
and can never be more than 0-1 blocks behind. The grace buffer was never needed with accurate
canonical-only height tracking.

Note: This constant was referenced in design docs but was never implemented in source code.
The concept it addressed — preventing dedup guard suppression when composite heights raced
ahead of BLOCK_DATA — is moot now that both sides are canonical.

### Q5: Implementation Ordering — What First?

The safest implementation order:

1. **Phase A (Foundation):** Add `trigger_unified_height` and `trigger_channel_height` to Snapshot. Add `round_unified_height`, `round_channel_height`, `push_unified_height`, `push_channel_height` as raw fields. Change `unified_height` and `channel_height` to canonical-only in `build_snapshot_locked()`. This is the single biggest change and the most impactful.

2. **Phase B (Cleanup):** Remove `AdvanceChannelTarget()`, eliminate `m_channel_target`, fix `OnTemplateReceived()` to write canonical.

3. **Phase C (Consumers):** Update `get_work()` dedup to use trigger heights. Update `validate_current_template()` (benefits automatically from Phase A). Remove `effective_channel_height` overrides.

4. **Phase D (Tests):** Update all test assertions. Add new canonical isolation tests.

5. **Phase E (Optional):** Remove ClientChannelManager if desired.

**Question:** Shall I implement Phases A through D in this session, or would you prefer to review the plan first and implement in a future session?

---

## Summary of Findings

### What I Learned That's New (Teaching Points)

1. **All internal heights are TIP-semantic.** GET_ROUND sends TARGETs on the wire, but NexusMiner normalizes them to TIPs with `−1`. This is handled correctly and consistently.

2. **`AdvanceChannelTarget()` is a band-aid for the composite problem.** The doom-loop it prevents can only occur because PUSH inflates the composite `channel_height` beyond canonical. Once composites are eliminated, the doom loop vanishes.

3. **ClientChannelManager's ValidateTemplate() is dead code in production.** Solo uses its own `validate_current_template()` backed by HeightTracker. CCM's template storage and validation are only used in standalone tests.

4. **The dual `m_channel_target` / `canonical_channel_target` system is a historical artifact.** It exists because the canonical separation was added incrementally. Both fields should converge to a single `canonical_channel_target`.

5. **Bug #4 (false template rejection) is the most impactful production bug.** During burst recovery (rapid PUSH→BLOCK_DATA cycles), valid templates from BLOCK_DATA are rejected because PUSH has advanced the composite `channel_height` beyond what the node reported. This can cause mining template starvation.

6. **The MAX_TEMPLATE_HEIGHT_LAG_BLOCKS buffer (~line 1150 in get_work) is a workaround for composite inflation.** It exists specifically to handle the case where the composite height is 2+ blocks ahead of the template's block.nHeight. Post-refactor, this workaround may be unnecessary.
