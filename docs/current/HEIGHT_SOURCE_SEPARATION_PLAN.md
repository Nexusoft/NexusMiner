# Height Source Separation Plan

## Eliminate Composite Height Mixing — Single-Source Height Architecture

**Status:** Implementation Plan  
**Problem:** Too many Height conflicts from various sources. The `max(canonical, push, round)` composition in `HeightTracker::Snapshot` conflates different height sources, making troubleshooting impossible and causing false-stale/false-advance decisions.

**Goal:** HeightTracker and ClientChannelManager use **only 1 height** in their sources. GET ROUND heights become a real GET BLOCK trigger (like PUSH). Block DATA canonical heights are kept **alone** as their own verifier. No more composite max() mixing.

---

## Table of Contents

1. [Current Architecture — What's Broken](#1-current-architecture--whats-broken)
2. [New Architecture — The Target State](#2-new-architecture--the-target-state)
3. [Implementation Items (Ordered)](#3-implementation-items-ordered)
4. [Detailed Design Per Item](#4-detailed-design-per-item)
5. [Test Strategy](#5-test-strategy)
6. [Migration Safety & Rollback](#6-migration-safety--rollback)
7. [Files Changed](#7-files-changed)

---

## 1. Current Architecture — What's Broken

### 1.1 The Core Problem: Composite Height Mixing

The `HeightTracker::Snapshot` computes its two most critical fields as **max() composites** of three different sources:

```cpp
// height_tracker.cpp — build_snapshot_locked()
s.unified_height = std::max({m_canonical.canonical_unified_height,
                             m_diagnostic.push_unified_height,
                             m_diagnostic.round_unified_height});

s.channel_height = std::max({m_canonical.canonical_channel_height,
                             m_diagnostic.push_channel_height,
                             m_diagnostic.round_channel_height});
```

These composite values are then used in **every staleness decision**:

| Decision Point | Uses `snap.unified_height` | Uses `snap.channel_height` | Problem |
|---|---|---|---|
| `is_template_stale()` | ✗ | ✓ composite | If PUSH advances channel_height beyond what the node actually reports, template is falsely declared stale |
| `is_tip_moved()` | ✓ composite | ✗ | If GET_ROUND advances unified beyond BLOCK_DATA, tip appears moved when it hasn't |
| `validate_current_template()` line 4640 | ✗ | ✓ composite | `tmpl->nChannelHeight <= snap.channel_height` → FALSE STALE if push is ahead |
| Dedup guard height key | ✓ composite | ✗ | Burst guard may suppress/allow based on the wrong source advancing |
| `effective_channel_height` selection (lines 1898, 3711) | ✗ | ✓ composite | `max(metadata, tracker)` overrides node-authoritative BLOCK_DATA with push/round data |
| `expected_template_target()` | ✗ | ✓ composite | Returns `channel_height + 1` where channel_height is the composite |
| `blocks_behind()` | ✗ | ✓ composite | Computes lag using composite, not node-authoritative height |

### 1.2 The Three Height Sources Today

| Source | What It Carries | When It Arrives | Latency | Authority |
|---|---|---|---|---|
| **PUSH** (BLOCK_AVAILABLE) | unified_height, channel_height, nBits, [hashPrevBlock] | Immediately on new block | ~0ms | Signal only — "something changed" |
| **GET_ROUND** (NEW_ROUND/OLD_ROUND) | unified_height, prime_height, hash_height, stake_height | Polled every ~10s or on demand | 0–10s | Normative snapshot of node state |
| **BLOCK_DATA** (GET_BLOCK response) | Full block + 12-byte metadata prefix (unified, channel, nBits) | After GET_BLOCK request | 50–500ms | **Authoritative** — this is what we mine on |

### 1.3 The Dual-System Divergence

Two systems track heights **independently** and **incompatibly**:

| System | Updated By | Height Source |
|---|---|---|
| **HeightTracker** → `Snapshot` | PUSH, GET_ROUND, BLOCK_DATA | max() composite of all three |
| **ClientChannelManager** | GET_ROUND only | GET_ROUND only |

`validate_current_template()` uses HeightTracker's composite snapshot, but `sync_template_state()` uses ClientChannelManager's GET_ROUND-only fork detection. This means fork detection and template validation operate on **different height realities**.

### 1.4 Specific Confliction Sites

**Site 1 — Lines 1898–1902 / 3711–3714 in solo.cpp:**
```cpp
auto ht_snap = m_height_tracker.GetSnapshot();
if (ht_snap.channel_height > effectiveChannelHeight) {
    effectiveChannelHeight = ht_snap.channel_height;  // Overrides node-authoritative metadata!
}
```

**Site 2 — Line 4640 in solo.cpp (validate_current_template):**
```cpp
if (tmpl->nChannelHeight <= snap.channel_height) {
    // FALSE STALE if push/round advanced channel_height beyond BLOCK_DATA
    discard_template("Channel height stale");
}
```

**Site 3 — OnTemplateReceived() captures `template_unified_height` from composite:**
```cpp
auto snap = build_snapshot_locked();
m_template_unified_height = snap.unified_height;  // Stores a composite, not canonical!
```

**Site 4 — `update_height_state()` only updates HeightTracker, not ClientChannelManager:**
PUSH → HeightTracker only. GET_ROUND → both systems. BLOCK_DATA → HeightTracker only.

---

## 2. New Architecture — The Target State

### 2.1 Design Principles

1. **GET ROUND = Trigger Source (like PUSH).** GET_ROUND and PUSH heights are used **only** to trigger GET_WORK/GET_BLOCK requests. They are **never** mixed into the canonical mining state.

2. **BLOCK_DATA Canonical Heights = Sole Verifier.** Only `OnBlockDataReceived()` data flows into the fields that `is_template_stale()`, `validate_current_template()`, and `expected_template_target()` use.

3. **No More max() Composition for Mining Decisions.** The `Snapshot` struct's `unified_height` and `channel_height` will be sourced from **canonical only** (BLOCK_DATA). Push/round heights remain available as separate diagnostic/trigger fields.

4. **Trigger Heights vs. Mining Heights are Separate Namespaces.** Trigger heights (push, round) tell us "go fetch a new block." Mining heights (canonical) tell us "is this template valid?"

5. **ClientChannelManager Alignment.** ClientChannelManager feeds from GET_ROUND for fork detection only. It never participates in template validation decisions — those are HeightTracker's canonical state only.

### 2.2 New Snapshot Field Layout

```
┌───────────────────────────────────────────────────────┐
│                   MINING STATE                        │
│              (from BLOCK_DATA only)                   │
│                                                       │
│  unified_height     ← canonical_unified_height        │
│  channel_height     ← canonical_channel_height        │
│  channel_target     ← canonical_channel_target        │
│  hash_prev_block    ← canonical_hash_prev_block       │
│  difficulty_nbits   ← canonical_difficulty_nbits      │
│                                                       │
│  Used by: is_template_stale()                         │
│           validate_current_template()                 │
│           expected_template_target()                  │
│           blocks_behind()                             │
│           drift_delta()                               │
│           is_tip_moved()                              │
├───────────────────────────────────────────────────────┤
│                TRIGGER STATE                          │
│         (from PUSH & GET_ROUND)                       │
│                                                       │
│  trigger_unified_height  ← max(push, round)           │
│  trigger_channel_height  ← max(push, round)           │
│  push_unified_height     ← raw push value             │
│  push_channel_height     ← raw push value             │
│  round_unified_height    ← raw GET_ROUND value        │
│  round_channel_height    ← raw GET_ROUND value        │
│                                                       │
│  Used by: GET_WORK/GET_BLOCK trigger decisions        │
│           Dedup guard height key                      │
│           Cross-channel stake detection               │
│           "Should I request a new block?" logic       │
├───────────────────────────────────────────────────────┤
│              DIAGNOSTIC STATE                         │
│         (for logging/Colin agent only)                │
│                                                       │
│  keepalive_unified_height  ← raw keepalive value      │
│  keepalive_prime_height    ← raw keepalive value      │
│  prime_height, hash_height, stake_height (per-source) │
│  fork_score, peak_fork_score                          │
│  All timing fields                                    │
│                                                       │
│  Used by: Logging, Colin agent, diagnostics           │
│           NEVER for mining decisions                  │
│           NEVER for trigger decisions                 │
└───────────────────────────────────────────────────────┘
```

### 2.3 New Decision Flow

```
    PUSH arrives                  GET_ROUND arrives
         │                              │
         ▼                              ▼
  Update trigger_*              Update trigger_*
  heights in HeightTracker      heights in HeightTracker
         │                              │
         ▼                              ▼
  Compare trigger height        Compare trigger height
  vs canonical height:          vs canonical height:
  "Has something changed?"      "Has something changed?"
         │                              │
    ┌────┴────┐                    ┌────┴────┐
    │ YES     │ NO                 │ YES     │ NO
    ▼         ▼                    ▼         ▼
  Issue     (do nothing)        Issue     (do nothing)
  GET_BLOCK                     GET_BLOCK
    │                              │
    ▼                              ▼
  BLOCK_DATA arrives (response)
    │
    ▼
  Update CANONICAL state (OnBlockDataReceived)
    │
    ▼
  validate_current_template() using CANONICAL ONLY
    │
    ▼
  Feed template to workers (or discard if stale)
```

---

## 3. Implementation Items (Ordered)

### Item 1: Separate Snapshot into Mining/Trigger/Diagnostic Zones
**Priority:** Critical (foundation for everything else)  
**Risk:** High (touches every snapshot consumer)  
**Files:** `height_tracker.hpp`, `height_tracker.cpp`

### Item 2: Source `unified_height` and `channel_height` from Canonical Only
**Priority:** Critical (eliminates the core confliction)  
**Risk:** High (changes staleness semantics)  
**Files:** `height_tracker.cpp` (`build_snapshot_locked`)

### Item 3: Introduce `trigger_unified_height` and `trigger_channel_height`
**Priority:** Critical (GET_ROUND + PUSH trigger fields)  
**Risk:** Medium  
**Files:** `height_tracker.hpp`, `height_tracker.cpp`

### Item 4: Update GET_WORK/GET_BLOCK Trigger Logic to Use Trigger Heights
**Priority:** Critical (GET_WORK must still fire on push/round)  
**Risk:** High (must not break mining responsiveness)  
**Files:** `solo.cpp` (`get_work`, dedup guard interactions)

### Item 5: Update `validate_current_template()` to Use Canonical Only
**Priority:** Critical (eliminates false-stale from push/round)  
**Files:** `solo.cpp`

### Item 6: Remove `effective_channel_height` max() Overrides
**Priority:** High (eliminates confliction sites 1898/3711)  
**Files:** `solo.cpp` (`on_block_data`, `on_stateless_get_block`)

### Item 7: Update `is_template_stale()` and Related Snapshot Methods
**Priority:** High (these use `channel_height` which is now canonical-only)  
**Files:** `height_tracker.hpp`

### Item 8: Add Trigger-vs-Canonical Drift Logging
**Priority:** Medium (replaces the lost diagnostic information)  
**Files:** `height_tracker.hpp`, `solo.cpp`

### Item 9: Update Dedup Guard to Use Trigger Heights for Key
**Priority:** High (dedup must fire on trigger advances, not canonical)  
**Files:** `solo.cpp`, `get_block_dedup_guard.hpp`

### Item 10: Update `OnTemplateReceived()` to Capture Canonical, Not Composite
**Priority:** Medium (fixes `template_unified_height` capture)  
**Files:** `height_tracker.cpp`

### Item 11: Align ClientChannelManager — Document Its GET_ROUND-Only Scope
**Priority:** Medium (no code change if it already only uses GET_ROUND)  
**Files:** `client_channel_manager.h`, `solo.cpp`

### Item 12: Update All Tests
**Priority:** Critical (tests must validate new separation)  
**Files:** All `*_test.cpp` files

### Item 13: Update Colin Agent Diagnostics
**Priority:** Low (Colin reads snapshot — needs new field names)  
**Files:** `colin_agent.cpp`

---

## 4. Detailed Design Per Item

### Item 1: Separate Snapshot into Mining/Trigger/Diagnostic Zones

**Current:** Single flat `Snapshot` struct with ~40 fields, no clear ownership.

**New:** Add semantic zone comments and introduce `trigger_*` fields. Keep the struct flat for ABI compatibility but organize fields into clearly labeled zones:

```cpp
struct Snapshot {
    uint64_t session_epoch{0};

    // ══════════════════════════════════════════════════════════════════════
    // MINING STATE — sourced from BLOCK_DATA canonical chain ONLY.
    // Used by: is_template_stale(), validate_current_template(),
    //          expected_template_target(), blocks_behind(), drift_delta()
    // RULE: These fields are NEVER updated by PUSH or GET_ROUND.
    // ══════════════════════════════════════════════════════════════════════
    uint32_t unified_height{0};            // canonical_unified_height (BLOCK_DATA only)
    uint32_t channel_height{0};            // canonical_channel_height (BLOCK_DATA only)
    uint32_t channel_target{0};            // canonical_channel_target (BLOCK_DATA only)
    uint32_t difficulty_nbits{0};          // canonical nBits (BLOCK_DATA only)
    uint1024_t hash_prev_block{};          // canonical hashPrevBlock (BLOCK_DATA only)
    uint32_t template_unified_height{0};   // unified_height at template receipt (canonical)

    // Typed aliases for submission-path guards
    UnifiedHeight unified_block_height{};
    ChannelHeight channel_tip_height{};
    ChannelHeight template_channel_target{};
    UnifiedHeight template_block_height{};
    uint32_t channel{0};                   // Mining channel (1=Prime, 2=Hash)

    // ══════════════════════════════════════════════════════════════════════
    // TRIGGER STATE — sourced from PUSH and GET_ROUND.
    // Used by: GET_WORK/GET_BLOCK trigger decisions, dedup guard height key,
    //          "should I request a new block?" logic.
    // RULE: These fields trigger GET_BLOCK requests. They do NOT participate
    //        in template validation or staleness checks.
    // ══════════════════════════════════════════════════════════════════════
    uint32_t trigger_unified_height{0};    // max(push_unified, round_unified)
    uint32_t trigger_channel_height{0};    // max(push_channel, round_channel)

    // Raw per-source trigger heights (for diagnostics & targeted logic)
    uint32_t push_unified_height{0};       // Raw push unified
    uint32_t push_channel_height{0};       // Raw push channel
    uint32_t round_unified_height{0};      // Raw GET_ROUND unified
    uint32_t round_channel_height{0};      // Raw GET_ROUND channel
    uint1024_t push_hash_prev_block{};     // hashPrevBlock hint from extended push

    // ══════════════════════════════════════════════════════════════════════
    // DIAGNOSTIC STATE — for logging, Colin agent, health checks ONLY.
    // RULE: NEVER used for mining or trigger decisions.
    // ══════════════════════════════════════════════════════════════════════
    // Cross-channel heights (per-source breakdown)
    uint32_t prime_height{0};              // max(keepalive, round, push) — diagnostic
    uint32_t hash_height{0};               // max(keepalive, round, push) — diagnostic
    uint32_t stake_height{0};              // max(keepalive, round, push) — diagnostic
    // ... (push-specific, keepalive-specific, fork_score, timing fields)
    // ... (same as today, but clearly labeled DIAGNOSTIC)

    // ══════════════════════════════════════════════════════════════════════
    // CANONICAL REFERENCE — raw copies for drift computation vs trigger.
    // These let consumers compare "what BLOCK_DATA says" vs "what triggers say"
    // without needing GetCanonicalSnapshot() separately.
    // ══════════════════════════════════════════════════════════════════════
    uint32_t canonical_unified_height{0};
    uint32_t canonical_channel_height{0};
    uint1024_t canonical_hash_prev_block{};
    std::chrono::steady_clock::time_point canonical_received_at{};

    UpdateSource last_update_source{UpdateSource::NONE};

    // ── Staleness methods (now purely canonical-based) ──────────────────
    bool is_template_stale() const {
        // Uses channel_height (now canonical only) and channel_target (canonical only)
        return (channel_height > 0 && channel_target > 0 &&
                channel_height >= channel_target);
    }

    bool is_tip_moved() const {
        // Uses unified_height (now canonical only) vs template capture
        return (template_unified_height > 0 && unified_height > template_unified_height);
    }

    // ── NEW: Trigger-based methods ──────────────────────────────────────
    bool has_trigger_advanced_beyond_canonical() const {
        return trigger_unified_height > canonical_unified_height ||
               trigger_channel_height > canonical_channel_height;
    }

    bool should_trigger_get_block() const {
        // Trigger fires when push/round sees a height beyond our template
        return (trigger_unified_height > template_unified_height) ||
               (trigger_channel_height > 0 && channel_target > 0 &&
                trigger_channel_height >= channel_target);
    }

    int32_t trigger_canonical_drift() const {
        return static_cast<int32_t>(trigger_unified_height) -
               static_cast<int32_t>(canonical_unified_height);
    }
};
```

### Item 2: Source `unified_height` and `channel_height` from Canonical Only

**Change in `build_snapshot_locked()`:**

```cpp
// BEFORE (current — composite):
s.unified_height = std::max({m_canonical.canonical_unified_height,
                             m_diagnostic.push_unified_height,
                             m_diagnostic.round_unified_height});
s.channel_height = std::max({m_canonical.canonical_channel_height,
                             m_diagnostic.push_channel_height,
                             m_diagnostic.round_channel_height});

// AFTER (new — canonical only):
s.unified_height = m_canonical.canonical_unified_height;
s.channel_height = m_canonical.canonical_channel_height;
```

This is the **single most important change**. Everything downstream that reads `snap.unified_height` and `snap.channel_height` now gets canonical-only values.

### Item 3: Introduce `trigger_unified_height` and `trigger_channel_height`

**Add to `build_snapshot_locked()`:**

```cpp
// Trigger heights: max of push and round (NOT canonical).
// These are used ONLY for GET_WORK/GET_BLOCK trigger decisions.
s.trigger_unified_height = std::max(m_diagnostic.push_unified_height,
                                     m_diagnostic.round_unified_height);
s.trigger_channel_height = std::max(m_diagnostic.push_channel_height,
                                     m_diagnostic.round_channel_height);

// Raw per-source values for targeted logic
s.push_unified_height = m_diagnostic.push_unified_height;
s.push_channel_height = m_diagnostic.push_channel_height;
s.round_unified_height = m_diagnostic.round_unified_height;
s.round_channel_height = m_diagnostic.round_channel_height;
```

### Item 4: Update GET_WORK/GET_BLOCK Trigger Logic to Use Trigger Heights

**In `get_work()` (solo.cpp ~line 1130–1185):**

The dedup guard currently uses `snap.unified_height` (composite) as its height key. Change it to use `snap.trigger_unified_height` so that push/round advances still break the dedup key:

```cpp
// BEFORE:
auto verdict = m_dedup_guard.check(reason, snap.unified_height, have_valid_template);

// AFTER:
auto verdict = m_dedup_guard.check(reason, snap.trigger_unified_height, have_valid_template);
```

Also in `record_transmission`:
```cpp
// BEFORE:
m_dedup_guard.record_transmission(snap.unified_height);

// AFTER:
m_dedup_guard.record_transmission(snap.trigger_unified_height);
```

**Rationale:** The dedup guard's purpose is to prevent duplicate GET_BLOCK requests. It should fire when push/round says "something changed" (trigger), not when we already have block data (canonical). If we used canonical, the dedup would never break until BLOCK_DATA arrives — too late.

### Item 5: Update `validate_current_template()` to Use Canonical Only

**In solo.cpp line 4630–4650:**

```cpp
// BEFORE (uses composite snap.channel_height):
if (tmpl->nChannelHeight <= snap.channel_height) {
    discard_template("Channel height stale");
}

// AFTER (uses canonical snap.channel_height — which is now canonical-only):
// No code change needed! Since snap.channel_height is now canonical-only
// (from Item 2), this comparison naturally becomes:
//   tmpl->nChannelHeight <= canonical_channel_height
// Which is correct: only BLOCK_DATA can authoritatively say the chain has
// surpassed our template's target.
```

The fix is entirely in Item 2 (the snapshot composition). No changes needed in `validate_current_template()` itself — it already uses `snap.channel_height`, which now means canonical-only.

### Item 6: Remove `effective_channel_height` max() Overrides

**In `on_block_data()` (solo.cpp lines 1898–1902) and `on_stateless_get_block()` (lines 3711–3714):**

```cpp
// BEFORE — overrides BLOCK_DATA metadata with push/round composite:
auto ht_snap = m_height_tracker.GetSnapshot();
if (ht_snap.channel_height > effectiveChannelHeight) {
    effectiveChannelHeight = ht_snap.channel_height;
}

// AFTER — trust BLOCK_DATA metadata (it IS the canonical source):
// REMOVE the override entirely. The BLOCK_DATA metadata IS authoritative.
// If it's "behind" push/round, that's expected — push/round are TRIGGERS,
// not authorities. The node knows its own block data.
//
// Add a DIAGNOSTIC log if trigger is ahead (for observability, not action):
auto ht_snap = m_height_tracker.GetSnapshot();
if (ht_snap.trigger_channel_height > effectiveChannelHeight) {
    m_logger->debug("[Solo BLOCK_DATA] Trigger channel_height {} ahead of "
                    "BLOCK_DATA metadata {} — expected during burst recovery",
                    ht_snap.trigger_channel_height, effectiveChannelHeight);
}
// effectiveChannelHeight stays as-is from BLOCK_DATA metadata.
```

### Item 7: Update `is_template_stale()` and Related Snapshot Methods

After Item 2, these methods automatically use canonical-only values. Verify each one:

| Method | Field Used | After Change | Correct? |
|---|---|---|---|
| `is_template_stale()` | `channel_height`, `channel_target` | Both canonical | ✓ Only BLOCK_DATA can say chain surpassed our target |
| `is_tip_moved()` | `unified_height`, `template_unified_height` | Both canonical | ✓ Only BLOCK_DATA confirms tip actually moved |
| `expected_template_target()` | `channel_height` | Canonical | ✓ Next target based on confirmed chain state |
| `blocks_behind()` | `channel_height`, `channel_target` | Both canonical | ✓ Lag measured against confirmed state |
| `drift_delta()` | `channel_height`, `channel_target` | Both canonical | ✓ Drift measured against confirmed state |

**One exception — `is_tip_moved()`:** After this change, `is_tip_moved()` will only fire when canonical advances. But we also need it to fire on trigger advances (push/round) for cross-channel detection. **Solution:** Add a new trigger-based equivalent:

```cpp
// NEW method for trigger-based tip detection:
bool has_trigger_tip_moved() const {
    return (template_unified_height > 0 &&
            trigger_unified_height > template_unified_height);
}
```

Then in solo.cpp, Stake/cross-channel detection should use `has_trigger_tip_moved()` instead of `is_tip_moved()`.

### Item 8: Add Trigger-vs-Canonical Drift Logging

Replace the lost observability from removing the composite:

```cpp
// In solo.cpp, after processing any height update:
auto snap = m_height_tracker.GetSnapshot();
auto drift = snap.trigger_canonical_drift();
if (drift > 0) {
    m_logger->debug("[Solo Height] Trigger {} blocks ahead of canonical {} "
                    "(trigger: push={} round={}, canonical: block_data={})",
                    drift,
                    snap.canonical_unified_height,
                    snap.push_unified_height,
                    snap.round_unified_height,
                    snap.canonical_unified_height);
}
```

### Item 9: Update Dedup Guard to Use Trigger Heights for Key

The `GetBlockDedupGuard::check()` height comparison and `record_transmission()` both use a `current_unified` parameter. This comes from the snapshot's `unified_height`.

**After Item 4**, the callers pass `trigger_unified_height` instead. No changes needed inside `GetBlockDedupGuard` itself — it's the caller's responsibility to pass the right height.

### Item 10: Update `OnTemplateReceived()` to Capture Canonical, Not Composite

```cpp
// BEFORE:
void HeightTracker::OnTemplateReceived(...) {
    auto snap = build_snapshot_locked();
    m_template_unified_height = snap.unified_height;  // Was composite!
}

// AFTER:
void HeightTracker::OnTemplateReceived(...) {
    // Capture the canonical unified height at template receipt time.
    // This is used by is_tip_moved() to detect when canonical advances.
    m_template_unified_height = m_canonical.canonical_unified_height;
}
```

### Item 11: Align ClientChannelManager — Document Its GET_ROUND-Only Scope

ClientChannelManager is already correct — it only receives GET_ROUND heights via `apply_channel_manager_update()`. No code changes needed, but add documentation:

```cpp
// In client_channel_manager.h:
/// @brief Channel-specific height tracking and fork detection.
///
/// HEIGHT SOURCE: GET_ROUND ONLY.
/// This manager is updated exclusively by GET_ROUND/NEW_ROUND responses.
/// It does NOT receive PUSH or BLOCK_DATA heights.
///
/// PURPOSE: Fork detection (2+ block regression = fork, 1-block = phantom stake).
/// It does NOT participate in template validation — that's HeightTracker's
/// canonical state via validate_current_template().
///
/// The GET_ROUND-only scope is intentional: fork detection needs raw node
/// snapshots, not BLOCK_DATA-delayed heights.
```

### Item 12: Update All Tests

Tests that verify snapshot composition need updating:

| Test File | What Changes |
|---|---|
| `height_tracker_test.cpp` | Tests that assert `snap.unified_height == max(canonical, push, round)` → change to assert `snap.unified_height == canonical`. Add new tests for `trigger_*` fields. |
| `mining_template_validation_test.cpp` | Tests that rely on push advancing `snap.channel_height` → update to use trigger fields for trigger-based logic. |
| `push_notification_lane_test.cpp` | Tests that check `snap.unified_height` after push → verify it remains canonical, check `trigger_unified_height` instead. |
| `get_block_dedup_recovery_test.cpp` | Dedup tests that use `snap.unified_height` → use `snap.trigger_unified_height`. |
| `fork_vs_tip_change_test.cpp` | Fork detection tests → verify they use canonical for validation, trigger for detection. |
| `degraded_recovery_test.cpp` | Recovery tests → verify trigger-based GET_BLOCK firing. |
| `phase2b_update_height_test.cpp` | Height update tests → update composite assertions. |

**New test cases to add:**
1. **Canonical isolation:** Push advances `trigger_unified_height` but `unified_height` stays at canonical — verify `is_template_stale()` does NOT fire.
2. **Trigger fires GET_BLOCK:** Push advances trigger → GET_WORK dedup allows → GET_BLOCK issued.
3. **False-stale prevention:** Push channel_height > BLOCK_DATA channel_height → template NOT falsely discarded.
4. **Drift logging:** Trigger ahead of canonical → diagnostic drift logged.

### Item 13: Update Colin Agent Diagnostics

Colin agent reads `GetSnapshot()` for observability. Ensure it uses the correct fields:

```cpp
// In colin_agent.cpp:
auto snap = m_height_tracker->GetSnapshot();
// For display purposes, show BOTH canonical and trigger:
report("canonical_unified", snap.unified_height);       // Now canonical-only
report("trigger_unified",  snap.trigger_unified_height); // New field
report("drift",            snap.trigger_canonical_drift());
```

---

## 5. Test Strategy

### 5.1 Regression Testing

Run ALL existing tests after Items 1–3 to identify breakage:
```bash
cd build/debug && make -j$(nproc)
./src/protocol/height_tracker_test
./src/protocol/mining_template_validation_test
./src/protocol/push_notification_lane_test
./src/protocol/get_block_dedup_recovery_test
./src/protocol/fork_vs_tip_change_test
./src/protocol/degraded_recovery_test
./src/protocol/solo_auth_resync_test
./src/protocol/phase2b_update_height_test
./src/mining/client_channel_manager_test
```

### 5.2 New Test Categories

1. **Canonical Isolation Tests** — Verify push/round CANNOT influence `unified_height` or `channel_height` in the snapshot.
2. **Trigger Responsiveness Tests** — Verify push/round DO influence `trigger_*` fields and still trigger GET_BLOCK.
3. **False-Stale Prevention Tests** — Verify templates are NOT discarded when only push/round have advanced.
4. **Drift Observability Tests** — Verify `trigger_canonical_drift()` reports correctly.

---

## 6. Migration Safety & Rollback

### 6.1 Feature Flag Approach (Recommended)

Add a compile-time or runtime flag to toggle between composite and canonical-only modes:

```cpp
// In height_tracker.hpp:
#ifndef NEXUS_HEIGHT_CANONICAL_ONLY
#define NEXUS_HEIGHT_CANONICAL_ONLY 1  // Set to 0 to revert to composite
#endif

// In build_snapshot_locked():
#if NEXUS_HEIGHT_CANONICAL_ONLY
    s.unified_height = m_canonical.canonical_unified_height;
    s.channel_height = m_canonical.canonical_channel_height;
#else
    s.unified_height = std::max({m_canonical.canonical_unified_height,
                                 m_diagnostic.push_unified_height,
                                 m_diagnostic.round_unified_height});
    s.channel_height = std::max({m_canonical.canonical_channel_height,
                                 m_diagnostic.push_channel_height,
                                 m_diagnostic.round_channel_height});
#endif
```

### 6.2 Transition Logging

During the transition period, log both old-composite and new-canonical values:

```cpp
auto old_unified = std::max({canonical, push, round});
auto new_unified = canonical;
if (old_unified != new_unified) {
    m_logger->info("[HeightTracker TRANSITION] composite={} vs canonical={} "
                   "(push={} round={})",
                   old_unified, new_unified, push, round);
}
```

### 6.3 Rollback Plan

If canonical-only mode causes mining stalls (template never refreshed):
1. Set `NEXUS_HEIGHT_CANONICAL_ONLY 0` → revert to composite
2. Investigate which trigger path failed to fire GET_BLOCK
3. Fix trigger logic, re-enable canonical-only mode

---

## 7. Files Changed

### Primary Changes (Must Change)

| File | Change Type | Items |
|---|---|---|
| `src/protocol/inc/protocol/height_tracker.hpp` | Struct reorganization, new trigger fields, new methods | 1, 3, 7 |
| `src/protocol/src/protocol/height_tracker.cpp` | `build_snapshot_locked()` composition change, `OnTemplateReceived()` fix | 2, 3, 10 |
| `src/protocol/src/protocol/solo.cpp` | GET_WORK trigger logic, validate_current_template, remove effective_channel_height overrides | 4, 5, 6, 8, 9 |
| `src/protocol/height_tracker_test.cpp` | Update all composite assertions, add canonical isolation tests | 12 |

### Secondary Changes (Should Change)

| File | Change Type | Items |
|---|---|---|
| `src/protocol/mining_template_validation_test.cpp` | Update snapshot-dependent tests | 12 |
| `src/protocol/push_notification_lane_test.cpp` | Update push-dependent snapshot tests | 12 |
| `src/protocol/get_block_dedup_recovery_test.cpp` | Update dedup height key tests | 12 |
| `src/protocol/fork_vs_tip_change_test.cpp` | Verify canonical isolation in fork detection | 12 |
| `src/protocol/degraded_recovery_test.cpp` | Verify trigger-based recovery | 12 |
| `src/protocol/phase2b_update_height_test.cpp` | Update height composition tests | 12 |

### Documentation/Diagnostic Changes (Nice to Have)

| File | Change Type | Items |
|---|---|---|
| `src/mining/client_channel_manager.h` | Add documentation comments | 11 |
| Colin agent source | Update field references | 13 |

---

## Clarifying Questions

Before implementation begins, the following should be confirmed:

1. **GET_ROUND as full GET_BLOCK trigger:** The problem statement says "GET ROUND HEIGHTS are now a real GET BLOCK and Discard source of Full Height INFO, similar to PUSH." Does this mean:
   - (a) GET_ROUND response should **always** trigger a GET_BLOCK (like a push does), OR
   - (b) GET_ROUND response should trigger GET_BLOCK only when it detects a height change (current behavior, but using trigger heights instead of composites)?
   
   This plan assumes (b) — GET_ROUND triggers GET_BLOCK on height change, using `trigger_*` fields. If (a), the dedup guard bypass policy for GET_ROUND reasons would need to be more aggressive.

2. **Discard on GET_ROUND:** "GET ROUND HEIGHTS are now a real ... Discard source of Full Height INFO." Does this mean that when GET_ROUND shows the channel has advanced past our template's target, we should **discard the template** using GET_ROUND heights (trigger)? Currently only `validate_current_template()` discards, and it would use canonical. We could add a trigger-based fast-path discard.

3. **ClientChannelManager's role going forward:** Should ClientChannelManager remain as a GET_ROUND-only fork detector, or should it be eliminated in favor of HeightTracker's canonical state handling all fork detection?

4. **`AdvanceChannelTarget()` from PUSH:** Currently, PUSH can advance `canonical_channel_target` via `AdvanceChannelTarget()`. In the new architecture, should this be:
   - (a) Removed entirely (PUSH only updates trigger state), OR
   - (b) Kept as a special case for preventing doom loops (push detects staleness faster than BLOCK_DATA)?

   This plan recommends (b) — keep `AdvanceChannelTarget()` as a safety valve, but clearly document it as the sole exception to "canonical updated only by BLOCK_DATA."

5. **Backward compatibility:** Are there external consumers of the Snapshot struct (beyond Colin agent) that depend on `unified_height` being a max() composite? If so, we may need a transition period with both `unified_height` (canonical) and `composite_unified_height` (legacy) available.
