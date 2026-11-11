# GET_ROUND Protocol: 12-Byte + Legacy Compatibility

## Overview

NexusMiner prefers the **12-byte protocol format** for GET_ROUND/NEW_ROUND/OLD_ROUND responses (LLL-TAO PR #151), but the legacy lane also accepts the 16-byte multi-channel format for compatibility.

## Protocol Specification

### Response Format (12 bytes preferred)

```
┌─────────────────────────────────────────────────────┐
│ GET_ROUND / NEW_ROUND / OLD_ROUND Response          │
├─────────────┬───────────────┬───────────────────────┤
│ Field       │ Bytes         │ Value (big-endian)    │
├─────────────┼───────────────┼───────────────────────┤
│ Unified     │ [0-3]         │ Current blockchain    │
│ Height      │               │ height (reference)    │
├─────────────┼───────────────┼───────────────────────┤
│ Channel     │ [4-7]         │ Miner's channel       │
│ Height      │               │ height (for staleness)│
├─────────────┼───────────────┼───────────────────────┤
│ Difficulty  │ [8-11]        │ Mining difficulty     │
│             │               │ target                │
└─────────────┴───────────────┴───────────────────────┘

Total: 12 bytes (preferred)
Legacy: 16 bytes (unified + prime + hash + stake) in legacy lane; miner derives channel height from the active channel.
```

> **Height semantics**: All heights in GET_ROUND/NEW_ROUND responses use **TIP** semantics
> (current chain state).  No normalization is needed — the node sends `tStateBest.nHeight`
> and `stateChannel.nChannelHeight` directly.  This differs from GET_HEIGHT which sends
> `nBestHeight + 1` (TARGET).

### Example (Prime Miner)

**Request:**
```
Miner → Node:  GET_ROUND (opcode 133)
               SET_CHANNEL(1) already sent (Prime)
```

**Response:**
```
Node → Miner:  NEW_ROUND (opcode 204)
               [0x00, 0x63, 0xB1, 0xAC]  ← Unified:  6533548
               [0x00, 0x23, 0x1D, 0x16]  ← Channel:  2301206 (Prime)
               [0x1D, 0x00, 0xFF, 0xFF]  ← Difficulty: 0x1D00FFFF
```

**Miner Action:**
```
1. Parse: unified=6533548, channel=2301206, difficulty=0x1D00FFFF
   (all values are TIP — no normalization needed)
2. Finalize template: nChannelHeight = 2301207 (channel_tip + 1 = channel_target)
3. Check staleness: is_template_stale() = (2301206 >= 2301207)? NO → not stale, continue mining
```

## Benefits of 12-Byte Format

| Feature | 4-byte | 16-byte | **12-byte** |
|---------|--------|---------|-------------|
| **Size** | 4 bytes | 16 bytes | **12 bytes** ✅ |
| **Channel Info** | ❌ None | All 3 channels | **Miner's channel** ✅ |
| **Difficulty** | ❌ Missing | ❌ Missing | **Included** ✅ |
| **Staleness** | Age-based | Multi-channel | **Channel-specific** ✅ |
| **Clarity** | Ambiguous | Redundant | **Clear** ✅ |

## Migration

**Legacy compatibility:** Legacy lane miners accept both 12-byte and 16-byte responses. Stateless lane remains strict.

## Validation Rules

**Legacy lane:** Accepts 12 or 16 bytes.
**Stateless lane:** Strict 12-byte responses only.

```cpp
// ✅ VALID (legacy lane)
packet.m_length == 12 && packet.m_data != nullptr
packet.m_length == 16 && packet.m_data != nullptr

// ❌ INVALID (legacy lane examples)
packet.m_length == 4   // Legacy
packet.m_length == 8   // Invalid
packet.m_length == 20  // Invalid
packet.m_data == nullptr  // Null data
```

## Expected Logs

### Successful Connection

```
[Solo] Connecting to 127.0.0.1:8323...
[Solo] ✓ Connected
[Solo GET_ROUND] NEW_ROUND response received
[Solo GET_ROUND] 🔔 NEW_ROUND:
[Solo GET_ROUND]   Unified height:  6533548 (reference)
[Solo GET_ROUND]   Prime height:    2301206
[Solo GET_ROUND]   Difficulty:      0x1D00FFFF
[Solo GET_ROUND] ✓ Template finalized:
[Solo GET_ROUND]   Node Prime height:     2301206
[Solo GET_ROUND]   Template Prime height: 2301207
[Solo GET_ROUND] ✓ Template valid - Prime height unchanged
```

### Protocol Error (Invalid Packet Length)

```
[Solo GET_ROUND] NEW_ROUND response received
[Solo GET_ROUND] ❌ PROTOCOL ERROR: Invalid packet length
[Solo GET_ROUND]   Expected:  12 bytes (unified + channel + difficulty)
[Solo GET_ROUND]              16 bytes (unified + prime + hash + stake)
[Solo GET_ROUND]   Received:  20 bytes
[Solo GET_ROUND]   Node may be running incompatible version
```

## Template Staleness Detection

### How It Works

1. **Template Creation**: Template builds NEXT block
   - Node reports: Channel height = N
   - Template uses: Channel height = N + 1

> ⚠️ **Superseded:** The two-axis `channel_advanced` / `tip_moved` decision model
> described below reflects the *initial* implementation.  It is **no longer the active
> push-handling logic**.  See **Current Model** immediately below for what the code
> actually does today.

#### Old Model (Two-Axis Staleness — superseded)

2. **Freshness Check** (every 5 seconds via GET_ROUND, or immediately on push):
   - Get current `unified_height` and `channel_height` from node
   - **`channel_advanced`**: `channel_height >= channel_target` → STALE, request new work
   - **`tip_moved`**: `unified_height > template_unified_height` → STALE anchor, request new work
   - Otherwise → Template is FRESH, continue mining

3. **Both Heights Matter**
   - `channel_height` tracks staleness of the block target (`channel_advanced`)
   - `unified_height` tracks staleness of `hashPrevBlock` anchor (`tip_moved`)
   - See [unified-tip-vs-channel-height.md](../mining/unified-tip-vs-channel-height.md)

#### Current Model (Unified-Height-Driven Push)

The push notification handler (`push_notification_handler.cpp`) uses a simplified
unified-height-driven model:

- **Every PUSH unconditionally triggers a template refresh** — `request_work_fn()` is
  always called.  A PUSH means the node found a block on *some* channel, which advances
  the unified tip and therefore changes `hashPrevBlock`.  The mined block must embed the
  correct parent hash, so the template must always be replaced.

- **Channel height is tracked for bookkeeping purposes only**: `is_template_stale()` triggers
  `AdvanceChannelTarget()` for doom-loop bookkeeping, but does **not** gate whether a
  work request is issued.  Similarly, `validate_current_template()` uses `unified_height`
  (block.nHeight vs canonical_unified_height) as the **primary staleness gate** (PR #606);
  channel_height from the push payload is informational only and is not used in stale
  detection within that function.

- **Same-height tip replacement**: when `channel_height` has NOT advanced but the node
  reports a different `hashBestChain` at the same height (same-height reorg), the current
  template is discarded via `discard_template()` before calling `request_work_fn()`.

- **Cross-channel PUSH** (wrong channel arrives): if `notification_unified_height >
  snap.unified_height`, `request_work_fn()` (or `cross_channel_request_fn()`) is called
  even for cross-channel pushes — the unified tip moved and `hashPrevBlock` is stale
  regardless of which channel mined.

The `GetBlockReason` enum (`get_block_reason.hpp`) labels each request semantically:

| Reason | When emitted |
|---|---|
| `PUSH_TIP_MOVED` | Unified tip moved on own channel (normal advance) |
| `PUSH_SAME_HEIGHT_TIP` | Hash mismatch at equal channel height (same-height reorg) |

> **Note**: `PUSH_STALE`, `PUSH_NO_TEMPLATE`, and `PUSH_CROSS_CHANNEL` have been
> removed — the NODE now auto-sends BLOCK_DATA after PUSH, so no GET_BLOCK is needed.

See also: [get-block-dedup-unified-height.md](../mining-protocols/get-block-dedup-unified-height.md)
for the dedup guard that prevents redundant back-to-back requests.

### Example Scenarios

**Scenario 1: Tip Moved — Other Channel Mines (PUSH_TIP_MOVED refresh)**
```
T0: Template for Prime height 2301207 (node: unified=6533548, prime=2301206)
T1: Hash block mined → unified advances, Prime unchanged
    Push received:  unified=6533549, prime=2301206
    Action: NODE auto-sends fresh BLOCK_DATA template [reason: PUSH_TIP_MOVED]
    (hashPrevBlock in old template now points to a stale ancestor)
```

**Scenario 2: Own Channel Mines — Template Auto-Refreshed**
```
T0: Template for Prime height 2301207 (node: unified=6533548, prime=2301206)
T1: Prime block mined → channel advances
    Push received:  unified=6533549, prime=2301207
    Action: NODE auto-sends fresh BLOCK_DATA template
    (no GET_BLOCK request needed)
```

**Scenario 3: Cross-Channel PUSH — Unified Tip Advanced**
```
T0: Prime miner has template for Prime height 2301207 (unified=6533548)
T1: Stake block mined → Hash/Stake PUSH received on Prime miner
    Push received:  channel=Stake, unified=6533549
    Action: NODE auto-sends fresh BLOCK_DATA template
    (hashPrevBlock changed — Prime template refreshed automatically)
```
