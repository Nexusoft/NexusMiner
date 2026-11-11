# Port-Lane Separation Implementation Summary

## Overview
This implementation enforces **strict port-lane separation** for the NexusMiner mining protocol, with NO fallback between lanes.

## Lane Rules

### Port 8323 → Legacy Lane
- **Framing**: 8-bit header (1-byte opcode)
- **Wire Format**: `[header:1B][length:4B][data]`
- **Behavior**: Push parity with stateless (MINER_READY + GET_BLOCK), 8-bit framing
- **Opcodes**: 0x00-0xFF (legacy single-byte opcodes)
- **Authentication**: Falcon + ChaCha20 (required)

### Port 9323 (and all others) → Stateless Lane
- **Framing**: 16-bit header (2-byte opcode, big-endian)
- **Wire Format**: `[header:2B][length:4B][data]`
- **Behavior**: Push (MINER_READY + GET_BLOCK), 16-bit mirror framing
- **Opcodes**: 0xD000-0xD0FF (mirror-mapped from legacy)
- **Authentication**: Falcon + ChaCha20 (required)

## Block Format Note

> Both lanes transmit **216-byte Tritium blocks**. The term "Legacy Lane" (Port 8323)
> refers only to the polling protocol behavior — NOT to the block serialization format.
>
> - **216-byte Tritium**: `nTime` NOT in wire template; set by node at `sign_block()` time
> - **220-byte Legacy**: `nTime` IS in wire template (historical; not used in current mining)
>
> See `docs/reference/block-formats.md` for the full field-by-field breakdown.

## Key Implementation Details

### 1. Lane Determination (packet.hpp)
```cpp
enum class ProtocolLane : uint8_t {
    UNKNOWN = 0,
    LEGACY = 1,     // Port 8323
    STATELESS = 2   // All other ports
};

inline ProtocolLane determine_lane_from_port(uint16_t port) {
    return (port == 8323) ? ProtocolLane::LEGACY : ProtocolLane::STATELESS;
}
```

### 2. Lane Initialization (solo.cpp)
- Called on first message in `process_messages()`
- Extracts remote port from connection
- Sets `m_protocol_lane` (never changes after initialization)
- Sets `m_stateless_protocol_active` flag
- Logs lane selection with loud formatting

### 3. Strict Lane Validation (solo.cpp)
In `process_messages()`, before processing any packet:
```cpp
bool expected_uint16 = (m_protocol_lane == ProtocolLane::STATELESS);
bool received_uint16 = packet.m_is_uint16_opcode;

if (expected_uint16 != received_uint16) {
    // LANE MISMATCH: Log error with port, expected vs received
    // Close connection immediately
    // NO FALLBACK ATTEMPTED
}
```

### 4. Protocol Flow by Lane

#### Legacy Lane (Port 8323)
After successful CHANNEL_ACK:
```
1. Send MINER_READY (0xD8) as [header][00000000]
2. Send GET_BLOCK (0x81) for the first template if no valid template exists
3. Accept push-triggered GET_BLOCK/BLOCK_DATA template delivery using 8-bit framing
```

#### Stateless Lane (Port 9323+)
After successful CHANNEL_ACK:
```
1. Send STATELESS_MINER_READY (0xD0D8) as [header16][00000000]
2. Send STATELESS_GET_BLOCK (0xD081) for the first template if no valid template exists
3. Accept push-triggered GET_BLOCK/BLOCK_DATA template delivery using 16-bit framing
```

### 5. Block Submission (solo.cpp)
Uses lane to select opcode:
```cpp
bool use_stateless_opcode = (m_protocol_lane == ProtocolLane::STATELESS);
uint16_t opcode = use_stateless_opcode ? 
    Packet::STATELESS_SUBMIT_BLOCK :  // 0xD001 for stateless
    Packet::SUBMIT_BLOCK;             // 0x01 for legacy
```

## Removed Features

### Auto-Negotiation (DEPRECATED)
- Previously: Miner would send MINER_READY, wait for response with 5s timeout
- Previously: On timeout, would fall back to legacy polling
- **Now**: Lane determined strictly by port, no negotiation or timeout

### Heuristic Detection
- Previously: Packet parsing tried to detect stateless opcodes heuristically
- **Now**: Lane-aware parsing based on connection port (see `extract_packet_from_buffer_with_lane`)
- Note: Old heuristic parser still exists for backward compatibility but is not used in strict mode

## Error Handling

### Lane Mismatch
When a packet arrives with wrong header format for the lane:
```
═══════════════════════════════════════════════════════════
PROTOCOL LANE MISMATCH - DISCONNECTING
═══════════════════════════════════════════════════════════
Remote:          192.168.1.100:9323
Remote Port:     9323
Expected Lane:   Stateless (16-bit header)
Received Header: BLOCK_DATA (0x00) - 8-bit format
═══════════════════════════════════════════════════════════
Server is speaking the WRONG protocol on this port!
- Port 9323 should use Stateless lane
- NO FALLBACK AVAILABLE (strict port-lane separation)
═══════════════════════════════════════════════════════════
```
Then disconnects immediately.

## Testing Recommendations

### Verify Legacy Lane (Port 8323)
1. Connect to node on port 8323
2. Check logs for "LEGACY LANE: Using push protocol"
3. Verify MINER_READY and an initial GET_BLOCK can be sent after CHANNEL_ACK
4. Verify SUBMIT_BLOCK uses 8-bit header (0x01)

### Verify Stateless Lane (Port 9323)
1. Connect to node on port 9323
2. Check logs for "STATELESS LANE: Using push protocol"
3. Verify STATELESS_MINER_READY (0xD0D8) sent after CHANNEL_ACK
4. Verify STATELESS_SUBMIT_BLOCK uses 16-bit header (0xD001)

### Verify No Fallback
1. Connect to port 8323 (legacy)
2. If server sends stateless packet (0xD0xx), should disconnect with LANE MISMATCH
3. Connect to port 9323 (stateless)
4. If server sends legacy packet (< 0x0100), should disconnect with LANE MISMATCH

## Files Modified

1. **src/LLP/packet.hpp**
   - Added `ProtocolLane` enum
   - Added `determine_lane_from_port()` helper
   - Added `extract_packet_from_buffer_with_lane()` for strict parsing

2. **src/protocol/inc/protocol/solo.hpp**
   - Added `m_protocol_lane` member
   - Added `initialize_protocol_lane()` declaration
   - Deprecated auto-negotiation members (kept for compatibility)

3. **src/protocol/src/protocol/solo.cpp**
   - Added `initialize_protocol_lane()` implementation
   - Added strict lane validation in `process_messages()`
   - Updated CHANNEL_ACK handler for lane-based flow
   - Updated `submit_block()` for lane-based opcodes
   - Removed fallback logic from CHANNEL_ACK

## Build Status
✅ Build succeeds with no compilation errors
✅ All existing functionality preserved (with strict enforcement added)

---

## Recovery Model — No Cross-Lane Recovery (Updated)

### Key Principle

**The miner's protocol lane is determined at initial connection time by port and NEVER
changes.**  All recovery operations (template refresh, reconnection, failover) happen on
the same protocol lane.  The only thing that changes during failover is the NODE endpoint.

### Removed: SIM-Link Cross-Lane Bypass

The former `on_lane_failed()` implementation in `DualConnectionManager` armed a bypass
on the **opposite** lane when a lane failed (stateless failure → arm legacy bypass;
legacy failure → arm stateless bypass).  This caused the miner to route recovery through
the wrong protocol and was a holdover from the original SIM-Link design.

**This cross-lane bypass has been removed.**

### New Behavior

`DualConnectionManager::on_lane_failed(dead_lane)` now arms the bypass on the **same**
lane that failed.  This prepares the reconnect path on the correct protocol lane so that
when the lane comes back up it can request a fresh template immediately without triggering
the node's GET_BLOCK rate limiter.

```cpp
// Before (wrong — crossed lanes):
void on_lane_failed(ProtocolLane dead_lane)
{
    if (dead_lane == ProtocolLane::STATELESS)
    {
        m_stateless_alive = false;
        arm_bypass(ProtocolLane::LEGACY);    // ← WRONG
    }
    else
    {
        m_legacy_alive = false;
        arm_bypass(ProtocolLane::STATELESS); // ← WRONG
    }
}

// After (correct — same lane):
void on_lane_failed(ProtocolLane dead_lane)
{
    if (dead_lane == ProtocolLane::STATELESS)
    {
        m_stateless_alive = false;
        arm_bypass(ProtocolLane::STATELESS); // Same lane: ready for reconnect
    }
    else
    {
        m_legacy_alive = false;
        arm_bypass(ProtocolLane::LEGACY);    // Same lane: ready for reconnect
    }
}
```

### Added: Mining Lane Tracking

`DualConnectionManager` now tracks `m_mining_lane` — the protocol lane the miner was
configured to mine on.  This is stamped **exactly once** inside
`NodeSession::connect_primary()`, after the primary connection is established and the
protocol lane is derived from `m_primary_connection->get_protocol_lane()`.  The call
is guarded by `m_dcm->mining_lane() == ProtocolLane::UNKNOWN` so that reconnection
and failover can never re-stamp it.

```cpp
// Called in NodeSession::connect_primary() on first connection only:
if (m_dcm && m_dcm->mining_lane() == ProtocolLane::UNKNOWN) {
    m_dcm->set_mining_lane(lane);
}

// Inspect during recovery (must equal the initial lane):
ProtocolLane lane = m_dcm->mining_lane();
```

### Recovery Sequence

| Step | Operation | Lane | Node |
|------|-----------|------|------|
| 1 | Primary retry | Same as initial | Same node |
| 2 | Failover (if primary retry exhausted) | Same as initial | **New node** |

Failover to a new node always performs a full RE-AUTH sequence:
1. **Unencrypted**: Tritium genesis exchange (`MINER_AUTH_INIT` → `MINER_AUTH_CHALLENGE`)
2. **Falcon**: Challenge/response (`MINER_AUTH_RESPONSE` → `MINER_AUTH_RESULT`)
3. **ChaCha20**: Session key derived from genesis; all subsequent traffic encrypted
4. `STATELESS_MINER_READY` / `MINER_READY` to subscribe to push notifications

### Files Changed

| File | Change |
|------|--------|
| `src/dual_connection_manager.hpp` | Fixed `on_lane_failed()` to arm same-lane bypass; added `m_mining_lane` field and accessors; updated doc comments |
| `src/dual_connection_manager_test.cpp` | Updated Tests 2 & 3 to verify same-lane bypass; added Test 6 (cross-lane bypass never armed) and Test 7 (mining lane immutability) |
| `docs/current/sim-link.md` | Updated `DualConnectionManager` section; added Recovery Model section; corrected Lane Semantics table |
| `docs/PROTOCOL_LANES.md` | Added "Recovery and Failover Lane Invariant" section |
| `IMPLEMENTATION_SUMMARY.md` | Added this Recovery Model section |
