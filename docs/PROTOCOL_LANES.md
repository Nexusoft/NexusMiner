# Protocol Lane Architecture

NexusMiner implements a dual-lane push-mining protocol that strictly separates legacy 8-bit wire framing from modern stateless 16-bit mirror-mapped wire framing.

## Overview

NexusMiner supports two protocol lanes, **strictly separated by port**:

* **Port 8323**: Legacy lane (8-bit opcodes, 1-byte headers)
* **Port 9323**: Stateless lane (16-bit opcodes, 2-byte headers, mirror-mapped 0xD0xx)

**Key principle:** Lane determination is port-based and immutable per connection. There is no heuristic byte-value detection and no fallback between lanes. Both lanes run the same push-notification mining flow (`MINER_READY` then `GET_BLOCK` template delivery); only opcode width and framing differ.

## Wire Format

### Legacy Lane (Port 8323)

```
[header: 1 byte][length: 4 bytes BE][payload: N bytes]
```

* **Header-only packets** (requests/responses): MUST emit an explicit zero-length frame on the wire: `[header][00 00 00 00]`
* **Data packets**: Include the 4-byte big-endian length field followed by the payload

### Stateless Lane (Port 9323)

```
[header: 2 bytes BE][length: 4 bytes BE][payload: N bytes]
```

* **Header-only packets** (requests/responses): MUST emit an explicit zero-length frame on the wire: `[header(2)][00 00 00 00]`
* **Data packets**: Include the 4-byte big-endian length field followed by the payload

### Zero-Length Framing Requirement

The current miner always transmits the 4-byte big-endian length field, even when the
payload length is zero. Bare-header packets are no longer accepted in beta builds; the
explicit zero-length frame is required on both lanes.

This fixes the historical framing bug where omitting the zero-length field on header-only
opcodes could leave the node waiting on a partial read, causing stalled submit parsing and
session drops under load. The miner and node now both send four zero bytes for zero-payload
opcodes so packet boundaries stay aligned.

## Lane Determination

The protocol lane is **strictly determined by the remote port**:

* **Port 8323** → LEGACY lane, 8-bit framing
* **Port 9323 (or any other port)** → STATELESS lane, 16-bit mirror framing

**Important:** This determination is immutable per connection. The protocol **never** attempts to detect the lane by examining byte values. Once a connection is established, its lane cannot change.

### No Cross-Lane Detection

Byte value 0xD0 (208) must **NOT** be used for cross-lane detection because:
* On the legacy lane, 0xD0 is MINER_AUTH_CHALLENGE (a valid legacy opcode)
* On the stateless lane, it is mirror-mapped to 0xD0D0

The parser uses the connection lane (determined from port) to decide header width and never guesses from byte values.

## Opcode Ranges

### Legacy Lane (8-bit opcodes: 0-255)

| Range | Type | Examples | Notes |
|-------|------|----------|-------|
| 0-127 | Data packets | BLOCK_DATA (0x00) | Include length + payload |
| 128-199 | Request packets | GET_BLOCK (0x81/129) | Zero-length framed (`len4=0`) |
| 200-203 | Response packets | ACCEPT (200), REJECT (201), COINBASE_SET (202), COINBASE_FAIL (203) | Zero-length framed (`len4=0`) |
| 204-205 | Round notifications | NEW_ROUND (204), OLD_ROUND (205) | Data-bearing (12 bytes preferred, 16 bytes legacy) |
| 206-212 | Auth/session | CHANNEL_ACK (206), MINER_AUTH_INIT (207), MINER_AUTH_CHALLENGE (208), MINER_AUTH_RESPONSE (209), MINER_AUTH_RESULT (210), SESSION_START (211), SESSION_KEEPALIVE (212) | Data-bearing |
| 213-214 | Reward management | MINER_SET_REWARD (213), MINER_REWARD_RESULT (214) | Data-bearing |
| 216 | Push notifications | MINER_READY (216) | Zero-length framed (`len4=0`) |
| 217-218 | Block availability | PRIME_BLOCK_AVAILABLE (217), HASH_BLOCK_AVAILABLE (218) | Data-bearing |
| 253-254 | Control | PING (253), CLOSE (254) | Zero-length framed (`len4=0`) |

### Stateless Lane (16-bit opcodes: 0xD000-0xD0FF)

| Range | Type | Examples | Notes |
|-------|------|----------|-------|
| 0xD000-0xD07F | Data packets | BLOCK_DATA (0xD000) | Include length + payload |
| 0xD080-0xD0C7 | Request packets | GET_BLOCK (0xD081) | **Data-bearing** (special case, 228-byte template push) |
| 0xD0C8-0xD0CB | Response packets | ACCEPT (0xD0C8), REJECT (0xD0C9), COINBASE_SET (0xD0CA), COINBASE_FAIL (0xD0CB) | Zero-length framed (`len4=0`) |
| 0xD0CC-0xD0CD | Round notifications | NEW_ROUND (0xD0CC), OLD_ROUND (0xD0CD) | Data-bearing (12 bytes preferred) |
| 0xD0CE-0xD0D4 | Auth/session | CHANNEL_ACK (0xD0CE), MINER_AUTH_INIT (0xD0CF), MINER_AUTH_CHALLENGE (0xD0D0), MINER_AUTH_RESPONSE (0xD0D1), MINER_AUTH_RESULT (0xD0D2), SESSION_START (0xD0D3), SESSION_KEEPALIVE (0xD0D4) | Data-bearing |
| 0xD0D5-0xD0D6 | Reward management | MINER_SET_REWARD (0xD0D5), MINER_REWARD_RESULT (0xD0D6) | Data-bearing |
| 0xD0D8 | Push notifications | MINER_READY (0xD0D8) | Zero-length framed (`len4=0`) |
| 0xD0D9-0xD0DA | Block availability | PRIME_BLOCK_AVAILABLE (0xD0D9), HASH_BLOCK_AVAILABLE (0xD0DA) | Data-bearing |
| 0xD0FD-0xD0FE | Control | PING (0xD0FD), CLOSE (0xD0FE) | Zero-length framed (`len4=0`) |

> **Block Format:** Both lanes transmit **216-byte Tritium blocks**. See
> [docs/reference/block-formats.md](reference/block-formats.md) for the complete
> field layout and the distinction from the historical 220-byte Legacy format.

## Block Format (Independent of Protocol Lane)

> **Important:** The protocol lane name ("Legacy" vs "Stateless") refers ONLY to the
> LLP framing and connection behaviour — NOT to the block serialization format.
> Both protocol lanes transmit the same 216-byte Tritium block format.

### Tritium Block (216 bytes) — Current Standard
Both lanes transmit this format:

| Offset | Size | Field | Notes |
|--------|------|-------|-------|
| 0 | 4 | nVersion | Block version |
| 4 | 128 | hashPrevBlock | Previous block hash |
| 132 | 64 | hashMerkleRoot | Merkle root |
| 196 | 4 | nChannel | Mining channel (1=Prime, 2=Hash) |
| 200 | 4 | nHeight | Block height |
| 204 | 4 | nBits | Difficulty target |
| 208 | 8 | nNonce | Miner nonce (mutable) |
| **Total** | **216** | | **nTime is NOT in wire template** |

`nTime` is **not transmitted** in the template. The node sets `nTime` internally via
`runtime::unifiedtimestamp()` during `sign_block()` validation.

### Legacy Block (220 bytes) — Historical Reference Only
The pre-Tritium format included `nTime` in the template:

| Offset | Size | Field | Notes |
|--------|------|-------|-------|
| 0–207 | 208 | (same as Tritium above) | |
| 208 | 8 | nNonce | Miner nonce |
| 216 | 4 | nTime | **Was included in wire template** |
| **Total** | **220** | | **No longer used in active mining** |

### Summary: What "Legacy" Means in This Codebase

| Term | Category | Meaning |
|------|----------|---------|
| "Legacy Lane" | Protocol framing | Port 8323, 8-bit opcodes, push (MINER_READY/GET_BLOCK) plus GET_ROUND fallback |
| "Stateless Lane" | Protocol framing | Port 9323, 16-bit opcodes, push (MINER_READY/GET_BLOCK) |
| "Legacy Block" | Block serialization | 220-byte format with `nTime` in template (historical) |
| "Tritium Block" | Block serialization | 216-byte format, `nTime` set by node (current standard for **both** lanes) |

## Key Design Decisions

### GET_BLOCK Behavior

Both lanes use push-notification behavior. After `CHANNEL_ACK`, the miner sends
`MINER_READY` and also sends `GET_BLOCK` for the first template because a spontaneous
push is not guaranteed to arrive immediately after authentication.

* **Legacy GET_BLOCK (0x81/129)**: 8-bit-framed request / template-delivery opcode.
* **Stateless GET_BLOCK (0xD081)**: 16-bit mirror-framed request / template-delivery opcode.

The intended semantic contract is the same: `GET_BLOCK` carries or requests the current
228-byte template (`12-byte metadata + 216-byte Tritium block`) depending on direction
and node timing.

### Auth Opcode Mirror Mapping

Authentication opcodes 206-218 on the legacy lane have mirror-mapped stateless definitions:

* CHANNEL_ACK: 206 → 0xD0CE
* MINER_AUTH_INIT: 207 → 0xD0CF
* **MINER_AUTH_CHALLENGE: 208 (0xD0) → 0xD0D0**
* MINER_AUTH_RESPONSE: 209 → 0xD0D1
* MINER_AUTH_RESULT: 210 → 0xD0D2
* SESSION_START: 211 → 0xD0D3
* SESSION_KEEPALIVE: 212 → 0xD0D4

**Critical note:** Auth opcode 208 (0xD0) is MINER_AUTH_CHALLENGE on the legacy lane. On the stateless lane, it is mirror-mapped to 0xD0D0. Because 0xD0 is a valid legacy auth opcode, it **must NOT** be used for cross-lane detection.

## TX/RX Enforcement

### Transmission (TX) Safety

The `Packet::get_bytes(ProtocolLane)` method enforces lane correctness during transmission:

* **LEGACY lane**: Rejects uint16 opcodes (stateless opcodes cannot be sent on legacy lane)
* **STATELESS lane**: Rejects uint8 opcodes (legacy opcodes cannot be sent on stateless lane)
* **UNKNOWN lane**: Refuses to transmit (returns empty buffer with error log)

This overload provides optional TX safety. It's not required if the packet is already lane-aware, but it provides an additional safety check.

### Reception (RX) Parsing

The parser uses the connection lane (determined from the remote port) to decide header width:

* **LEGACY lane**: Reads 1-byte headers and requires explicit 4-byte zero lengths for zero-payload opcodes
* **STATELESS lane**: Reads 2-byte big-endian headers and requires explicit 4-byte zero lengths for zero-payload opcodes

The parser **never** guesses the lane from byte values. It uses the port-determined lane for the entire connection lifecycle.

## Mirror Mapping Formula

Stateless opcodes are derived from legacy opcodes using the mirror mapping formula:

```
stateless_opcode = 0xD000 | legacy_opcode
```

This is implemented via the `LLP::MirrorOpcode()` function. For example:
* Legacy BLOCK_DATA (0x00) → Stateless 0xD000
* Legacy GET_BLOCK (0x81) → Stateless 0xD081
* Legacy ACCEPT (200) → Stateless 0xD0C8

This consistent mapping makes it easy to translate between legacy and stateless opcodes while maintaining strict lane separation.

## Summary

The dual-lane architecture provides:
* **Strict separation**: Legacy and stateless protocols never intermix
* **Port-based determination**: Lane is determined once at connection time based on port
* **No fallback**: Connections stay on their assigned lane for their entire lifecycle
* **Mirror mapping**: Consistent opcode translation between lanes
* **TX/RX enforcement**: Compile-time and runtime checks prevent lane violations

## Recovery and Failover Lane Invariant

**Protocol lanes are NEVER crossed during recovery or failover.**

The mining lane is determined at connection time (from the remote port via
`determine_lane_from_port()`) and is **immutable** for the session lifetime.

- **Primary retry**: same node, same lane, same port.
- **Failover**: different node, **same lane**, same port — with a full RE-AUTH sequence.

`DualConnectionManager::on_lane_failed()` arms the bypass on the **same** lane that
failed, not the opposite lane.  The `m_mining_lane` field is set once during initial
connection and is never modified by any recovery or failover operation.

This eliminates the former SIM-Link cross-lane bypass pattern where a stateless lane
failure could route recovery through the legacy lane and vice versa.

## Recovery-Epoch Gating Is Lane-Agnostic

There is exactly **one** mining engine (`Worker_manager`) and exactly **one**
`protocol::Solo` / `MiningTemplateInterface` implementation — there is no
`SoloLegacy`/`SoloStateless` split. `ProtocolLane` only tells the packet
serialization layer which opcode width and framing to use on the wire; it has
no bearing on template validation, recovery, or degraded-mode logic.

Consequently, the soft-vs-hard recovery policy in `Worker_manager` — gating
`mark_recovery_initiated()` behind `protocol::should_initiate_recovery_epoch()`
in both the template-validation-failure handler and the recovery-initiated
handler (`src/worker_manager.cpp`) — runs identically regardless of which lane
(Legacy port 8323 or Stateless port 9323) the active `NodeSession` is bound
to. A known, immediately-retriable template-local rejection (e.g.
`PRIME_ORIGIN_TOO_LOW`) is retried with a single `GET_BLOCK` and never
triggers an unnecessary `RecoveryPhase` transition or degraded-mode stats
flip, on either lane. There is no separate "Stateless Miner" code path that
would need this fix applied independently.

## Related: Mining Tip Anchoring

Push notifications (`PRIME_BLOCK_AVAILABLE` / `HASH_BLOCK_AVAILABLE`) are sent on
**both** legacy and stateless lanes and carry identical 12-byte payloads.  The
miner uses these events to detect two distinct staleness conditions:
`channel_advanced` and `tip_moved`.  See the canonical mining architecture doc
for details:

* **[docs/current/mining/unified-tip-vs-channel-height.md](current/mining/unified-tip-vs-channel-height.md)**
  — unified best tip, channel height, channel target, and two refresh reasons
