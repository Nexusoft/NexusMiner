# Block Serialization Formats

This document describes the two block wire formats used by the Nexus mining protocol.
These are **independent** of the protocol lane (see [PROTOCOL_LANES.md](../PROTOCOL_LANES.md)).

## Quick Reference

| Format | Size | nTime in template? | Used by |
|--------|------|--------------------|---------|
| Tritium | 216 bytes | ❌ No (set by node) | Both protocol lanes (current) |
| Legacy | 220 bytes | ✅ Yes | Historical only |

## Tritium Block Format (216 bytes) — Current Standard

Both the Legacy Lane (port 8323) and the Stateless Lane (port 9323) use this format.

```
Offset  Size  Field           Description
------  ----  -----           -----------
0       4     nVersion        Block version number
4       128   hashPrevBlock   Previous block hash (big-endian)
132     64    hashMerkleRoot  Merkle root hash (big-endian)
196     4     nChannel        Mining channel (1 = Prime, 2 = Hash)
200     4     nHeight         Block height
204     4     nBits           Encoded difficulty target
208     8     nNonce          Miner nonce (miner increments this)
─────────────────────────────────────────────────────
TOTAL   216   bytes
```

**Key property:** `nTime` is **not included** in the wire template. The node populates
`nTime` internally during `sign_block()` using `runtime::unifiedtimestamp()`. Miners
must NOT attempt to set or parse `nTime` from the template.

### Parsing in block_utils.hpp

```cpp
// Tritium blocks (216 bytes) — nTime NOT in wire format:
// [0-3]     nVersion (4 bytes)
// [4-131]   hashPrevBlock (128 bytes)
// [132-195] hashMerkleRoot (64 bytes)
// [196-199] nChannel (4 bytes)
// [200-203] nHeight (4 bytes)
// [204-207] nBits (4 bytes)
// [208-215] nNonce (8 bytes)
// TOTAL: 216 bytes

block.nTime = 0; // nTime not in Tritium template; set by node at sign_block() time
```

## BLOCK_DATA Metadata Prefix (12 bytes)

Every `BLOCK_DATA` response (from both GET_BLOCK and push-triggered template sends)
prepends a **12-byte metadata prefix** before the 216-byte Tritium block payload.
Total wire size: 228 bytes (12 + 216).

```
Offset  Size  Field           Semantics
------  ----  -----           ---------
0       4     nUnifiedHeight  Unified chain **TIP** (tStateBest.nHeight)
4       4     nChannelHeight  Channel **TIP** (stateChannel.nChannelHeight)
8       4     nBits           Difficulty target (compact format)
─────────────────────────────────────────────────────
TOTAL   12    bytes (all big-endian)
```

### Height semantics: TIP vs TARGET

| Wire field | Semantics | Value relative to chain state |
|-----------|-----------|-------------------------------|
| Metadata `nUnifiedHeight` [0–3] | **TIP** | `tStateBest.nHeight` |
| Metadata `nChannelHeight` [4–7] | **TIP** | `stateChannel.nChannelHeight` |
| Block `nHeight` [200–203] | **TARGET** | `tStateBest.nHeight + 1` |

**Critical**: `block.nHeight` (inside the 216-byte block at offset 200) is always
**one higher** than the metadata `nUnifiedHeight`.  The metadata is the current chain
state (TIP); the block header contains the height of the block being *mined* (TARGET).

NexusMiner uses metadata heights for HeightTracker (staleness detection) and
`block.nHeight` for ProofHash computation.  They must never be confused.

> **Node**: Only the [NamecoinGithub/LLL-TAO](https://github.com/NamecoinGithub/LLL-TAO)
> fork (branch `NODE`) sends this 12-byte metadata prefix.  The standard Nexusoft/LLL-TAO
> node sends raw 216-byte blocks without metadata.

## Legacy Block Format (220 bytes) — Historical Reference

> ⚠️ This format is **not used** in active mining. It is documented here for historical
> reference and compatibility context only.

```
Offset  Size  Field           Description
------  ----  -----           -----------
0       4     nVersion        Block version number
4       128   hashPrevBlock   Previous block hash (big-endian)
132     64    hashMerkleRoot  Merkle root hash (big-endian)
196     4     nChannel        Mining channel (1 = Prime, 2 = Hash)
200     4     nHeight         Block height
204     4     nBits           Encoded difficulty target
208     8     nNonce          Miner nonce
216     4     nTime           Unix timestamp (WAS in wire template)
─────────────────────────────────────────────────────
TOTAL   220   bytes
```

The 4-byte difference is `nTime` at offset 216. In the legacy format, miners set this
field in the template before submission. This required clock synchronization between
miner and node — a design flaw eliminated by the Tritium format.

## Why the Distinction Matters

The word **"Legacy"** in this codebase refers to:

1. **"Legacy Protocol Lane"** (Port 8323): polling behavior with 8-bit opcodes.
   Does NOT mean Legacy block format.
2. **"Legacy Block Format"** (220 bytes): historical serialization with `nTime`.

Both current protocol lanes (Legacy Lane + Stateless Lane) use **Tritium blocks (216 bytes)**.
This is intentional: the protocol polling behavior was decoupled from the block format
when Tritium mining was introduced.

### Confirmation in Live Logs

The log line `✓ Timestamp updated` from `sign_block()` on the node confirms the Tritium
path is active — the node is setting `nTime` itself, not trusting a miner-provided value.

## Related Documents

- [Protocol Lane Architecture](../PROTOCOL_LANES.md) — framing/opcode differences between lanes
- [IMPLEMENTATION_SUMMARY.md](../../IMPLEMENTATION_SUMMARY.md) — port-lane separation implementation
- [Stateless Mining Protocol](../current/mining-protocols/stateless-mining.md)
