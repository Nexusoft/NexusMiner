# RISC-V Considerations for Miner Architecture

## Overview

The miner-side RISC-V work is not only about getting the code to compile on a
new ISA.  The more important requirement is architectural neutrality: packet
serialization, reward decoding, Genesis handling, and crypto inputs must remain
stable regardless of whether the miner is running on x86-64, ARM64, or RISC-V.

## Non-negotiable rules

1. **No host-endian assumptions in packet bytes.**
2. **No host-endian assumptions in reward decode bytes.**
3. **No architecture-specific interpretation of GenesisHash bytes.**
4. **No architecture-specific changes to ChaCha20 inputs or AAD.**
5. **No lock or atomic behavior that depends on x86-specific ordering folklore.**

## Why this is miner-specific

The miner is responsible for several architecture-sensitive boundaries:

- decoding a reward address into canonical bytes
- building wire payloads for auth, reward, keepalive, template, and submit
- preserving block/header bytes exactly as the node expects
- logging diagnostics without changing byte order or field ownership

A node-side portability bug is bad; a miner-side portability bug can directly
produce invalid reward binds or invalid submits.

## Current design direction

### Architecture-independent serialization

Every byte sequence that reaches the wire or a crypto primitive should already
be in canonical order before it touches any architecture-optimized path.
Vectorized or accelerated code may speed up processing, but it must not change
field order, padding, or replay semantics.

### Canonical decoded identities

Genesis hashes, reward hashes, and submit identifiers should be treated as
opaque canonical byte arrays.  Code should avoid casual conversion to host-sized
integers unless the conversion is purely diagnostic and clearly bounded.

### Lock and atomic discipline

RISC-V portability is also about synchronization.  Code that is “accidentally
safe” on x86 due to stronger ordering should be rewritten around explicit lock
ownership or well-defined atomic semantics.

## Relationship to the general RISC-V docs

The existing repository-level RISC-V docs describe build profiles, extension
maps, and performance goals.  This page narrows the focus to the miner session,
packet, and submit paths so portability rules are tied directly to consensus and
session safety.

## Checklist for future refactors

- Keep packet parsers and builders architecture-neutral first, optimized second.
- Preserve canonical byte arrays for reward, Genesis, and session identifiers.
- Avoid reinterpreting byte storage with host-endian casts.
- Test serialization on at least one non-x86 target or emulator.
- Treat crypto input layout as part of the network contract, not an internal
  convenience.

## Related docs

- [../riscv/README.md](../riscv/README.md)
- [../riscv/endianness-and-serialization.md](../riscv/endianness-and-serialization.md)
- [../riscv/atomic-locking-considerations.md](../riscv/atomic-locking-considerations.md)
- [../../../riscv/RISCV-OVERVIEW.md](../../../riscv/RISCV-OVERVIEW.md)
