# Miner RISC-V Notes

This section mirrors the node-side portability effort for the miner-specific
session, packet, reward, and submit paths.

## Scope

These pages focus on miner-specific portability rules:

- packet serialization must remain architecture-independent
- reward decode bytes must remain architecture-independent
- Genesis hash bytes must remain architecture-independent
- ChaCha20 inputs and AAD must remain architecture-independent
- locking and atomic assumptions must not rely on x86-only behavior

## Pages

| Document | Purpose |
|----------|---------|
| [build-notes.md](build-notes.md) | How the miner-specific RISC-V docs relate to the repo-wide build guides |
| [portability-checklist.md](portability-checklist.md) | Short checklist for portable session/packet changes |
| [endianness-and-serialization.md](endianness-and-serialization.md) | Canonical byte-order and serialization rules |
| [atomic-locking-considerations.md](atomic-locking-considerations.md) | Lock and atomic considerations for session/container code |
| [diagnostics-and-testing.md](diagnostics-and-testing.md) | Cross-architecture diagnostics and test expectations |
| [julia-qtv-hooks.md](julia-qtv-hooks.md) | Recommended Julia fixture hooks and isolated C++ bridge/engine boundary |

## Relationship to existing RISC-V docs

For general build and ecosystem information, see:

- [docs/riscv/RISCV-OVERVIEW.md](../../../riscv/RISCV-OVERVIEW.md)
- [docs/riscv/BUILD-RISCV.md](../../../riscv/BUILD-RISCV.md)
- [docs/riscv/RISCV-DIAGRAMS.md](../../../riscv/RISCV-DIAGRAMS.md)

This miner-side section is narrower: it documents the architecture-independent
rules that keep miner packets and crypto inputs identical across hosts.
