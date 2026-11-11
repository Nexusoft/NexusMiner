# Miner RISC-V Build Notes

## Overview

NexusMiner already ships repository-level RISC-V build documentation.  This page
records the additional miner-side expectations for the session, packet, and
submit code paths.

## Build guidance

1. Use the existing repository build guidance first:
   - [BUILD.md](../../../../BUILD.md)
   - [docs/riscv/BUILD-RISCV.md](../../../riscv/BUILD-RISCV.md)
2. Treat any miner-specific architecture optimization as optional until the
   canonical scalar path and serialization checks pass.
3. Validate the same session/reward/submit artifacts on non-RISC-V and RISC-V
   targets before declaring a portability change complete.

## Miner-side requirements

- The build may select different instruction sets or crypto back ends.
- The build must not change packet layouts or decoded reward bytes.
- The build must not change Genesis hash interpretation.
- The build must not change ChaCha20 nonce/key/AAD ordering.

## Practical note

When bringing up a new RISC-V environment, verify the miner-specific packet and
crypto tests before spending time on performance tuning.  Correct bytes matter
more than fast bytes.
