# Miner RISC-V Diagnostics and Testing

## Goal

Make cross-architecture issues visible before they become session or submit
failures.

## What to compare across architectures

- session diagnostics snapshots
- decoded reward hash bytes
- keepalive payload bytes
- template-derived submit plaintext bytes
- ChaCha20 inputs and AAD labels
- accept/reject attribution after submit snapshot consumption

## Recommended checks

1. Run the same targeted miner/session tests on x86-64 and a RISC-V target or
   emulator-backed environment.
2. Compare serialized artifacts, not only pass/fail status.
3. Preserve failing artifacts so mismatches can be diagnosed offline.
4. Re-run after any change to packet builders, reward decoding, or crypto
   context accessors.

## Acceptance rule

A miner portability change is not complete until it shows that wire-visible and
crypto-visible bytes remain architecture-independent.
