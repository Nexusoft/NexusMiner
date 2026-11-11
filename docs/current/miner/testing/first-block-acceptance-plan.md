# First-Block Acceptance Plan

## Goal

Define the remaining test work needed to reach a first accepted mined block with
confidence after the current session-container refactor series.

## Acceptance harness target

The harness should exercise the miner through the full happy path while keeping
artifacts available for diagnosis:

1. MINER_AUTH / auth response
2. SESSION_START and session-context population
3. reward bind request and reward bind success
4. channel selection and readiness
5. template receipt and template validation
6. mining loop / nonce discovery or deterministic fixture-based win
7. submit encode/transmit
8. ACCEPT / GOOD_BLOCK handling
9. accepted-submission snapshot consumption

## Harness requirements

### Deterministic orchestration

The harness should be able to drive node and miner state deterministically so a
submit is accepted predictably instead of relying on an uncontrolled live race.

### Artifact capture

Capture at least:

- session diagnostics before submit
- session event journal for auth / reward / submit transitions
- reward bind diagnostics
- template metadata and hashPrevBlock anchor
- submit snapshot contents
- node accept/reject response
- post-accept session state

### Dual validation mode

Support both:

- **fast acceptance runs** for iteration speed
- **full validation runs** with additional consistency checks and event logging

## Test matrix required for acceptance confidence

| Test area | Why it is required |
|-----------|--------------------|
| First-mined-block acceptance harness | Proves the end-to-end path is real, not only unit-tested |
| Reconnect and reward-binding persistence tests | Prevents payout/session regressions after reconnect |
| Submit-key ownership tests | Ensures submit authority belongs to the active session only |
| Session ingress resync tests | Confirms packet handlers run after resync, not before |
| Lock-scoped validation race tests | Protects authoritative container invariants under concurrency |
| Lane-transition and lane-mismatch tests | Prevents incorrect framing or mixed-lane recovery behavior |
| Multi-session / multi-miner collision tests | Verifies one miner cannot submit using another miner’s identity |
| Cross-architecture serialization checks | Ensures x86/ARM64/RISC-V produce identical bytes |
| Accepted submission snapshot lifecycle tests | Verifies accept/reject reporting remains tied to the submitted block |
| Packet-ingress preflight validation tests | Ensures preflight is mandatory for every packet-specific handler |

## Suggested harness phases

### Phase 1 — Controlled success path

Use a deterministic fixture or harness node mode that guarantees a valid submit
can be accepted.  The immediate goal is not benchmark realism; it is proving the
entire miner-side session and submit chain holds together.

### Phase 2 — Failure injection

Repeat the harness with injected failures:

- reward bind failure
- stale template before submit
- session mismatch ACKs
- reconnect between auth and reward bind
- reconnect between reward bind and submit
- template replacement between submit and accept

### Phase 3 — Cross-architecture replay

Replay the same harness artifacts on x86-64, ARM64, and RISC-V to verify packet
bytes, decoded reward hash bytes, Genesis hash bytes, and ChaCha20 inputs stay
identical.

The repository now includes deterministic protocol-level replay tests for those
serialization invariants and the full-validation harness captures event-journal
artifacts alongside accepted-submission snapshot consumption.

## Exit criteria

The miner should not claim the refactor is “acceptance ready” until the harness
can demonstrate at least one accepted block and preserve enough state to explain
why it succeeded or failed.

## Related diagram

- [10-first-block-harness-diagram.txt](../diagrams/10-first-block-harness-diagram.txt)
