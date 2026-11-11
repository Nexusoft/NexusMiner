# Session Recovery and Reconnect Tests

## Goal

Document the reconnect-, resync-, and race-oriented coverage for the current
authoritative-session-first recovery model, and call out the remaining gaps.

## Coverage now in-tree

The current test suite already covers the highest-value recovery transitions:

- `degraded_recovery_test` exercises the explicit local recovery phases,
  including:
  - `SESSION_EXPIRED` escalation out of `WAITING_TEMPLATE`
  - blocking degraded exit while local state is in `SESSION_RECOVERY`
  - reconnect / template-recovery phase transitions
  - the 300-second `WAITING_TEMPLATE` reconnect window used by both Hash and Prime
- `node_session_test` covers transport-generation fencing, including the
  regression where a late pre-reset `connection_ok` must not resurrect the old
  session after `reset()`
- `node_session_context_test` covers authoritative recovery-state transitions,
  including `clear_for_reauth()` now entering `RECOVERY_IN_PROGRESS`
- `session_binding_test` covers the shared authoritative readiness predicates
  (`session_requires_full_recovery()`, `may_request_work()`,
  `is_fully_mining_ready()`)

## Required test buckets

### Reconnect and reward-binding persistence

Verify that reconnect does not lose:

- session Genesis / crypto context needed for secure reward and submit flows
- reward address string used for diagnostics and rebind intent
- decoded reward hash used as canonical wire bytes
- the distinction between “configured” reward intent and “live bind” result

### Controlled auth budget / terminal-stop policy

Add coverage around the bounded recovery policy so the miner cannot drift back
into silent reconnect churn:

- immediate first in-band auth attempt
- 90-second buffered retries after subsequent failures
- hard stop once the 3-attempt budget is exhausted
- hard stop once dead recovery exceeds the 300-second reconnect window

### Session ingress resync

Exercise packet ingress where local cached state is stale or empty but the
session container is already authoritative.  Each relevant packet path should
prove that resync occurs before packet-specific handling.

### Packet-ingress preflight validation

Add tests that assert the following order of operations:

1. connection/lane metadata refresh
2. cached session resync
3. lane/framing validation
4. session/container validation
5. packet-specific logic

### Lock-scoped validation race tests

Stress the authoritative session container with concurrent-looking transitions
and prove validation observes coherent snapshots.  Target scenarios include:

- auth success racing with reconnect cleanup
- reward bind result racing with template/update notifications
- lane update racing with packet processing
- submit readiness toggling while diagnostics are built

### Lane-transition tests

Even though lane changes are intentionally strict, tests should verify:

- unknown → resolved lane initialization
- lane mismatch disconnect behavior
- no fallback from wrong framing on the wrong port
- session container remains internally consistent after rejection

### Accepted submission snapshot lifecycle

Prove that:

- submit snapshot is created before encode/transmit
- accept/reject consumes the snapshot exactly once
- fallback behavior is explicit when the snapshot is absent
- later templates do not rewrite the accepted/rejected block attribution

### Cross-architecture / serialization stability

For x86-64, ARM64, and RISC-V (or emulator-backed equivalents), compare:

- reward decode bytes
- Genesis hash bytes
- keepalive payload bytes
- submit payload plaintext bytes before encryption
- ChaCha20 ciphertext inputs and AAD choices

## Recommended sequencing

1. ingress preflight order assertions
2. reconnect/reward persistence coverage
3. lock-scoped validation race tests
4. submit snapshot lifecycle tests
5. cross-architecture replay checks

## Related diagrams

- [04-live-vs-reconnect-diagram.txt](../diagrams/04-live-vs-reconnect-diagram.txt)
- [11-submit-snapshot-lifecycle-diagram.txt](../diagrams/11-submit-snapshot-lifecycle-diagram.txt)
- [12-validation-modes-diagram.txt](../diagrams/12-validation-modes-diagram.txt)
- [14-packet-preflight-gate-diagram.txt](../diagrams/14-packet-preflight-gate-diagram.txt)
