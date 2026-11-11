# C++ Refactor Upgrade Path

## Goal

Complete the current miner-side C++ refactor series without regressing the
session-container model or losing the path to a first accepted mined block.

## What is already in place

The current codebase direction already includes the most important structural
moves:

- authoritative per-session container
- local cache resync into `Solo`
- reward binding persistence fields in the session container
- packet-ingress session resync before dispatch
- lock-scoped container validation and diagnostics
- accepted-submission snapshot handling
- lane-state ownership with strict mismatch rejection

This roadmap focuses on what remains.

## Upgrade path

### Phase 1 — Make the canonical contracts explicit

1. Define a shared `SessionBinding`-style value object or equivalent helper API
   so session identity, reward identity, and crypto identity are passed together.
2. Promote the current container validator into the canonical consistency entry
   point used by auth, reconnect, reward, submit, and diagnostics.
3. Replace loosely related booleans with stronger enums for auth, reward,
   reconnect, and lane state transitions where that improves clarity.

### Phase 2 — Harden packet ingress and staged updates

1. Introduce an explicit packet-ingress preflight gate so no handler can skip
   resync, lane validation, or session validation.
2. Adopt a scoped update/staged merge pattern for session mutations that span
   multiple fields.
3. Define conflict-resolution rules for reconnect cases where local caches,
   persistent state, and live packet data disagree.

### Phase 3 — Unify crypto context access

1. Expose a canonical crypto-context accessor for reward send and submit flows.
2. Keep reward address string and decoded reward hash as separate first-class
   values with documented semantics.
3. Make submit-key ownership and session-key ownership clearer in public APIs.

### Phase 4 — Improve observability and type safety

1. Add per-session event journals / ring buffers for auth, reconnect, reward,
   keepalive, template, and submit events.
2. Introduce strong semantic wrappers for session ID, Genesis hash, reward hash,
   Falcon key identifiers, and session fingerprints.
3. Separate fast validation from full validation so debug harnesses can request
   richer checks without penalizing hot paths unnecessarily.

### Phase 5 — Finish the acceptance harness

1. Build the first-mined-block acceptance harness.
2. Add multi-session and multi-miner collision coverage.
3. Add cross-architecture serialization stability checks.
4. Validate accepted-submission snapshot lifecycle under template churn.

## Current implementation status

- A shared `SessionBinding` value object now batches authoritative session,
  crypto, reward, and readiness semantics.
- `SessionManager` and `NodeSessionContext` now expose typed session-binding
  accessors so callers can consume one canonical bundle instead of piecemeal
  fields.
- `MiningTemplateInterface` now accepts typed `SessionId` construction and a
  batched `SessionBinding` update path while retaining backward-compatible raw
  overloads.
- `Solo` now propagates template ownership through the canonical binding bundle
  rather than separate session-id / epoch / identity calls.
- `SessionBinding` now also carries authoritative auth and ChaCha20 crypto
  state so reward-send and submit paths can batch their reads from one
  canonical source.
- Session-loss handling now treats the old session as non-viable: workers stop,
  then the miner performs full re-auth on the current connection or reconnects /
  fails over to a configured node if no active connection remains.
- The first-block acceptance harness now captures session-event-journal
  artifacts in full-validation mode and verifies accepted-submission snapshot
  consumption under template churn.
- The protocol test suite now includes deterministic cross-architecture
  serialization replay coverage and multi-miner/session isolation coverage.

## Next coding sequence for phases 5-6

1. Expand acceptance coverage from deterministic protocol fixtures into broader
   reconnect / failure-injection cases.
2. Grow the multi-miner matrix from protocol isolation tests into larger mixed
   lane / reconnect-storm coverage.
3. Remove transitional raw overloads only after all production and test call
   sites have switched to typed or batched APIs.
4. Add targeted observability around binding changes so future sweeps can prove
   cache resync paths stay aligned with the authoritative session container.

## Diagram-guided map

| Topic | Diagram |
|-------|---------|
| Session binding package | [01-session-binding-diagram.txt](../diagrams/01-session-binding-diagram.txt) |
| Consistency entry point | [02-validate-consistency-diagram.txt](../diagrams/02-validate-consistency-diagram.txt) |
| State-machine upgrade | [03-state-machine-diagram.txt](../diagrams/03-state-machine-diagram.txt) |
| Reconnect handoff | [04-live-vs-reconnect-diagram.txt](../diagrams/04-live-vs-reconnect-diagram.txt) |
| Ownership model | [05-identity-ownership-diagram.txt](../diagrams/05-identity-ownership-diagram.txt) |
| Staged merge | [06-scoped-update-guard-diagram.txt](../diagrams/06-scoped-update-guard-diagram.txt) |
| Reward semantics | [07-reward-semantics-diagram.txt](../diagrams/07-reward-semantics-diagram.txt) |
| Scale test model | [08-multi-miner-tests-diagram.txt](../diagrams/08-multi-miner-tests-diagram.txt) |
| Canonical crypto view | [09-crypto-context-diagram.txt](../diagrams/09-crypto-context-diagram.txt) |
| Acceptance harness | [10-first-block-harness-diagram.txt](../diagrams/10-first-block-harness-diagram.txt) |
| Submit lifecycle | [11-submit-snapshot-lifecycle-diagram.txt](../diagrams/11-submit-snapshot-lifecycle-diagram.txt) |
| Validation split | [12-validation-modes-diagram.txt](../diagrams/12-validation-modes-diagram.txt) |
| Event journal | [13-session-event-journal-diagram.txt](../diagrams/13-session-event-journal-diagram.txt) |
| Ingress gate | [14-packet-preflight-gate-diagram.txt](../diagrams/14-packet-preflight-gate-diagram.txt) |
| Typed IDs | [15-strong-id-types-diagram.txt](../diagrams/15-strong-id-types-diagram.txt) |

## Definition of done for this roadmap

The refactor series is not done when the code merely “looks cleaner.”  It is
done when the miner can:

- reconnect without losing authoritative session meaning
- prove reward binding and submit ownership stay in sync
- validate ingress state before packet-specific logic
- survive multi-miner collision scenarios
- produce a first accepted mined block under a documented acceptance harness
