# Current Miner Architecture

This section documents the current NexusMiner architecture direction after the
session-container and stateless-mining refactor series.  It mirrors the
node-side documentation effort so the miner and node describe the same design
boundaries, ownership rules, and upgrade path.

## Design Principles

- **Authoritative session container** — `SessionManager::MinerSessionContainer`
  is the per-session source of truth for auth, reward binding, lane metadata,
  and crypto context.
- **Explicit ownership** — `Solo` may cache values for hot paths, but it must
  resync from `NodeSessionContext` before submit, reward, and packet-ingress
  decisions.
- **Reconnect-safe reward binding** — the original reward string and decoded
  32-byte reward hash travel together so reconnect and resync paths keep the
  same payout semantics.
- **Lock-scoped validation** — session consistency is validated while the
  session mutex is held, so diagnostics and guards see a coherent snapshot.
- **Architecture-independent bytes** — packet bytes, reward hashes, genesis
  hashes, and ChaCha20 inputs must not vary by host architecture.
- **First accepted block as the acceptance target** — roadmap and tests are
  organized around the remaining work needed to reach a first accepted mined
  block safely and repeatably.

## Document Map

| Area | Document | Purpose |
|------|----------|---------|
| Architecture | [architecture/session-container-architecture.md](architecture/session-container-architecture.md) | Authoritative container, ownership, invariants |
| Architecture | [architecture/epoch-coordinator-architecture.md](architecture/epoch-coordinator-architecture.md) | Single source of truth for all epoch counters; prevents epoch desync |
| Architecture | [architecture/reconnect-and-resync-model.md](architecture/reconnect-and-resync-model.md) | Reconnect flow, packet-ingress resync, persistence |
| Architecture | [architecture/submit-path-ownership.md](architecture/submit-path-ownership.md) | Reward binding, crypto context, submit snapshot lifecycle |
| Architecture | [architecture/riscv-considerations.md](architecture/riscv-considerations.md) | Miner-side portability rules and RISC-V constraints |
| Roadmap | [roadmap/cpp-refactor-upgrade-path.md](roadmap/cpp-refactor-upgrade-path.md) | Remaining C++ refactors and upgrade sequencing |
| Roadmap | [roadmap/remaining-bottlenecks-and-optimizations.md](roadmap/remaining-bottlenecks-and-optimizations.md) | Honest list of open bottlenecks and follow-on improvements |
| Testing | [testing/first-block-acceptance-plan.md](testing/first-block-acceptance-plan.md) | Acceptance harness needed for first accepted mined block |
| Testing | [testing/multi-miner-scale-tests.md](testing/multi-miner-scale-tests.md) | Multi-session and multi-miner collision coverage |
| Testing | [testing/session-recovery-and-reconnect-tests.md](testing/session-recovery-and-reconnect-tests.md) | Reconnect, reward persistence, ingress preflight, and race tests |
| RISC-V | [riscv/README.md](riscv/README.md) | Miner-specific RISC-V landing page, including Julia QTV hook guidance |

## Upgrade-Path Diagram Index

The full-size diagrams live in [diagrams/](diagrams/).  Each diagram is
screenshot-friendly ASCII with thick-box formatting for PR review.

| # | Diagram | Summary |
|---|---------|---------|
| 01 | [Shared SessionBinding value object](diagrams/01-session-binding-diagram.txt) | Keep identity, reward, session, and lane data moving together. |
| 02 | [Canonical `ValidateConsistency()`](diagrams/02-validate-consistency-diagram.txt) | One validation entry point reused by auth, reconnect, reward, and submit. |
| 03 | [Stronger state machines](diagrams/03-state-machine-diagram.txt) | Separate auth, reward, reconnect, and lane states with explicit transitions. |
| 04 | [Live container vs reconnect snapshot](diagrams/04-live-vs-reconnect-diagram.txt) | Runtime state resyncs from the authoritative container after reconnect. |
| 05 | [Identity ownership model](diagrams/05-identity-ownership-diagram.txt) | Genesis, reward, session, fingerprint, and submit authority are distinct. |
| 06 | [Scoped update guard](diagrams/06-scoped-update-guard-diagram.txt) | Stage under lock, validate once, then commit or roll back. |
| 07 | [Reward string vs decoded reward hash](diagrams/07-reward-semantics-diagram.txt) | Preserve both human input and canonical 32-byte send payload. |
| 08 | [Multi-miner collision tests](diagrams/08-multi-miner-tests-diagram.txt) | Prove one miner session cannot satisfy another miner’s submit path. |
| 09 | [Canonical crypto context](diagrams/09-crypto-context-diagram.txt) | Reward send and block submit read the same authoritative crypto state. |
| 10 | [First accepted block harness](diagrams/10-first-block-harness-diagram.txt) | Auth → reward bind → channel → template → mine → submit → accept. |
| 11 | [Submit snapshot lifecycle](diagrams/11-submit-snapshot-lifecycle-diagram.txt) | Submitted block state must outlive template replacement until accept/reject. |
| 12 | [Fast vs full validation modes](diagrams/12-validation-modes-diagram.txt) | Keep hot-path guards cheap while preserving strict diagnostics. |
| 13 | [Per-session event journal](diagrams/13-session-event-journal-diagram.txt) | Ring-buffer event capture for auth, reward, reconnect, and submit debugging. |
| 14 | [Packet-ingress preflight gate](diagrams/14-packet-preflight-gate-diagram.txt) | Resync and validate before packet-specific logic runs. |
| 15 | [Strong semantic ID types](diagrams/15-strong-id-types-diagram.txt) | Compile-time wrappers reduce mix-ups between hashes, fingerprints, and IDs. |
| 16 | [Epoch coordinator — single source of truth](diagrams/16-epoch-coordinator-diagram.txt) | One coordinator owns session_epoch and recovery_epoch; monotonic, never resets. |

## Current Architecture Direction

### 1. Authoritative session ownership

`SessionManager::MinerSessionContainer` stores the miner-side session identity:
remote/local endpoints, lane, auth state, session ID, Falcon identity,
ChaCha20 key material, reward binding, channel state, and keepalive metadata.
`NodeSessionContext` exposes that state as the session-facing API.

### 2. `Solo` is no longer the independent source of truth

`Solo` still keeps local cached fields such as `m_authenticated`,
`m_session_id`, and `m_reward_bound`, but the current direction is explicit:
those values are derivative.  `refresh_cached_session_state()` and
`resync_auth_from_session_context()` pull from the authoritative session
container before reward send, GET_BLOCK, submit, and packet-ingress handling.

### 3. Submit and reward flows share the same crypto/session picture

The same session container exposes the session ID, Falcon identity, genesis,
ChaCha20 session key, reward string, reward hash, and lane metadata used by the
reward-bind path and the submit path.  This prevents “config-time state” and
“submit-time state” from drifting apart.

### 4. Lock-scoped validation is the intended guardrail

`validate_miner_session()` and `build_miner_session_diagnostics()` run against a
single snapshot under the session mutex.  The miner should continue to evolve
around canonical validation helpers instead of growing more ad hoc checks.

### 5. The acceptance bar is a first accepted mined block

The roadmap is not complete until the miner can produce a first accepted mined
block with strong coverage for reconnect, reward persistence, ingress resync,
collision isolation, and cross-architecture serialization stability.

## Related Existing Docs

- [docs/NODE_SESSION_CONTEXT.md](../../NODE_SESSION_CONTEXT.md)
- [docs/current/mining-protocols/stateless-mining.md](../mining-protocols/stateless-mining.md)
- [docs/current/mining/unified-tip-vs-channel-height.md](../mining/unified-tip-vs-channel-height.md)
- [docs/current/mining/height-tracker-canonical-state.md](../mining/height-tracker-canonical-state.md)
- [docs/riscv/RISCV-OVERVIEW.md](../../riscv/RISCV-OVERVIEW.md)
- [docs/riscv/BUILD-RISCV.md](../../riscv/BUILD-RISCV.md)
