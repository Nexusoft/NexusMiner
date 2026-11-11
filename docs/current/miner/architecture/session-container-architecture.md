# Session Container Architecture

## Overview

The miner-side architecture now revolves around a single authoritative
per-session container: `SessionManager::MinerSessionContainer`.  This container
is the place where session identity, crypto readiness, reward binding, lane
metadata, and submit readiness are stored together.

The public surface for the rest of the miner is `NodeSessionContext`, which
wraps `SessionManager` and exposes session-centric operations such as
`get_session_info()`, `validate_miner_session()`, and
`build_miner_session_diagnostics()`.

## Why this matters

Without a canonical session container, the miner risks carrying stale copies of
session ID, reward binding state, or crypto keys in multiple classes.  The
recent refactor direction is explicitly trying to prevent that drift.

## Current ownership model

| Concern | Authoritative owner | Notes |
|---------|---------------------|-------|
| Session ID | `MinerSessionContainer::session_id` | Required for authenticated session state |
| Falcon identity | `falcon_pubkey`, `falcon_key_id`, `falcon_authenticated` | Set once auth succeeds |
| Genesis / ChaCha20 context | `session_genesis`, `chacha20_session_key`, `chacha20_key_fingerprint`, `chacha20_ready` | Shared by reward and submit flows |
| Reward binding | `reward_address_string`, `reward_hash`, `reward_bound`, `reward_binding_source` | Keeps config intent and decoded bytes together |
| Lane metadata | `active_lane`, `channel`, `ready_for_submit`, `ready_for_get_block` | Keeps packet framing and mining readiness aligned |
| Keepalive/fork canary input | `prevblock_suffix`, `last_keepalive`, `keepalive_count` | Session-scoped observability |

## Core invariants

The current validator already documents the direction the rest of the refactor
should preserve:

1. **Connected sessions must know their lane.**
2. **Authenticated sessions must carry a non-zero session ID and Falcon auth.**
3. **ChaCha20-ready sessions must carry genesis, key bytes, and a matching
   fingerprint.**
4. **Submit readiness must not be set ahead of reward binding.**
5. **Reward-bound sessions must preserve the decoded reward hash, not just the
    user-facing address string.**
6. **Reward-bind readiness must be decided from the authoritative session
   snapshot, including genesis and ChaCha20 key readiness, rather than from
   stale local caches.**

Those invariants are enforced through
`validate_miner_session_container_locked()` while `m_session_mutex` is held.
That lock-scoped validation is the miner-side equivalent of a canonical
`ValidateConsistency()` entry point.  Reward-send code should likewise use the
authoritative readiness predicate instead of guessing from partial local state.

## Authoritative container vs. local caches

`Solo` still uses local fields for hot-path convenience, but the architecture
now treats them as caches.  The canonical flow is:

1. Packet ingress updates or queries `NodeSessionContext`.
2. `Solo::refresh_cached_session_state()` resyncs local auth/session/reward/lane
   fields from `get_session_info()`.
3. Reward send, GET_BLOCK, submit, and packet handlers run against those
   refreshed values.
4. Any inconsistency is diagnosed using `validate_miner_session()` plus
   `build_miner_session_diagnostics()`.

The important direction is **not** “remove every local cache immediately.”  It
is “never let a local cache become a competing source of truth.”

## SessionBinding as the mental model

The codebase now exposes a `SessionBinding` value object, and the session
container remains the authoritative source used to populate it. Documentation
and future refactors should treat these fields as a package:

- session identity (`session_id`, `active_lane`)
- Genesis/Falcon/ChaCha20 identity (`session_genesis`, `falcon_key_id`,
  `chacha20_key_fingerprint`)
- reward identity (`reward_address_string`, `reward_hash`, `reward_bound`)
- readiness and recovery gates (`channel`, `ready_for_submit`,
  `ready_for_get_block`, `full_recovery_required`, `work_request_allowed`,
  `mining_ready`)

That framing prevents future changes from passing these values piecemeal and
reintroducing ownership bugs.

The important consequence is that recovery exit must consult the binding-level
predicates, not just raw template availability.  Local orchestration can say
“fresh work arrived,” but only the authoritative session container can say
whether that work is actually usable by a restored session.

## Remaining C++ refactor direction

### Canonical validation entry point

Today the validator lives in `SessionManager`.  A useful next step is to make
that consistency entry point more explicit and reusable by auth, reconnect,
reward, submit, and diagnostics code paths.

### Stronger state machines

Boolean fields are enough to express current state, but they are not the final
shape.  The roadmap points toward more explicit enums for auth, reward binding,
reconnect, and lane transitions.

### Strong semantic ID wrappers

`session_id`, reward hash bytes, genesis hash bytes, Falcon key identifiers, and
session fingerprints are distinct concepts.  The code would benefit from typed
wrappers so those values cannot be mixed accidentally.

## Related diagrams

- [01-session-binding-diagram.txt](../diagrams/01-session-binding-diagram.txt)
- [02-validate-consistency-diagram.txt](../diagrams/02-validate-consistency-diagram.txt)
- [03-state-machine-diagram.txt](../diagrams/03-state-machine-diagram.txt)
- [15-strong-id-types-diagram.txt](../diagrams/15-strong-id-types-diagram.txt)
- [16-epoch-coordinator-diagram.txt](../diagrams/16-epoch-coordinator-diagram.txt)

## Related documents

- [epoch-coordinator-architecture.md](epoch-coordinator-architecture.md) — Epoch desync root cause, coordinator design, wiring model, and invariants
