# Remaining Bottlenecks and Optimizations

## Overview

This page records the honest follow-on work after the current architectural
cleanup.  The goal is not to pretend the refactor is complete; it is to make
the remaining bottlenecks visible so contributors do not rediscover them
piecemeal.

## Structural bottlenecks

### 1. Multiple booleans still represent state machines

Auth, reward binding, reconnect state, and lane readiness are still expressed by
clusters of booleans.  That works, but it leaves room for impossible state
combinations and ad hoc transitions.

### 2. Session updates can still span multiple calls

The session container is authoritative, but some transitions still update
identity, reward, readiness, and diagnostics in multiple steps.  A scoped guard
would make these transitions easier to reason about and easier to roll back on
failure.

### 3. Packet ingress relies on discipline

The current packet-ingress path already resyncs and validates early, but the
shape is still procedural.  A named preflight gate would make it harder for a
future handler to bypass canonical validation.

### 4. Diagnostics are snapshot-heavy, history-light

`build_miner_session_diagnostics()` is useful, but many bugs are transition
bugs.  A per-session event journal would capture the “how we got here” story.

### 5. Type ambiguity still exists

Reward hash bytes, Genesis hash bytes, session fingerprints, and session IDs are
all meaningful values but still easy to confuse at the API level.

## Performance-oriented opportunities

### Fast vs full validation

A future validation split should keep the hot path small:

- fast checks for lane, auth, reward-ready, and minimal template safety
- full checks for reconnect recovery, acceptance harnesses, and debug runs

### Conflict-resolution strategy

Reconnect/recovery logic should have a documented policy for choosing among:

- cached local fields
- session-container state
- persisted values
- just-received packet data

### Acceptance-path instrumentation

The first-block acceptance harness will need timing and transition instrumentation
around auth, reward, template, mine, submit, and accept.  That same work can
feed future performance tuning.

## Optimization constraints

Any optimization must preserve the current architecture direction:

- the session container remains authoritative
- no host-endian or architecture-specific assumptions alter bytes on the wire
- no optimization reintroduces a second source of truth for session or reward
  state
- no validation shortcut bypasses the canonical session guard

## Suggested sequencing

1. packet-ingress preflight gate
2. canonical validation accessor(s)
3. staged merge / scoped update guard
4. stronger typed IDs and enums
5. per-session event journal
6. fast vs full validation split
7. first-block acceptance harness and scale tests

## Related docs

- [cpp-refactor-upgrade-path.md](cpp-refactor-upgrade-path.md)
- [../testing/first-block-acceptance-plan.md](../testing/first-block-acceptance-plan.md)
- [../testing/multi-miner-scale-tests.md](../testing/multi-miner-scale-tests.md)
- [../testing/session-recovery-and-reconnect-tests.md](../testing/session-recovery-and-reconnect-tests.md)
