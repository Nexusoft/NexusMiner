# Multi-Miner Scale Tests

## Goal

Document the remaining tests needed to prove that the authoritative session
model remains safe when more than one miner session is active or reconnecting.

## Why scale tests are required

The miner-side refactor is specifically trying to avoid “shared mental model”
bugs where a session ID, reward binding, or submit key from one miner leaks into
another miner’s view.  Multi-miner tests are therefore architectural tests, not
just performance tests.

## Required scenarios

### 1. Two miners, same node, separate sessions

Verify that miners A and B each get:

- independent session IDs
- independent reward binding state
- independent submit readiness
- independent accept/reject snapshots

### 2. Reconnect collision

Force miner A to reconnect while miner B stays live.  The test should prove that
A’s reconnect cannot overwrite B’s live session container view and that B’s
local caches do not “helpfully” resync from A’s state.

### 3. Shared reward string, distinct sessions

Use the same reward address string on multiple miners and prove that the session
container still tracks per-session ownership correctly.  The same payout target
must not imply shared session authority.

### 4. Submit collision

Construct a case where:

- miner A prepares a submit snapshot
- miner B receives a fresh template or accept/reject event
- accept/reject handling remains tied to the correct miner/session snapshot

### 5. Lane collision model

Run mixed legacy/stateless or mixed channel tests to confirm lane validation and
submit framing remain per-session and per-connection.

## Metrics worth capturing

- session IDs and lane assignments
- reward-binding source and decoded reward hash
- packet-ingress resync count
- submit snapshot create/consume counts
- accept vs reject attribution per miner
- any lane mismatch or session mismatch warnings

## Minimal scale matrix

| Matrix | Purpose |
|--------|---------|
| 2 miners × 1 node | baseline collision test |
| 4 miners × 1 node | stress reconnect and packet ingress |
| 2 miners × reconnect storm | resync and reward persistence |
| 2 miners × mixed lane/channel | framing and ownership isolation |
| 2 arch variants × 2 miners | serialization invariance under scale |

## Acceptance criteria

A multi-miner run passes only if every event can be attributed back to the
correct miner/session pair and no authoritative session container reports drift
caused by another miner’s activity.

## Current protocol-level coverage

The repository now includes focused collision/isolation tests that prove:

- the same reward string does not imply shared session authority
- miner A re-authentication does not mutate miner B session state or journal
- template ownership remains bound to the correct miner/session identity

Larger-scale reconnect-storm and mixed-lane matrices remain future work.

## Related diagram

- [08-multi-miner-tests-diagram.txt](../diagrams/08-multi-miner-tests-diagram.txt)
