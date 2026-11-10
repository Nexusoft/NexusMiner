# Reconnect and Resync Model

## Overview

The reconnect model is built around a simple rule: the authoritative session
container survives longer than any one hot-path cache.  When network or packet
ordering causes local state to lag, the miner resyncs from the container rather
than reconstructing a new truth from scattered fields.

## Runtime model

### Live container

The live container is the authoritative `MinerSessionContainer` instance guarded
by `m_session_mutex`.  It records:

- whether the miner is connected and authenticated
- which lane is active
- which session ID and Falcon identity are in force
- the reward binding source and decoded reward hash
- the current channel readiness and keepalive metadata

### Reconnect snapshot

`Solo` may temporarily hold stale copies of auth, session ID, reward binding,
and lane.  Those copies are intentionally treated as reconnect/recovery
snapshots, not as independent sources of truth.

`refresh_cached_session_state()` is the mechanism that reconciles the two.
Expected reconnect drift is logged as a resync event; unexpected drift is logged
as a warning.

## Packet-ingress preflight direction

`process_messages()` already does several important preflight tasks before it
enters packet-specific logic:

1. attach the current connection
2. update connection metadata
3. initialize protocol lane once from the port
4. resync cached session state
5. reject lane-mismatch packets immediately
6. reject invalid packets before dispatch

This is the current miner-side shape of a packet-ingress preflight gate.  The
remaining refactor work is to make that preflight even more explicit so future
packet handlers cannot bypass it.

## Reward binding persistence across reconnect

The reward-binding model has two pieces that must survive reconnect safely:

- the original reward address string supplied by config or runtime setup
- the decoded 32-byte reward hash actually used on the wire

`send_set_reward()` writes the reward address and decoded hash into the
session container before the encrypted packet is built.  `handle_reward_result()`
updates the same container when the node confirms or rejects the bind.

That split matters for reconnect because the miner needs both:

- the human-facing address for diagnostics and rebind intent
- the canonical bytes for submit readiness and cross-arch determinism

## Session start and resync

When a session start/auth sequence completes, the miner should converge on a
single state picture:

1. session ID established
2. Genesis/session key recorded
3. ChaCha20 readiness derived from that Genesis
4. reward binding preserved or re-established from the same authoritative state
5. lane/channel readiness updated only after the required prerequisites hold

The current code already logs this transition through session-start summaries and
`validate_authoritative_session()` calls.

## Session-loss rule

There is only one authoritative live session at a time. If that session is
lost, the miner does not keep mining under a "session recovery" model. Instead
it must:

1. stop workers immediately
2. establish a fresh authenticated session through in-band re-auth if the
   current node connection is still usable
3. reconnect or fail over to a configured alternate node if no active
   connection remains

Template refresh / GET_BLOCK recovery is still valid while a session remains
authenticated, but once the session is gone the miner must treat all previous
session-bound work as stale.

## Established transport loss

A mid-session socket close is not `SESSION_EXPIRED`. The node is gone, so
in-band login cannot succeed, and a push timestamp from before the interrupt
is not evidence that the TCP session is still alive.

`NodeSession::mark_lane_socket_failed()` therefore:

1. clears that lane's RX accumulator so a partial frame cannot poison the next
   connection
2. ends the authoritative session when the primary socket was already up
3. posts `transport_lost` unless `connect()` still owns the failure

`Worker_manager` stops workers and force-reconnects. It does not consult the
push-alive guard on that path. The transport-lost handler is registered from
the first `connect()`, after shared ownership exists — a `weak_ptr` captured
in the constructor is empty and would drop every notification. Duplicate
notifications are ignored only while a retry timer or `connect()` attempt is
actually outstanding. A socket that drops after the connect callback has
completed, while the manager is still `RECONNECTING` and waiting for work,
schedules a new reconnect immediately. The template health monitor repeats
the same recovery if the socket is down and no retry is armed, because
`get_active_protocol()` is null for the entire reconnect window and cannot be
the only place stall detection runs.

A solved block is snapshotted before the authentication and template gates.
If the socket is already gone, that worker-owned snapshot is kept for up to
180 seconds. Payload construction and transmission wait until the replacement
session can build them. The retry is one shot. A tip that is already known to
have moved is discarded locally; otherwise a moved tip is the node's rejection
to make.

## Local recovery phases vs authoritative session truth

The local `Worker_manager` recovery phase and the authoritative
`SessionManager` recovery state now have distinct roles:

- `WAITING_TEMPLATE` means the session is still authoritative enough to request
  fresh work, but the miner is waiting for a valid replacement template.
- `SESSION_RECOVERY` means the authoritative session itself is being restored
  (for example, after `SESSION_EXPIRED` or during an in-band re-auth flow).
- `RECONNECTING` means transport reconnect / failover is underway.

The current reconnect policy keeps both Hash and Prime in `WAITING_TEMPLATE`
for up to 300 seconds before transport reconnect is forced.  That longer
window reflects bursty chain behavior: even though the global chain often
advances faster, channel-specific template replacement can still see multi-
minute gaps while `hashPrevBlock` remains sensitive to fresh tip changes.

If the session is still alive but `SESSION_EXPIRED` or equivalent authority loss
is detected, controlled in-band auth is allowed only in a bounded window:

1. immediate first `AUTH_INT`
2. up to 3 total controlled auth attempts
3. 90 seconds between buffered retries
4. full stop after ~5 minutes of dead recovery or retry-budget exhaustion

Healthy exit requires both layers to agree.  A template arriving during
`SESSION_RECOVERY` is cached and may still be distributed to workers, but it
must **not** clear degraded state until the authoritative session reports:

1. no full-recovery requirement
2. work requests allowed
3. a live authenticated session on the active lane

That prevents the miner from treating “I have a template” as equivalent to
“I have a valid session.”

## Transport generation fencing

`NodeSession` now stamps each connect/reset cycle with a transport generation.
Every lane callback, deferred connect finalizer, and auth kickoff must match the
active generation before it can mutate session state.  In practice this acts
like a TCP epoch: once a reconnect/reset begins, any late event from the old
socket is ignored instead of resurrecting stale auth or lane state.

## Roadmap items still open

### Conflict resolution

The code can already detect stale-vs-authoritative drift, but reconnect recovery
still depends on call-site discipline.  The remaining work is to define which
class resolves conflicts for every field rather than only logging them.

### Fast vs full validation

Packet ingress and submit paths need a documented split between cheap hot-path
checks and more expensive diagnostics.  The current direction is:

- **fast validation** for lane, auth, reward-ready, and minimal template guards
- **full validation** for diagnostics, recovery, test harnesses, and debug builds

### Event journals

Reconnect bugs are hard to root-cause from a single snapshot.  A per-session
ring buffer of events would make auth/reconnect/reward/submit transitions easier
to audit.

## Test coverage still needed

- reconnect with cached reward binding and session metadata
- packet ingress resync after local cache loss
- lane mismatch rejection without corrupting authoritative session state
- reward binding persistence after disconnect/reconnect cycles
- packet preflight order assertions so handlers cannot skip resync/validate

## Related diagrams

- [04-live-vs-reconnect-diagram.txt](../diagrams/04-live-vs-reconnect-diagram.txt)
- [12-validation-modes-diagram.txt](../diagrams/12-validation-modes-diagram.txt)
- [13-session-event-journal-diagram.txt](../diagrams/13-session-event-journal-diagram.txt)
- [14-packet-preflight-gate-diagram.txt](../diagrams/14-packet-preflight-gate-diagram.txt)
