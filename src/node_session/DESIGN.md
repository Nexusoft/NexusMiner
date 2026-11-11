# NodeSession Design

## Current role

`NodeSession` now models **one configured mining node/lane session**.

That means:

- one active endpoint selected from config
- one active protocol lane (`8323` legacy or `9323` stateless)
- reconnect and re-auth stay on that same lane
- optional second node is handled by `Worker_manager` failover, not by opening a live same-node opposite lane inside `NodeSession`

## Contract

`NodeSession::connect(endpoint, callback)` means:

- TCP connection established
- Falcon authentication completed
- session-bound mining flow is ready

`Worker_manager` is allowed to treat the callback as “fully authenticated and ready”.

## Lane rules

- The configured port selects the lane.
- Lane selection is immutable for the lifetime of that session attempt.
- Reconnect on the same node reuses the same lane.
- Failover to another node also reuses the same lane.
- `NodeSession` does not perform cross-lane failover.

## Separation of responsibilities

### NodeSession

- owns the active mining connection/protocol pair
- owns authoritative session state via `NodeSessionContext`
- transmits and re-authenticates on the configured active lane
- forwards template/session/shutdown callbacks

### Worker_manager

- chooses the initial node endpoint
- decides when to reconnect
- decides when to switch to a failover node
- keeps failover on the same configured lane/port

### DualConnectionManager

- remains a lane-health/state bookkeeper
- does not imply dual live mining lanes

## Notes on dormant secondary plumbing

Some secondary-node plumbing still exists in `NodeSession` for compatibility with older
refactors, but it is no longer part of the active mining model. The active path is the
configured primary lane only.

## Testing focus

Current `NodeSession` coverage should continue to prove:

- connect callback waits for full authentication
- configured legacy and stateless primary lanes both work
- re-auth uses the active configured lane
- session-expired and node-shutdown events propagate correctly
- malformed packets stay diagnostic-only
- connection failure updates DCM for the configured lane
