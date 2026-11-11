# SIM Link — Lane Health and Same-Lane Recovery

`sim_link` no longer means “open two live mining lanes to the same node”.

The current model is:

- **One configured node + one configured lane per active `NodeSession`**
- **Reconnect/re-auth stay on that same configured lane**
- **Optional second node is failover only**
- **`DualConnectionManager` (DCM) is lane-health bookkeeping, not dual-live-lane mining**

---

## Current Architecture

```
                 ┌─────────────────────────────────┐
                 │          NexusMiner             │
                 │                                 │
                 │  Worker_manager                 │
                 │   ├─ NodeSession (active node)  │── TCP to configured port
                 │   ├─ DualConnectionManager      │   (8323 or 9323)
                 │   └─ ColinAgent                 │
                 │                                 │
                 │  Worker pool                    │
                 └─────────────────────────────────┘
```

If failover is configured, `Worker_manager` switches the **node endpoint** after repeated
primary failures, but it keeps the **same mining lane/port**.

---

## Recovery Model

Lane selection is immutable for a session lifetime.

1. **Initial connect**
   - `NodeSession::connect(..., callback)` targets the lane selected by TOML config.
   - The callback means the session is **fully authenticated and ready for session-bound mining flow**.

2. **Reconnect on the same node**
   - Retry the same endpoint/lane.
   - Re-authenticate on that same lane.

3. **Failover to a second node**
   - Switch only the node IP/host.
   - Keep the same lane/port as the primary session.
   - Perform a fresh full Falcon authentication handshake and obtain a new session ID.

The miner must **not** switch from Stateless→Legacy or Legacy→Stateless during reconnect or failover.

---

## What `sim_link` Means Now

`sim_link` is retained for compatibility with the existing lane-health / diagnostics plumbing.
It does **not** imply dual live same-node template delivery.

Today it mainly means:

- enable the existing lane-health bookkeeping paths
- preserve DCM/diagnostic reporting hooks
- keep reconnect/re-auth policy aligned with the configured primary lane

---

## DualConnectionManager (DCM)

`DualConnectionManager` remains a lightweight bookkeeper for lane state:

- which protocol lane is the miner’s configured mining lane
- whether a lane should be treated as alive/dead for diagnostics
- one-shot bypass flags used on same-lane recovery paths

DCM does **not** change the miner into dual-live-lane mode.

---

## Failover Rules

- `failover_wallet_ip = ""` disables failover
- `failover_port = 0` means “use the same port as the configured primary lane”
- if a mismatched failover port is configured, NexusMiner pins failover back to the primary lane/port

That keeps reconnect and failover semantics simple:

- **same node** → same lane, re-auth
- **different node** → same lane, full fresh auth

---

## Configuration

```toml
[wallet]
ip = "127.0.0.1"
port = 9323
failover_wallet_ip = "127.0.0.2"   # optional second node
failover_port = 0                  # 0 = same lane/port as primary

[network]
sim_link = true
```

---

## Summary

- **Primary** = one configured node/lane
- **Reconnect** = same node, same lane, re-auth
- **Secondary/failover** = different node, same lane, fresh auth
- **No live same-node opposite-lane mining session by default**
