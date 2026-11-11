# Failover Node Support

NexusMiner supports automatic failover to a secondary Nexus node when the primary becomes unreachable.
Colin's diagnostic report reflects the active/standby node status so operators can act quickly on failover events.

## How It Works

When a connection to the primary node fails repeatedly, the miner switches to the failover node after
`failover_max_retries` consecutive failures.  Once the failover node itself accumulates
`failover_max_retries` consecutive failures, the miner cycles back to the primary.  This cycling
behaviour continues indefinitely until one of the nodes accepts a connection.

```
Primary fails (N retries) → switch to Failover
Failover fails (N retries) → retry Primary
Primary fails (N retries) → switch to Failover
…
```

The switch is logged at `WARN` level so it is easy to spot in logs:

```
[Failover] Primary 192.168.1.10:9323 failed 5 times — switching to failover 192.168.1.11:9323
[Failover] Retrying primary 192.168.1.10:9323 after failover failures
```

## Configuration

Add the following keys to the `[wallet]` section of your `.config` file:

```toml
[wallet]
ip                  = "192.168.1.10"   # Primary node IP
port                = 9323             # Primary node port
failover_wallet_ip  = "192.168.1.11"   # Failover node IP
failover_port       = 9323             # Failover port (0 = same as primary)
failover_max_retries = 5               # Switch after this many consecutive primary failures
```

### Two pool IPs

```toml
[wallet]
ip                   = "pool1.example.com"
port                 = 9323
failover_wallet_ip   = "pool2.example.com"
failover_port        = 9323
failover_max_retries = 3
```

### Two home-node IPs (LAN)

```toml
[wallet]
ip                   = "192.168.1.10"
port                 = 9323
failover_wallet_ip   = "192.168.1.11"
failover_port        = 9323
failover_max_retries = 10   # tolerant of brief reboots
```

### Datacenter cluster

```toml
[wallet]
ip                   = "10.0.0.1"
port                 = 9323
failover_wallet_ip   = "10.0.0.2"
failover_port        = 9323
failover_max_retries = 3   # fast failover + monitoring alert on Colin warn
```

## `failover_max_retries` Tuning Guide

| Scenario | Recommended value | Rationale |
|---|---|---|
| LAN nodes | 10 | Tolerant of brief reboots or network blips |
| Remote / pool | 3–5 | Fast failover — remote outages are rarely transient |
| Production cluster | 3 | Fast failover + Colin warn triggers monitoring alert |

## Colin Diagnostic Report

Colin reports the failover state in every periodic diagnostic report.

### Healthy — primary active

```
[Colin]  FailoverNode │ ✅ PRIMARY active: 192.168.1.10:9323  │  standby: 192.168.1.11:9323  │  fails: 0/5
```

### Failover active — primary down

```
[Colin]  FailoverNode │ ⚠️  FAILOVER ACTIVE: 192.168.1.11:9323  │  primary DOWN: 192.168.1.10:9323  │  active for: 142s
[Colin]  ── Warnings ──────────────────────────────────────────
[Colin]    • FAILOVER NODE ACTIVE for 142s — primary node 192.168.1.10:9323 appears down
```

### Extended failover (> 5 minutes)

After 300 seconds on failover, Colin adds an extended-failover warning and a remediation recommendation:

```
[Colin]    • Extended failover: primary has been unreachable for 8 min — check primary node
[Colin]  ── Recommendations ──────────────────────────────────
[Colin]    • Inspect primary node 192.168.1.10:9323 — restart or check network
```

### No failover configured

When `failover_wallet_ip` is empty the failover section is silent (logged only at `DEBUG` level).
The report appears as though the section does not exist — single-node mode is the default.

## Warning Thresholds

| Condition | Severity | Threshold |
|---|---|---|
| Failover just activated | `WARN` | immediate (any failover active) |
| Extended failover | `WARN` + recommendation | > 300 s (5 min) |

Both warnings also appear in the `run_diagnostics()` pre-report check phase so they are
counted in the warnings list even if the report is not yet due.

## Session re-authentication on failover

When the miner switches to the failover node, NexusMiner performs a complete fresh Falcon handshake
(`MINER_AUTH_INIT` → `MINER_AUTH_CHALLENGE` → `MINER_AUTH_RESPONSE` → `MINER_AUTH_RESULT`) with the
failover node.  The old session ID from the primary node is discarded.  The failover node issues a new
session ID.  Mining resumes automatically once the fresh session is established. The failover
connection stays on the same configured lane/port as the primary session; only the node endpoint changes.

The protocol state is reset before scheduling the reconnect timer, so `login()` always sends
`MINER_AUTH_INIT` as if it were a brand-new connection.  The new session ID is logged on successful
authentication:

```
[Failover] Resetting protocol state for fresh Falcon re-authentication on 192.168.1.11:9323
[Failover] Fresh session established on failover node: session_id=3827461920
```
