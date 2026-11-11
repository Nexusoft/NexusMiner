# PR-B Architecture Diagram — Degraded Mode Recovery Hardening

## Problem: The Old Doom Loop

```
Connection drops (TCP dead)
         │
         ▼
check_template_health() fires (every 30s)
         │
         ▼
Stage 3: both signals dead > 180s → retry_connect()
         │
         ▼  retry_connect() push-alive guard:
         │  since_push_s < 300s ?  ← YES (push was 5 min ago — still within 300s)
         │
         ▼
In-band re-auth (login()) on DEAD TCP socket
         │
         ▼
Payload transmitted → no response → 30s passes
         │
         ▼
check_template_health() fires again
         │
         ▼  ← SAME PATH: push still "recent" (now 5m30s ago < 300s)
         │
         ▼
In-band re-auth on dead socket again
         │
         └─────────────────────────────────────► DOOM LOOP (no TCP reconnect ever fires)
```

## Fix: The New Fast Recovery Path

```
Connection drops (TCP dead)
         │
         ▼
check_template_health() fires (30s after degraded entry)
         │
         ▼ NEW: Stage 0 fast-path check:
         │   ACK stale > 90s?  YES
         │   Push stale > 90s? YES
         │   Degraded > 30s?   YES
         │   Reconnect in progress? NO
         │
         ▼
retry_connect() called immediately
         │
         ▼  NEW push-alive guard threshold: 30s (was 300s)
         │  since_push_s < 30s?  NO (push was 5 min ago)
         │
         ▼
TRUE TCP RECONNECT proceeds:
  transition_to(RecoveryPhase::RECONNECTING, "tcp_reconnect")
  on_phase_enter(RECONNECTING) records entered_at timestamp
  m_primary_node_session->reset()
  exponential backoff timer fires → connect()
         │
         ▼
Falcon handshake → session_id received
         │
         ▼
transition_to(RecoveryPhase::HARD_RECOVERY, "reconnect_complete")
on_phase_exit(RECONNECTING) clears reconnect bookkeeping
m_recovery.degraded_since cleared by transition_to()
retry_template_request() → GET_BLOCK → template → mining resumes
```

## Escape Ladder (Updated)

| Stage | Trigger | Action | New Behaviour |
|-------|---------|--------|---------------|
| **Stage 0** | Both signals dead > 90s AND degraded > 30s | Immediate TCP reconnect | **NEW** — bypasses 60–180s ladder |
| **Stage 1** | 0–60s degraded | Retry GET_BLOCK | Unchanged |
| **Stage 2** | 60–180s, both stale | In-band re-auth via login() | Unchanged |
| **Stage 3** | > 180s, both stale | TCP reconnect | No longer blocked by stale push guard |
| **Hard Limit** | > 300s any state | Unconditional reconnect | Push guard tightened to 30s |

## The Push-Alive Guard Fix

```
BEFORE:
  retry_connect() push guard threshold = PUSH_ALIVE_THRESHOLD_SECONDS = 300s
  
  Effect: Any session degraded for < 300s with a push received in the last
  300s (almost always true) would NEVER do a TCP reconnect — only in-band
  re-auth on potentially dead socket.

AFTER:
  retry_connect() push guard threshold = RETRY_CONNECT_PUSH_LIVE_SECONDS = 30s
  
  Effect: Only suppress TCP reconnect if a push was received in the last 30s.
  A 30-second-old push proves the TCP connection is alive RIGHT NOW.
  A 5-minute-old push proves nothing about current TCP state.
```

## Reconnect Timeout Guard

```
transition_to(RecoveryPhase::RECONNECTING, "tcp_reconnect")
on_phase_enter(RECONNECTING) records m_recovery.entered_at = now()

check_template_health() (30s later):
  if is_reconnecting() && age > 60s:
    transition_to(RecoveryPhase::HARD_RECOVERY, "reconnect_timeout")
    on_phase_exit(RECONNECTING) clears reconnect bookkeeping
    → escape ladder resumes normally
    → retry_connect() will be called again if still degraded
```
