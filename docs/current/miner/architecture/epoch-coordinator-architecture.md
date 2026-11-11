# Epoch Coordinator Architecture

## Overview

Prior to this change, the NexusMiner contained **five independent epoch counters**
that could silently desync, causing the miner to send `GET_BLOCK` requests on the
wrong epoch — most visibly as:

```
[TemplateInterface] Session epoch set to 1
[TemplateInterface] Session ID set to 0x548c90d3
[Worker_manager] → GET_BLOCK sent (recovery epoch 0)   ← MISMATCH
```

`EpochCoordinator` is a thread-safe utility class that serves as the **single
source of truth** for all epoch counters.  All components read from — and all
advances go through — this one coordinator, eliminating the desync by design.

## The Problem: Independent Counters That Drifted

| # | Counter | Location | Reset behaviour |
|---|---------|----------|-----------------|
| 1 | `session_epoch` | `SessionManager::SessionInfo` | Reset to 0 via `m_session = SessionInfo{}` in `clear_runtime_session_locked()` |
| 2 | `epoch` | `RecoveryContext` in `worker_manager.hpp` | Reset to 0 in `on_phase_enter(HEALTHY)` |
| 3 | `m_session_epoch` | `Solo` | Resynced from `SessionManager` but with a drift window |
| 4 | `m_session_epoch` | `HeightTracker` | Set via `set_session_epoch()` from Solo — further downstream |
| 5 | `m_session_epoch` | `MiningTemplateInterface` | Set via `set_session_epoch()` from Solo — even further downstream |

### Root cause

`clear_runtime_session_locked()` in `SessionManager` did:

```cpp
m_session = SessionInfo{};   // ← session_epoch silently reset to 0 here
```

Then `transition_to_authenticated_locked()` did:

```cpp
++m_session.session_epoch;   // ← 0 + 1 = 1 again
```

Meanwhile `RecoveryContext::epoch` in `Worker_manager` was never in a non-HEALTHY
phase during that auth cycle, so it stayed at 0.  Both counters now showed
plausible-looking values (0 or 1) but referred to different things, and nothing
tied them together.

## The Fix: EpochCoordinator

### Design

```cpp
class EpochCoordinator : public std::enable_shared_from_this<EpochCoordinator> {
public:
    // Epoch domains
    uint64_t session_epoch() const;
    uint64_t advance_session_epoch(const char* reason);   // monotonic ↑

    uint64_t recovery_epoch() const;
    uint64_t advance_recovery_epoch(const char* reason);  // monotonic ↑

    uint64_t global_epoch() const;   // max(session_epoch, recovery_epoch)

    // Observer pattern (called outside mutex)
    using EpochObserver = std::function<void(const char* domain,
                                             uint64_t old_val,
                                             uint64_t new_val)>;
    void add_observer(EpochObserver observer);

    // Atomic snapshot of all three values
    struct Snapshot { uint64_t session_epoch, recovery_epoch, global_epoch; };
    Snapshot snapshot() const;

    // Human-readable dump for logs/diagnostics
    std::string diagnostics() const;
};
```

### Key invariants

1. **Monotonicity** — Both `session_epoch` and `recovery_epoch` are
   monotonically increasing.  `advance_*()` never decreases the stored value,
   and `on_phase_enter(HEALTHY)` in `Worker_manager` no longer resets
   `recovery_epoch` to 0.

2. **No silent resets** — `clear_runtime_session_locked()` now restores epoch
   from the coordinator immediately after the `m_session = SessionInfo{}` reset,
   preventing the regression to 0:
   ```cpp
   const auto saved_epoch = m_session.session_epoch;
   m_session = SessionInfo{};
   if (m_epoch_coordinator) {
       m_session.session_epoch = m_epoch_coordinator->session_epoch();
   } else {
       m_session.session_epoch = saved_epoch;
   }
   ```

3. **Thread safety** — All reads and writes go through `m_mutex` in the
   coordinator.  Observers are called outside `m_mutex` to prevent deadlocks.

4. **Backward compatibility** — If no coordinator is wired (unit tests, minimal
   harnesses), `SessionManager` falls back to its local counter and emits a
   one-time warning log.

## Ownership and Wiring

`Worker_manager` creates and owns the coordinator.  The coordinator is wired
to `NodeSession` immediately after creation:

```cpp
// Worker_manager constructor
m_epoch_coordinator = std::make_shared<protocol::EpochCoordinator>();
m_primary_node_session->set_epoch_coordinator(m_epoch_coordinator);
```

`NodeSession::set_epoch_coordinator()` delegates to `SessionManager`:

```cpp
void NodeSession::set_epoch_coordinator(
    std::shared_ptr<protocol::EpochCoordinator> coordinator)
{
    auto session_mgr = m_session_context->get_session_manager();
    session_mgr->set_epoch_coordinator(std::move(coordinator));
}
```

`Solo` also accepts `set_epoch_coordinator()` for future downstream use.

## Epoch Advance Points

| Event | Call | Effect |
|-------|------|--------|
| Successful authentication | `transition_to_authenticated_locked()` → `coordinator.advance_session_epoch("authenticated")` | `session_epoch` increments; never resets |
| Recovery phase entry (non-HEALTHY) | `Worker_manager::transition_to()` → `coordinator.advance_recovery_epoch(reason)` | `recovery_epoch` increments; never resets |
| Transition to HEALTHY | `on_phase_enter(HEALTHY)` — does **NOT** reset recovery_epoch | Counter preserved across recovery cycles |
| Session clear / reauth prep | `clear_runtime_session_locked()` — restores epoch from coordinator | Prevents regression to 0 |

## Coordinator Durability Across Reconnect and Reauth

A key merge risk for any coordinator pattern is that it gets silently
disconnected from the component it guards during reconnect or reauth.  The
following chain proves the coordinator survives both paths unchanged.

### TCP reconnect path

```
Worker_manager::retry_connect()
  → transition_to(RecoveryPhase::RECONNECTING, "tcp_reconnect")
  → m_primary_node_session->reset()
      → NodeSession::reset()
          → m_session_context->end_session()
              → SessionManager::end_session()
                  → SessionManager::clear_for_disconnect()
                      → SessionManager::clear_runtime_session_locked()
```

`clear_runtime_session_locked()` only resets `m_session = SessionInfo{}` and
then restores the epoch from the coordinator.  It does **NOT** touch
`m_epoch_coordinator`.  The `SessionManager` member `m_epoch_coordinator` is a
`shared_ptr` that is set once by `set_epoch_coordinator()` and never cleared by
any session-lifecycle method (`clear_runtime_session_locked`,
`clear_for_disconnect`, `clear_for_reauth`, `end_session`).

After TCP reconnect completes and the Falcon handshake succeeds,
`transition_to_authenticated_locked()` calls
`m_epoch_coordinator->advance_session_epoch("authenticated")` on the same
live coordinator instance.  No re-wiring is needed.

### In-band reauth path

```
Worker_manager (in-band login path)
  → NodeSession login (does NOT call reset())
  → Solo processes AUTH_OK
  → SessionManager::transition_to_authenticated_locked()
      → m_epoch_coordinator->advance_session_epoch("authenticated")
```

The in-band reauth path never calls `reset()` at all; the coordinator is
trivially preserved.

### Summary

The coordinator is created once, stored in `Worker_manager`, and passed to
`SessionManager` once at startup.  Neither the TCP reconnect path nor the
in-band reauth path replaces `m_primary_node_session`, `SessionManager`, or
clears `m_epoch_coordinator`.  **No re-wiring after reconnect or reauth is
required.**

## HeightTracker and MiningTemplateInterface: Accepted Propagation Residual

`HeightTracker` and `MiningTemplateInterface` do not read from
`EpochCoordinator` directly.  They receive the epoch via
`Solo::refresh_cached_session_state()`, which is the same mechanism used for
session ID, auth state, and reward binding.

### Propagation chain

```
EpochCoordinator::advance_session_epoch()
  → SessionManager::m_session.session_epoch  (immediate, same call)
  → Solo::m_session_epoch                    (next refresh_cached_session_state())
      → HeightTracker::set_session_epoch()   (inside refresh_cached_session_state())
      → MiningTemplateInterface::set_session_epoch()
            (via propagate_session_to_template_interface() on auth resync or init)
```

### Why the lag is bounded and harmless

`refresh_cached_session_state()` is called at the top of every packet-ingress
path (`process_messages()`, GET_BLOCK handler, submit handler, reward handler,
auth handler).  The propagation lag therefore ends at the **first packet ingress
after the epoch advance** — in practice within milliseconds of authentication.

More importantly, the epoch fields in these two components are **diagnostic
only**:

| Component | How epoch is used | Wire path impact |
|-----------|-------------------|-----------------|
| `HeightTracker` | Tags `Snapshot.session_epoch` for diagnostics | None — not used for packet framing or submit decisions |
| `MiningTemplateInterface` | Stamps `MiningTemplate.session_epoch` at template adoption time | None — only used for template ownership logging and validation diagnostics |

Neither component's epoch field is consulted when building or transmitting a
wire packet.  A template stamped with epoch N that arrives in the brief window
before the first `refresh_cached_session_state()` call will still be processed
correctly; only its diagnostic tag will show the slightly stale epoch.

### Accepted residual

The propagation lag for `HeightTracker` and `MiningTemplateInterface` is
**accepted residual** behavior, consistent with how all other cached session
fields (session ID, auth state, reward binding) propagate via the same
`refresh_cached_session_state()` resync mechanism.  Direct coordinator access
for these two components would break the existing single-resync-point
architecture without correctness benefit.

## Observer Pattern

Components that need to react to epoch changes can register an observer:

```cpp
coordinator.add_observer([](const char* domain,
                            uint64_t old_val,
                            uint64_t new_val) {
    // domain = "session" | "recovery"
    // old_val, new_val: the epoch values before and after the advance
});
```

Observers are called after each advance, **outside** the internal mutex, so they
can safely re-enter the coordinator for reads.

## Diagnostics

The coordinator provides a human-readable snapshot for log messages and
diagnostic tools:

```
[EpochCoordinator] session=2 recovery=3 global=3
```

`Worker_manager` log lines that previously read `m_recovery.epoch` now call
`m_epoch_coordinator->recovery_epoch()` — the same authoritative value that
`SessionManager` increments.

## File Locations

| File | Purpose |
|------|---------|
| `src/protocol/inc/protocol/epoch_coordinator.hpp` | Class declaration |
| `src/protocol/src/protocol/epoch_coordinator.cpp` | Implementation |
| `src/protocol/epoch_coordinator_test.cpp` | 25 unit tests |

## Related Diagrams and Documents

- [16-epoch-coordinator-diagram.txt](../diagrams/16-epoch-coordinator-diagram.txt)
- [session-container-architecture.md](session-container-architecture.md)
- [reconnect-and-resync-model.md](reconnect-and-resync-model.md)
- [submit-path-ownership.md](submit-path-ownership.md)

## Tests

`epoch_coordinator_test.cpp` covers:

| Test | What it verifies |
|------|-----------------|
| Initial state | All epochs start at 0 |
| Session epoch monotonic | `advance_session_epoch()` always returns a value ≥ previous |
| Recovery epoch monotonic | `advance_recovery_epoch()` always returns a value ≥ previous |
| Global epoch | `global_epoch()` == `max(session_epoch, recovery_epoch)` |
| Observer notification | Observer receives correct domain, old, new values |
| Thread safety | 8 threads × 100 concurrent advances yield exactly 800 |
| No reset after session clear | Epoch continues from last value, never regresses to 0 |
