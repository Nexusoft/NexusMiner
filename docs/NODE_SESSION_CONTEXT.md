# NodeSessionContext Implementation

## Overview

This document describes `protocol::NodeSessionContext` as a thin façade over `SessionManager`, with `SessionManager` remaining the single authoritative mutable session record for the NexusMiner protocol stack.

## Problem Statement

Prior to this implementation, session management was fragmented across multiple components:

1. **Solo::m_session_id** - Legacy session ID storage (duplicated)
2. **Solo::m_session_manager->m_session.session_id** - Authoritative source (new)
3. **NodeSession::m_session_id** - Atomic uint32_t (unused/never updated)
4. **NodeSession::m_authenticated** - Atomic bool (duplicated state)

This duplication created several issues:
- **Unclear ownership**: Which component owns the session ID?
- **Stale data risk**: If Solo::m_session_id and SessionManager diverge
- **No synchronization**: No mutex protecting consistency between sources
- **Maintenance burden**: Changes must be applied to multiple locations

## Solution: NodeSessionContext

### Architecture

```
NodeSession
    └── NodeSessionContext (authoritative)
            └── SessionManager (wrapped)
                    └── SessionInfo
                            └── session_id (uint32_t)
                            └── state (SessionState enum)
```

### Key Design Principles

1. **Single Source of Truth**: `SessionManager` owns authoritative live session truth
2. **Thin Façade Pattern**: `NodeSessionContext` wraps `SessionManager` without adding competing business logic
3. **Simplified Interface**: Provides session-centric accessors for protocol constants
4. **Wire Protocol Helpers**: Colocates session parsing utilities (e.g., parse_session_start)
5. **Explicit Transition APIs**: Auth, reward, recovery, expiry, and reset flows are requested through named `SessionManager` transitions

### Responsibilities

`NodeSessionContext` exposes the authoritative session-domain decisions owned by `SessionManager`:

- **Authoritative runtime snapshot** - queried by all components
- **Session state machine** - DISCONNECTED → AUTHENTICATING → AUTHENTICATED → ACTIVE → EXPIRED
- **Reward lifecycle state** - NONE / REQUIRED / BINDING / BOUND / REJECTED / STALE
- **Recovery lifecycle state** - HEALTHY / RECOVERY_PENDING / RECOVERY_IN_PROGRESS / FORCED_REAUTH / RECONNECT_REQUIRED
- **Expiry/replay bookkeeping** - keepalive ACK state, expiry reason, deferred replay allowances
- **Lane-aware packet building** - `build_keepalive_packet()`, `build_session_status_packet()`
- **Session constants** - Keepalive cadence rules, retry caps (from ProtocolConstants)
- **Expiry detection** - "Invalidate session" actions (e.g., on mismatch ACK)
- **Wire parsing helpers** - `parse_session_start()` for SESSION_START packets

## Implementation Details

### Files Added

1. **src/protocol/inc/protocol/node_session_context.hpp**
   - Header file defining the NodeSessionContext class
   - Thin façade over SessionManager
   - Session-centric accessors for ProtocolConstants

2. **src/protocol/src/protocol/node_session_context.cpp**
   - Implementation of NodeSessionContext methods
   - Delegates all calls to underlying SessionManager
   - Implements `parse_session_start()` wire protocol helper

3. **src/protocol/node_session_context_test.cpp**
   - Comprehensive unit tests for NodeSessionContext
   - Tests session lifecycle, constants, wire parsing, etc.
   - All tests passing ✓

### Changes to Existing Files

#### NodeSession (src/node_session/)

**Before:**
```cpp
// Duplicate session state
std::shared_ptr<protocol::SessionManager> m_session_manager;
std::atomic<uint32_t> m_session_id{0};
std::atomic<bool> m_authenticated{false};

uint32_t session_id() const {
    if (m_primary_protocol) {
        return m_primary_protocol->get_session_id();  // Query Solo
    }
    return 0;
}

bool is_authenticated() const {
    return m_primary_protocol->is_authenticated() ||
           m_secondary_protocol->is_authenticated();
}
```

**After:**
```cpp
// Single authoritative source
std::shared_ptr<protocol::NodeSessionContext> m_session_context;

uint32_t session_id() const {
    return m_session_context ? m_session_context->get_session_id() : 0;
}

bool is_authenticated() const {
    return m_session_context ? m_session_context->is_authenticated() : false;
}

bool is_session_active() const {
    return m_session_context ? m_session_context->is_active() : false;
}
```

**Key Changes:**
- Removed `m_session_id` (atomic duplicate)
- Removed `m_authenticated` (atomic duplicate)
- Replaced `m_session_manager` with `m_session_context`
- All session queries now go through `m_session_context`

## API Reference

### Core Methods

```cpp
// Session ID (authoritative)
uint32_t get_session_id() const;

// Session state queries
bool is_authenticated() const;
bool is_active() const;
SessionManager::SessionState get_state() const;

// Session lifecycle
void begin_auth_handshake(const std::string& detail = "");
void start_session(uint32_t session_id,
                   const std::vector<uint8_t>& session_key = {},
                   const std::vector<uint8_t>& tritium_genesis = {});
void commit_authenticated_session(uint32_t session_id,
                                  const std::vector<uint8_t>& pubkey,
                                  const std::string& key_id,
                                  const std::vector<uint8_t>& tritium_genesis = {});
void begin_reward_binding(const std::string& reward_address,
                          const std::vector<uint8_t>& reward_hash = {},
                          const std::string& source = "");
void commit_reward_bound(const std::string& reward_address,
                         const std::vector<uint8_t>& reward_hash,
                         const std::string& source = "");
void commit_reward_rejected(const std::string& reward_address,
                            const std::string& source = "",
                            const std::string& reason = "");
void mark_session_expired(const std::string& reason);
void clear_for_disconnect(...);
void clear_for_reauth(...);
void end_session();
void set_state(SessionManager::SessionState state);

// Packet building
network::Shared_payload build_keepalive_packet() const;
network::Shared_payload build_session_status_packet(
    bool degraded, bool has_template,
    bool workers_running, bool secondary_up) const;

// Session constants accessors
static constexpr uint32_t get_keepalive_safety_divisor();
static constexpr uint32_t get_max_session_auth_retries();
static constexpr uint32_t get_base_session_retry_ms();
static constexpr uint32_t get_max_session_retry_ms();

// Wire protocol helpers
static bool parse_session_start(
    const std::vector<uint8_t>& packet_data,
    uint32_t& out_session_id,
    uint32_t& out_timeout,
    std::vector<uint8_t>& out_genesis);
```

`clear_for_reauth()` now represents **authoritative session restoration in
progress**, not a generic transport disconnect.  New code should preserve the
distinction between:

- local template refresh / degraded orchestration
- authoritative session recovery (`RECOVERY_IN_PROGRESS`)
- transport reconnect / failover

The corresponding `SessionManager` / `SessionBinding` helpers are:

- `session_requires_full_recovery()`
- `session_may_request_work()`
- `session_is_fully_mining_ready()`

## Benefits

### 1. Clear Ownership
- `SessionManager` is the authoritative source for mutable session state
- `NodeSessionContext` is the façade that keeps other components from mutating overlapping local truth
- No more ambiguity about which component owns session_id

### 2. Prevented Duplication Recurrence
- Components query `NodeSessionContext` instead of caching locally
- Eliminates the pattern that caused the original duplication

### 3. Simplified Testing
- Single component to test for session logic
- Mock-friendly interface for unit tests

### 4. Improved Maintainability
- Session-related changes in one place
- Easier to reason about session state flow

### 5. Thread-Safe by Design
- SessionManager's internal mutex protects session state
- All queries go through synchronized accessors

## Future Work

While this implementation establishes NodeSessionContext as the authoritative source for NodeSession, there are opportunities for further consolidation:

1. **Solo Integration**: Currently, Solo creates its own SessionManager internally. Future work could:
   - Pass a shared SessionManager to Solo's constructor
   - Remove Solo::m_session_id (legacy field)
   - Make Solo query NodeSessionContext for session state

2. **Worker_manager Integration**: Update Worker_manager to use NodeSessionContext directly

3. **Session Context Ownership**: Consider making NodeSessionContext the top-level owner that both NodeSession and Solo share

## Testing

All tests pass successfully:

```
Running NodeSessionContext unit tests...
✓ Session lifecycle test passed!
✓ Session constants test passed!
✓ parse_session_start test passed!
✓ Keepalive interval test passed!
✓ Tritium genesis test passed!
✓ Session manager access test passed!

All NodeSessionContext tests passed!
```

NodeSession tests also pass with the updated implementation:
```
=== NodeSession Unit Tests ===
✓ NodeSession created successfully
✓ Initial state is unauthenticated
✓ Initial session_id is 0
✓ Miner keys set
✓ Reward address set
...
=== All NodeSession tests passed! ===
```

## Conclusion

The current `NodeSessionContext`/`SessionManager` split addresses the session management duplication issues identified in the problem statement. `SessionManager` owns the runtime snapshot, transition validation, reset semantics, event journal, epoch advancement, and readiness/replay predicates, while `NodeSessionContext` remains a thin bridge for callers that should not depend on all `SessionManager` internals directly.
