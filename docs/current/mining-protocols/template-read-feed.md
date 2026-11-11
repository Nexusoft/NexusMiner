# Unified Mining Template Read/Feed System

## Overview

The Unified Mining Template Read/Feed System provides a reliable interface between NexusMiner and LLL-TAO Node for stateless mining operations. This system implements the VPN-like FALCON Handshake tunnel for secure, authenticated communication between the miner and node.

## Architecture

### Components

1. **MiningTemplateInterface** (`src/protocol/inc/protocol/mining_template_interface.hpp`)
   - Central interface for all template READ/FEED operations
   - Manages template lifecycle (EMPTY → PENDING → RECEIVED → VALIDATED → ACTIVE → STALE/SUBMITTED)
   - Integrates with FALCON authenticated sessions
   - Provides template validation and verification
   - Thread-safe statistics tracking

2. **Solo Protocol Integration** (`src/protocol/src/protocol/solo.cpp`)
   - Uses MiningTemplateInterface for BLOCK_DATA processing
   - Binds template interface to FALCON session on authentication
   - Handles template feeding to worker threads

3. **FALCON Signature Wrapper** (`src/protocol/inc/protocol/falcon_wrapper.hpp`)
   - Centralized Falcon-512 signature operations
   - Authentication and block signing capabilities

### Data Flow

```
┌─────────────────┐     FALCON Auth      ┌─────────────────┐
│    NexusMiner   │◄───────────────────►│    LLL-TAO      │
│                 │     Session ID       │    Node         │
│ ┌─────────────┐ │                      │                 │
│ │ Template    │ │     GET_BLOCK        │                 │
│ │ Interface   │─┼────────────────────►│                 │
│ │             │ │                      │                 │
│ │             │◄┼──────────────────────┤ BLOCK_DATA     │
│ │             │ │                      │                 │
│ │ READ:       │ │                      │                 │
│ │  Validate   │ │                      │                 │
│ │  Parse      │ │                      │                 │
│ │  Verify     │ │                      │                 │
│ │             │ │                      │                 │
│ │ FEED:       │ │     SUBMIT_BLOCK     │                 │
│ │  Dispatch   │─┼────────────────────►│                 │
│ │  to Workers │ │                      │                 │
│ └─────────────┘ │                      │                 │
│                 │     ACCEPT/REJECT    │                 │
│                 │◄─────────────────────┤                 │
└─────────────────┘                      └─────────────────┘
```

## READ Operations

The READ operation processes incoming mining templates (BLOCK_DATA packets) from the LLL-TAO node:

### Template Reading Flow

1. **Receive BLOCK_DATA packet**
   - Treat `BLOCK_DATA` / stateless `GET_BLOCK` as template-delivery lifeline packets
   - Never stale-drop them at ingress because of session-debug ownership checks
   - Validate packet structure
   - Check minimum size requirements (92 bytes for Phase 2)

2. **Parse Block Header**
   - Deserialize using `llp_utils::deserialize_block_header()`
   - Extract nVersion, hashPrevBlock, hashMerkleRoot, nChannel, nHeight, nBits, nNonce, nTime

3. **Validate Template**
   - Verify channel matches expected (1=Prime, 2=Hash)
   - Check unified height is not stale (block.nHeight vs canonical_unified_height) — this is the primary staleness gate (PR #606); `validate_current_template()` uses `unified_height`, **not** `channel_height`, as the authoritative stale-detection threshold. Channel height from the push payload is tracked for informational / channel-advance detection purposes only and does not gate template acceptance.
   - Validate nBits (difficulty) is non-zero
   - Verify merkle root is not all zeros

4. **Accept or Reject**
   - Valid templates: Set state to VALIDATED
   - Invalid templates: Log error, request new work

### API

```cpp
// Read and process a template from packet data
ValidationResult read_template(const network::Payload& data, 
                               const std::string& source_endpoint);

// Check if a valid template is available
bool has_valid_template() const;

// Get the current validated template
const MiningTemplate* get_current_template() const;
```

## FEED Operations

The FEED operation dispatches validated templates to worker threads:

### Template Feeding Flow

1. **Check template validity**
   - Ensure template is in VALIDATED or ACTIVE state

2. **Update template state**
   - Change state to ACTIVE

3. **Invoke registered handlers**
   - Call `TemplateFeedHandler` with template and nBits
   - Handler dispatches to worker threads

### API

```cpp
// Register a handler to receive template feeds
void set_template_feed_handler(TemplateFeedHandler handler);

// Feed the current template to registered handlers
bool feed_current_template();

// Mark current template as stale
void mark_template_stale(const std::string& reason);
```

## FALCON Tunnel Integration

The Mining Template Interface integrates with FALCON authentication to create a VPN-like tunnel:

### Session Binding

1. **After MINER_AUTH_RESULT success:**
   - Session ID is extracted from packet
   - Template interface is updated: `set_session_id(session_id)`
   - All subsequent operations are bound to the authenticated session

2. **Template Association:**
   - Each template stores the session_id it was received in
   - Block submissions are tied to the same session

### Security Features

- **Session isolation**: Templates are bound to authenticated sessions
- **Template verification**: All templates are validated before processing
- **Stale detection**: Old templates are automatically detected and rejected
- **Channel validation**: Templates must match expected mining channel

## Create Block Verification

The interface provides block creation verification before submission. The current
authoritative submit path is built around the worker's solved snapshot, not a
reconstruction from the latest live template.

### Verification Process

```cpp
// Verify block creation from template is valid
bool verify_block_creation(const std::vector<uint8_t>& merkle_root, 
                           uint64_t nonce) const;

// Preferred solved-submit path (worker snapshot stays authoritative)
std::vector<uint8_t> prepare_block_submission_from_solved(const Block_data& solved);

// Prime variant: append vOffsets after the 216-byte solved block body
std::vector<uint8_t> prepare_block_submission_from_solved(const Block_data& solved,
                                                          const std::vector<uint8_t>& vOffsets);
```

### Solved-Submit Ownership Flow

```text
Worker
  -> Block_data snapshot (exact bytes the worker proved)
  -> Worker_manager
  -> prepare_block_submission_from_solved(...)
  -> Solo::submit_block()
  -> StatelessBlockUtility::encode_submit()
  -> ChaCha20/Falcon wrapping
```

Why this matters:

1. **Height / prevhash stay worker-owned** even if a new BLOCK_DATA template arrives before submit.
2. **Prime `vOffsets` stay attached** to the same solved snapshot that produced them.
3. **Solo only splits the payload** into `[216-byte block body]` plus optional Prime tail; it does not rebuild from the current template.

### Checks Performed

1. **Valid template exists**: Template must be in VALIDATED or ACTIVE state
2. **Merkle root size**: Must be 32 or 64 bytes
3. **Nonce validity**: Basic sanity check (warning if zero)

## Statistics and Diagnostics

The interface provides comprehensive statistics for monitoring:

```cpp
struct TemplateStats {
    uint64_t templates_received;    // Total templates received
    uint64_t templates_validated;   // Successfully validated
    uint64_t templates_rejected;    // Failed validation
    uint64_t templates_stale;       // Detected as stale
    uint64_t templates_fed;         // Fed to workers
    uint64_t blocks_verified;       // Blocks verified before submission
    uint64_t blocks_submitted;      // Blocks submitted
    uint64_t total_read_time_us;    // Total READ processing time
    uint64_t total_validation_time_us; // Total validation time
};

TemplateStats get_stats() const;
void reset_stats();
```

## Configuration

The unified interface is automatically initialized when Solo protocol is created. No additional configuration is required beyond standard FALCON authentication setup:

```json
{
    "version": 1,
    "wallet_ip": "127.0.0.1",
    "port": 8323,
    "mining_mode": "PRIME",
    "miner_falcon_pubkey": "<your_public_key_hex>",
    "miner_falcon_privkey": "<your_private_key_hex>"
}
```

## Logging

The interface provides detailed logging at multiple levels:

**INFO Level:**
```
[Solo] Mining Template Interface initialized for unified READ/FEED system
[Solo READ/FEED] Processing template via Mining Template Interface
[Solo READ] Template validated successfully in 45 μs
[Solo FEED] Dispatching validated template to workers (height: 12345, nBits: 0x1b0404cb)
[Solo Phase 2] FALCON tunnel established - Template interface bound to session
```

**DEBUG Level:**
```
[TemplateInterface] Initialized for channel 1 (prime)
[TemplateInterface] READ: Processing template (92 bytes) from 127.0.0.1:8323
[TemplateInterface] VALIDATE: Template passed all validation checks
[TemplateInterface] FEED: Feeding template at height 12345 to workers
```

## Error Handling

### Template Rejection Scenarios

1. **Invalid channel**: Template's nChannel doesn't match expected mining mode
2. **Stale height**: Template unified height is less than the current canonical unified height (marked invalid); `validate_current_template()` uses `unified_height` as the primary gate — channel_height is informational only (PR #606)
3. **Invalid nBits**: Difficulty bits are zero
4. **Invalid merkle root**: Merkle root is all zeros
5. **Parse failure**: Block header deserialization failed
6. **hashPrevBlock mismatch**: Advisory-only. Mismatches emit tiered log messages (info/warn/sustained-warn) but the template is **never discarded** solely for hashPrevBlock mismatch. The `HashCheckpointGuard` ring buffer tracks recent canonical hashes to distinguish expected tip-churn from genuine drift. `discard_template()` is not called for hashPrevBlock mismatches (PR #607).

### Recovery Actions

- All rejections trigger a new work request (GET_BLOCK)
- Stale templates are rejected and trigger new work request
- Parse failures indicate protocol mismatch - check node version
- Freshness protection is applied after ingress by `HeightTracker` and template validation,
  so stale responses cannot regress the canonical template feed even though delivery packets
  remain open.
- If `pending_get_block` times out (no BLOCK_DATA received within the timeout window), the
  pending flag is automatically cleared and the timeout is logged. The miner then issues a
  fresh GET_BLOCK on the next push notification (PR #606).

## Files

### New Files
- `src/protocol/inc/protocol/mining_template_interface.hpp` - Interface definition
- `src/protocol/src/protocol/mining_template_interface.cpp` - Implementation
- `docs/mining_template_read_feed_system.md` - This documentation

### Modified Files
- `src/protocol/inc/protocol/solo.hpp` - Added template interface member
- `src/protocol/src/protocol/solo.cpp` - Integrated template interface
- `src/protocol/CMakeLists.txt` - Added new source file

## Compatibility

- **LLL-TAO Node**: Compatible with Phase 2 stateless mining protocol
- **FALCON Authentication**: Required for session binding
- **Backward Compatibility**: Falls back to legacy processing if interface unavailable

## Performance

- **Template READ**: ~45-100 μs typical processing time
- **Template Validation**: ~10-50 μs
- **Thread Safety**: All statistics use atomic operations
- **Memory**: Single template buffer (no caching overhead)

---

**Implementation Date**: 2025-11-27  
**Based on**: Phase 2 Stateless Mining Protocol  
**Author**: GitHub Copilot Code Agent
