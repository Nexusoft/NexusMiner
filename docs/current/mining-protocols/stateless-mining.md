# Stateless Mining Protocol (LLL-TAO PR #170)

> ⚠️ **Historical note**: Some opcode values and protocol flow descriptions in
> this document reflect an earlier design iteration (e.g. opcodes 0xD007,
> 0xD008, 0xD009 for `MINER_READY` / `GET_BLOCK` / `NEW_BLOCK`).  The current
> implementation uses the values documented in
> [PROTOCOL_LANES.md](../../PROTOCOL_LANES.md) and
> [push-notifications.md](push-notifications.md).  Push notifications are now
> used on both legacy 8-bit and stateless 16-bit lanes; the lanes differ only by
> wire opcode width/framing.  Push notifications are sent on **any** channel
> block (universal PoW tip push) and the miner refreshes for two distinct
> reasons (`channel_advanced` and `tip_moved`).  Refer to
> [unified-tip-vs-channel-height.md](../mining/unified-tip-vs-channel-height.md)
> for the authoritative description.

> **Update (PRs #605–#607, 2026-04):** The `hashPrevBlock` 3-strike discard
> mechanism described in earlier revisions has been **removed**. hashPrevBlock
> mismatches are now advisory-only; the miner logs tiered warnings but never
> discards or returns false for mismatch alone. A `HashCheckpointGuard` ring
> buffer of the last 10 canonical tip hashes is used to distinguish expected
> tip-churn from genuine drift. Additionally, `validate_current_template()` now
> uses `unified_height` (not `channel_height`) as the staleness gate, and the
> GET_BLOCK pending flag (`m_pending_get_block`) is now atomic and self-clearing
> on timeout.

## Overview

The stateless mining protocol is a modern push-notification based protocol that eliminates polling overhead and provides instant block updates. The same push behavior is now used on the legacy 8-bit lane; the lane names describe framing, not polling-vs-push behavior.

### Miner / Node Responsibility Boundary

> **Important:** The miner does **not** load, store, or transmit the transaction set.
>
> - **Node** — holds the full transaction set, builds and validates the complete block, assembles
>   the final serialized block when the miner submits a solved header.
> - **Miner** — receives a compact block template (header metadata only, ~216/220 bytes) from the
>   node, finds a valid nonce/hash, and submits the solved compact header plus a Falcon signature
>   back to the node.  The miner never handles 2 MB full block payloads.
>
> Any previous documentation or comments implying that the miner sends a full transaction-bearing
> block to the node reflected an early beta design that was not carried forward.

## Key Features

✅ **Push Notifications** - Node pushes templates to miner on both lanes (no polling)  
✅ **Instant Updates** - NEW_BLOCK pushed when blockchain advances  
✅ **Port-Selected Framing** - 8323 uses 8-bit framing; 9323 uses 16-bit mirror framing  
✅ **Lower Latency** - Immediate template delivery after MINER_READY  
✅ **Reduced Network Traffic** - No periodic GET_ROUND polling required  
✅ **Session-Based** - Maintains persistent connection with keepalive

---

## Protocol Flow

### Initial Connection and Authentication

```
Miner                                    Node
  |                                       |
  |--- MINER_AUTH (0xD000) ------------->|
  |    [genesis(32)][pubkey_len(2)]      |
  |    [pubkey(1793)][miner_id]          |
  |    (Falcon-1024 default; 897 bytes   |
  |     when Falcon-512 opt-in is used)  |
  |                                       |
  |<-- MINER_AUTH_RESPONSE (0xD001) -----|
  |    [success(1)][session_id(32)]      |
  |    [nonce(12)][challenge(64)]        |
  |                                       |
```

### Push Readiness

```
Miner                                    Node
  |                                       |
  |--- MINER_READY --------------------->|
  |    legacy 0xD8 / stateless 0xD0D8    |
  |                                       |
  |--- GET_BLOCK ----------------------->|
  |    legacy 0x81 / stateless 0xD081    |
  |                                       |
  |<-- GET_BLOCK / BLOCK_DATA -----------|
  |    [template data]                   |
  |    (Initial mining template)         |
  |                                       |
```

### Mining Loop (Stateless)

```
Miner                                    Node
  |                                       |
  |    ⛏️  Mining block...                 |
  |                                       |
  |<-- PRIME/HASH_BLOCK_AVAILABLE -------|
  |    [height/difficulty metadata]      |
  |    (Pushed when chain advances)      |
  |                                       |
  |--- GET_BLOCK ----------------------->|
  |<-- GET_BLOCK / BLOCK_DATA -----------|
  |    [template data]                   |
  |                                       |
  |    ⛏️  Switch to new template          |
  |    ⛏️  Continue mining...              |
  |                                       |
  |--- SUBMIT_BLOCK -------------------->|
  |    [block data]                      |
  |    (Found valid block)               |
  |                                       |
  |<-- ACCEPT (0x01) or REJECT (0x00) ---|
  |                                       |
```

### No Cross-Lane Fallback

```
8323 stays on 8-bit framing.
9323 stays on 16-bit mirror framing.
Wrong-lane packets are rejected/closed instead of falling back.
```

---

## Opcodes

### Stateless Tritium Protocol Opcodes (uint16_t, port 9323)

| Opcode | Value | Direction | Description |
|--------|-------|-----------|-------------|
| `MINER_AUTH` | 0xD000 | Miner → Node | Genesis-first authentication |
| `MINER_AUTH_RESPONSE` | 0xD001 | Node → Miner | Authentication result + session |
| `STATELESS_MINER_READY` | 0xD0D8 | Miner → Node | Signal stateless protocol support |
| `STATELESS_PRIME_BLOCK_AVAILABLE` | 0xD0D9 | Node → Miner | Push notification (Prime channel) |
| `STATELESS_HASH_BLOCK_AVAILABLE` | 0xD0DA | Node → Miner | Push notification (Hash channel) |
| `STATELESS_GET_BLOCK` | 0xD081 | Miner → Node | Request mining template |
| `SUBMIT_BLOCK` | 0x0005 | Miner → Node | Submit found block |
| `ACCEPT` / `REJECT` | 0x01 / 0x00 | Node → Miner | Block submission result |

### Legacy Tritium Protocol Opcodes (uint8_t, port 8323)

| Opcode | Value | Direction | Description |
|--------|-------|-----------|-------------|
| `PRIME_BLOCK_AVAILABLE` | 0xD9 | Node → Miner | Push notification (Prime channel) |
| `HASH_BLOCK_AVAILABLE` | 0xDA | Node → Miner | Push notification (Hash channel) |
| `GET_ROUND` | 0x05 | Miner → Node | Poll for template (legacy) |
| `BLOCK_DATA` | 0x06 | Node → Miner | Template response (legacy) |

**See:** [docs/reference/opcodes-reference.md](../../reference/opcodes-reference.md) for complete opcode reference.

---

## Push Notification Flow (Post-PR #122)

After authentication and channel selection:

1. Miner sends: `STATELESS_MINER_READY (0xD0D8)`
2. Node immediately sends: `STATELESS_PRIME_BLOCK_AVAILABLE (0xD0D9)` or `STATELESS_HASH_BLOCK_AVAILABLE (0xD0DA)`
3. Payload: 12 bytes [unified_height, channel_height, difficulty]
4. Miner requests template: `STATELESS_GET_BLOCK (0xD081)`
5. Node sends: 228-byte template [12 metadata + 216 block]
6. On new block: Node pushes step 2 again (event-driven, no polling)

---

## Packet Formats

### MINER_AUTH (0xD000)

**Genesis-First Authentication Packet**

```
Offset | Size | Field          | Description
-------|------|----------------|----------------------------------
0      | 32   | hashGenesis    | Tritium account genesis hash
32     | 2    | pubkey_len     | Falcon public key length
34     | 1793 | falcon_pubkey  | Falcon-1024 public key (unwrapped) [default]
       | or   |                | OR
34     | 1821 | falcon_pubkey  | Falcon-1024 public key (ChaCha20 wrapped)
       |      |                | [nonce(12)][ciphertext(1793)][tag(16)]
       | or   |                | OR (Falcon-512 opt-in only)
34     | 897  | falcon_pubkey  | Falcon-512 public key (unwrapped)
       | or   |                | OR
34     | 925  | falcon_pubkey  | Falcon-512 public key (ChaCha20 wrapped)
34+N   | 2    | miner_id_len   | Miner ID string length (N = pubkey field size above)
36+N   | var  | miner_id       | Miner ID string (e.g., "NexusMiner")
```

**Notes:**
- Genesis hash sent FIRST enables key derivation before pubkey parsing
- **Falcon-1024 is the default** (1793-byte public key, 1577-byte CT signature)
- Falcon-512 is available as an explicit opt-in for compatibility/testing only
- ChaCha20 wrapping auto-enabled for remote mining
- Session key = SHA256("nexus-mining-chacha20-v1" || genesis)

**See:** [docs/current/authentication/genesis-first-protocol.md](../authentication/genesis-first-protocol.md)

---

### MINER_AUTH_RESPONSE (0xD001)

**Authentication Result + Session Establishment**

```
Offset | Size | Field          | Description
-------|------|----------------|----------------------------------
0      | 1    | success        | 1=success, 0=failure
1      | 32   | session_id     | Session identifier
33     | 12   | nonce          | ChaCha20 nonce for challenge
45     | 64   | challenge      | Encrypted challenge data (for future use)
```

---

### MINER_READY (legacy 0xD8 / stateless 0xD0D8)

**Push Protocol Signal**

```
Offset | Size | Field          | Description
-------|------|----------------|----------------------------------
legacy:    [0xD8][00 00 00 00]
stateless: [0xD0D8][00 00 00 00]
```

**Purpose:**
- Signals to node that miner is ready for push template delivery
- Used on both 8-bit legacy and 16-bit stateless lanes
- Zero-length framing bytes are required

---

### GET_BLOCK (legacy 0x81 / stateless 0xD081)

**Initial Template Push**

```
Offset | Size | Field          | Description
-------|------|----------------|----------------------------------
0      | var  | template       | Mining template data
       |      |                | (format matches legacy BLOCK_DATA)
```

**Trigger:**
- Sent immediately after receiving MINER_READY
- Contains current best template for mining

---

### NEW_BLOCK (removed; use block-available + GET_BLOCK)

**Historical Updated Template Push**

```
Offset | Size | Field          | Description
-------|------|----------------|----------------------------------
0      | var  | template       | Updated mining template data
       |      |                | (format matches legacy BLOCK_DATA)
```

**Trigger:**
- Pushed when blockchain advances (new block found)
- Pushed when template becomes stale
- Miner should immediately switch to new template

---

## Auto-Negotiation Process

The miner automatically detects whether the connected node supports stateless protocol:

### Step 1: Authentication
```cpp
// Miner sends MINER_AUTH with genesis-first format
send_packet(MINER_AUTH, auth_data);

// Wait for MINER_AUTH_RESPONSE
auto response = recv_packet();
if (response.opcode == MINER_AUTH_RESPONSE && response.success) {
    session_established = true;
}
```

### Step 2: Protocol Negotiation
```cpp
// Attempt stateless protocol
send_packet(MINER_READY);

// Wait for response with timeout
auto response = recv_packet_with_timeout(5s);

if (response.opcode == GET_BLOCK) {
    // ✅ Node supports stateless protocol
    stateless_mode = true;
    current_template = parse_template(response.data);
    start_mining();
} else {
    // ❌ Node doesn't support stateless
    // Fallback to legacy polling
    stateless_mode = false;
    start_legacy_polling();
}
```

### Step 3: Mining Loop

**Stateless Mode:**
```cpp
while (mining) {
    // Mine current template
    mine_template(current_template);
    
    // Listen for NEW_BLOCK pushes
    if (packet_received(NEW_BLOCK)) {
        current_template = parse_template(packet.data);
        restart_mining();
    }
    
    // Submit found blocks
    if (block_found) {
        send_packet(SUBMIT_BLOCK, block_data);
        wait_for_accept_or_reject();
    }
}
```

**Legacy Mode:**
```cpp
while (mining) {
    // Poll for template periodically
    sleep(poll_interval);
    send_packet(GET_ROUND);
    
    auto response = recv_packet();
    if (response.opcode == BLOCK_DATA) {
        current_template = parse_template(response.data);
    }
    
    // Mine and submit as usual
    mine_and_submit();
}
```

---

## Implementation Details

### Keepalive Mechanism

To maintain session cache on the node:

```cpp
// Default keepalive interval: 24 hours
const auto keepalive_interval = config.keepalive_interval_hours;

// Send periodic keepalive pings
timer.schedule_periodic(keepalive_interval, []() {
    send_packet(KEEPALIVE_PING);
});
```

**See:** [docs/current/authentication/falcon-handshake-cache.md](../authentication/falcon-handshake-cache.md)

---

### Session Recovery

If connection is lost:

```cpp
// On disconnect
if (connection_lost) {
    // Re-authenticate with same genesis/keys
    send_packet(MINER_AUTH, auth_data);
    
    // Node may restore session from cache
    // Or create new session if cache expired
    auto response = recv_packet(MINER_AUTH_RESPONSE);
    
    // Resume stateless protocol
    if (response.success) {
        send_packet(MINER_READY);
        // Continue mining
    }
}
```

---

## Logging Output

### Successful Push-Lane Connection

```
[Falcon Auth] ✅ Authentication successful
[Falcon Auth]    Session ID: a1b2c3d4...
[Solo Protocol] LEGACY/STATELESS LANE: Using push protocol
[Solo Push] Sending MINER_READY
[Solo] ✨ GET_BLOCK template received!
[Solo]    Channel: 1 (Prime)
[Solo]    Height: 4523891
[Mining] ⛏️  Started mining on template...
```

### Template Push Notification

```
[Solo Push] 🔔 PRIME/HASH_BLOCK_AVAILABLE notification received!
[Solo Stateless]    New height: 4523892
[Solo Stateless]    Switching to new template...
[Mining] ⛏️  Restarted mining on new template
```

### Legacy 8-bit Push Lane

```
[Solo Protocol] LEGACY LANE: Using push protocol
[Solo Push] Sending MINER_READY (0xD8)
[Worker_manager] → GET_BLOCK sent
```

---

## Performance Comparison

### Stateless Protocol

| Metric | Value |
|--------|-------|
| **Template Latency** | < 10ms (instant push) |
| **Network Traffic** | ~100 bytes/block (NEW_BLOCK only) |
| **CPU Overhead** | Minimal (event-driven) |
| **Stale Work** | Near zero (instant updates) |

### Legacy 8-bit Push Protocol

| Metric | Value |
|--------|-------|
| **Template Latency** | < 10ms after push / immediate after initial GET_BLOCK |
| **Network Traffic** | ~100 bytes/block (push only) |
| **CPU Overhead** | Minimal (event-driven) |
| **Stale Work** | Near zero (instant updates) |

**Efficiency Gain:** ~95% reduction in network traffic, <90% reduction in template latency

---

## Node Requirements

### Node Support Detection

Stateless protocol requires:
- LLL-TAO node with PR #170 or later
- `miningport` enabled (default: 8323)
- Falcon authentication enabled

**Check node version:**
```bash
# In Nexus wallet console:
system/get/info

# Look for version >= 5.1.0 or stateless mining support
```

---

## Configuration

### Miner Configuration

**No special configuration needed!** The miner automatically:
1. Attempts stateless protocol
2. Falls back to legacy if unsupported
3. Uses optimal protocol for the connected node

**Example miner.conf (TOML):**
```toml
[wallet]
ip = "127.0.0.1"
port = 8323  # Stateless mining port

[mining]
channel = 1
genesis = "your_genesis_hash"
reward_address = "your_nxs_address"

[falcon]
pubkey = "your_falcon_pubkey"
privkey = "your_falcon_privkey"
```

**See:** [docs/reference/nexus.conf.md](../../reference/nexus.conf.md) for complete configuration reference.

---

### Node Configuration

**Example nexus.conf:**
```ini
# Enable mining
mining=1

# Mining port (default: 8323)
miningport=8323

# Whitelist miner public keys (optional)
# minerallowkey=<falcon_pubkey_hex>
```

---

## Troubleshooting

### "No stateless protocol support detected"

**Possible causes:**
1. Node version too old (< 5.1.0)
2. Node doesn't have PR #170
3. Mining not enabled on node

**Solution:**
- Update node to latest version
- Verify `mining=1` in nexus.conf
- Check node logs for mining server startup

---

### "Connection lost during mining"

**Possible causes:**
1. Network interruption
2. Node restart
3. Session expired (keepalive not working)

**Solution:**
- Miner will auto-reconnect
- Check keepalive_interval setting
- Verify network stability

---

### "Falling back to legacy protocol"

**This is normal!** The miner is designed to work with both new and old nodes.

**If you want stateless protocol:**
- Ensure node supports it (version check)
- Verify `port = 8323` in miner config
- Check node logs for stateless protocol messages

---

## Security Considerations

### Session Security

- Session IDs are cryptographically random (32 bytes)
- Genesis binding ensures rewards go to correct account
- ChaCha20 encryption protects public key in transit
- Falcon signatures provide quantum-resistant authentication

**See:** [docs/current/security/security-overview.md](../security/security-overview.md)

---

### Replay Protection

- Each authentication generates new session ID
- Nonces ensure unique encryption for each packet
- Session expiration prevents long-term replay attacks

---

## Migration from Legacy Protocol

### For Users

**No action required!** The miner automatically:
- Detects node capabilities
- Uses best available protocol
- Falls back gracefully if needed

### For Node Operators

**To enable stateless protocol:**
1. Update to LLL-TAO with PR #170+
2. Ensure `mining=1` in nexus.conf
3. Use `miningport=8323` (default)
4. Miners will auto-detect and use stateless protocol

**See:** [docs/upgrade-guides/legacy-to-stateless.md](../../upgrade-guides/legacy-to-stateless.md)

---

## References

- **Implementation PR:** LLL-TAO PR #170
- **Miner PR:** NexusMiner PR #91
- **Opcode Reference:** [docs/reference/opcodes-reference.md](../../reference/opcodes-reference.md)
- **Genesis-First Protocol:** [docs/current/authentication/genesis-first-protocol.md](../authentication/genesis-first-protocol.md)
- **Configuration Reference:** [docs/reference/nexus.conf.md](../../reference/nexus.conf.md)
