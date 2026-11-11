# Falcon Handshake and Adaptive Cache Management

## Overview

This document describes the Falcon Handshake and Adaptive Cache Management features introduced in NexusMiner to support both private localhost and decentralized public mining with enhanced security and efficiency.

## Features

### 1. Enhanced Falcon Handshake

The enhanced handshake protocol provides secure key exchange and session establishment with the LLL-TAO Node:

**Key Components:**
- **ChaCha20-Poly1305 Encryption**: Optional wrapping of Falcon Public Keys during handshake
- **Tritium GenesisHash Binding**: Links mining rewards to specific Tritium accounts
- **Session Key Exchange**: Establishes secure session keys for ongoing communication

**Handshake Flow:**
1. Miner sends `MINER_AUTH_RESPONSE` with:
   - Falcon Public Key (optionally ChaCha20-wrapped)
   - Authentication timestamp
   - Falcon signature
   - Optional Tritium GenesisHash (32 bytes)

2. Node responds with `MINER_AUTH_RESULT`:
   - Authentication status (success/fail)
   - Session ID (4 bytes, LE)
   - Optional error code

3. Node sends `SESSION_START` (optional):
   - Session timeout parameters
   - Falcon Session Key
   - Confirmed GenesisHash

### 2. ChaCha20 Encryption Wrapper

Protects Falcon Public Keys during transmission:

**Features:**
- ChaCha20-Poly1305 AEAD encryption
- 256-bit keys, 96-bit nonces
- Authenticated encryption with 128-bit tags
- OpenSSL 1.1.1+ or 3.0+ compatible

**Usage:**
- **Localhost Mining**: Optional (direct IPC communication is secure)
- **Remote Mining**: Auto-enabled (mandatory for public miners over HTTPS)

**Configuration:**
```json
{
    "enable_disposable_falcon": true
}
```

Note: Disposable Falcon signing defaults to true and is required for block signing. The configuration field can be used to explicitly confirm this setting.

### 3. Session Management

Manages authenticated sessions with adaptive cache management:

**Session States:**
- `DISCONNECTED`: No connection
- `AUTHENTICATING`: Handshake in progress
- `AUTHENTICATED`: Session established
- `ACTIVE`: Session active with keepalive
- `EXPIRED`: Session expired, needs re-onboarding

**Session Information:**
- Session ID (4 bytes)
- Falcon Session Key (from node)
- Tritium GenesisHash binding
- Keepalive tracking
- Session uptime statistics

### 3.1 Initial handshake key separation

The initial handshake is security-sensitive because three similarly named values
participate in different parts of the flow:

1. **Falcon identity key material** — the miner's Falcon public/private keypair
   proves miner identity and is what the node authenticates.
2. **Falcon session key** — an optional node-provided field from
   `SESSION_START`; this is distinct from the reward/submission encryption path.
3. **ChaCha20 session key** — a deterministic 32-byte key derived from the
   agreed `tritium_genesis` anchor and used for handshake wrapping, reward
   binding, and encrypted submit traffic.

`tritium_genesis` is therefore not "just another field". It is the blockchain
anchor that both peers use to derive the same ChaCha20 key and to agree on a
verifiable handshake context.

#### Miner-side invariants

- An authenticated authoritative session that still requires reward binding must
  retain the ChaCha20 key material needed by `MINER_SET_REWARD`.
- If generation-bound local caches are invalidated after a session epoch
  advance, the authoritative session snapshot must already hold the genesis and
  ChaCha20 state needed to repopulate those caches.
- Reset/clear helpers must not drop the authoritative ChaCha20 reward key during
  `begin_auth_handshake()` / auth-success transition unless the entire handshake
  is being abandoned.

### 4. Adaptive Keep-Alive Protocol

Maintains miner presence in node's cache through periodic pings:

**Features:**
- Configurable ping frequency (default: 24 hours = 1 ping/day)
- Adaptive scheduling (can increase to 2 pings/day in summer months)
- Automatic re-onboarding on cache expiry
- Session timeout tracking

**Configuration:**
```json
{
    "keepalive_interval": 24
}
```

Valid range: 1-168 hours (1 hour to 1 week)

**Recommended Settings:**
- **Normal Operation**: 24 hours (1 ping/day)
- **High Reliability**: 12 hours (2 pings/day)
- **Development/Testing**: 1-4 hours

### 5. Tritium GenesisHash Reward Binding

Links mined blocks to specific Tritium accounts for reward distribution:

**Configuration:**
```json
{
    "tritium_genesis": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
}
```

Format: 64 hexadecimal characters (32 bytes)

**Purpose:**
- Ties mining rewards to your Tritium account
- Enables stateless mining with proper reward attribution
- Validates miner ownership of the account

## Configuration Reference

### Complete Example Configuration

```json
{
    "version": 1,
    "wallet_ip": "127.0.0.1",
    "port": 8323,
    "local_ip": "127.0.0.1",
    "mining_mode": "PRIME",
    
    "miner_falcon_pubkey": "<your_897_byte_falcon_public_key_hex>",
    "miner_falcon_privkey": "<your_1281_byte_falcon_private_key_hex>",
    
    "tritium_genesis": "<your_32_byte_tritium_genesis_hash_hex>",
    "keepalive_interval": 24,
    "enable_disposable_falcon": true,
    
    "connection_retry_interval": 5,
    "get_height_interval": 2,
    "ping_interval": 10,
    "log_level": 2,
    "logfile": "miner.log",
    
    "stats_printers": [
        {
            "stats_printer": {
                "mode": "console"
            }
        }
    ],
    
    "print_statistics_interval": 10,
    
    "workers": [
        {
            "worker": {
                "id": "cpu0",
                "mode": {
                    "hardware": "cpu"
                }
            }
        }
    ]
}
```

### Configuration Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `tritium_genesis` | string | "" | Tritium account genesis hash (64 hex chars = 32 bytes) |
| `keepalive_interval` | number | 24 | Hours between SESSION_KEEPALIVE pings (1-168) |
| `enable_disposable_falcon` | boolean | true | Enable disposable Falcon block signing (required) |

## Usage Scenarios

### Scenario 1: Localhost Solo Mining

**Configuration:**
```json
{
    "wallet_ip": "127.0.0.1",
    "port": 8323,
    "tritium_genesis": "<your_genesis>",
    "keepalive_interval": 24,
    "enable_disposable_falcon": true
}
```

**Characteristics:**
- Direct IPC communication
- ChaCha20 encryption active (core security)
- Disposable Falcon signatures enabled (required for block signing)
- Standard keepalive (24 hours)
- Optimized for local trust environment

### Scenario 2: Remote/Public Mining

**Configuration:**
```json
{
    "wallet_ip": "mining.pool.com",
    "port": 8323,
    "tritium_genesis": "<your_genesis>",
    "keepalive_interval": 12,
    "enable_disposable_falcon": true
}
```

**Characteristics:**
- HTTPS/TLS connection
- ChaCha20 encryption active (core security)
- Disposable Falcon signatures enabled (required for block signing)
- Frequent keepalive (12 hours for reliability)
- Enhanced security for untrusted networks

### Scenario 3: Summer Mining (Reduced Uptime)

**Configuration:**
```json
{
    "keepalive_interval": 12
}
```

**Purpose:**
- Increase ping frequency during periods of reduced miner uptime
- Maintain cache presence despite intermittent operation
- Prevent cache eviction during downtime

## Protocol Details

### MINER_AUTH_RESPONSE Packet Format

```
[pubkey_len (2 bytes, LE)]
[pubkey data (897 bytes raw OR 913+ bytes wrapped)]
[timestamp (8 bytes, LE)]
[sig_len (2 bytes, LE)]
[signature (~690 bytes)]
[optional: tritium_genesis (32 bytes)]
```

**Total Size:**
- Without wrapping: ~1607 bytes + optional 32 bytes
- With wrapping: ~1623+ bytes + optional 32 bytes

### SESSION_START Packet Format (from Node)

```
[timeout (4 bytes, LE)]
[optional: session_key (32 bytes)]
[optional: genesis_hash (32 bytes)]
```

### SESSION_KEEPALIVE Packet Format

**Request:**
```
[session_id (4 bytes, LE)]
```

**Response:**
```
[remaining_timeout (4 bytes, LE)]
```

## Security Considerations

### ChaCha20 Encryption

**Strengths:**
- Modern, fast stream cipher
- AEAD (Authenticated Encryption with Associated Data)
- Immune to timing attacks
- 256-bit key security

**When to Use:**
- Always for remote/public mining
- Optional for localhost (adds defense in depth)

### Session Key Management

**Best Practices:**
- Do not conflate the Falcon identity/session fields with the genesis-derived
  ChaCha20 session key; reward binding depends on the latter.
- The authoritative post-auth session snapshot must expose the crypto material
  needed by both `MINER_SET_REWARD` and encrypted submit flows.
- Session keys are derived during handshake
- Keys are memory-cleared on session end
- Re-authentication required after expiry
- No key reuse across sessions

### Tritium Genesis Binding

**Security Benefits:**
- Prevents reward theft/misdirection
- Validates miner account ownership
- Enables trustless reward distribution
- Compatible with stateless mining

## Troubleshooting

### ChaCha20 Wrapping Failures

**Symptom:** "ChaCha20 wrapping failed" warning

**Solutions:**
1. Check OpenSSL version (require 1.1.1+ or 3.0+)
2. Verify key/nonce generation
3. Falls back to unwrapped key automatically

### Session Expiry

**Symptom:** "Session expired" or cache eviction

**Solutions:**
1. Reduce `keepalive_interval` (e.g., from 24 to 12 hours)
2. Check network connectivity
3. Monitor SESSION_KEEPALIVE responses
4. Allow automatic re-onboarding

### Authentication Failures

**Symptom:** "Authentication FAILED" error

**Possible Causes:**
1. Invalid Falcon keys
2. Tritium GenesisHash mismatch
3. Public key not whitelisted on node
4. Timestamp drift (check system clock)

**Solutions:**
1. Verify Falcon keys are correct (897/1281 bytes)
2. Confirm Tritium genesis matches node
3. Whitelist public key on node: `-minerallowkey=<pubkey>`
4. Synchronize system time

## Performance Impact

### ChaCha20 Wrapping

- **Encryption Time**: ~50-200 microseconds per pubkey
- **Size Overhead**: +16 bytes (authentication tag)
- **Impact**: Negligible on handshake (one-time per session)

### Keep-Alive Pings

- **Frequency**: Default 24 hours
- **Packet Size**: 4 bytes request, 4 bytes response
- **Network Impact**: Minimal (<1KB/day)

### Session Management

- **Memory**: ~200 bytes per session
- **CPU**: Negligible (state tracking only)
- **Impact**: None on mining performance

## Future Enhancements

Planned improvements:

1. **Challenge-Response**: Add nonce-based challenge to prevent replay attacks
2. **Multi-Session**: Support multiple concurrent authenticated sessions
3. **Key Rotation**: Periodic session key rotation for enhanced security
4. **Hardware Acceleration**: Utilize CPU extensions for faster ChaCha20
5. **Dynamic Keepalive**: Automatically adjust based on node cache policy

## References

- Falcon-512 Specification: https://falcon-sign.info/
- ChaCha20-Poly1305: RFC 8439
- LLL-TAO Phase 2 Stateless Mining
- Tritium Account System Documentation

---

**Version**: 1.0  
**Last Updated**: 2025-12-04  
**Compatibility**: NexusMiner 1.5+, LLL-TAO 5.1+
