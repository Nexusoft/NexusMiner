# NexusMiner Configuration Reference

Complete reference for all configuration options in NexusMiner.

## Table of Contents
1. [Configuration Formats](#configuration-formats)
2. [Connection Settings](#connection-settings)
3. [Authentication](#authentication)
4. [Mining Configuration](#mining-configuration)
5. [Worker Configuration](#worker-configuration)
6. [Hardware-Specific Settings](#hardware-specific-settings)
7. [Network Settings](#network-settings)
8. [Logging](#logging)
9. [Advanced Options](#advanced-options)
10. [Security Best Practices](#security-best-practices)

---

## Configuration Formats

NexusMiner supports two configuration formats:

### TOML Format (.config) - Recommended
Simpler, more readable format with comments support.
```toml
[wallet]
ip = "127.0.0.1"
port = 8323

[mining]
channel = 1
```

**Example Files:**
- [configs/cpu-prime-solo.config](../../configs/cpu-prime-solo.config) - CPU Prime starter
- [configs/gpu-hash-solo.config](../../configs/gpu-hash-solo.config) - NVIDIA GPU Hash starter
- [configs/MASTER-REFERENCE.config](../../configs/MASTER-REFERENCE.config) - Options reference

See [docs/reference/toml-format.md](toml-format.md) for complete TOML guide.

### JSON Format (.conf) - Legacy
Filenames not ending in `.config`, including `.conf` and `.toml`, use the JSON parser.
```json
{
    "wallet_ip": "127.0.0.1",
    "port": 8323,
    "mining_mode": "SOLO"
}
```

**Example Files:**
- `docs/reference/config-examples/MASTER_REFERENCE.conf` - Complete JSON documentation
- `docs/reference/config-examples/basic-example.conf` - Simple JSON template

---

## Connection Settings

### `wallet_ip` / `[wallet] ip`
**Type:** String  
**Default:** `"127.0.0.1"`  
**Description:** IP address of the Nexus node to connect to.

**Examples:**
```toml
# Local node
[wallet]
ip = "127.0.0.1"

# Remote node
[wallet]
ip = "192.168.1.100"
```

**Notes:**
- Use `127.0.0.1` for local node mining
- Use remote IP for network mining
- Ensure node is running and accessible

---

### `port` / `[wallet] port`
**Type:** Integer  
**Default:** `8323`  
**Valid Range:** 1-65535  
**Description:** Mining port on the Nexus node.

**Mining Lanes:**
- Port `8323`: Legacy 8-bit framing with modern push behavior
- Port `9323`: Stateless 16-bit mirror framing with the same push behavior
- Both lanes send `MINER_READY`, use `GET_BLOCK` for template delivery, and require explicit zero-length frames for zero-payload opcodes

**Examples:**
```toml
# Legacy 8-bit push lane
[wallet]
port = 8323

# Stateless 16-bit push lane
[wallet]
port = 9323
```

**See:** [docs/current/mining-protocols/stateless-mining.md](../current/mining-protocols/stateless-mining.md) for protocol details.

---

## Authentication

### `miner_falcon_pubkey` / `[falcon] pubkey`
**Type:** String (Hex)  
**Length:** 897 bytes (Falcon-512) or 1793 bytes (Falcon-1024), encoded as hex
**Description:** Falcon public key for miner authentication.

**Generation:**
```bash
# Generate keys; copy the public and private hex strings into [falcon]
./NexusMiner --create-keys
```

**Security:**
- ✅ Falcon-1024: 256-bit quantum security (recommended, default)
- ✅ Falcon-512: 128-bit quantum security
- ⚠️ Public key can be shared safely

**Example:**
```toml
[falcon]
pubkey = "0123456789abcdef..."  # 897 or 1793 bytes hex
```

**See:** [docs/current/authentication/falcon-keygen-guide.md](../current/authentication/falcon-keygen-guide.md)

---

### `miner_falcon_privkey` / `[falcon] privkey`
**Type:** String (Hex)  
**Length:** depends on the Falcon key type; store the complete hex output from the key generator.
**Description:** Falcon private key for signing authentication.

**Security:**
- 🔴 **NEVER share your private key**
- 🔴 **NEVER commit to version control**
- 🔴 **Store in secure location**
- ✅ Use file permissions (chmod 600 on Linux)

**File Permissions:**
```bash
chmod 600 miner.conf
```

**Example:**
```toml
[falcon]
privkey = "fedcba9876543210..."  # 2305 or 4609 bytes hex
```

---

### `tritium_genesis` / `[mining] genesis`
**Type:** String (Hex)  
**Length:** 64 characters (32 bytes)  
**Description:** Genesis hash of your Nexus Tritium account.

**How to get:**
```bash
# In Nexus wallet console:
system/get/info

# Copy the "genesis" field value
```

**Purpose:**
- Links mining rewards to your Tritium account
- Required for solo mining
- Enables stateless mining protocol
- Used for ChaCha20 session key derivation

**Example:**
```toml
[mining]
genesis = "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
```

**See:** [docs/current/authentication/genesis-first-protocol.md](../current/authentication/genesis-first-protocol.md)

---

## Mining Configuration

### `channel` / `[mining] channel`
**Type:** Integer  
**Values:** `1` (Prime) or `2` (Hash)  
**Description:** Mining channel selection.

**Channel Types:**
- **1 (Prime Channel):**
  - CPU mining
  - Proof-of-Work: Prime number discovery
  - Best for: Multi-core CPUs, servers
  
- **2 (Hash Channel):**
  - GPU/FPGA mining
  - Proof-of-Work: SK1024 hashing
  - Best for: GPUs, FPGAs

**Examples:**
```toml
# Prime mining (CPU)
[mining]
channel = 1

# Hash mining (GPU/FPGA)
[mining]
channel = 2
```

---

### `reward_address` / `[mining] reward_address`
**Type:** String  
**Description:** 64-character Tritium genesis hex hash to receive mining rewards.

**How to get:**
```bash
# In Nexus wallet or via API:
system/get/info

# Copy the "genesis" field (64-character hex string)
# This is the same value as the `genesis` field in your miner config.
```

**Example:**
```toml
[mining]
reward_address = "a1000000000000000000000000000000000000000000000000000000000000ab"
# 64-char hex genesis hash from system/get/info (same as genesis above; NOT a Base58 account address)
```

**Notes:**
- Must be a valid Tritium Genesis Hash (leading byte `0xa1` on mainnet)
- **NOT** a Base58 NXS account address — register addresses will fail `Coinbase::Verify`
- Typically the same value as the `genesis` field unless routing rewards to a separate sigchain
- Advanced: supply a different genesis hash to route rewards to a different Tritium sigchain

---

### `mining_mode` (JSON only)
**Type:** String  
**Values:** `"PRIME"` or `"HASH"`  
**Description:** Mining channel selection (JSON format only).

**Example:**
```json
{
    "mining_mode": "PRIME"
}
```

**Note:** In TOML format, use `[mining] channel = 1` or `channel = 2` instead.

---

## Worker Configuration

### `count` / `[workers] count`
**Type:** Integer  
**Default:** 1  
**Valid Range:** 1-256  
**Description:** Number of mining workers to create.

**Recommendations:**

**CPU (Prime Channel):**
```toml
# Use number of physical cores
[workers]
count = 8  # For 8-core CPU

# High-core-count systems
[workers]
count = 96  # For Threadripper 128-core (leave some for OS)
```

**GPU (Hash Channel):**
```toml
# One worker per GPU; device indices are consecutive from [gpu] device.
[workers]
count = 4
hardware = "gpu"
[gpu]
device = 0  # Uses devices 0, 1, 2, and 3
```

**Hybrid (CPU + GPU):**
```json
{
    "workers": [
        {
            "worker": {
                "id": "cpu0",
                "mode": {"hardware": "cpu"}
            }
        },
        {
            "worker": {
                "id": "gpu0",
                "mode": {"hardware": "gpu", "device": 0}
            }
        }
    ]
}
```

**Example Files:**
- `docs/reference/config-examples/threadripper_128core_prime.conf` - 96-worker CPU config
- `docs/reference/config-examples/hybrid_cpu_gpu.conf` - Hybrid config
- `docs/reference/config-examples/multi_gpu_hash.conf` - Multi-GPU config

---

### `[cpu] threads`
**Type:** Integer  
**Default:** 1  
**Description:** Number of threads per CPU worker.

**Recommendations:**
- **Single-threaded (1):** Best for most CPUs
- **Multi-threaded (2+):** Experimental, may reduce efficiency

**Example:**
```toml
[cpu]
threads = 1
```

---

### `[cpu] efficiency_cores`
**Type:** Boolean  
**Default:** `true`  
**Description:** Use efficiency (E) cores on hybrid CPUs (Intel 12th gen+).

**Recommendations:**
- **true:** Use all cores (P+E) - maximum hashrate
- **false:** Use only P-cores - better power efficiency

**Example:**
```toml
[cpu]
efficiency_cores = true
```

---

## Hardware-Specific Settings

### GPU Power Controls

**Not enforced:** the current workers do not apply the GPU power, clock, fan,
or target-hashrate fields below. The `.config` GPU parser accepts only `device`.

#### `power_limit_percent`
**Type:** Integer  
**Valid Range:** 50-100  
**Default:** 100  
**Description:** GPU power limit as percentage of TDP.

**Example:**
```json
{
    "worker": {
        "mode": {
            "hardware": "gpu",
            "device": 0,
            "power_limit_percent": 80
        }
    }
}
```

---

#### `core_clock_offset`
**Type:** Integer  
**Valid Range:** -500 to +500 MHz  
**Default:** 0  
**Description:** GPU core clock offset in MHz.

**Examples:**
```json
"core_clock_offset": 100   // +100 MHz overclock
"core_clock_offset": -200  // -200 MHz underclock
```

---

#### `memory_clock_offset`
**Type:** Integer  
**Valid Range:** -1000 to +1000 MHz  
**Default:** 0  
**Description:** GPU memory clock offset in MHz.

**Examples:**
```json
"memory_clock_offset": 200   // +200 MHz memory OC
"memory_clock_offset": -500  // -500 MHz memory UC
```

---

#### `fan_speed`
**Type:** Integer  
**Valid Range:** 0 (auto) or 1-100%  
**Default:** 0 (auto)  
**Description:** Manual fan speed control.

**Examples:**
```json
"fan_speed": 0   // Auto (default)
"fan_speed": 70  // Fixed 70%
```

---

#### `target_hashrate`
**Type:** Integer  
**Default:** 0 (unlimited)  
**Description:** Target hashrate limit in H/s (0 = maximum).

---

### CPU Power Controls

#### `priority`
**Type:** Integer  
**Valid Range:** 0-4  
**Default:** 2  
**Description:** Thread priority level.

**Levels:**
- 0: Lowest
- 1: Below normal
- 2: Normal (default)
- 3: Above normal
- 4: Highest

**Example:**
```json
{
    "worker": {
        "mode": {
            "hardware": "cpu",
            "priority": 3
        }
    }
}
```

---

#### `power_limit_percent` (CPU)
**Type:** Integer  
**Valid Range:** 50-100  
**Default:** 100  
**Description:** Parsed but not enforced by the current CPU workers.

---

#### `hyperthreading`
**Type:** Boolean  
**Default:** `true`  
**Description:** Enable/disable SMT (Simultaneous Multithreading).

**Example:**
```json
{
    "worker": {
        "mode": {
            "hardware": "cpu",
            "hyperthreading": true
        }
    }
}
```

---

## Network Settings

### `local_ip` / `[network] local_ip`
**Type:** String  
**Default:** `"0.0.0.0"` (bind to all interfaces)  
**Description:** Local IP address to bind the miner's outbound socket to.

**Values:**
- `"127.0.0.1"`: Bind to localhost only — **use this for same-machine node**
- `"0.0.0.0"`: Bind to all interfaces — **use this for remote or VPN nodes** (default)
- `"auto"`: Bind to `127.0.0.1` for localhost nodes, otherwise bind to
  `0.0.0.0` and let the OS choose the source address; no internet DNS lookup
  is needed for this choice.

> **Note:** Local starter templates explicitly use `127.0.0.1`. For a remote
> node, use `0.0.0.0` or `auto` rather than binding to localhost.

**Example (localhost mining):**
```toml
[network]
local_ip = "127.0.0.1"   # Same machine as node
```

**Example (remote/VPN mining):**
```toml
[network]
local_ip = "0.0.0.0"     # Bind to all interfaces
```

---

### `keepalive_interval` / `[network] keepalive_interval`
**Type:** Integer  
**Default:** 24 hours  
**Valid Range:** 1-168 hours  
**Description:** Session keepalive ping interval in hours. Controls how often SESSION_KEEPALIVE packets are sent to prevent session cache eviction on the node.

**Behavior:**
- A 10-second early keepalive is always sent immediately after session establishment (not configurable)
- After the early keepalive, regular keepalives are sent at the configured interval
- The interval can be overridden dynamically by the node's SESSION_START packet (node sends its preferred timeout)

**Recommendations:**
- **1 hour**: Recommended for production miners (ensures session stays active)
- **12 hours**: Intermittent or part-time mining
- **24 hours**: Always-on dedicated miners with stable connections (default)
- **168 hours**: Maximum (7 days) for highly stable connections only

**Example:**
```toml
[network]
keepalive_interval = 1
```

**See:** [docs/current/authentication/falcon-handshake-cache.md](../current/authentication/falcon-handshake-cache.md)

---

### TLS/SSL Settings

**Not active for mining connections:** these fields are parsed but are not
used by the current mining connection. Use a secure tunnel for remote mining;
setting `enable_tls` does not encrypt the connection.

#### `enable_tls`
**Type:** Boolean  
**Default:** `false`
**Description:** Reserved setting; currently does not enable TLS.

**Example:**
```json
{
    "enable_tls": true,
    "tls_verify_peer": true
}
```

**Historical design:** [docs/current/security/tls-https.md](../current/security/tls-https.md)

---

#### `tls_ca_cert_path`
**Type:** String  
**Default:** `""` (use system CA certificates)  
**Description:** Path to CA certificate bundle file.

---

#### `tls_verify_peer`
**Type:** Boolean  
**Default:** `true`  
**Description:** Verify peer TLS certificate.

⚠️ Only disable for testing! Production should always verify.

---

#### `tls_server_name`
**Type:** String  
**Default:** `""` (use wallet_ip)  
**Description:** Server Name Indication for TLS handshake.

---

### Mutual TLS (Client Certificates)

#### `tls_client_cert_path`
**Type:** String  
**Description:** Path to client certificate file (PEM format).

**Example:**
```json
{
    "tls_client_cert_path": "/path/to/client-cert.pem",
    "tls_client_key_path": "/path/to/client-key.pem"
}
```

**See:** [docs/current/security/mutual-tls.md](../current/security/mutual-tls.md)

---

#### `tls_client_key_path`
**Type:** String  
**Description:** Path to client private key file (PEM format).

---

#### `tls_client_key_password`
**Type:** String  
**Description:** Password for encrypted client key (optional).

---

## Logging

### `log_level` / `[logging] level`
**Type:** Integer  
**Valid Range:** 0-3  
**Default:** 2  
**Description:** Logging verbosity level.

**Levels:**
- **0:** Trace (extremely verbose)
- **1:** Debug (detailed debug information)
- **2:** Info (default, recommended)
- **3:** Warn (warnings only)
- **4:** Error (errors only)
- **5:** Critical (critical errors only)

**Example:**
```toml
[logging]
level = 2
```

**Note:** Higher numbers = less verbose. Level 2 (Info) is recommended for production.

---

## Advanced Options

### Stateless Mining Protocol

**Push Readiness:**
- Miner selects framing from the configured port
- Sends `MINER_READY` after authentication (`0xD8` on 8323, `0xD0D8` on 9323)
- Sends/receives `GET_BLOCK` for template delivery using the selected lane framing
- Wrong-lane packets disconnect instead of falling back

**Configuration:**
- No manual configuration needed
- Port `8323` uses legacy 8-bit framing with push behavior
- Port `9323` uses stateless 16-bit mirror framing with the same push behavior

**Verification:**
```
[Solo Protocol] LEGACY/STATELESS LANE: Using push protocol
[Solo Push] Sending MINER_READY
[Solo] ✨ GET_BLOCK template received!
```

**See:** [docs/current/mining-protocols/stateless-mining.md](../current/mining-protocols/stateless-mining.md)

---

### Physical Falcon Signatures

#### `enable_physical_falcon` / `[falcon] enable_block_signing`
**Type:** Boolean  
**Default:** `false`  
**Description:** Enable physical block signatures (stored on blockchain).

**Impact:**
- Adds ~809 bytes per block submission (Falcon-512)
- Adds ~1577 bytes per block submission (Falcon-1024)
- Provides emergency backup authentication
- OFF by default for zero blockchain overhead

**Example:**
```toml
[falcon]
enable_block_signing = false
```

**See:** [docs/current/authentication/unified-falcon-protocol.md](../current/authentication/unified-falcon-protocol.md)

---

### ChaCha20 Encryption

**Status:** ALWAYS ON (core protocol security)

ChaCha20-Poly1305 encryption is always enabled for:
- Session ID protection
- Falcon public key wrapping (for remote mining)
- Genesis-derived key agreement

**See:**
- [docs/current/security/chacha20-encryption.md](../current/security/chacha20-encryption.md)
- [docs/current/authentication/genesis-first-protocol.md](../current/authentication/genesis-first-protocol.md)

---

## Security Best Practices

### Private Key Protection

**DO:**
- ✅ Set file permissions to 600 (`chmod 600 miner.conf`)
- ✅ Store config files in secure location
- ✅ Use encrypted filesystem if possible
- ✅ Back up keys securely (offline storage)

**DON'T:**
- 🔴 Never commit private keys to version control
- 🔴 Never share private keys
- 🔴 Never send keys over unencrypted channels
- 🔴 Never store in publicly accessible locations

---

### Remote Mining

Use a secure tunnel between the miner and node. The TLS configuration fields
are not connected to the mining socket and do not encrypt it.

**Falcon Authentication:**
- Always use Falcon-1024 for remote mining
- Verify node authenticity
- Use ChaCha20 encryption for public key transmission

---

### File Permissions (Linux/macOS)

```bash
# Secure your config file
chmod 600 miner.conf

# Verify permissions
ls -l miner.conf
# Should show: -rw------- (owner read/write only)
```

---

## Example Configurations

See the [mining setup guide](../../configs/README.md) for the recommended
CPU Prime and NVIDIA GPU Hash examples. Retained JSON examples are under
`docs/reference/config-examples/`.

### JSON Examples
- `MASTER_REFERENCE.conf` - Complete JSON documentation
- `basic-example.conf` - Simple JSON template
- `hybrid_cpu_gpu.conf` - Combined CPU+GPU
- `multi_gpu_hash.conf` - Multi-GPU with power controls
- `threadripper_128core_prime.conf` - High-core-count CPU
- `remote_tls_mining.conf` - Historical TLS example (not active in mining connections)

---

## Migration Guides

For upgrading from older versions:
- [docs/upgrade-guides/legacy-to-stateless.md](../upgrade-guides/legacy-to-stateless.md) - Protocol migration
- [docs/upgrade-guides/legacy-features-removed.md](../upgrade-guides/legacy-features-removed.md) - Deprecated features

---

## Protocol Reference

For opcode and packet format details:
- [docs/reference/opcodes-reference.md](opcodes-reference.md) - LLP protocol opcodes

---

## Troubleshooting

For common issues and solutions:
- [docs/current/troubleshooting/enhanced-diagnostics.md](../current/troubleshooting/enhanced-diagnostics.md)
- [docs/current/troubleshooting/dynamic-port-detection.md](../current/troubleshooting/dynamic-port-detection.md)
- [docs/current/troubleshooting/cross-validation-recovery.md](../current/troubleshooting/cross-validation-recovery.md)

---

**Last Updated:** January 2026  
**Version:** 1.5+  
**Protocol:** Stateless mining (LLL-TAO PR #170)

---

## Quick Start Checklist

Follow [mining setup](../../configs/README.md) for the current steps and
matching binary and config. Do not use the JSON-only `-c` check for `.config`
files; a successful check also starts mining.
