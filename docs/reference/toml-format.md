# TOML Configuration Format (.config)

## Overview

NexusMiner now supports TOML-style `.config` files as a simpler, more readable alternative to JSON `.conf` files. The TOML format is easier to read and edit, with support for comments and a cleaner syntax.

## File Format Detection

- Files ending in `.config` are parsed as TOML
- Other filenames, including `.conf` and `.toml`, are parsed as JSON
- All existing `.conf` files continue to work without modification

## TOML Configuration Format

### Basic Structure

TOML uses sections (in brackets) and key-value pairs:

```toml
[section_name]
key = "value"
number = 123
boolean = true
```

### Supported Sections

#### [wallet]
Connection settings for your Nexus node:
```toml
[wallet]
ip = "127.0.0.1"                # Node IP address
port = 8323                     # Mining port (default: 8323)
failover_wallet_ip = ""         # (optional) Failover node IP — leave empty to disable
failover_port = 0               # (optional) Failover port — 0 = same lane/port as primary
failover_max_retries = 5        # Switch to failover after this many consecutive primary failures
```

##### Failover: cycling behaviour

When the primary node fails `failover_max_retries` consecutive times the miner automatically
switches to the failover node.  If the failover node also fails `failover_max_retries` times,
the miner cycles back to the primary.  This alternation continues until one of the nodes answers.

##### Failover: Colin visibility

Colin reports the current failover state in every periodic diagnostic report.  Example output:

**Primary active (healthy):**
```
[Colin]  FailoverNode │ ✅ PRIMARY active: 192.168.1.10:9323  │  standby: 192.168.1.11:9323  │  fails: 0/5
```

**Failover active (primary down):**
```
[Colin]  FailoverNode │ ⚠️  FAILOVER ACTIVE: 192.168.1.11:9323  │  primary DOWN: 192.168.1.10:9323  │  active for: 142s
[Colin]  ── Warnings ──────────────────────────────────────────
[Colin]    • FAILOVER NODE ACTIVE for 142s — primary node 192.168.1.10:9323 appears down
```

After 300 s (5 min) on failover an extended-failover warning and a remediation recommendation are added.
See [docs/current/failover.md](../current/failover.md) for the full tuning guide.

#### [network]
Network configuration:
```toml
[network]
local_ip = "0.0.0.0"          # Local bind address (0.0.0.0 = all interfaces)
keepalive_interval = 24        # Keepalive interval in hours (1-168)
```

#### [mining]
Mining configuration:
```toml
[mining]
channel = 1                    # 1 = Prime (CPU), 2 = Hash (GPU/FPGA)
genesis = "abc...xyz"          # 64-character genesis hash (from wallet)
reward_address = "a1...64hex" # 64-char genesis hash (from system/get/info; NOT a Base58 address)
```

#### [workers]
Worker configuration:
```toml
[workers]
count = 8                      # Number of workers to create
```

#### [cpu]
CPU mining settings (for Prime channel):
```toml
[cpu]
threads = 1                    # Threads per worker (default: 1)
efficiency_cores = true        # Use E-cores on hybrid CPUs (default: true)
priority = 2                   # Thread priority: 0=low, 1=below_normal, 2=normal, 3=above_normal, 4=high
hyperthreading = true          # Use hyperthreading/SMT (default: true)
```

`power_limit_percent` and `target_hashrate` are parsed but not enforced by
the current workers.

#### [gpu]
GPU mining settings (for Hash channel):
```toml
[gpu]
device = 0                     # GPU device index (default: 0)
```

#### [stats]
Statistics output configuration:
```toml
[stats]
mode = "console"               # File stats mode is not implemented for TOML
```

#### [logging]
Logging configuration:
```toml
[logging]
level = 2                      # 0=trace, 1=debug, 2=info, 3=warning
```

## Getting Started

Follow the [mining setup guide](../../configs/README.md) to choose a real
template, configure the genesis and Falcon keys, and start the miner.

## Features and Benefits

### TOML Advantages
- **Readable**: Clear section headers and key-value pairs
- **Comments**: Use `#` for documentation
- **Simple**: No need for quotes on numbers, no trailing commas
- **Type-safe**: Automatic type detection (strings, numbers, booleans)

### Backward Compatibility
- All existing `.conf` (JSON) files work unchanged
- No migration required
- Use whichever format you prefer

### Error Handling
- Invalid lines are warned but don't stop parsing
- Provides line numbers for errors
- Validates data ranges (e.g., keepalive_interval clamped to 1-168 hours)

## Advanced Options

### Remote Mining
```toml
[wallet]
ip = "192.168.1.100"      # Remote node IP
port = 8323

[network]
local_ip = "0.0.0.0"      # Bind to all interfaces
                          # For localhost: use 127.0.0.1
                          # Or use "auto" to select the local bind address
```

The current mining connection does not use the configured TLS settings.
Use a secure tunnel for remote mining.

### Multi-threaded Workers
```toml
[workers]
count = 4                 # 4 workers

[cpu]
threads = 2               # 2 threads each = 8 total threads
```

### Debug Logging
```toml
[logging]
level = 1                 # Debug verbosity (0 = trace)
```

## Comparison: JSON vs TOML

### JSON (.conf)
```json
{
    "version": 1,
    "wallet_ip": "127.0.0.1",
    "port": 8323,
    "mining_mode": "PRIME",
    "workers": [
        {"worker": {"id": "cpu0", "mode": {"hardware": "cpu", "threads": 1}}},
        {"worker": {"id": "cpu1", "mode": {"hardware": "cpu", "threads": 1}}}
    ]
}
```

### TOML (.config)
```toml
[wallet]
ip = "127.0.0.1"
port = 8323

[mining]
channel = 1

[workers]
count = 2

[cpu]
threads = 1
```

The TOML format is much simpler and easier to understand!

## Limitations

The TOML parser currently supports a subset of TOML features focused on mining configuration:
- Sections with `[section_name]`
- Key-value pairs with `key = value`
- Strings (quoted), integers, and booleans
- Comments with `#`

Common mining configurations support CPU workers, GPU workers, and console
stats output. The GPU section currently parses only `device`.

**GPU Configuration**: `[workers] count` creates GPU workers on consecutive
indices starting at `[gpu] device`. For individual nonconsecutive indices or
mixed CPU/GPU workers in one instance, use JSON `.conf` files.

## Reference Files

- [CPU Prime](../../configs/cpu-prime-solo.config) and
  [GPU Hash](../../configs/gpu-hash-solo.config) starter configs
- [MASTER-REFERENCE.config](../../configs/MASTER-REFERENCE.config) - options reference

## Support

For questions or issues:
1. Check the [mining setup guide](../../configs/README.md) and options reference
2. See `example_configs/` for more examples
3. Refer to the README.md for general miner documentation
