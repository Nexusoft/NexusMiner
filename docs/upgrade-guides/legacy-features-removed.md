# Legacy Features Removed - Migration Guide

## Overview

This document explains what legacy features were removed from NexusMiner and provides migration instructions.

---

## 1. Pool Mining Removed

### Why?

**The Nexus Node IS the Pool** - Pool mining has been superseded by the node's built-in pool functionality. The node itself now provides pool-like features through its native mining interface.

### What Was Removed?

- Pool protocol implementation (`src/protocol/pool.cpp`, `src/protocol/pool.hpp`)
- Pool configuration structures (`src/config/pool.hpp`)
- Pool-specific stats printers
- Pool connection logic in worker manager
- Pool example configs (`simple_prime_pool.conf`)

### Migration Path

**For Pool Miners:**
- Connect directly to Nexus nodes that provide pool services
- Use SOLO mining mode with Falcon authentication
- The node handles reward distribution and pooling internally

**Configuration Changes:**
```json
// OLD (Pool Mode)
{
    "wallet_ip": "primepool.nexus.io",
    "port": 50000,
    "pool": {
        "username": "<NXS_ADDRESS>",
        "display_name": "MyMiner"
    }
}

// NEW (Solo Mode - Node IS the Pool)
{
    "wallet_ip": "127.0.0.1",  // or node IP
    "port": 8323,
    "miner_falcon_pubkey": "<YOUR_PUBKEY>",
    "miner_falcon_privkey": "<YOUR_PRIVKEY>",
    "tritium_genesis": "<YOUR_GENESIS>"
}
```

### Pool-Specific URLs Removed

These pool services are no longer referenced in NexusMiner:
- `primepool.nexus.io`
- `hashpool.nexus.io`

Contact the Nexus community for current pool/node mining options.

---

## 2. Simplified Config System Removed

### Why?

The simplified `.config` format (version 2.0) was a separate system that caused confusion:
- Users didn't know which format to use (`.conf` vs `.config`)
- Options like `power_limit_percent` in simplified configs were ignored when using legacy configs
- Maintaining two parallel systems added complexity
- Power controls have now been **unified** into the legacy `.conf` format

### What Was Removed?

- Simplified config parser (`src/config/simplified_config.cpp/hpp`)
- Simplified config documentation (`docs/simplified_config_files.md`)
- Beginner template configs (`beginner_hash_gpu.config`)
- CLI options: `--create-config`, `--import-config`, `--export-config`

### Migration Path

**All users now use ONE config format: `.conf` files with `version: 1`**

**Power controls that were ONLY in simplified configs are NOW available in legacy configs:**

```json
// GPU Power Controls (NOW AVAILABLE)
{
    "workers": [
        {
            "worker": {
                "id": "gpu0",
                "mode": {
                    "hardware": "gpu",
                    "device": 0,
                    "power_limit_percent": 80,        // NEW!
                    "core_clock_offset": 100,         // NEW!
                    "memory_clock_offset": 200,       // NEW!
                    "fan_speed": 70,                  // NEW!
                    "target_hashrate": 0              // NEW!
                }
            }
        }
    ]
}

// CPU Power Controls (NOW AVAILABLE)
{
    "workers": [
        {
            "worker": {
                "id": "cpu0",
                "mode": {
                    "hardware": "cpu",
                    "threads": 1,
                    "priority": 3,                    // NEW!
                    "power_limit_percent": 90,        // NEW!
                    "hyperthreading": true,           // NEW!
                    "efficiency_cores": true,         // NEW!
                    "target_hashrate": 0              // NEW!
                }
            }
        }
    ]
}
```

### File Conversion

**If you had a simplified `.config` file:**

1. Rename it to `.conf`
2. Change `"config_version": "2.0"` to `"version": 1`
3. Update structure to match examples in `example_configs/`
4. Remove simplified-only fields like `"preset"` and `"power_profile"`
5. Map simplified worker configs to legacy structure

**Example:**
```json
// OLD .config format
{
    "config_version": "2.0",
    "workers": [{
        "id": "gpu0",
        "hardware": "gpu",
        "gpu": {"device": 0, "power_limit_percent": 80}
    }]
}

// NEW .conf format
{
    "version": 1,
    "workers": [{
        "worker": {
            "id": "gpu0",
            "mode": {
                "hardware": "gpu",
                "device": 0,
                "power_limit_percent": 80
            }
        }
    }]
}
```

---

## 3. Default `local_ip` Changed

### What Changed?

- **OLD DEFAULT**: `"local_ip": "127.0.0.1"` (localhost only)
- **NEW DEFAULT**: `"local_ip": "0.0.0.0"` (all interfaces)

### Why?

The new default works for ALL scenarios:
- ✅ Localhost mining (`127.0.0.1`)
- ✅ VPN mining
- ✅ Remote mining
- ✅ Multi-interface systems

### Migration

**No action required** - The new default is more flexible. If you explicitly set `local_ip` in your config, it will still be respected.

To revert to old behavior (localhost only):
```json
{
    "local_ip": "127.0.0.1"
}
```

---

## New Features Available

### 1. GPU Power Controls

Control GPU power consumption and performance:
- `power_limit_percent`: 50-100% (reduce power consumption)
- `core_clock_offset`: MHz offset for core clock
- `memory_clock_offset`: MHz offset for memory clock
- `fan_speed`: 0=auto, 1-100=fixed percentage
- `target_hashrate`: Target hashrate limit (0=max)

### 2. CPU Power Controls

Optimize CPU mining:
- `priority`: Thread priority level (0=low, 2=normal, 4=high)
- `power_limit_percent`: 50-100% CPU power limit
- `hyperthreading`: Enable/disable SMT cores
- `efficiency_cores`: Use E-cores on hybrid CPUs
- `target_hashrate`: Target hashrate limit (0=max)

### 3. High-Core-Count Support

See `example_configs/threadripper_128core_prime.conf` for 96+ worker configurations.

### 4. Unified Example Configs

New comprehensive example configs:
- `MASTER_simple.conf` - Quick start template
- `MASTER_reference.conf` - Complete option reference
- `threadripper_128core_prime.conf` - High-core CPU mining
- `multi_gpu_hash.conf` - Multi-GPU with power controls
- `hybrid_cpu_gpu.conf` - Combined CPU+GPU mining
- `remote_tls_mining.conf` - Secure remote mining

---

## Getting Help

If you have questions about migration:

1. Check example configs in `example_configs/`
2. See `MASTER_reference.conf` for ALL available options
3. Join [Nexus Miners Telegram](https://t.me/NexusMiners)
4. Review README.md for updated documentation

---

## Summary

✅ **ONE config system** - No more confusion between `.conf` and `.config`  
✅ **ALL power controls** - GPU and CPU power management unified  
✅ **Solo mining only** - Node IS the pool  
✅ **Better defaults** - `local_ip` works everywhere  
✅ **More examples** - Comprehensive config templates  

**Result**: Simpler, more powerful, unified configuration system!
