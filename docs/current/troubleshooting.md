# NexusMiner Troubleshooting Guide

Common issues and solutions for NexusMiner configuration, mining, and connectivity.

## Table of Contents

1. [Connection Issues](#connection-issues)
2. [Authentication Problems](#authentication-problems)
3. [Mining Issues](#mining-issues)
4. [Performance Problems](#performance-problems)
5. [Configuration Errors](#configuration-errors)
6. [Hardware Issues](#hardware-issues)
7. [Advanced Diagnostics](#advanced-diagnostics)

---

## Connection Issues

### "Connection refused" or "Cannot connect to node"

**Symptoms:**
```
[Network] Failed to connect to 127.0.0.1:8323
[Network] Connection refused
```

**Solutions:**

1. **Check node is running:**
   ```bash
   ps aux | grep nexus
   # or
   systemctl status nexus
   ```

2. **Verify node mining is enabled:**
   ```ini
   # In nexus.conf:
   mining=1
   miningport=8323
   ```

3. **Check firewall rules:**
   ```bash
   # Linux
   sudo ufw status
   sudo ufw allow 8323/tcp
   
   # Check if port is listening
   netstat -tuln | grep 8323
   ```

4. **Verify correct IP/port in miner.conf:**
   ```toml
   [wallet]
   ip = "127.0.0.1"  # Use correct node IP
   port = 8323       # Use correct port
   ```

---

### "Connection timeout"

**Symptoms:**
```
[Network] Connection timeout after 30 seconds
```

**Solutions:**

1. **Check network connectivity:**
   ```bash
   ping <node_ip>
   telnet <node_ip> 8323
   ```

2. **Increase timeout (if high latency):**
   ```json
   {
       "connection_timeout": 60
   }
   ```

3. **Check for network restrictions:**
   - VPN/proxy interfering
   - ISP blocking ports
   - Router NAT configuration

---

### "Wrong lane or stale protocol logs"

**Symptoms:**
```
[NodeSession:PRIMARY] Cross-lane packet prefix ... closing instead of byte-drop resync
```

This means the node and miner disagree on the port-selected framing. The miner no
longer falls back between lanes.

**To fix:**
1. Verify the node mining port.
2. Use `8323` for legacy 8-bit framing or `9323` for stateless 16-bit mirror framing.
3. Restart miner and node after changing the port.

**See:** [docs/upgrade-guides/legacy-to-stateless.md](../upgrade-guides/legacy-to-stateless.md)

---

## Authentication Problems

### "Authentication failed"

**Symptoms:**
```
[Falcon Auth] ❌ Authentication failed
[Falcon Auth]    Reason: Invalid signature
```

**Solutions:**

1. **Verify Falcon keys are correct:**
   ```bash
   # Generate new keys
   ./NexusMiner --create-keys
   
   # Copy output to miner.conf
   ```

2. **Check genesis hash:**
   ```bash
   # In Nexus wallet console:
   system/get/info
   # Copy "genesis" field (32 bytes hex = 64 chars)
   ```

3. **Ensure keys match:**
   - Public key and private key must be from same keypair
   - Don't mix Falcon-512 and Falcon-1024 keys

4. **Check node whitelist (if enabled):**
   ```ini
   # In nexus.conf:
   # minerallowkey=<your_falcon_pubkey_hex>
   ```

---

### "No genesis - ChaCha20 encryption unavailable"

**Symptoms:**
```
[Falcon Auth] ⚠️  No genesis - ChaCha20 encryption unavailable
```

**Solution:**

Add genesis hash to miner.conf:

```toml
[mining]
genesis = "your_32_byte_genesis_hash_hex"
```

Get genesis from wallet:
```bash
# In Nexus wallet:
system/get/info
# Copy "genesis" field
```

**See:** [docs/current/authentication/genesis-first-protocol.md](../current/authentication/genesis-first-protocol.md)

---

### "Invalid Falcon key length"

**Symptoms:**
```
[Config] Error: Falcon public key must be 897 or 1793 bytes (hex)
```

**Solutions:**

1. **Check key format:**
   - Falcon-512: 897 bytes = 1794 hex characters
   - Falcon-1024: 1793 bytes = 3586 hex characters
   - Must be hex encoded (0-9, a-f)

2. **Regenerate keys:**
   ```bash
   ./NexusMiner --create-keys
   ```

3. **Verify no extra spaces/newlines:**
   ```bash
   # Remove whitespace
   echo -n "YOUR_KEY_HERE" | wc -c
   ```

---

## Mining Issues

### GET_BLOCK returns TEMPLATE_NOT_READY

Authentication and channel setup can succeed even when the node is not ready
to issue mining templates. `BLOCK_REJECTED` with an eight-byte control payload
is a GET_BLOCK response, not a rejected block submission. Older miners may
report it as a malformed `d0 c9 00 00 00 08 01 04` packet and disconnect.
Update the miner to handle the response and retry without closing the session.

On the node, inspect `system/get/info.mining_readiness`, especially `ready`,
`reason`, `fresh_peers`, and `required_peers`. Restore synchronization and
healthy peer connections as indicated by the reason. `TEMPLATE_NOT_READY` can
also indicate session/channel setup or a template-building problem; check the
node's GET_BLOCK and template logs if readiness is already true. Public-network mining
requires two distinct supporting peers by default. `-miningminpeers=1` is an
explicit reduction in safety evidence for an intentional single-peer setup,
not a way to make an unsynchronized node ready. Do not bypass readiness merely
to suppress the message.

### "No work received"

**Symptoms:**
```
[Mining] Waiting for work...
[Mining] No work received after 60 seconds
```

**Solutions:**

1. **Check node is synced:**
   ```bash
   # In Nexus wallet:
   system/get/info
   # Verify "blocks" matches network height
   ```

2. **Verify mining channel:**
   ```toml
   [mining]
   channel = 1  # 1=Prime, 2=Hash
   ```

3. **Check node has mining enabled:**
   ```ini
   # In nexus.conf:
   mining=1
   ```

4. **Enable debug logging:**
   ```toml
   [logging]
   level = 1  # Debug level
   ```

---

### "All blocks rejected - stale"

**Symptoms:**
```
[Mining] Block found!
[Submit] ❌ Block rejected: stale
```

**Solutions:**

1. **Check system time:**
   ```bash
   # Sync system clock
   sudo ntpdate pool.ntp.org
   # or
   sudo systemctl start systemd-timesyncd
   ```

2. **Reduce stale work:**
   - Update to stateless protocol (instant updates)
   - Reduce polling interval (if using legacy)
   - Improve network latency to node

3. **Check node connectivity:**
   - High latency to node
   - Packet loss
   - Node overloaded

---

### "KEEPALIVE ACK stale but push notifications arriving"

**Symptoms:**
```
[Worker_manager] KEEPALIVE ACK STALE (420s) but push notifications received 12s ago — TCP session alive, retrying template request (node-side keepalive responder may be malfunctioning)
```

**Explanation:**

NexusMiner uses a **two-signal liveness model** to avoid false "session dead" detection:

| Signal | What it proves | When it can go silent |
|--------|---------------|-----------------------|
| `KEEPALIVE_V2_ACK` | Node's keepalive responder is answering | Node-side bug, rate limiting, or responder hang |
| Push notifications (`PRIME/HASH_BLOCK_AVAILABLE`) | TCP session is alive & authenticated | Only when the TCP connection is actually dead |

If push notifications are arriving on a connection, the TCP session is demonstrably alive —
the miner will **not** tear it down just because `KEEPALIVE_V2_ACK` responses have gone silent.
Instead, it retries the template request (`retry_template_request`) and logs a warning.

A full TCP reconnect (`retry_connect`) is only triggered when **both** signals are stale:
- No `KEEPALIVE_V2_ACK` for > 360 seconds, **AND**
- No push notifications received for > 120 seconds

**What to investigate:**

1. **Node-side keepalive responder:** The node may have a bug or rate-limiting issue causing
   it to stop sending `KEEPALIVE_V2_ACK` responses while still serving block push notifications.
   Check node logs for keepalive-related errors.

2. **No further action if mining continues:** If the miner continues to receive push
   notifications and template updates, mining will proceed normally. The stale ACK warning
   is informational — the connection is healthy.

3. **If both signals go stale:** If push notifications also stop arriving (> 120s), a full
   reconnect will be triggered automatically.

---

### "Template age exceeds emergency timeout"

**Symptoms:**
```
[Worker_manager] ❌ EMERGENCY (Hash channel): template 305s old — no push received
[Worker_manager]    channel_height 0 / channel_target 0 (chain not yet advanced in tracker)
[Worker_manager]    Forcing hard recovery (discard + stop + retry)
```

**Explanation:**

The miner monitors template age to detect when the push-notification connection has gone dead.
There are two age thresholds:

- **`MAX_TEMPLATE_AGE_SECONDS` = 300 s** (`client_block.h`) — per-template validation gate in
  `ClientChannelManager::ValidateTemplate()`.  This is the first line of defence: it rejects
  a template that has not been refreshed by a push in 300 seconds.
- **`TEMPLATE_AGE_EMERGENCY_TIMEOUT_SECONDS` = 600 s** (`protocol_constants.hpp`) — the
  hard dead-connection detector.  If 600 seconds pass with no push the connection is
  declared dead and a forced recovery is triggered.

Both thresholds must be above any realistic block drought.  A 275 s drought with no block
on **any** channel (Prime, Hash, or Stake combined) has been observed in production.  The
300 s client-validation gate is set to be safely above this observed worst case.

In the push-driven protocol the node pushes a fresh template within ~2 s of every unified
chain tip advance.  Hash blocks advance the unified chain every ~18 s on average, but can
take up to 300+ s when no block arrives on any channel.  Prime blocks typically take 2–10
minutes, during which Hash blocks keep the template fresh.

If 300 seconds pass without any push the `ClientChannelManager` will mark the template
stale and request a new one via GET_BLOCK.  If 600 seconds pass the connection is declared
dead and a hard recovery is forced.  The miner will then:
1. Discard the stale template
2. Stop all workers
3. Re-request a fresh template (retry / re-subscribe)

A warning is logged when the template approaches the emergency threshold to give operators
an early signal.

**Logging distinguishes two sub-cases:**

- **Chain advanced** — `channel_height >= channel_target`: both a missed push *and* a
  chain advance were detected; clear emergency.
- **Chain unchanged** — tracker shows no advance: push was missed while the chain was
  (apparently) still at the same height.  For Prime this *could* be a genuinely long
  block (2–10 min is normal), but 300 s without any hash-block push still indicates a
  likely dead connection.

**Solutions:**

1. **Check push notifications:**
   - Verify node supports stateless protocol (LLL-TAO 5.1.0+)
   - Check node logs for push notification delivery
   - Push notifications are the primary update mechanism

2. **Check for actual node issues:**
   - Node may be disconnected or crashed
   - Network connectivity problems
   - Node overloaded or syncing

3. **Verify GET_ROUND polling (legacy fallback):**
   - If using legacy protocol, ensure polling is working
   - Check network connectivity to node
   - Review node logs for GET_ROUND requests

**When to investigate:**
- Repeated emergency triggers on either channel
- Emergency fires without subsequent recovery (no new template received)
- Consistent emergencies paired with low hash-rate or zero shares submitted

**See also:**
- [Channel Management](../mining-protocols/channel-management.md)
- [Connection Recovery](./cross-validation-recovery.md)
- [Template Age Policy](../diagrams/protocols/template-age-policy.md)

---

### "Low hash rate"

**Symptoms:**
- Hash rate lower than expected
- GPU/CPU not fully utilized

**Solutions:**

1. **Check worker count:**
   ```toml
   [workers]
   count = 8  # Adjust for your hardware
   ```

2. **Verify hardware settings:**
   
   **CPU:**
   ```toml
   [cpu]
   threads = 1
   efficiency_cores = true
   ```
   
   **GPU:**
   ```json
   {
       "worker": {
           "mode": {
               "power_limit_percent": 100,
               "core_clock_offset": 0
           }
       }
   }
   ```

3. **Check system resources:**
   ```bash
   # CPU usage
   top
   htop
   
   # GPU usage
   nvidia-smi  # Nvidia
   radeontop   # AMD
   ```

4. **Disable power saving:**
   - Set CPU governor to "performance"
   - Disable GPU power management
   - Check BIOS settings

---

## Stuck in Degraded Mode

**Symptoms:**
```
[Worker_manager] ⚠️  STOPPING ALL WORKERS (DEGRADED MODE)
[Worker_manager] Mining stopped - waiting for valid template
... (miner remains in degraded mode for 1-2+ hours without recovery)
```

**Explanation:**

Degraded mode is entered when the miner has no valid mining template or no
usable authoritative session (for example after a node restart, `SESSION_EXPIRED`,
or a network interruption). Recovery is now **authoritative-session-first**:
the miner distinguishes “session alive, waiting for template” from “session
must be restored,” and it only exits degraded state when both the template layer
and the authoritative session layer agree the miner is ready again.

The miner still uses a **3-stage escape ladder** with a hard time limit to
guarantee recovery within 300 seconds.

### Escape Ladder Stages

| Stage | Trigger | Action |
|-------|---------|--------|
| **Stage 1** | 0–60 s in degraded | Retry GET_BLOCK while the session is still authoritative, or wait for explicit session recovery to complete |
| **Stage 2** | authoritative session expired / dead but TCP still usable | Enter local `SESSION_RECOVERY`, perform in-band re-authentication, then request fresh work |
| **Stage 3** | transport dead, reconnect timeout, or both liveness signals stale long enough | Force full TCP reconnect / configured failover via `retry_connect()` |
| **Hard Limit** | > 300 s in degraded (any state) | **Unconditional** `retry_connect()` — no miner should be stuck longer than this |

### Two-Signal Liveness Model

Before escalating to a TCP reconnect, the miner checks two independent liveness signals:

1. **KEEPALIVE_V2_ACK** — keepalive response from the node; stale if silent > 360 s
2. **Push notifications** (`PRIME_BLOCK_AVAILABLE`, `BLOCK_DATA`) — updated by `HeightTracker::OnPushNotification()`; stale if silent > 300 s

**If push notifications are arriving but keepalive ACK is silent:** The TCP session is demonstrably alive. The miner logs a warning and retries GET_BLOCK instead of tearing down the connection. This prevents spurious reconnects caused by a node-side keepalive responder issue.

**If BOTH signals are stale for > 300 s:** The connection is presumed dead and `retry_connect()` is called.

### Session ID Mismatch Handling

When a `KEEPALIVE_V2_ACK` carries a session ID that doesn't match the miner's local session ID, the miner **does not immediately self-expire**. Instead:

1. A mismatch counter is incremented and a warning is logged.
2. After **3 consecutive mismatches** with no intervening successful ACK, the session is expired and re-authentication is triggered.
3. A successful (matching) ACK resets the mismatch counter to 0.

This prevents premature session expiry due to late/replayed ACKs or node-side race conditions during re-authentication.

### Diagnostic Log Messages

| Log Message | Meaning |
|-------------|---------|
| `Stage 1 (Xs in degraded): retrying recovery request` | Normal — waiting for GET_BLOCK response |
| `Stage 2 (Xs in degraded, both signals stale) — attempting in-band re-authentication` | Retrying login on existing TCP connection |
| `Stage 3 ESCALATION — forcing reconnect` | Both signals dead > 180 s; reconnecting |
| `DEGRADED MODE HARD LIMIT — forcing reconnect` | Exceeded 300 s hard limit; reconnecting unconditionally |
| `KEEPALIVE ACK STALE (Xs) but push received Xs ago — TCP session alive` | Push proving connection live; keepalive silent is a node-side issue |
| `Session ID mismatch #N` | Counting mismatches; will expire after 3 consecutive mismatches |

### Common Causes and Fixes

1. **Node restarted:** The miner detects SESSION_EXPIRED and re-authenticates. Should recover within Stage 1 (< 60 s). If not, check node is accepting connections.

2. **Network intermittent:** Push signals will be stale. The miner first tries
   authoritative session recovery if the TCP session is still usable, then
   reconnects / fails over when transport health is no longer good enough.

3. **Node keepalive responder silent:** If push notifications still arrive but keepalive ACKs are silent, this is a node-side issue. The miner will NOT reconnect while pushes are arriving — it retries GET_BLOCK on the live session. Update the node software.

4. **Miner stuck > 5 min:** Should never happen with the hard limit. If it does, check for crashes or deadlocks in the miner log.

---

## Performance Problems

### "High CPU usage"

**Symptoms:**
- Miner using 100% CPU unexpectedly
- System slow/unresponsive

**Solutions:**

1. **Adjust worker count:**
   ```toml
   [workers]
   count = 6  # Leave some cores for OS
   ```

2. **Lower thread priority:**
   ```json
   {
       "worker": {
           "mode": {
               "priority": 1  # Below normal (0-4)
           }
       }
   }
   ```

3. **Check for polling loop:**
   - Legacy protocol polls continuously
   - Update to stateless protocol for event-driven mining

---

### "High memory usage"

**Symptoms:**
- Miner using excessive RAM
- System swapping

**Solutions:**

1. **Reduce worker count:**
   ```toml
   [workers]
   count = 4  # Fewer workers = less memory
   ```

2. **Check for memory leaks:**
   ```bash
   # Monitor memory over time
   watch -n 1 "ps aux | grep NexusMiner"
   ```

3. **Update to latest version:**
   - Memory leaks fixed in recent versions
   - Check GitHub releases

---

### "Network traffic too high"

**Symptoms:**
- Excessive bandwidth usage
- ISP throttling

**Solutions:**

1. **Use stateless protocol:**
   - 95% reduction in network traffic
   - Update miner and node

2. **Reduce polling (if legacy):**
   ```json
   {
       "poll_interval": 5000  # 5 seconds (milliseconds)
   }
   ```

3. **Monitor traffic:**
   ```bash
   # Linux
   iftop
   nethogs
   ```

---

## Configuration Errors

### "Failed to parse config file"

**Symptoms:**
```
[Config] Error parsing miner.conf: Invalid TOML syntax
```

**Solutions:**

1. **Check TOML syntax:**
   ```toml
   # Correct
   [section]
   key = "value"
   
   # Wrong
   [section
   key = value  # Missing quotes
   ```

2. **Validate TOML:**
   ```bash
   # Online validator: https://www.toml-lint.com/
   ```

3. **Check JSON syntax:**
   ```json
   {
       "key": "value"  // No trailing comma on last item
   }
   ```

4. **Use config validation:**
   ```bash
   ./NexusMiner -c miner.conf
   ```

---

### "Missing required field"

**Symptoms:**
```
[Config] Error: Missing required field 'falcon.pubkey'
```

**Solutions:**

1. **Check all required fields:**
   ```toml
   [falcon]
   pubkey = "..."
   privkey = "..."
   
   [mining]
   genesis = "..."
   reward_address = "..."
   channel = 1
   ```

2. **Use example config:**
   ```bash
   cp docs/reference/config-examples/stateless-mining.conf miner.conf
   # Edit with your values
   ```

---

### "Config file not found"

**Symptoms:**
```
[Config] Error: Could not open miner.conf
```

**Solutions:**

1. **Specify full path:**
   ```bash
   ./NexusMiner /full/path/to/miner.conf
   ```

2. **Check file permissions:**
   ```bash
   ls -l miner.conf
   chmod 600 miner.conf
   ```

3. **Create default config:**
   ```bash
   ./NexusMiner --create-falcon-config
   ```

---

## Hardware Issues

### GPU Not Detected

**Symptoms:**
```
[GPU] No CUDA devices found
[GPU] No AMD devices found
```

**Solutions:**

1. **Check GPU drivers:**
   ```bash
   # Nvidia
   nvidia-smi
   
   # AMD
   rocm-smi
   ```

2. **Verify build options:**
   ```bash
   # Rebuild with GPU support
   cmake -DWITH_GPU_CUDA=On ..  # Nvidia
   cmake -DWITH_GPU_AMD=On ..   # AMD
   ```

3. **Check GPU visibility:**
   ```bash
   lspci | grep -i nvidia
   lspci | grep -i amd
   ```

---

### GPU Crashes / Instability

**Symptoms:**
- Miner crashes during mining
- GPU driver resets
- System freezes

**Solutions:**

1. **Reduce overclock:**
   ```json
   {
       "core_clock_offset": 0,      // Remove overclock
       "memory_clock_offset": 0
   }
   ```

2. **Lower power limit:**
   ```json
   {
       "power_limit_percent": 80    // Reduce from 100%
   }
   ```

3. **Check temperatures:**
   ```bash
   nvidia-smi
   # If temp > 80°C, increase fan speed or improve cooling
   ```

4. **Test GPU stability:**
   ```bash
   # Run GPU stress test
   # If fails, GPU may be faulty
   ```

---

### CPU Mining Not Working

**Symptoms:**
```
[CPU] Prime mining not supported
```

**Solutions:**

1. **Verify build with PRIME support:**
   ```bash
   # Rebuild with Prime support
   cmake -DWITH_PRIME=On ..
   make -j4
   ```

2. **Check dependencies:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install libgmp-dev libboost-all-dev
   ```

3. **Verify channel setting:**
   ```toml
   [mining]
   channel = 1  # Must be 1 for Prime (CPU)
   ```

---

## Advanced Diagnostics

### Enable Debug Logging

```toml
[logging]
level = 1  # 0=trace, 1=debug, 2=info
```

**Output to file:**
```bash
./NexusMiner miner.conf > miner.log 2>&1
```

---

### Network Packet Capture

```bash
# Capture mining traffic
sudo tcpdump -i any port 8323 -w mining.pcap

# Analyze with Wireshark
wireshark mining.pcap
```

---

### Check Protocol Opcodes

Look for these in debug logs:

**Push protocol:**
- legacy lane: `MINER_READY (0xD8)`, `GET_BLOCK (0x81)`
- stateless lane: `MINER_READY (0xD0D8)`, `GET_BLOCK (0xD081)`
- zero-payload opcodes include the 4-byte zero length on both lanes

**See:** [docs/reference/opcodes-reference.md](../reference/opcodes-reference.md)

---

### System Information

Collect this info for bug reports:

```bash
# NexusMiner version
./NexusMiner --version

# System info
uname -a
cat /etc/os-release

# Hardware info
lscpu
nvidia-smi  # For Nvidia GPUs
rocm-smi    # For AMD GPUs

# Node version
# In Nexus wallet:
system/get/info
```

---

## Additional Resources

### Detailed Guides
- **Enhanced Diagnostics:** [enhanced-diagnostics.md](enhanced-diagnostics.md)
- **Dynamic Port Detection:** [dynamic-port-detection.md](dynamic-port-detection.md)
- **Cross Validation Recovery:** [cross-validation-recovery.md](cross-validation-recovery.md)

### Configuration
- **Configuration Reference:** [docs/reference/nexus.conf.md](../reference/nexus.conf.md)
- **Example Configs:** [docs/reference/config-examples/](../reference/config-examples/)
- **TOML Format Guide:** [docs/reference/toml-format.md](../reference/toml-format.md)

### Protocol Documentation
- **Stateless Mining:** [docs/current/mining-protocols/stateless-mining.md](../current/mining-protocols/stateless-mining.md)
- **Opcodes Reference:** [docs/reference/opcodes-reference.md](../reference/opcodes-reference.md)
- **Migration Guide:** [docs/upgrade-guides/legacy-to-stateless.md](../upgrade-guides/legacy-to-stateless.md)

### Security
- **Genesis-First Protocol:** [docs/current/authentication/genesis-first-protocol.md](../current/authentication/genesis-first-protocol.md)
- **ChaCha20 Encryption:** [docs/current/security/chacha20-encryption.md](../current/security/chacha20-encryption.md)
- **Security Overview:** [docs/current/security/security-overview.md](../current/security/security-overview.md)

---

## Getting Help

If you're still having issues:

1. **Check GitHub Issues:**
   - [NexusMiner Issues](https://github.com/Nexusoft/NexusMiner/issues)
   - Search for similar problems

2. **Join Telegram:**
   - [Nexus Miners](https://t.me/NexusMiners)
   - Community support

3. **Create Bug Report:**
   - Include miner version
   - Include config file (remove private keys!)
   - Include relevant logs
   - Include system information

---

**Last Updated:** January 2026  
**NexusMiner Version:** 1.5+  
**Protocol:** Stateless mining support
