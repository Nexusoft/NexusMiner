# Migration Guide: Legacy to Stateless Mining Protocol

This guide helps you transition from the legacy polling-based mining protocol to the modern stateless push-notification protocol.

## Overview

**Good news:** Both mining lanes now use the same push-notification behavior. The selected port only changes the wire framing width.

## What Changed

### Legacy Lane (Port 8323)
- **8-bit opcode framing**
- **Push behavior** with `MINER_READY` and `GET_BLOCK`
- Header-only opcodes require `[opcode][00000000]`

### Stateless Lane (Port 9323+)
- **16-bit mirror-mapped opcode framing**
- **Push notifications** (`MINER_READY`, `GET_BLOCK`)
- Minimal network traffic
- < 10ms template latency
- Event-driven (lower CPU overhead)

---

## Migration Steps

### For Miners

#### Step 1: Update NexusMiner

```bash
# Download latest release
wget https://github.com/Nexusoft/NexusMiner/releases/latest

# Or build from source
git clone https://github.com/Nexusoft/NexusMiner.git
cd NexusMiner
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
```

#### Step 2: Update Configuration

**No configuration changes needed!** Your existing `miner.conf` will work.

**Optional:** Choose the wire framing your node exposes:

```toml
[wallet]
port = 8323  # Legacy 8-bit framing, push behavior
# port = 9323  # Stateless 16-bit mirror framing, same push behavior
```

#### Step 3: Generate Falcon Keys (if not already done)

```bash
./NexusMiner --create-keys
```

Add the generated keys to your `miner.conf`:

```toml
[falcon]
pubkey = "your_falcon_pubkey_here"
privkey = "your_falcon_privkey_here"

[mining]
genesis = "your_genesis_hash_here"
reward_address = "your_nxs_address_here"
```

#### Step 4: Start Mining

```bash
./NexusMiner miner.conf
```

**Look for these log messages:**

**Successful Push-Lane Connection:**
```
[Falcon Auth] ✅ Authentication successful
[Solo Protocol] LEGACY/STATELESS LANE: Using push protocol
[Solo Push] Sending MINER_READY
[Worker_manager] → GET_BLOCK sent
```

---

### For Node Operators

#### Step 1: Update Node Software

Update to LLL-TAO with stateless mining support:

```bash
# Check current version
./nexus system/get/info

# Update to latest (5.1.0+)
# Follow official Nexus update guide
```

**Minimum version:** LLL-TAO 5.1.0 with PR #170

#### Step 2: Configure Node

Enable mining in `nexus.conf`:

```ini
# Enable mining server
mining=1

# Set mining port (default: 8323)
miningport=8323

# Optional: Whitelist specific miner public keys
# minerallowkey=<falcon_pubkey_hex>
```

#### Step 3: Restart Node

```bash
# Restart Nexus node
./nexus -daemon
# or
systemctl restart nexus
```

#### Step 4: Verify

Check node logs for mining server startup:

```
[Mining] Mining LLP server initialized on port 8323
[Mining] Stateless protocol support: ENABLED
```

---

## Compatibility Matrix

| Miner Version | Node Version | Protocol Used | Status |
|---------------|--------------|---------------|--------|
| Current NexusMiner | Current NamecoinGithub/LLL-TAO | **Push on 8323 or 9323** | ✅ Best performance |
| Current NexusMiner | Node exposes 8323 only | **Legacy 8-bit push lane** | ✅ Same behavior, 8-bit framing |
| Current NexusMiner | Node exposes 9323+ | **Stateless 16-bit push lane** | ✅ Same behavior, 16-bit framing |
| Older miner/node | Pre-push protocol | **Unsupported beta path** | ⚠️ Update both before release |

---

## Configuration Changes

### What Stays the Same

✅ **No changes needed for:**
- Falcon keys (pubkey/privkey)
- Genesis hash
- Reward address
- Mining channel (Prime/Hash)
- Worker configuration
- Hardware settings (GPU/CPU power controls)
- TLS settings

### What's New (Optional)

**Keepalive Interval:**
```toml
[network]
keepalive_interval = 24  # Hours (1-168)
```

Default: 24 hours. Maintains session cache on node.

**See:** [docs/current/authentication/falcon-handshake-cache.md](../current/authentication/falcon-handshake-cache.md)

---

## Troubleshooting

### "Node doesn't support stateless protocol"

**This is not an error!** The miner will automatically use legacy protocol.

**To enable stateless:**
1. Update node to LLL-TAO 5.1.0+ with PR #170
2. Verify `mining=1` in nexus.conf
3. Ensure `miningport=8323`
4. Restart node
5. Restart miner (will auto-detect new capabilities)

---

### "Authentication failed with new miner"

**Possible causes:**
1. Missing genesis hash in config
2. Invalid Falcon keys
3. Pubkey not whitelisted on node

**Solution:**
```bash
# Generate new keys
./NexusMiner --create-keys

# Get genesis from wallet
# In Nexus wallet: system/get/info
# Copy "genesis" field

# Add to miner.conf
[mining]
genesis = "your_genesis_here"

[falcon]
pubkey = "your_pubkey_here"
privkey = "your_privkey_here"
```

---

### "Performance seems worse with new protocol"

**Check that push mode is active:**

```bash
# Look for this in miner logs:
grep "Using push protocol" miner.log
```

**If not active:**
- Node may be outdated
- Miner may be connected to a non-mining service/port
- Wrong framing on the selected port should disconnect, not fall back

**If active but slow:**
- Check network latency to node
- Verify node isn't overloaded
- Check system resources (CPU, RAM)

---

## Performance Comparison

### Real-World Measurements

**Network Traffic:**
- Legacy 8-bit lane: ~100 bytes/block (push only)
- Stateless 16-bit lane: ~100 bytes/block (push only)
- Difference: framing width only

**Template Latency:**
- Legacy 8-bit lane: < 10ms after push / immediate after initial GET_BLOCK
- Stateless 16-bit lane: < 10ms after push / immediate after initial GET_BLOCK

**CPU Overhead:**
- Both lanes: event-driven

**Stale Work:**
- Both lanes: near zero under normal push delivery

### Hash Rate Impact

**For most miners:**
- No change in hash rate (same mining algorithm)
- Slightly more accepted blocks (less stale work)
- Better efficiency (less wasted hashrate)

**For high-latency connections:**
- Significant improvement (instant updates vs polling delay)
- More competitive block finding

---

## Choosing a Lane

If you need 8-bit framing:

```toml
[wallet]
port = 8323  # Legacy 8-bit framing, same push behavior
```

If you need 16-bit mirror framing, use `port = 9323`. Both lanes use the same
push protocol and require explicit zero-length frames for zero-payload opcodes.

---

## Best Practices

### During Migration

✅ **DO:**
- Update miner first (automatic fallback ensures compatibility)
- Test with small worker count initially
- Monitor logs for protocol detection
- Update node when convenient
- Keep config files backed up

❌ **DON'T:**
- Don't update node without updating miner (older miners work but miss benefits)
- Don't disable logging until verified working
- Don't change config file format unnecessarily (both work)

### After Migration

✅ **DO:**
- Verify "Stateless protocol ACTIVE" in logs
- Monitor for NEW_BLOCK notifications
- Adjust keepalive_interval if needed (default 24h is fine)
- Keep miner and node up to date

❌ **DON'T:**
- Don't disable stateless protocol (no option to disable)
- Don't increase keepalive_interval too high (max 168 hours)

---

## Security Improvements

### Stateless Protocol Security

The new protocol includes several security improvements:

**Genesis-First Authentication:**
- Genesis hash sent first enables key derivation
- No TLS certificates needed
- Simpler, more secure key agreement

**Session Management:**
- Unique session IDs per connection
- Automatic session recovery
- Cache-based authentication

**ChaCha20 Encryption:**
- Always-on encryption for sensitive data
- Quantum-resistant Falcon signatures
- AAD (Additional Authenticated Data) protection

**See:**
- [docs/current/authentication/genesis-first-protocol.md](../current/authentication/genesis-first-protocol.md)
- [docs/current/security/chacha20-encryption.md](../current/security/chacha20-encryption.md)
- [docs/current/security/security-overview.md](../current/security/security-overview.md)

---

## FAQ

### Q: Will my existing config file work?
**A:** Yes! Both TOML (.config) and JSON (.conf) formats continue to work.

### Q: Do I need to generate new Falcon keys?
**A:** No, your existing keys work with the new protocol.

### Q: What if my node doesn't support stateless yet?
**A:** The miner automatically falls back to legacy protocol. Everything works normally.

### Q: Will I lose blocks during the switch?
**A:** No. The protocol switch happens during miner startup. No mining interruption.

### Q: Can I force legacy protocol?
**A:** Yes, use `port = 9323` in config, but not recommended (stateless is better).

### Q: How do I know if stateless is working?
**A:** Check logs for "Stateless protocol ACTIVE" message.

### Q: Does stateless protocol change the mining algorithm?
**A:** No, same Prime/Hash algorithms. Only communication protocol changes.

### Q: Will stateless improve my hash rate?
**A:** Hash rate stays the same, but you'll waste less work on stale templates.

### Q: Is stateless more secure?
**A:** Yes, includes genesis-first authentication and better session management.

### Q: Can I run both legacy and stateless miners?
**A:** Yes, same node supports both simultaneously (different sessions).

---

## Additional Resources

- **Stateless Protocol Details:** [docs/current/mining-protocols/stateless-mining.md](../current/mining-protocols/stateless-mining.md)
- **Configuration Reference:** [docs/reference/nexus.conf.md](../reference/nexus.conf.md)
- **Opcode Reference:** [docs/reference/opcodes-reference.md](../reference/opcodes-reference.md)
- **Falcon Authentication:** [docs/current/authentication/falcon-integration.md](../current/authentication/falcon-integration.md)

---

## Support

- **Telegram:** [Nexus Miners](https://t.me/NexusMiners)
- **GitHub Issues:** [NexusMiner Issues](https://github.com/Nexusoft/NexusMiner/issues)
- **Documentation:** [docs/](../)

---

**Migration Status:** ✅ Production Ready  
**Recommended:** All users should update to NexusMiner 1.5+  
**Timeline:** No deadline - automatic backward compatibility  
**Last Updated:** January 2026
