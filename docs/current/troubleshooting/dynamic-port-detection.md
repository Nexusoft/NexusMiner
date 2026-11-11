# Dynamic Port Detection Enhancement

## Overview

NexusMiner has been enhanced with comprehensive dynamic port detection and validation capabilities. This feature allows the miner to automatically adapt to and validate the LLP ports assigned by LLL-TAO nodes, providing better diagnostics and flexibility for different network configurations.

## Motivation

Previously, NexusMiner relied solely on static port configuration from `miner.conf`. While this works well for standard deployments, it provided limited visibility into:
- The actual port being used during runtime
- Whether the configured port matches the node's actual LLP port
- Potential misconfigurations between miner.conf and nexus.conf
- Port information during different connection states (initial connection, authentication, channel setup)

## Implementation Details

### 1. Connection Establishment Port Detection

**Location:** `src/miner.cpp` and `src/worker_manager.cpp`

When NexusMiner initializes a connection to a LLL-TAO node, it now:
- Logs the configured port from `miner.conf`
- Logs the actual connected remote endpoint (IP:Port)
- Logs the local endpoint information
- Provides clear debugging output for network troubleshooting

**Example Output:**
```
[info] === NexusMiner Port Configuration ===
[info] Configured wallet IP: 127.0.0.1
[info] Configured LLP port: 8323 (from miner.conf)
[info] Note: This is the miningport in LLL-TAO's nexus.conf
[info] =====================================
[info] [Solo] Connecting to wallet 127.0.0.1:8323
[info] [Solo] Port Configuration: Using port 8323 from miner.conf
[info] [Solo] Dynamic Port Detection: Successfully connected to 127.0.0.1:8323
[info] [Solo] Local endpoint: 192.168.1.100:54321
```

### 2. Authentication Session Port Validation

**Location:** `src/protocol/src/protocol/solo.cpp` - `MINER_AUTH_RESULT` handler

After successful Falcon authentication, NexusMiner logs:
- The authenticated session ID
- The actual port used for the authenticated session
- Port validation confirmation

**Example Output:**
```
[info] [Solo Phase 2] ✓ Authentication SUCCEEDED - Session ID: 0x12345678
[info] [Solo Phase 2] Authenticated session established on 127.0.0.1:8323
[debug] [Solo] Port Validation: Authenticated mining session using LLP port 8323
```

### 3. Channel Acknowledgment Port Detection

**Location:** `src/protocol/src/protocol/solo.cpp` - `CHANNEL_ACK` handler

When receiving CHANNEL_ACK from the node, NexusMiner now:
- Logs the actual connected port information
- Supports optional extended CHANNEL_ACK format (future protocol enhancement)
- Validates port consistency if the node communicates port information

**Standard CHANNEL_ACK Output:**
```
[info] [Solo Phase 2] Received CHANNEL_ACK from node
[info] [Solo] Dynamic Port Detection: Connected to 127.0.0.1:8323
[debug] [Solo] Port Validation: Using dynamically detected LLP port 8323
[info] [Solo] Channel acknowledged: 2 (hash)
```

**Extended CHANNEL_ACK (Future Enhancement):**
If the LLL-TAO protocol is extended to include port information in CHANNEL_ACK:
- Format: `[channel(1)][port(2, big-endian)]`
- NexusMiner will parse and validate the port
- Logs warnings if there's a mismatch between advertised and actual ports

```
[info] [Solo] Node communicated LLP port: 8323
[info] [Solo] Port validation successful: Both using port 8323
```

Or if there's a mismatch:
```
[warn] [Solo] Port mismatch: Node advertised port 9325 but connected to port 8323
```

### 4. Per-Packet Port Logging (Debug Level)

**Location:** `src/protocol/src/protocol/solo.cpp` - `process_messages()`

At debug log level, every packet processed includes endpoint information:
```
[debug] [Solo] Processing packet: header=129 (GET_BLOCK) length=0 | Remote: tcp://127.0.0.1#8323 | Local: tcp://192.168.1.100#54321
```

## Configuration

### miner.conf

No changes required to existing configurations. The `port` field continues to work as before:

```json
{
  "version": 1,
  "wallet_ip": "127.0.0.1",
  "port": 8323,
  ...
}
```

### nexus.conf (LLL-TAO Node)

Ensure the `miningport` matches the port configured in `miner.conf`:

```
miningport=8323
```

## Use Cases

### 1. Standard Configuration Validation

The enhanced logging helps confirm that miner and node are using the same port:
- Miner configured port: 8323 (from miner.conf)
- Node's miningport: 8323 (from nexus.conf)
- Actual connected port: 8323 (detected at runtime)

All three should match for proper operation.

### 2. Troubleshooting Connection Issues

If connection fails:
1. Check the initial "Port Configuration" output for the configured port
2. Verify it matches the node's `miningport` in nexus.conf
3. Check firewall rules for the specific port
4. Examine the connection error messages

### 3. Multi-Node Setup

When connecting to different nodes (testnet, mainnet, local), the logging makes it clear which port is being used for each connection attempt.

### 4. Custom Port Configurations

For nodes using non-default ports:
1. Set custom `miningport` in nexus.conf (e.g., 9325)
2. Update `port` in miner.conf to match
3. Verify connection logs show the correct port

## Future Protocol Enhancement: Extended CHANNEL_ACK

The implementation includes forward-compatible support for an extended CHANNEL_ACK format where the node can communicate its actual LLP port back to the miner.

**Current CHANNEL_ACK:**
```
[channel(1 byte)]
```

**Extended CHANNEL_ACK (Future):**
```
[channel(1 byte)][port(2 bytes, big-endian)]
```

Benefits:
- Node can explicitly communicate its LLP port
- Miner can validate configuration matches node's actual port
- Enables automatic port detection for advanced scenarios
- Backward compatible (extended fields are optional)

### Implementation in LLL-TAO

To support this in LLL-TAO nodes (future enhancement):

```cpp
// In miner.cpp CHANNEL_ACK response:
std::vector<uint8_t> response;
response.push_back(channel);  // Current behavior

// Add optional port information
uint16_t miningport = GetArg("-miningport", 8323);
response.push_back((miningport >> 8) & 0xFF);  // High byte
response.push_back(miningport & 0xFF);         // Low byte

return response;
```

NexusMiner will automatically detect and validate this information once nodes start sending it.

## Debugging

### Enable Debug Logging

Set `log_level` to 0 or 1 in miner.conf for maximum detail:

```json
{
  "log_level": 0,  // 0=trace, 1=debug, 2=info
  ...
}
```

### Key Log Messages

**Port Configuration:**
- `"=== NexusMiner Port Configuration ==="` - Initial configuration summary
- `"Configured LLP port: X (from miner.conf)"` - Port from config file
- `"Port Configuration: Using port X from miner.conf"` - Connection attempt port

**Connection Success:**
- `"Dynamic Port Detection: Successfully connected to IP:PORT"` - Actual connection
- `"Local endpoint: IP:PORT"` - Local side of connection

**Authentication:**
- `"Authenticated session established on IP:PORT"` - Session port info
- `"Port Validation: Authenticated mining session using LLP port X"` - Validation

**Channel Setup:**
- `"Dynamic Port Detection: Connected to IP:PORT"` - CHANNEL_ACK port info
- `"Port Validation: Using dynamically detected LLP port X"` - Final validation

## Benefits

1. **Improved Diagnostics:** Clear visibility into port configuration and actual connections
2. **Configuration Validation:** Easier to verify miner.conf matches nexus.conf
3. **Troubleshooting:** Port-related issues are immediately visible in logs
4. **Future-Proof:** Ready for extended protocol with node-advertised ports
5. **Backward Compatible:** Works with all existing nodes and configurations
6. **Multi-Environment Support:** Better logging for testnet/mainnet/local setups

## Testing

The enhancement has been tested with:
- ✅ Compilation successful on Linux
- ✅ Code structure validated
- ✅ Backward compatibility maintained
- ✅ No breaking changes to existing protocol
- ✅ Forward-compatible with future protocol extensions

## Related Documentation

- [Mining LLP Protocol](mining-llp-protocol.md) - LLP protocol specification
- [Falcon Authentication](falcon_authentication.md) - Phase 2 authentication
- [Phase 2 Integration](../PHASE2_INTEGRATION.md) - Phase 2 overview

## Summary

This enhancement provides comprehensive port detection and validation throughout the connection lifecycle:
1. **Initial Configuration:** Logs configured port from miner.conf
2. **Connection Establishment:** Detects and logs actual connected port
3. **Authentication:** Validates port for authenticated sessions
4. **Channel Setup:** Confirms port usage and supports future extensions
5. **Runtime:** Per-packet port logging at debug level

The implementation is backward compatible, non-invasive, and provides excellent diagnostic capabilities for troubleshooting port-related issues in NexusMiner deployments.
