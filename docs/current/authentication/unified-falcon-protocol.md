# Unified Hybrid Falcon Signature Protocol - Implementation Guide

## Overview

The Unified Hybrid Falcon Signature Protocol provides a centralized wrapper for Falcon-based
signature operations in NexusMiner. The miner currently uses **Falcon-1024 by default**
while preserving **Falcon-512 compatibility** for testing and interoperability.

This implementation builds on the stateless MINER_AUTH_RESPONSE flow and the current
encrypted submit path shared with the node.

## Two Types of Falcon Signatures

### Disposable Falcon Signatures (ALWAYS ON)

**Purpose**: Session authentication for mining protocol
**Status**: Core protocol feature, ALWAYS ENABLED, cannot be disabled
**Blockchain Storage**: NOT stored (0 bytes overhead)
**Use Case**: Authenticates each block submission to the node

Disposable signatures are ephemeral authentication tokens that prove the miner has the private key without storing anything on the blockchain. They are generated for every block submission and are verified by the node but not permanently recorded.

### Physical Falcon Signatures (CONFIGURABLE)

**Purpose**: Optional blockchain-stored proof of authorship
**Status**: Optional feature, OFF by default (lazy miner economics)
**Blockchain Storage**: Adds 809 bytes (Falcon-512) or 1577 bytes (Falcon-1024) per block
**Use Case**: Future blockchain integration for permanent mining authorship records

Physical signatures are the optional dual-signature system that stores the signature permanently on the blockchain for future proof of mining work. Most miners keep this OFF to save 61% blockchain space.

## Architecture

### FalconSignatureWrapper Class

The `FalconSignatureWrapper` class (`src/protocol/inc/protocol/falcon_wrapper.hpp`) provides:

- **Centralized signature operations**: Falcon-512 and Falcon-1024 signatures go through a single interface
- **Multiple signature types**: Authentication, block submission, and generic payload signing
- **Performance optimization**: Signature caching, timing metrics, and performance statistics
- **Thread safety**: Safe for multi-worker mining environments
- **Security**: Automatic private key clearing on destruction

### Key Features

#### 1. Authentication Signatures (MINER_AUTH_RESPONSE)

Used for the Phase 2 stateless miner authentication protocol:

```cpp
auto sig_result = wrapper->sign_authentication(address, timestamp);
if (sig_result.success) {
    // Use sig_result.signature for MINER_AUTH_RESPONSE packet
    // sig_result.generation_time tracks performance
}
```

**Message format**: `address + timestamp (8 bytes LE)`

#### 2. Disposable Block Signatures (ALWAYS ON)

Core protocol feature for session authentication:

```cpp
auto sig_result = wrapper->sign_payload(block_bytes_plus_timestamp,
                                        SignatureType::BLOCK);
if (sig_result.success) {
    // Included in SUBMIT_BLOCK packet
    // Provides cryptographic proof for this mining session
    // NOT stored on blockchain (0 bytes overhead)
}
```

**Signed message format**: `serialized_block [+ Prime vOffsets] + timestamp (8 bytes LE)`

**Submit plaintext format**:
`serialized_block [+ Prime vOffsets] + timestamp (8 bytes LE) + sig_len (2 bytes LE) + signature`

**Note**: Disposable Falcon signatures are **ALWAYS ON** (core protocol). Physical Falcon signatures are optional and configurable separately.

#### 3. Generic Payload Signatures

For protocol extensions and custom validation:

```cpp
auto sig_result = wrapper->sign_payload(data, SignatureType::PAYLOAD);
```

### Integration with Solo Protocol

The `Solo` protocol class has been updated to use the wrapper:

1. **Wrapper initialization**: Created when `set_miner_keys()` is called
2. **Authentication**: `login()` uses wrapper for MINER_AUTH_RESPONSE signatures
3. **Disposable signatures**: `submit_block()` always signs the solved submit payload before ChaCha20 wrapping
4. **Physical signatures**: Optional via `enable_physical_falcon()` API (OFF by default)

## Configuration

### Basic Configuration (Authentication Only)

Standard SOLO mining configuration:

```json
{
    "version": 1,
    "wallet_ip": "127.0.0.1",
    "port": 8323,
    "local_ip": "127.0.0.1",
    "mining_mode": "PRIME",
    "miner_falcon_pubkey": "<your_public_key_hex>",
    "miner_falcon_privkey": "<your_private_key_hex>",
    "workers": [...]
}
```

**Default behavior**: 
- Disposable Falcon signatures: **ALWAYS ON** (core protocol, cannot be disabled)
- Physical Falcon signatures: **OFF** (optional, for blockchain storage)
- ChaCha20 encryption: **ALWAYS ON** (core security)

### Note on Configuration

**Important**: Disposable Falcon signatures and ChaCha20 encryption are ALWAYS ON and cannot be configured. The configuration options `enable_block_signing` and `enable_chacha20_wrapping` are deprecated and ignored (forced to true).

For Physical Falcon signatures (optional blockchain storage), use the `enable_physical_falcon()` API method instead of configuration files.

## Performance Considerations

### Signature Generation Times

Typical signature generation time depends on the configured Falcon key size and CPU.

- Disposable Falcon: Per block submission (always enabled, minimal overhead)
- Physical Falcon: Per block submission when enabled (optional)

### Performance Statistics

The wrapper tracks comprehensive performance metrics:

```cpp
auto stats = wrapper->get_stats();
// stats.total_signatures
// stats.auth_signatures
// stats.block_signatures
// stats.average_time_microseconds
```

Statistics are logged periodically by the Solo protocol.

### Optimization Features

1. **Lazy initialization**: Wrapper created only when Falcon keys are set
2. **Smart fallback**: Falls back to direct signing if wrapper fails
3. **Submit-path signing**: Always enabled for disposable session signatures
4. **Memory safety**: Private key automatically cleared on destruction

## Protocol Flow

### Phase 2 Authentication (MINER_AUTH_RESPONSE)

```
1. Miner: Create FalconSignatureWrapper with keys
2. Miner: wrapper->sign_authentication(address, timestamp)
3. Miner: Build MINER_AUTH_RESPONSE packet:
          [pubkey_len(2)][pubkey][sig_len(2)][signature]
4. Miner → Node: MINER_AUTH_RESPONSE
5. Node → Miner: MINER_AUTH_RESULT (session_id)
6. Miner: SET_CHANNEL
7. Node: CHANNEL_ACK
8. Mining begins...
```

### Block Submission (current wire shape)

```
1. Worker finds a block against its solved snapshot
2. Miner serializes the full 216-byte Tritium block body from that solved snapshot
3. Prime channel appends vOffsets after the 216-byte block body
4. Miner appends timestamp (8 LE) + sig_len (2 LE) + Falcon signature
5. Miner encrypts the plaintext payload with ChaCha20-Poly1305
6. Node decrypts, validates, and accepts/rejects
```

Notes:

- **Hash plaintext** = `216-byte block + timestamp + sig_len + signature`
- **Prime plaintext** = `216-byte block + vOffsets + timestamp + sig_len + signature`
- **Falcon-1024 is the default**, so Prime submit sizes are larger than the old Falcon-512-era examples
- Node-side fixed-offset `sig_len` probing is obsolete for Prime because `vOffsets` make the submit payload variable-length

## Logging

The wrapper provides detailed logging at multiple levels:

**INFO level**:
- Wrapper initialization
- Signature generation success/failure
- Performance metrics

**DEBUG level**:
- Detailed payload structures
- Signature sizes
- Generation times

**Example logs**:
```
[FalconWrapper] Initialized successfully
[FalconWrapper]   - Public key: 1793 bytes
[FalconWrapper]   - Private key: 2305 bytes
[Solo Auth] Using Falcon Signature Wrapper for authentication
[Solo Auth] Wrapper signature generated in 234 μs
[Solo Submit] Block signing enabled - generating signature
[FalconWrapper] Block signature generated successfully
[FalconWrapper]   - Signature size: 1577 bytes
[FalconWrapper]   - Generation time: 187 μs
```

## Security Features

1. **Private key protection**: Automatically cleared from memory on wrapper destruction
2. **Signature validation**: Wrapper validates key sizes on initialization
3. **Error handling**: Comprehensive error messages for debugging
4. **Fallback support**: Falls back to direct signing if wrapper fails

## Future Enhancements

Planned features for the Unified Falcon Signature Protocol:

1. **Signature caching**: Cache recent signatures to avoid redundant computation
2. **Batch signing**: Sign multiple payloads in a single operation
3. **Hardware acceleration**: Utilize CPU instructions for faster signing
4. **Genesis binding**: Support Tritium account binding per LLL-TAO Phase 2 spec
5. **Adaptive signing**: Automatically enable/disable block signing based on network conditions

## API Reference

### FalconSignatureWrapper

**Constructor**:
```cpp
FalconSignatureWrapper(const std::vector<uint8_t>& pubkey,
                      const std::vector<uint8_t>& privkey);
```

**Methods**:
- `sign_authentication(address, timestamp)`: Sign auth message for MINER_AUTH_RESPONSE
- `sign_block(block_data, nonce)`: Legacy helper retained for compatibility
- `sign_payload(data, type)`: Sign arbitrary payload, including current submit plaintexts
- `is_valid()`: Check if wrapper is initialized
- `get_stats()`: Get performance statistics
- `reset_stats()`: Reset performance counters

**SignatureResult Structure**:
```cpp
struct SignatureResult {
    bool success;
    std::vector<uint8_t> signature;
    std::string error_message;
    std::chrono::microseconds generation_time;
};
```

### Solo Protocol Extensions

**New methods**:
- `enable_block_signing(bool enable)`: Backward-compatible no-op for legacy configs
- `is_block_signing_enabled()`: Reports the disposable-signature compatibility flag

## Troubleshooting

### Wrapper Initialization Failed

**Symptom**: `Falcon Signature Wrapper initialization failed - invalid keys`

**Solutions**:
- Verify public key is exactly 897 bytes (Falcon-512) or 1793 bytes (Falcon-1024)
- Verify private key is exactly 1281 bytes (Falcon-512) or 2305 bytes (Falcon-1024)
- Regenerate keys: `./NexusMiner --create-keys`

### Signature Generation Failed

**Symptom**: `Falcon Wrapper signature failed: <error>`

**Solutions**:
- Check private key is valid and not corrupted
- Verify payload data is not empty
- Check system memory availability
- Review wrapper logs for detailed error messages

### Performance Impact

**Symptom**: Lower hashrate during signature generation

**Expected**: Submit-path signing adds a small per-found-block cost and does not affect steady-state hashing

**Solutions**:
- Use faster CPU for signature generation
- Submit-path signing only affects submission, not mining performance

## Compatibility

- **LLL-TAO Phase 2**: Fully compatible with stateless miner protocol
- **Backward compatible**: Falls back to direct signing if wrapper unavailable
- **Pool mining**: N/A - wrapper is for SOLO mining only
- **Legacy nodes**: Works with Phase 2 nodes implementing MINER_AUTH opcodes

## Files Modified

1. `src/protocol/inc/protocol/falcon_wrapper.hpp` - Wrapper interface (new)
2. `src/protocol/src/protocol/falcon_wrapper.cpp` - Wrapper implementation (new)
3. `src/protocol/inc/protocol/solo.hpp` - Added wrapper integration
4. `src/protocol/src/protocol/solo.cpp` - Use wrapper for signatures
5. `src/protocol/CMakeLists.txt` - Added falcon_wrapper.cpp to build

---

**Implementation Date**: 2025-11-26  
**Based on**: PR #24 (Phase 2 Direct MINER_AUTH_RESPONSE Protocol)  
**Author**: GitHub Copilot Code Agent
