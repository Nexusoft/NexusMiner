# TLS/HTTPS Integration for Secure Remote Mining

## Overview

NexusMiner now supports TLS/HTTPS encrypted connections for secure remote mining. This feature ensures that all communication with the LLL-TAO Node is encrypted and authenticated when mining over untrusted networks (e.g., the Internet).

**NEW:** Mutual TLS (mTLS) authentication is now supported! See [mutual_tls_authentication.md](mutual_tls_authentication.md) for client certificate configuration.

## Features

### 1. Automatic TLS Enablement

TLS/HTTPS is automatically enabled for remote (non-localhost) connections when configured. For localhost connections, TLS is optional since the communication stays within the local machine.

**Auto-Detection Logic:**
- **Localhost** (127.0.0.1, ::1, localhost): TLS optional (disabled by default)
- **Remote connections**: TLS recommended (can be forced with `enable_tls: true`)

### 2. Secure Default Settings

The TLS implementation enforces modern security standards:

**Protocol Versions:**
- TLS 1.2 (minimum)
- TLS 1.3 (preferred)
- SSL v2/v3, TLS 1.0, TLS 1.1 disabled

**Cipher Suites (in priority order):**

TLS 1.3:
- TLS_CHACHA20_POLY1305_SHA256
- TLS_AES_256_GCM_SHA384
- TLS_AES_128_GCM_SHA256

TLS 1.2:
- ECDHE-ECDSA-CHACHA20-POLY1305
- ECDHE-RSA-CHACHA20-POLY1305
- ECDHE-ECDSA-AES256-GCM-SHA384
- ECDHE-RSA-AES256-GCM-SHA384
- ECDHE-ECDSA-AES128-GCM-SHA256
- ECDHE-RSA-AES128-GCM-SHA256

All cipher suites provide:
- Forward secrecy (ECDHE key exchange)
- Strong encryption (ChaCha20-Poly1305, AES-GCM)
- Authentication (ECDSA or RSA signatures)

### 3. Certificate Validation

**Client Mode (Miner):**
- Peer certificate verification enabled by default
- Uses system CA certificate bundle by default
- Optional custom CA certificate path
- Server Name Indication (SNI) support
- Hostname verification

**Server Mode (Pool):**
- Certificate and private key configuration
- Optional password-protected private keys
- Full TLS 1.2/1.3 support

## Configuration

### Configuration Fields

```json
{
    "enable_tls": false,          // Enable TLS/HTTPS (auto for remote)
    "tls_ca_cert_path": "",       // CA bundle path (empty = system default)
    "tls_verify_peer": true,      // Verify server certificate
    "tls_server_name": ""         // Server name for SNI (empty = wallet_ip)
}
```

### Field Descriptions

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_tls` | boolean | false | Force enable TLS (auto-enabled for remote) |
| `tls_ca_cert_path` | string | "" | Path to CA certificate bundle (empty = use system default) |
| `tls_verify_peer` | boolean | true | Verify server certificate (always recommended) |
| `tls_server_name` | string | "" | Server name for SNI and verification (empty = use wallet_ip) |

## Usage Scenarios

### Scenario 1: Localhost Mining (No TLS)

```json
{
    "wallet_ip": "127.0.0.1",
    "port": 8323,
    "enable_tls": false
}
```

**Characteristics:**
- No TLS overhead
- Fastest performance
- Secure within local machine
- Recommended for solo mining on same machine

### Scenario 2: Remote Mining with System CA Bundle

```json
{
    "wallet_ip": "mining.pool.com",
    "port": 8323,
    "enable_tls": true,
    "tls_verify_peer": true
}
```

**Characteristics:**
- TLS 1.2/1.3 encryption
- Uses system CA certificates
- Full certificate validation
- Recommended for public pools with valid certificates

### Scenario 3: Remote Mining with Custom CA Certificate

```json
{
    "wallet_ip": "private-pool.internal",
    "port": 8323,
    "enable_tls": true,
    "tls_ca_cert_path": "/path/to/ca-bundle.crt",
    "tls_verify_peer": true,
    "tls_server_name": "private-pool.internal"
}
```

**Characteristics:**
- Custom CA for private/self-signed certificates
- Full certificate validation
- SNI with custom server name
- Recommended for private pools or internal networks

### Scenario 4: Testing/Development (Verification Disabled)

```json
{
    "wallet_ip": "test-node.local",
    "port": 8323,
    "enable_tls": true,
    "tls_verify_peer": false
}
```

**⚠️ WARNING:** Disabling peer verification is INSECURE and should only be used for testing/development!

**Characteristics:**
- TLS encryption without verification
- Vulnerable to man-in-the-middle attacks
- NOT recommended for production use
- Use only for testing with self-signed certificates

## Security Considerations

### Certificate Validation

**Always Verify Peer Certificates:**
- Set `tls_verify_peer: true` (default)
- Ensures you're connecting to the intended server
- Prevents man-in-the-middle attacks

**Use System CA Bundle:**
- Leave `tls_ca_cert_path` empty for public pools
- System CA bundles are regularly updated
- Automatically validates certificates from trusted CAs

**Custom CA Certificates:**
- Use for private pools or self-signed certificates
- Keep CA certificates secure and updated
- Validate certificate chain before deployment

### Cipher Suite Security

The default cipher suite list prioritizes:
1. **Forward Secrecy**: ECDHE key exchange
2. **Modern Encryption**: ChaCha20-Poly1305, AES-GCM
3. **No Weak Ciphers**: RC4, DES, MD5 excluded

### Server Name Indication (SNI)

SNI is used for:
- Virtual hosting support
- Hostname verification
- Certificate selection (server-side)

**Default Behavior:**
- `tls_server_name` defaults to `wallet_ip`
- Override for IP-based connections to domain-named servers

## Performance Impact

### TLS Overhead

**Handshake (once per connection):**
- ~10-50ms additional latency
- CPU overhead for key exchange
- Amortized over long-lived connections

**Data Transfer (per packet):**
- ~5-10% throughput reduction
- Minimal CPU overhead (AES-NI, ChaCha20 optimized)
- Negligible impact on mining performance

**Recommendations:**
- Use TLS 1.3 for improved performance (faster handshake)
- Long-lived connections minimize handshake overhead
- Hardware acceleration (AES-NI) reduces encryption cost

## Troubleshooting

### Certificate Verification Failures

**Symptom:** "Certificate verification failed" or "Unknown CA"

**Solutions:**
1. Ensure server certificate is valid and not expired
2. Verify CA certificate bundle is up-to-date
3. Check `tls_server_name` matches certificate CN/SAN
4. Use custom CA path for self-signed certificates

### Connection Timeouts

**Symptom:** Connection times out or fails to establish

**Solutions:**
1. Verify TLS is enabled on server/pool
2. Check firewall allows TLS port (typically 8323)
3. Ensure server supports TLS 1.2/1.3
4. Review server logs for TLS errors

### Cipher Suite Mismatch

**Symptom:** "No shared cipher" or "Cipher mismatch"

**Solutions:**
1. Ensure server supports modern cipher suites
2. Update OpenSSL to latest version
3. Check server cipher configuration
4. Verify TLS 1.2/1.3 support on both sides

### Hostname Verification Failures

**Symptom:** "Hostname verification failed"

**Solutions:**
1. Set `tls_server_name` to match certificate CN/SAN
2. Use IP address in certificate if connecting by IP
3. Ensure DNS resolution is correct
4. Check certificate Subject Alternative Names (SAN)

## Integration with Existing Features

### ChaCha20 Wrapper Integration

TLS and ChaCha20 Falcon key wrapping work together:

**TLS Layer:**
- Encrypts all network traffic
- Protects entire connection
- Uses ChaCha20-Poly1305 cipher (when available)

**ChaCha20 Wrapper:**
- Encrypts Falcon Public Keys
- Additional layer of protection
- Application-level encryption

**Combined Security:**
- Defense in depth approach
- Double encryption of Falcon keys
- Maximum security for remote mining

### Localhost Detection

Automatic security optimization based on connection type:

**Localhost (127.0.0.1, ::1):**
- TLS disabled by default
- ChaCha20 wrapping optional
- Maximum performance

**Remote (any other IP):**
- TLS auto-enabled (recommended)
- ChaCha20 wrapping auto-enabled
- Maximum security

## Best Practices

### For Solo Miners

**Localhost Mining:**
```json
{
    "wallet_ip": "127.0.0.1",
    "enable_tls": false
}
```
> **Note:** ChaCha20 encryption is always active (core security) — no configuration needed.

**Remote Mining:**
```json
{
    "wallet_ip": "my-node.example.com",
    "enable_tls": true,
    "tls_verify_peer": true
}
```
> **Note:** ChaCha20 encryption is always active (core security) — no configuration needed.

### For Pool Miners

**Public Pool with Valid Certificate:**
```json
{
    "wallet_ip": "pool.nexus.io",
    "enable_tls": true,
    "tls_verify_peer": true
}
```

**Private Pool with Self-Signed Certificate:**
```json
{
    "wallet_ip": "private-pool.local",
    "enable_tls": true,
    "tls_ca_cert_path": "/etc/nexus/pool-ca.crt",
    "tls_verify_peer": true,
    "tls_server_name": "private-pool.local"
}
```

### Certificate Management

1. **Obtain Certificates:**
   - Use Let's Encrypt for public pools (free, automated)
   - Generate self-signed for private/testing
   - Purchase from CA for commercial pools

2. **Certificate Renewal:**
   - Automate renewal (Let's Encrypt)
   - Monitor expiration dates
   - Update miner configuration if paths change

3. **CA Bundle Updates:**
   - Keep system CA bundle updated
   - Use `apt-get update && apt-get upgrade ca-certificates` (Debian/Ubuntu)
   - Review custom CA certificates periodically

## Technical Details

### TLS Implementation

**Library:** ASIO SSL (Boost.Asio SSL wrapper)
**Backend:** OpenSSL 1.1.1+ or 3.0+
**Async I/O:** Non-blocking operations
**Thread Safety:** Context-level thread safety

### Cipher Suite Selection

The cipher suite list is optimized for:
- **Security**: No weak or deprecated ciphers
- **Performance**: Hardware-accelerated algorithms
- **Compatibility**: Works with most modern servers

### Protocol Negotiation

1. Client sends ClientHello with supported versions/ciphers
2. Server selects best version and cipher
3. TLS 1.3 preferred (if both support)
4. Fallback to TLS 1.2 if needed
5. Connection fails if no common version/cipher

## Future Enhancements

Planned improvements:

1. ~~**Client Certificates**: Mutual TLS authentication~~ ✅ **IMPLEMENTED** - See [mutual_tls_authentication.md](mutual_tls_authentication.md)
2. **Certificate Pinning**: Pin specific certificates for enhanced security
3. **OCSP Stapling**: Real-time certificate revocation checking
4. **Session Resumption**: Faster reconnections with session tickets
5. **Performance Monitoring**: TLS handshake and throughput metrics

## Mutual TLS Authentication

For the highest level of security, NexusMiner now supports mutual TLS (mTLS) where both client and server authenticate each other using certificates.

**Quick Configuration:**
```json
{
    "enable_tls": true,
    "tls_client_cert_path": "/path/to/client-cert.pem",
    "tls_client_key_path": "/path/to/client-key.pem",
    "tls_client_key_password": "optional_password"
}
```

**See [mutual_tls_authentication.md](mutual_tls_authentication.md) for:**
- Certificate generation and installation
- Server-side configuration
- Security best practices
- Troubleshooting guide

## References

- TLS 1.3: RFC 8446
- TLS 1.2: RFC 5246
- ChaCha20-Poly1305: RFC 8439
- OpenSSL Documentation: https://www.openssl.org/docs/
- ASIO SSL: https://think-async.com/Asio/

---

**Version**: 1.0  
**Last Updated**: 2025-12-06  
**Compatibility**: NexusMiner 1.5+, OpenSSL 1.1.1+/3.0+
