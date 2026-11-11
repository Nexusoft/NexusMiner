# Mutual TLS (mTLS) Authentication for NexusMiner

## Overview

Mutual TLS (mTLS) authentication extends standard TLS by requiring both the client (miner) and server (pool/node) to authenticate each other using certificates. This provides the highest level of security for remote mining connections.

## What is Mutual TLS?

In standard TLS:
- **Server** authenticates to **client** using a certificate
- **Client** verifies the server's identity
- **Server** does NOT verify the client's identity

In mutual TLS (mTLS):
- **Server** authenticates to **client** using a certificate
- **Client** authenticates to **server** using a certificate
- **Both parties** verify each other's identity

## Benefits of Mutual TLS

1. **Strong Authentication**: Cryptographic proof of identity for both parties
2. **No Credentials in Config**: No need to store passwords or API keys
3. **Certificate-Based Access Control**: Server can restrict access based on client certificates
4. **Compliance**: Meets strict security requirements for regulated environments
5. **Non-Repudiation**: Cryptographic proof of who connected and when

## Configuration

### Configuration Fields

```json
{
    "enable_tls": true,
    "tls_verify_peer": true,
    "tls_ca_cert_path": "/path/to/ca-bundle.crt",
    "tls_client_cert_path": "/path/to/client-cert.pem",
    "tls_client_key_path": "/path/to/client-key.pem",
    "tls_client_key_password": "optional_password"
}
```

### Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `enable_tls` | boolean | Yes | Must be `true` for mTLS |
| `tls_verify_peer` | boolean | Yes | Must be `true` for mTLS |
| `tls_ca_cert_path` | string | Yes | CA bundle that includes server's CA |
| `tls_client_cert_path` | string | Yes | Path to client certificate (PEM) |
| `tls_client_key_path` | string | Yes | Path to client private key (PEM) |
| `tls_client_key_password` | string | No | Password for encrypted private key |

## Certificate Generation

### Option 1: Using OpenSSL (Self-Signed)

**Step 1: Create a Certificate Authority (CA)**

```bash
# Generate CA private key
openssl genrsa -out ca-key.pem 4096

# Generate CA certificate (valid for 10 years)
openssl req -new -x509 -days 3650 -key ca-key.pem -out ca-cert.pem \
    -subj "/CN=NexusMiner CA/O=Your Organization/C=US"
```

**Step 2: Create Client Certificate**

```bash
# Generate client private key
openssl genrsa -out client-key.pem 2048

# Generate certificate signing request (CSR)
openssl req -new -key client-key.pem -out client.csr \
    -subj "/CN=miner-001/O=Your Organization/C=US"

# Sign the CSR with your CA
openssl x509 -req -days 365 -in client.csr \
    -CA ca-cert.pem -CAkey ca-key.pem -CAcreateserial \
    -out client-cert.pem
```

**Step 3: (Optional) Password-Protect Private Key**

```bash
# Convert to password-protected format
openssl rsa -aes256 -in client-key.pem -out client-key-encrypted.pem
```

**Step 4: Create Server Certificate (for pool operators)**

```bash
# Generate server private key
openssl genrsa -out server-key.pem 2048

# Generate server CSR
openssl req -new -key server-key.pem -out server.csr \
    -subj "/CN=mining.pool.com/O=Your Organization/C=US"

# Sign the server CSR
openssl x509 -req -days 365 -in server.csr \
    -CA ca-cert.pem -CAkey ca-key.pem -CAcreateserial \
    -out server-cert.pem
```

### Option 2: Using Let's Encrypt (Public Pools)

Let's Encrypt does not issue client certificates. For public pools:
- Use Let's Encrypt for **server** certificates
- Generate self-signed **client** certificates as shown above
- Configure server to trust your client CA

### Option 3: Using Commercial CA

For enterprise deployments:
1. Purchase certificates from a commercial CA (DigiCert, GlobalSign, etc.)
2. Request both server and client certificates
3. Follow CA-specific instructions for CSR generation
4. Install certificates as described below

## Installation

### File Structure

```
/etc/nexus/certs/
├── ca-cert.pem                # CA certificate bundle
├── client-cert.pem            # Client certificate
├── client-key.pem             # Client private key
└── client-key-encrypted.pem   # Password-protected key (optional)
```

### File Permissions

```bash
# Set restrictive permissions
chmod 600 /etc/nexus/certs/client-key.pem
chmod 644 /etc/nexus/certs/client-cert.pem
chmod 644 /etc/nexus/certs/ca-cert.pem

# Change ownership to miner user
chown nexusminer:nexusminer /etc/nexus/certs/*
```

## Configuration Examples

### Example 1: Mutual TLS with Self-Signed Certificates

```json
{
    "wallet_ip": "mining.pool.com",
    "port": 8323,
    "enable_tls": true,
    "tls_verify_peer": true,
    "tls_ca_cert_path": "/etc/nexus/certs/ca-cert.pem",
    "tls_client_cert_path": "/etc/nexus/certs/client-cert.pem",
    "tls_client_key_path": "/etc/nexus/certs/client-key.pem"
}
```

**Use Case:**
- Private mining pool
- Self-managed certificate infrastructure
- Maximum control over authentication

### Example 2: Mutual TLS with Password-Protected Key

```json
{
    "wallet_ip": "secure-pool.internal",
    "port": 8323,
    "enable_tls": true,
    "tls_verify_peer": true,
    "tls_ca_cert_path": "/etc/nexus/certs/ca-cert.pem",
    "tls_client_cert_path": "/etc/nexus/certs/client-cert.pem",
    "tls_client_key_path": "/etc/nexus/certs/client-key-encrypted.pem",
    "tls_client_key_password": "your_secure_password"
}
```

**Use Case:**
- High-security environments
- Compliance requirements (PCI-DSS, HIPAA, etc.)
- Additional layer of protection for private keys

### Example 3: Mutual TLS with Commercial Certificates

```json
{
    "wallet_ip": "enterprise-pool.company.com",
    "port": 8323,
    "enable_tls": true,
    "tls_verify_peer": true,
    "tls_ca_cert_path": "/etc/ssl/certs/ca-bundle.crt",
    "tls_client_cert_path": "/etc/nexus/certs/client-cert.pem",
    "tls_client_key_path": "/etc/nexus/certs/client-key.pem",
    "tls_server_name": "enterprise-pool.company.com"
}
```

**Use Case:**
- Enterprise deployments
- Commercial CA certificates
- Multiple mining pools with different CAs

## Server-Side Configuration (For Pool Operators)

### Nginx Example

```nginx
server {
    listen 8323 ssl;
    server_name mining.pool.com;
    
    # Server certificate
    ssl_certificate /etc/nginx/certs/server-cert.pem;
    ssl_certificate_key /etc/nginx/certs/server-key.pem;
    
    # Client certificate verification
    ssl_client_certificate /etc/nginx/certs/ca-cert.pem;
    ssl_verify_client on;
    ssl_verify_depth 2;
    
    # TLS settings
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers 'ECDHE+AESGCM:ECDHE+CHACHA20';
    
    location / {
        proxy_pass http://localhost:8324;
        proxy_set_header X-Client-Cert $ssl_client_cert;
    }
}
```

### Apache Example

```apache
<VirtualHost *:8323>
    ServerName mining.pool.com
    
    SSLEngine on
    SSLCertificateFile /etc/apache2/certs/server-cert.pem
    SSLCertificateKeyFile /etc/apache2/certs/server-key.pem
    
    SSLVerifyClient require
    SSLVerifyDepth 2
    SSLCACertificateFile /etc/apache2/certs/ca-cert.pem
    
    SSLProtocol -all +TLSv1.2 +TLSv1.3
    SSLCipherSuite ECDHE+AESGCM:ECDHE+CHACHA20
    
    ProxyPass / http://localhost:8324/
</VirtualHost>
```

### Direct LLL-TAO Node Configuration

If the LLL-TAO node supports mTLS directly, configure it to:
1. Require client certificates
2. Trust your client CA certificate
3. Verify client certificate chain
4. Extract client identity from certificate CN/SAN

## Security Best Practices

### Certificate Management

1. **Key Generation:**
   - Use strong key sizes (2048-bit RSA minimum, 4096-bit recommended)
   - Consider ECDSA for better performance (256-bit or 384-bit)
   - Generate keys on the client machine (never transmit private keys)

2. **Certificate Lifetime:**
   - Client certificates: 1 year (renewable)
   - CA certificates: 5-10 years
   - Automate renewal before expiration

3. **Key Protection:**
   - Use password-protected private keys in production
   - Store keys in secure locations with restrictive permissions
   - Consider hardware security modules (HSM) for critical deployments

4. **Certificate Revocation:**
   - Maintain a Certificate Revocation List (CRL)
   - Implement OCSP for real-time revocation checking
   - Promptly revoke compromised certificates

### Access Control

1. **Certificate-Based Authorization:**
   - Use certificate CN/SAN for user identification
   - Implement role-based access control (RBAC)
   - Maintain audit logs of certificate usage

2. **Certificate Whitelisting:**
   - Server maintains list of allowed client certificates
   - Block unknown certificates by default
   - Regular review of authorized certificates

### Monitoring

1. **Certificate Expiration:**
   - Monitor certificate expiration dates
   - Alert 30 days before expiration
   - Automate renewal process

2. **Connection Logging:**
   - Log all mTLS connection attempts
   - Record certificate fingerprints
   - Alert on authentication failures

## Troubleshooting

### Client Certificate Not Found

**Symptom:** "Failed to load client certificate"

**Solutions:**
1. Verify file path is correct and absolute
2. Check file exists and is readable
3. Ensure certificate is in PEM format
4. Verify file permissions (should be readable by miner user)

### Private Key Password Issues

**Symptom:** "Failed to load private key" or "Bad password"

**Solutions:**
1. Verify password is correct
2. Check if key is actually password-protected
3. Try loading key manually: `openssl rsa -in client-key.pem -check`
4. Ensure password doesn't contain special characters that need escaping

### Server Rejects Client Certificate

**Symptom:** "Client certificate verification failed" or connection refused

**Solutions:**
1. Verify server is configured to accept client certificates
2. Check if server trusts your client CA
3. Ensure certificate is not expired (`openssl x509 -in client-cert.pem -noout -dates`)
4. Verify certificate chain is complete
5. Check server logs for specific error messages

### Certificate Validation Errors

**Symptom:** "Certificate chain validation failed"

**Solutions:**
1. Ensure CA certificate is included in `tls_ca_cert_path`
2. Verify certificate chain is complete (client cert → CA cert)
3. Check for certificate revocation
4. Ensure time synchronization (NTP) is correct

## Performance Considerations

### Handshake Overhead

**mTLS Handshake:**
- Additional ~5-15ms latency vs standard TLS
- Client certificate verification by server
- Minimal impact on long-lived connections

**Mitigation:**
- Use session resumption (TLS 1.3)
- Maintain long-lived connections
- Use connection pooling

### Key Size Impact

| Algorithm | Key Size | Handshake Time | Recommendation |
|-----------|----------|----------------|----------------|
| RSA | 2048-bit | ~10ms | Minimum |
| RSA | 4096-bit | ~40ms | Secure |
| ECDSA | 256-bit | ~2ms | **Recommended** |
| ECDSA | 384-bit | ~5ms | High Security |

**Recommendation:** Use ECDSA certificates for better performance.

## Integration with Existing Features

### Falcon Authentication

mTLS complements Falcon authentication:

**mTLS Layer:**
- Transport-level authentication
- Verifies network identity
- Prevents unauthorized connections

**Falcon Authentication:**
- Application-level authentication
- Verifies mining identity
- Ties to blockchain account

**Combined Benefits:**
- Defense in depth
- Multiple authentication factors
- Maximum security

### ChaCha20 Wrapper

Both mTLS and ChaCha20 can be used together:

**mTLS:** Encrypts entire connection
**ChaCha20:** Additional encryption for Falcon keys

This provides layered security for maximum protection.

## Migration Guide

### From Standard TLS to mTLS

**Step 1: Generate Certificates** (as shown above)

**Step 2: Update Configuration**

```json
{
    "enable_tls": true,
    "tls_client_cert_path": "/etc/nexus/certs/client-cert.pem",
    "tls_client_key_path": "/etc/nexus/certs/client-key.pem"
}
```

**Step 3: Test Connection**

```bash
# Test with OpenSSL
openssl s_client -connect mining.pool.com:8323 \
    -cert /etc/nexus/certs/client-cert.pem \
    -key /etc/nexus/certs/client-key.pem \
    -CAfile /etc/nexus/certs/ca-cert.pem
```

**Step 4: Monitor Logs**

Check miner logs for successful mTLS authentication:
```
[TLS] Loaded client certificate: /etc/nexus/certs/client-cert.pem
[TLS] Loaded client private key: /etc/nexus/certs/client-key.pem
[TLS] Client certificate configured for mutual TLS authentication
```

### Rollback Plan

If issues occur:
1. Remove `tls_client_cert_path` and `tls_client_key_path` from config
2. Restart miner
3. Connection falls back to standard TLS
4. Investigate and fix certificate issues

## Advanced Topics

### Certificate Pinning

For maximum security, pin specific certificates:

```cpp
// Custom verification callback
bool verify_callback(bool preverified, asio::ssl::verify_context& ctx) {
    // Get certificate
    X509* cert = X509_STORE_CTX_get_current_cert(ctx.native_handle());
    
    // Calculate fingerprint
    std::string fingerprint = get_cert_fingerprint(cert);
    
    // Compare with pinned fingerprint
    return (fingerprint == "expected_fingerprint_sha256");
}
```

### Hardware Security Modules (HSM)

For enterprise deployments, store private keys in HSM:
- Prevents key extraction
- FIPS 140-2 compliance
- Requires PKCS#11 integration

### Certificate Automation

Automate certificate lifecycle:
- Use cert-manager (Kubernetes)
- Implement auto-renewal
- Centralized certificate management

## References

- RFC 5246: TLS 1.2
- RFC 8446: TLS 1.3
- RFC 5280: X.509 Certificates
- OpenSSL Documentation: https://www.openssl.org/docs/

---

**Version**: 1.0  
**Last Updated**: 2025-12-06  
**Compatibility**: NexusMiner 1.5+, OpenSSL 1.1.1+/3.0+
