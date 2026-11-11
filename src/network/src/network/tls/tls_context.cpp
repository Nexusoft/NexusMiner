#include "network/tls/tls_context.hpp"
#include <openssl/ssl.h>
#include <openssl/err.h>

namespace nexusminer {
namespace network {
namespace tls {

TlsContext::TlsContext(Mode mode)
    : m_context(mode == Mode::CLIENT ? asio::ssl::context::tlsv12_client : asio::ssl::context::tlsv12_server)
    , m_mode(mode)
    , m_configured(false)
    , m_logger(spdlog::get("logger"))
{
    if (!m_logger) {
        m_logger = spdlog::default_logger();
    }
    
    apply_secure_defaults();
    
    m_logger->info("[TLS] Context initialized in {} mode", 
                  mode == Mode::CLIENT ? "CLIENT" : "SERVER");
}

TlsContext::~TlsContext()
{
    m_logger->debug("[TLS] Context destroyed");
}

void TlsContext::apply_secure_defaults()
{
    // Disable insecure SSL/TLS versions
    // This enforces TLS 1.2 and TLS 1.3 only
    m_context.set_options(
        asio::ssl::context::default_workarounds |
        asio::ssl::context::no_sslv2 |
        asio::ssl::context::no_sslv3 |
        asio::ssl::context::no_tlsv1 |
        asio::ssl::context::no_tlsv1_1 |
        asio::ssl::context::single_dh_use
    );
    
    m_logger->debug("[TLS] Secure defaults applied (TLS 1.2+ only)");
}

std::string TlsContext::get_default_cipher_list()
{
    // Prioritize modern, secure cipher suites
    // TLS 1.3 ciphers are automatically included
    // This list focuses on:
    // 1. Forward secrecy (ECDHE)
    // 2. Strong encryption (ChaCha20-Poly1305, AES-256-GCM)
    // 3. Modern algorithms (no RC4, no MD5, no DES)
    
    return "TLS_CHACHA20_POLY1305_SHA256:"
           "TLS_AES_256_GCM_SHA384:"
           "TLS_AES_128_GCM_SHA256:"
           "ECDHE-ECDSA-CHACHA20-POLY1305:"
           "ECDHE-RSA-CHACHA20-POLY1305:"
           "ECDHE-ECDSA-AES256-GCM-SHA384:"
           "ECDHE-RSA-AES256-GCM-SHA384:"
           "ECDHE-ECDSA-AES128-GCM-SHA256:"
           "ECDHE-RSA-AES128-GCM-SHA256";
}

bool TlsContext::set_cipher_list(const std::string& cipher_list)
{
    try {
        // For TLS 1.3 ciphersuites
        SSL_CTX* ctx = m_context.native_handle();
        if (SSL_CTX_set_ciphersuites(ctx, cipher_list.c_str()) != 1) {
            m_logger->warn("[TLS] Failed to set TLS 1.3 cipher suites");
        }
        
        // For TLS 1.2 and earlier cipher suites
        if (SSL_CTX_set_cipher_list(ctx, cipher_list.c_str()) != 1) {
            m_logger->error("[TLS] Failed to set cipher list");
            return false;
        }
        
        m_logger->info("[TLS] Cipher list configured");
        return true;
    }
    catch (const std::exception& e) {
        m_logger->error("[TLS] Exception setting cipher list: {}", e.what());
        return false;
    }
}

bool TlsContext::configure_client(const std::string& ca_cert_path, bool verify_peer)
{
    try {
        if (m_mode != Mode::CLIENT) {
            m_logger->error("[TLS] configure_client() called in SERVER mode");
            return false;
        }
        
        // Set verification mode
        if (verify_peer) {
            m_context.set_verify_mode(asio::ssl::verify_peer);
            m_logger->info("[TLS] Client mode: peer verification ENABLED");
        } else {
            m_context.set_verify_mode(asio::ssl::verify_none);
            m_logger->warn("[TLS] Client mode: peer verification DISABLED (insecure!)");
        }
        
        // Load CA certificates
        if (!ca_cert_path.empty()) {
            m_context.load_verify_file(ca_cert_path);
            m_logger->info("[TLS] Loaded CA certificates from: {}", ca_cert_path);
        } else {
            // Use system default CA bundle
            m_context.set_default_verify_paths();
            m_logger->info("[TLS] Using system default CA certificates");
        }
        
        // Set default cipher list
        if (!set_cipher_list(get_default_cipher_list())) {
            m_logger->error("[TLS] Failed to set default cipher list");
            return false;
        }
        
        m_configured = true;
        m_logger->info("[TLS] Client configuration complete");
        return true;
    }
    catch (const std::exception& e) {
        m_logger->error("[TLS] Exception configuring client: {}", e.what());
        return false;
    }
}

bool TlsContext::configure_client_certificate(const std::string& cert_path,
                                              const std::string& key_path,
                                              const std::string& password)
{
    try {
        if (m_mode != Mode::CLIENT) {
            m_logger->error("[TLS] configure_client_certificate() only available in CLIENT mode");
            return false;
        }
        
        // Set password callback if provided
        if (!password.empty()) {
            m_context.set_password_callback(
                [password](std::size_t, asio::ssl::context::password_purpose) {
                    return password;
                }
            );
        }
        
        // Load client certificate
        m_context.use_certificate_chain_file(cert_path);
        m_logger->info("[TLS] Loaded client certificate: {}", cert_path);
        
        // Load client private key
        m_context.use_private_key_file(key_path, asio::ssl::context::pem);
        m_logger->info("[TLS] Loaded client private key: {}", key_path);
        
        m_logger->info("[TLS] Client certificate configured for mutual TLS authentication");
        return true;
    }
    catch (const std::exception& e) {
        m_logger->error("[TLS] Exception configuring client certificate: {}", e.what());
        return false;
    }
}

bool TlsContext::configure_server(const std::string& cert_path,
                                  const std::string& key_path,
                                  const std::string& password)
{
    try {
        if (m_mode != Mode::SERVER) {
            m_logger->error("[TLS] configure_server() called in CLIENT mode");
            return false;
        }
        
        // Set password callback if provided
        if (!password.empty()) {
            m_context.set_password_callback(
                [password](std::size_t, asio::ssl::context::password_purpose) {
                    return password;
                }
            );
        }
        
        // Load server certificate
        m_context.use_certificate_chain_file(cert_path);
        m_logger->info("[TLS] Loaded server certificate: {}", cert_path);
        
        // Load private key
        m_context.use_private_key_file(key_path, asio::ssl::context::pem);
        m_logger->info("[TLS] Loaded private key: {}", key_path);
        
        // Set default cipher list
        if (!set_cipher_list(get_default_cipher_list())) {
            m_logger->error("[TLS] Failed to set default cipher list");
            return false;
        }
        
        m_configured = true;
        m_logger->info("[TLS] Server configuration complete");
        return true;
    }
    catch (const std::exception& e) {
        m_logger->error("[TLS] Exception configuring server: {}", e.what());
        return false;
    }
}

bool TlsContext::enable_hostname_verification(const std::string& hostname)
{
    try {
        if (m_mode != Mode::CLIENT) {
            m_logger->error("[TLS] Hostname verification only available in CLIENT mode");
            return false;
        }

        if (hostname.empty()) {
            m_logger->error("[TLS] Hostname verification requires a non-empty hostname");
            return false;
        }

        SSL_CTX* ctx = m_context.native_handle();

        // Configure the default X509_VERIFY_PARAM to check the peer
        // certificate's CN / SAN against the expected hostname.
        X509_VERIFY_PARAM* param = SSL_CTX_get0_param(ctx);
        if (!param) {
            m_logger->error("[TLS] Failed to get X509_VERIFY_PARAM from SSL_CTX");
            return false;
        }

        // Enable hostname checking (matches CN and SubjectAltName)
        X509_VERIFY_PARAM_set_hostflags(param, X509_CHECK_FLAG_NO_PARTIAL_WILDCARDS);
        if (X509_VERIFY_PARAM_set1_host(param, hostname.c_str(), hostname.size()) != 1) {
            m_logger->error("[TLS] Failed to set expected hostname '{}' for verification", hostname);
            return false;
        }

        // Ensure peer verification is enabled (hostname check is meaningless without it)
        m_context.set_verify_mode(asio::ssl::verify_peer);

        m_logger->info("[TLS] Hostname verification enabled for: {}", hostname);
        return true;
    }
    catch (const std::exception& e) {
        m_logger->error("[TLS] Exception enabling hostname verification: {}", e.what());
        return false;
    }
}

} // namespace tls
} // namespace network
} // namespace nexusminer
