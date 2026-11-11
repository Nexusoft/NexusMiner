#ifndef NEXUSMINER_NETWORK_TLS_CONTEXT_HPP
#define NEXUSMINER_NETWORK_TLS_CONTEXT_HPP

#include <memory>
#include <string>
#include "asio/ssl.hpp"
#include "spdlog/spdlog.h"

namespace nexusminer {
namespace network {
namespace tls {

/**
 * @brief TLS/SSL Context Manager for secure connections
 * 
 * This class manages SSL/TLS context configuration for both client and server modes.
 * It provides secure default settings including:
 * - TLS 1.2 and TLS 1.3 support (no SSL v2/v3)
 * - Strong cipher suites (ECDHE, ChaCha20-Poly1305, AES-GCM)
 * - Certificate validation for client connections
 * - Optional certificate loading for server mode
 */
class TlsContext {
public:
    
    /**
     * @brief TLS mode enumeration
     */
    enum class Mode {
        CLIENT,  // Client mode (connects to remote server)
        SERVER   // Server mode (accepts incoming connections)
    };
    
    /**
     * @brief Constructor
     * @param mode TLS mode (client or server)
     */
    explicit TlsContext(Mode mode);
    
    /**
     * @brief Destructor
     */
    ~TlsContext();
    
    /**
     * @brief Get the underlying ASIO SSL context
     * @return Reference to SSL context
     */
    asio::ssl::context& get_context() { return m_context; }
    const asio::ssl::context& get_context() const { return m_context; }
    
    /**
     * @brief Configure client mode with certificate verification
     * 
     * @param ca_cert_path Path to CA certificate bundle (empty for system default)
     * @param verify_peer Enable peer certificate verification (default: true)
     * @return true on success, false on failure
     */
    bool configure_client(const std::string& ca_cert_path = "", bool verify_peer = true);
    
    /**
     * @brief Configure client certificate for mutual TLS authentication
     * 
     * Enables the client to authenticate itself to the server using a certificate.
     * This is used for mutual TLS (mTLS) where both client and server verify each other.
     * 
     * @param cert_path Path to client certificate (PEM format)
     * @param key_path Path to client private key (PEM format)
     * @param password Password for private key (optional)
     * @return true on success, false on failure
     */
    bool configure_client_certificate(const std::string& cert_path,
                                      const std::string& key_path,
                                      const std::string& password = "");
    
    /**
     * @brief Configure server mode with certificates
     * 
     * @param cert_path Path to server certificate
     * @param key_path Path to private key
     * @param password Password for private key (optional)
     * @return true on success, false on failure
     */
    bool configure_server(const std::string& cert_path,
                         const std::string& key_path,
                         const std::string& password = "");
    
    /**
     * @brief Set allowed cipher suites
     * 
     * Default cipher list prioritizes:
     * 1. TLS 1.3 ciphers (ChaCha20-Poly1305, AES-GCM)
     * 2. TLS 1.2 ciphers with forward secrecy (ECDHE)
     * 
     * @param cipher_list OpenSSL cipher list string
     * @return true on success, false on failure
     */
    bool set_cipher_list(const std::string& cipher_list);
    
    /**
     * @brief Enable hostname verification for client connections
     * 
     * @param hostname Expected hostname for verification
     * @return true on success, false on failure
     */
    bool enable_hostname_verification(const std::string& hostname);
    
    /**
     * @brief Get default secure cipher list
     * 
     * Recommended cipher suites:
     * - TLS 1.3: TLS_CHACHA20_POLY1305_SHA256, TLS_AES_256_GCM_SHA384
     * - TLS 1.2: ECDHE-ECDSA-CHACHA20-POLY1305, ECDHE-RSA-CHACHA20-POLY1305
     * - TLS 1.2: ECDHE-ECDSA-AES256-GCM-SHA384, ECDHE-RSA-AES256-GCM-SHA384
     * 
     * @return OpenSSL cipher list string
     */
    static std::string get_default_cipher_list();
    
    /**
     * @brief Check if TLS is properly configured
     * @return true if configured and ready
     */
    bool is_configured() const { return m_configured; }
    
    /**
     * @brief Get TLS mode
     * @return Current mode (client or server)
     */
    Mode get_mode() const { return m_mode; }

private:
    
    // SSL context
    asio::ssl::context m_context;
    
    // Mode and state
    Mode m_mode;
    bool m_configured;
    
    // Logger
    std::shared_ptr<spdlog::logger> m_logger;
    
    /**
     * @brief Apply secure default settings
     */
    void apply_secure_defaults();
};

} // namespace tls
} // namespace network
} // namespace nexusminer

#endif // NEXUSMINER_NETWORK_TLS_CONTEXT_HPP
