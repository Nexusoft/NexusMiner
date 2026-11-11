#pragma once
#ifndef NEXUSMINER_CONFIG_MINER_CONFIG_HPP
#define NEXUSMINER_CONFIG_MINER_CONFIG_HPP

#include <string>
#include <map>
#include <LLC/flkey.h>

namespace config {

/** Default configuration values - OPTIMIZED FOR SECURITY AND ECONOMICS */

// Falcon version (DEFAULT: 1024 for maximum quantum security) ✅
constexpr bool DEFAULT_FALCON1024 = true;

/** Miner configuration class 
 * 
 * Manages Falcon signature configuration for the stateless mining protocol.
 * 
 * Only Disposable Falcon signatures are used — block signing does not
 * store signatures on the blockchain (zero blockchain overhead).
 * 
 * Configuration File Format (miner.conf):
 * ```
 * # Falcon version (1=1024, 0=512)
 * falcon1024=1
 * 
 * # Falcon keys
 * [falcon]
 * version = 1024
 * pubkey = "hex_encoded_pubkey"
 * privkey = "hex_encoded_privkey"
 * ```
 */
class MinerConfig
{
private:
    bool m_bFalcon1024;
    std::string m_strConfigFile;
    
public:
    /** Constructor with defaults */
    MinerConfig()
    : m_bFalcon1024(DEFAULT_FALCON1024)  // ✅ Falcon-1024 by default
    , m_strConfigFile("miner.conf")
    {
    }
    
    /** Load configuration from file 
     * 
     * Reads miner.conf and parses:
     * - falcon1024 (default: 1)
     * - [falcon] section with version, pubkey, privkey
     * 
     * @param filename Path to configuration file
     * @return true if loaded successfully, false otherwise
     */
    bool Load(const std::string& filename);
    
    /** Save configuration to file 
     * 
     * Writes miner.conf with comments explaining defaults.
     * 
     * @param filename Path to configuration file
     * @return true if saved successfully, false otherwise
     */
    bool Save(const std::string& filename) const;
    
    /** Get Falcon version setting 
     * 
     * @return true for Falcon-1024 (default), false for Falcon-512
     */
    bool GetFalcon1024() const { return m_bFalcon1024; }
    
    /** Set Falcon version setting 
     * 
     * @param value true for Falcon-1024, false for Falcon-512
     */
    void SetFalcon1024(bool value) { m_bFalcon1024 = value; }
    
    /** Get Falcon version enum for key generation 
     * 
     * Converts boolean setting to LLC::FalconVersion enum.
     * 
     * @return FalconVersion::FALCON_1024 or FALCON_512
     */
    LLC::FalconVersion GetFalconVersion() const
    {
        return m_bFalcon1024 
            ? LLC::FalconVersion::FALCON_1024 
            : LLC::FalconVersion::FALCON_512;
    }
    
    /** Get expected signature size (CT) matching node requirements 
     * 
     * Returns constant-time signature size for protocol compliance.
     * Node REJECTS signatures that don't match this exact size!
     * 
     * @return 1577 bytes for Falcon-1024, 809 bytes for Falcon-512
     */
    size_t GetSignatureSize() const
    {
        return m_bFalcon1024 
            ? 1577  // Falcon-1024 CT
            : 809;  // Falcon-512 CT
    }
    
    /** Get public key size for packet construction 
     * 
     * Returns public key size for building MINER_AUTH packets.
     * Node detects version from this size!
     * 
     * @return 1793 bytes for Falcon-1024, 897 bytes for Falcon-512
     */
    size_t GetPublicKeySize() const
    {
        return m_bFalcon1024 
            ? 1793  // Falcon-1024
            : 897;  // Falcon-512
    }
    
    /** Get private key size 
     * 
     * @return 2305 bytes for Falcon-1024, 1281 bytes for Falcon-512
     */
    size_t GetPrivateKeySize() const
    {
        return m_bFalcon1024 
            ? 2305  // Falcon-1024
            : 1281; // Falcon-512
    }
    
};

} // namespace config

#endif // NEXUSMINER_CONFIG_MINER_CONFIG_HPP
