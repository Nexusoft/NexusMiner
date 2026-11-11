#ifndef NEXUSMINER_MINER_KEYS_HPP
#define NEXUSMINER_MINER_KEYS_HPP

#include <vector>
#include <string>
#include <cstdint>

namespace nexusminer
{
namespace keys
{

/**
 * @brief Generate a Falcon-1024 keypair for miner authentication
 * 
 * This function generates a quantum-resistant Falcon-1024 keypair suitable
 * for miner authentication with LLL-TAO nodes. Falcon-1024 provides 256-bit
 * quantum security (maximum protection) and is the default/recommended version.
 * 
 * @param[out] pubkey Generated public key (1793 bytes for Falcon-1024)
 * @param[out] privkey Generated private key (2305 bytes for Falcon-1024)
 * @return true if generation succeeded, false otherwise
 */
bool generate_falcon_keypair(std::vector<uint8_t>& pubkey, std::vector<uint8_t>& privkey);

/**
 * @brief Sign data using Falcon private key
 * 
 * @param privkey The Falcon private key
 * @param data The data to sign
 * @param[out] signature The resulting signature
 * @return true if signing succeeded, false otherwise
 */
bool falcon_sign(const std::vector<uint8_t>& privkey, 
                 const std::vector<uint8_t>& data, 
                 std::vector<uint8_t>& signature);

/**
 * @brief Verify a Falcon signature
 * 
 * @param pubkey The Falcon public key
 * @param data The data that was signed
 * @param signature The signature to verify
 * @return true if signature is valid, false otherwise
 */
bool falcon_verify(const std::vector<uint8_t>& pubkey,
                   const std::vector<uint8_t>& data,
                   const std::vector<uint8_t>& signature);

/**
 * @brief Convert binary data to hexadecimal string
 * 
 * @param data The binary data
 * @return Hexadecimal string representation
 */
std::string to_hex(const std::vector<uint8_t>& data);

/**
 * @brief Convert hexadecimal string to binary data
 * 
 * @param hex The hexadecimal string
 * @param[out] data The resulting binary data
 * @return true if conversion succeeded, false otherwise
 */
bool from_hex(const std::string& hex, std::vector<uint8_t>& data);

/**
 * @brief Generate a Falcon miner configuration file
 * 
 * Creates a complete falconminer.conf file with all mandatory fields populated
 * for SOLO PRIME mining with Falcon authentication.
 * 
 * @param config_filename The filename to write the config to (default: "falconminer.conf")
 * @param include_privkey Whether to include the private key in the config file (default: false)
 * @param miner_id The miner ID to use in the Falcon config (default: "default")
 * @return true if config generation succeeded, false otherwise
 */
bool create_falcon_config(const std::string& config_filename = "falconminer.conf",
                         bool include_privkey = false,
                         const std::string& miner_id = "default");

} // namespace keys
} // namespace nexusminer

#endif
