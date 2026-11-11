#ifndef NEXUSMINER_CONFIG_HPP
#define NEXUSMINER_CONFIG_HPP

#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "config/worker_config.hpp"
#include "config/stats_printer_config.hpp"
#include "config/types.hpp"
#include "protocol_lane.hpp"

namespace spdlog { class logger; }
namespace nexusminer
{
namespace config
{
#define CONFIG_VERSION 1

// Tritium GenesisHash validation constant
constexpr size_t TRITIUM_GENESIS_HEX_LENGTH = 64;  // 32 bytes as hex = 64 chars

// Mining configuration for stateless mining (MINER_SET_REWARD protocol)
struct MiningConfig
{
    std::string m_reward_address;  // NXS account address for mining rewards
};

class Config
{
	friend class TomlConfig;  // Allow TOML parser to use setters
	
public:

	explicit Config(std::shared_ptr<spdlog::logger> logger);

	bool read_config(std::string const& miner_config_file);

	std::uint16_t get_version() const { return m_version; }
	std::string const& get_wallet_ip() const { return m_wallet_ip; }
	// Phase 2: port refers to the stateless miner LLP port (miningport in nexus.conf)
	// Default is 8323 to match LLL-TAO's default miningport
	std::uint16_t get_port() const { return m_port; }
	std::string const& get_local_ip() const { return m_local_ip; }
	Mining_mode get_mining_mode() const { return m_mining_mode; }
	std::uint8_t get_log_level() const { return m_log_level; }
	std::string const& get_logfile() const { return m_logfile; }
	std::uint16_t get_connection_retry_interval() const { return m_connection_retry_interval; }
	std::uint16_t get_print_statistics_interval() const { return m_print_statistics_interval; }
	std::uint16_t get_height_interval() const { return m_get_height_interval; }
	std::uint16_t get_ping_interval() const { return m_ping_interval; }
	std::vector<Worker_config>& get_worker_config() { return m_worker_config; }
	std::vector<Stats_printer_config>& get_stats_printer_config() { return m_stats_printer_config; }
	std::string const& get_miner_falcon_pubkey() const { return m_miner_falcon_pubkey; }
	std::string const& get_miner_falcon_privkey() const { return m_miner_falcon_privkey; }
	bool has_miner_falcon_keys() const { return !m_miner_falcon_pubkey.empty() && !m_miner_falcon_privkey.empty(); }
	bool get_enable_disposable_falcon() const { return m_enable_disposable_falcon; }
	// Legacy method name (deprecated - kept for backward compatibility)
	bool get_enable_block_signing() const { return m_enable_disposable_falcon; }
	
	// Tritium GenesisHash and session management
	std::string const& get_tritium_genesis() const { return m_tritium_genesis; }
	bool has_tritium_genesis() const { return !m_tritium_genesis.empty(); }
	std::uint16_t get_keepalive_interval() const { return m_keepalive_interval; }
	bool get_enable_chacha20_wrapping() const { return m_enable_chacha20_wrapping; }
	bool is_localhost_mining() const;
	
	// TLS/HTTPS configuration
	bool get_enable_tls() const { return m_enable_tls; }
	// SSL mining port (0 = use plaintext port, non-zero = connect to this port over TLS)
	std::uint16_t get_ssl_port() const { return m_ssl_port; }
	std::string const& get_tls_ca_cert_path() const { return m_tls_ca_cert_path; }
	bool get_tls_verify_peer() const { return m_tls_verify_peer; }
	std::string const& get_tls_server_name() const { return m_tls_server_name; }
	
	// Mutual TLS (client certificate) configuration
	bool has_tls_client_certificate() const { return !m_tls_client_cert_path.empty() && !m_tls_client_key_path.empty(); }
	std::string const& get_tls_client_cert_path() const { return m_tls_client_cert_path; }
	std::string const& get_tls_client_key_path() const { return m_tls_client_key_path; }
	std::string const& get_tls_client_key_password() const { return m_tls_client_key_password; }
	
	// Mining configuration for stateless mining (MINER_SET_REWARD protocol)
	MiningConfig const& get_mining() const { return m_mining; }
	bool has_reward_address() const { return !m_mining.m_reward_address.empty(); }
	std::string const& get_reward_address() const { return m_mining.m_reward_address; }
	
	// SIM Link / failover flags retained for lane-health + second-node failover wiring.
	bool get_enable_sim_link() const { return m_enable_sim_link; }
	/// Returns the opposite-lane companion port for legacy compatibility helpers only.
	std::uint16_t get_secondary_port() const
	{
		return (m_port == ProtocolPorts::STATELESS_PORT) ? ProtocolPorts::LEGACY_PORT : ProtocolPorts::STATELESS_PORT;
	}

	// GET_BLOCK miner-side rate limit (milliseconds, default 2000)
	uint32_t get_get_block_interval_ms() const { return m_get_block_interval_ms; }

	// Colin AI diagnostic agent
	bool get_colin_enabled() const { return m_colin_enabled; }
	uint32_t get_colin_report_interval_seconds() const { return m_colin_report_interval_seconds; }

	// Failover node configuration
	std::string const& get_failover_wallet_ip() const { return m_failover_wallet_ip; }
	std::uint16_t get_failover_port() const { return m_failover_port; }
	uint32_t get_failover_max_retries() const { return m_failover_max_retries; }
	bool has_failover() const { return !m_failover_wallet_ip.empty(); }

	// Setters for TOML parser
	void set_wallet_ip(const std::string& ip) { m_wallet_ip = ip; }
	void set_port(std::uint16_t port) { m_port = port; }
	void set_local_ip(const std::string& ip) { m_local_ip = ip; }
	void set_mining_mode(Mining_mode mode) { m_mining_mode = mode; }
	void set_tritium_genesis(const std::string& genesis) { m_tritium_genesis = genesis; }
	void set_reward_address(const std::string& address) { m_mining.m_reward_address = address; }
	void set_worker_count(std::uint32_t count);
	void set_keepalive_interval(std::uint16_t interval) { m_keepalive_interval = interval; }
	void set_log_level(std::uint8_t level) { m_log_level = level; }
	void set_logfile(const std::string& logfile) { m_logfile = logfile; }
	
	// Advanced timing setters
	void set_connection_retry_interval(std::uint16_t interval) { m_connection_retry_interval = interval; }
	void set_print_statistics_interval(std::uint16_t interval) { m_print_statistics_interval = interval; }
	void set_ping_interval(std::uint16_t interval) { m_ping_interval = interval; }
	void set_get_height_interval(std::uint16_t interval) { m_get_height_interval = interval; }
	
	// Falcon authentication setters
	void set_miner_falcon_pubkey(const std::string& pubkey) { m_miner_falcon_pubkey = pubkey; }
	void set_miner_falcon_privkey(const std::string& privkey) { m_miner_falcon_privkey = privkey; }
	void set_enable_disposable_falcon(bool enable) { m_enable_disposable_falcon = enable; }
	// Legacy method name (deprecated - kept for backward compatibility)
	void set_enable_block_signing(bool enable) { m_enable_disposable_falcon = enable; }
	
	// ChaCha20 and TLS setters
	void set_enable_chacha20_wrapping(bool enable) { m_enable_chacha20_wrapping = enable; }
	void set_enable_tls(bool enable) { m_enable_tls = enable; }
	// Dedicated TLS mining port (0 = disabled/use plaintext port)
	void set_ssl_port(std::uint16_t port) { m_ssl_port = port; }
	void set_enable_sim_link(bool enable) { m_enable_sim_link = enable; }
	void set_get_block_interval_ms(uint32_t ms) { m_get_block_interval_ms = ms; }
	void set_colin_enabled(bool enabled) { m_colin_enabled = enabled; }
	void set_colin_report_interval_seconds(uint32_t secs) { m_colin_report_interval_seconds = secs; }
	void set_failover_wallet_ip(const std::string& ip) { m_failover_wallet_ip = ip; }
	void set_failover_port(std::uint16_t port) { m_failover_port = port; }
	void set_failover_max_retries(uint32_t n) { m_failover_max_retries = n; }
	void set_tls_ca_cert_path(const std::string& path) { m_tls_ca_cert_path = path; }
	void set_tls_verify_peer(bool verify) { m_tls_verify_peer = verify; }
	void set_tls_server_name(const std::string& name) { m_tls_server_name = name; }
	void set_tls_client_cert_path(const std::string& path) { m_tls_client_cert_path = path; }
	void set_tls_client_key_path(const std::string& path) { m_tls_client_key_path = path; }
	void set_tls_client_key_password(const std::string& password) { m_tls_client_key_password = password; }

private:

	bool read_stats_printer_config(nlohmann::json& j);
	bool read_worker_config(nlohmann::json& j);
	void print_global_config() const;
	void print_worker_config() const;

	std::shared_ptr<spdlog::logger> m_logger;
	std::uint16_t m_version;
	std::string  m_wallet_ip;
	std::uint16_t m_port;
	std::string m_local_ip;
	Mining_mode	 m_mining_mode;
	std::uint8_t m_log_level;
	std::string  m_logfile;

	// stats printers
	std::vector<Stats_printer_config> m_stats_printer_config;

	// workers
	std::vector<Worker_config> m_worker_config;

	// advanced configs
	std::uint16_t m_connection_retry_interval;
	std::uint16_t m_print_statistics_interval;
	std::uint16_t m_get_height_interval;
	std::uint16_t m_ping_interval;

	// Falcon miner authentication keys (optional)
	std::string m_miner_falcon_pubkey;
	std::string m_miner_falcon_privkey;
	
	// Unified Falcon Signature Protocol options
	bool m_enable_disposable_falcon;  // Disposable Falcon signing (ALWAYS ON - core protocol, 0 blockchain overhead)
	
	// Tritium GenesisHash and adaptive cache management (Phase 2 enhancement)
	std::string m_tritium_genesis;  // Tritium account genesis hash (32 bytes hex) for reward binding
	std::uint16_t m_keepalive_interval;  // Keep-alive ping interval in hours (default: 24)
	bool m_enable_chacha20_wrapping;  // Enable ChaCha20 wrapping of Falcon pubkey (auto for remote, optional for localhost)
	
	// TLS/HTTPS configuration (auto-enabled for remote connections)
	bool m_enable_tls;  // Enable TLS/SSL for remote connections (default: auto-detect)
	std::uint16_t m_ssl_port;  // Dedicated TLS mining port (0 = disabled/use plaintext port)
	std::string m_tls_ca_cert_path;  // Path to CA certificate bundle (empty = system default)
	bool m_tls_verify_peer;  // Verify peer certificate (default: true)
	std::string m_tls_server_name;  // Server name for SNI and verification (default: wallet_ip)
	
	// Mutual TLS (client certificate) configuration
	std::string m_tls_client_cert_path;  // Path to client certificate (PEM format)
	std::string m_tls_client_key_path;  // Path to client private key (PEM format)
	std::string m_tls_client_key_password;  // Password for client private key (optional)
	
	// Mining configuration for stateless mining (MINER_SET_REWARD protocol)
	MiningConfig m_mining;
	
	// SIM Link: dual-lane (stateless + legacy) simultaneous connection (default: enabled)
	bool m_enable_sim_link;

	// GET_BLOCK miner-side rate limit in milliseconds (default: 2500)
	uint32_t m_get_block_interval_ms;

	// Colin AI diagnostic agent configuration
	bool m_colin_enabled;
	uint32_t m_colin_report_interval_seconds;

	// Failover node configuration (optional, for cluster HA)
	std::string  m_failover_wallet_ip;          // empty = disabled
	std::uint16_t m_failover_port{0};           // 0 = use same port as primary
	uint32_t m_failover_max_retries{5};         // switch after this many consecutive primary failures

};
}
}
#endif 
