#ifndef NEXUSMINER_CONFIG_HPP
#define NEXUSMINER_CONFIG_HPP

#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "config/worker_config.hpp"
#include "config/stats_printer_config.hpp"
#include "config/pool.hpp"
#include "config/types.hpp"

namespace spdlog { class logger; }
namespace nexusminer
{
namespace config
{
#define CONFIG_VERSION 1

class Config
{
public:

	explicit Config(std::shared_ptr<spdlog::logger> logger);

	bool read_config(std::string const& miner_config_file);

	std::uint16_t get_version() const { return m_version; }
	std::string const& get_wallet_ip() const { return m_wallet_ip; }
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
	Pool const& get_pool_config() const { return m_pool_config; }

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
	Pool 		 m_pool_config; 
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

};
}
}
#endif 