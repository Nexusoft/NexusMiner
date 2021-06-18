#ifndef NEXUSMINER_CONFIG_HPP
#define NEXUSMINER_CONFIG_HPP

#include <string>
#include <vector>
#include "json/json.hpp"
#include "worker_config.hpp"
#include "stats_printer_config.hpp"
#include "pool.hpp"
#include "types.hpp"

namespace nexusminer
{
namespace config
{
class Config
{
public:

	Config();

	bool read_config(std::string const& miner_config_file);
	void print_config() const;

	std::string const& get_wallet_ip() const { return m_wallet_ip; }
	std::uint16_t get_port() const { return m_port; }
	std::string const& get_local_ip() const { return m_local_ip; }
	Mining_mode get_mining_mode() const { return m_mining_mode; }
	bool get_use_bool() const { return m_use_pool; }
	std::string const& get_logfile() const { return m_logfile; }
	std::uint16_t get_connection_retry_interval() const { return m_connection_retry_interval; }
	std::uint16_t get_print_statistics_interval() const { return m_print_statistics_interval; }
	std::uint16_t get_height_interval() const { return m_get_height_interval; }
	std::vector<Worker_config>& get_worker_config() { return m_worker_config; }
	std::vector<Stats_printer_config>& get_stats_printer_config() { return m_stats_printer_config; }
	Pool const& get_pool_config() const { return m_pool_config; }

private:

	bool read_stats_printer_config(nlohmann::json& j);
	bool read_worker_config(nlohmann::json& j);

	std::string  m_wallet_ip;
	std::uint16_t m_port;
	std::string m_local_ip;
	Mining_mode	 m_mining_mode;
	bool		 m_use_pool;
	Pool 		 m_pool_config; 
	std::string  m_logfile;

	// stats printers
	std::vector<Stats_printer_config> m_stats_printer_config;

	// workers
	std::vector<Worker_config> m_worker_config;

	// advanced configs
	std::uint16_t m_connection_retry_interval;
	std::uint16_t m_print_statistics_interval;
	std::uint16_t m_get_height_interval;

};
}
}
#endif 