#ifndef NEXUSMINER_CONFIG_HPP
#define NEXUSMINER_CONFIG_HPP

#include <string>
#include <vector>
#include <variant>
#include "json/json.hpp"

namespace nexusminer
{
struct Worker_config_cpu
{

};

struct Worker_config_fpga
{
	std::string serial_port{};

};

struct Worker_config_gpu
{

};

class Worker_config
{
public:

	enum Mode
	{
		CPU = 0,
		FPGA,
		GPU
	};

	std::string m_id{};
	std::uint16_t m_internal_id{0U};
	Mode m_mode{Mode::CPU};
	std::variant<Worker_config_cpu, Worker_config_fpga, Worker_config_gpu>
		m_worker_mode;
};

class Config
{
public:

	enum Mining_mode
	{
		PRIME = 0,
		HASH = 1
	};

	Config();

	bool read_config();
	void print_config() const;

	std::string const& get_wallet_ip() const { return m_wallet_ip; }
	std::uint16_t get_port() const { return m_port; }
	Mining_mode get_mining_mode() const { return m_mining_mode; }
	bool get_use_bool() const { return m_use_pool; }
	std::uint32_t get_min_share() const { return m_min_share; }
	std::string const& get_logfile() const { return m_logfile; }
	std::uint16_t get_connection_retry_interval() const { return m_connection_retry_interval; }
	std::uint16_t get_print_statistics_interval() const { return m_print_statistics_interval; }
	std::uint16_t get_height_interval() const { return m_get_height_interval; }
	std::vector<Worker_config>& get_worker_config() { return m_worker_config; }

private:

	bool read_worker_config(nlohmann::json& j);

	std::string  m_wallet_ip;
	std::uint16_t m_port;
	Mining_mode	 m_mining_mode;
	bool		 m_use_pool;
	std::uint32_t m_min_share;
	std::string  m_logfile;

	// workers
	std::vector<Worker_config> m_worker_config;

	// advanced configs
	std::uint16_t m_connection_retry_interval;
	std::uint16_t m_print_statistics_interval;
	std::uint16_t m_get_height_interval;

};

}
#endif 