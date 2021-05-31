#ifndef NEXUSMINER_CONFIG_HPP
#define NEXUSMINER_CONFIG_HPP

#include <string>

namespace nexusminer
{
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

private:

	std::string  m_wallet_ip;
	std::uint16_t m_port;
	Mining_mode	 m_mining_mode;
	bool		 m_use_pool;
	std::uint32_t m_min_share;
	std::string  m_logfile;
	std::uint16_t m_connection_retry_interval;
	std::uint16_t m_print_statistics_interval;

};

}
#endif 