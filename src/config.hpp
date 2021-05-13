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
	std::uint32_t get_connection_threads() const { return m_connection_threads; }
	bool get_use_ddos() const { return m_use_ddos; }

	std::uint32_t get_min_share() const { return m_min_share; }

	std::string const& get_logfile() const { return m_logfile; }


private:

	std::string  m_wallet_ip;
	std::uint16_t m_port;
	Mining_mode	 m_mining_mode;
	std::uint32_t m_connection_threads;
	bool         m_use_ddos;
	int          m_r_score;
	int          m_c_score;
	std::uint32_t     m_min_share;
	std::string  m_logfile;

};

}


#endif 