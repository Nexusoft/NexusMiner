
#include "config.hpp"
#include "json/json.hpp"

using json = nlohmann::json;

#include <fstream>
#include <iostream>

namespace nexusminer
{
	Config::Config()
		: m_wallet_ip{ "127.0.0.1" }
		, m_port{ 9323 }
		, m_mining_mode{ Mining_mode::PRIME}
		, m_connection_threads{ 20 }
		, m_use_ddos{true}
		, m_r_score{20}
		, m_c_score{2}
		, m_min_share{ 40000000 }
		, m_logfile{""}		// no logfile usage, default
	{
	}

	void Config::print_config() const
	{
		std::cout << "Configuration: " << std::endl;
		std::cout << "-------------" << std::endl;
		std::cout << "Wallet IP: " << m_wallet_ip << std::endl;
		std::cout << "Port: " << m_port << std::endl;

		std::cout << "Mining Mode: " << m_mining_mode << std::endl;
		std::cout << "Connection Threads: " << m_connection_threads << std::endl;

		std::cout << "bDDOS: " << m_use_ddos << std::endl;;
		std::cout << "rScore: " << m_r_score << std::endl;
		std::cout << "cScore: " << m_c_score << std::endl;
		std::cout << "Min Share Diff: " << m_min_share << std::endl;

		std::cout << "Logfile: " << m_logfile << std::endl;

		std::cout << "-------------" << std::endl;

	}

	bool Config::read_config()
	{
		std::cout << "Reading config file miner.conf" << std::endl;

		std::ifstream config_file("miner.conf");
		if (!config_file.is_open())
		{
			std::cerr << "Unable to read miner.conf";
			return false;
		}

		json j = json::parse(config_file);
		j.at("wallet_ip").get_to(m_wallet_ip);
		j.at("port").get_to(m_port);
		j.at("mining_mode").get_to(m_mining_mode);
		
		j.at("connection_threads").get_to(m_connection_threads);
		j.at("enable_ddos").get_to(m_use_ddos);
		j.at("ddos_rscore").get_to(m_r_score);
		j.at("ddos_cscore").get_to(m_c_score);
		j.at("min_share").get_to(m_min_share);

		j.at("logfile").get_to(m_logfile);

		print_config();
		// TODO Need to add exception handling here and set bSuccess appropriately
		return true;
	}


} // end namespace