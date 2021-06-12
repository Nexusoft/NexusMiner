
#include "config/config.hpp"

#include <fstream>
#include <iostream>

using json = nlohmann::json;

namespace nexusminer
{
namespace config
{
	Config::Config()
		: m_wallet_ip{ "127.0.0.1" }
		, m_port{ 9323 }
		, m_mining_mode{ Mining_mode::HASH}
		, m_use_pool{false}
		, m_min_share{ 40000000 }
		, m_logfile{""}		// no logfile usage, default
		, m_connection_retry_interval{5}
		, m_print_statistics_interval{5}
		, m_get_height_interval{2}
	{
	}

	void Config::print_config() const
	{
		std::cout << "Configuration: " << std::endl;
		std::cout << "-------------" << std::endl;
		std::cout << "Wallet IP: " << m_wallet_ip << std::endl;
		std::cout << "Port: " << m_port << std::endl;

		std::cout << "Mining Mode: " << ((m_mining_mode == Mining_mode::HASH) ? "HASH" : "PRIME") << std::endl;

		std::cout << "Connection Retry Interval: " << m_connection_retry_interval << std::endl;
		std::cout << "Print Statistics Interval: " << m_print_statistics_interval << std::endl;		
		std::cout << "Get Height Interval: " << m_get_height_interval << std::endl;		

		std::cout << "Pool: " << m_use_pool << std::endl;;
		std::cout << "Min Share Diff: " << m_min_share << std::endl;

		std::cout << "Logfile: " << m_logfile << std::endl;

		std::cout << "-------------" << std::endl;

	}

	bool Config::read_config(std::string const& miner_config_file)
	{
		std::string miner_config_tmp = miner_config_file.empty() ? "miner.conf" : miner_config_file;
		std::cout << "Reading config file " << miner_config_tmp << std::endl;

		std::ifstream config_file(miner_config_tmp);
		if (!config_file.is_open())
		{
			std::cerr << "Unable to read " << miner_config_tmp << std::endl;
			return false;
		}

		json j = json::parse(config_file);
		j.at("wallet_ip").get_to(m_wallet_ip);
		j.at("port").get_to(m_port);

		std::string mining_mode = j["mining_mode"];
		std::for_each(mining_mode.begin(), mining_mode.end(), [](char & c) {
        	c = ::tolower(c);
    	});

		if(mining_mode == "prime")
		{
			m_mining_mode = Mining_mode::PRIME;
		}
		else
		{
			m_mining_mode = Mining_mode::HASH;
		}

		j.at("use_pool").get_to(m_use_pool);
		j.at("min_share").get_to(m_min_share);

		// read worker config
		if(!read_stats_printer_config(j))
		{
			return false;
		}

		// read worker config
		if(!read_worker_config(j))
		{
			return false;
		}

		// advanced configs
		if (j.count("connection_retry_interval") != 0)
		{
			j.at("connection_retry_interval").get_to(m_connection_retry_interval);
		}
		if (j.count("print_statistics_interval") != 0)
		{
			j.at("print_statistics_interval").get_to(m_print_statistics_interval);
		}
		if (j.count("get_height_interval") != 0)
		{
			j.at("get_height_interval").get_to(m_get_height_interval);
		}

		j.at("logfile").get_to(m_logfile);

		print_config();
		// TODO Need to add exception handling here and set return value appropriately
		return true;
	}

	bool Config::read_stats_printer_config(nlohmann::json& j)
	{
		for (auto& stats_printers_json : j["stats_printers"])
		{
			for(auto& stats_printer_config_json : stats_printers_json)
			{
				Stats_printer_config stats_printer_config;
				auto stats_printer_mode = stats_printer_config_json["mode"];

				if(stats_printer_mode == "console")
				{
					stats_printer_config.m_mode = Stats_printer_mode::CONSOLE;
					stats_printer_config.m_printer_mode = Stats_printer_config_console{};
				}
				else if(stats_printer_mode == "file")
				{
					stats_printer_config.m_mode = Stats_printer_mode::FILE;
					stats_printer_config.m_printer_mode = Stats_printer_config_file{};
				}
				else
				{
					// invalid config
					return false;
				}

				m_stats_printer_config.push_back(stats_printer_config);		
			}
		}
		return true;	
	}

	bool Config::read_worker_config(nlohmann::json& j)
	{
		for (auto& workers_json : j["workers"])
		{
			for(auto& worker_config_json : workers_json)
			{
				Worker_config worker_config;
				worker_config.m_id = worker_config_json["id"];

				auto& worker_mode_json = worker_config_json["mode"];

				if(worker_mode_json["hardware"] == "cpu")
				{
					worker_config.m_mode = Worker_mode::CPU;
					worker_config.m_worker_mode = Worker_config_cpu{};
				}
				else if(worker_mode_json["hardware"] == "gpu")
				{
					worker_config.m_mode = Worker_mode::GPU;
					worker_config.m_worker_mode = Worker_config_gpu{};
				}
				else if(worker_mode_json["hardware"] == "fpga")
				{
					worker_config.m_mode = Worker_mode::FPGA;
					worker_config.m_worker_mode = Worker_config_fpga{worker_mode_json["serial_port"]};
				}
				else
				{
					// invalid config
					return false;
				}

				m_worker_config.push_back(worker_config);		
			}
		}
		return true;	
	}
}
}