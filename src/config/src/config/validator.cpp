#include "config/validator.hpp"
#include "config/types.hpp"
#include "json/json.hpp"

#include <fstream>
#include <sstream>
#include <iostream>

using json = nlohmann::json;

namespace nexusminer
{
namespace config
{
Validator::Validator()
: m_mandatory_fields{}
, m_optional_fields{}
{
}

bool Validator::check(std::string const& config_file)
{
    std::ifstream config(config_file);
    if (!config.is_open())
    {
        std::cerr << "Unable to read " << config_file << std::endl;
        return false;
    }

    try
    {
        json j = json::parse(config);
        std::string mining_mode = "hash";
        // check mandatory fields;
        if (j.count("wallet_ip") == 0)
        {
            m_mandatory_fields.push_back(Validator_error{"wallet_ip", ""});
        }

        if (j.count("port") == 0)
        {
            m_mandatory_fields.push_back(Validator_error{"port", ""});
        }
        else
        {
            if(!j.at("port").is_number())
            {
                m_mandatory_fields.push_back(Validator_error{"port", "Not a number"});
            }
        }

        if (j.count("mining_mode") == 0)
        {
            m_mandatory_fields.push_back(Validator_error{"mining_mode", ""});
        }
        else
        {
            // check content of mining_mode
            mining_mode = j["mining_mode"];
            std::for_each(mining_mode.begin(), mining_mode.end(), [](char & c) {
                c = ::tolower(c);
    	    });
            if(mining_mode != "prime" && mining_mode != "hash")
            {
                m_mandatory_fields.push_back(Validator_error{"mining_mode", "Not 'prime' or 'hash'"});
            }
        }

        if(j.count("use_pool") != 0)
        {
            if(!j.at("use_pool").is_boolean())
            {
                m_optional_fields.push_back(Validator_error{"use_pool", "Not a boolean"});
            }

            if(j.count("pool")["username"] == 0)
            {
                m_mandatory_fields.push_back(Validator_error{"pool/username", ""});
            }
        }

        //stats printers
        for (auto& stats_printers_json : j["stats_printers"])
        {
            for(auto& stats_printer_config_json : stats_printers_json)
            {
                auto stats_printer_mode = stats_printer_config_json["mode"];
                if(!stats_printer_mode.is_string())
                {
                    m_optional_fields.push_back(Validator_error{"stats_printers/stats_printer/mode", "Not a string"});
                    break;
                }

                if(stats_printer_mode != "console" && stats_printer_mode != "file")
                {
                    m_optional_fields.push_back(Validator_error{"stats_printers/stats_printer/mode", "Not 'console' or 'prime"});
                    break;
                }
            }
        }

        // workers
        if(j.count("workers") == 0)
        {
            m_mandatory_fields.push_back(Validator_error{"workers", ""});
        }
        else
        {
            for (auto& workers_json : j["workers"])
            {
                for(auto& worker_config_json : workers_json)
                {
                    if(!worker_config_json["id"].is_string())
                    {
                        m_mandatory_fields.push_back(Validator_error{"workers/worker/id", "Not a string"});
                        break;
                    }

                    auto& worker_mode_json = worker_config_json["mode"];
                    if(worker_mode_json["hardware"] != "cpu" &&
                    worker_mode_json["hardware"] != "gpu" &&
                    worker_mode_json["hardware"] != "fpga")
                    {
                        m_mandatory_fields.push_back(Validator_error{"workers/worker/mode/hardware", "Not 'cpu', 'gpu' or 'fpga'"});
                        break;
                    }

                    if(worker_mode_json["hardware"] == "fpga")
                    {
                        if(worker_mode_json.count("serial_port") == 0)
                        {
                            m_mandatory_fields.push_back(Validator_error{"workers/worker/mode/serial_port", ""});
                            break;
                        }

                        if (mining_mode == "prime")
                        {
                            m_mandatory_fields.push_back(Validator_error{ "workers/worker/mode/hardware", "FPGA is not supported for PRIME mining" });
                            break;
                        }
                    }

                    if (worker_mode_json["hardware"] == "gpu")
                    {
                        if (mining_mode == "prime")
                        {
                            m_mandatory_fields.push_back(Validator_error{ "workers/worker/mode/hardware", "GPU is currently not supported for PRIME mining" });
                            break;
                        }

                        if (worker_mode_json.count("device") == 0)
                        {
                            m_mandatory_fields.push_back(Validator_error{ "workers/worker/mode/device", "" });
                        }
                        else
                        {
                            if (!worker_mode_json["device"].is_number())
                            {
                                m_mandatory_fields.push_back(Validator_error{ "workers/worker/mode/device", "Not a number" });
                            }
                        }
                    }
                }
            }
        }

        //advanced config
		if (j.count("connection_retry_interval") != 0)
		{
            if(!j.at("connection_retry_interval").is_number())
            {
                m_optional_fields.push_back(Validator_error{"connection_retry_interval", "Not a number"});
            }
		}
		if (j.count("print_statistics_interval") != 0)
		{
            if(!j.at("print_statistics_interval").is_number())
            {
                m_optional_fields.push_back(Validator_error{"print_statistics_interval", "Not a number"});
            }
		}
		if (j.count("get_height_interval") != 0)
		{
                        if(!j.at("get_height_interval").is_number())
            {
                m_optional_fields.push_back(Validator_error{"get_height_interval", "Not a number"});
            }
		}
    }
    catch(const std::exception& e)
    {
        std::cerr << "Incomplete json file" <<std::endl;
        std::cerr << e.what();
        return false;
    }

    if(m_mandatory_fields.empty() && m_optional_fields.empty())
    {
        return true;
    }
    else
    {
        return false;
    }
}

std::string Validator::get_check_result() const
{
    std::stringstream result;
    result << "Mandatory fields errors: " << m_mandatory_fields.size() << std::endl;
    result << "Optional fields errors: " << m_optional_fields.size() << std::endl;

    if(!m_mandatory_fields.empty())
    {
        result << "--- Mandatory fields ---" << std::endl;
    }
    for(auto const& error : m_mandatory_fields)
    {
        result << "[" << error.m_field << "]\t" << (error.m_message.empty() ? " is missing" : error.m_message) << std::endl; 
    }

    if(!m_optional_fields.empty())
    {
        result << "--- Optional fields ---" << std::endl;
    }
    for(auto const& error : m_optional_fields)
    {
        result << "[" << error.m_field << "]\t" << (error.m_message.empty() ? " is missing" : error.m_message) << std::endl; 
    }

    return result.str();
}

}
}
