#ifndef NEXUSMINER_CONFIG_STATS_PRINTER_CONFIG_HPP
#define NEXUSMINER_CONFIG_STATS_PRINTER_CONFIG_HPP

#include <string>
#include <variant>
#include "config/types.hpp"

namespace nexusminer
{
namespace config
{

struct Stats_printer_config_console
{

};

struct Stats_printer_config_file
{
	std::string file_name{};
};

class Stats_printer_config
{
public:

	Stats_printer_mode m_mode{Stats_printer_mode::CONSOLE};
	std::variant<Stats_printer_config_console, Stats_printer_config_file>
		m_printer_mode;
};

}
}
#endif