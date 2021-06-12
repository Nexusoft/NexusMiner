#ifndef NEXUSMINER_STATS_PRINTER_CONFIG_HPP
#define NEXUSMINER_STATS_PRINTER_CONFIG_HPP

#include <string>
#include <variant>

namespace nexusminer
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

	enum Mode
	{
		CONSOLE = 0,
		FILE
	};

	Mode m_mode{Mode::CONSOLE};
	std::variant<Stats_printer_config_console, Stats_printer_config_file>
		m_printer_mode;
};

}
#endif