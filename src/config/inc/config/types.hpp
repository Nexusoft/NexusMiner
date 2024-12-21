#ifndef NEXUSMINER_CONFIG_TYPES_HPP
#define NEXUSMINER_CONFIG_TYPES_HPP

#include <string>
#include <variant>
#include <cstdint>

namespace nexusminer
{
namespace config
{
	enum class Mining_mode : uint8_t
	{
		PRIME = 0,
		HASH = 1
	};

    enum class Stats_printer_mode : int8_t
	{
		CONSOLE = 0,
		FILE
	};

    enum class Worker_mode : uint8_t
	{
		CPU = 0,
		FPGA,
		GPU
	};
}
}
#endif
