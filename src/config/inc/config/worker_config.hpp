#ifndef NEXUSMINER_CONFIG_WORKER_CONFIG_HPP
#define NEXUSMINER_CONFIG_WORKER_CONFIG_HPP

#include <string>
#include <variant>
#include "config/types.hpp"

namespace nexusminer
{
namespace config
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

	std::string m_id{};
	std::uint16_t m_internal_id{0U};
	Worker_mode m_mode{Worker_mode::CPU};
	std::variant<Worker_config_cpu, Worker_config_fpga, Worker_config_gpu>
		m_worker_mode;
};

}
}
#endif