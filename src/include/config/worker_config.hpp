#ifndef NEXUSMINER_WORKER_CONFIG_HPP
#define NEXUSMINER_WORKER_CONFIG_HPP

#include <string>
#include <variant>

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

}
#endif