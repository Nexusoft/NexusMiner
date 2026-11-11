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
	// Number of CPU threads to use for mining (default: 1)
	// Multi-threading is supported for hash mining with nonce partitioning
	std::uint16_t m_threads{1};
	
	// CPU affinity mask for thread pinning (default: 0, no affinity)
	// When 0, affinity may be automatically set based on hyperthreading/efficiency_cores settings
	std::uint64_t m_affinity_mask{0};
	
	// CPU power controls
	std::uint8_t m_priority_level{2};             // 0=low, 1=below_normal, 2=normal, 3=above_normal, 4=high
	std::uint8_t m_power_limit_percent{100};      // 50-100%
	bool m_enable_hyperthreading{true};           // Use SMT/HT threads (auto-disables if false)
	bool m_enable_efficiency_cores{true};         // For hybrid CPUs (P-cores/E-cores)
	std::uint32_t m_target_hashrate{0};           // 0=max

	// Stone 3: prime-channel mining backend selector.
	// "workers" (default) — today's N independent single-threaded Worker_prime
	//                       instances, each with its own Sieve and segment cursor.
	// "engine"            — reserved for the upcoming PrimeMiningEngine
	//                       (Option 3).  Currently logs and falls back to "workers".
	std::string m_engine_mode{"workers"};
};

struct Worker_config_fpga
{
	std::string serial_port{};

};

struct Worker_config_gpu
{
	std::uint8_t m_device{0};
	
	// NEW: GPU power controls
	std::uint8_t m_power_limit_percent{100};      // 50-100%
	std::int16_t m_core_clock_offset{0};          // MHz offset
	std::int16_t m_memory_clock_offset{0};        // MHz offset
	std::uint8_t m_fan_speed_percent{0};          // 0=auto, 1-100%
	std::uint32_t m_target_hashrate{0};           // 0=max
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