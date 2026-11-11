#ifndef NEXUSMINER_CPU_THREAD_UTILS_HPP
#define NEXUSMINER_CPU_THREAD_UTILS_HPP

#include <cstdint>
#include <vector>

namespace nexusminer
{
namespace cpu
{

/**
 * @brief Set the priority level of the current thread
 * @param level Priority level: 0=low, 1=below_normal, 2=normal, 3=above_normal, 4=high
 * @return true on success, false on failure
 */
bool set_thread_priority(std::uint8_t level);

/**
 * @brief Set CPU affinity mask for the current thread
 * @param mask Bitmask of allowed CPU cores (bit 0 = core 0, etc.)
 * @return true on success, false on failure
 */
bool set_thread_affinity(std::uint64_t mask);

/**
 * @brief Get the number of physical CPU cores (not including hyperthreading)
 * @return Number of physical cores, or 0 on error
 */
std::uint32_t get_physical_core_count();

/**
 * @brief Check if Simultaneous Multi-Threading (SMT/Hyperthreading) is enabled
 * @return true if SMT is enabled, false otherwise
 */
bool is_smt_enabled();

/**
 * @brief Check if the current core is an efficiency core (E-core) on Intel hybrid CPUs
 * @return true if running on E-core, false if P-core or non-hybrid CPU
 */
bool is_efficiency_core();

/**
 * @brief Get list of performance core (P-core) IDs on Intel hybrid CPUs
 * @return Vector of P-core IDs, empty if not a hybrid CPU
 */
std::vector<std::uint32_t> get_performance_cores();

/**
 * @brief Get list of efficiency core (E-core) IDs on Intel hybrid CPUs
 * @return Vector of E-core IDs, empty if not a hybrid CPU
 */
std::vector<std::uint32_t> get_efficiency_cores();

} // namespace cpu
} // namespace nexusminer

#endif // NEXUSMINER_CPU_THREAD_UTILS_HPP
