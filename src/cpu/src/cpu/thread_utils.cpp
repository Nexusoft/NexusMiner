#include "cpu/thread_utils.hpp"
#include <spdlog/spdlog.h>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#include <malloc.h>
#elif defined(__linux__)
#include <unistd.h>
#include <pthread.h>
#include <sys/resource.h>
#include <fstream>
#include <sstream>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace nexusminer
{
namespace cpu
{

static auto s_logger = spdlog::get("logger");

bool set_thread_priority(std::uint8_t level)
{
    if (level > 4)
    {
        if (s_logger) s_logger->warn("Invalid priority level {}, must be 0-4", level);
        return false;
    }

#ifdef _WIN32
    int priority_class;
    switch (level)
    {
        case 0: priority_class = THREAD_PRIORITY_LOWEST; break;
        case 1: priority_class = THREAD_PRIORITY_BELOW_NORMAL; break;
        case 2: priority_class = THREAD_PRIORITY_NORMAL; break;
        case 3: priority_class = THREAD_PRIORITY_ABOVE_NORMAL; break;
        case 4: priority_class = THREAD_PRIORITY_HIGHEST; break;
        default: priority_class = THREAD_PRIORITY_NORMAL; break;
    }
    
    if (SetThreadPriority(GetCurrentThread(), priority_class))
    {
        if (s_logger) s_logger->debug("Set thread priority to level {}", level);
        return true;
    }
    else
    {
        if (s_logger) s_logger->warn("Failed to set thread priority: {}", GetLastError());
        return false;
    }
#elif defined(__linux__)
    // On Linux, use nice values: -20 (highest) to 19 (lowest)
    // Map our 0-4 scale to nice values
    int nice_value;
    switch (level)
    {
        case 0: nice_value = 10; break;   // Low priority
        case 1: nice_value = 5; break;    // Below normal
        case 2: nice_value = 0; break;    // Normal
        case 3: nice_value = -5; break;   // Above normal
        case 4: nice_value = -10; break;  // High
        default: nice_value = 0; break;
    }
    
    // setpriority requires root for negative values, so we'll try but not fail
    if (setpriority(PRIO_PROCESS, 0, nice_value) == 0)
    {
        if (s_logger) s_logger->debug("Set thread priority to level {} (nice={})", level, nice_value);
        return true;
    }
    else
    {
        // Don't fail on EPERM (permission denied) for negative nice values
        if (errno == EPERM && nice_value < 0)
        {
            if (s_logger) s_logger->warn("Cannot set elevated priority (requires root), continuing with default priority");
            return true;  // Continue without error
        }
        if (s_logger) s_logger->warn("Failed to set thread priority: {}", strerror(errno));
        return false;
    }
#else
    if (s_logger) s_logger->warn("Thread priority setting not supported on this platform");
    return false;
#endif
}

bool set_thread_affinity(std::uint64_t mask)
{
    if (mask == 0)
    {
        // No affinity specified (0 = allow all cores), this is not an error
        if (s_logger) s_logger->debug("No CPU affinity constraint specified (mask=0), thread can run on any core");
        return true;
    }

#ifdef _WIN32
    DWORD_PTR affinity_mask = static_cast<DWORD_PTR>(mask);
    if (SetThreadAffinityMask(GetCurrentThread(), affinity_mask))
    {
        if (s_logger) s_logger->debug("Set thread affinity mask to 0x{:016x}", mask);
        return true;
    }
    else
    {
        if (s_logger) s_logger->warn("Failed to set thread affinity: {}", GetLastError());
        return false;
    }
#elif defined(__linux__)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    
    // Set bits in cpu_set_t based on mask
    for (int i = 0; i < 64; i++)
    {
        if (mask & (1ULL << i))
        {
            CPU_SET(i, &cpuset);
        }
    }
    
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) == 0)
    {
        if (s_logger) s_logger->debug("Set thread affinity mask to 0x{:016x}", mask);
        return true;
    }
    else
    {
        if (s_logger) s_logger->warn("Failed to set thread affinity: {}", strerror(errno));
        return false;
    }
#else
    if (s_logger) s_logger->warn("Thread affinity setting not supported on this platform");
    return false;
#endif
}

std::uint32_t get_physical_core_count()
{
#ifdef _WIN32
    DWORD length = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length);
    
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER)
    {
        return 0;
    }
    
    auto buffer = static_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(_alloca(length));
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, buffer, &length))
    {
        return 0;
    }
    
    std::uint32_t core_count = 0;
    DWORD offset = 0;
    while (offset < length)
    {
        auto current = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(
            reinterpret_cast<char*>(buffer) + offset);
        if (current->Relationship == RelationProcessorCore)
        {
            core_count++;
        }
        offset += current->Size;
    }
    
    return core_count;
#elif defined(__linux__)
    // Read from /sys/devices/system/cpu/cpu*/topology/thread_siblings_list
    std::uint32_t physical_cores = 0;
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    std::uint32_t processors = 0;
    std::uint32_t cores = 0;
    
    while (std::getline(cpuinfo, line))
    {
        if (line.find("processor") == 0) processors++;
        if (line.find("cpu cores") == 0)
        {
            std::istringstream iss(line);
            std::string key, colon;
            iss >> key >> key >> colon >> cores;
        }
    }
    
    // If we found core count, return it
    if (cores > 0)
    {
        return cores;
    }
    
    // Fallback: assume all processors are cores
    return processors > 0 ? processors : std::thread::hardware_concurrency();
#else
    return std::thread::hardware_concurrency();
#endif
}

bool is_smt_enabled()
{
    std::uint32_t logical_cores = std::thread::hardware_concurrency();
    std::uint32_t physical_cores = get_physical_core_count();
    
    if (physical_cores == 0)
    {
        return false;  // Unknown
    }
    
    return logical_cores > physical_cores;
}

bool is_efficiency_core()
{
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    // Check for Intel hybrid architecture using CPUID
    // NOTE: This function checks the core type of the calling thread at the moment
    // it executes. Due to thread migration in multithreaded environments, the thread
    // may move to a different core between calls. This function provides a snapshot
    // of the current core type but does not guarantee thread stays on that core.
    // Check if CPUID leaf 0x1A is supported (hybrid information)
    #ifdef _MSC_VER
    int cpuid_info[4];
    __cpuid(cpuid_info, 0);
    if (cpuid_info[0] < 0x1A)
    {
        return false;  // Leaf 0x1A not supported, not a hybrid CPU
    }
    
    __cpuid(cpuid_info, 0x1A);
    std::uint32_t core_type = (cpuid_info[0] >> 24) & 0xFF;
    #else
    std::uint32_t eax, ebx, ecx, edx;
    __get_cpuid(0, &eax, &ebx, &ecx, &edx);
    if (eax < 0x1A)
    {
        return false;  // Leaf 0x1A not supported, not a hybrid CPU
    }
    
    __get_cpuid(0x1A, &eax, &ebx, &ecx, &edx);
    std::uint32_t core_type = (eax >> 24) & 0xFF;
    #endif
    
    // Core type: 0x20 = Atom (E-core), 0x40 = Core (P-core)
    return core_type == 0x20;
#else
    return false;  // Not x86/x64 or not supported
#endif
}

std::vector<std::uint32_t> get_performance_cores()
{
    std::vector<std::uint32_t> p_cores;
    
    // This is a simplified implementation that makes assumptions about core ordering
    // NOTE: Proper hybrid CPU detection requires platform-specific APIs:
    // - Windows: GetLogicalProcessorInformationEx with RelationProcessorCore
    // - Linux: Reading /sys/devices/system/cpu/cpu*/topology/core_cpus_list
    // For now, we use a heuristic that may not be accurate on all systems.
    // TODO: Implement proper hybrid CPU core enumeration using platform APIs
    
    // Simple heuristic: Check if this appears to be a hybrid system
    // and assume P-cores come first (common on Intel 12th+ gen)
    if (is_smt_enabled())
    {
        std::uint32_t total_cores = std::thread::hardware_concurrency();
        std::uint32_t physical_cores = get_physical_core_count();
        
        // Simplified heuristic: assume first half are P-cores
        // This is NOT reliable and should be replaced with proper detection
        for (std::uint32_t i = 0; i < physical_cores / 2; i++)
        {
            p_cores.push_back(i);
        }
    }
    
    return p_cores;
}

std::vector<std::uint32_t> get_efficiency_cores()
{
    std::vector<std::uint32_t> e_cores;
    
    // This is a simplified implementation that makes assumptions about core ordering
    // NOTE: Proper hybrid CPU detection requires platform-specific APIs:
    // - Windows: GetLogicalProcessorInformationEx with RelationProcessorCore
    // - Linux: Reading /sys/devices/system/cpu/cpu*/topology/core_cpus_list
    // For now, we use a heuristic that may not be accurate on all systems.
    // TODO: Implement proper hybrid CPU core enumeration using platform APIs
    
    // Simple heuristic: Check if this appears to be a hybrid system
    // and assume E-cores come after P-cores (common on Intel 12th+ gen)
    if (is_smt_enabled())
    {
        std::uint32_t total_cores = std::thread::hardware_concurrency();
        std::uint32_t physical_cores = get_physical_core_count();
        
        // Simplified heuristic: assume second half are E-cores
        // This is NOT reliable and should be replaced with proper detection
        for (std::uint32_t i = physical_cores / 2; i < physical_cores; i++)
        {
            e_cores.push_back(i);
        }
    }
    
    return e_cores;
}

} // namespace cpu
} // namespace nexusminer
