#include "config/toml_config.hpp"
#include "config/config.hpp"
#include "config/types.hpp"
#include "config/worker_config.hpp"
#include "config/stats_printer_config.hpp"
#include <limits>
#include <spdlog/spdlog.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>

namespace nexusminer
{
namespace config
{
    // Constants for keepalive interval validation
    constexpr int MIN_KEEPALIVE_HOURS = 1;
    constexpr int MAX_KEEPALIVE_HOURS = 168;  // 1 week
    
    TomlConfig::TomlConfig(std::shared_ptr<spdlog::logger> logger)
        : m_logger(std::move(logger))
    {
    }

    bool TomlConfig::parse_file(const std::string& filename, Config& config)
    {
        std::ifstream file(filename);
        if (!file.is_open())
        {
            m_logger->critical("Unable to open TOML config file: {}", filename);
            return false;
        }

        std::string current_section;
        std::string line;
        int line_number = 0;
        
        // Track worker configuration
        int worker_count = 0;
        int cpu_threads = 1;  // Default threads per worker
        bool efficiency_cores = true;  // Default
        int cpu_priority = 2;  // Default: normal (0=low, 1=below_normal, 2=normal, 3=above_normal, 4=high)
        int cpu_power_limit = 100;  // Default: 100%
        bool cpu_hyperthreading = true;
        int cpu_target_hashrate = 0;  // Default: 0 = max
        std::string cpu_engine_mode = "workers";  // Stone 3: prime backend selector
        std::string worker_hardware = "cpu";  // Default hardware type
        int gpu_device = 0;  // Default GPU device
        bool stats_console = false;  // Default: no stats printer

        try
        {
            while (std::getline(file, line))
            {
                line_number++;
                line = trim(line);

                // Skip empty lines and comments
                if (line.empty() || is_comment(line))
                {
                    continue;
                }

                // Check for section headers
                std::string section_name;
                if (is_section(line, section_name))
                {
                    current_section = section_name;
                    continue;
                }

                // Parse key-value pairs
                std::string key, value;
                if (!parse_key_value(line, key, value))
                {
                    m_logger->warn("Invalid line {} in {}: {}", line_number, filename, line);
                    continue;
                }

                // Process based on current section
                if (current_section == "wallet")
                {
                    if (key == "ip")
                    {
                        config.set_wallet_ip(parse_string_value(value));
                    }
                    else if (key == "port")
                    {
                        config.set_port(static_cast<std::uint16_t>(parse_int_value(value)));
                    }
                    else if (key == "failover_wallet_ip")
                    {
                        config.set_failover_wallet_ip(parse_string_value(value));
                    }
                    else if (key == "failover_port")
                    {
                        config.set_failover_port(static_cast<std::uint16_t>(parse_int_value(value)));
                    }
                    else if (key == "failover_max_retries")
                    {
                        config.set_failover_max_retries(static_cast<uint32_t>(parse_int_value(value)));
                    }
                }
                else if (current_section == "mining")
                {
                    if (key == "channel")
                    {
                        int channel = parse_int_value(value);
                        if (channel == 1)
                        {
                            config.set_mining_mode(Mining_mode::PRIME);
                        }
                        else if (channel == 2)
                        {
                            config.set_mining_mode(Mining_mode::HASH);
                        }
                        else
                        {
                            m_logger->warn("Invalid mining channel: {}. Use 1 for PRIME or 2 for HASH", channel);
                        }
                    }
                    else if (key == "genesis")
                    {
                        std::string genesis = parse_string_value(value);
                        if (!genesis.empty() && genesis.length() != TRITIUM_GENESIS_HEX_LENGTH)
                        {
                            m_logger->warn("tritium_genesis must be {} hex characters (32 bytes). Ignoring invalid value.", 
                                          TRITIUM_GENESIS_HEX_LENGTH);
                        }
                        else
                        {
                            config.set_tritium_genesis(genesis);
                        }
                    }
                    else if (key == "reward_address")
                    {
                        std::string address = parse_string_value(value);
                        config.set_reward_address(address);
                        m_logger->info("Mining reward address configured: {}", address);
                    }
                }
                else if (current_section == "workers")
                {
                    if (key == "count")
                    {
                        worker_count = parse_int_value(value);
                    }
                    else if (key == "hardware")
                    {
                        worker_hardware = parse_string_value(value);
                    }
                }
                else if (current_section == "cpu")
                {
                    if (key == "threads")
                    {
                        cpu_threads = parse_int_value(value);
                    }
                    else if (key == "efficiency_cores")
                    {
                        efficiency_cores = parse_bool_value(value);
                    }
                    else if (key == "priority")
                    {
                        cpu_priority = parse_int_value(value);
                        if (cpu_priority < 0 || cpu_priority > 4)
                        {
                            m_logger->warn("Invalid priority {} out of range [0-4], using default 2", cpu_priority);
                            cpu_priority = 2;
                        }
                    }
                    else if (key == "power_limit_percent")
                    {
                        cpu_power_limit = parse_int_value(value);
                        if (cpu_power_limit < 50 || cpu_power_limit > 100)
                        {
                            m_logger->warn("Invalid power_limit_percent {} out of range [50-100], using default 100", cpu_power_limit);
                            cpu_power_limit = 100;
                        }
                    }
                    else if (key == "hyperthreading")
                    {
                        cpu_hyperthreading = parse_bool_value(value);
                    }
                    else if (key == "target_hashrate")
                    {
                        cpu_target_hashrate = parse_int_value(value);
                    }
                    else if (key == "engine_mode")
                    {
                        cpu_engine_mode = parse_string_value(value);
                        if (cpu_engine_mode != "workers" && cpu_engine_mode != "engine")
                        {
                            m_logger->warn("Invalid engine_mode '{}' (expected 'workers' or 'engine'); using 'workers'.",
                                           cpu_engine_mode);
                            cpu_engine_mode = "workers";
                        }
                    }
                }
                else if (current_section == "gpu")
                {
                    if (key == "device")
                    {
                        gpu_device = parse_int_value(value);
                        if (gpu_device < 0 || gpu_device > 255)
                        {
                            m_logger->warn("Invalid GPU device {} out of range [0-255], using default 0", gpu_device);
                            gpu_device = 0;
                        }
                        // [gpu] section sets hardware to GPU (last parsed section takes precedence)
                        worker_hardware = "gpu";
                    }
                }
                else if (current_section == "stats")
                {
                    if (key == "mode")
                    {
                        std::string mode = parse_string_value(value);
                        if (mode == "console")
                        {
                            stats_console = true;
                        }
                        else if (mode == "file")
                        {
                            // File mode not yet implemented in TOML parser
                            m_logger->warn("Stats mode 'file' is not yet implemented in TOML config, ignoring");
                        }
                        else
                        {
                            m_logger->warn("Invalid stats mode '{}', expected 'console' or 'file'", mode);
                        }
                    }
                }
                else if (current_section == "network")
                {
                    if (key == "local_ip")
                    {
                        config.set_local_ip(parse_string_value(value));
                    }
                    else if (key == "keepalive_interval")
                    {
                        int interval = parse_int_value(value);
                        // Clamp to reasonable range
                        if (interval < MIN_KEEPALIVE_HOURS) interval = MIN_KEEPALIVE_HOURS;
                        if (interval > MAX_KEEPALIVE_HOURS) interval = MAX_KEEPALIVE_HOURS;
                        config.set_keepalive_interval(static_cast<std::uint16_t>(interval));
                    }
                    else if (key == "sim_link")
                    {
                        config.set_enable_sim_link(parse_bool_value(value));
                    }
                    else if (key == "get_block_interval_ms")
                    {
                        constexpr int MIN_GET_BLOCK_MS = 100;
                        constexpr int MAX_GET_BLOCK_MS = 60000;
                        int ms = parse_int_value(value);
                        if (ms >= MIN_GET_BLOCK_MS && ms <= MAX_GET_BLOCK_MS)
                            config.set_get_block_interval_ms(static_cast<uint32_t>(ms));
                        else
                            m_logger->warn("[Config] get_block_interval_ms={} out of range [{},{}] — using default",
                                ms, MIN_GET_BLOCK_MS, MAX_GET_BLOCK_MS);
                    }
                }
                else if (current_section == "colin")
                {
                    if (key == "enabled")
                    {
                        config.set_colin_enabled(parse_bool_value(value));
                    }
                    else if (key == "report_interval_seconds")
                    {
                        constexpr int MIN_COLIN_INTERVAL = 10;
                        constexpr int MAX_COLIN_INTERVAL = 3600;
                        int secs = parse_int_value(value);
                        if (secs >= MIN_COLIN_INTERVAL && secs <= MAX_COLIN_INTERVAL)
                            config.set_colin_report_interval_seconds(static_cast<uint32_t>(secs));
                        else
                            m_logger->warn("[Config] colin.report_interval_seconds={} out of range [{},{}] — using default",
                                secs, MIN_COLIN_INTERVAL, MAX_COLIN_INTERVAL);
                    }
                }
                else if (current_section == "logging")
                {
                    if (key == "level")
                    {
                        int level = parse_int_value(value);
                        if (level >= 0 && level <= 3)
                        {
                            config.set_log_level(static_cast<std::uint8_t>(level));
                        }
                    }
                    else if (key == "file")
                    {
                        config.set_logfile(parse_string_value(value));
                    }
                }
                else if (current_section == "falcon")
                {
                    if (key == "pubkey")
                    {
                        config.set_miner_falcon_pubkey(parse_string_value(value));
                    }
                    else if (key == "privkey")
                    {
                        config.set_miner_falcon_privkey(parse_string_value(value));
                    }
                    else if (key == "enable_block_signing" || key == "enable_disposable_falcon")
                    {
                        // Support both old and new config keys for backward compatibility
                        config.set_enable_disposable_falcon(parse_bool_value(value));
                    }
                }
                else if (current_section == "advanced")
                {
                    if (key == "connection_retry_interval")
                    {
                        config.set_connection_retry_interval(static_cast<std::uint16_t>(parse_int_value(value)));
                    }
                    else if (key == "print_statistics_interval")
                    {
                        config.set_print_statistics_interval(static_cast<std::uint16_t>(parse_int_value(value)));
                    }
                    else if (key == "ping_interval")
                    {
                        config.set_ping_interval(static_cast<std::uint16_t>(parse_int_value(value)));
                    }
                    else if (key == "get_height_interval")
                    {
                        config.set_get_height_interval(static_cast<std::uint16_t>(parse_int_value(value)));
                    }
                    else if (key == "enable_chacha20_wrapping")
                    {
                        config.set_enable_chacha20_wrapping(parse_bool_value(value));
                    }
                }
                else if (current_section == "tls")
                {
                    if (key == "enable" || key == "enable_tls")
                    {
                        config.set_enable_tls(parse_bool_value(value));
                    }
                    else if (key == "ssl_port")
                    {
                        // Dedicated TLS mining port (0 = use [wallet] port with TLS)
                        // Forward-reserved: 9325 (stateless TLS) / 8325 (legacy TLS)
                        int port = parse_int_value(value);
                        if (port >= 0 && port <= std::numeric_limits<std::uint16_t>::max())
                            config.set_ssl_port(static_cast<std::uint16_t>(port));
                        else
                            m_logger->warn("[Config] ssl_port={} out of range [0-{}] — using default (0)",
                                          port, std::numeric_limits<std::uint16_t>::max());
                    }
                    else if (key == "ca_cert_path" || key == "tls_ca_cert_path")
                    {
                        config.set_tls_ca_cert_path(parse_string_value(value));
                    }
                    else if (key == "verify_peer" || key == "tls_verify_peer")
                    {
                        config.set_tls_verify_peer(parse_bool_value(value));
                    }
                    else if (key == "server_name" || key == "tls_server_name")
                    {
                        config.set_tls_server_name(parse_string_value(value));
                    }
                    else if (key == "client_cert_path" || key == "tls_client_cert_path")
                    {
                        config.set_tls_client_cert_path(parse_string_value(value));
                    }
                    else if (key == "client_key_path" || key == "tls_client_key_path")
                    {
                        config.set_tls_client_key_path(parse_string_value(value));
                    }
                    else if (key == "client_key_password" || key == "tls_client_key_password")
                    {
                        config.set_tls_client_key_password(parse_string_value(value));
                    }
                }
            }

            // Create worker configurations after parsing all settings
            if (worker_count > 0)
            {
                // Clear existing workers
                config.m_worker_config.clear();
                
                // Create workers based on hardware type
                if (worker_hardware == "gpu")
                {
                    // Create GPU workers
                    for (int i = 0; i < worker_count; ++i)
                    {
                        Worker_config worker_config;
                        worker_config.m_id = "gpu" + std::to_string(i);
                        worker_config.m_mode = Worker_mode::GPU;
                        
                        Worker_config_gpu gpu_config{};
                        const int enumerated_device = gpu_device + i;
                        if (enumerated_device > 255)
                        {
                            m_logger->warn("GPU worker {} would exceed device index range with enumerated device {}. Stopping GPU worker creation.",
                                i, enumerated_device);
                            break;
                        }
                        gpu_config.m_device = static_cast<std::uint8_t>(enumerated_device);
                        
                        worker_config.m_worker_mode = gpu_config;
                        config.m_worker_config.push_back(worker_config);
                    }
                }
                else
                {
                    // Create CPU workers with all configured settings
                    for (int i = 0; i < worker_count; ++i)
                    {
                        Worker_config worker_config;
                        worker_config.m_id = "cpu" + std::to_string(i);
                        worker_config.m_mode = Worker_mode::CPU;
                        
                        Worker_config_cpu cpu_config{};
                        cpu_config.m_threads = static_cast<std::uint16_t>(cpu_threads);
                        cpu_config.m_affinity_mask = 0;  // No affinity by default
                        cpu_config.m_enable_efficiency_cores = efficiency_cores;
                        cpu_config.m_priority_level = static_cast<std::uint8_t>(cpu_priority);
                        cpu_config.m_power_limit_percent = static_cast<std::uint8_t>(cpu_power_limit);
                        cpu_config.m_enable_hyperthreading = cpu_hyperthreading;
                        cpu_config.m_target_hashrate = static_cast<std::uint32_t>(cpu_target_hashrate);
                        cpu_config.m_engine_mode = cpu_engine_mode;
                        
                        worker_config.m_worker_mode = cpu_config;
                        config.m_worker_config.push_back(worker_config);
                    }
                }
            }
            
            // Add stats printer configuration if console mode is enabled
            if (stats_console)
            {
                Stats_printer_config stats_config;
                stats_config.m_mode = Stats_printer_mode::CONSOLE;
                stats_config.m_printer_mode = Stats_printer_config_console{};
                config.m_stats_printer_config.push_back(stats_config);
            }

            return true;
        }
        catch (const std::exception& e)
        {
            m_logger->critical("Error parsing TOML config at line {}: {}", line_number, e.what());
            return false;
        }
    }

    std::string TomlConfig::trim(const std::string& str)
    {
        size_t first = str.find_first_not_of(" \t\r\n");
        if (first == std::string::npos)
            return "";
        
        size_t last = str.find_last_not_of(" \t\r\n");
        return str.substr(first, last - first + 1);
    }

    bool TomlConfig::is_comment(const std::string& line)
    {
        return !line.empty() && line[0] == '#';
    }

    bool TomlConfig::is_section(const std::string& line, std::string& section_name)
    {
        if (line.empty() || line[0] != '[')
            return false;

        size_t end = line.find(']');
        if (end == std::string::npos)
            return false;

        section_name = trim(line.substr(1, end - 1));
        return true;
    }

    bool TomlConfig::parse_key_value(const std::string& line, std::string& key, std::string& value)
    {
        size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos)
            return false;

        key = trim(line.substr(0, eq_pos));
        value = trim(line.substr(eq_pos + 1));
        
        // Key must not be empty, but value can be empty (for optional settings)
        return !key.empty();
    }

    std::string TomlConfig::parse_string_value(const std::string& value)
    {
        std::string result = trim(value);
        
        if (result.empty())
            return result;
        
        // Handle quoted strings: find closing quote, discard trailing comment
        // Note: This does not handle escaped quotes (e.g., "value with \" inside")
        // as they are not currently used in this codebase's config files.
        if (result.front() == '"' || result.front() == '\'')
        {
            char quote_char = result.front();
            size_t close_quote = result.find(quote_char, 1);
            if (close_quote != std::string::npos)
            {
                // Check if there's a trailing comment after the closing quote
                if (close_quote + 1 < result.size())
                {
                    std::string trailing = result.substr(close_quote + 1);
                    if (trailing.find('#') != std::string::npos)
                    {
                        m_logger->debug("[TOML] Stripped inline comment from value on this line. "
                                        "Move comments to their own line to avoid confusion.");
                    }
                }
                // Truncate at closing quote (discard any trailing "  # comment")
                result = result.substr(0, close_quote + 1);
            }
            // Remove the surrounding quotes
            if (result.size() >= 2 &&
                result.front() == quote_char && result.back() == quote_char)
            {
                result = result.substr(1, result.size() - 2);
            }
        }
        else
        {
            // Unquoted value: strip from first '#' onwards, then trim
            size_t comment_pos = result.find('#');
            if (comment_pos != std::string::npos)
            {
                m_logger->debug("[TOML] Stripped inline comment from value on this line. "
                                "Move comments to their own line to avoid confusion.");
                result = result.substr(0, comment_pos);
            }
            result = trim(result);
        }
        
        return result;
    }

    int TomlConfig::parse_int_value(const std::string& value)
    {
        std::string cleaned = trim(value);
        
        // Strip inline comment first (before removing quotes)
        size_t comment_pos = cleaned.find('#');
        if (comment_pos != std::string::npos)
        {
            m_logger->debug("[TOML] Stripped inline comment from value on this line. "
                            "Move comments to their own line to avoid confusion.");
            cleaned = trim(cleaned.substr(0, comment_pos));
        }
        
        // Remove quotes if present
        if (cleaned.size() >= 2 &&
            ((cleaned.front() == '"' && cleaned.back() == '"') ||
             (cleaned.front() == '\'' && cleaned.back() == '\'')))
        {
            cleaned = cleaned.substr(1, cleaned.size() - 2);
        }
        
        try
        {
            return std::stoi(cleaned);
        }
        catch (const std::invalid_argument&) { return 0; }
        catch (const std::out_of_range&)     { return 0; }
    }

    bool TomlConfig::parse_bool_value(const std::string& value)
    {
        std::string cleaned = trim(value);
        
        // Strip inline comment first (before any transformations)
        size_t comment_pos = cleaned.find('#');
        if (comment_pos != std::string::npos)
        {
            m_logger->debug("[TOML] Stripped inline comment from value on this line. "
                            "Move comments to their own line to avoid confusion.");
            cleaned = trim(cleaned.substr(0, comment_pos));
        }
        
        // Remove quotes if present
        if (cleaned.size() >= 2 &&
            ((cleaned.front() == '"' && cleaned.back() == '"') ||
             (cleaned.front() == '\'' && cleaned.back() == '\'')))
        {
            cleaned = cleaned.substr(1, cleaned.size() - 2);
        }
        
        // Convert to lowercase for comparison
        std::transform(cleaned.begin(), cleaned.end(), cleaned.begin(), ::tolower);
        
        return cleaned == "true" || cleaned == "1" || cleaned == "yes";
    }

} // namespace config
} // namespace nexusminer
