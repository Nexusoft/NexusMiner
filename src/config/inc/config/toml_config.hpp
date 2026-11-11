#ifndef NEXUSMINER_TOML_CONFIG_HPP
#define NEXUSMINER_TOML_CONFIG_HPP

#include <string>
#include <memory>

namespace spdlog { class logger; }

namespace nexusminer
{
namespace config
{
    class Config;

    // TOML configuration parser for .config files
    // Provides a simpler, more readable alternative to JSON .conf files
    class TomlConfig
    {
    public:
        explicit TomlConfig(std::shared_ptr<spdlog::logger> logger);
        
        // Parse a TOML .config file and populate the Config object
        bool parse_file(const std::string& filename, Config& config);

    private:
        std::shared_ptr<spdlog::logger> m_logger;
        
        // Helper methods for parsing
        std::string trim(const std::string& str);
        bool is_comment(const std::string& line);
        bool is_section(const std::string& line, std::string& section_name);
        bool parse_key_value(const std::string& line, std::string& key, std::string& value);
        
        // Value parsing helpers
        std::string parse_string_value(const std::string& value);
        int parse_int_value(const std::string& value);
        bool parse_bool_value(const std::string& value);
    };

} // namespace config
} // namespace nexusminer

#endif // NEXUSMINER_TOML_CONFIG_HPP
