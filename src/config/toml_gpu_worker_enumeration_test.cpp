#include "config/config.hpp"
#include "config/toml_config.hpp"

#include <spdlog/spdlog.h>
#include <spdlog/logger.h>
#include <spdlog/sinks/null_sink.h>

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace
{
std::shared_ptr<spdlog::logger> install_test_logger()
{
    spdlog::drop("logger");
    auto sink = std::make_shared<spdlog::sinks::null_sink_mt>();
    auto logger = std::make_shared<spdlog::logger>("logger", std::move(sink));
    logger->set_level(spdlog::level::off);
    spdlog::register_logger(logger);
    return logger;
}
}

int main()
{
    auto logger = install_test_logger();

    const auto temp_dir = std::filesystem::temp_directory_path();
    const auto config_path = temp_dir / "nexusminer-toml-gpu-workers.config";

    std::ofstream config_file(config_path);
    config_file << "[wallet]\n";
    config_file << "ip = \"127.0.0.1\"\n";
    config_file << "port = 8323\n\n";
    config_file << "[mining]\n";
    config_file << "channel = 2\n\n";
    config_file << "[workers]\n";
    config_file << "count = 3\n";
    config_file << "hardware = \"gpu\"\n\n";
    config_file << "[gpu]\n";
    config_file << "device = 2\n";
    config_file.close();

    nexusminer::config::Config config{logger};
    nexusminer::config::TomlConfig parser{logger};

    const bool parsed = parser.parse_file(config_path.string(), config);
    assert(parsed);
    assert(config.get_worker_config().size() == 3);

    for (std::size_t i = 0; i < config.get_worker_config().size(); ++i)
    {
        auto& worker = config.get_worker_config()[i];
        assert(worker.m_mode == nexusminer::config::Worker_mode::GPU);
        const auto& gpu_config = std::get<nexusminer::config::Worker_config_gpu>(worker.m_worker_mode);
        assert(gpu_config.m_device == static_cast<std::uint8_t>(2 + i));
    }

    std::filesystem::remove(config_path);
    std::cout << "toml_gpu_worker_enumeration_test: passed\n";
    return 0;
}
