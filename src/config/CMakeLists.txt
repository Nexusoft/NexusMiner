add_library(config STATIC "src/config/config.cpp"
                          "src/config/validator.cpp"
                          "src/config/toml_config.cpp"
                          "src/config/miner_config.cpp")
                    
target_include_directories(config
    PUBLIC 
        $<INSTALL_INTERFACE:inc>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(config spdlog nlohmann_json::nlohmann_json LLC)

add_nexusminer_test(
    NAME        toml_gpu_worker_enumeration_test
    SOURCES     toml_gpu_worker_enumeration_test.cpp
    LINK_LIBS   config spdlog::spdlog
    INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/inc
)
