cmake_minimum_required(VERSION 3.19)

add_library(config STATIC "src/config/config.cpp"
                          "src/config/validator.cpp")
                    
target_include_directories(config
    PUBLIC 
        $<INSTALL_INTERFACE:inc>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(config spdlog nlohmann_json::nlohmann_json)