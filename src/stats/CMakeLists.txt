cmake_minimum_required(VERSION 3.19)

add_library(stats STATIC src/stats/stats_collector.cpp)
                    
target_include_directories(stats
    PUBLIC 
        $<INSTALL_INTERFACE:inc>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(stats PRIVATE config spdlog::spdlog)