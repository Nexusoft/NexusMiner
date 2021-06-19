cmake_minimum_required(VERSION 3.19)

add_library(stats STATIC ${CMAKE_CURRENT_SOURCE_DIR}/stats_collector.cpp
                         ${CMAKE_CURRENT_SOURCE_DIR}/stats_printer_console.cpp
                         ${CMAKE_CURRENT_SOURCE_DIR}/stats_printer_file.cpp)
                    
target_include_directories(stats
    PUBLIC 
        $<INSTALL_INTERFACE:.>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/.>
)

target_link_libraries(stats config)