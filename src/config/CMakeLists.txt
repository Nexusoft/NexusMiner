cmake_minimum_required(VERSION 3.20)

add_library(config STATIC ${CMAKE_CURRENT_SOURCE_DIR}/config.cpp)
                    
target_include_directories(config
    PUBLIC 
        $<INSTALL_INTERFACE:.>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/.>
)