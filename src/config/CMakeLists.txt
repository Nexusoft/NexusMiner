cmake_minimum_required(VERSION 3.19)

add_library(config STATIC ${CMAKE_CURRENT_SOURCE_DIR}/config.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/validator.cpp)
                    
target_include_directories(config
    PUBLIC 
        $<INSTALL_INTERFACE:.>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/.>
)