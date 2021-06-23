cmake_minimum_required(VERSION 3.19)

add_library(cpu STATIC ${CMAKE_CURRENT_SOURCE_DIR}/worker_hash.cpp)
                    
target_include_directories(cpu
    PUBLIC 
        $<INSTALL_INTERFACE:.>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/.>
)