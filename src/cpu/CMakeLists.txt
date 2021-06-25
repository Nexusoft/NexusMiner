cmake_minimum_required(VERSION 3.19)

add_library(cpu STATIC "src/cpu/worker_hash.cpp")
                    
target_include_directories(cpu
    PUBLIC 
        $<INSTALL_INTERFACE:inc>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

# TODO reduce dependencies
target_link_libraries(cpu hash worker LLP stats config)