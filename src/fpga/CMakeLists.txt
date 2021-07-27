cmake_minimum_required(VERSION 3.19)

add_library(fpga STATIC "src/fpga/worker_hash.cpp")
                    
target_include_directories(fpga
    PUBLIC 
        $<INSTALL_INTERFACE:inc>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(fpga hash worker LLP stats config)