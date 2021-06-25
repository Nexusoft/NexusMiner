cmake_minimum_required(VERSION 3.19)

add_library(gpu STATIC "src/gpu/worker_hash.cpp")
                    
target_include_directories(gpu
    PUBLIC 
        $<INSTALL_INTERFACE:inc>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(gpu hash worker LLP stats config)