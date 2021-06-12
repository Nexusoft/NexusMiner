cmake_minimum_required(VERSION 3.20)

add_library(hash STATIC ${CMAKE_CURRENT_SOURCE_DIR}/nexus_keccak.cpp 
                ${CMAKE_CURRENT_SOURCE_DIR}/nexus_skein.cpp)
                    
target_include_directories(hash
    PUBLIC 
        $<INSTALL_INTERFACE:.>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/.>
)