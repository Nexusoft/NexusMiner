cmake_minimum_required(VERSION 3.19)

add_library(hash STATIC src/hash/nexus_keccak.cpp
                        src/hash/nexus_skein.cpp)
                    
target_include_directories(hash
    PUBLIC 
        $<INSTALL_INTERFACE:inc>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
    PRIVATE src
)