cmake_minimum_required(VERSION 3.19)

add_library(protocol STATIC src/protocol/solo.cpp
                            src/protocol/pool.cpp 
                            src/protocol/pool_legacy.cpp)
                    
target_include_directories(protocol
    PUBLIC 
        $<INSTALL_INTERFACE:inc>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(protocol PRIVATE network LLP stats config)