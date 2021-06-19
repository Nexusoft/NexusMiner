cmake_minimum_required(VERSION 3.19)

add_library(protocol STATIC ${CMAKE_CURRENT_SOURCE_DIR}/solo.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/pool.cpp)
                    
target_include_directories(protocol
    PUBLIC 
        $<INSTALL_INTERFACE:.>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/.>
)

target_link_libraries(protocol network)