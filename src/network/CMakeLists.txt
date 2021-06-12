cmake_minimum_required(VERSION 3.20)

add_library(network STATIC ${CMAKE_CURRENT_SOURCE_DIR}/create_component.cpp 
                    ${CMAKE_CURRENT_SOURCE_DIR}/endpoint.cpp)

target_include_directories(network
    PUBLIC 
        $<INSTALL_INTERFACE:.>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/.>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/tcp
)