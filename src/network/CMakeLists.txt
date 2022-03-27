cmake_minimum_required(VERSION 3.19)

add_library(network STATIC "src/network/create_component.cpp"
                           "src/network/endpoint.cpp")

target_include_directories(network
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
        $<INSTALL_INTERFACE:inc>
    PRIVATE src
)

target_link_libraries(network asio)