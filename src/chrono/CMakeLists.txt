cmake_minimum_required(VERSION 3.19)

add_library(chrono INTERFACE)
target_include_directories(chrono INTERFACE .)

target_link_libraries(chrono INTERFACE asio)