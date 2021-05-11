cmake_minimum_required(VERSION 3.12)

file(GLOB main_SRC CONFIGURE_DEPENDS "*.h" "*.cpp")

add_executable(miner ${main_SRC})