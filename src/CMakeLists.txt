cmake_minimum_required(VERSION 3.10)

include_directories(${CMAKE_SOURCE_DIR}/src/include/)
include_directories(${CMAKE_SOURCE_DIR}/src/hash/)
include_directories(${CMAKE_SOURCE_DIR}/src/LLP/)
include_directories(${CMAKE_SOURCE_DIR}/src/network/)
include_directories(${CMAKE_SOURCE_DIR}/src/worker_software_hash/)
include_directories(${CMAKE_SOURCE_DIR}/src/fpga/)

add_subdirectory(chrono)



file(GLOB_RECURSE main_SRC CONFIGURE_DEPENDS "*.hpp" "*.h" "*.cpp")

add_executable(miner ${main_SRC})

target_link_libraries(miner chrono)
target_link_libraries(miner ${OPENSSL_LIBRARIES})
target_link_libraries(miner Threads::Threads)