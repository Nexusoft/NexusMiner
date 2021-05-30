cmake_minimum_required(VERSION 3.10)

include_directories(${CMAKE_SOURCE_DIR}/src/include/)
include_directories(${CMAKE_SOURCE_DIR}/src/worker_software_hash/)
include_directories(${CMAKE_SOURCE_DIR}/src/fpga/)

add_subdirectory(chrono)
add_subdirectory(network)
add_subdirectory(hash)

add_executable(miner config.cpp main.cpp miner.cpp worker_manager.cpp worker_software_hash/worker_software_hash.cpp)

target_link_libraries(miner chrono)
target_link_libraries(miner network)
target_link_libraries(miner hash)
target_link_libraries(miner ${OPENSSL_LIBRARIES})
target_link_libraries(miner Threads::Threads)