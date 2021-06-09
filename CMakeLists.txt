cmake_minimum_required(VERSION 3.10)

project(NexusMiner CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR}/include/)

find_package(Threads REQUIRED)

# OpenSSL
find_package(OpenSSL REQUIRED)
include_directories(${OPENSSL_INCLUDE_DIR})

#ASIO
add_definitions(-DASIO_STANDALONE)
include_directories(${CMAKE_SOURCE_DIR}/include/asio-1.18.1/include)

if(UNIX)
    add_definitions(-DUNIX)
endif()

if(WIN32)
    add_definitions(-D_WIN32_WINT=0x501 -DNOMINMAX -D_CRT_SECURE_NO_WARNINGS)
endif()

if(MSVC)
    # enable mt build
    add_definitions(/MP)     
    #enable C++ standard conformance       
    add_definitions(/permissive-)
endif()

include_directories(src/include/)

# add submodules
add_subdirectory(src/chrono)
add_subdirectory(src/network)
add_subdirectory(src/hash)

add_executable(NexusMiner 
                src/config.cpp 
                src/main.cpp 
                src/miner.cpp 
                src/worker_manager.cpp 
                src/timer_manager.cpp
                src/stats_collector.cpp
                src/cpu/worker_software_hash.cpp
                src/fpga/worker_fpga.cpp)

target_link_libraries(NexusMiner chrono)
target_link_libraries(NexusMiner network)
target_link_libraries(NexusMiner hash)
target_link_libraries(NexusMiner ${OPENSSL_LIBRARIES})
target_link_libraries(NexusMiner Threads::Threads)
