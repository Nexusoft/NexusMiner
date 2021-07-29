cmake_minimum_required(VERSION 3.19)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules")

set(CMAKE_LIBRARY_PATH "${CMAKE_SOURCE_DIR}/libs")

project(NexusMiner VERSION 1.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

configure_file(src/version.h.in version.h)

include_directories(${CMAKE_SOURCE_DIR}/include)
link_directories(${CMAKE_SOURCE_DIR}/libs)

option(WITH_GPU_CUDA "Build with gpu workers, CUDA needed" OFF)
option(WITH_PRIME "Build with PRIME mining support, BOOST and Primesieve needed" OFF)

if(UNIX)
    add_definitions(-DUNIX)
endif()

if(WIN32)
    add_definitions(-D_WIN32_WINT=0x0601 -DNOMINMAX -D_CRT_SECURE_NO_WARNINGS)
endif()

find_package(Threads REQUIRED)

# OpenSSL
find_package(OpenSSL REQUIRED)
include_directories(${OPENSSL_INCLUDE_DIR})

#ASIO
add_definitions(-DASIO_STANDALONE)
include_directories(${CMAKE_SOURCE_DIR}/include/asio-1.18.1/include)

#GPU
if(WITH_GPU_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        add_definitions(-DGPU_ENABLED)
    else()
        message(STATUS "No CUDA compiler found")
    endif()
endif()

#PRIME
if(WITH_PRIME)
    find_package(Boost)
    include_directories(${Boost_INCLUDE_DIRS}) 
    find_package(primesieve REQUIRED)
    # if(WIN32)
    #    find_package(MPIR REQUIRED)
    # else()
    #    find_package(GMP REQUIRED)
    # endif()
    add_definitions(-DPRIME_ENABLED)
endif()


# add submodules
add_subdirectory(src/chrono)
add_subdirectory(src/network)
add_subdirectory(src/LLP)
add_subdirectory(src/LLC)
add_subdirectory(src/hash)
add_subdirectory(src/worker)
add_subdirectory(src/config)
add_subdirectory(src/stats)
add_subdirectory(src/protocol)
add_subdirectory(src/cpu)
add_subdirectory(src/fpga)

if(WITH_GPU_CUDA)
    add_subdirectory(src/gpu)
endif()

add_executable(NexusMiner 
                src/main.cpp 
                src/miner.cpp 
                src/worker_manager.cpp 
                src/timer_manager.cpp)

target_include_directories(NexusMiner PUBLIC "${PROJECT_BINARY_DIR}")
target_link_libraries(NexusMiner chrono network config stats protocol cpu fpga LLP worker)
if(WITH_GPU_CUDA)
    target_link_libraries(NexusMiner gpu)
endif()

target_link_libraries(NexusMiner ${OPENSSL_LIBRARIES})
target_link_libraries(NexusMiner Threads::Threads)


