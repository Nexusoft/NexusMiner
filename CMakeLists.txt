cmake_minimum_required(VERSION 3.19)

project(NexusMiner CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR}/include/)

option(WITH_GPU_CUDA "Build with gpu workers, CUDA needed" OFF)

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
    add_definitions(-D_WIN32_WINT=0x0601 -DNOMINMAX -D_CRT_SECURE_NO_WARNINGS)
endif()

if(MSVC)
    # enable mt build
    add_definitions(/MP)     
    #enable C++ standard conformance       
    add_definitions(/permissive-)
endif()

#GPU
if(WITH_GPU_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
    else()
        message(STATUS "No CUDA compiler found")
    endif()
endif()

include_directories(src/include/)
include_directories(src/hash/)

# add submodules
add_subdirectory(src/chrono)
add_subdirectory(src/network)
add_subdirectory(src/hash)
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

target_link_libraries(NexusMiner chrono network config stats protocol cpu fpga)
if(WITH_GPU_CUDA)
    set_target_properties(NexusMiner PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(NexusMiner gpu)
endif()
target_link_libraries(NexusMiner ${OPENSSL_LIBRARIES})
target_link_libraries(NexusMiner Threads::Threads)
