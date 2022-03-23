cmake_minimum_required(VERSION 3.19)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules")

set(CMAKE_LIBRARY_PATH "${CMAKE_SOURCE_DIR}/libs")

project(NexusMiner VERSION 1.5 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

configure_file(src/version.h.in version.h)

set(CPM_DOWNLOAD_VERSION 0.35.0) 
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")

if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
    message(STATUS "Downloading CPM.cmake")
    file(DOWNLOAD https://github.com/TheLartians/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake ${CPM_DOWNLOAD_LOCATION})
endif()

include(${CPM_DOWNLOAD_LOCATION})

find_package(Threads REQUIRED)

#spdlog
CPMAddPackage(NAME spdlog GITHUB_REPOSITORY gabime/spdlog VERSION 1.9.2)

#nlohmann json
CPMAddPackage(NAME nlohmann_json GITHUB_REPOSITORY nlohmann/json  VERSION 3.10.5)

include_directories(${CMAKE_SOURCE_DIR}/include)
link_directories(${CMAKE_SOURCE_DIR}/libs)

option(WITH_GPU_CUDA "Build with gpu workers, CUDA needed" OFF)
option(WITH_PRIME "Build with PRIME mining support, BOOST and GMP or MPIR needed" OFF)
option(STATIC_OPENSSL "Build with static OpenSSL" ON)

if(UNIX)
    add_definitions(-DUNIX)
endif()

if(WIN32)
    add_definitions(-D_WIN32_WINT=0x0601 -DNOMINMAX -D_CRT_SECURE_NO_WARNINGS)
endif()

# OpenSSL
if(STATIC_OPENSSL)
    set(OPENSSL_USE_STATIC_LIBS TRUE)
endif()
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
        find_package(CUDAToolkit)
        include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) 
        add_definitions(-DGPU_ENABLED)
    else()
        message(STATUS "No CUDA compiler found")
    endif()
endif()

#PRIME
if(WITH_PRIME)
    find_package(Boost)
    include_directories(${Boost_INCLUDE_DIRS}) 
    include(FetchContent)
    FetchContent_Declare(primesieve GIT_REPOSITORY https://github.com/kimwalisch/primesieve.git GIT_TAG master)
    FetchContent_MakeAvailable(primesieve)
    # find_package(primesieve REQUIRED)
    if(WIN32)
       find_package(MPIR REQUIRED)
    else()
       find_package(GMP REQUIRED)
    endif()
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
add_subdirectory(src/TAO)


if(WITH_GPU_CUDA)
    add_subdirectory(src/gpu)
endif()

add_executable(NexusMiner 
                src/main.cpp 
                src/miner.cpp 
                src/worker_manager.cpp 
                src/timer_manager.cpp)

target_include_directories(NexusMiner PUBLIC "${PROJECT_BINARY_DIR}")
target_link_libraries(NexusMiner chrono network config stats protocol cpu fpga LLP worker TAO)
if(WITH_GPU_CUDA)
    target_link_libraries(NexusMiner gpu)
endif()

target_link_libraries(NexusMiner ${OPENSSL_LIBRARIES})
target_link_libraries(NexusMiner Threads::Threads)


