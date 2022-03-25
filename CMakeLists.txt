cmake_minimum_required(VERSION 3.21)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules")

set(CMAKE_LIBRARY_PATH "${CMAKE_SOURCE_DIR}/libs")

project(NexusMiner VERSION 1.4 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

configure_file(src/version.h.in version.h)

include_directories(${CMAKE_SOURCE_DIR}/include)
link_directories(${CMAKE_SOURCE_DIR}/libs)

option(WITH_GPU_AMD "Build with AMD gpu workers, Requires HIP and clang" OFF)
option(WITH_GPU_CUDA "Build with Nvidia gpu workers, CUDA needed" OFF)
option(WITH_PRIME "Build with PRIME mining support, BOOST and GMP or MPIR needed" OFF)
option(STATIC_OPENSSL "Build with static OpenSSL" ON)

if(UNIX)
    add_definitions(-DUNIX)
endif()

if(WIN32)
    add_definitions(-D_WIN32_WINT=0x0601 -DNOMINMAX -D_CRT_SECURE_NO_WARNINGS)
endif()

find_package(Threads REQUIRED)

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
        add_definitions(-DGPU_CUDA_ENABLED)
        add_definitions(-DGPU_ENABLED)
    else()
        message(STATUS "No CUDA compiler found")
    endif()
endif()

if(WITH_GPU_AMD)
    # Search for rocm in common locations
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
    find_package(hip)
    add_definitions(-DGPU_AMD_ENABLED)
    add_definitions(-DGPU_ENABLED)    
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


if(WITH_GPU_CUDA OR WITH_GPU_AMD)
    add_subdirectory(src/gpu)
endif()

set(MAIN_SOURCE_FILES src/main.cpp
                src/miner.cpp 
                src/worker_manager.cpp 
                src/timer_manager.cpp)

set(GPU_TARGETS "gfx1030" CACHE STRING "AMD GPU targets")
add_executable(NexusMiner ${MAIN_SOURCE_FILES})
if (WITH_GPU_AMD AND WITH_PRIME)
    set(GPU_FILES 
        src/gpu/src/gpu/cuda_prime/cuda_chain.cu  
        src/gpu/src/gpu/cuda_prime/sieve.cu
        src/gpu/src/gpu/cuda_prime/sieve_impl.cu
        src/gpu/src/gpu/cuda_prime/find_chain.cu
        src/gpu/src/gpu/cuda_prime/fermat_prime/fermat_prime.cu
        src/gpu/src/gpu/cuda_prime/fermat_prime/fermat_prime_impl.cu
        src/gpu/src/gpu/cuda_prime/fermat_prime/fermat_utils.cu)

    set(HOST_FILES 
        src/gpu/src/gpu/worker_prime.cpp
        src/gpu/src/gpu/prime/prime.cpp
        src/gpu/src/gpu/prime/prime_tests.cpp
        src/gpu/src/gpu/prime/chain.cpp
        src/gpu/src/gpu/prime/sieve.cpp
        src/gpu/src/gpu/prime/small_sieve_tools.cpp)
    set_source_files_properties(${GPU_FILES} PROPERTIES COMPILE_FLAGS "-fgpu-rdc" )
    set_source_files_properties(${GPU_FILES} PROPERTIES LANGUAGE CXX)

    target_sources(NexusMiner PRIVATE ${GPU_FILES} ${HOST_FILES})
    include_directories(${CMAKE_SOURCE_DIR}/src/gpu/inc)
    target_link_libraries(NexusMiner hip::device "--hip-link -fgpu-rdc")
endif() 
              

target_include_directories(NexusMiner PUBLIC "${PROJECT_BINARY_DIR}")
target_link_libraries(NexusMiner chrono network config stats protocol cpu fpga LLP worker TAO)
if(WITH_GPU_CUDA)
    target_link_libraries(NexusMiner gpu)
endif()


target_link_libraries(NexusMiner ${OPENSSL_LIBRARIES})
target_link_libraries(NexusMiner Threads::Threads)


