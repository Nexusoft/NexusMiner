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

include_directories(${CMAKE_SOURCE_DIR}/src/include/)
include_directories(${CMAKE_SOURCE_DIR}/src/worker_software_hash/)

# add submodules
add_subdirectory(${CMAKE_SOURCE_DIR}/src/chrono)
add_subdirectory(${CMAKE_SOURCE_DIR}/src/network)
add_subdirectory(${CMAKE_SOURCE_DIR}/src/hash)

add_executable(NexusMiner 
                ${CMAKE_SOURCE_DIR}/src/config.cpp 
                ${CMAKE_SOURCE_DIR}/src/main.cpp 
                ${CMAKE_SOURCE_DIR}/src/miner.cpp 
                ${CMAKE_SOURCE_DIR}/src/worker_manager.cpp 
                ${CMAKE_SOURCE_DIR}/src/worker_software_hash/worker_software_hash.cpp)

target_link_libraries(NexusMiner chrono)
target_link_libraries(NexusMiner network)
target_link_libraries(NexusMiner hash)
target_link_libraries(NexusMiner ${OPENSSL_LIBRARIES})
target_link_libraries(NexusMiner Threads::Threads)
