cmake_minimum_required(VERSION 3.19)

add_library(cpu STATIC src/cpu/worker_hash.cpp)

if(WITH_PRIME)
    target_sources(cpu PRIVATE src/cpu/worker_prime.cpp src/cpu/prime/prime.cpp src/cpu/prime/chain_sieve.cpp)
endif()
                    
target_include_directories(cpu
    PUBLIC 
        $<INSTALL_INTERFACE:inc>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
    PRIVATE src
)

# TODO reduce dependencies
target_link_libraries(cpu hash worker LLP LLC stats config spdlog::spdlog)
if(WITH_PRIME)
    target_link_libraries(cpu libprimesieve-static)
    if(WIN32)
        target_link_libraries(cpu mpir) # OpenSSL::Crypto OpenSSL::applink)
    else()
       target_link_libraries(cpu gmp) # OpenSSL::Crypto OpenSSL::applink)
    endif()
endif()