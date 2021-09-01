cmake_minimum_required(VERSION 3.19)

add_library(gpu STATIC "src/gpu/worker_hash.cpp" "src/gpu/cuda_hash/util.cu" "src/gpu/cuda_hash/constants.cu" "src/gpu/cuda_hash/sk1024.cu")

if(WITH_PRIME)
    target_sources(gpu PRIVATE src/gpu/worker_prime.cpp src/gpu/prime/prime.cpp src/gpu/prime/chain_sieve.cpp src/gpu/cuda_prime/fermat_test.cu src/gpu/cuda_prime/sieve.cu)
endif()

set_target_properties(gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET gpu PROPERTY CUDA_ARCHITECTURES 61 75)
#set(CMAKE_CUDA_FLAGS "-arch=sm_61")
#set(CMAKE_CUDA_FLAGS "-gencode arch=compute_61,code=sm_61 -gencode arch=compute_75,code=sm_75")
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) 

#-DNDEBUG causes nvcc build errors for some architectures
foreach(flag_var
    CMAKE_CUDA_FLAGS CMAKE_CUDA_FLAGS_RELEASE CMAKE_CUDA_FLAGS_RELWITHDEBINFO CMAKE_CUDA_FLAGS_MINSIZEREL)
  if(${flag_var} MATCHES "-DNDEBUG")
    string(REGEX REPLACE "-DNDEBUG" "" ${flag_var} "${${flag_var}}")
  endif()
endforeach()

                    
target_include_directories(gpu
    PUBLIC 
        $<INSTALL_INTERFACE:inc>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(gpu LLC worker LLP stats config)

if(WITH_PRIME)
    target_link_libraries(gpu primesieve::primesieve)
    if(WIN32)
        target_link_libraries(gpu mpir)
    else()
       target_link_libraries(gpu gmp)
    endif()
endif()