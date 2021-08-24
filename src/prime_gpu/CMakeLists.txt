cmake_minimum_required(VERSION 3.19)

add_library(gpu_prime STATIC "fermat_test.cu")

set_target_properties(gpu_prime PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET gpu_prime PROPERTY CUDA_ARCHITECTURES 75)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) 
include_directories(/usr/local/cuda/include) 
                    
target_include_directories(gpu_prime
    PUBLIC 
        $<INSTALL_INTERFACE:inc>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(gpu_prime gmp)