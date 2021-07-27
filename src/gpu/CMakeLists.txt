cmake_minimum_required(VERSION 3.19)

add_library(gpu STATIC "src/gpu/worker_hash.cpp" "src/gpu/cuda/util.cu" "src/gpu/cuda/constants.cu" "src/gpu/cuda/sk1024.cu")

set_target_properties(gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET gpu PROPERTY CUDA_ARCHITECTURES 61 72)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) 
                    
target_include_directories(gpu
    PUBLIC 
        $<INSTALL_INTERFACE:inc>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(gpu LLC worker LLP stats config OpenSSL::Crypto OpenSSL::applink)