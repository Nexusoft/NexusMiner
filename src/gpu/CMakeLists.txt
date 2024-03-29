cmake_minimum_required(VERSION 3.21)

if (WITH_GPU_CUDA)
    add_definitions(-DGPU_CUDA_ENABLED)
    add_library(gpu STATIC "src/gpu/worker_hash.cpp" "src/gpu/cuda_hash/util.cu" "src/gpu/cuda_hash/constants.cu" "src/gpu/cuda_hash/sk1024.cu")
    if(WITH_PRIME)
        target_sources(gpu PRIVATE src/gpu/worker_prime.cpp src/gpu/prime/prime.cpp src/gpu/prime/prime_tests.cpp
            src/gpu/prime/chain.cpp src/gpu/prime/sieve.cpp
            src/gpu/cuda_prime/cuda_chain.cu
            src/gpu/cuda_prime/sieve.cu
            src/gpu/cuda_prime/sieve_impl.cu  src/gpu/cuda_prime/find_chain.cu
            src/gpu/prime/small_sieve_tools.cpp
            src/gpu/cuda_prime/fermat_prime/fermat_prime.cu src/gpu/cuda_prime/fermat_prime/fermat_prime_impl.cu
            src/gpu/cuda_prime/fermat_prime/fermat_utils.cu
        )
    endif()

    set_target_properties(gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET gpu PROPERTY CUDA_ARCHITECTURES 61 75 86)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-line-info")
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) 
    include_directories(${CUDA_HOME}/samples/common/inc/)

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
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>)

    target_link_libraries(gpu LLC worker LLP stats config TAO spdlog::spdlog)


    if(WITH_PRIME)
        target_link_libraries(gpu libprimesieve-static)
        if(WIN32)
            target_link_libraries(gpu mpir)
        else()
        target_link_libraries(gpu gmp)
        endif()
        
    endif()
endif()

if (WITH_GPU_AMD)
    add_definitions(-DGPU_AMD_ENABLED)
endif()

#if (WITH_GPU_AMD AND WITH_PRIME)
#    set(GPU_FILES 
#    src/gpu/cuda_prime/cuda_chain.cu  
#    src/gpu/cuda_prime/sieve.cu
#    src/gpu/cuda_prime/sieve_impl.cu  src/gpu/cuda_prime/find_chain.cu
#    src/gpu/cuda_prime/fermat_prime/fermat_prime.cu src/gpu/cuda_prime/fermat_prime/fermat_prime_impl.cu
#    src/gpu/cuda_prime/fermat_prime/fermat_utils.cu)

#    set(HOST_FILES src/gpu/worker_prime.cpp src/gpu/prime/prime.cpp src/gpu/prime/prime_tests.cpp
#    src/gpu/prime/chain.cpp src/gpu/prime/sieve.cpp src/gpu/prime/small_sieve_tools.cpp)
#    set_source_files_properties(${GPU_FILES} PROPERTIES COMPILE_FLAGS "-fgpu-rdc" )

    #set_source_files_properties(${GPU_FILES} PROPERTIES COMPILE_OPTIONS "-x hip")
    # set_source_files_properties(${HOST_FILES} PROPERTIES COMPILE_OPTIONS "-fPIC")

#    set_source_files_properties(${GPU_FILES} PROPERTIES LANGUAGE CXX)

    #add_library(gpu STATIC)
    #target_sources(gpu PRIVATE ${GPU_FILES} ${HOST_FILES})

    
#endif()


                    
