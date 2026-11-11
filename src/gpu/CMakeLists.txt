if (WITH_GPU_CUDA)
    add_definitions(-DGPU_CUDA_ENABLED)
    add_library(gpu STATIC "src/gpu/worker_hash.cpp" "src/gpu/cuda_memory.cpp" "src/gpu/cuda_hash/util.cu" "src/gpu/cuda_hash/constants.cu" "src/gpu/cuda_hash/sk1024.cu")
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
    if(MSVC)
        set_target_properties(gpu PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    endif()
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
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/cuda_prime)

    if(WITH_PRIME)
        target_include_directories(gpu PRIVATE ${CMAKE_SOURCE_DIR}/src/cpu/inc)
    endif()

    target_link_libraries(gpu LLC worker LLP stats config TAO spdlog::spdlog)


    if(WITH_PRIME)
        target_link_libraries(gpu cpu)
        target_link_libraries(gpu libprimesieve-static)
        if(NEXUSMINER_PRIME_BIGINT_LIB)
            target_link_libraries(gpu ${NEXUSMINER_PRIME_BIGINT_LIB})
        else()
            message(FATAL_ERROR "WITH_PRIME is enabled but no PRIME bigint backend was selected; install GMP/MPIR and re-run CMake configure")
        endif()
        
    endif()
endif()

# ---- Host-stubs build: GPU workers against stub CUDA functions ----
# WITH_GPU_HOST_STUBS=ON builds the host-side portions of the GPU hash and
# prime workers against plain-C++ stubs (no nvcc / CUDA runtime required).
# This lets contributors without a CUDA toolchain validate:
#   * The static_assert(offsetof(Block_data, nNonce) == 26*8) layout contract.
#   * The linear (not quadratic) hashes-done accumulation invariant.
#   * The keccak-mismatch fault-counter logic.
#   * GPU-prime worker lifecycle/stat publication still compiles without nvcc.
# WITH_GPU_HOST_STUBS and WITH_GPU_CUDA are mutually exclusive (stub symbols
# would conflict with the real kernel object files).
if(WITH_GPU_HOST_STUBS AND NOT WITH_GPU_CUDA)
    # GPU_CUDA_ENABLED keeps the #include guards in worker_hash.cpp consistent
    # with a real CUDA build; the symbol is intentionally the same so no
    # #ifdef ladder is needed inside the production source.
    add_library(gpu STATIC
        "src/gpu/worker_hash.cpp"
        "src/gpu/worker_prime.cpp"
        "src/gpu/prime/chain.cpp"
        "src/gpu/host_stubs/util_stub.cpp"
        "src/gpu/host_stubs/sk1024_stub.cpp"
        "src/gpu/host_stubs/prime_kernel_stub.cpp")

    target_compile_definitions(gpu PRIVATE GPU_CUDA_ENABLED GPU_HOST_STUBS)

    target_include_directories(gpu
        PUBLIC
            $<INSTALL_INTERFACE:inc>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
        PRIVATE
            ${CMAKE_SOURCE_DIR}/src
            ${CMAKE_SOURCE_DIR}/src/cpu/inc
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu)

    target_link_libraries(gpu LLC worker LLP stats config TAO spdlog::spdlog)

    # Integration test — drives Worker_hash::step_once() via the stub kernel.
    add_executable(worker_hash_integration_test
        "test/worker_hash_integration_test.cpp")

    target_include_directories(worker_hash_integration_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/inc
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu
        ${CMAKE_SOURCE_DIR}/src/config/inc
        ${CMAKE_SOURCE_DIR}/src/worker/inc
        ${CMAKE_SOURCE_DIR}/src/stats/inc)

    target_link_libraries(worker_hash_integration_test
        gpu LLC worker LLP stats config TAO spdlog::spdlog asio)

    add_test(NAME worker_hash_integration_test COMMAND worker_hash_integration_test)

    add_executable(worker_prime_host_stubs_test
        "test/worker_prime_host_stubs_test.cpp")

    target_include_directories(worker_prime_host_stubs_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/inc
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/cpu/inc
        ${CMAKE_SOURCE_DIR}/src/config/inc
        ${CMAKE_SOURCE_DIR}/src/worker/inc
        ${CMAKE_SOURCE_DIR}/src/stats/inc)

    target_link_libraries(worker_prime_host_stubs_test
        gpu LLC worker LLP stats config TAO spdlog::spdlog asio)

    add_test(NAME worker_prime_host_stubs_test COMMAND worker_prime_host_stubs_test)

    # Prime-channel vOffsets regression test for the host-stubs preset.  This is
    # intentionally CUDA-free and does not require WITH_PRIME because it compiles
    # the shared prime_validation.cpp path directly.
    add_executable(prime_voffsets_length8_host_stubs_test
        "test/prime_voffsets_length8_host_stubs_test.cpp"
        "${CMAKE_SOURCE_DIR}/src/cpu/src/cpu/prime_validation.cpp")

    target_include_directories(prime_voffsets_length8_host_stubs_test PRIVATE
        ${CMAKE_SOURCE_DIR}/src/cpu/inc
        ${CMAKE_SOURCE_DIR}/src/LLC/inc
        ${CMAKE_SOURCE_DIR}/src/mining
        ${CMAKE_CURRENT_SOURCE_DIR}/inc)

    target_link_libraries(prime_voffsets_length8_host_stubs_test
        NexusMiner_platform LLC spdlog::spdlog)

    add_test(NAME prime_voffsets_length8_host_stubs_test COMMAND prime_voffsets_length8_host_stubs_test)
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


# Stone — host-runnable GPU unit tests.
# These tests exercise the host-only portions of the GPU code (currently
# `src/gpu/src/gpu/prime/chain.cpp` — the host-side Chain class used by the
# CPU-fallback batch test path and by the get_chains/get_long_chains
# converters).  They are NOT gated on WITH_GPU_CUDA because the files under
# test compile cleanly without any CUDA toolchain — chain.hpp only includes
# `prime_common.hpp` plus standard headers.
#
# Gated on WITH_PRIME because the GPU-prime stack (and therefore chain.hpp)
# is meaningful only in prime builds.
# Header-only.  Built for host-stubs and prime presets so the region-offset
# formula can be checked without nvcc.
add_executable(gpu_sieve_region_offset_test
    sieve_region_offset_test.cpp)
target_include_directories(gpu_sieve_region_offset_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
add_test(NAME gpu_sieve_region_offset_test COMMAND gpu_sieve_region_offset_test)

add_executable(gpu_chain_slot_test chain_slot_test.cpp)
target_include_directories(gpu_chain_slot_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(gpu_chain_slot_test PRIVATE Threads::Threads)
add_test(NAME gpu_chain_slot_test COMMAND gpu_chain_slot_test)

if(WITH_PRIME)
    add_executable(gpu_chain_test
        gpu_chain_test.cpp
        src/gpu/prime/chain.cpp)
    target_include_directories(gpu_chain_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/inc
        ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME gpu_chain_test COMMAND gpu_chain_test)
endif()



                    
