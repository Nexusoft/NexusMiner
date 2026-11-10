cmake_minimum_required(VERSION 3.21)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules")

set(CMAKE_LIBRARY_PATH "${CMAKE_SOURCE_DIR}/libs")

project(NexusMiner VERSION 1.5.1 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

configure_file(src/version.h.in version.h)

# Use vendored CPM.cmake (respects CPM_SOURCE_CACHE for offline builds)
include(${CMAKE_SOURCE_DIR}/cmake/CPM.cmake)

find_package(Threads REQUIRED)

#spdlog
CPMAddPackage(NAME spdlog GITHUB_REPOSITORY gabime/spdlog VERSION 1.17.0 OPTIONS "SPDLOG_BUILD_SHARED OFF")

#nlohmann json
CPMAddPackage(NAME nlohmann_json GITHUB_REPOSITORY nlohmann/json  VERSION 3.11.3)

include_directories(${CMAKE_SOURCE_DIR}/include)
link_directories(${CMAKE_SOURCE_DIR}/libs)

option(WITH_GPU_AMD "Build with AMD gpu workers, Requires HIP and clang" OFF)
option(WITH_GPU_CUDA "Build with Nvidia gpu workers, CUDA needed" OFF)
option(WITH_GPU_HOST_STUBS "Build gpu/worker_hash.cpp against stub CUDA functions (no CUDA toolchain required); mutually exclusive with WITH_GPU_CUDA" OFF)
option(WITH_PRIME "Build with PRIME mining support, BOOST and GMP or MPIR needed" OFF)
# Selected PRIME bigint backend library identifier ("gmp" or "mpir"), resolved
# during dependency discovery and consumed by cpu/gpu subdirectories.
set(NEXUSMINER_PRIME_BIGINT_LIB "")

# Default OFF for macOS (Homebrew OpenSSL is typically dynamic)
if(APPLE)
    option(NEXUSMINER_STATIC_OPENSSL "Build with static OpenSSL" OFF)
else()
    option(NEXUSMINER_STATIC_OPENSSL "Build with static OpenSSL" ON)
endif()

# Enable testing early so subdirectories can register tests
enable_testing()

# DRY helper for test executables (used by subdirectories)
include(${CMAKE_SOURCE_DIR}/cmake/AddTest.cmake)

# Create platform INTERFACE library for scoped compile definitions
add_library(NexusMiner_platform INTERFACE)
if(UNIX)
    target_compile_definitions(NexusMiner_platform INTERFACE UNIX)
endif()
if(WIN32)
    target_compile_definitions(NexusMiner_platform INTERFACE _WIN32_WINNT=0x0601 NOMINMAX _CRT_SECURE_NO_WARNINGS)
endif()

# OpenSSL
if(NEXUSMINER_STATIC_OPENSSL)
    set(OPENSSL_USE_STATIC_LIBS TRUE)
endif()
find_package(OpenSSL REQUIRED)
include_directories(${OPENSSL_INCLUDE_DIR})

# Detect OpenSSL version and set appropriate compile definitions
if(OPENSSL_FOUND)
    message(STATUS "OpenSSL version: ${OPENSSL_VERSION}")
    if(OPENSSL_VERSION VERSION_GREATER_EQUAL "3.0.0")
        message(STATUS "OpenSSL 3.0+ detected - adding compatibility flags")
        target_compile_definitions(NexusMiner_platform INTERFACE OPENSSL_3_0_OR_LATER)
        # Suppress deprecation warnings for OpenSSL 3.0 when using legacy APIs
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
            target_compile_options(NexusMiner_platform INTERFACE -Wno-deprecated-declarations)
        elseif(MSVC)
            target_compile_options(NexusMiner_platform INTERFACE /wd4996)
        endif()
    else()
        message(STATUS "OpenSSL 1.1.1 detected - using legacy API")
    endif()
endif()

#ASIO
CPMAddPackage("gh:chriskohlhoff/asio#asio-1-30-2@1.30.2")

# ASIO doesn't use CMake, we have to configure it manually. Extra notes for using on Windows:
#
# 1) If _WIN32_WINNT is not set, ASIO assumes _WIN32_WINNT=0x0501, i.e. Windows XP target, which is
# definitely not the platform which most users target.
#
# 2) WIN32_LEAN_AND_MEAN is defined to make Winsock2 work.
if(asio_ADDED)
  add_library(asio INTERFACE)

  target_include_directories(asio SYSTEM INTERFACE ${asio_SOURCE_DIR}/asio/include)

  target_compile_definitions(asio INTERFACE ASIO_STANDALONE ASIO_NO_DEPRECATED)

  target_link_libraries(asio INTERFACE Threads::Threads)

  if(WIN32)
    # macro see @ https://stackoverflow.com/a/40217291/1746503
    macro(get_win32_winnt version)
      if(CMAKE_SYSTEM_VERSION)
        set(ver ${CMAKE_SYSTEM_VERSION})
        string(REGEX MATCH "^([0-9]+).([0-9])" ver ${ver})
        string(REGEX MATCH "^([0-9]+)" verMajor ${ver})
        # Check for Windows 10, b/c we'll need to convert to hex 'A'.
        if("${verMajor}" MATCHES "10")
          set(verMajor "A")
          string(REGEX REPLACE "^([0-9]+)" ${verMajor} ver ${ver})
        endif("${verMajor}" MATCHES "10")
        # Remove all remaining '.' characters.
        string(REPLACE "." "" ver ${ver})
        # Prepend each digit with a zero.
        string(REGEX REPLACE "([0-9A-Z])" "0\\1" ver ${ver})
        set(${version} "0x${ver}")
      endif()
    endmacro()

    if(NOT DEFINED _WIN32_WINNT)
      get_win32_winnt(ver)
      set(_WIN32_WINNT ${ver})
    endif()

    message(STATUS "Set _WIN32_WINNT=${_WIN32_WINNT}")

    target_compile_definitions(asio INTERFACE _WIN32_WINNT=${_WIN32_WINNT} WIN32_LEAN_AND_MEAN)
  endif()
endif()

#GPU
if(WITH_GPU_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit)
        include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
        target_compile_definitions(NexusMiner_platform INTERFACE GPU_CUDA_ENABLED GPU_ENABLED)
    else()
        message(STATUS "No CUDA compiler found")
    endif()
endif()

if(WITH_GPU_AMD)
    # Search for rocm in common locations
    list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)
    find_package(hip)
    target_compile_definitions(NexusMiner_platform INTERFACE GPU_AMD_ENABLED GPU_ENABLED)
endif()


#PRIME
if(WITH_PRIME)
    find_package(Boost)
    if(NOT Boost_FOUND)
        message(WARNING "Boost not found. Disabling PRIME support. Run: sudo apt-get install libboost-all-dev")
        set(WITH_PRIME OFF)
    endif()
    
    if(WITH_PRIME)
        if(MSVC)
            find_package(MPIR)
            if(MPIR_FOUND)
                set(NEXUSMINER_PRIME_BIGINT_LIB mpir)
            else()
                message(WARNING "MPIR not found with MSVC toolchain. GMP fallback will be attempted.")
            endif()
        endif()

        if(NOT NEXUSMINER_PRIME_BIGINT_LIB)
            find_package(GMP)
            if(GMP_FOUND)
                set(NEXUSMINER_PRIME_BIGINT_LIB gmp)
            elseif(MSVC)
                message(WARNING "Neither MPIR nor GMP found. Disabling PRIME support. Install MPIR or provide GMPDIR.")
                set(WITH_PRIME OFF)
            else()
                message(WARNING "GMP not found. Disabling PRIME support. Run: sudo apt-get install libgmp-dev")
                set(WITH_PRIME OFF)
            endif()
        endif()
    endif()
    
    if(WITH_PRIME)
        message(STATUS "PRIME big integer backend: ${NEXUSMINER_PRIME_BIGINT_LIB}")
        include_directories(${Boost_INCLUDE_DIRS})
        include(FetchContent)
        FetchContent_Declare(primesieve GIT_REPOSITORY https://github.com/kimwalisch/primesieve.git GIT_TAG master)
        FetchContent_MakeAvailable(primesieve)
        # find_package(primesieve REQUIRED)
        target_compile_definitions(NexusMiner_platform INTERFACE PRIME_ENABLED)
    endif()
endif()


# add submodules
add_subdirectory(src/chrono)
add_subdirectory(src/Util)
add_subdirectory(src/network)
add_subdirectory(src/LLP)
add_subdirectory(src/LLC)
add_subdirectory(src/hash)
add_subdirectory(src/worker)
add_subdirectory(src/config)
add_subdirectory(src/stats)
add_subdirectory(src/protocol)
add_subdirectory(src/node_session)
add_subdirectory(src/cpu)
add_subdirectory(src/fpga)
add_subdirectory(src/TAO)
add_subdirectory(src/tools)


if(WITH_GPU_CUDA OR WITH_GPU_AMD OR WITH_PRIME OR WITH_GPU_HOST_STUBS)
    # Stone — also enter src/gpu when WITH_PRIME=ON (without GPU) so the
    # host-runnable GPU unit tests under that subdirectory build and run on
    # CPU-only CI.  The actual `gpu` static library remains gated on
    # WITH_GPU_CUDA / WITH_GPU_AMD / WITH_GPU_HOST_STUBS inside
    # src/gpu/CMakeLists.txt itself.
    add_subdirectory(src/gpu)
endif()

set(MAIN_SOURCE_FILES src/main.cpp
                src/miner.cpp 
                src/miner_keys.cpp
                src/worker_manager.cpp 
                src/timer_manager.cpp
                src/dual_connection_manager.cpp
                src/colin_agent.cpp
                src/liveness_watchdog.cpp)

add_executable(NexusMiner ${MAIN_SOURCE_FILES})
add_test(NAME nexusminer_version_test COMMAND NexusMiner --version)
string(REPLACE "." "[.]" NEXUSMINER_VERSION_REGEX "${PROJECT_VERSION}")
set_tests_properties(nexusminer_version_test PROPERTIES
    PASS_REGULAR_EXPRESSION "^NexusMiner version: ${NEXUSMINER_VERSION_REGEX}(\n|$)"
)
if (WITH_GPU_AMD AND WITH_PRIME)
    # set(GPU_TARGETS "gfx1030" CACHE STRING "AMD GPU targets")  #this limits gpu hardware targets and can shorten compile time
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
target_link_libraries(NexusMiner NexusMiner_platform chrono network config stats protocol node_session cpu fpga LLP worker TAO asio spdlog::spdlog)
if(WITH_GPU_CUDA)
    target_link_libraries(NexusMiner gpu)
endif()


target_link_libraries(NexusMiner ${OPENSSL_LIBRARIES})
target_link_libraries(NexusMiner Threads::Threads)

# SIM Link DualConnectionManager unit test
add_nexusminer_test(
    NAME        dual_connection_manager_test
    SOURCES     src/dual_connection_manager_test.cpp src/dual_connection_manager.cpp
    LINK_LIBS   # (none required — header-only)
)

# Liveness watchdog (independent io_context stall detector) unit test
add_nexusminer_test(
    NAME        liveness_watchdog_test
    SOURCES     src/liveness_watchdog_test.cpp src/liveness_watchdog.cpp
    LINK_LIBS   chrono asio spdlog::spdlog Threads::Threads
)

if(NOT WITH_GPU_AMD)
    add_nexusminer_test(
        NAME        worker_manager_startup_recovery_test
        SOURCES     src/worker_manager_startup_recovery_test.cpp src/worker_manager.cpp
                    src/timer_manager.cpp src/dual_connection_manager.cpp src/colin_agent.cpp
                    src/miner_keys.cpp
        LINK_LIBS   NexusMiner_platform chrono network config stats protocol node_session cpu fpga
                    LLP worker TAO asio spdlog::spdlog ${OPENSSL_LIBRARIES} Threads::Threads
        INCLUDE_DIRS ${PROJECT_BINARY_DIR}
    )
    if(WITH_GPU_CUDA)
        target_link_libraries(worker_manager_startup_recovery_test PRIVATE gpu)
    endif()
    set_tests_properties(worker_manager_startup_recovery_test PROPERTIES TIMEOUT 20)
endif()

# ============================================================================
# Installation and packaging
# ============================================================================
include(GNUInstallDirs)

# Install the main executable
install(TARGETS NexusMiner
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Install configuration files
# Keep both patterns for compatibility during .conf -> .config migration.
install(DIRECTORY configs/
    DESTINATION ${CMAKE_INSTALL_DATADIR}/nexusminer/configs
    FILES_MATCHING
        PATTERN "*.conf"
        PATTERN "*.config"
)

# Install documentation
install(FILES README.md
    DESTINATION ${CMAKE_INSTALL_DOCDIR}
    OPTIONAL
)

# CPack configuration
set(CPACK_PACKAGE_NAME "NexusMiner")
set(CPACK_PACKAGE_VENDOR "Nexus")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Nexus blockchain miner")
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})

# Set license and readme if they exist
if(EXISTS "${CMAKE_SOURCE_DIR}/LICENSE")
    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")
endif()
if(EXISTS "${CMAKE_SOURCE_DIR}/README.md")
    set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")
endif()

# Platform-specific packaging
if(WIN32)
    set(CPACK_GENERATOR "ZIP;NSIS")
    set(CPACK_NSIS_MODIFY_PATH ON)
    set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)
else()
    set(CPACK_GENERATOR "TGZ;ZIP")
endif()

include(CPack)
