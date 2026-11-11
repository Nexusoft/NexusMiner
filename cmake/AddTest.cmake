# cmake/AddTest.cmake  —  DRY helper for NexusMiner test executables
#
# Usage:
#   add_nexusminer_test(
#       NAME            <test-executable-name>
#       SOURCES         <source1.cpp> [source2.cpp ...]
#       LINK_LIBS       <lib1> [lib2 ...]
#       [INCLUDE_DIRS   <dir1> [dir2 ...]]
#   )
#
# Every test automatically gets:
#   • ${CMAKE_CURRENT_SOURCE_DIR}/inc  (if it exists)
#   • ${CMAKE_SOURCE_DIR}/src
# as PRIVATE include directories, plus any extras from INCLUDE_DIRS.
# The executable is registered with CTest via add_test().

function(add_nexusminer_test)
    cmake_parse_arguments(
        ARG                         # prefix
        ""                          # options  (none)
        "NAME"                      # one-value keywords
        "SOURCES;LINK_LIBS;INCLUDE_DIRS"  # multi-value keywords
        ${ARGN}
    )

    if(NOT ARG_NAME)
        message(FATAL_ERROR "add_nexusminer_test: NAME is required")
    endif()
    if(NOT ARG_SOURCES)
        message(FATAL_ERROR "add_nexusminer_test(${ARG_NAME}): SOURCES is required")
    endif()

    add_executable(${ARG_NAME} ${ARG_SOURCES})

    target_link_libraries(${ARG_NAME} PRIVATE ${ARG_LINK_LIBS})

    # Build the include-directory list: always add the two common paths,
    # then append any caller-supplied extras.
    set(_inc_dirs)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/inc")
        list(APPEND _inc_dirs "${CMAKE_CURRENT_SOURCE_DIR}/inc")
    endif()
    list(APPEND _inc_dirs "${CMAKE_SOURCE_DIR}/src")

    if(ARG_INCLUDE_DIRS)
        list(APPEND _inc_dirs ${ARG_INCLUDE_DIRS})
    endif()

    target_include_directories(${ARG_NAME} PRIVATE ${_inc_dirs})

    add_test(NAME ${ARG_NAME} COMMAND ${ARG_NAME})
endfunction()
