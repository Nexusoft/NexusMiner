cmake_minimum_required(VERSION 3.19)

add_library(TAO STATIC  
src/TAO/Ledger/difficulty.cpp
)

                    
target_include_directories(TAO
    PUBLIC 
        $<INSTALL_INTERFACE:inc>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
    PRIVATE src
)

