cmake_minimum_required(VERSION 3.19)

add_library(LLC STATIC  src/LLC/hash/SK/skein.cpp
                        src/LLC/hash/SK/KeccakSponge.cpp
                        src/LLC/hash/SK/KeccakHash.cpp 
                        src/LLC/hash/SK/skein_block.cpp 
                        src/LLC/types/base_uint.cpp 
                        src/LLC/hash/SK/Keccak-compact64.cpp
                        src/LLC/types/bignum.cpp)
                    
target_include_directories(LLC
    PUBLIC 
        $<INSTALL_INTERFACE:inc>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
    PRIVATE src
)