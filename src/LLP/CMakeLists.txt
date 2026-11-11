add_library(LLP INTERFACE)
target_include_directories(LLP INTERFACE .)

target_link_libraries(LLP INTERFACE network LLC)

# Unit tests for LLP packet framing and protocol-lane wire behavior
add_nexusminer_test(
    NAME        packet_framing_test
    SOURCES     packet_framing_test.cpp
    LINK_LIBS   LLP
)

add_nexusminer_test(
    NAME        protocol_lane_header_test
    SOURCES     protocol_lane_header_test.cpp
    LINK_LIBS   LLP
)
