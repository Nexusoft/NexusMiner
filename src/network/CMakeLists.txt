add_library(network STATIC "src/network/create_component.cpp"
                           "src/network/endpoint.cpp"
                           "src/network/tls/tls_context.cpp"
                           "src/network/tls/tls_socket_wrapper.cpp")

target_include_directories(network
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
        $<INSTALL_INTERFACE:inc>
    PRIVATE src
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(network NexusMiner_platform asio spdlog ${OPENSSL_LIBRARIES})