# Build falcon-keygen tool
add_executable(falcon-keygen falcon_keygen.cpp)

target_include_directories(falcon-keygen PUBLIC 
    ${CMAKE_SOURCE_DIR}/src/LLC/inc
    ${CMAKE_SOURCE_DIR}/src/config/inc
)

target_link_libraries(falcon-keygen LLC config spdlog::spdlog)

# Install the tool
install(TARGETS falcon-keygen DESTINATION bin)
