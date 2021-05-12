cmake_minimum_required(VERSION 3.12)

file(GLOB main_SRC CONFIGURE_DEPENDS "*.h" "*.cpp")

find_package(OpenSSL REQUIRED)
include_directories(${OPENSSL_INCLUDE_DIR})


add_executable(miner ${main_SRC})

target_link_libraries(miner ${OPENSSL_LIBRARIES})