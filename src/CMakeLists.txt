cmake_minimum_required(VERSION 3.12)

file(GLOB_RECURSE main_SRC CONFIGURE_DEPENDS "*.hpp", "*.h" "*.cpp")

add_executable(miner ${main_SRC})

target_link_libraries(miner ${OPENSSL_LIBRARIES})
target_link_libraries(miner Threads::Threads)