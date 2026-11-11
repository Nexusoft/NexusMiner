add_library(worker INTERFACE)
target_include_directories(worker INTERFACE .)

target_link_libraries(worker INTERFACE LLP LLC hash)

add_nexusminer_test(
    NAME        template_feed_test
    SOURCES     template_feed_test.cpp
    LINK_LIBS   worker
)

add_nexusminer_test(
    NAME        node_shutdown_policy_test
    SOURCES     node_shutdown_policy_test.cpp
    LINK_LIBS   worker
)
