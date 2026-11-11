add_library(stats STATIC src/stats/stats_collector.cpp)
                    
target_include_directories(stats
    PUBLIC 
        $<INSTALL_INTERFACE:inc>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(stats PRIVATE config spdlog::spdlog)

add_nexusminer_test(
    NAME        prime_stats_snapshot_test
    SOURCES     prime_stats_snapshot_test.cpp
    LINK_LIBS   stats
)

add_nexusminer_test(
    NAME        stats_collector_typed_test
    SOURCES     stats_collector_typed_test.cpp
    LINK_LIBS   stats config spdlog::spdlog
)
