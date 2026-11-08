add_library(protocol STATIC src/protocol/solo.cpp
                            src/protocol/pool_base.cpp
                            src/protocol/falcon_wrapper.cpp
                            src/protocol/chacha20_wrapper.cpp
                            src/protocol/session_manager.cpp
                            src/protocol/node_session_context.cpp
                            src/protocol/mining_template_interface.cpp
                            src/protocol/push_notification_handler.cpp
                            src/protocol/height_tracker.cpp
                            src/protocol/packet_builder.cpp
                            src/protocol/epoch_coordinator.cpp
                            src/protocol/packet_router.cpp
                            ${CMAKE_SOURCE_DIR}/src/LLP/colin_ping_handler.cpp
                            ${CMAKE_SOURCE_DIR}/src/LLP/stateless_block_utility.cpp)
                    
target_include_directories(protocol
    PUBLIC 
        $<INSTALL_INTERFACE:inc>    
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/Util/include
        ${CMAKE_SOURCE_DIR}/src/LLP/include
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(protocol PRIVATE network LLP LLC stats config spdlog nlohmann_json::nlohmann_json ${OPENSSL_LIBRARIES})

# ============================================================================
# Protocol test executables — uses add_nexusminer_test() from cmake/AddTest.cmake.
# Every test automatically gets ${CMAKE_CURRENT_SOURCE_DIR}/inc and ${CMAKE_SOURCE_DIR}/src.
# ============================================================================

# Common include-directory shorthands used by many tests
set(_LLP_INC  ${CMAKE_SOURCE_DIR}/src/LLP/inc
              ${CMAKE_SOURCE_DIR}/src/LLP
              ${CMAKE_SOURCE_DIR}/src/LLP/include)
set(_LLC_INC  ${CMAKE_SOURCE_DIR}/src/LLC/inc)

# --- Template & height validation ---
add_nexusminer_test(NAME mining_template_validation_test
    SOURCES     mining_template_validation_test.cpp
    LINK_LIBS   protocol LLP TAO spdlog
    INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/src/TAO/inc ${CMAKE_SOURCE_DIR}/src/LLP/inc)

add_nexusminer_test(NAME height_tracker_test
    SOURCES     height_tracker_test.cpp
    LINK_LIBS   protocol
    INCLUDE_DIRS ${_LLC_INC})

add_nexusminer_test(NAME phase2b_update_height_test
    SOURCES     phase2b_update_height_test.cpp
    LINK_LIBS   protocol LLP spdlog
    INCLUDE_DIRS ${_LLP_INC})

# --- Crypto ---
add_nexusminer_test(NAME chacha20_test
    SOURCES     chacha20_test.cpp
    LINK_LIBS   protocol spdlog ${OPENSSL_LIBRARIES})

add_nexusminer_test(NAME falcon_qtv_parity_test
    SOURCES     falcon_qtv_parity_test.cpp
    LINK_LIBS   ${OPENSSL_LIBRARIES})

# --- Serialization helpers ---
add_nexusminer_test(NAME genesis_serialization_utils_test
    SOURCES     genesis_serialization_utils_test.cpp
    LINK_LIBS   protocol)

add_nexusminer_test(NAME session_start_parser_test
    SOURCES     session_start_parser_test.cpp
    LINK_LIBS   protocol)

# --- Packet / wire protocol ---
add_nexusminer_test(NAME packet_builder_test
    SOURCES     packet_builder_test.cpp
    LINK_LIBS   protocol LLP spdlog
    INCLUDE_DIRS ${_LLP_INC})

add_nexusminer_test(NAME get_block_control_test
    SOURCES     get_block_control_test.cpp ${CMAKE_SOURCE_DIR}/src/miner_keys.cpp
    LINK_LIBS   protocol LLP LLC network asio spdlog stats config ${OPENSSL_LIBRARIES}
    INCLUDE_DIRS ${_LLP_INC} ${_LLC_INC})

add_nexusminer_test(NAME keepalive_v2_test
    SOURCES     keepalive_v2_test.cpp
    LINK_LIBS   protocol LLP spdlog
    INCLUDE_DIRS ${_LLP_INC})

add_nexusminer_test(NAME session_status_wire_test
    SOURCES     session_status_wire_test.cpp
    LINK_LIBS   LLP
    INCLUDE_DIRS ${_LLP_INC})

add_nexusminer_test(NAME node_shutdown_wire_test
    SOURCES     node_shutdown_wire_test.cpp
    LINK_LIBS   protocol LLP spdlog
    INCLUDE_DIRS ${_LLP_INC})

add_nexusminer_test(NAME packet_ingress_preflight_test
    SOURCES     packet_ingress_preflight_test.cpp
    LINK_LIBS   protocol network asio spdlog)

# --- Push notifications & lane detection ---
add_nexusminer_test(NAME push_notification_lane_test
    SOURCES     push_notification_lane_test.cpp
                ${CMAKE_SOURCE_DIR}/src/miner_keys.cpp
    LINK_LIBS   protocol LLP LLC spdlog
    INCLUDE_DIRS ${_LLP_INC})

add_nexusminer_test(NAME fork_vs_tip_change_test
    SOURCES     fork_vs_tip_change_test.cpp
    LINK_LIBS   protocol LLP LLC spdlog
    INCLUDE_DIRS ${_LLP_INC})

# --- Block submission ---
add_nexusminer_test(NAME stateless_block_utility_test
    SOURCES     stateless_block_utility_test.cpp
                ${CMAKE_SOURCE_DIR}/src/miner_keys.cpp
    LINK_LIBS   protocol LLP spdlog
    INCLUDE_DIRS ${_LLP_INC} ${_LLC_INC})

add_nexusminer_test(NAME stateless_metadata_divergence_test
    SOURCES     stateless_metadata_divergence_test.cpp
    LINK_LIBS   protocol LLP spdlog
    INCLUDE_DIRS ${_LLP_INC})

add_nexusminer_test(NAME e2e_block_submission_test
    SOURCES     e2e_block_submission_test.cpp
                ${CMAKE_SOURCE_DIR}/src/miner_keys.cpp
    LINK_LIBS   protocol LLP spdlog ${OPENSSL_LIBRARIES}
    INCLUDE_DIRS ${_LLP_INC} ${_LLC_INC})

add_nexusminer_test(NAME prepare_block_submission_from_solved_test
    SOURCES     prepare_block_submission_from_solved_test.cpp
    LINK_LIBS   protocol LLP spdlog
    INCLUDE_DIRS ${_LLP_INC} ${_LLC_INC}
                 ${CMAKE_SOURCE_DIR}/src/worker)

add_nexusminer_test(NAME submit_result_gate_test
    SOURCES     submit_result_gate_test.cpp
    LINK_LIBS   protocol)

add_nexusminer_test(NAME first_block_acceptance_test
    SOURCES     first_block_acceptance_test.cpp
                ${CMAKE_SOURCE_DIR}/src/miner_keys.cpp
    LINK_LIBS   protocol network LLP asio spdlog ${OPENSSL_LIBRARIES}
    INCLUDE_DIRS ${_LLP_INC} ${_LLC_INC})

add_nexusminer_test(NAME cross_architecture_serialization_test
    SOURCES     cross_architecture_serialization_test.cpp
                ${CMAKE_SOURCE_DIR}/src/miner_keys.cpp
    LINK_LIBS   protocol network LLP asio spdlog ${OPENSSL_LIBRARIES}
    INCLUDE_DIRS ${_LLP_INC} ${_LLC_INC})

add_nexusminer_test(NAME multi_miner_session_isolation_test
    SOURCES     multi_miner_session_isolation_test.cpp
                ${CMAKE_SOURCE_DIR}/src/miner_keys.cpp
    LINK_LIBS   protocol network LLP asio spdlog ${OPENSSL_LIBRARIES}
    INCLUDE_DIRS ${_LLP_INC} ${_LLC_INC})

add_nexusminer_test(NAME mined_block_cache_dedup_test
    SOURCES     ${CMAKE_SOURCE_DIR}/src/stats/mined_block_cache_test.cpp
    LINK_LIBS   protocol LLP stats spdlog
    INCLUDE_DIRS ${_LLP_INC} ${_LLC_INC}
                 ${CMAKE_SOURCE_DIR}/src/stats/inc)

# --- Session & dedup recovery ---
add_nexusminer_test(NAME get_block_dedup_recovery_test
    SOURCES     get_block_dedup_recovery_test.cpp
    LINK_LIBS   protocol LLP spdlog
    INCLUDE_DIRS ${_LLP_INC})

add_nexusminer_test(NAME degraded_recovery_test
    SOURCES     degraded_recovery_test.cpp
    LINK_LIBS   protocol LLP spdlog
    INCLUDE_DIRS ${_LLP_INC})

add_nexusminer_test(NAME solo_auth_resync_test
    SOURCES     solo_auth_resync_test.cpp
    LINK_LIBS   protocol spdlog)

# --- Node session context & epochs ---
add_nexusminer_test(NAME node_session_context_test
    SOURCES     node_session_context_test.cpp
    LINK_LIBS   protocol network asio spdlog)

add_nexusminer_test(NAME epoch_coordinator_test
    SOURCES     epoch_coordinator_test.cpp
    LINK_LIBS   protocol network asio spdlog)

# --- MerkleRoot feed guard ---
add_nexusminer_test(NAME merkle_root_feed_guard_test
    SOURCES     merkle_root_feed_guard_test.cpp
    LINK_LIBS   protocol
    INCLUDE_DIRS ${_LLC_INC})

# --- Session identity ---
add_nexusminer_test(NAME session_identity_test
    SOURCES     session_identity_test.cpp
    LINK_LIBS   protocol LLC
    INCLUDE_DIRS ${_LLC_INC})

add_nexusminer_test(NAME session_binding_test
    SOURCES     session_binding_test.cpp
    LINK_LIBS   protocol network asio spdlog)

# --- NEW_ROUND recovery debounce (2s symmetric gate) ---
# Host-only: no network connection, no GPU.  Drives the asio timer with
# run_for() to validate that the deferred recovery GET_BLOCK fires /
# is cancelled correctly for each operator-identified race-order scenario.
add_nexusminer_test(NAME new_round_recovery_debounce_test
    SOURCES     new_round_recovery_debounce_test.cpp
                ${CMAKE_SOURCE_DIR}/src/miner_keys.cpp
    LINK_LIBS   protocol LLP LLC network asio spdlog ${OPENSSL_LIBRARIES}
    INCLUDE_DIRS ${_LLP_INC} ${_LLC_INC})

add_nexusminer_test(NAME post_adoption_suppression_test
    SOURCES     post_adoption_suppression_test.cpp
                ${CMAKE_SOURCE_DIR}/src/miner_keys.cpp
    LINK_LIBS   protocol LLP LLC network asio spdlog ${OPENSSL_LIBRARIES}
    INCLUDE_DIRS ${_LLP_INC} ${_LLC_INC})

add_nexusminer_test(NAME same_tip_reconfirmation_test
    SOURCES     same_tip_reconfirmation_test.cpp
                ${CMAKE_SOURCE_DIR}/src/miner_keys.cpp
    LINK_LIBS   protocol LLP LLC network asio spdlog ${OPENSSL_LIBRARIES}
    INCLUDE_DIRS ${_LLP_INC} ${_LLC_INC})

# --- Packet router ---
add_nexusminer_test(NAME packet_router_test
    SOURCES     packet_router_test.cpp
    LINK_LIBS   protocol LLP spdlog
    INCLUDE_DIRS ${_LLP_INC}
                 ${CMAKE_SOURCE_DIR}/src/network/inc)
