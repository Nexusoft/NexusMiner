#include "worker_manager.hpp"
#include "worker/worker.hpp"
#include "worker/template_feed.hpp"
#include "worker/node_shutdown_policy.hpp"
#include "colin_agent.hpp"
#include "cpu/worker_hash.hpp"

#include "fpga/worker_hash.hpp"
#ifdef GPU_CUDA_ENABLED
#include "gpu/worker_hash.hpp"
#endif
#if defined(PRIME_ENABLED) && defined(GPU_ENABLED)
#include "gpu/worker_prime.hpp"
#endif
#ifdef PRIME_ENABLED
#include "cpu/worker_prime.hpp"
#include "cpu/prime/prime_mining_engine.hpp"
#include "cpu/prime/chain_sieve.hpp"
#endif
#include "packet.hpp"
#include "config/config.hpp"
#include "config/types.hpp"
#include "LLP/block.hpp"
#include "mining/client_block.h"
#include "stats/stats_printer_console.hpp"
#include "stats/stats_printer_file.hpp"
#include "stats/stats_collector.hpp"
#include "miner_keys.hpp"
#include "protocol/solo.hpp"
#include "protocol/protocol_constants.hpp"
#include "protocol/session_status_policy.hpp"
#include "protocol/get_block_reason.hpp"
#include "worker/block_header_utils.hpp"
#include <asio/steady_timer.hpp>
#include <variant>
#include <algorithm>
#include <limits>
#include <random>
#include <iomanip>
#include <sstream>

namespace nexusminer
{

// Recovery window: if no template arrives within this many seconds after a
// GET_BLOCK recovery is initiated, the health monitor escalates to hard recovery.
// Channel-aware: Prime blocks genuinely take 2-5+ min, so a 60 s window causes
// spurious escalations during normal long Prime blocks. Hash blocks arrive every
// ~18 s so 60 s (≈ 3 blocks) is appropriate for Hash.
namespace {
    constexpr int64_t RECOVERY_WINDOW_SECONDS_HASH  = 300;   // Hash can still see long burst-driven gaps; allow 5 min before forced reconnect
    constexpr int64_t RECOVERY_WINDOW_SECONDS_PRIME = 300;   // Prime blocks also tolerate long gaps; align both channels on the same 5 min window
    // Minimum interval between successive GET_BLOCK sends by the health monitor
    // during an active recovery.  Prevents rapid-fire GETs while still allowing
    // periodic retries if the first attempt is not answered.
    // 15s gives 4 attempts in a 60s hash-block window (t=0, t=15, t=30, t=45),
    // well above the node's 2s rate-limit floor (GET_BLOCK_COOLDOWN_SECONDS).
    constexpr int64_t RECOVERY_RESEND_INTERVAL_SECONDS = 15;

    // Minimum interval between successive hard escalations (stop workers + hard recovery).
    // Prevents re-escalation before the new epoch's GET_BLOCK has had time to be answered.
    // Must be at least RECOVERY_RESEND_INTERVAL_SECONDS * 3 so that a GET_BLOCK sent during
    // escalation has time to be answered before we escalate again (minimum = 15 × 3 = 45s).
    // 90s provides 2× the minimum — comfortable margin for slow nodes.
    constexpr int64_t MIN_ESCALATION_INTERVAL_SECONDS = 90;

    // Multiplier on RECOVERY_RESEND_INTERVAL_SECONDS for the per-epoch no-re-escalate window.
    // 3 × 15s = 45s gives the new epoch's GET_BLOCK at least 3 resend intervals to be answered
    // before another escalation is permitted.
    constexpr int64_t EPOCH_NO_ESCALATE_MULTIPLIER = 3;

    // Push-notification liveness threshold: if a push notification
    // (PRIME/HASH_BLOCK_AVAILABLE) was received within this window, the TCP session
    // is demonstrably alive and authenticated.  PUSH is the sole authoritative
    // signal for session liveness — keepalive ACK is diagnostic only.
    constexpr int64_t PUSH_ALIVE_THRESHOLD_SECONDS = protocol::ProtocolConstants::PUSH_LIVENESS_THRESHOLD_SECONDS;

    // Push-alive guard in retry_connect(): suppress TCP reconnect if a push
    // notification was received within this window, proving the connection is alive.
    // Aligned with PUSH_ALIVE_THRESHOLD_SECONDS to prevent conflicting liveness
    // decisions: retry_connect() must not tear down a session that
    // check_template_health() still considers alive (Bug 2 fix).
    constexpr int64_t RETRY_CONNECT_PUSH_LIVE_SECONDS = PUSH_ALIVE_THRESHOLD_SECONDS;

    // Aggressive secondary reconnect delay during degraded mode.
    // Unified height drift threshold: if HeightTracker.unified_height exceeds
    // template.block.nHeight by more than this many blocks, the template is
    // presumed stale (hashPrevBlock is wrong) and must be discarded.
    // On a 3-channel Nexus blockchain, the unified height advances whenever any
    // channel (Prime, Hash, Stake) finds a block. A drift of 1-3 blocks between
    // push notification and new BLOCK_DATA template is normal during the propagation
    // window. Set threshold to 5 to avoid false-positive template discards.
    constexpr uint32_t UNIFIED_DRIFT_THRESHOLD = 5;
    constexpr int64_t FORCED_RETRY_JITTER_MIN_MS = 100;
    constexpr int64_t FORCED_RETRY_JITTER_MAX_MS = 250;

}

Worker_manager::Worker_manager(std::shared_ptr<asio::io_context> io_context, Config& config,
    chrono::Timer_factory::Sptr timer_factory, network::Socket::Sptr socket)
: m_io_context{std::move(io_context)}
, m_config{config}
, m_socket{std::move(socket)}
, m_logger{spdlog::get("logger")}
, m_stats_collector{stats::make_collector(m_config)}
, m_timer_manager{std::move(timer_factory)}
{
    // Falcon miner authentication is mandatory for solo mining
    if (!m_config.has_miner_falcon_keys())
    {
        m_logger->error("[Worker_manager] CRITICAL: Falcon authentication keys are required for solo mining");
        m_logger->error("[Worker_manager] Legacy authentication has been removed. Please configure Falcon keys:");
        m_logger->error("[Worker_manager]   1. Generate keys: ./NexusMiner --create-keys");
        m_logger->error("[Worker_manager]   2. Add keys to miner.conf (falcon_miner_pubkey and falcon_miner_privkey)");
        m_logger->error("[Worker_manager]   3. Whitelist your public key on the node:");
        m_logger->error("[Worker_manager]      - Config file: Add 'minerallowkey=<pubkey>' to nexus.conf");
        m_logger->error("[Worker_manager]      - Command line: Start nexus with -minerallowkey=<pubkey>");
        m_logger->error("[Worker_manager] See docs/falcon_authentication.md for detailed instructions");
        throw std::runtime_error("Falcon authentication keys are required for solo mining");
    }

        m_logger->info("[Worker_manager] Configuring Falcon miner authentication");

        std::vector<uint8_t> pubkey, privkey;
        if (!keys::from_hex(m_config.get_miner_falcon_pubkey(), pubkey) ||
            !keys::from_hex(m_config.get_miner_falcon_privkey(), privkey))
        {
            m_logger->error("[Worker_manager] CRITICAL: Failed to parse Falcon keys from config - invalid hex format");
            m_logger->error("[Worker_manager] Keys must be valid hexadecimal strings");
            m_logger->error("[Worker_manager] Use ./NexusMiner --create-keys to generate valid keys");
            throw std::runtime_error("Invalid Falcon key format in configuration");
        }

        // Create the primary lane-scoped NodeSession.
        // Lane selection follows the configured node port; failover happens by
        // reconnecting to a secondary node endpoint with a full auth cycle.
        m_primary_node_session = std::make_shared<NodeSession>(
            m_io_context,
            m_config,
            m_socket,
            m_stats_collector,
            "PRIMARY",
            &m_sim_link);  // Pass DualConnectionManager for lane health tracking

        // Create EpochCoordinator and wire it to the session manager before any auth begins.
        m_epoch_coordinator = std::make_shared<protocol::EpochCoordinator>();
        m_primary_node_session->set_epoch_coordinator(m_epoch_coordinator);

        // Configure miner keys
        m_primary_node_session->set_miner_keys(pubkey, privkey);

        // Configure Tritium GenesisHash if provided
        if (m_config.has_tritium_genesis()) {
            std::vector<uint8_t> genesis;
            if (keys::from_hex(m_config.get_tritium_genesis(), genesis)) {
                m_primary_node_session->set_tritium_genesis(genesis);
                m_logger->info("[Worker_manager] Tritium GenesisHash configured for reward binding");
            } else {
                m_logger->warn("[Worker_manager] Failed to parse Tritium GenesisHash - invalid hex format");
            }
        }

        // Configure keepalive interval
        m_primary_node_session->set_keepalive_interval(get_effective_keepalive_interval());
        m_logger->info("[Worker_manager] Keepalive interval: {} hours", get_effective_keepalive_interval());

        m_logger->info("[Worker_manager] ChaCha20 encryption: ENABLED (ALWAYS ON - core security)");
        m_logger->info("[Worker_manager] Disposable Falcon signing: ENABLED (ALWAYS ON - core protocol, 0 blockchain overhead)");

        // Configure reward address for stateless mining (MINER_SET_REWARD protocol)
        if (m_config.has_reward_address()) {
            m_primary_node_session->set_reward_address(m_config.get_reward_address());
            m_logger->info("[Worker_manager] Reward address configured: {}", m_config.get_reward_address());
        } else {
            m_logger->debug("[Worker_manager] No reward address configured - using node default");
        }

        m_logger->info("[Worker_manager] Falcon keys loaded from config");
        m_logger->info("[Worker_manager] Auth address: {}", m_config.get_local_ip());

        /* ========== REGISTER TEMPLATE DISTRIBUTION HANDLER ========== */
        /* Connects protocol layer (validated templates) to worker layer (mining threads) */
        /* This lambda is called by the template feed handler (PR #62) when templates arrive */
        m_primary_node_session->set_template_handler(
            [this](const ::LLP::CBlock& block, uint32_t nBits) -> bool {
                if (m_terminal_stop_requested.load(std::memory_order_acquire)) {
                    m_logger->warn("[Worker_manager] Ignoring template after terminal stop request");
                    return false;
                }

                if (m_node_shutdown_quarantine.load(std::memory_order_acquire)) {
                    const auto current_generation = m_session_generation.load(std::memory_order_acquire);
                    const auto shutdown_generation = m_node_shutdown_generation.load(std::memory_order_acquire);
                    if (current_generation <= shutdown_generation) {
                        m_logger->warn("[Worker_manager] Ignoring template from NODE_SHUTDOWN generation "
                                       "(current={}, shutdown={})",
                                       current_generation, shutdown_generation);
                        return false;
                    }
                }

                // Dead-on-arrival gate: mirrors the node's Prime-channel origins
                // check (Block::VerifyWork(): nVersion >= 5 && ProofHash() <
                // bnPrimeMinOrigins => "prime origins below 1016-bits"). This is a
                // per-TEMPLATE property (depends only on fixed header fields, never
                // on nNonce), so it is evaluated once here -- before any worker is
                // handed the template -- rather than per solved candidate. Without
                // this gate the miner can spend minutes mining a Prime chain that
                // satisfies every miner-side check yet is unconditionally rejected
                // by the node's Check() for a reason unrelated to difficulty.
                if (!HasSufficientPrimeOrigins(block)) {
                    m_logger->error("[Worker_manager] ⚠️  Template DEAD ON ARRIVAL: height={} channel={} "
                                    "reason=PRIME_ORIGIN_TOO_LOW (ProofHash below bnPrimeMinOrigins / 1016-bits) "
                                    "-- discarding without mining, requesting fresh template",
                                    block.nHeight, block.nChannel);
                    return false;
                }

                std::size_t worker_count = 0;
                {
                    std::lock_guard lock(m_worker_mutex);
                    worker_count = m_workers.size();
                }
                m_logger->info("[Worker_manager] ═══════════════════════════════════════");
                m_logger->info("[Worker_manager] DISTRIBUTING TEMPLATE TO {} WORKERS", worker_count);
                m_logger->info("[Worker_manager]   Height:     {}", block.nHeight);
                m_logger->info("[Worker_manager]   Channel:    {} ({})", 
                              block.nChannel,
                              (block.nChannel == 1) ? "prime" : "hash");
                m_logger->info("[Worker_manager]   Difficulty: 0x{:08x}", nBits);
                m_logger->info("[Worker_manager]   Merkle:     {}...",
                              block.hashMerkleRoot.ToString().substr(0, 16));
                m_logger->info("[Worker_manager]   PrevHash:   {}...",
                              block.hashPrevBlock.ToString().substr(0, 20));
                m_logger->info("[Worker_manager] ═══════════════════════════════════════");

                // Note: Template feed debounce is now handled in MiningTemplateInterface
                // (unified dedup gate). This handler is only called after the template
                // passes the debounce check, so no additional checking is needed here.

                // Update mined-block cache confirmations based on new chain height.
                m_mined_block_cache.update_confirmations(block.nHeight);

                // ═══════════════════════════════════════════════════════════════
                // Recovery state (WAITING_TEMPLATE) is cleared by the phase transition
                // in clear_recovery_state() below (workers_fed > 0 branch).
                // Do not clear here — we must confirm workers actually received the
                // template first before exiting recovery state.
                // ═══════════════════════════════════════════════════════════════

                // Recovery restart guard: workers may have been stopped and reset
                // by stop_all_workers().  Restart them now so set_block() below
                // actually starts mining threads; without this the template is
                // silently dropped and workers_fed falsely reads 0 keeping the
                // miner in a doom loop.
                //
                // Exclude the terminal stop: mining is intentionally halted there
                // (stop_all_workers banner/logs), and a late/in-flight template
                // must not recreate workers and resume mining unexpectedly.
                size_t workers_fed = 0;
                size_t total_worker_count = 0;
                std::vector<std::shared_ptr<Worker>> worker_snapshot;
                std::shared_ptr<WorkPackage> work_package;
                std::shared_ptr<WorkerTemplateFeed> feed_snapshot;
                {
                    std::lock_guard lock(m_worker_mutex);
                    if (is_recovery_active() && !is_degraded_terminal() &&
                        !m_recovery_workers_spawned && m_workers.empty()) {
                        m_logger->info("[Worker_manager] Recovery mode: restarting workers before feeding recovery template");
                        create_workers_locked();
                        m_recovery_workers_spawned = !m_workers.empty();  // set AFTER success for exception safety
                    }

                    /* Safety check - workers should be created by now */
                    if (m_workers.empty()) {
                        m_logger->error("[Worker_manager] CRITICAL: No workers available for mining!");
                        m_logger->error("[Worker_manager]   Workers may not be initialized yet");
                        m_logger->error("[Worker_manager]   Template will be lost - mining cannot start");
                        return false;
                    }

                    /* Create shared WorkPackage once for all workers */
                    work_package = std::make_shared<WorkPackage>(block, nBits);
                    m_logger->debug("[Worker_manager] Created shared WorkPackage (block height: {}, nBits: 0x{:08x})",
                                    block.nHeight, nBits);

#ifdef PRIME_ENABLED
                    /* Optimization: For prime channel, precompute base hash (Skein+Keccak) once
                     * instead of having each worker compute it independently */
                    if (block.nChannel == 1) {  // Prime channel
                        Block_data temp_block{block};
                        work_package->set_prime_base_hash(temp_block.GetPrimeBaseHash());

                        m_logger->debug("[Worker_manager] Precomputed prime base hash for all workers");
                    }
#endif

                    // Stone 4: snapshot the worker vector and the feed pointer under
                    // m_worker_mutex, then release it before fanning out.  The
                    // shared template feed is published into below; per-worker
                    // set_block() shim calls (for unmigrated subclasses) run
                    // *outside* the parent mutex so the stats path no longer
                    // contends with N CV notifications during template handoff.
                    //
                    // Note: we snapshot the full vector (including any nulls)
                    // so the loop index below still matches each worker's
                    // original m_workers position — the warn() messages are
                    // diagnostic and need that mapping to stay aligned with
                    // m_internal_id (== creation index).
                    worker_snapshot = m_workers;
                    total_worker_count = 0;
                    for (auto const& w : worker_snapshot) {
                        if (w) ++total_worker_count;
                    }
                    feed_snapshot = m_template_feed;
                }  // ── m_worker_mutex released here ───────────────────────────

                // Build the found-block callback ONCE per template (instead of
                // re-capturing it per worker as the prior fanout did).  Both
                // the WorkerTemplateFeed publish path and the legacy per-worker
                // set_block() shim path share the same handler instance, which
                // also makes future Stone-5 migrations of individual worker
                // subclasses transparent — they pull the same handler from the
                // published epoch.
                //
                // Option D: the handler captures a weak_ptr to its own epoch
                // so the FIRST find on this template can mark_consumed() at
                // the top of the lambda.  This is the cross-worker /
                // cross-epoch source of truth that future feed-consumer
                // workers will gate on via WorkerTemplateFeed::latest_unconsumed().
                // Legacy workers also self-track an equivalent per-worker
                // flag in their own mining loops; the two work together.
                //
                // The epoch_ptr is built BEFORE on_found so on_found can
                // capture a weak_ptr; ownership is then transferred into the
                // feed (or — when no feed exists — destroyed at the end of
                // this scope, in which case the lambda's weak_ptr expires and
                // mark_consumed() degrades to a no-op, but the per-worker
                // flag still fires the consume on the legacy path).
                auto epoch_ptr = std::make_shared<TemplateEpoch>();
                epoch_ptr->work_package = work_package;
                std::weak_ptr<const TemplateEpoch> epoch_weak = epoch_ptr;

                Worker::Block_found_handler on_found =
                    [this, epoch_weak](std::uint32_t id, std::unique_ptr<Block_data> block_data)
                {
                    // Option D: mark the epoch consumed BEFORE the validity
                    // gate.  Even if the prev-hash check below discards the
                    // solution, the worker should not keep grinding the same
                    // template — a "tip moved out from under us" mismatch
                    // means the template is structurally spent regardless of
                    // whether THIS solution submits.
                    if (auto epoch = epoch_weak.lock()) {
                        const bool was_already = epoch->mark_consumed();
                        if (!was_already) {
                            m_logger->debug("[Worker_manager] Epoch #{} marked consumed by worker {}",
                                            epoch->epoch_id, id);
                        }
                    }

                    m_logger->info("════════════════════════════════════════════════════════");
                    m_logger->info("💎 BLOCK FOUND CALLBACK INVOKED!");
                    m_logger->info("   Worker ID:  {}", id);
                    m_logger->info("   Height:     {}", block_data->nHeight);
                    m_logger->info("   Nonce:      0x{:016x}", block_data->nNonce);
                    m_logger->info("════════════════════════════════════════════════════════");

                    // Snapshot is retained before auth/template gates. A socket
                    // drop ends the session before this callback runs, so returning
                    // here used to discard the only copy of the solved header.
                    accept_solved_block(std::move(block_data));
                };

                // Stone 4: publish a single TemplateEpoch into the shared feed.
                // Workers that consume from the feed (uses_template_feed() == true)
                // will observe this epoch on their next mining-loop iteration
                // without any per-worker mutex acquisition by the manager.
                [[maybe_unused]] std::uint64_t published_epoch_id = 0;
                if (feed_snapshot) {
                    epoch_ptr->on_found = on_found;  // shared by all consumers of this epoch
                    published_epoch_id = feed_snapshot->publish(std::move(epoch_ptr));
                    m_logger->debug("[Worker_manager] Published TemplateEpoch #{} to feed",
                                    published_epoch_id);
                }
                // If feed_snapshot is null, epoch_ptr falls out of scope here
                // and the on_found lambda's weak_ptr expires.  That's fine —
                // legacy workers self-mark their own per-worker consumed flag
                // and the cross-epoch consumed signal is unused without a
                // feed-consumer worker to read it.

                // Fan out to workers that have not yet been migrated to consume
                // from the feed.  This loop runs OUTSIDE m_worker_mutex, so the
                // stats path is no longer blocked while N per-worker mutexes +
                // CV notifications take effect.  Once a worker subclass is
                // migrated and starts returning uses_template_feed() == true,
                // it is skipped here automatically.
                //
                // `i` here is the worker's original m_workers index, which is
                // assigned to m_config.m_internal_id at construction time, so
                // diagnostic warn() messages stay aligned with the internal id
                // operators see in other logs.
                for (size_t i = 0; i < worker_snapshot.size(); ++i) {
                    auto& worker = worker_snapshot[i];
                    if (!worker) {
                        m_logger->warn("[Worker_manager] Skipping null worker at index {}", i);
                        continue;
                    }
                    if (!worker->uses_template_feed()) {
                        worker->set_block(work_package, on_found);
                    }
                    if (worker->is_running()) {
                        workers_fed++;
                        m_logger->debug("[Worker_manager] Template sent to worker {}/{}",
                                       workers_fed, total_worker_count);
                    } else if (!worker->uses_template_feed()) {
                        // Only meaningful for legacy shim path: set_block() was
                        // expected to wake the worker.  Feed-consuming workers
                        // observe epochs lock-free at their natural rebind
                        // point, so is_running()==false here is unrelated to
                        // the template handoff and would be a misleading warn.
                        m_logger->warn(
                            "[Worker_manager] Worker {} did not start after set_block() — not counted",
                            i);
                    } else {
                        // Feed-consuming worker not currently running; not an
                        // error in the template-handoff sense, but worth
                        // flagging at debug level since it won't pick up the
                        // freshly-published epoch until it transitions back
                        // to running.
                        m_logger->debug(
                            "[Worker_manager] Feed-consuming worker {} not running — epoch #{} will be picked up on next start",
                            i, published_epoch_id);
                    }
                }
                
                if (workers_fed > 0) {
                    m_logger->info("[Worker_manager] ✓ Template distributed to {} workers - MINING STARTED", 
                                  workers_fed);
                    return true;
                } else {
                    m_logger->error("[Worker_manager] FAILED: No workers received template!");
                    // The failed-feed callback tears down this generation after
                    // the template interface has released its mutex.
                    return false;
                }
            }
        );
        
        m_logger->info("[Worker_manager] Template distribution handler registered");

        m_primary_node_session->set_template_feed_failed_handler([this]() {
            mark_recovery_initiated("template_distribution_failed");
            stop_all_workers();
        });

        m_primary_node_session->set_template_fed_handler([this]() {
            if (m_node_shutdown_quarantine.exchange(false, std::memory_order_acq_rel)) {
                m_logger->warn("[Worker_manager] NODE_SHUTDOWN quarantine cleared after fresh template distribution");
            }
            flush_pending_submit("template_fed");
            clear_recovery_state();

            auto protocol = m_primary_node_session ? m_primary_node_session->get_active_protocol() : nullptr;
            if (!protocol && m_primary_node_session) protocol = m_primary_node_session->get_primary_protocol();
            if (!protocol) return;
            const auto ht = protocol->get_height_tracker_snapshot();
            if (ht.last_push_notification_at == std::chrono::steady_clock::time_point{}) {
                m_logger->info("[Worker_manager] No push notification ever received — skipping resubscribe");
                return;
            }
            const auto now = std::chrono::steady_clock::now();
            const auto since_push = std::chrono::duration_cast<std::chrono::seconds>(
                now - ht.last_push_notification_at).count();
            const auto since_recovery = m_recovery.last_completed_at == std::chrono::steady_clock::time_point{}
                ? INT64_MAX : std::chrono::duration_cast<std::chrono::seconds>(
                    now - m_recovery.last_completed_at).count();
            if (since_recovery >= 60 && since_push > 400) {
                m_logger->warn("[Worker_manager] Push notifications silent for {}s after recovery — re-subscribing",
                               since_push);
                protocol->resubscribe_push_notifications();
            }
        });

        /* ========== REGISTER VALIDATION FAILURE HANDLER ========== */
        /* This handler is called when template validation fails */
        /* It stops workers and requests fresh work/GET_BLOCK */
        auto solo_protocol = m_primary_node_session->get_primary_protocol();
        if (solo_protocol) {
            auto* template_interface = solo_protocol->get_template_interface();
            if (template_interface) {
            template_interface->set_validation_failure_handler(
                [this](const protocol::MiningTemplateInterface::ValidationResult& result) {
                    m_logger->error("[Worker_manager] ════════════════════════════════════════");
                    m_logger->error("[Worker_manager] ⚠️  TEMPLATE VALIDATION FAILED");
                    m_logger->error("[Worker_manager]    Reason: {}", result.error_message);
                    m_logger->error("[Worker_manager] ════════════════════════════════════════");
                    
                    // Template hygiene failures are template-local, not session
                    // failures.  Keep the current workers alive (or idle) while
                    // requesting replacement work so a single dead-on-arrival
                    // template cannot strand the miner with no worker/stat timer
                    // activity.  Full worker stops remain reserved for session
                    // recovery, transport reconnect, degraded mode, and true
                    // template distribution failures.
                    //
                    // Route through the SAME soft-vs-hard policy the
                    // recovery_initiated_handler already uses
                    // (protocol::should_initiate_recovery_epoch), instead of
                    // unconditionally bumping the recovery epoch here.  A
                    // known, immediately-retriable template-local rejection
                    // (e.g. PRIME_ORIGIN_TOO_LOW — ~1/256 of Prime templates,
                    // see HasSufficientPrimeOrigins) does not need a
                    // RecoveryPhase transition, a degraded_since timestamp, or
                    // a degraded-mode stats flip: it self-heals with a single
                    // GET_BLOCK retry.  Reasons the shared policy actually
                    // classifies as recovery-worthy still escalate normally.
                    // (This mirrors RecoveryHandlerHarness::invoke() in
                    // get_block_dedup_recovery_test.cpp, which already
                    // asserted this exact routing.)
                    if (protocol::should_initiate_recovery_epoch(result.retry_reason)) {
                        mark_recovery_initiated(protocol::reason_name(result.retry_reason));
                    } else {
                        m_logger->debug("[Worker_manager] Validation failure with soft reason {} — "
                                        "issuing GET_BLOCK retry without epoch transition",
                                        protocol::reason_name(result.retry_reason));
                    }
                    retry_template_request(result.retry_reason);
                }
            );
            m_logger->info("[Worker_manager] Validation failure handler registered");
        } else {
            m_logger->warn("[Worker_manager] Template interface not available - validation failure handler not registered");
        }
    } else {
        m_logger->warn("[Worker_manager] Primary protocol not available - validation failure handler not registered");
    }

        /* ========== REGISTER RECOVERY INITIATED HANDLER ========== */
        /* Multi-block/channel-stale recovery remains a hard stop path. Same-height  */
        /* canonical replacement now takes the dedicated soft-refresh handler below, */
        /* which withholds submissions while the replacement template is fetched.    */
        m_primary_node_session->set_recovery_initiated_handler(
            [this](protocol::GetBlockReason reason) -> bool {
                if (protocol::should_initiate_recovery_epoch(reason)) {
                    mark_recovery_initiated(protocol::reason_name(reason));
                } else {
                    m_logger->debug("[Worker_manager] Recovery handler invoked with soft reason {} — "
                                    "issuing GET_BLOCK retry without epoch transition",
                                    protocol::reason_name(reason));
                }
                // Workers keep running while we request fresh work/GET_BLOCK.
                // Return value: true iff a GET_BLOCK was transmitted on the wire.
                // Solo uses this to decide whether to issue a local fallback request
                // (single-chokepoint contract — see Solo::dispatch_recovery_or_fallback).
                return retry_template_request_returning(reason);
            }
        );
        m_logger->info("[Worker_manager] Recovery handler registered");

        /* ========== REGISTER SESSION EXPIRED HANDLER ========== */
        /* Called by Solo when SESSION_EXPIRED opcode is received from the node.     */
        /* The TCP connection is still alive — we must re-authenticate in-band       */
        /* using login() on the existing connection, NOT reset() which tears down    */
        /* the TCP connection. Uses existing session auth backoff infrastructure.    */
        m_primary_node_session->set_session_expired_handler(
            [this]() {
                // recovery_in_progress tracks the authoritative session-restoration
                // window so template-only recovery never masks SESSION_EXPIRED.
                if (is_reconnecting() || is_session_recovery() ||
                    m_recovery.recovery_in_progress.load(std::memory_order_acquire)) {
                    m_logger->warn("[Worker_manager] Session EXPIRED ignored: authoritative session restoration already in progress "
                                   "(phase={}, recovery_in_progress={}, recovery_epoch={})",
                                   phase_name(m_recovery.phase.load(std::memory_order_relaxed)),
                                   m_recovery.recovery_in_progress.load(std::memory_order_relaxed),
                                   m_epoch_coordinator->recovery_epoch());
                    return;
                }

                m_logger->warn("[Worker_manager] Session EXPIRED — stopping workers and requiring a fresh session");
                ++m_session_generation;  // Invalidate any in-flight timer dispatches for the old session
                stop_all_workers();
                m_pending_recovery_completion_kind = RecoveryCompletionKind::PRIMARY_SESSION_REAUTH;
                transition_to(RecoveryPhase::SESSION_RECOVERY, "session_expired");

                // Use the current session auth fail count to calculate backoff delay.
                // NOTE: We do NOT increment m_session_auth_fail_count here.
                // The session_authenticated_handler will increment it if the subsequent
                // authentication fails (session_id == 0), avoiding double-counting.

                // First controlled in-band re-auth is immediate; subsequent ones use an
                // exponential backoff (see next_session_auth_retry_delay_seconds()) so
                // early retries are fast and later ones patiently wait out a storm.
                auto delay_seconds = static_cast<uint16_t>(
                    m_session_auth_fail_count > 0 ? next_session_auth_retry_delay_seconds() : 0);

                if (delay_seconds > 0) {
                    m_logger->warn("[Session] Scheduling in-band re-authentication in {}s ({} prior failures, controlled retry window)",
                        delay_seconds, m_session_auth_fail_count);
                } else {
                    m_logger->info("[Session] Scheduling immediate in-band re-authentication (no prior auth failures)");
                }

                // Schedule re-auth via io_context to avoid calling login()
                // from within a packet-receive callback (stack depth / reentrancy safety).
                if (m_io_context && m_primary_node_session) {
                    auto timer = std::make_shared<asio::steady_timer>(*m_io_context, std::chrono::seconds(delay_seconds));
                    const auto generation = m_session_generation.load(std::memory_order_acquire);
                    timer->async_wait([self = shared_from_this(), timer, generation](const asio::error_code& ec) {
                        if (ec) {
                            if (ec != asio::error::operation_aborted) {
                                self->m_logger->error("[Session] Re-auth timer error: {}", ec.message());
                            }
                            return;
                        }

                        if (self->m_terminal_stop_requested.load(std::memory_order_acquire) ||
                            self->m_session_generation.load(std::memory_order_acquire) != generation) {
                            return;
                        }

                        if (!self->m_primary_node_session) {
                            self->m_logger->warn("[Session] Node session destroyed before re-auth could execute");
                            return;
                        }

                        self->m_logger->info("[Session] Executing in-band re-authentication (calling login() on existing connection)");

                        // Use login_on_active_connection() to ensure the auth payload
                        // is framed for the same lane that will carry it (avoids the
                        // STATELESS-on-LEGACY mismatch when primary is down).
                        bool sent = self->m_primary_node_session->login_on_active_connection(
                            [self](bool login_result) {
                                if (!login_result) {
                                    self->m_logger->error("[Session] In-band re-authentication login() call failed");
                                    // The session_authenticated_handler will handle retry logic
                                } else {
                                    self->m_logger->info("[Session] In-band re-authentication login() call succeeded, awaiting MINER_AUTH_RESULT");
                                }
                            });

                        if (!sent) {
                            // In-band re-auth failed — no active connection to send on.
                            // Schedule a full TCP reconnect instead of silently stalling.
                            self->m_logger->warn("[Session] In-band re-auth failed (no active connection) — "
                                                  "scheduling TCP reconnect as fallback");
                            if (self->m_primary_endpoint.is_valid()) {
                                // No socket to log in on. A push timestamp from the
                                // dead session must not suppress the TCP reconnect.
                                self->retry_connect(self->m_primary_endpoint, true);
                            } else {
                                self->m_logger->error("[Session] No endpoint available for reconnect fallback");
                            }
                        }
                    });
                }
            }
        );
        m_logger->info("[Worker_manager] Session expired handler registered");

        // Transport-lost is registered from connect(), not here. weak_from_this()
        // is empty until the owning shared_ptr finishes constructing this object.

        /* ========== REGISTER RECOVERY-CONFIRMED HANDLER ========== */
        /* Called by Solo when the node re-serves BLOCK_DATA identical to the template   */
        /* already loaded (same unified height + hashPrevBlock) while the same-height    */
        /* feed guard suppresses redistributing it to workers. This is proof the current */
        /* template is still canonical, so an in-flight recovery can clear immediately   */
        /* rather than waiting for a genuinely new tip (or escalating toward degraded    */
        /* mode) — see handle_recovery_confirmed().                                      */
        m_primary_node_session->set_recovery_confirmed_handler(
            [this]() {
                handle_recovery_confirmed("same_tip_reconfirmed");
            }
        );
        m_logger->info("[Worker_manager] Recovery-confirmed handler registered");

        /* ========== REGISTER SESSION AUTHENTICATED HANDLER (BUG FIX #2) ========== */
        /* Called by Solo after MINER_AUTH_RESULT is fully processed and session_id is set. */
        /* This is the correct place to check session_id=0 (not in the login callback which */
        /* fires before MINER_AUTH_RESULT arrives). Triggers retry with exponential backoff. */
        m_primary_node_session->set_session_authenticated_handler(
            [this](protocol::SessionId session_id) {
                // CRITICAL: If session_id = 0, the node rejected authentication or didn't provide a session.
                // Mining cannot proceed without a valid session_id (work submissions will be silently rejected).
                // Continue reconnecting with capped backoff until the session returns.
                if (session_id.is_default())
                {
                    if (m_session_auth_fail_count < std::numeric_limits<uint32_t>::max())
                        ++m_session_auth_fail_count;
                    ++m_session_generation;  // Failed auth is still a session transition
                    m_pending_recovery_completion_kind = RecoveryCompletionKind::PRIMARY_SESSION_REAUTH;
                    if (!is_reconnecting() && !is_degraded_terminal()) {
                        transition_to(RecoveryPhase::SESSION_RECOVERY, "session_authentication_failed");
                    }
                    stop_all_workers();
                    m_logger->error("[Session] CRITICAL: Node returned session_id=0x00000000 after authentication (attempt #{})",
                        m_session_auth_fail_count);
                    m_logger->error("[Session] This indicates the node rejected the session or is misconfigured");
                    m_logger->error("[Session] Work submissions cannot proceed without a valid session ID");

                    auto delay_seconds = next_session_auth_retry_delay_seconds();

                    m_logger->warn("[Session] Scheduling controlled reconnection retry in {}s",
                        delay_seconds);

                    if (m_session_auth_retry_timer) {
                        m_session_auth_retry_timer->cancel();
                    }
                    auto timer = std::make_shared<asio::steady_timer>(
                        *m_io_context, std::chrono::seconds(delay_seconds));
                    m_session_auth_retry_timer = timer;
                    const auto generation = m_session_generation.load(std::memory_order_acquire);
                    timer->async_wait([weak_self = weak_from_this(), timer, generation](const asio::error_code& ec) {
                        auto self = weak_self.lock();
                        if (!self || ec == asio::error::operation_aborted) return;
                        if (ec) {
                            self->m_logger->error("[Session] Authentication retry timer failed: {}", ec.message());
                            return;
                        }
                        if (self->m_session_generation.load(std::memory_order_acquire) != generation ||
                            self->m_terminal_stop_requested.load(std::memory_order_acquire)) return;
                        if (!self->m_primary_endpoint.is_valid()) {
                            self->m_logger->error("[Session] Authentication retry has no valid node endpoint");
                            return;
                        }
                        self->retry_connect(self->m_primary_endpoint, true);
                    });
                    return;
                }

                // Successful authentication: reset session auth failure counter
                flush_pending_submit("session_authenticated");
                m_session_auth_fail_count = 0;
                ++m_session_generation;  // New session — invalidate stale timer dispatches
                if (m_session_auth_retry_timer) {
                    m_session_auth_retry_timer->cancel();
                }

                if (m_using_failover)
                {
                    m_logger->info("[Failover] Fresh session established on failover node: session_id=0x{:08x}",
                        session_id.get());
                }
                else
                {
                    m_logger->info("[Primary] Fresh session established on primary node: session_id=0x{:08x}",
                        session_id.get());
                }
            }
        );
        m_logger->info("[Worker_manager] Session authenticated handler registered");

        m_primary_node_session->set_work_ready_handler(
            [this]() {
                auto solo_protocol = m_primary_node_session
                    ? m_primary_node_session->get_active_protocol()
                    : nullptr;
                if (!solo_protocol || has_valid_template_available(solo_protocol)) {
                    return;
                }

                const bool recovering_session = is_session_recovery();
                if (recovering_session || is_reconnecting()) {
                    m_pending_recovery_completion_kind = recovering_session
                        ? RecoveryCompletionKind::PRIMARY_SESSION_REAUTH
                        : RecoveryCompletionKind::TRANSPORT_RECONNECT;
                    transition_to(RecoveryPhase::WAITING_TEMPLATE, "authoritative_work_ready");
                }

                if (is_recovery_active()) {
                    restart_recovery_window("authoritative_work_ready");
                }

                const auto reason = recovering_session
                    ? protocol::GetBlockReason::SESSION_REAUTH
                    : protocol::GetBlockReason::INITIAL_REQUEST;
                retry_template_request(reason);
            }
        );
        m_logger->info("[Worker_manager] Work-ready handler registered");

        /* ========== REGISTER SESSION START HANDLER ========== */
        /* Called by Solo when SESSION_START is received and keepalive interval has been  */
        /* auto-adjusted from the node-advertised timeout. Updates m_node_keepalive_interval_hours */
        /* so future secondary/failover connections use the correct node-derived interval. */
        // Handlers registered here must not capture weak_from_this()/shared_from_this()
        // at construction time: enable_shared_from_this is not yet wired when
        // Worker_manager is created via std::make_shared. Capture `this` and
        // obtain weak_from_this() only when the callback runs (post-construction).
        m_primary_node_session->set_session_start_handler(
            [this](uint16_t keepalive_hours) {
                auto self = weak_from_this().lock();
                if (!self) return;
                self->m_node_keepalive_interval_hours.store(keepalive_hours);
                self->m_logger->info("[Worker_manager] Node-advertised keepalive interval: {} hours (will seed future connections)",
                    keepalive_hours);
            }
        );
        m_logger->info("[Worker_manager] Session start handler registered");

        /* ========== REGISTER BLOCK ACCEPTED HANDLER ========== */
        /* Records accepted blocks into the three-tier mined-block cache. */
        m_primary_node_session->set_block_accepted_handler(
            [this](uint32_t height, uint1024_t hash_prev_block, uint32_t channel, uint64_t nonce) {
                auto self = weak_from_this().lock();
                if (!self) return;
                self->m_mined_block_cache.record_accepted_block(height, hash_prev_block, channel, nonce);
                self->m_logger->info("[Worker_manager] ⛏ Block recorded in mined-block cache — height={} ch={} total={}",
                    height, channel == 1 ? "Prime" : "Hash", self->m_mined_block_cache.total_blocks());
                self->log_mined_block_cache();
            }
        );
        m_logger->info("[Worker_manager] Block accepted handler registered");

        // Node shutdown handler: stop workers and log the reason when the node
        // sends NODE_SHUTDOWN (0xD0FF).  The Solo protocol already applies its
        // own reconnect backoff (NODE_SHUTDOWN_BACKOFF_S); we just need to park
        // the workers so they don't churn on stale work.
        m_primary_node_session->set_node_shutdown_handler(
            [this](uint8_t reason) {
                auto mgr = weak_from_this().lock();
                if (!mgr) return;
                mgr->handle_node_shutdown(reason);
            }
        );
        m_logger->info("[Worker_manager] Node shutdown handler registered");

        // Replacement-pending handler: arm a precise one-shot timer at the exact
        // deadline reported by mark_replacement_pending() (via the PUSH path),
        // so an expired "BLOCK_DATA never arrived" promise is caught immediately
        // instead of waiting out the remainder of the 30s general-purpose
        // template-health poll cycle. See check_template_health()'s
        // replacement-pending safety net for the actual expiry handling.
        m_primary_node_session->set_replacement_pending_handler(
            [this](std::chrono::steady_clock::time_point deadline) {
                auto mgr = weak_from_this().lock();
                if (!mgr) return;
                auto now = std::chrono::steady_clock::now();
                auto delay = std::max(
                    std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now),
                    std::chrono::milliseconds(0));
                mgr->m_timer_manager.start_replacement_pending_timer(delay, mgr);
            }
        );
        m_logger->info("[Worker_manager] Replacement-pending handler registered");

        m_logger->info("[Worker_manager] NodeSession configured and handlers registered");

    create_stats_printers();
    create_workers();
}

void Worker_manager::create_stats_printers()
{
    bool printer_console_created = false;
    bool printer_file_created = false;
    for(auto& stats_printer_config : m_config.get_stats_printer_config())
    {
        switch(stats_printer_config.m_mode)
        {
            case config::Stats_printer_mode::FILE:
            {
                if(!printer_file_created)
                {
                    printer_file_created = true;
                    auto& stats_printer_config_file = std::get<config::Stats_printer_config_file>(stats_printer_config.m_printer_mode);
                    m_stats_printers.push_back(std::make_shared<stats::Printer_file<stats::Printer_solo>>(stats_printer_config_file.file_name,
                        m_config.get_mining_mode(), m_config.get_worker_config(), *m_stats_collector));
                }
                break;
            }
            case config::Stats_printer_mode::CONSOLE:    // falltrough
            default:
            {
                if(!printer_console_created)
                {
                    printer_console_created = true;
                    m_stats_printers.push_back(std::make_shared<stats::Printer_console<stats::Printer_solo>>(m_config.get_mining_mode(),
                        m_config.get_worker_config(), *m_stats_collector));
                }
                break;
            }
        }
    }

    if(m_stats_printers.empty())
    {
        m_logger->warn("No stats printer configured.");
    }
}

void Worker_manager::create_workers_locked()
{
    // Guard: if workers are already alive, do not spawn duplicates.
    // While m_worker_mutex is held, !m_workers.empty() means the prior generation
    // still exists. stop_all_workers() resets the shared_ptrs under the same mutex,
    // and each worker destructor blocks until its mining threads have fully joined.
    if (!m_workers.empty()) {
        m_logger->warn("[Worker_manager] create_workers_locked() called with {} workers already alive — skipping duplicate spawn",
                       m_workers.size());
        return;
    }

#ifdef PRIME_ENABLED
    // Stone 1 hardening: pre-touch the process-shared CPU sieving prime table
    // on the manager thread before spawning workers.  Without this, on
    // high-core-count machines all prime workers race into the magic-static
    // once-init in parallel during construction.  This call is cheap on
    // subsequent invocations (per-epoch worker recreation just reuses the
    // already-built singleton).
    if (m_config.get_mining_mode() == config::Mining_mode::PRIME) {
        bool has_cpu_prime_worker = false;
        for (const auto& worker_config : m_config.get_worker_config()) {
            if (worker_config.m_mode == config::Worker_mode::CPU) {
                has_cpu_prime_worker = true;
                break;
            }
        }
        if (has_cpu_prime_worker) {
            cpu::Worker_prime::prewarm_shared_state();
        }
    }
#endif

    auto internal_id = 0U;
    // Stone 4: ensure a shared template feed exists for this worker batch.
    // Created lazily here (instead of in the constructor) so stop_all_workers
    // can release it together with the workers, guaranteeing the feed always
    // outlives any worker holding a shared_ptr to it.
    if (!m_template_feed) {
        m_template_feed = std::make_shared<WorkerTemplateFeed>();
    }
    for(auto& worker_config : m_config.get_worker_config())
    {
        worker_config.m_internal_id = internal_id;
        switch(worker_config.m_mode)
        {
            case config::Worker_mode::FPGA:
            {
                if (m_config.get_mining_mode() == config::Mining_mode::PRIME)
                {
                    m_logger->error("FPGA worker is not supported for PRIME mining!");
                }
                else
                {
                    m_workers.push_back(std::make_shared<fpga::Worker_hash>(m_io_context, worker_config));
                }
                break;
            }
            case config::Worker_mode::GPU:
            {
#ifdef GPU_ENABLED
                if (m_config.get_mining_mode() == config::Mining_mode::PRIME)
                {
#ifdef PRIME_ENABLED
                    m_workers.push_back(std::make_shared<gpu::Worker_prime>(m_io_context, worker_config));
#else
                    m_logger->error("NexusMiner not built 'WITH_PRIME' -> no worker created!");
#endif
                }
#if defined(GPU_CUDA_ENABLED) && !defined(PRIME_ENABLED)
                else
                {
                    m_workers.push_back(std::make_shared<gpu::Worker_hash>(m_io_context, worker_config));
                }                
#elif defined(GPU_AMD_ENABLED) && !defined(PRIME_ENABLED)
                m_logger->error("NexusMiner 'WITH_GPU_AMD' but not 'WITH_PRIME'.  Hash mode on AMD is not supported. -> no worker created!");
#endif
#else
                m_logger->error("NexusMiner not built 'WITH_GPU_CUDA' or 'WITH_GPU_AMD' -> no worker created!");
#endif
                break;
            }
            case config::Worker_mode::CPU:    // falltrough
            default:
            {
                if (m_config.get_mining_mode() == config::Mining_mode::PRIME)
                {
#ifdef PRIME_ENABLED
                    m_workers.push_back(std::make_shared<cpu::Worker_prime>(m_io_context, worker_config));
#else
                    m_logger->error("NexusMiner not built 'WITH_PRIME' -> no worker created!");
#endif
                }
                else
                {
                    m_workers.push_back(std::make_shared<cpu::Worker_hash>(m_io_context, worker_config));
                }
                break;
            }
        }
        internal_id++;
    }

#ifdef PRIME_ENABLED
    // ── Stone 7: PrimeMiningEngine wiring ─────────────────────────────────
    //
    // When [cpu] engine_mode = "engine" and any CPU prime worker exists,
    // construct one PrimeMiningEngine for the channel.  The engine owns the
    // sieve pool threads + consumer thread; Worker_prime instances under
    // engine mode are thin adapters registered with the engine for stats
    // fan-in.
    //
    // Construction order (matches Stone 7 brief):
    //   1. feed exists (guaranteed above)
    //   2. workers exist (just constructed)
    //   3. construct engine from lowest internal_id Worker_prime's config
    //   4. register every Worker_prime with the engine
    //   5. bind_to_engine on every Worker_prime so update_statistics can
    //      pull from the engine snapshot
    //   6. attach_template_feed loop runs below (existing — engine-mode
    //      Worker_prime overrides attach_template_feed as a no-op).
    //
    // Lowest internal_id is the channel's representative worker (gets credit
    // on found blocks via EngineSession::internal_id_for_solution).
    if (m_config.get_mining_mode() == config::Mining_mode::PRIME)
    {
        // Collect (worker, internal_id) pairs for every CPU prime worker.
        // m_internal_id was assigned to each worker_config above as
        // creation-order, so the parallel walk over m_workers and
        // get_worker_config() yields a stable (worker, id) zip.
        struct PrimeWorkerEntry {
            std::shared_ptr<cpu::Worker_prime> worker;
            std::uint32_t internal_id;
        };
        std::vector<PrimeWorkerEntry> prime_workers;
        const auto& worker_configs = m_config.get_worker_config();
        const std::size_t pair_count = std::min(m_workers.size(), worker_configs.size());
        for (std::size_t i = 0; i < pair_count; ++i) {
            if (auto wp = std::dynamic_pointer_cast<cpu::Worker_prime>(m_workers[i])) {
                prime_workers.push_back({wp, worker_configs[i].m_internal_id});
            }
        }
        if (!prime_workers.empty())
        {
            // Resolve engine_mode from the FIRST CPU prime worker's config
            // (engine_mode is a [cpu] section key, not per-worker — all
            // CPU prime workers in the same TOML batch share the value).
            std::string engine_mode = "workers";
            for (const auto& wc : worker_configs) {
                if (wc.m_mode == config::Worker_mode::CPU
                    && std::holds_alternative<config::Worker_config_cpu>(wc.m_worker_mode))
                {
                    engine_mode = std::get<config::Worker_config_cpu>(wc.m_worker_mode).m_engine_mode;
                    break;
                }
            }

            if (engine_mode == "engine")
            {
                std::uint32_t lowest_internal_id = std::numeric_limits<std::uint32_t>::max();
                for (const auto& e : prime_workers) {
                    lowest_internal_id = std::min<std::uint32_t>(lowest_internal_id, e.internal_id);
                }

                // Stone 7 cleanup: cpu::Sieve::get_segment_size() is now
                // a static constexpr accessor for the compile-time
                // m_segment_size constant, so we no longer need to
                // construct a throwaway Sieve (whose ctor allocates
                // non-trivial per-thread buffers) just to read it.
                const std::uint64_t segment_size = cpu::Sieve::get_segment_size();

                cpu::Engine_config cfg;
                cfg.segment_size            = segment_size;
                // Channel-level starting nonce baseline: lowest registered
                // worker's `internal_id << 48`.  Pool threads add this when
                // computing the absolute sieve start (matches the per-worker
                // convention lifted to the channel level).
                cfg.channel_starting_nonce  = static_cast<std::uint64_t>(lowest_internal_id) << 48;
                cfg.internal_id_for_solution = lowest_internal_id;
                // Honor the operator's intent: under engine mode, the
                // [workers] count = N value (== prime_workers.size()) is
                // the pool_threads count.  Without this, count = N would
                // become meaningless for compute (only stats-display
                // partitioning would care about it) and the engine would
                // silently spawn min(hardware_concurrency(), 32) sieve
                // threads instead — a real behavioural surprise on a
                // 16-core box where the operator wrote count = 4 because
                // they wanted 4 cores busy.  prime_workers is non-empty
                // here (guarded above), so this is always >= 1.
                cfg.pool_threads            =
                    static_cast<std::uint32_t>(prime_workers.size());
                cfg.io_context              = m_io_context;

                try {
                    m_prime_engine = std::make_shared<cpu::PrimeMiningEngine>(
                        cfg, m_template_feed);

                    for (auto& e : prime_workers) {
                        m_prime_engine->register_worker(e.worker);
                    }
                    const std::uint32_t share_count =
                        static_cast<std::uint32_t>(prime_workers.size());
                    for (std::uint32_t i = 0; i < share_count; ++i) {
                        prime_workers[i].worker->bind_to_engine(m_prime_engine, i, share_count);
                    }
                    m_logger->info("[Worker_manager] PrimeMiningEngine wired ({} workers, "
                                   "internal_id_for_solution={}, segment_size={}, "
                                   "pool_threads={} (== [workers] count))",
                                   share_count, lowest_internal_id, segment_size,
                                   share_count);
                } catch (const std::exception& e) {
                    m_logger->error("[Worker_manager] Failed to construct PrimeMiningEngine: {} — "
                                    "engine-mode Worker_prime instances have no run-thread; "
                                    "workers will be inert until engine_mode is set back to \"workers\".",
                                    e.what());
                    m_prime_engine.reset();
                }
            }
        }
    }
#endif

    // Stone 4: attach the shared template feed to every worker so subclasses
    // that have been migrated to consume work via WorkerTemplateFeed (see
    // Worker::uses_template_feed()) bind to it.  Subclasses that have not
    // been migrated keep the default no-op attach_template_feed() and continue
    // to receive work via the legacy set_block(WorkPackage,...) shim, which
    // Worker_manager now invokes outside m_worker_mutex.
    //
    // NOTE: this runs AFTER every worker constructor has returned, and several
    // worker subclasses start their mining thread inside the constructor.  An
    // override that wants to read the feed from that already-running thread
    // MUST publish the pointer through a thread-safe slot (atomic shared_ptr)
    // and treat "not yet attached" as idle — see the contract on
    // Worker::attach_template_feed in src/worker/worker.hpp for details.
    for (auto& worker : m_workers) {
        if (worker) {
            worker->attach_template_feed(m_template_feed);
        }
    }
}

void Worker_manager::create_workers()
{
    std::lock_guard lock(m_worker_mutex);
    create_workers_locked();
}

void Worker_manager::stop()
{
    m_terminal_stop_requested.store(true, std::memory_order_release);
    m_connection_retry_scheduled = false;
    m_timer_manager.stop();
    if (m_session_auth_retry_timer) {
        m_session_auth_retry_timer->cancel();
    }

    // Destroy workers before tearing down transport so worker-side submit/request
    // callbacks cannot race a closing session or cleared connection state.
    stop_all_workers();
    clear_pending_submit();

    if (m_colin_agent)
    {
        m_colin_agent->stop();
        m_colin_agent.reset();
    }

    // Destroy workers before tearing down protocol/session state so no worker can
    // race a late submit/request against a closing NodeSession.  stop_all_workers()
    // above already moved out and joined every worker, so m_workers is normally
    // already empty here; snapshot-then-release (rather than resetting under the
    // lock) keeps this defensive pass consistent with stop_all_workers()'s
    // no-join-while-locked invariant in case that ever changes.
    std::vector<std::shared_ptr<Worker>> stragglers;
    {
        std::lock_guard lock(m_worker_mutex);
        stragglers = std::move(m_workers);
        m_workers.clear();
    }
    nexusminer::util::assert_no_tracked_locks_held("Worker_manager::stop (straggler worker teardown)");
    for (auto& worker : stragglers)
    {
        worker.reset();
    }
    stragglers.clear();

    if (m_primary_node_session) {
        m_primary_node_session->stop();
    }
}

void Worker_manager::collect_worker_statistics()
{
    std::lock_guard lock(m_worker_mutex);
    for (auto& worker : m_workers) {
        if (worker) {
            worker->update_statistics(*m_stats_collector);
        }
    }
}

void Worker_manager::enter_terminal_degraded_mode(int signal_number)
{
    enter_terminal_degraded_mode_internal(signal_number, "signal_terminal_exit");
}

void Worker_manager::enter_terminal_degraded_mode_internal(int signal_number, const char* reason)
{
    if (m_terminal_stop_requested.exchange(true, std::memory_order_acq_rel)) {
        return;
    }

    m_recovery.degraded_signal.store(signal_number, std::memory_order_relaxed);
    transition_to(RecoveryPhase::DEGRADED_TERMINAL, reason);
    if (signal_number != 0) {
        m_logger->critical("[Worker_manager] TERMINAL DEGRADED MODE entered by signal {} — full stop", signal_number);
    } else {
        m_logger->critical("[Worker_manager] TERMINAL DEGRADED MODE entered ({}) — full stop, operator action required",
                           reason ? reason : "unknown");
    }
    stop();
}

void Worker_manager::handle_node_shutdown(uint8_t reason)
{
    m_logger->warn("[Worker_manager] NODE_SHUTDOWN received (reason=0x{:02X})", reason);

    const auto invalidation = nexusminer::node_shutdown_work_invalidation();
    const auto shutdown_generation =
        m_session_generation.fetch_add(1, std::memory_order_acq_rel) + 1;
    if (invalidation.quarantine_current_generation) {
        m_node_shutdown_generation.store(shutdown_generation, std::memory_order_release);
        m_node_shutdown_quarantine.store(true, std::memory_order_release);
    }

    auto active_protocol = m_primary_node_session
        ? m_primary_node_session->get_active_protocol()
        : nullptr;
    if (!active_protocol && m_primary_node_session) {
        active_protocol = m_primary_node_session->get_primary_protocol();
    }
    if (invalidation.discard_template && active_protocol) {
        if (auto* template_interface = active_protocol->get_template_interface()) {
            template_interface->discard_template("node_shutdown");
        }
    }

    if (invalidation.stop_workers) {
        stop_all_workers();
    }

    if (invalidation.reset_session && m_primary_node_session) {
        m_primary_node_session->reset();
    }

    const auto action = nexusminer::decide_node_shutdown_action(m_config.has_failover());
    if (action == NodeShutdownAction::RETRY_PRIMARY_NODE)
    {
        if (!m_primary_endpoint.is_valid()) {
            m_logger->critical("[Worker_manager] NODE_SHUTDOWN received but primary endpoint is invalid");
            enter_terminal_degraded_mode_internal(0, "node_shutdown_invalid_primary");
            return;
        }
        transition_to(RecoveryPhase::RECONNECTING, "node_shutdown_retry_primary");
        m_logger->warn("[Worker_manager] NODE_SHUTDOWN: no failover configured; retrying primary in {}s",
                       protocol::Solo::NODE_SHUTDOWN_BACKOFF_S);
        schedule_connection_retry(
            static_cast<std::uint16_t>(protocol::Solo::NODE_SHUTDOWN_BACKOFF_S), m_primary_endpoint);
        return;
    }

    network::Endpoint standby_endpoint;
    bool target_is_failover = !m_using_failover;
    if (target_is_failover)
    {
        standby_endpoint = m_failover_endpoint;
        if (!standby_endpoint.is_valid())
        {
            m_logger->critical("[Worker_manager] NODE_SHUTDOWN requested failover, but failover endpoint is invalid — full stop");
            enter_terminal_degraded_mode_internal(0, "node_shutdown_invalid_failover");
            return;
        }

        m_logger->warn("[Worker_manager] NODE_SHUTDOWN on primary — switching immediately to failover {}",
                       standby_endpoint.to_string());
    }
    else
    {
        standby_endpoint = m_primary_endpoint;
        if (!standby_endpoint.is_valid())
        {
            m_logger->critical("[Worker_manager] NODE_SHUTDOWN on failover, but primary endpoint is invalid — full stop");
            enter_terminal_degraded_mode_internal(0, "node_shutdown_invalid_primary_standby");
            return;
        }

        m_logger->warn("[Worker_manager] NODE_SHUTDOWN on failover — switching immediately back to primary {}",
                       standby_endpoint.to_string());
    }

    transition_to(RecoveryPhase::RECONNECTING, "node_shutdown_standby_switch");

    m_using_failover = target_is_failover;
    m_primary_fail_count = 0;
    m_connection_retry_count = 0;
    m_connection_backoff.reset();
    m_node_keepalive_interval_hours.store(0);
    if (m_using_failover) {
        m_failover_activated_at = std::chrono::steady_clock::now();
    } else {
        m_failover_activated_at = {};
    }
    m_sim_link.set_failover_active(m_using_failover, standby_endpoint.to_string());

    m_logger->warn("[Worker_manager] NODE_SHUTDOWN standby reconnect scheduled immediately");
    schedule_connection_retry(0, standby_endpoint);
}

uint16_t Worker_manager::get_effective_keepalive_interval() const
{
    uint16_t node_interval = m_node_keepalive_interval_hours.load();
    return (node_interval > 0)
        ? node_interval
        : m_config.get_keepalive_interval();
}

void Worker_manager::retry_connect(network::Endpoint const& wallet_endpoint, bool force_transport_reset)
{
    if (m_terminal_stop_requested.load(std::memory_order_acquire)) {
        return;
    }

    // Safety guard: if push notifications have been received recently, the TCP
    // connection is demonstrably alive from the node's perspective. Tearing it
    // down here would destroy a valid node session. Use in-band re-auth instead.
    if (!force_transport_reset && m_primary_node_session) {
        // Use get_active_protocol() so push liveness and auth state are read from
        // the SAME protocol instance that login_on_active_connection() will use.
        // Before this fix, get_primary_protocol() could read stale push/auth data
        // from the primary protocol while get_active_protocol() returned the secondary
        // (or nullptr), causing protocol mismatch and stuck states.
        auto active_protocol = m_primary_node_session->get_active_protocol();
        if (active_protocol) {
            auto ht_snap = active_protocol->get_height_tracker_snapshot();
            bool push_received = (ht_snap.last_push_notification_at != std::chrono::steady_clock::time_point{});
            if (push_received) {
                auto now = std::chrono::steady_clock::now();
                auto since_push_s = std::chrono::duration_cast<std::chrono::seconds>(
                    now - ht_snap.last_push_notification_at).count();
                if (since_push_s < RETRY_CONNECT_PUSH_LIVE_SECONDS) {
                    // Conditional auth guard: suppress duplicate login() only if auth is recent.
                    // If auth has been in-flight longer than 10s it is likely stuck from a dead
                    // TCP mid-handshake — reset and retry rather than silently blocking.
                    double in_flight_s = active_protocol->auth_in_flight_seconds();
                    constexpr double AUTH_STALE_THRESHOLD_S = 10.0;

                    if (in_flight_s > 0.0 && in_flight_s <= AUTH_STALE_THRESHOLD_S) {
                        m_logger->info("[Worker_manager] Auth in-flight and recent ({:.1f}s < {}s) — "
                                       "skipping duplicate login()", in_flight_s, AUTH_STALE_THRESHOLD_S);
                        return;
                    }
                    if (in_flight_s > AUTH_STALE_THRESHOLD_S) {
                        m_logger->warn("[Worker_manager] Auth in-flight but stale ({:.1f}s > {}s) — "
                                       "resetting auth state before retry", in_flight_s, AUTH_STALE_THRESHOLD_S);
                    }
                    // Reset auth state so login() starts from a clean NOT_AUTHENTICATED state.
                    // This clears any stale WAITING_FOR_CHALLENGE / WAITING_FOR_RESULT left
                    // over from a previous attempt on a now-dead TCP connection.
                    active_protocol->reset_auth_state();
                    m_logger->warn("[Worker_manager] retry_connect() suppressed — push received {}s ago "
                                  "(TCP alive). Triggering in-band re-auth instead.", since_push_s);
                    // Use login_on_active_connection() to ensure the auth payload
                    // is framed for the same lane that will carry it.
                    bool sent = m_primary_node_session->login_on_active_connection(
                        [weak_self = weak_from_this()](bool login_result) {
                            auto self = weak_self.lock();
                            if (!self) return;
                            if (!login_result) {
                                self->m_logger->error("[Worker_manager] In-band re-auth (retry_connect guard) login() failed");
                            } else {
                                self->m_logger->info("[Worker_manager] In-band re-auth (retry_connect guard) login() sent, "
                                                     "awaiting MINER_AUTH_RESULT");
                            }
                        });
                    if (sent) {
                        return;  // Re-auth dispatched — wait for MINER_AUTH_RESULT
                    }
                    // In-band re-auth failed (connection died between push guard check
                    // and login attempt). Fall through to full TCP reconnect rather than
                    // returning with no recovery action — that would leave the miner stuck.
                    m_logger->warn("[Worker_manager] retry_connect guard: in-band re-auth failed "
                                   "(connection lost) — falling through to TCP reconnect");
                } else {
                    m_logger->warn("[Worker_manager] retry_connect() push guard expired "
                                   "(push {}s ago > {}s threshold) — proceeding with TCP reconnect",
                                   since_push_s, RETRY_CONNECT_PUSH_LIVE_SECONDS);
                }
            }
        }
    }

    // Transition to RECONNECTING phase — guards against stale callbacks and
    // prevents session_expired from re-entering during TCP reconnect window.
    ++m_session_generation;  // TCP disconnect — invalidate stale timer dispatches
    transition_to(RecoveryPhase::RECONNECTING, "tcp_reconnect");
    m_recovery.reconnect_started_at = std::chrono::steady_clock::now();

    // Reset NodeSession for reconnection
    if (m_primary_node_session) {
        m_primary_node_session->reset();
    }

    stats::Global_delta global_stats{};
    global_stats.m_connection_retries = 1;
    m_stats_collector->update_global_stats(global_stats);

    ++m_connection_retry_count;

    // Exponential backoff: start at the configured interval, double each failure, cap at 60s
    auto const base_delay = static_cast<uint32_t>(m_config.get_connection_retry_interval());
    if (m_connection_backoff.base == 0) {
        m_connection_backoff.base = base_delay;  // Set base delay from config on first use
    }
    auto current_delay = m_connection_backoff.next_delay();

    // ── Failover switchover logic ─────────────────────────────────────────────
    network::Endpoint effective_endpoint = wallet_endpoint;
    bool failover_switched = false;  // Track if we're switching nodes

    if (m_config.has_failover())
    {
        if (!m_using_failover)
        {
            ++m_primary_fail_count;
            if (m_primary_fail_count >= m_config.get_failover_max_retries())
            {
                m_using_failover = true;
                m_primary_fail_count = 0;
                m_connection_backoff.reset();  // reset backoff for failover attempt
                m_failover_activated_at = std::chrono::steady_clock::now();
                m_logger->warn("[Failover] Primary {} failed {} times — switching to failover {}",
                    m_primary_endpoint.to_string(),
                    m_config.get_failover_max_retries(),
                    m_failover_endpoint.to_string());
                effective_endpoint = m_failover_endpoint;
                failover_switched = true;
            }
        }
        else
        {
            ++m_primary_fail_count;
            if (m_primary_fail_count >= m_config.get_failover_max_retries())
            {
                m_using_failover = false;
                m_primary_fail_count = 0;
                m_connection_backoff.reset();
                m_logger->info("[Failover] Retrying primary {} after failover failures",
                    m_primary_endpoint.to_string());
                effective_endpoint = m_primary_endpoint;
                failover_switched = true;
            }
            else
            {
                effective_endpoint = m_failover_endpoint;
            }
        }

        // Clear cached keepalive interval for failover switch
        if (failover_switched)
        {
            m_logger->info("[Failover] Resetting NodeSession for fresh authentication on {}",
                effective_endpoint.to_string());
            m_node_keepalive_interval_hours.store(0);
            m_logger->info("[Failover] Cleared node-advertised keepalive interval (will re-learn from failover SESSION_START)");
        }
    }

    if (effective_endpoint.is_valid()) {
        m_sim_link.set_failover_active(m_using_failover, effective_endpoint.to_string());
    }

    if (m_connection_retry_count > protocol::ProtocolConstants::CONNECTION_RETRY_ERROR_THRESHOLD)
        m_logger->error("Connection retry #{} - {} consecutive failures",
                        m_connection_retry_count, m_connection_retry_count);
    else
        m_logger->info("Connection retry {} seconds (attempt #{})",
                       current_delay, m_connection_retry_count);

    schedule_connection_retry(static_cast<std::uint16_t>(current_delay), effective_endpoint);
}

Worker_manager::FailoverStatus Worker_manager::get_failover_status() const
{
    FailoverStatus fs;
    fs.has_failover_configured = m_config.has_failover();
    fs.using_failover          = m_using_failover;
    fs.primary_fail_count      = m_primary_fail_count;
    fs.failover_max_retries    = m_config.get_failover_max_retries();
    fs.primary_endpoint_str    = m_primary_endpoint.is_valid()  ? m_primary_endpoint.to_string()  : "";
    fs.failover_endpoint_str   = m_failover_endpoint.is_valid() ? m_failover_endpoint.to_string() : "";
    fs.failover_activated_at   = m_failover_activated_at;
    return fs;
}



int64_t Worker_manager::next_forced_retry_jitter_ms()
{
    thread_local std::mt19937 rng([]() {
        std::random_device rd;
        auto now = static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
        std::seed_seq seed{
            rd(), rd(),
            static_cast<uint32_t>(now & 0xffffffffu),
            static_cast<uint32_t>((now >> 32) & 0xffffffffu)
        };
        return std::mt19937(seed);
    }());
    std::uniform_int_distribution<int64_t> dist(FORCED_RETRY_JITTER_MIN_MS, FORCED_RETRY_JITTER_MAX_MS);
    return dist(rng);
}

bool Worker_manager::has_valid_template_available(const std::shared_ptr<protocol::Solo>& solo_protocol) const
{
    auto* template_interface = solo_protocol ? solo_protocol->get_template_interface() : nullptr;
    return (template_interface && template_interface->has_valid_template());
}


void Worker_manager::schedule_forced_recovery_retry(const char* trigger_reason)
{
    if (!m_io_context || !is_recovery_active()) {
        return;
    }
    if (m_forced_retry_timer_pending) {
        return;
    }

    auto now = std::chrono::steady_clock::now();
    auto delay_ms = next_forced_retry_jitter_ms();
    auto due = now + std::chrono::milliseconds(delay_ms);

    auto wait_ms = std::max<int64_t>(
        1,
        std::chrono::duration_cast<std::chrono::milliseconds>(due - now).count());
    m_forced_retry_timer_pending = true;
    ++m_forced_retry_timer_token;
    const auto token = m_forced_retry_timer_token;
    m_forced_retry_timer = std::make_shared<asio::steady_timer>(*m_io_context, std::chrono::milliseconds(wait_ms));
    m_logger->info("[Worker_manager] Queue forced GET_BLOCK retry token={} in {}ms (reason={})",
                   token, wait_ms, trigger_reason ? trigger_reason : "unknown");
    m_forced_retry_timer->async_wait([self = shared_from_this(), token](const asio::error_code& ec) {
        if (ec) {
            // Always clear the pending flag so future retries are not permanently blocked.
            self->m_forced_retry_timer_pending = false;
            if (ec != asio::error::operation_aborted) {
                self->m_logger->warn("[Worker_manager] forced_retry_timer error: {}", ec.message());
            } else {
                self->m_logger->debug("[Worker_manager] forced_retry_timer cancelled (operation_aborted)");
            }
            return;
        }
        if (!self->is_recovery_active() || token != self->m_forced_retry_timer_token) {
            self->m_forced_retry_timer_pending = false;
            return;
        }
        self->m_forced_retry_timer_pending = false;
        self->retry_template_request(protocol::GetBlockReason::RECOVERY_TIMER);
    });
}

bool Worker_manager::connect(network::Endpoint const& wallet_endpoint)
{
    if (m_terminal_stop_requested.load(std::memory_order_acquire)) {
        m_logger->warn("[Worker_manager] connect() ignored - terminal stop requested");
        return false;
    }

    // Save the primary endpoint on the very first connect() call from Miner::run()
    if (!m_primary_endpoint.is_valid())
    {
        m_primary_endpoint = wallet_endpoint;
        if (m_config.has_failover())
        {
            auto const fo_ip   = m_config.get_failover_wallet_ip();
            auto fo_port = m_config.get_failover_port() != 0
                               ? m_config.get_failover_port()
                               : wallet_endpoint.port();
            if (fo_port != wallet_endpoint.port())
            {
                m_logger->warn("[Failover] Ignoring configured failover port {} and pinning failover lane "
                               "to primary port {} so reconnect/failover stay on the configured lane",
                               fo_port,
                               wallet_endpoint.port());
                fo_port = wallet_endpoint.port();
            }
            m_failover_endpoint = network::Endpoint{network::Transport_protocol::tcp, fo_ip, fo_port};
            m_logger->info("[Failover] Configured: {}:{} (switch after {} primary failures)",
                           fo_ip, fo_port, m_config.get_failover_max_retries());
        }
    }

    std::string wallet_addr;
    wallet_endpoint.address(wallet_addr);
    uint16_t configured_port = wallet_endpoint.port();

    m_logger->info("[Solo] Connecting to wallet {}:{}", wallet_addr, configured_port);
    m_logger->info("[Solo] Port Configuration: Using port {} from miner.conf", configured_port);
    m_logger->debug("[Solo] Connection initiated to endpoint: {}", wallet_endpoint.to_string());

    // Register only after shared ownership exists. Doing this in the constructor
    // captures an empty weak_from_this(), so every later notification no-ops.
    ensure_transport_lost_handler();

    // The retry timer has fired (or this is the initial connect). The in-flight
    // attempt is now NodeSession's pending connect callback, not the timer.
    m_connection_retry_scheduled = false;

    // Use NodeSession to connect on the configured lane; no cross-lane fallback.
    std::weak_ptr<Worker_manager> weak_self = shared_from_this();
    const auto generation = m_session_generation.load(std::memory_order_acquire);
    return m_primary_node_session->connect(wallet_endpoint, [weak_self, wallet_endpoint, generation](bool success) {
        auto self = weak_self.lock();
        if (!self) return;
        // reset() completes the old connect callback synchronously. A forced
        // retry already owns recovery, so it must not schedule a second retry.
        if (self->m_terminal_stop_requested.load(std::memory_order_acquire) ||
            self->m_session_generation.load(std::memory_order_acquire) != generation) {
            self->m_logger->debug("[Worker_manager] Ignoring superseded connect callback");
            return;
        }

        if (!success) {
            self->m_logger->error("[Solo] Connection to wallet {} not successful", wallet_endpoint.to_string());
            self->retry_connect(wallet_endpoint);
            return;
        }

        // Connection and authentication succeeded
        self->m_logger->info("[Solo] Successfully connected and authenticated to {}", wallet_endpoint.to_string());

        // Reset retry counters on successful connection
        self->m_connection_retry_count = 0;
        self->m_connection_backoff.reset();
        self->m_primary_fail_count = 0;
        self->m_sim_link.set_failover_active(self->m_using_failover, wallet_endpoint.to_string());

        if (self->is_reconnecting()) {
            self->m_pending_recovery_completion_kind =
                self->m_using_failover
                    ? RecoveryCompletionKind::FAILOVER_SESSION
                    : RecoveryCompletionKind::TRANSPORT_RECONNECT;
            self->m_logger->info("[Worker_manager] Transport reconnected — waiting for authoritative work readiness");
        }

        // Start timers once only (guarded by flags)
        auto const print_statistics_interval = self->m_config.get_print_statistics_interval();
        if (!self->m_stats_timers_started)
        {
            self->m_stats_timers_started = true;
            self->m_timer_manager.start_stats_collector_timer(print_statistics_interval, self, self->m_stats_collector);
            self->m_timer_manager.start_stats_printer_timer(print_statistics_interval, self->m_stats_printers);
        }

        // Start template health monitor
        constexpr uint16_t TEMPLATE_HEALTH_INTERVAL = 30;
        if (!self->m_template_health_timer_started)
        {
            self->m_template_health_timer_started = true;
            self->m_timer_manager.start_template_health_timer(TEMPLATE_HEALTH_INTERVAL, self);
            self->m_logger->info("[Worker_manager] Template health monitor started (30s interval)");
        }

        // Start GET_ROUND timer (uses Worker_manager to fetch current connection on each tick)
        static_assert(protocol::Solo::POLL_INTERVAL_MIN_MS % 1000 == 0,
            "GET_ROUND timer tick expects whole-second poll intervals");
        constexpr uint16_t GET_ROUND_TIMER_INTERVAL =
            static_cast<uint16_t>(protocol::Solo::POLL_INTERVAL_MIN_MS / 1000);
        if (!self->m_get_round_timer_started)
        {
            self->m_get_round_timer_started = true;
            self->m_timer_manager.start_get_round_timer(
                GET_ROUND_TIMER_INTERVAL,
                self);
            self->m_logger->info("[Solo Poll] GET_ROUND polling timer started ({}s tick, {}--{}s adaptive interval, both lanes)",
                GET_ROUND_TIMER_INTERVAL,
                protocol::Solo::POLL_INTERVAL_MIN_MS / 1000,
                protocol::Solo::POLL_INTERVAL_MAX_MS / 1000);
        }

        // Start lane health check timer
        constexpr uint16_t LANE_HEALTH_INTERVAL = 120;
        if (!self->m_lane_health_timer_started)
        {
            self->m_lane_health_timer_started = true;
            self->m_timer_manager.start_lane_health_check_timer(LANE_HEALTH_INTERVAL, self);
        }

        // Start Colin agent on first successful connect
        if (!self->m_colin_agent && self->m_config.get_colin_enabled())
        {
            self->m_colin_agent = std::make_shared<ColinAgent>(
                self->m_io_context,
                &self->m_sim_link,
                self->m_stats_collector,
                self->m_logger,
                self->m_config.get_colin_report_interval_seconds());

            // Wire up diagnostic sources
            auto solo_protocol_ptr = self->m_primary_node_session->get_primary_protocol();
            if (solo_protocol_ptr)
            {
                std::weak_ptr<protocol::Solo> weak_proto = solo_protocol_ptr;

                // Wire up PING_DIAG source
                self->m_colin_agent->set_ping_source(
                    [weak_proto]() -> ::LLP::ReceivedPingFrame {
                        auto proto = weak_proto.lock();
                        return proto ? proto->last_received_ping() : ::LLP::ReceivedPingFrame{};
                    });

                // Wire up SESSION_STATUS_ACK source
                self->m_colin_agent->set_status_source(
                    [weak_proto]() -> std::pair<::LLP::SessionStatusAckFrame,
                                                 std::chrono::steady_clock::time_point> {
                        auto proto = weak_proto.lock();
                        if (!proto) return {};
                        return { proto->last_session_status_ack(),
                                 proto->last_session_status_ack_time() };
                    });

                // Wire up HeightTracker
                self->m_colin_agent->set_height_tracker(&solo_protocol_ptr->get_height_tracker());

                // Wire up HashCheckpointGuard for reorg diagnostics
                self->m_colin_agent->set_checkpoint_guard_source(
                    [weak_proto]() -> ColinAgent::CheckpointGuardSnapshot {
                        ColinAgent::CheckpointGuardSnapshot snap;
                        auto proto = weak_proto.lock();
                        if (!proto) return snap;
                        const auto& guard = proto->get_hash_checkpoint_guard();
                        snap.reorg_depth_estimate = guard.reorg_depth_estimate();
                        snap.consecutive_mismatch = guard.consecutive_mismatch_count();
                        snap.checkpoint_count     = guard.checkpoint_count();
                        return snap;
                    });

                // Wire up MiningTemplateInterface
                if (auto* tmpl_iface = solo_protocol_ptr->get_template_interface())
                {
                    self->m_colin_agent->set_template_source(
                        [weak_proto]() -> ColinAgent::TemplateSnapshot {
                            ColinAgent::TemplateSnapshot snap;
                            auto proto = weak_proto.lock();
                            if (!proto) return snap;
                            auto* iface = proto->get_template_interface();
                            if (!iface) return snap;

                            auto stats = iface->get_stats();
                            snap.templates_received       = stats.templates_received;
                            snap.templates_validated      = stats.templates_validated;
                            snap.templates_rejected       = stats.templates_rejected;
                            snap.templates_stale          = stats.templates_stale;
                            snap.templates_fed            = stats.templates_fed;
                            snap.templates_expired_age    = stats.templates_expired_age;
                            snap.templates_expired_height = stats.templates_expired_height;

                            if (iface->has_valid_template())
                            {
                                if (const auto* tmpl = iface->get_current_template())
                                {
                                    snap.has_valid_template = true;
                                    snap.unified_height = tmpl->block.nHeight;
                                    snap.nBits          = tmpl->nBits;
                                    snap.channel_height = tmpl->nChannelHeight;
                                    snap.channel        = tmpl->block.nChannel;
                                    snap.state_name     = protocol::MiningTemplateInterface::state_to_string(tmpl->state);
                                    // Use HeightTracker for canonical age calculation
                                    auto ht_snap = proto->get_height_tracker_snapshot();
                                    snap.age_seconds = ht_snap.get_template_age_seconds();
                                }
                            }
                            return snap;
                        });
                }

                // Wire up ColinPingHandler
                self->m_colin_agent->set_pong_telemetry_source(
                    [weak_proto]() -> ColinAgent::PongTelemetrySnapshot {
                        ColinAgent::PongTelemetrySnapshot pt;
                        auto proto = weak_proto.lock();
                        if (!proto) return pt;
                        const auto& handler = proto->get_ping_handler();
                        pt.ping_count  = handler.ping_count();
                        pt.last_rtt_us = handler.last_rtt_us();
                        return pt;
                    });

                self->m_logger->info("[Worker_manager] Colin diagnostic sources wired");
            }

            // Wire MinedBlockCache source
            std::weak_ptr<Worker_manager> weak_wm = self->shared_from_this();
            self->m_colin_agent->set_mined_block_cache_source(
                [weak_wm]() -> std::vector<ColinAgent::MinedBlockSnapshot> {
                    std::vector<ColinAgent::MinedBlockSnapshot> result;
                    auto wm = weak_wm.lock();
                    if (!wm) return result;
                    const auto& tier1 = wm->m_mined_block_cache.tier1();
                    result.reserve(tier1.size());
                    for (const auto& rec : tier1) {
                        ColinAgent::MinedBlockSnapshot snap;
                        snap.height = rec.height;
                        snap.channel = rec.channel;
                        snap.confirmations = rec.confirmations;
                        snap.status_emoji = rec.status_emoji();
                        snap.channel_name = rec.channel_name();
                        snap.hash_prev_block_hex = rec.hash_prev_block.GetHex();
                        result.push_back(std::move(snap));
                    }
                    return result;
                });

            // Wire failover state source
            self->m_colin_agent->set_failover_source(
                [weak_wm]() -> ColinAgent::FailoverSnapshot {
                    ColinAgent::FailoverSnapshot snap;
                    auto wm = weak_wm.lock();
                    if (!wm) return snap;
                    auto fs = wm->get_failover_status();
                    snap.has_failover_configured = fs.has_failover_configured;
                    snap.using_failover          = fs.using_failover;
                    snap.primary_fail_count      = fs.primary_fail_count;
                    snap.failover_max_retries    = fs.failover_max_retries;
                    snap.active_endpoint_str  = fs.using_failover ? fs.failover_endpoint_str : fs.primary_endpoint_str;
                    snap.standby_endpoint_str = fs.using_failover ? fs.primary_endpoint_str  : fs.failover_endpoint_str;
                    if (fs.using_failover && fs.failover_activated_at != std::chrono::steady_clock::time_point{}) {
                        snap.failover_active_seconds = static_cast<uint64_t>(
                            std::chrono::duration_cast<std::chrono::seconds>(
                                std::chrono::steady_clock::now() - fs.failover_activated_at).count());
                    }
                    snap.secondary_ip = wm->m_sim_link.get_active_node_ip();
                    return snap;
                });

            self->m_colin_agent->start();
            self->m_logger->info("[Worker_manager] Colin agent started");
        }
    });
}


bool Worker_manager::transmit_prepared_solution(const std::vector<uint8_t>& full_block_bytes, uint64_t nNonce)
{
    if (!m_primary_node_session ||
        !m_primary_node_session->is_primary_connected() ||
        !m_primary_node_session->is_authenticated())
    {
        m_logger->error("[Worker_manager] Block submission failed — transport or session is not usable "
                        "(connected={}, authenticated={})",
                        m_primary_node_session && m_primary_node_session->is_primary_connected(),
                        m_primary_node_session && m_primary_node_session->is_authenticated());
        return false;
    }

    auto packet = m_primary_node_session->submit_block(full_block_bytes, nNonce);
    if (!packet || packet->empty())
    {
        m_logger->error("[Worker_manager] Block submission failed — protocol did not build a payload");
        return false;
    }
    if (!m_primary_node_session->transmit(packet))
    {
        m_logger->error("[Worker_manager] Block submission failed — payload was not queued");
        return false;
    }
    return true;
}

void Worker_manager::schedule_connection_retry(std::uint16_t delay_seconds,
                                               network::Endpoint const& endpoint)
{
    m_connection_retry_scheduled = true;
    m_timer_manager.start_connection_retry_timer(delay_seconds, shared_from_this(), endpoint);
}

void Worker_manager::ensure_transport_lost_handler()
{
    if (!m_primary_node_session) {
        return;
    }

    // Captured here, after std::make_shared has attached enable_shared_from_this.
    // The constructor runs before that attachment, so a weak_ptr taken there
    // stays empty and every notification returns without reconnecting.
    m_primary_node_session->set_transport_lost_handler([weak_self = weak_from_this()]() {
        auto self = weak_self.lock();
        if (!self) {
            return;
        }
        self->handle_transport_lost("primary_socket_closed");
    });
    m_logger->info("[Worker_manager] Transport-lost handler registered");
}

bool Worker_manager::transport_recovery_outstanding() const
{
    return m_connection_retry_scheduled ||
           (m_primary_node_session && m_primary_node_session->has_pending_connect());
}

void Worker_manager::accept_solved_block(std::unique_ptr<Block_data> block_data)
{
    if (!block_data) {
        m_logger->error("[Worker_manager] Solved-block callback had no block data");
        return;
    }
    if (m_terminal_stop_requested.load(std::memory_order_acquire)) {
        m_logger->warn("[Worker_manager] Solved block ignored after terminal stop");
        return;
    }

    // Preserve the worker-owned snapshot before authentication or template gates.
    // mark_lane_socket_failed() ends the session immediately, so a callback that
    // arrives after the drop never reaches payload construction. Session-dependent
    // work is deferred to flush_pending_submit().
    m_pending_submit = PendingSubmit{
        std::shared_ptr<Block_data>(std::move(block_data)),
        std::chrono::steady_clock::now()
    };
    schedule_pending_submit_expiry();
    flush_pending_submit("block_found");
}

void Worker_manager::schedule_pending_submit_expiry()
{
    if (m_pending_submit_expiry_timer) {
        m_pending_submit_expiry_timer->cancel();
    }
    auto timer = std::make_shared<asio::steady_timer>(
        *m_io_context, m_pending_submit->found_at + std::chrono::seconds(PENDING_SUBMIT_MAX_AGE_SECONDS));
    m_pending_submit_expiry_timer = timer;
    timer->async_wait([weak_self = weak_from_this(), timer](const asio::error_code& ec) {
        if (ec == asio::error::operation_aborted) return;
        auto self = weak_self.lock();
        // A replaced snapshot may have an old timer callback already queued.
        if (!self || self->m_pending_submit_expiry_timer != timer) return;
        if (ec) {
            self->m_logger->error("[Worker_manager] Solved-block expiry timer failed: {}", ec.message());
            return;
        }
        self->flush_pending_submit("expiry_timer");
    });
}

void Worker_manager::clear_pending_submit()
{
    m_pending_submit.reset();
    if (m_pending_submit_expiry_timer) {
        m_pending_submit_expiry_timer->cancel();
        m_pending_submit_expiry_timer.reset();
    }
}

void Worker_manager::flush_pending_submit(const char* reason)
{
    // Flushes a worker-found solved block that may have been stashed while
    // transport/session state was unstable. This routine is intentionally
    // conservative: it only submits when all runtime gates are healthy, and it
    // drops stale cached work to avoid emitting obsolete submissions.

    // Preconditions: there must be a cached solved snapshot to submit.
    // If none exists, or it has already been cleared, there is nothing to do.
    if (!m_pending_submit || !m_pending_submit->solved)
    {
        return;
    }

    // Staleness guard: solved blocks buffered during outages are only valid for
    // a bounded window. After TTL expiry we must discard to avoid submitting
    // stale work against a newer chain/template state.
    const auto now = std::chrono::steady_clock::now();
    const auto age_s = std::chrono::duration_cast<std::chrono::seconds>(
        now - m_pending_submit->found_at).count();
    if (now >= m_pending_submit->found_at + std::chrono::seconds(PENDING_SUBMIT_MAX_AGE_SECONDS))
    {
        m_logger->error("[Worker_manager] Dropping solved block stashed during transport loss "
                        "(age {}s >= {}s)",
                        age_s, PENDING_SUBMIT_MAX_AGE_SECONDS);
        clear_pending_submit();
        return;
    }

    const auto solved = m_pending_submit->solved;
    const char* why = reason ? reason : "recovery";

    if (!m_primary_node_session ||
        !m_primary_node_session->is_primary_connected() ||
        !m_primary_node_session->is_authenticated())
    {
        m_logger->error("[Worker_manager] No authenticated session. Solved block retained for recovery "
                        "(height={} nonce=0x{:016x}, reason={})",
                        solved->nHeight, solved->nNonce, why);
        return;
    }

    auto solo_protocol = m_primary_node_session->get_primary_protocol();
    if (!solo_protocol)
    {
        m_logger->error("[Worker_manager] Failed to get protocol from NodeSession — solved block retained");
        return;
    }

    auto* template_interface = solo_protocol->get_template_interface();
    if (!template_interface)
    {
        m_logger->error("[Worker_manager] Template interface not available — solved block retained");
        return;
    }

    auto ht_snap = solo_protocol->get_height_tracker_snapshot();
    uint64_t template_age = ht_snap.get_template_age_seconds();
    const bool tip_known = (ht_snap.hash_prev_block != uint1024_t(0));
    const bool prev_hash_mismatch =
        tip_known && (solved->previous_hash != ht_snap.hash_prev_block);

    if (prev_hash_mismatch)
    {
        const char* soft_refresh_reason = "submit_side_prev_hash_mismatch";
        m_logger->error("[Worker_manager] ❌ Solution found but tip has moved — hashPrevBlock mismatch (age: {}s)",
                       template_age);
        m_logger->error("[Worker_manager]    solved_prev={}... canonical_tip={}...",
                       solved->previous_hash.SubString(),
                       ht_snap.hash_prev_block.SubString());
        m_logger->error("[Worker_manager]    Discarding stale solution (node would reject as 'stale block')");
        clear_pending_submit();
        mark_soft_refresh_requested(soft_refresh_reason);
        template_interface->discard_template("hashPrevBlock no longer matches canonical tip");

        m_logger->info("[Worker_manager] Requesting fresh work/GET_BLOCK via NodeSession");
        auto work_payload = m_primary_node_session->request_work(protocol::GetBlockReason::HEALTH_CHANNEL_STALE);
        if (work_payload && !work_payload->empty()) {
            if (m_primary_node_session->transmit(work_payload)) {
                auto proto = m_primary_node_session->get_active_protocol();
                if (proto) proto->mark_get_block_pending(protocol::GetBlockReason::HEALTH_CHANNEL_STALE);
            }
        }
        return;
    }

    if (ht_snap.is_template_age_stale()) {
        m_logger->warn("[Worker_manager] ⚠️  Submitting solution on aged template ({}s > 600s) — "
                       "hashPrevBlock still matches tip; push pipe may be quiet",
                       template_age);
    } else if (ht_snap.is_template_age_old()) {
        m_logger->info("[Worker_manager] Submitting solution on aged template ({}s) — "
                       "hashPrevBlock still matches tip",
                       template_age);
    }

    m_logger->info("[Worker_manager] 💎 Solution found! Age: {}s, hashPrevBlock matches canonical tip ✅ - SUBMITTING", template_age);
    m_logger->info("[Worker_manager] Preparing full block submission ({})", why);
    m_logger->info("[Worker_manager]   Height: {}", solved->nHeight);
    m_logger->info("[Worker_manager]   Nonce:  0x{:016x}", solved->nNonce);
    if (!solved->vOffsets.empty())
        m_logger->info("[Worker_manager]   vOffsets: {} bytes (Prime channel)",
                       solved->vOffsets.size());

    auto full_block_bytes = (solved->nChannel == 1 && !solved->vOffsets.empty())
        ? template_interface->prepare_block_submission_from_solved(*solved, solved->vOffsets)
        : template_interface->prepare_block_submission_from_solved(*solved);

    if (full_block_bytes.empty())
    {
        m_logger->error("[Worker_manager] Failed to prepare block submission — solved block retained "
                        "until a template can build the payload");
        return;
    }

    m_logger->info("[Worker_manager] Full block serialized: {} bytes", full_block_bytes.size());
    if (!transmit_prepared_solution(full_block_bytes, solved->nNonce))
    {
        m_logger->warn("[Worker_manager] Solved block still queued after {} (age {}s)",
                       why, age_s);
        return;
    }

    m_logger->info("[Worker_manager] Submitted solved block ({}, age {}s)", why, age_s);
    clear_pending_submit();
}

void Worker_manager::handle_transport_lost(const char* reason)
{
    if (m_terminal_stop_requested.load(std::memory_order_acquire) || is_degraded_terminal())
    {
        return;
    }
    // RECONNECTING covers the whole recovery phase, including the window after
    // the connect callback has completed and before work is ready. Ignore a
    // duplicate only when a retry timer or connect attempt is actually outstanding.
    if (transport_recovery_outstanding())
    {
        m_logger->debug("[Worker_manager] Transport lost ({}) ignored — reconnect already scheduled",
                        reason ? reason : "unknown");
        return;
    }
    if (!m_primary_endpoint.is_valid())
    {
        m_logger->error("[Worker_manager] Transport lost ({}) but no endpoint is available for reconnect",
                        reason ? reason : "unknown");
        return;
    }

    m_logger->error("[Worker_manager] Primary transport lost ({}) — stopping workers and reconnecting",
                    reason ? reason : "unknown");
    stop_all_workers();
    m_pending_recovery_completion_kind = RecoveryCompletionKind::TRANSPORT_RECONNECT;
    // force_transport_reset: a recent push timestamp must not suppress TCP
    // reconnect or attempt in-band login on the socket that just died.
    retry_connect(m_primary_endpoint, true);
}

void Worker_manager::log_lane_health()
{
    bool primary_alive = m_primary_node_session && m_primary_node_session->is_authenticated();

    m_logger->info("[NodeSession] Session health — Authenticated: {}", primary_alive ? "YES" : "NO");

    // Refresh the aggregate lane-alive heartbeat on every periodic tick so it
    // reflects sustained liveness (not just alive-transition edges), matching
    // how push liveness is tracked.
    m_sim_link.touch_alive_heartbeat();

    send_session_status_if_due();
}

void Worker_manager::poll_get_round()
{
    if (!m_primary_node_session)
        return;

    // Session generation guard: capture at entry so we detect transitions
    // that happen between protocol lookup and transmit.
    const uint64_t gen_at_entry = m_session_generation.load(std::memory_order_acquire);

    auto solo_protocol = m_primary_node_session->get_active_protocol();
    if (!solo_protocol)
        return;

    // Intelligent polling: only send if protocol says it's time
    if (!solo_protocol->should_send_get_round())
        return;

    // send_get_round() sends GET_ROUND on all lanes (legacy: 0x85,
    // stateless: 0xD085) as a pure height/difficulty sanity probe.
    // Template recovery is handled separately by Worker_manager via
    // send_recovery_work_request() (GET_BLOCK).
    auto payload = solo_protocol->send_get_round();
    if (!payload || payload->empty()) {
        return;  // Build failure (not authenticated, etc.) — counter was not touched.
    }

    // Re-check generation before transmit: if a session transition happened
    // while building the packet, the node would reject it anyway.
    if (m_session_generation.load(std::memory_order_acquire) != gen_at_entry) {
        m_logger->debug("[Worker_manager] GET_ROUND discarded pre-transmit: session generation changed ({} → {})",
                       gen_at_entry, m_session_generation.load(std::memory_order_relaxed));
        return;  // Counter not incremented — nothing to roll back.
    }

    if (!m_primary_node_session->transmit(payload)) {
        m_logger->warn("[Worker_manager] GET_ROUND not queued — counter untouched (no shadow-ban account)");
        return;
    }

    // Commit: packet is on the wire.  Only now do we count it as 'unanswered'.
    solo_protocol->note_get_round_transmitted();

    // Shadow-ban detection: combine count + elapsed-time + push silence to avoid
    // false positives from transient network jitter or slow replies.
    constexpr uint32_t SHADOW_BAN_UNANSWERED_THRESHOLD = 5;
    constexpr int64_t  SHADOW_BAN_MIN_ELAPSED_SECONDS  = 90;  // At 15s cadence, 5 sends ≈ 60–75s; require >= 90s

    const auto unanswered = solo_protocol->get_unanswered_get_round_count();
    if (unanswered >= SHADOW_BAN_UNANSWERED_THRESHOLD) {
        auto earliest = solo_protocol->get_earliest_unanswered_get_round_at();
        int64_t elapsed_s = (earliest != std::chrono::steady_clock::time_point{})
            ? std::chrono::duration_cast<std::chrono::seconds>(
                  std::chrono::steady_clock::now() - earliest).count()
            : 0;

        auto ht = solo_protocol->get_height_tracker_snapshot();
        bool push_received = (ht.last_push_notification_at != std::chrono::steady_clock::time_point{});
        int64_t push_age_s = push_received
            ? std::chrono::duration_cast<std::chrono::seconds>(
                  std::chrono::steady_clock::now() - ht.last_push_notification_at).count()
            : INT64_MAX;
        bool push_silent = (push_age_s >= PUSH_ALIVE_THRESHOLD_SECONDS);

        constexpr int64_t ONE_WAY_SHADOW_BAN_ELAPSED_SECONDS = 180;
        const auto last_status_ack_at = solo_protocol->last_session_status_ack_time();
        const int64_t status_ack_age_s =
            last_status_ack_at != std::chrono::steady_clock::time_point{}
                ? std::chrono::duration_cast<std::chrono::seconds>(
                      std::chrono::steady_clock::now() - last_status_ack_at).count()
                : -1;

        if (elapsed_s >= SHADOW_BAN_MIN_ELAPSED_SECONDS && push_silent) {
            m_logger->error("[Worker_manager] SHADOW BAN DETECTED: {} consecutive GET_ROUNDs "
                            "unanswered over {}s, push silent {}s — forcing session recovery",
                            unanswered, elapsed_s, push_age_s == INT64_MAX ? -1 : push_age_s);
            mark_recovery_initiated("shadow_ban_unanswered_get_round");
            if (solo_protocol->is_get_block_retry_deferred()) {
                retry_connect(m_primary_endpoint, true);
            }
        } else if (elapsed_s >= ONE_WAY_SHADOW_BAN_ELAPSED_SECONDS && !push_silent && !is_reconnecting()) {
            m_logger->error("[Worker_manager] ONE-WAY SESSION FAILURE SUSPECTED: {} consecutive GET_ROUNDs "
                            "unanswered over {}s while push is still live (push_age={}s, status_ack_age={}s) "
                            "— forcing hard transport reconnect",
                            unanswered,
                            elapsed_s,
                            push_age_s == INT64_MAX ? -1 : push_age_s,
                            status_ack_age_s);
            mark_recovery_initiated("one_way_shadow_ban");
            retry_connect(m_primary_endpoint, true);
        } else {
            m_logger->warn("[Worker_manager] Unanswered GET_ROUND count {} >= threshold {}, "
                           "but shadow-ban suppressed (elapsed={}s, push_age={}s, status_ack_age={}s)",
                           unanswered, SHADOW_BAN_UNANSWERED_THRESHOLD,
                           elapsed_s,
                           push_age_s == INT64_MAX ? -1 : push_age_s,
                           status_ack_age_s);
        }
    }
}

void Worker_manager::send_session_status_if_due()
{
    auto now = std::chrono::steady_clock::now();
    constexpr int64_t SESSION_STATUS_INTERVAL_SECONDS = 300;
    if (std::chrono::duration_cast<std::chrono::seconds>(
            now - m_last_session_status_sent).count() < SESSION_STATUS_INTERVAL_SECONDS)
        return;

    // Session generation guard: don't send status for a session in transition
    const uint64_t gen_at_entry = m_session_generation.load(std::memory_order_acquire);

    bool degraded    = is_degraded();
    bool workers_run = !is_degraded() && !m_workers.empty();

    // Send session status via NodeSession (handles both lanes internally)
    if (m_primary_node_session && m_primary_node_session->is_authenticated())
    {
        auto solo_protocol = m_primary_node_session->get_active_protocol();
        if (solo_protocol)
        {
            // NodeSession is lane-scoped; lane health is diagnostic only.
            auto pkt = solo_protocol->build_session_status_packet(degraded, workers_run, false);
            if (pkt && !pkt->empty()) {
                if (m_session_generation.load(std::memory_order_acquire) != gen_at_entry) {
                    m_logger->debug("[Worker_manager] SESSION_STATUS discarded: session generation changed ({} → {})",
                                   gen_at_entry, m_session_generation.load(std::memory_order_relaxed));
                    return;
                }
                if (!m_primary_node_session->transmit(pkt)) {
                    m_logger->warn("[Worker_manager] SESSION_STATUS not queued on active node session");
                    return;
                }
                m_last_session_status_sent = now;
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Worker Control Methods (Degraded Mode Support)
// ═══════════════════════════════════════════════════════════════════════

// ── State machine helpers ────────────────────────────────────────────────

const char* Worker_manager::phase_name(RecoveryPhase phase) {
    switch (phase) {
        case RecoveryPhase::HEALTHY:              return "HEALTHY";
        case RecoveryPhase::WAITING_TEMPLATE:     return "WAITING_TEMPLATE";
        case RecoveryPhase::SESSION_RECOVERY:     return "SESSION_RECOVERY";
        case RecoveryPhase::RECONNECTING:         return "RECONNECTING";
        case RecoveryPhase::DEGRADED_TERMINAL:    return "DEGRADED_TERMINAL";
    }
    return "UNKNOWN";
}

bool Worker_manager::is_valid_transition(RecoveryPhase from, RecoveryPhase to) {
    if (from == to) return true;

    // DEGRADED_TERMINAL is genuinely terminal — no legal exit, ever.
    if (from == RecoveryPhase::DEGRADED_TERMINAL) return false;

    if (to == RecoveryPhase::DEGRADED_TERMINAL) return true;

    switch (from) {
        case RecoveryPhase::HEALTHY:
            return to == RecoveryPhase::WAITING_TEMPLATE ||
                   to == RecoveryPhase::SESSION_RECOVERY ||
                   to == RecoveryPhase::RECONNECTING;
        case RecoveryPhase::WAITING_TEMPLATE:
            return to == RecoveryPhase::HEALTHY ||
                   to == RecoveryPhase::SESSION_RECOVERY ||
                   to == RecoveryPhase::RECONNECTING;
        case RecoveryPhase::SESSION_RECOVERY:
            return to == RecoveryPhase::WAITING_TEMPLATE ||
                   to == RecoveryPhase::RECONNECTING;
        case RecoveryPhase::RECONNECTING:
            return to == RecoveryPhase::HEALTHY ||
                   to == RecoveryPhase::WAITING_TEMPLATE;
    }
    // DEGRADED_TERMINAL is handled above.
    return false;
}

void Worker_manager::on_phase_exit(RecoveryPhase old_phase) {
    switch (old_phase) {
        case RecoveryPhase::SESSION_RECOVERY:
            m_recovery.recovery_in_progress.store(false, std::memory_order_release);
            break;
        case RecoveryPhase::RECONNECTING:
            m_recovery.reconnect_started_at = {};
            break;
        default:
            break;
    }
}

void Worker_manager::on_phase_enter(RecoveryPhase new_phase) {
    switch (new_phase) {
        case RecoveryPhase::HEALTHY: {
            auto now = std::chrono::steady_clock::now();
            m_recovery.degraded_since = {};
            m_recovery.timeout.reset();
            m_recovery.last_completed_at = now;
            m_recovery.entered_at = {};
            m_recovery.recovery_in_progress.store(false, std::memory_order_release);
            // Template successfully adopted — reset GET_BLOCK mismatch backoff.
            m_get_block_backoff_ms    = 0;
            m_get_block_backoff_until = {};
            m_stats_collector->set_degraded_mode(false);
            break;
        }
        case RecoveryPhase::WAITING_TEMPLATE: {
            auto now = std::chrono::steady_clock::now();
            bool is_new_outage = (m_recovery.degraded_since == std::chrono::steady_clock::time_point{});
            if (is_new_outage) {
                m_recovery.degraded_since = now;
                ++m_degraded_enter_total;
            }
            m_recovery.timeout.start_if_unset(m_recovery.degraded_since);
            m_stats_collector->set_degraded_mode(true);
            break;
        }
        case RecoveryPhase::SESSION_RECOVERY: {
            auto now = std::chrono::steady_clock::now();
            bool is_new_outage = (m_recovery.degraded_since == std::chrono::steady_clock::time_point{});
            if (is_new_outage) {
                m_recovery.degraded_since = now;
                ++m_degraded_enter_total;
            }
            m_recovery.timeout.start_if_unset(m_recovery.degraded_since);
            m_recovery.recovery_in_progress.store(true, std::memory_order_release);
            m_stats_collector->set_degraded_mode(true);
            break;
        }
        case RecoveryPhase::RECONNECTING: {
            m_recovery.reconnect_started_at = std::chrono::steady_clock::now();
            break;
        }
        case RecoveryPhase::DEGRADED_TERMINAL: {
            m_stats_collector->set_degraded_mode(true);
            break;
        }
    }
}


bool Worker_manager::should_reset_stats_on_recovery_completion() const
{
    const auto kind = m_pending_recovery_completion_kind;
    auto solo_protocol = m_primary_node_session ? m_primary_node_session->get_primary_protocol() : nullptr;
    auto* session_manager = solo_protocol ? solo_protocol->get_session_manager() : nullptr;
    const auto recovery_state = session_manager
        ? session_manager->get_runtime_snapshot().recovery_state
        : protocol::SessionManager::RecoveryState::HEALTHY;

    switch (kind) {
        case RecoveryCompletionKind::FAILOVER_SESSION:
            return true;
        case RecoveryCompletionKind::PRIMARY_SESSION_REAUTH:
            return true; // Safe default until same-node preservation policy is refined.
        case RecoveryCompletionKind::TRANSPORT_RECONNECT:
            return true;
        case RecoveryCompletionKind::TEMPLATE_REFRESH:
        case RecoveryCompletionKind::NONE:
            return recovery_state == protocol::SessionManager::RecoveryState::RECOVERY_IN_PROGRESS ||
                   recovery_state == protocol::SessionManager::RecoveryState::FORCED_REAUTH ||
                   recovery_state == protocol::SessionManager::RecoveryState::RECONNECT_REQUIRED;
    }
    return false;
}

void Worker_manager::mark_recovery_completion_kind_template_refresh()
{
    if (m_pending_recovery_completion_kind == RecoveryCompletionKind::NONE ||
        m_pending_recovery_completion_kind == RecoveryCompletionKind::TEMPLATE_REFRESH) {
        m_pending_recovery_completion_kind = RecoveryCompletionKind::TEMPLATE_REFRESH;
    }
}

uint16_t Worker_manager::next_session_auth_retry_delay_seconds() const
{
    // m_session_auth_fail_count is 1-based by the time this is consulted
    // (incremented before the check in both call sites), matching
    // ExponentialBackoff::calculate_delay_seconds()'s 1-based attempt_count
    // (attempt 1 = first retry after the first failure, delay = base;
    // attempt 2 = base*2; ...). Passing it through directly, with no -1/+1
    // adjustment, is what keeps the two indexing schemes aligned.
    return static_cast<uint16_t>(m_session_auth_backoff.calculate_delay_seconds(m_session_auth_fail_count));
}

void Worker_manager::transition_to(RecoveryPhase new_phase, const char* reason) {
    RecoveryPhase old_phase = m_recovery.phase.load(std::memory_order_acquire);
    if (old_phase == new_phase) {
        return;  // Already in this phase — no-op
    }

    if (!is_valid_transition(old_phase, new_phase)) {
        m_logger->error("[Worker_manager] ⚡ ILLEGAL TRANSITION: {} → {} (reason: {})",
                        phase_name(old_phase), phase_name(new_phase),
                        reason ? reason : "unknown");
        // In debug builds, assert to catch illegal transitions early during development.
        // In release builds, log and proceed to avoid hard crashes in production.
#ifndef NDEBUG
        assert(false && "Illegal RecoveryPhase transition — see error log above");
#endif
    }

    // Pre-transition accounting: if exiting WAITING_TEMPLATE to HEALTHY,
    // accumulate total time spent in degraded mode.
    if (new_phase == RecoveryPhase::HEALTHY &&
        old_phase != RecoveryPhase::HEALTHY &&
        old_phase != RecoveryPhase::DEGRADED_TERMINAL) {
        auto now = std::chrono::steady_clock::now();
        if (m_recovery.degraded_since != std::chrono::steady_clock::time_point{}) {
            auto elapsed_ms = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    now - m_recovery.degraded_since).count());
            m_time_in_degraded_ms += elapsed_ms;
        }
        ++m_degraded_exit_total;
    }

    on_phase_exit(old_phase);

    m_recovery.phase.store(new_phase, std::memory_order_release);
    m_recovery.reason = reason;

    // Reset per-epoch state for all non-HEALTHY phases
    if (new_phase != RecoveryPhase::HEALTHY) {
        m_epoch_coordinator->advance_recovery_epoch(reason ? reason : "unknown");
        m_recovery.entered_at = std::chrono::steady_clock::now();
        m_recovery.get_block_confirmed = false;
        m_recovery.last_get_block_at = {};
        m_forced_retry_timer_pending = false;
        ++m_forced_retry_timer_token;
        if (m_forced_retry_timer) {
            m_forced_retry_timer->cancel();
        }
    }

    on_phase_enter(new_phase);

    if (new_phase == RecoveryPhase::HEALTHY && should_reset_stats_on_recovery_completion()) {
        m_stats_collector->reset_global_counters();
        m_stats_collector->reset_start_time();
    }
    if (new_phase == RecoveryPhase::HEALTHY) {
        m_pending_recovery_completion_kind = RecoveryCompletionKind::NONE;
    }

    m_logger->info("[Worker_manager] ⚡ TRANSITION: {} → {} (epoch={}, reason={})",
                   phase_name(old_phase), phase_name(new_phase),
                   m_epoch_coordinator->recovery_epoch(), reason ? reason : "unknown");
}


void Worker_manager::mark_recovery_initiated(const char* reason)
{
    // Idempotent: if already in WAITING_TEMPLATE or RECONNECTING,
    // a new epoch is already running — do NOT reset it.
    if (is_recovery_active()) {
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - m_recovery.entered_at).count();
        m_logger->info("[Worker_manager] Recovery already pending (epoch {}, {}s elapsed, reason: {})",
                       m_epoch_coordinator->recovery_epoch(), elapsed, reason ? reason : "unknown");
        return;
    }
    // From HEALTHY → WAITING_TEMPLATE (new epoch; workers keep running)
    mark_recovery_completion_kind_template_refresh();
    transition_to(RecoveryPhase::WAITING_TEMPLATE, reason);
    m_logger->warn("[Worker_manager] ⚑ RECOVERY INITIATED — epoch {} (reason: {})",
                   m_epoch_coordinator->recovery_epoch(), reason ? reason : "unknown");
    m_logger->warn("[Worker_manager]   Workers keep running with current template while requesting fresh one");
}

void Worker_manager::mark_soft_refresh_requested(const char* reason)
{
    // Template-only refresh maps to WAITING_TEMPLATE.
    // Workers keep running; no submissions withheld in the current model.
    mark_recovery_initiated(reason);
}

void Worker_manager::restart_recovery_window(const char* reason)
{
    // Reset the current-epoch timing state while staying in the current recovery phase.
    // Used after successful re-authentication to give the new session a clean recovery
    // window clock without exiting degraded mode (workers are still stopped, no template yet).
    const bool had_pending_recovery = is_recovery_active();
    const bool had_forced_retry_timer = m_forced_retry_timer_pending;
    int64_t elapsed_s = 0;
    if (had_pending_recovery && m_recovery.entered_at != std::chrono::steady_clock::time_point{}) {
        elapsed_s = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - m_recovery.entered_at).count();
    }

    // Reset per-epoch state in-place (no phase change)
    m_recovery.entered_at = std::chrono::steady_clock::now();  // fresh epoch clock
    m_recovery.degraded_since = m_recovery.entered_at;
    m_recovery.timeout.reset();
    m_recovery.timeout.start_if_unset(m_recovery.degraded_since);
    m_recovery.last_get_block_at = {};
    m_recovery.get_block_confirmed = false;
    m_forced_retry_timer_pending = false;
    ++m_forced_retry_timer_token;
    if (m_forced_retry_timer) {
        m_forced_retry_timer->cancel();
    }

    if (had_pending_recovery || had_forced_retry_timer) {
        m_logger->info("[Worker_manager] Restarting recovery window after {} "
                       "(prior_epoch={} prior_elapsed={}s phase={})",
                       reason ? reason : "unknown",
                       m_epoch_coordinator->recovery_epoch(),
                       elapsed_s,
                       phase_name(m_recovery.phase.load(std::memory_order_relaxed)));
    }
}

void Worker_manager::handle_recovery_confirmed(const char* reason)
{
    // Fires unconditionally from Solo whenever the same-height feed guard
    // suppresses a re-feed, whether or not a recovery is currently active.
    // Only act when there is actually something to clear.
    if (!is_recovery_active()) {
        return;
    }

    if (is_degraded_terminal()) {
        return;
    }

    m_logger->info("[Worker_manager] Recovery-confirmed signal received ({}): node re-served the "
                   "template already loaded — current tip is still canonical. Attempting to clear "
                   "recovery without a worker re-feed (phase={})",
                   reason ? reason : "unknown",
                   phase_name(m_recovery.phase.load(std::memory_order_relaxed)));

    // clear_recovery_state() independently re-validates that a valid template is
    // available and the authoritative session is restored before actually clearing —
    // it is safe to call here even though no new distribution to workers occurred.
    const bool was_active = is_recovery_active();
    clear_recovery_state();

    if (was_active && !is_recovery_active()) {
        ++m_recovery_confirmed_clears_total;
        m_logger->warn("[Worker_manager] ✅ Recovery cleared via same-tip reconfirmation ({}) — "
                       "no worker re-feed was necessary (cumulative reconfirmation-clears={})",
                       reason ? reason : "unknown", m_recovery_confirmed_clears_total);
    }
}

void Worker_manager::clear_recovery_state()
{
    if (!is_degraded() && !is_recovery_active())
        return;  // Already HEALTHY — nothing to clear

    auto solo_protocol = m_primary_node_session ? m_primary_node_session->get_active_protocol() : nullptr;
    if (!solo_protocol && m_primary_node_session) solo_protocol = m_primary_node_session->get_primary_protocol();
    bool has_valid_template_now = has_valid_template_available(solo_protocol);
    if (!has_valid_template_now) {
        m_logger->warn("[Worker_manager] clear_recovery_state() deferred: no valid template accepted yet");
        return;
    }

    auto* session_manager = solo_protocol ? solo_protocol->get_session_manager() : nullptr;
    if (!session_manager) {
        m_logger->warn("[Worker_manager] clear_recovery_state() deferred: authoritative session manager unavailable");
        return;
    }

    const auto binding = session_manager->get_session_binding();
    if (binding.active_lane == nexusminer::ProtocolLane::UNKNOWN ||
        binding.session_requires_full_recovery() ||
        !binding.may_request_work()) {
        m_logger->warn("[Worker_manager] clear_recovery_state() deferred: authoritative session not restored "
                       "(session_requires_full_recovery={}, may_request_work={}, authenticated={}, has_session={}, active_lane={})",
                       binding.session_requires_full_recovery(),
                       binding.may_request_work(),
                       binding.authenticated,
                       binding.has_session(),
                       static_cast<int>(binding.active_lane));
        return;
    }

    auto now = std::chrono::steady_clock::now();
    auto recovery_age_s =
        (m_recovery.entered_at != std::chrono::steady_clock::time_point{})
        ? std::chrono::duration_cast<std::chrono::seconds>(now - m_recovery.entered_at).count()
        : 0;
    m_logger->info("[Worker_manager] Recovery exit gate: valid_template={} recovery_age={}s phase={}",
                   has_valid_template_now ? "YES" : "NO",
                   recovery_age_s,
                   phase_name(m_recovery.phase.load(std::memory_order_relaxed)));

    const auto session_snapshot = session_manager->get_runtime_snapshot();
    if (session_snapshot.recovery_state != protocol::SessionManager::RecoveryState::HEALTHY ||
        !session_snapshot.recovery_reason.empty()) {
        solo_protocol->mark_authoritative_recovery_healthy("worker_manager_clear_recovery_state");
    }

    m_logger->info("[Worker_manager] Clearing recovery state — exiting {} phase",
                   phase_name(m_recovery.phase.load(std::memory_order_relaxed)));

    // Note: m_recovery_workers_spawned is intentionally NOT reset here.
    // It is only reset in stop_all_workers() which actually destroys workers,
    // preventing a mid-recovery clear_recovery_state() call (e.g. from a
    // different epoch's template feed) from allowing duplicate worker creation.

    // transition_to(HEALTHY) handles: degraded time accounting, global stats,
    // stats reset, last_completed_at, clearing of all recovery fields.
    transition_to(RecoveryPhase::HEALTHY, "template_distributed");

    m_logger->info("[Worker_manager] Recovery state cleared — degraded_exit_count={} cumulative_degraded_time_ms={}",
                   m_degraded_exit_total, m_time_in_degraded_ms);
}

void Worker_manager::stop_all_workers()
{
    // ── Lock-scope discipline ──────────────────────────────────────────────
    // Destroying a Worker (shared_ptr reset) joins its mining thread in the
    // destructor; destroying the PrimeMiningEngine joins its pool/consumer/
    // stats threads.  Doing either of those while still holding
    // m_worker_mutex would mean the io_context thread (which calls
    // stop_all_workers() synchronously from several recovery/shutdown paths)
    // blocks on a join *while holding the lock* — any other code path that
    // ever needs m_worker_mutex from inside one of those joined threads
    // (directly, or transitively) would deadlock permanently.  No such path
    // exists today, but nothing enforces that invariant either, and it is
    // exactly the "silent freeze" failure mode this function must never
    // become an instance of.
    //
    // The fix: snapshot every shared_ptr/vector that teardown needs into
    // locals while the lock is held, release the lock, then destroy/join the
    // locals with no lock held at all.  This mirrors the "Stone 4" pattern
    // already used by the template-distribution fan-out below (see its
    // comment: "m_worker_mutex released here").
    std::shared_ptr<WorkerTemplateFeed> feed_to_teardown;
    std::vector<std::shared_ptr<Worker>> workers_to_teardown;
#ifdef PRIME_ENABLED
    std::shared_ptr<cpu::PrimeMiningEngine> engine_to_teardown;
#endif
    bool truly_degraded = false;
    bool session_active = false;

    {
        std::lock_guard lock(m_worker_mutex);

        // Header wording depends on WHY workers are being stopped: a normal
        // self-healing recovery pause (WAITING_TEMPLATE / SESSION_RECOVERY /
        // RECONNECTING — workers resume automatically once a fresh template/
        // session arrives) reads very differently from an operator/fatal stop.
        // Conflating the two in logs is what previously made a normal, ~35s
        // reorg-storm recovery look like a permanent failure.
        truly_degraded = is_degraded_terminal();

        // Snapshot (move) the teardown targets out of member state.  From
        // this point on, any concurrent reader of m_workers/m_prime_engine/
        // m_template_feed (e.g. create_workers(), collect_worker_statistics(),
        // the template-distribution fan-out) observes the post-teardown empty
        // state immediately — there is no window where the members are
        // non-empty but their underlying threads are already being joined.
        workers_to_teardown = std::move(m_workers);
        m_workers.clear();
#ifdef PRIME_ENABLED
        engine_to_teardown = std::move(m_prime_engine);
#endif
        feed_to_teardown = std::move(m_template_feed);

        // Clear the recovery gate so the next epoch can re-create workers
        m_recovery_workers_spawned = false;
    }  // ── m_worker_mutex released here ────────────────────────────────────

    // Regression guard: from here to the end of this function we are about
    // to join threads (PrimeMiningEngine's pool/consumer/stats threads, and
    // each Worker's mining thread).  Assert (debug builds only) that
    // m_worker_mutex is not held on this thread — see
    // Util/include/debug_lock_audit.h for why this specific invariant
    // matters.
    nexusminer::util::assert_no_tracked_locks_held("Worker_manager::stop_all_workers (before join teardown)");

    const char* banner = truly_degraded
        ? "STOPPING ALL WORKERS (DEGRADED MODE — terminal, operator action required)"
        : "STOPPING ALL WORKERS (RECOVERY PAUSE — self-healing, will resume automatically)";

    m_logger->warn("[Worker_manager] ════════════════════════════════════════");
    m_logger->warn("[Worker_manager] ⚠️  {}", banner);
    m_logger->warn("[Worker_manager] ════════════════════════════════════════");

    if (auto solo_protocol = m_primary_node_session ? m_primary_node_session->get_primary_protocol() : nullptr) {
        solo_protocol->reset_worker_feed_guards();
        if (auto* session_manager = solo_protocol->get_session_manager()) {
            const auto binding = session_manager->get_session_binding();
            session_active = !binding.session_requires_full_recovery() && binding.has_session();
        }
        if (auto secondary = m_primary_node_session ? m_primary_node_session->get_secondary_protocol() : nullptr) {
            secondary->reset_worker_feed_guards();
        }
        if (session_active) {
            solo_protocol->mark_authoritative_recovery_required("workers_stopped_waiting_for_valid_template");
        }
    }

    // Stone 4: wake any worker blocked in WorkerTemplateFeed::wait_for_epoch_after
    // so it can observe its own m_shutdown / m_stop flag and exit cleanly before
    // its destructor joins the mining thread below.  We notify *before* resetting
    // the workers (the destructors below set those flags and join), but the feed
    // shared_ptr is kept alive locally, so the wake call is safe even if a
    // worker has already begun teardown on another thread.  No lock is held
    // here or below — every remaining step in this function only touches the
    // local snapshots taken above.
    if (feed_to_teardown) {
        feed_to_teardown->notify_wake();
    }

#ifdef PRIME_ENABLED
    // Stone 7: Reset the PrimeMiningEngine BEFORE workers and BEFORE the feed.
    //
    //   * BEFORE workers — engine pool threads dispatch on_found via
    //     asio::post(io_context); the captured lambda touches Worker_manager
    //     state.  Joining the engine here drains in-flight dispatches so
    //     workers and Worker_manager state are still alive while they run.
    //
    //   * BEFORE the feed — the engine consumer thread parks inside
    //     WorkerTemplateFeed::wait_for_epoch_after.  Engine destructor calls
    //     m_feed->notify_wake() to unpark it; the feed must therefore be alive.
    //
    // Engine destructor sets shutdown, wakes pool CV, joins pool threads
    // (Stone 6 ordering), then joins consumer.  No work for us beyond this
    // shared_ptr reset.  This join happens with NO lock held (see above).
    if (engine_to_teardown) {
        // Re-check immediately before the join-capable teardown step so a future
        // edit that re-acquires a tracked lock between the earlier assertion and
        // this point is still caught at the actual join site.
        nexusminer::util::assert_no_tracked_locks_held(
            "Worker_manager::stop_all_workers (before PrimeMiningEngine teardown)");
        m_logger->info("[Worker_manager] Tearing down PrimeMiningEngine");
        engine_to_teardown.reset();
    }
#endif

    // Reset all worker instances so that the next create_workers() call starts fresh
    // without duplicating existing workers.  The shared_ptr reset() destroys the Worker
    // object (and joins its mining thread in the destructor), effectively stopping it.
    // This join happens with NO lock held (see above).
    // Re-check immediately before the join-capable worker destructor loop so a
    // future edit that re-acquires a tracked lock before this point is still
    // caught at the actual join site.
    nexusminer::util::assert_no_tracked_locks_held(
        "Worker_manager::stop_all_workers (before worker teardown)");
    for (auto& worker : workers_to_teardown) {
        worker.reset();
    }
    workers_to_teardown.clear();

    // Stone 4: now that every worker has been destroyed (and joined its mining
    // thread), it is safe to release the feed.  A fresh feed will be created on
    // the next create_workers_locked() so the next worker batch starts at
    // epoch_id 1 rather than inheriting a stale epoch counter from the prior
    // generation (workers compare the loaded epoch_id against their last-seen
    // value, and a fresh feed gives them a clean baseline).
    feed_to_teardown.reset();

    if (truly_degraded) {
        m_logger->warn("[Worker_manager] Mining halted — terminal stop, operator action (or process restart) required");
        m_logger->warn("[Worker_manager] ⬆  DEGRADED MODE (terminal) ENTERED");
    } else if (session_active) {
        m_logger->warn("[Worker_manager] Mining paused - waiting for valid template");
        m_logger->warn("[Worker_manager] Workers stopped and cleared — will be restarted on recovery");
        m_logger->warn("[Worker_manager] ⬆  RECOVERY PAUSE — GET_BLOCK retry in progress (0xD081), self-healing");
    } else {
        m_logger->warn("[Worker_manager] Mining paused - session unavailable");
        m_logger->warn("[Worker_manager] Workers stopped and cleared — waiting for full re-auth or configured failover");
        m_logger->warn("[Worker_manager] ⬆  RECOVERY PAUSE — no work requests until a fresh session is established");
    }
    m_logger->warn("[Worker_manager] degraded_enter_total={}", m_degraded_enter_total);
}


void Worker_manager::retry_template_request(protocol::GetBlockReason reason)
{
    // Wrapper maintains the historical void signature for the many existing
    // callers that don't care whether the GET_BLOCK was transmitted; new
    // chokepoint callers (the Solo recovery handler) use the
    // retry_template_request_returning() variant directly so they can decide
    // whether to issue a local fallback.
    (void) retry_template_request_returning(reason);
}

bool Worker_manager::retry_template_request_returning(protocol::GetBlockReason reason)
{
    auto now = std::chrono::steady_clock::now();

    // Bug 5 fix: Prevent burst duplicate GET_BLOCK requests when both the forced
    // retry timer (100-250ms jitter) and health monitor (5s cycle) fire within
    // the same short window.  Suppress if the last request was sent < 500ms ago,
    // unless the reason bypasses all dedup (recovery-critical).
    constexpr int64_t GET_BLOCK_BURST_GUARD_MS = 500;
    if (!should_bypass_all_dedup(reason) &&
        m_last_get_block_request_time != std::chrono::steady_clock::time_point{}) {
        auto since_last_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - m_last_get_block_request_time).count();
        if (since_last_ms < GET_BLOCK_BURST_GUARD_MS) {
            m_logger->debug("[Worker_manager] GET_BLOCK burst-suppressed: {}ms since last request "
                           "(guard={}ms, reason={})", since_last_ms, GET_BLOCK_BURST_GUARD_MS, reason_name(reason));
            return false;
        }
    }

    m_logger->info("[Worker_manager] Requesting fresh work/GET_BLOCK... (reason={})", reason_name(reason));

    // Domain-isolation contract:
    //   - retry_template_request(): fresh work/GET_BLOCK request semantics only
    //   - no reconnect/session-state mutation here
    // Reconnect/recovery transitions are handled by explicit orchestration helpers/callers.
    if (!m_primary_node_session || !m_primary_node_session->is_authenticated()) {
        m_logger->debug("[Worker_manager] GET_BLOCK suppressed: context=missing_authenticated_session");
        m_logger->info("[Worker_manager] GET_BLOCK deferred — no authenticated session available");
        return false;
    }

    auto solo_protocol = m_primary_node_session->get_active_protocol();
    if (!solo_protocol) {
        m_logger->debug("[Worker_manager] GET_BLOCK suppressed: context=no_protocol");
        m_logger->error("[Worker_manager] Failed to get protocol from NodeSession");
        return false;
    }

    // ── GET_BLOCK exponential backoff (hashPrevBlock mismatch storm guard) ───
    // If validate_current_template() has discarded the last several templates due to
    // hashPrevBlock mismatch (consecutive > 0), a rapid-fire GET_BLOCK ↔ discard loop
    // is forming.  Apply an exponential delay before the next request so we don't
    // hammer a node that is already under load.  Forced recovery calls (should_bypass_all_dedup)
    // coming from the health-monitor still respect the backoff; the health-monitor's
    // own tick period (~5 s) already provides a natural floor delay.
    {
        uint32_t consecutive = solo_protocol->get_hashprev_mismatch_consecutive();
        if (consecutive > 0) {
            // Compute the target backoff for this mismatch depth (doubles each step, 30 s cap).
            // consecutive==1 → 2 s, consecutive==2 → 4 s, … consecutive>=4 → 30 s
            // Cap the iteration count to avoid a pathologically large loop; the backoff
            // plateaus at GET_BLOCK_BACKOFF_MAX_MS well before the cap is reached.
            constexpr uint32_t MAX_BACKOFF_ITERATIONS = 16;
            uint32_t iterations = std::min(consecutive - 1, MAX_BACKOFF_ITERATIONS);
            int64_t target_backoff_ms = GET_BLOCK_BACKOFF_INITIAL_MS;
            for (uint32_t i = 0; i < iterations; ++i) {
                target_backoff_ms = std::min(target_backoff_ms * 2, GET_BLOCK_BACKOFF_MAX_MS);
                if (target_backoff_ms >= GET_BLOCK_BACKOFF_MAX_MS) break;
            }

            // Extend the deadline if the new target is longer than the current one.
            auto desired_until = now + std::chrono::milliseconds(target_backoff_ms);
            if (desired_until > m_get_block_backoff_until) {
                m_get_block_backoff_ms    = target_backoff_ms;
                m_get_block_backoff_until = desired_until;
                m_logger->warn("[Worker_manager] ⏳ GET_BLOCK backoff set to {}ms "
                               "({} consecutive hashPrevBlock mismatches) — deferring request",
                               target_backoff_ms, consecutive);
            }
        } else {
            // No consecutive mismatches — clear any residual backoff (template was
            // recently adopted and the mismatch chain is broken).  The HEALTHY phase
            // transition handler also resets the backoff as a second safety net for
            // the case where the consecutive count is queried after adoption.
            m_get_block_backoff_ms    = 0;
            m_get_block_backoff_until = {};
        }

        // Suppress this GET_BLOCK if we are still within the backoff window.
        // Forced live-session refresh requests bypass this guard so stalled/waiting
        // recovery can always attempt a fresh BLOCK_DATA pull immediately.
        if (!should_bypass_all_dedup(reason) &&
            m_get_block_backoff_until != std::chrono::steady_clock::time_point{} &&
            now < m_get_block_backoff_until)
        {
            auto remaining_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                m_get_block_backoff_until - now).count();
            m_logger->info("[Worker_manager] GET_BLOCK suppressed: context=hashprev_mismatch_backoff "
                           "({}ms remaining)", remaining_ms);
            return false;
        } else if (should_bypass_all_dedup(reason) &&
                   m_get_block_backoff_until != std::chrono::steady_clock::time_point{} &&
                   now < m_get_block_backoff_until) {
            auto remaining_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                m_get_block_backoff_until - now).count();
            m_logger->warn("[Worker_manager] GET_BLOCK hashprev backoff bypassed by forced live-session request "
                           "({}ms remaining)", remaining_ms);
        }
    }

    // Early-exit if not yet authenticated — the channel-advanced staleness detector
    // can fire during the brief window after a push increments channel_height but
    // before MINER_AUTH_RESULT has set m_authenticated.  This is a normal transient
    // startup/reconnect condition; the health monitor will retry at the next tick.
    if (!solo_protocol->is_authenticated()) {
        m_logger->debug("[Worker_manager] GET_BLOCK suppressed: context=solo_not_authenticated");
        m_logger->info("[Worker_manager] GET_BLOCK deferred — not yet authenticated (auth in progress); "
                       "health monitor will retry when session is established");
        return false;
    }

    // Early-exit if primary TCP connection is down — request_work() would silently
    // return nullptr in this case (NodeSession checks m_primary_connected before
    // calling get_work).  Detect it here so we can log the real reason explicitly.
    if (!m_primary_node_session->is_primary_connected()) {
        m_logger->debug("[Worker_manager] GET_BLOCK suppressed: context=primary_disconnected");
        m_logger->warn("[Worker_manager] GET_BLOCK deferred — primary TCP connection is not established");
        return false;
    }

    bool no_valid_template = !has_valid_template_available(solo_protocol);
    bool is_forced = should_bypass_height_dedup(reason) && (is_recovery_active() || no_valid_template) && solo_protocol->is_authenticated();

    // Gate: SessionManager::can_request_get_block() must be true before transmitting GET_BLOCK.
    // This guard is bypassed when the reason bypasses height dedup (e.g. recovery, validation failure)
    // so the recovery ladder can always make progress even when push notifications are absent
    // (no PUSH subscription, post-reorg push silence, or initial connection before first push).
    if (!solo_protocol->can_request_get_block()) {
        m_logger->debug("[Worker_manager] GET_BLOCK suppressed: context=session_not_ready_for_get_block");
        m_logger->info("[Worker_manager] GET_BLOCK deferred — session not ready for GET_BLOCK; "
                       "health monitor will retry at next tick");
        return false;
    }

    // Request fresh work via NodeSession (wire-level GET_BLOCK)
    m_logger->info("[Worker_manager] Requesting fresh work via NodeSession (reason={}, forced={})",
                   reason_name(reason), is_forced ? "true" : "false");
    auto work_payload = m_primary_node_session->request_work(reason);
    if (work_payload && !work_payload->empty()) {
        if (m_primary_node_session->transmit(work_payload)) {
            // Mark the GET_BLOCK as in-flight AFTER confirmed transmission so that
            // PendingGetBlock.active is never set when the packet wasn't actually sent
            // (e.g. socket closed between request_work() and transmit()).
            solo_protocol->mark_get_block_pending(reason);
            m_recovery.last_get_block_at = std::chrono::steady_clock::now();
            m_recovery.get_block_confirmed = true;
            m_last_get_block_request_time = std::chrono::steady_clock::now();  // Bug 5: burst guard timestamp
            ++m_get_block_sent_total;
            if (is_forced) {
                ++m_get_block_forced_retry_total;
            }
            m_logger->info("[Worker_manager] → GET_BLOCK sent (recovery epoch {})", m_epoch_coordinator->recovery_epoch());
            return true;
        } else {
            m_logger->warn("[Worker_manager] GET_BLOCK payload built but not queued on transport");
            return false;
        }
    } else {
        auto last_status = solo_protocol->get_last_get_block_request_status();
        const char* status_str = "request_work_empty";
        switch (last_status) {
            case protocol::Solo::GetBlockRequestStatus::NODE_RETRY_DELAY:
                m_logger->debug("[Worker_manager] GET_BLOCK deferred until node retry deadline");
                return false;
            case protocol::Solo::GetBlockRequestStatus::DUPLICATE_WINDOW:
                status_str = "duplicate_window";
                break;
            case protocol::Solo::GetBlockRequestStatus::UNAUTHENTICATED:
            case protocol::Solo::GetBlockRequestStatus::SESSION_INVALID:
                status_str = "unauthenticated";
                break;
            default:
                break;
        }
        m_logger->debug("[Worker_manager] GET_BLOCK suppressed: context={}", status_str);
        // request_work() returned empty despite passing all guards above.
        // Most likely cause: 100ms GET_BLOCK dedup guard in Solo::get_work() or
        // transient reward-binding gap.  The recovery timer will retry at the next tick.
        m_logger->warn("[Worker_manager]   GET_BLOCK not sent — request_work() returned empty "
                       "(authenticated={}, primary_connected={}, see Solo logs for specific suppression reason)",
                       solo_protocol->is_authenticated(),
                       m_primary_node_session->is_primary_connected());
        if (solo_protocol->is_authenticated() && no_valid_template) {
            // Epoch 0 fix: when HEALTHY (recovery_epoch == 0) but no valid template
            // and GET_BLOCK was just suppressed by the dedup guard, the miner is stuck:
            // the health monitor retries with non-bypassing reasons (HEALTH_CHANNEL_ADVANCE)
            // which the dedup guard blocks at the same height,
            // and no forced retry is scheduled because is_recovery_active() is false.
            // Initiate recovery so the epoch advances (0 → 1), enabling
            // RECOVERY_FORCED/RECOVERY_TIMER which bypass ALL dedup guards.
            if (!is_recovery_active()) {
                mark_recovery_initiated("get_block_suppressed_no_template");
            }
            schedule_forced_recovery_retry("request_work_empty");
        }
        return false;
    }
}

void Worker_manager::check_template_health()
{
    if (m_terminal_stop_requested.load(std::memory_order_acquire)) {
        return;
    }
    if (!m_primary_node_session) {
        return;
    }

    // get_active_protocol() is null whenever the primary socket is down, including
    // the entire RECONNECTING window. Stall detection and transport-loss recovery
    // must run before that lookup or a dropped connection is invisible here.
    if (is_reconnecting()) {
        if (m_recovery.reconnect_started_at == std::chrono::steady_clock::time_point{}) {
            m_logger->warn("[Worker_manager] RECONNECTING but reconnect_started_at unset — "
                           "logic error detected; stamping now to allow timeout guard");
            m_recovery.reconnect_started_at = std::chrono::steady_clock::now();
        }
        auto reconnect_age_s = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - m_recovery.reconnect_started_at).count();
        // Extended from 60s to 90s to avoid racing exponential backoff retries
        // that may have a scheduled attempt at 60-65s.
        constexpr int64_t MAX_RECONNECT_WAIT_SECONDS = 90;
        if (reconnect_age_s <= MAX_RECONNECT_WAIT_SECONDS) {
            // A replacement socket can die after the connect callback returns,
            // while this phase is still RECONNECTING and no retry is armed.
            // Do not wait out the stall guard for that loss.
            if (!m_primary_node_session->is_primary_connected() &&
                !transport_recovery_outstanding()) {
                m_logger->warn("[Worker_manager] Primary socket down during RECONNECTING with no "
                               "retry scheduled — forcing TCP reconnect");
                handle_transport_lost("health_primary_disconnected");
            }
            return;
        }
        if (!m_primary_node_session->is_primary_connected() ||
            m_primary_node_session->has_pending_connect()) {
            m_logger->warn("[Worker_manager] Reconnect stalled for {}s > {}s — "
                           "resetting the timed-out transport and scheduling a fresh TCP reconnect",
                           reconnect_age_s, MAX_RECONNECT_WAIT_SECONDS);
            stop_all_workers();
            m_pending_recovery_completion_kind = RecoveryCompletionKind::TRANSPORT_RECONNECT;
            retry_connect(m_primary_endpoint, true);
            return;
        }
        // The socket came back. Leave RECONNECTING so the template ladder below
        // can request work instead of tearing down a live session.
        m_logger->warn("[Worker_manager] Reconnect stalled for {}s > {}s — clearing RECONNECTING phase",
                       reconnect_age_s, MAX_RECONNECT_WAIT_SECONDS);
        transition_to(RecoveryPhase::WAITING_TEMPLATE, "reconnect_timeout");
    }

    if (!m_primary_node_session->is_primary_connected()) {
        handle_transport_lost("health_primary_disconnected");
        return;
    }

    auto solo_protocol = m_primary_node_session->get_active_protocol();
    if (!solo_protocol) {
        return;
    }

    // ── Session health summary (every 60s) ──────────────────────────────────
    {
        auto now = std::chrono::steady_clock::now();
        constexpr int64_t HEALTH_LOG_INTERVAL_SECONDS = 60;
        auto since_last_log = std::chrono::duration_cast<std::chrono::seconds>(
            now - m_last_session_health_log).count();
        if (since_last_log >= HEALTH_LOG_INTERVAL_SECONDS) {
            m_last_session_health_log = now;
            auto ht = solo_protocol->get_height_tracker_snapshot();
            bool push_received = (ht.last_push_notification_at != std::chrono::steady_clock::time_point{});
            int64_t push_age_s = push_received
                ? std::chrono::duration_cast<std::chrono::seconds>(now - ht.last_push_notification_at).count()
                : -1;
            int64_t lane_alive_age_s = std::chrono::duration_cast<std::chrono::seconds>(
                now - m_sim_link.last_alive_at()).count();
            m_logger->info("[Session Health] authenticated={} session=0x{:08x}"
                           " push={}s_ago unanswered_rounds={}"
                           " preflight_drops={} phase={}"
                           " lane_alive={}s_ago"
                           " get_blocks_sent={} accepted={} rejected={}",
                           solo_protocol->is_authenticated(),
                           solo_protocol->get_session_id().get(),
                           push_age_s,
                           solo_protocol->get_unanswered_get_round_count(),
                           solo_protocol->get_preflight_reject_count(),
                           phase_name(m_recovery.phase.load(std::memory_order_relaxed)),
                           lane_alive_age_s,
                           m_get_block_sent_total,
                           solo_protocol->get_blocks_accepted(),
                           solo_protocol->get_blocks_rejected());
        }
    }

    // A node-requested wait is not a template outage. A down socket is handled
    // before this function looks up the active protocol.
    const bool node_retry_deferred = !is_reconnecting() &&
        m_primary_node_session->is_primary_connected() &&
        solo_protocol->is_authenticated() &&
        (solo_protocol->is_get_block_retry_deferred() || solo_protocol->is_node_template_wait_active());

    if (is_recovery_active()) {
        const auto now = std::chrono::steady_clock::now();
        const bool session_usable = !is_reconnecting() &&
            m_primary_node_session->is_primary_connected() &&
            solo_protocol->is_authenticated();
        m_recovery.timeout.observe_node_retry(node_retry_deferred, session_usable, now);
    }

    // WAITING_TEMPLATE: check timing and connection health, retry GET_BLOCK
    if (is_recovery_active() && !is_reconnecting() && !node_retry_deferred) {
        auto now = std::chrono::steady_clock::now();
        if (m_recovery.degraded_since == std::chrono::steady_clock::time_point{}) {
            m_recovery.degraded_since = now;
        }
        m_recovery.timeout.start_if_unset(m_recovery.degraded_since);
        auto degraded_secs = m_recovery.timeout.elapsed_seconds(now);

        auto ht_snap = solo_protocol->get_height_tracker_snapshot();
        bool push_received = (ht_snap.last_push_notification_at != std::chrono::steady_clock::time_point{});
        int64_t since_push_s = push_received
            ? std::chrono::duration_cast<std::chrono::seconds>(now - ht_snap.last_push_notification_at).count()
            : INT64_MAX;
        bool push_recent = push_received && (since_push_s <= PUSH_ALIVE_THRESHOLD_SECONDS);

        // Worst-case outage aggregator (Option B): take the MAX of template-outage
        // duration, push silence, and lane-alive silence so a connectivity blip that
        // started before (or outlasts) the current WAITING_TEMPLATE epoch cannot be
        // masked by a template-only clock that looks "recent enough" in isolation.
        // When no push has ever been received, since_push_s is INT64_MAX (sentinel for
        // logging/resubscribe). Do not feed that sentinel into the aggregator — it would
        // make worst_case_secs trip the hard-stop immediately on first WAITING_TEMPLATE.
        // Pass 0 so the hard-stop is driven only by real elapsed outage signals.
        int64_t lane_alive_age_s = std::chrono::duration_cast<std::chrono::seconds>(
            now - m_sim_link.last_alive_at()).count();
        int64_t worst_case_secs = m_recovery.timeout.worst_case_outage_seconds(
            now, push_received ? since_push_s : 0, std::max<int64_t>(lane_alive_age_s, 0));

        m_logger->info("[Worker_manager] WAITING_TEMPLATE: {}s elapsed, push {}s ago (recent={}), "
                       "worst_case_outage={}s",
                       degraded_secs,
                       push_received ? since_push_s : static_cast<int64_t>(-1),
                       push_recent ? "YES" : "NO",
                       worst_case_secs);

        const auto recovery_window_seconds =
            (solo_protocol->get_template_interface() &&
             solo_protocol->get_template_interface()->get_channel() == mining::CHANNEL_PRIME)
                ? RECOVERY_WINDOW_SECONDS_PRIME
                : RECOVERY_WINDOW_SECONDS_HASH;

        // After the channel-aware recovery window without template, check connection health.
        // IMPORTANT: push silence is orchestration-layer only — it may trigger reconnect,
        // but must not suppress authoritative GET_BLOCK refresh attempts.
        // Intentional ordering: when this branch fires we do BOTH operations in the same
        // tick (reconnect attempt + forced GET_BLOCK retry path) so orchestration signals
        // can feed recovery without replacing authoritative refresh/submission decisions.
        if (degraded_secs > recovery_window_seconds && !push_recent) {
            m_logger->error("[Worker_manager] ⛔ {}s timeout: no template and push is dead — reconnecting",
                            recovery_window_seconds);
            retry_connect(m_primary_endpoint);
            // Fall through intentionally: still run forced GET_BLOCK retry below.
        }

        // Retry GET_BLOCK on every health check tick (forced: bypass local gates)
        retry_template_request(protocol::GetBlockReason::RECOVERY_FORCED);

        /* ── Reorg Resubscription Guard ──────────────────────────────────────────
         * After the 300s WAITING_TEMPLATE window with a live authenticated session but no
         * incoming push, proactively re-send MINER_READY.  This covers the reorg
         * case: the node recovered the session but the push subscription state was
         * lost during the TCP disconnect, so the node's heartbeat cycle (480s) is
         * the only thing that would otherwise restore it.
         * Cooldown: once per 60s to prevent rapid-fire. */
        constexpr int64_t REORG_RESUBSCRIBE_THRESHOLD_SECONDS = 60;
        constexpr int64_t REORG_RESUBSCRIBE_COOLDOWN_SECONDS  = 60;

        bool push_silent_in_recovery = !push_recent &&
            (degraded_secs > protocol::ProtocolConstants::DEGRADED_MODE_STAGE2_SECONDS);

        if (push_silent_in_recovery && solo_protocol->is_authenticated())
        {
            auto since_last_resub_s =
                (m_last_resubscribe_at == std::chrono::steady_clock::time_point{})
                ? INT64_MAX
                : std::chrono::duration_cast<std::chrono::seconds>(
                      now - m_last_resubscribe_at).count();

            if (since_last_resub_s >= REORG_RESUBSCRIBE_COOLDOWN_SECONDS &&
                since_push_s >= REORG_RESUBSCRIBE_THRESHOLD_SECONDS)
            {
                int64_t display_push_silence_s = (since_push_s == INT64_MAX) ? -1 : since_push_s;
                m_logger->warn("[Worker_manager] ⚡ REORG RESUBSCRIBE GUARD: {}s in recovery, "
                               "push silent {}s, sending MINER_READY to restore subscription",
                               degraded_secs, display_push_silence_s);
                solo_protocol->resubscribe_push_notifications();
                m_last_resubscribe_at = now;
            }
        }

        return;
    }

    auto* template_interface = solo_protocol->get_template_interface();
    if (!template_interface) {
        return;
    }

    uint8_t channel = template_interface->get_channel();
    std::string channel_name = (channel == mining::CHANNEL_PRIME) ? "Prime" : "Hash";
    const bool has_valid_template = template_interface->has_valid_template();

    // Replacement-pending safety net: a prior PUSH (e.g. same-height tip update)
    // marked the current template as about-to-be-superseded so workers could
    // keep mining through the BLOCK_DATA round-trip.  If the deadline has
    // elapsed without read_template() consuming it, fall back to the normal
    // discard + HEALTH_NO_TEMPLATE recovery path so the miner does not sit
    // indefinitely on a template that the node has implicitly invalidated.
    {
        std::string expired_reason;
        if (template_interface->take_expired_replacement_pending(expired_reason)) {
            m_logger->warn("[Worker_manager] Replacement-pending timeout ({}) — "
                           "promised BLOCK_DATA never arrived; falling back to "
                           "HEALTH_NO_TEMPLATE recovery",
                           expired_reason.empty() ? "unspecified" : expired_reason);
            template_interface->discard_template("replacement_timeout:" +
                (expired_reason.empty() ? std::string("unspecified") : expired_reason));
            retry_template_request(protocol::GetBlockReason::HEALTH_NO_TEMPLATE);
            return;
        }
    }

    if (node_retry_deferred && !has_valid_template) {
        m_logger->debug("[Worker_manager] Template health escalation deferred by node retry deadline");
        return;
    }

    if (!has_valid_template) {
        // No template in HEALTHY state — force fresh work/GET_BLOCK so local gates
        // do not stall BLOCK_DATA delivery while the primary session is alive.
        retry_template_request(protocol::GetBlockReason::HEALTH_NO_TEMPLATE);
        return;
    }

    // Get HeightTracker snapshot for staleness and age checks (single source of truth)
    auto ht_snap = solo_protocol->get_height_tracker_snapshot();
    uint64_t template_age = ht_snap.get_template_age_seconds();

    // Channel height-based staleness detection removed: is_template_stale() was
    // structurally false under canonical-only semantics (OnBlockDataReceived auto-
    // advances target = channel + 1, so channel_height >= channel_target never holds).
    // Staleness detection lives in PUSH, GET_ROUND, and Health Monitor trigger paths.

    // Unified tip moved on another channel — log it but do NOT send GET_BLOCK.
    // GET_ROUND is the backup to PUSH for tip changes, and GET_ROUND triggers
    // GET_BLOCK on tip changes.  Health monitor should not send GET_BLOCK directly.
    if (ht_snap.is_tip_moved()) {
        m_logger->info("[Worker_manager] Unified tip moved (template_unified_height {} → unified_height {}) — "
                       "node will auto-send via PUSH; GET_ROUND backup active; workers continue on valid channel template",
                       ht_snap.template_unified_height, ht_snap.unified_height);
        return;
    }

    // Unified height drift detection
    {
        auto ht_snap2 = solo_protocol->get_height_tracker_snapshot();
        uint32_t tmpl_height = template_interface->get_template_height();

        if (ht_snap2.unified_height > 0 && tmpl_height > 0 &&
            ht_snap2.unified_height > tmpl_height + UNIFIED_DRIFT_THRESHOLD)
        {
            int32_t drift = static_cast<int32_t>(ht_snap2.unified_height) -
                            static_cast<int32_t>(tmpl_height);
            m_logger->warn("[Worker_manager] ⚠️  HEIGHT_DRIFT: unified={} vs template.nHeight={} (drift={}) — template on stale tip",
                ht_snap2.unified_height, tmpl_height, drift);
            template_interface->discard_template("Unified height drift: " +
                std::to_string(drift) + " blocks behind");
            mark_recovery_initiated("height_drift");
            retry_template_request(protocol::GetBlockReason::HEIGHT_DRIFT);
            return;
        }
    }

    // ── Fork / tip mismatch detection (DIAGNOSTIC ONLY) ────────────────────────
    // Fork score and hash_tip_lo32 come from keepalive ACKs (DiagnosticObserverState).
    // They are INFORMATIONAL and must NOT be used to trigger hard stops, as keepalive
    // data can lag chain advancement by 45s and cause false-positive fork detections
    // during normal block-finding events.
    //
    // Real fork detection uses canonical_hash_prev_block vs push hashPrevBlock
    // (handled by the TEMPLATE_ANCHOR in solo.cpp, not here).
    {
        auto diag = solo_protocol->get_diagnostic_snapshot();
        if (diag.keepalive_peak_fork_score > 0) {
            m_logger->warn("[Worker_manager] [Colin] FORK CANARY: peak_fork_score={} fork_score={} hash_tip_lo32=0x{:08x} (diagnostic only — not stopping workers)",
                diag.keepalive_peak_fork_score, diag.keepalive_fork_score, diag.keepalive_hash_tip_lo32);
        }
        // Do NOT stop workers based on keepalive fork_score alone.
    }

    // BLOCK_DATA timeout warning: 480s gives operators a 120s (2-minute) window
    // before the 600s BLOCK_DATA timeout trigger fires.
    // Both channels use the same threshold — in the push-driven protocol the node pushes
    // on every unified tip advance (~18s apart via hash blocks), so 480s without a push
    // is unusual for either channel.
    // Proactively request fresh work at the warning threshold so the age clock resets
    // before the 600s emergency fires.  Workers continue mining on the same height in the
    // meantime; only the timestamp is refreshed.  TEMPLATE_AGE_WARNING does not bypass
    // so it respects the can_request_get_block() session gate and hashPrevBlock mismatch backoff.
    //
    // Cooldown guard: fire at most once per TEMPLATE_AGE_COOLDOWN_SECONDS to prevent
    // the 5-second health tick from exhausting the node's 25/60s GET_BLOCK rate limit.
    if (template_age > protocol::ProtocolConstants::TEMPLATE_AGE_WARNING_SECONDS &&
        template_age <= protocol::ProtocolConstants::TEMPLATE_AGE_EMERGENCY_TIMEOUT_SECONDS) {
        auto now = std::chrono::steady_clock::now();
        auto since_last = std::chrono::duration_cast<std::chrono::seconds>(
            now - m_last_template_age_request_at).count();
        if (m_last_template_age_request_at == std::chrono::steady_clock::time_point{} ||
            since_last >= TEMPLATE_AGE_COOLDOWN_SECONDS)
        {
            m_logger->warn("[Worker_manager] ⚠️  {} template age {}s (warning threshold {}s, emergency {}s)",
                channel_name, template_age,
                protocol::ProtocolConstants::TEMPLATE_AGE_WARNING_SECONDS,
                protocol::ProtocolConstants::TEMPLATE_AGE_EMERGENCY_TIMEOUT_SECONDS);
            m_logger->warn("[Worker_manager]    No push received for {}s — requesting proactive fresh work/GET_BLOCK", template_age);
            retry_template_request(protocol::GetBlockReason::TEMPLATE_AGE_WARNING);
            m_last_template_age_request_at = now;
        } else {
            m_logger->debug("[Worker_manager] Template age {}s > warning threshold but cooldown active ({}s since last request, cooldown {}s)",
                template_age, since_last, TEMPLATE_AGE_COOLDOWN_SECONDS);
        }
    }

    // BLOCK_DATA timeout trigger (600s) — dead-connection detector for both channels.
    //
    // In the push-driven era the node pushes fresh BLOCK_DATA within ~2s of every unified
    // tip advance.  Even during long Prime blocks, hash blocks keep advancing the unified
    // chain every ~18s, so a push should arrive well within 600s.
    //
    // If template_age > 600s, treat this as an orchestration-layer recovery signal.
    // It may escalate refresh/reconnect behavior, but MUST NOT invalidate a valid
    // template or halt mining by itself.
    if (template_age > protocol::ProtocolConstants::TEMPLATE_AGE_EMERGENCY_TIMEOUT_SECONDS) {
        constexpr const char* BLOCK_DATA_TIMEOUT_TRIGGER_REASON = "block_data_timeout";

        // Bug D fix: During an active recovery epoch, the miner is already waiting for a
        // fresh GET_BLOCK response.  Discarding the template now makes recovery self-defeating:
        // we'd have no template AND no recovery in progress simultaneously (doom-loop).
        // Suppress the hard emergency discard while recovery is pending; the recovery window
        // escalation logic (above) handles escalation if the window expires without a template.
        if (is_recovery_active()) {
            m_logger->debug("[Worker_manager] BLOCK_DATA timeout trigger ({}s) suppressed during active recovery (epoch {})",
                            template_age, m_epoch_coordinator->recovery_epoch());
            // Resend GET_BLOCK (non-force: respects RECOVERY_RESEND_INTERVAL_SECONDS rate-limit)
            // so recovery keeps making progress without escalating to hard stop/restart.
            retry_template_request(protocol::GetBlockReason::TEMPLATE_AGE_DEFERRED);
            return;
        }

        auto ht_snap = solo_protocol->get_height_tracker_snapshot();
        // is_template_stale() removed (structurally false under canonical-only semantics).
        // Use recent_push as the sole signal for deferring recovery.

        // If a push notification was received recently, the TCP session is
        // demonstrably alive — defer the hard recovery to avoid spurious stops
        // during slow-block scenarios (e.g. long Prime blocks).
        // PUSH is the sole authoritative signal; keepalive ACK is not used here.
        bool recent_push = (ht_snap.last_push_notification_at != std::chrono::steady_clock::time_point{}) &&
            (std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - ht_snap.last_push_notification_at).count() < PUSH_ALIVE_THRESHOLD_SECONDS);
        if (recent_push) {
            auto since_push_s = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - ht_snap.last_push_notification_at).count();
            m_logger->warn("[Worker_manager] BLOCK_DATA timeout deferred: push notification is recent ({}s ago) — "
                           "connection alive, awaiting fresh BLOCK_DATA",
                           since_push_s);
            retry_template_request(protocol::GetBlockReason::TEMPLATE_AGE_DEFERRED);
            return;
        }

        m_logger->error("[Worker_manager] ⚠️  BLOCK_DATA TIMEOUT TRIGGER ({} channel): template {}s old with no recent push — escalating recovery",
                        channel_name, template_age);
        m_logger->error("[Worker_manager]    channel_height {} / channel_target {}",
                        ht_snap.channel_height, ht_snap.channel_target);
        if (channel == mining::CHANNEL_PRIME) {
            m_logger->error("[Worker_manager]    Prime blocks may be long; treating this as recovery signal only");
        }

        if (node_retry_deferred) {
            m_logger->debug("[Worker_manager] BLOCK_DATA timeout recovery deferred by node retry deadline");
            return;
        }

        // Cooldown guard: do not escalate recovery on every 5-second health tick.
        {
            auto now_em = std::chrono::steady_clock::now();
            auto since_last_em = std::chrono::duration_cast<std::chrono::seconds>(
                now_em - m_last_template_age_request_at).count();
            if (m_last_template_age_request_at != std::chrono::steady_clock::time_point{} &&
                since_last_em < TEMPLATE_AGE_COOLDOWN_SECONDS)
            {
                m_logger->debug("[Worker_manager] Emergency recovery cooldown active ({}s since last, cooldown {}s)",
                    since_last_em, TEMPLATE_AGE_COOLDOWN_SECONDS);
                return;
            }
            m_last_template_age_request_at = now_em;
        }

        // Non-authoritative path: do NOT discard valid template and do NOT halt workers here.
        // Escalate orchestration only (reconnect + forced GET_BLOCK lane).
        mark_recovery_initiated(BLOCK_DATA_TIMEOUT_TRIGGER_REASON);
        retry_connect(m_primary_endpoint);
        retry_template_request(protocol::GetBlockReason::TEMPLATE_AGE_EMERGENCY);
    }
}

void Worker_manager::log_mined_block_cache() const
{
    auto summary = m_mined_block_cache.format_cache_summary();
    // Log each line individually for proper formatting
    std::istringstream iss(summary);
    std::string line;
    while (std::getline(iss, line)) {
        if (!line.empty())
            m_logger->info("[MinedBlockCache] {}", line);
    }
}

}
