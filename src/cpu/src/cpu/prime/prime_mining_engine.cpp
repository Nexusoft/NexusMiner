#include "cpu/prime/prime_mining_engine.hpp"

#include "cpu/prime/chain_sieve.hpp"
#include "cpu/prime_validation.hpp"
#include "worker/template_feed.hpp"

#include <asio.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>

namespace nexusminer {
namespace cpu {

PrimeMiningEngine::PrimeMiningEngine(Engine_config cfg,
                                     std::shared_ptr<WorkerTemplateFeed> feed)
    : m_cfg{cfg}
    , m_feed{std::move(feed)}
    , m_logger{spdlog::get("logger")}
    , m_segment_allocator{std::make_unique<Shared_segment_allocator>(cfg.segment_size)}
{
    if (!m_feed)
    {
        // A null feed is a programmer error: the engine has no other source
        // of templates and would block forever.  Fail loudly rather than
        // silently spawning an idle consumer thread.
        throw std::invalid_argument{"PrimeMiningEngine requires a non-null WorkerTemplateFeed"};
    }
    if (m_cfg.segment_size == 0)
    {
        throw std::invalid_argument{"PrimeMiningEngine requires segment_size > 0"};
    }

    // Stone 6.5 — validate chunk size.  [1, 1024]: 1 reproduces broken per-
    // segment behaviour for the regression test, 1024 caps memory/latency.
    if (m_cfg.pool_chunk_segments < 1 || m_cfg.pool_chunk_segments > 1024)
    {
        throw std::invalid_argument{
            "PrimeMiningEngine requires pool_chunk_segments in [1, 1024]"};
    }
    m_pool_chunk_segments = m_cfg.pool_chunk_segments;

    // Stone 6 — cursor domain is RELATIVE.  The cooperative cursor always
    // starts at 0 on a new template; pool threads add `session->starting_nonce`
    // (== Engine_config::channel_starting_nonce) themselves at use time.
    // Seeding at 0 here is defensive: a pool thread that races to
    // next_segment_start() before any session arrives still gets a sensible
    // relative offset (the value will be re-reset on the first publish).
    m_segment_allocator->reset(0);

    // Resolve pool-thread count up front so it is observable via
    // pool_thread_count() before any pool thread has actually spawned.
    m_pool_thread_count = resolve_pool_thread_count(m_cfg.pool_threads, m_logger);

    // Session-only mode: with no io_context there is nowhere to dispatch
    // found blocks, so spawning pool threads would just sieve into a void.
    // Force the count to zero and warn — the consumer thread still runs and
    // builds sessions, which is exactly what tests / future stages that only
    // care about session creation need.
    if (!m_cfg.io_context && m_pool_thread_count > 0)
    {
        if (m_logger)
        {
            m_logger->warn("[PrimeMiningEngine] io_context is null; running in "
                           "session-only mode (pool_threads forced from {} to 0)",
                           m_pool_thread_count);
        }
        m_pool_thread_count = 0;
    }

    if (m_logger)
    {
        m_logger->info("[PrimeMiningEngine] starting consumer thread "
                       "(segment_size={}, channel_starting_nonce={}, "
                       "internal_id_for_solution={}, pool_threads={}, "
                       "pool_chunk_segments={})",
                       m_cfg.segment_size,
                       m_cfg.channel_starting_nonce,
                       m_cfg.internal_id_for_solution,
                       m_pool_thread_count,
                       m_pool_chunk_segments);
    }

    // Spawn the consumer first so all members are fully constructed before
    // it can observe them.
    m_consumer = std::thread{&PrimeMiningEngine::run_consumer, this};

    // Option 2 — per-pool-thread published histogram snapshots.  Each
    // Atomic_snapshot is value-initialised (default Pool_histogram_snapshot
    // has zeroed Prime_histogram arrays) so the stats path can read it
    // safely even before any pool thread has published.  Allocated BEFORE
    // pool threads are launched so the very first pool-thread iteration
    // can safely publish into its slot.
    if (m_pool_thread_count > 0)
    {
        m_pool_histogram_snapshots =
            std::make_unique<stats::Atomic_snapshot<Pool_histogram_snapshot>[]>(
                m_pool_thread_count);
    }

    // Spawn the pool threads AFTER the consumer is running.  Pool threads must
    // observe a fully-constructed engine; spawning them last guarantees this.
    m_pool_threads.reserve(m_pool_thread_count);
    for (std::uint32_t i = 0; i < m_pool_thread_count; ++i)
    {
        m_pool_threads.emplace_back(&PrimeMiningEngine::run_pool_thread, this, i);
    }

    // Stone 6.5 — periodic stats logger (low-frequency timer thread).  Only
    // useful when the engine actually drives pool sieving work — in session-
    // only mode (no io_context, no pool threads) the stats line would be all
    // zeros and just spam logs.
    if (m_pool_thread_count > 0)
    {
        m_stats_logger = std::thread{&PrimeMiningEngine::run_stats_logger, this};
    }
}

PrimeMiningEngine::~PrimeMiningEngine()
{
    m_shutdown.store(true, std::memory_order_release);

    // Wake pool threads parked on m_pool_cv waiting for a usable session.
    // This is required even when no template was ever published — the
    // destructor MUST be safe to call on an idle engine.
    {
        std::lock_guard<std::mutex> lock(m_pool_mtx);
    }
    m_pool_cv.notify_all();

    // Wake the consumer if it is parked in WorkerTemplateFeed::wait_for_epoch_after.
    // The feed may have been reset by Worker_manager already in pathological
    // teardown orderings — the contract is that the feed outlives the engine,
    // so this should never be null in practice, but the null-check keeps the
    // destructor safe under test fixtures that drop the feed first.
    if (m_feed)
    {
        m_feed->notify_wake();
    }

    // Wake any test waiter blocked in wait_for_sessions_published_after().
    {
        std::lock_guard<std::mutex> lock(m_publish_mtx);
    }
    m_publish_cv.notify_all();

    // Join pool threads BEFORE the consumer.  The consumer publishes sessions
    // that pool threads rely on; if it died first, pool threads might be
    // observing a torn-down engine.  Pool-threads-first keeps the dependency
    // graph clean.
    for (auto& th : m_pool_threads)
    {
        if (th.joinable())
        {
            th.join();
        }
    }

    if (m_consumer.joinable())
    {
        m_consumer.join();
    }

    // Stone 6.5 — join the periodic stats logger.  It parks on m_pool_cv with
    // a 30s timeout and exits as soon as it observes m_shutdown.
    if (m_stats_logger.joinable())
    {
        m_stats_logger.join();
    }

    if (m_logger)
    {
        m_logger->info("[PrimeMiningEngine] joined "
                       "(sessions_published={}, allocator_resets={}, "
                       "same_base_short_circuits={}, segments_processed={}, "
                       "segments_discarded_epoch_changed={}, "
                       "segments_skipped_consumed={}, candidates_dispatched={}, "
                       "chunks_drawn={}, starting_multiples_calls={}, "
                       "pool_threads_crashed={})",
                       m_sessions_published.load(std::memory_order_relaxed),
                       m_allocator_resets.load(std::memory_order_relaxed),
                       m_same_base_short_circuits.load(std::memory_order_relaxed),
                       m_segments_processed.load(std::memory_order_relaxed),
                       m_segments_discarded_epoch_changed.load(std::memory_order_relaxed),
                       m_segments_skipped_consumed.load(std::memory_order_relaxed),
                       m_candidates_dispatched.load(std::memory_order_relaxed),
                       m_chunks_drawn.load(std::memory_order_relaxed),
                       m_starting_multiples_calls.load(std::memory_order_relaxed),
                       m_pool_threads_crashed.load(std::memory_order_relaxed));
    }
}

void PrimeMiningEngine::register_worker(std::shared_ptr<Worker_prime> worker)
{
    if (!worker)
    {
        return;
    }
    std::lock_guard<std::mutex> lock(m_registered_mtx);
    m_registered.emplace_back(std::move(worker));
}

PrimeMiningEngine::Engine_stats_snapshot PrimeMiningEngine::snapshot_stats() const
{
    Engine_stats_snapshot snap{};
    snap.segments_processed   = m_segments_processed.load(std::memory_order_relaxed);
    snap.candidates_dispatched = m_candidates_dispatched.load(std::memory_order_relaxed);
    snap.chains_found_by_sieve = m_chains_found_by_sieve.load(std::memory_order_relaxed);
    snap.chains_pushed_long    = m_chains_pushed_long.load(std::memory_order_relaxed);
    snap.popcount_windows_passed =
        m_popcount_windows_passed.load(std::memory_order_relaxed);
    snap.validate_attempts     = m_validate_attempts.load(std::memory_order_relaxed);
    snap.validate_rejected_base_not_prime =
        m_validate_rejected_base_not_prime.load(std::memory_order_relaxed);
    snap.validate_rejected_below_diff =
        m_validate_rejected_below_diff.load(std::memory_order_relaxed);
    snap.validate_rejected_malformed =
        m_validate_rejected_malformed.load(std::memory_order_relaxed);
    snap.segment_size         = m_cfg.segment_size;
    snap.sessions_published   = m_sessions_published.load(std::memory_order_relaxed);
    snap.pool_thread_count    = m_pool_thread_count;
    snap.chunks_drawn         = m_chunks_drawn.load(std::memory_order_relaxed);
    snap.starting_multiples_calls =
        m_starting_multiples_calls.load(std::memory_order_relaxed);
    snap.segments_discarded_epoch_changed =
        m_segments_discarded_epoch_changed.load(std::memory_order_relaxed);
    snap.segments_skipped_consumed =
        m_segments_skipped_consumed.load(std::memory_order_relaxed);
    snap.same_base_short_circuits =
        m_same_base_short_circuits.load(std::memory_order_relaxed);
    snap.allocator_resets =
        m_allocator_resets.load(std::memory_order_relaxed);
    snap.pool_threads_running =
        m_pool_threads_running.load(std::memory_order_relaxed);
    snap.pool_threads_crashed =
        m_pool_threads_crashed.load(std::memory_order_relaxed);
    snap.pool_chunk_segments  = m_pool_chunk_segments;
    snap.best_difficulty      = m_best_difficulty.load(std::memory_order_relaxed);

    if (auto session = current_session())
    {
        snap.nbits = session->nbits;
    }

    // Option 2 — fan-in chain histograms across every pool thread's PUBLISHED
    // snapshot (Pool_histogram_snapshot), NOT the live Sieve*.  Pool threads
    // republish at chunk boundaries from the owning thread, so the stats
    // path observes only immutable, atomically-swapped arrays — restoring
    // the documented invariant in stats/prime_stats_snapshot.hpp:20-22 that
    // diagnostic counters are published from the owning thread and the
    // stats path never reads live (mutable) sieve state.
    //
    // Buckets are saturating-summed into the fixed-size stats::Prime_histogram
    // array (any over-length input buckets beyond the array's capacity are
    // dropped — they cannot occur today since the CPU Sieve sizes
    // m_chain_histogram to 12 == kPrimeHistogramBuckets after Option 5).
    if (m_pool_histogram_snapshots)
    {
        for (std::uint32_t i = 0; i < m_pool_thread_count; ++i)
        {
            const auto local = m_pool_histogram_snapshots[i].load();
            if (!local) continue;
            const std::size_t n = std::min(local->best.size(),
                                           snap.chain_histogram.size());
            for (std::size_t b = 0; b < n; ++b)
            {
                const std::uint64_t sum_best =
                    static_cast<std::uint64_t>(snap.chain_histogram[b])
                  + static_cast<std::uint64_t>(local->best[b]);
                snap.chain_histogram[b] = stats::saturating_prime_stat(sum_best);
                const std::uint64_t sum_attempted =
                    static_cast<std::uint64_t>(snap.chain_histogram_attempted[b])
                  + static_cast<std::uint64_t>(local->attempted[b]);
                snap.chain_histogram_attempted[b] =
                    stats::saturating_prime_stat(sum_attempted);
            }
        }
    }

    return snap;
}

std::size_t PrimeMiningEngine::registered_worker_count() const
{
    std::lock_guard<std::mutex> lock(m_registered_mtx);
    std::size_t live = 0;
    for (const auto& w : m_registered)
    {
        if (!w.expired()) ++live;
    }
    return live;
}

std::uint64_t PrimeMiningEngine::wait_for_sessions_published_after(
    std::uint64_t last_seen,
    std::chrono::milliseconds timeout)
{
    std::unique_lock<std::mutex> lock(m_publish_mtx);
    m_publish_cv.wait_for(lock, timeout, [&] {
        return m_sessions_published.load(std::memory_order_acquire) > last_seen
            || m_shutdown.load(std::memory_order_acquire);
    });
    return m_sessions_published.load(std::memory_order_acquire);
}

void PrimeMiningEngine::run_consumer()
{
    std::uint64_t last_seen = 0;

    while (true)
    {
        // Block until the feed advances or we are asked to shut down.
        last_seen = m_feed->wait_for_epoch_after(last_seen, [this] {
            return m_shutdown.load(std::memory_order_acquire);
        });

        if (m_shutdown.load(std::memory_order_acquire))
        {
            break;
        }

        // Load the latest published epoch.  wait_for_epoch_after only updates
        // last_seen to the latest visible epoch_id at the time it returns,
        // which is exactly the epoch we want to process.
        const auto epoch = m_feed->load();
        if (!epoch)
        {
            // A reset_for_new_batch() could have wiped the slot between the
            // wake-up and the load.  Treat as no-op and re-park.
            continue;
        }

        // Build the immutable session.  Block_data is constructed from the
        // shared CBlock; the prime base hash is taken from the precomputed
        // WorkPackage value if present (Worker_manager precomputes it for the
        // prime channel) or recomputed locally otherwise.
        const auto& work_package = epoch->work_package;
        if (!work_package)
        {
            // Empty publish (e.g. a heartbeat).  Nothing to mine.
            continue;
        }

        auto session = std::make_shared<EngineSession>();
        session->epoch_id = epoch->epoch_id;
        session->block_data = Block_data{work_package->get_block()};
        session->nbits = work_package->get_nbits();
        session->starting_nonce = m_cfg.channel_starting_nonce;
        session->internal_id_for_solution = m_cfg.internal_id_for_solution;
        session->on_found = epoch->on_found;
        const bool precomputed_hash_available = work_package->get_prime_base_hash().has_value();
        if (precomputed_hash_available)
        {
            session->base_hash = work_package->get_prime_base_hash().value();
        }
        else
        {
            // Compute on the consumer thread (rather than per pool thread) so
            // every pool thread sees the same base_hash without a race.
            session->base_hash = session->block_data.GetPrimeBaseHash();
        }

        // Same-proof-space short-circuit: the cooperative cursor is preserved
        // when the new template targets the same block height as the previous
        // session (same hashPrevBlock + same nHeight).
        //
        // In production LLL-TAO rotates hashMerkleRoot on every KEEPALIVE
        // (coinbase update), causing base_hash to change each publish even
        // though the chain tip and proof-hash space have not moved.  Using the
        // STABLE fields (prev_hash + height) instead of base_hash ensures the
        // short-circuit fires on those Merkle-rotation-only updates —
        // eliminating the allocator_resets thrash and the associated nonce-
        // space restart that otherwise discards all cooperative sieve progress.
        //
        // Pool threads still rebind their sieve whenever base_hash changes
        // (i.e. on every Merkle rotation) because the sieve's starting-
        // multiples depend on the exact base_hash.  Only the cooperative cursor
        // (nonce-space offset) is preserved here, not the sieve state.
        const auto previous = current_session();
        const bool same_base =
            previous
            && previous->block_data.previous_hash == session->block_data.previous_hash
            && previous->block_data.nHeight       == session->block_data.nHeight;

        if (m_logger)
        {
            // Emit the first 16 hex digits of base_hash so operators can
            // confirm in live logs whether base_hash is changing each publish.
            std::ostringstream oss;
            oss << std::hex << session->base_hash;
            const auto s = oss.str();
            const std::string hash_prefix = "0x" + s.substr(0, std::min(s.size(), std::size_t{16}));
            m_logger->debug("[PrimeMiningEngine] consumer: epoch={} base_hash={} "
                            "(precomputed={}) height={} same_base={}",
                            session->epoch_id,
                            hash_prefix,
                            precomputed_hash_available,
                            session->block_data.nHeight,
                            same_base);
        }

        if (!same_base)
        {
            // Stone 6 — RELATIVE cursor: always restart at 0 on a new
            // template.  Pool threads add `session->starting_nonce` themselves
            // when computing the absolute sieve start.
            m_segment_allocator->reset(0);
            m_allocator_resets.fetch_add(1, std::memory_order_relaxed);
        }
        else
        {
            m_same_base_short_circuits.fetch_add(1, std::memory_order_relaxed);
        }

        // Publish the new session.  Release-store pairs with the acquire-load
        // in current_session() so any pool thread observing the new session
        // also observes the cursor reset above (when same_base is false).
        std::atomic_store_explicit(&m_session,
                                   std::shared_ptr<const EngineSession>(session),
                                   std::memory_order_release);

        {
            std::lock_guard<std::mutex> lock(m_publish_mtx);
            m_sessions_published.fetch_add(1, std::memory_order_release);
        }
        m_publish_cv.notify_all();

        // Stone 6 — wake pool threads parked waiting for a usable session.
        // Every publish wakes them: the post-segment session re-check then
        // tells each thread whether to continue (epoch_id matches) or
        // discard-and-rebind (different epoch / different base).
        {
            std::lock_guard<std::mutex> lock(m_pool_mtx);
        }
        m_pool_cv.notify_all();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stone 6 — pool sieve thread implementation.
// ─────────────────────────────────────────────────────────────────────────────

namespace {

using uint1k = boost::multiprecision::uint1024_t;

// Local boost-uint1024 → LLC uint1024 conversion.  Worker_prime owns the
// canonical implementation as a private member (verified by
// prime_validation_test); duplicating the limb copy here keeps the engine
// from depending on Worker_prime internals while Stone 7 wiring is pending.
uint1024_t boost_uint1k_to_uint1024(const uint1k& p)
{
    constexpr std::size_t kLimbBytes = sizeof(boost::multiprecision::limb_type);
    uint1024_t result{};
    const auto limb_bytes = p.backend().size() * kLimbBytes;
    if (limb_bytes > sizeof(result))
    {
        throw std::runtime_error("Boost uint1024 limb storage exceeds LLC uint1024_t size");
    }
    std::memcpy(&result, p.backend().limbs(), limb_bytes);
    return result;
}

constexpr std::size_t kPrimeOffsetFractionBytes = nexusminer::prime::kPrimeOffsetFractionBytes;
constexpr std::size_t kMinSerializedPrimeOffsets = nexusminer::prime::kMinSerializedPrimeOffsets;
constexpr std::size_t kMaxSerializedPrimeOffsets = nexusminer::prime::kMaxSerializedPrimeOffsets;

}  // namespace

std::uint32_t PrimeMiningEngine::resolve_pool_thread_count(
    std::uint32_t configured,
    std::shared_ptr<spdlog::logger>& logger)
{
    const std::uint32_t hw = std::max<std::uint32_t>(1, std::thread::hardware_concurrency());
    if (configured == 0)
    {
        return std::min<std::uint32_t>(hw, pool_threads_max_auto_cap);
    }
    if (configured > hw && logger)
    {
        logger->warn("[PrimeMiningEngine] pool_threads={} exceeds "
                     "hardware_concurrency={}; honoring as-is",
                     configured, hw);
    }
    return configured;
}

void PrimeMiningEngine::run_stats_logger()
{
    // Stone 6.5 — periodic engine-stats info-level log line every 30s.
    // Park on m_pool_cv (which the destructor wakes via notify_all) so
    // shutdown is responsive: no 30-second teardown stall.
    using namespace std::chrono_literals;
    constexpr auto kInterval = 30s;

    while (!m_shutdown.load(std::memory_order_acquire))
    {
        {
            std::unique_lock<std::mutex> lock(m_pool_mtx);
            m_pool_cv.wait_for(lock, kInterval, [this] {
                return m_shutdown.load(std::memory_order_acquire);
            });
        }
        if (m_shutdown.load(std::memory_order_acquire))
        {
            break;
        }
        if (!m_logger)
        {
            continue;
        }
        const auto snap = snapshot_stats();
        m_logger->info("[PrimeMiningEngine] stats: "
                       "segments_processed={} candidates_dispatched={} "
                       "segments_discarded_epoch_changed={} segments_skipped_consumed={} "
                       "chunks_drawn={} starting_multiples_calls={} "
                       "pool_threads_running={} pool_threads_crashed={} "
                       "sessions_published={} same_base_short_circuits={} "
                       "allocator_resets={}",
                       snap.segments_processed,
                       snap.candidates_dispatched,
                       snap.segments_discarded_epoch_changed,
                       snap.segments_skipped_consumed,
                       snap.chunks_drawn,
                       snap.starting_multiples_calls,
                       snap.pool_threads_running,
                       snap.pool_threads_crashed,
                       snap.sessions_published,
                       snap.same_base_short_circuits,
                       snap.allocator_resets);

        // Stone 6.9 — find→test→dispatch funnel diagnostic line.  Logged
        // separately from the engine "stats:" line so operators (and grep)
        // can tell at a glance whether candidates_dispatched=0 is caused by
        // (a) the sieve never producing chains (chains_found_by_sieve low),
        // (b) Fermat truncation (chains_pushed_long ≪ chains_found_by_sieve),
        // or (c) the network-difficulty gate silently rejecting otherwise-
        // valid candidates (validate_rejected_below_diff dominant).
        m_logger->info("[PrimeMiningEngine] funnel: "
                       "popcount_windows_passed={} "
                       "chains_found_by_sieve={} chains_pushed_long={} "
                       "validate_attempts={} "
                       "rejected_base_not_prime={} rejected_below_diff={} "
                       "rejected_malformed={}",
                       snap.popcount_windows_passed,
                       snap.chains_found_by_sieve,
                       snap.chains_pushed_long,
                       snap.validate_attempts,
                       snap.validate_rejected_base_not_prime,
                       snap.validate_rejected_below_diff,
                       snap.validate_rejected_malformed);
    }
}

void PrimeMiningEngine::run_pool_thread(std::uint32_t pool_index)
{
    m_pool_threads_running.fetch_add(1, std::memory_order_relaxed);

    try
    {
        // Per-thread Sieve owned in this stack frame (Option A — one Sieve
        // per pool thread).  Allocated lazily so test_skip_sieve avoids the
        // (heavy) Sieve construction entirely.
        std::unique_ptr<Sieve> sieve;
        if (!m_cfg.test_skip_sieve)
        {
            sieve = std::make_unique<Sieve>();
        }

        // Per-thread per-session bookkeeping.  Reset on every session rebind
        // so each new template re-runs the sieve prepare path.
        //
        // Option 1 — `bound_session` captures the EngineSession the in-flight
        // chunk was sieved against.  Dispatch uses this captured pointer so
        // a KEEPALIVE Merkle-rotation republish that lands mid-chunk doesn't
        // change which block_data the candidate is dispatched against.  The
        // post-segment discard now keys on the STABLE (previous_hash, nHeight)
        // pair (matching the publish-layer discriminator from PR #674) rather
        // than base_hash, so Merkle-only rotations no longer abandon work.
        std::shared_ptr<const EngineSession> bound_session;
        std::uint64_t local_nonce = 0;        // session->starting_nonce, rounded
        uint1k local_sieve_start{};            // == bound_session->base_hash + local_nonce, rounded
        bool bound = false;

        while (!m_shutdown.load(std::memory_order_acquire))
        {
            // ── Acquire-load the latest session (single re-load gives us a
            // consistent (epoch_id, is_consumed) tuple snapshot).
            auto session = current_session();
            if (!session || session->is_consumed())
            {
                // Idle.  When the session is consumed (vs. simply not yet
                // published), record the skipped-segment counter so operators
                // can distinguish "engine has spent template" from "engine
                // hasn't started yet".
                if (session && session->is_consumed())
                {
                    m_segments_skipped_consumed.fetch_add(1, std::memory_order_relaxed);
                }
                bound = false;
                std::unique_lock<std::mutex> lock(m_pool_mtx);
                m_pool_cv.wait_for(lock, std::chrono::milliseconds{50}, [&] {
                    if (m_shutdown.load(std::memory_order_acquire))
                        return true;
                    auto s = current_session();
                    return s && !s->is_consumed();
                });
                continue;
            }

            // ── Capture the bound session by shared_ptr up front (Option 1).
            // The pool thread's in-flight chunk runs against THIS session's
            // (base_hash, block_data) — not against whatever current_session()
            // returns later.  This decouples mid-chunk Merkle-rotation
            // republishes (which change base_hash but preserve the cooperative
            // cursor) from chunk-abort decisions: the chunk completes against
            // its bound session, and dispatch uses bound_session->block_data
            // (the block whose proof-hash the candidate was actually sieved
            // against).  See post-segment re-check below for the stable
            // (previous_hash, nHeight) discriminator that DOES abort the chunk.
            const std::uint64_t my_epoch = session->epoch_id;
            const uint1k my_base_hash = session->base_hash;
            const std::uint32_t my_nbits = session->nbits;

            // ── Rebind on base-hash change OR nbits change.
            //
            // Sieve starting multiples depend on the EXACT base_hash, so any base_hash
            // change requires sieve->prepare() to be re-run.  Same-base republishes
            // (epoch advances but base_hash unchanged) do NOT need a sieve re-prepare
            // for starting multiples — local_sieve_start is determined entirely by
            // (base_hash, starting_nonce).
            //
            // HOWEVER, the sieve's target_length (which controls close_chain_min,
            // popcount_window_floor, and the per-segment chain-quality gate) is also
            // rebound inside this block, derived from session->nbits.  If nbits
            // changes across two same-base templates (rare but possible when the node
            // serves a singleflight waiter a template built pre-retarget while the
            // local view of difficulty has already advanced), and we DON'T re-enter
            // this block, the sieve keeps running with the stale target_length and
            // silently drops length-N candidates as below-difficulty.  The bug class
            // surfaces as: stats show Best 7.x climbing while bucket 7 stays at 0 and
            // rejected_below_diff dominates validate_attempts.
            //
            // Extending the gate to also fire on nbits change closes the hole.
            const bool need_rebind = !bound
                || !bound_session
                || my_base_hash != bound_session->base_hash
                || my_nbits != bound_session->nbits;
            if (need_rebind)
            {
                // Stone 6.9 — derive the per-session target Cunningham chain
                // length from session->nbits the same way Worker_prime does
                // (nbits / 10'000'000.0).  Floor → ceil so that at difficulty
                // 6.93 we target length 7 (not 7+1=8, which is what the
                // hard-coded m_min_chain_length=8 was effectively forcing
                // and which produced the bucket-7=0 symptom).  Clamp at 2 to
                // never disable the sieve's chain-cluster filter entirely.
                const double required_difficulty =
                    static_cast<double>(session->nbits) / 10000000.0;
                int target_length = nexusminer::mining::derive_target_length(session->nbits);

                // Stone 6.9.2 — observability for difficulty-driven gate
                // changes.  Logged at INFO with a stable, greppable prefix
                // so operators can correlate any throughput step-change
                // with the underlying target_length transition (the
                // popcount/close_chain thresholds both auto-scale with
                // target_length via mining/prime_thresholds.hpp).  Logged
                // only when the value actually changes, and only when this
                // pool thread had a previous bound session — initial bind
                // is implicit in pool startup and would just be noise.
                if (sieve && bound_session
                    && sieve->get_target_length() != target_length
                    && m_logger)
                {
                    const char* rebind_reason =
                        (my_base_hash != bound_session->base_hash && my_nbits != bound_session->nbits) ? "base+nbits" :
                        (my_nbits != bound_session->nbits)                                              ? "nbits"      :
                                                                                                          "base";
                    m_logger->info("[PrimeMiningEngine] pool[{}] target_length "
                                   "{} -> {} (nbits={}, difficulty={:.2f}, "
                                   "popcount_floor={}, close_chain_min={}, reason={})",
                                   pool_index,
                                   sieve->get_target_length(), target_length,
                                   session->nbits, required_difficulty,
                                   nexusminer::mining::popcount_window_floor(target_length),
                                   nexusminer::mining::close_chain_min(target_length),
                                   rebind_reason);
                }

                if (sieve)
                {
                    const uint1k startprime = my_base_hash + session->starting_nonce;
                    local_sieve_start = sieve->prepare(startprime, target_length);
                    local_nonce = static_cast<std::uint64_t>(local_sieve_start - my_base_hash);
                }
                else
                {
                    local_nonce = session->starting_nonce;
                    local_sieve_start = my_base_hash + session->starting_nonce;
                }
                bound = true;
            }
            // Always re-capture the latest bound_session so dispatch uses the
            // freshest (block_data, on_found, internal_id_for_solution) tuple
            // that matches the base_hash we are currently sieving against.
            bound_session = session;

            // ── Stone 6.5: draw a CHUNK of contiguous segments from the
            // cooperative cursor.  Inside the chunk the segments are
            // contiguous, so Sieve::sieve_segment()'s wheel-advance-by-one
            // mechanic is valid and calculate_starting_multiples() only
            // needs to be called ONCE at the chunk top instead of per
            // segment.  This is the per-segment-overhead amortisation that
            // restores engine-mode throughput to legacy "workers"-mode
            // territory.
            //
            // The static_cast is safe: the engine constructor unconditionally
            // builds a Shared_segment_allocator (see PrimeMiningEngine ctor).
            // The polymorphic Segment_allocator base only exists so tests can
            // inject fakes against next_segment_start(); chunk drawing is
            // engine-internal and bypasses that abstraction.
            const std::uint64_t chunk_segments = m_pool_chunk_segments;
            const std::uint64_t chunk_base = static_cast<Shared_segment_allocator&>(
                *m_segment_allocator).next_segment_chunk(chunk_segments);
            m_chunks_drawn.fetch_add(1, std::memory_order_relaxed);

            // ── Chunk-top: prime the wheel state ONCE for the whole chunk.
            // The wheel then advances correctly across the chunk's contiguous
            // segments via the per-segment Sieve::sieve_segment() calls in
            // the inner loop.  Do NOT add any per-segment wheel-state reset
            // inside the inner loop — it would re-introduce the Stone 6
            // performance bug.
            if (sieve)
            {
                const uint1k chunk_start =
                    local_sieve_start + static_cast<std::uint64_t>(chunk_base);
                sieve->calculate_starting_multiples(chunk_start);
                m_starting_multiples_calls.fetch_add(1, std::memory_order_relaxed);
            }

            // ── Inner segment loop: iterate chunk_segments contiguous
            // segments.  Each iteration runs the per-segment sieve pipeline
            // and the post-segment session re-check (mid-chunk template
            // changes abandon the rest of the chunk and re-enter the outer
            // loop where the rebind happens).
            bool chunk_aborted = false;
            for (std::uint64_t seg_index = 0;
                 seg_index < chunk_segments && !chunk_aborted;
                 ++seg_index)
            {
                if (m_shutdown.load(std::memory_order_acquire)) break;

                const std::uint64_t low =
                    chunk_base + seg_index * m_cfg.segment_size;

                // ── Run the sieve pipeline (or skip in test mode).
                std::vector<std::uint64_t> segment_chain_offsets;
                if (sieve)
                {
                    sieve->reset_sieve();
                    if (m_shutdown.load(std::memory_order_acquire)) break;
                    sieve->clear_chains();
                    if (m_shutdown.load(std::memory_order_acquire)) break;
                    // NOTE: NO per-segment calculate_starting_multiples() here.
                    // The wheel was primed once at the chunk top above; the
                    // contiguous sieve_segment() calls advance it correctly.
                    sieve->sieve_segment();
                    // Stone 6.9 — snapshot the per-sieve "chains found by
                    // close_chain" diag counter BEFORE find_chains so we can
                    // attribute the delta produced by THIS segment to the
                    // engine-level funnel counter.  test_chains() then bumps
                    // m_diag_chains_pushed_long for chains it pushed onto
                    // m_long_chain_starts; we mirror that delta to
                    // m_chains_pushed_long.  Using deltas (rather than reading
                    // the absolute counter once per loop) keeps the engine
                    // counter monotone across same-base chunk continuations.
                    const std::uint64_t found_before =
                        sieve->m_diag_chain_candidates_found.load(std::memory_order_relaxed);
                    const std::uint64_t pushed_before =
                        sieve->m_diag_chains_pushed_long.load(std::memory_order_relaxed);
                    const std::uint64_t popcount_before =
                        sieve->m_diag_popcount_windows_passed.load(std::memory_order_relaxed);
                    sieve->find_chains(low, false);
                    sieve->test_chains(local_sieve_start);
                    const std::uint64_t found_after =
                        sieve->m_diag_chain_candidates_found.load(std::memory_order_relaxed);
                    const std::uint64_t pushed_after =
                        sieve->m_diag_chains_pushed_long.load(std::memory_order_relaxed);
                    const std::uint64_t popcount_after =
                        sieve->m_diag_popcount_windows_passed.load(std::memory_order_relaxed);
                    if (found_after > found_before)
                    {
                        m_chains_found_by_sieve.fetch_add(found_after - found_before,
                                                         std::memory_order_relaxed);
                    }
                    if (pushed_after > pushed_before)
                    {
                        m_chains_pushed_long.fetch_add(pushed_after - pushed_before,
                                                       std::memory_order_relaxed);
                    }
                    if (popcount_after > popcount_before)
                    {
                        m_popcount_windows_passed.fetch_add(
                            popcount_after - popcount_before,
                            std::memory_order_relaxed);
                    }
                    segment_chain_offsets = sieve->m_long_chain_starts;
                }

                if (m_cfg.test_force_candidate_per_segment)
                {
                    // Synthetic candidate at the segment's start.  Only
                    // meaningful for tests that bypass the real sieve.
                    segment_chain_offsets.push_back(low);
                }

                // ── Test seam: simulated_segment_latency widens the race
                // window between segment draw and the post-segment session
                // re-check so tests like test_heavy_churn_drives_discards
                // are deterministic even when the sieve pipeline is skipped
                // (test_skip_sieve).  Production callers leave this at zero.
                if (m_cfg.simulated_segment_latency.count() > 0)
                {
                    std::this_thread::sleep_for(m_cfg.simulated_segment_latency);
                }

                // ── REQUIRED post-segment re-check (Stone 6 correctness gate).
                // Runs per SEGMENT, not per chunk: a real chain-tip advance
                // can land at any time and we must not grind the remaining
                // chunk_segments-1 segments against a stale tip.
                //
                // Option 1 — discriminator is the STABLE (previous_hash, nHeight)
                // pair (matching the publish-layer same-base discriminator from
                // PR #674), NOT base_hash.  KEEPALIVE Merkle-rotations rotate
                // base_hash without advancing the chain tip; the in-flight
                // sieve output is still cryptographically valid for the OLD
                // base_hash, so we let the chunk complete and dispatch against
                // bound_session->block_data (the block whose proof-hash space
                // the candidates were sieved against).  Only a real chain-tip
                // change (different prev_hash or nHeight) — or a found-block
                // consumed signal — abandons the chunk.
                //
                // The legacy fast paths (force-rebind on shutdown / null-fresh)
                // remain: a null `fresh` means the consumer hasn't published
                // anything yet, which is unreachable here because we already
                // bound a session at the top of the outer loop, but we keep
                // the defensive null-check for symmetry with the idle path.
                auto fresh = current_session();
                const bool tip_advanced =
                    fresh && bound_session
                    && (fresh->block_data.previous_hash != bound_session->block_data.previous_hash
                        || fresh->block_data.nHeight   != bound_session->block_data.nHeight);
                if (!fresh
                    || tip_advanced
                    || (bound_session && bound_session->is_consumed())
                    || fresh->is_consumed())
                {
                    m_segments_discarded_epoch_changed.fetch_add(1,
                        std::memory_order_relaxed);
                    bound = false;       // force rebind on next outer iter
                    chunk_aborted = true; // abandon the rest of this chunk
                    break;
                }
                (void)my_epoch;  // captured for traceability/future logging

                // ── Dispatch each chain candidate via asio::post.
                //
                // Option 1 — dispatch references `bound_session`, NOT `fresh`.
                // The candidate offsets in this segment came from a sieve
                // primed against bound_session->base_hash; submitting them
                // against any other block_data would produce an invalid block.
                // bound_session is held by shared_ptr so it remains valid
                // even if a later publish has already swapped m_session.
                const bool dispatch_real = !m_cfg.test_skip_sieve;
                bool session_consumed_here = false;
                for (auto x : segment_chain_offsets)
                {
                    Block_data candidate_block = bound_session->block_data;
                    candidate_block.nNonce = local_nonce + x;

                    std::vector<std::uint8_t> offsets;
                    bool is_valid = false;
                    double actual_difficulty = 0.0;

                    if (dispatch_real)
                    {
                        const uint1k chain_start =
                            bound_session->base_hash + candidate_block.nNonce;
                        const uint1024_t hashPrime = boost_uint1k_to_uint1024(chain_start);
                        // Required network difficulty derived from the session's
                        // nBits the same way Worker_prime::getNetworkDifficulty()
                        // does it (nbits / 10'000'000.0).  Without this, every
                        // Fermat-passing candidate would be flagged "valid" and
                        // dispatched, flooding the engine with false positives.
                        const double required_difficulty =
                            static_cast<double>(bound_session->nbits) / 10000000.0;
                        m_validate_attempts.fetch_add(1, std::memory_order_relaxed);
                        is_valid = nexusminer::prime::ValidatePrimeCandidate(
                            hashPrime,
                            required_difficulty,
                            offsets,
                            actual_difficulty);
                        if (is_valid && !nexusminer::prime::is_well_formed_prime_offsets(offsets))
                        {
                            if (m_logger)
                            {
                                m_logger->error("[PrimeMiningEngine] pool[{}] rejecting "
                                                "candidate with malformed offsets ({} bytes, "
                                                "expected size in [{}..{}])",
                                                pool_index, offsets.size(),
                                                kMinSerializedPrimeOffsets,
                                                kMaxSerializedPrimeOffsets);
                            }
                            is_valid = false;
                            m_validate_rejected_malformed.fetch_add(1, std::memory_order_relaxed);
                        }
                        else if (!is_valid)
                        {
                            // Disambiguate the two ValidatePrimeCandidate
                            // failure modes using its post-conditions
                            // (prime_validation.cpp:212-237):
                            //   * base PrimeCheck failed → vOffsets cleared
                            //     and nDifficulty set to 0.
                            //   * difficulty < required → vOffsets retained,
                            //     nDifficulty > 0 but < required.
                            // Anything else (offsets non-empty AND
                            // actual_difficulty == 0) cannot occur today
                            // but is folded into below_diff for safety.
                            if (offsets.empty() && actual_difficulty == 0.0)
                            {
                                m_validate_rejected_base_not_prime.fetch_add(
                                    1, std::memory_order_relaxed);
                            }
                            else
                            {
                                m_validate_rejected_below_diff.fetch_add(
                                    1, std::memory_order_relaxed);
                            }
                        }
                    }
                    else
                    {
                        // Test seam: no real validation; force-success path.
                        // Synthesize a minimum-well-formed offsets vector
                        // (chain length 2 = 1 gap byte + 4-byte fraction = 5
                        // bytes).  Decoupled from the production gate's upper
                        // bound so a future bump of kMaxSerializedPrimeOffsets
                        // can't accidentally make the test seam emit
                        // malformed-by-the-gate output.
                        is_valid = m_cfg.test_force_candidate_per_segment;
                        offsets.assign(kMinSerializedPrimeOffsets, 0);
                    }

                    if (!is_valid)
                    {
                        continue;
                    }

                    // Stone 6.8 — channel-wide best-difficulty fan-in.  Mirror
                    // the legacy Worker_prime path: m_segmented_sieve->m_best_chain
                    // is updated with std::max(actual_difficulty, ...) only on
                    // valid candidates (i.e. those that meet required network
                    // difficulty).  Engine mode tracks the same value as a
                    // single channel-wide atomic so every registered
                    // Worker_prime can publish it as its "Best".  std::atomic
                    // <double> has no fetch_max, hence the CAS loop.
                    {
                        double current = m_best_difficulty.load(std::memory_order_relaxed);
                        while (actual_difficulty > current &&
                               !m_best_difficulty.compare_exchange_weak(
                                   current, actual_difficulty,
                                   std::memory_order_relaxed,
                                   std::memory_order_relaxed))
                        {
                            // current was reloaded by compare_exchange_weak;
                            // loop until either we win the CAS or another
                            // thread has installed a value >= ours.
                        }
                    }

                    // Mark the bound session consumed BEFORE asio::post so other
                    // pool threads, on their next session re-check, observe the
                    // consumed bit and idle.  Single-found-block-wins.  Also
                    // mark `fresh` consumed (when distinct from bound_session)
                    // because in the Merkle-rotation case the consumer has
                    // already moved on; idling threads should observe the
                    // consumed transition on whichever session they re-load.
                    bound_session->mark_consumed();
                    if (fresh && fresh.get() != bound_session.get())
                    {
                        fresh->mark_consumed();
                    }
                    m_candidates_dispatched.fetch_add(1, std::memory_order_relaxed);
                    session_consumed_here = true;

                    if (bound_session->on_found && m_cfg.io_context)
                    {
                        auto session_for_dispatch = bound_session;
                        auto block_copy = candidate_block;
                        auto captured_offsets = std::move(offsets);
                        ::asio::post(*m_cfg.io_context,
                            [session_for_dispatch,
                             block_copy,
                             captured_offsets = std::move(captured_offsets)]() mutable {
                                auto bd = std::make_unique<Block_data>(block_copy);
                                bd->vOffsets = std::move(captured_offsets);
                                session_for_dispatch->on_found(
                                    session_for_dispatch->internal_id_for_solution,
                                    std::move(bd));
                            });
                    }

                    // After dispatch the session is consumed; break out of
                    // the candidate loop and let the outer loop re-check
                    // and idle.
                    break;
                }

                m_segments_processed.fetch_add(1, std::memory_order_relaxed);

                // If we just consumed the session, abandon the rest of the
                // chunk: subsequent segments would just observe is_consumed()
                // in the post-segment re-check anyway.  Falling through to
                // the outer loop lets the idle/wakeup machinery do its job.
                if (session_consumed_here)
                {
                    chunk_aborted = true;
                    break;
                }
            }

            // Option 2 — at every chunk boundary republish this pool thread's
            // chain-length histogram snapshot from the OWNING thread.  The
            // stats path reads only the snapshot (Pool_histogram_snapshot in
            // the engine), never the live Sieve* — restoring the documented
            // invariant in stats/prime_stats_snapshot.hpp that diagnostic
            // counters are published from the worker thread.  Snapshotting
            // per chunk (instead of per segment) bounds the cost: the histogram
            // is two 12-element arrays, so the publish is two atomic shared_ptr
            // swaps amortised over pool_chunk_segments segments (default 64).
            if (sieve && m_pool_histogram_snapshots
                && pool_index < m_pool_thread_count)
            {
                Pool_histogram_snapshot snap{};
                const auto best_local = sieve->snapshot_chain_histogram();
                const auto attempted_local = sieve->snapshot_chain_histogram_attempted();
                const std::size_t nb = std::min(best_local.size(), snap.best.size());
                for (std::size_t b = 0; b < nb; ++b)
                {
                    snap.best[b] = best_local[b];
                }
                const std::size_t na = std::min(attempted_local.size(),
                                                snap.attempted.size());
                for (std::size_t b = 0; b < na; ++b)
                {
                    snap.attempted[b] = attempted_local[b];
                }
                m_pool_histogram_snapshots[pool_index].store(std::move(snap));
            }
        }
    }
    catch (const std::exception& e)
    {
        m_pool_threads_crashed.fetch_add(1, std::memory_order_relaxed);
        if (m_logger)
        {
            m_logger->error("[PrimeMiningEngine] pool thread {} crashed: {}",
                            pool_index, e.what());
        }
    }
    catch (...)
    {
        m_pool_threads_crashed.fetch_add(1, std::memory_order_relaxed);
        if (m_logger)
        {
            m_logger->error("[PrimeMiningEngine] pool thread {} crashed (unknown exception)",
                            pool_index);
        }
    }

    m_pool_threads_running.fetch_sub(1, std::memory_order_relaxed);
}

} // namespace cpu
} // namespace nexusminer
