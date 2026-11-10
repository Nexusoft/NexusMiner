#include "gpu/worker_hash.hpp"
#include "config/worker_config.hpp"
#include "stats/stats_collector.hpp"
#include "block.hpp"
#include <asio/io_context.hpp>
#include <asio/post.hpp>
#include "cuda_hash/util.h"
#include "cuda_hash/sk1024.h"
#include "LLC/hash/SK.h"
#include "LLC/types/uint1024.h"
#include "LLC/types/bignum.h"
#include "TAO/Ledger/difficulty.h"
#include <cstddef>
#include <stdexcept>

namespace nexusminer
{
namespace gpu
{

// PR #681 follow-up §5: the cuda_sk1024_hash kernel reads the start nonce as
// `((uint64_t*)TheData)[26]` (sk1024.cu:776, 822) where TheData is the byte
// pointer to Block_data::nVersion.  The aliasing trick is undocumented and
// fragile against any reordering of Block_data fields — lock the layout
// assumption in here so a future "harmless" refactor breaks the build instead
// of silently corrupting hash-rate statistics.  Block_data is not strictly
// standard-layout (it has user-provided constructors), but offsetof on a
// trivially-destructible aggregate-ish type is universally supported by the
// compilers we ship for and is the cleanest way to express the contract.
static_assert(offsetof(Block_data, nNonce) == 26 * sizeof(std::uint64_t),
    "cuda_sk1024_hash assumes Block_data::nNonce lives at "
    "((uint64_t*)&Block_data::nVersion)[26]; do not reorder Block_data "
    "fields without updating sk1024.cu's TheData[26] uses.");

std::uint32_t Worker_hash::device_id() const
{
    return std::get<config::Worker_config_gpu>(m_config.m_worker_mode).m_device;
}

bool Worker_hash::bind_device_context(const char* phase)
{
    const auto gpu_device = device_id();
    const auto device_count = cuda_num_devices();
    if (device_count == 0)
    {
        m_logger->error("{}No CUDA devices detected during {}", m_log_leader, phase);
        return false;
    }

    if (gpu_device >= device_count)
    {
        m_logger->error("{}Configured CUDA device {} is out of range during {} ({} device(s) detected)",
            m_log_leader, gpu_device, phase, device_count);
        return false;
    }

    cuda_init(gpu_device);
    return true;
}

Worker_hash::Worker_hash(std::shared_ptr<asio::io_context> io_context, Worker_config& config)
: m_io_context{std::move(io_context)}
, m_logger{spdlog::get("logger")}
, m_config{config}
, m_found_nonce_callback{}
, m_stop{true}
, m_log_leader{ "GPU Worker " + m_config.m_id + ": " }
, m_pool_nbits{0}
, m_threads_per_block{896}
, m_best_leading_zeros{0}
, m_met_difficulty_count{0}
{
    if (!bind_device_context("worker initialization"))
    {
        throw std::runtime_error("GPU hash worker initialization failed");
    }

    // Allocate memory associated with Device Hashing
    cuda_sk1024_init(device_id());

    // Compute the intensity by determining number of multiprocessors
    const auto multiprocessors = cuda_device_multiprocessors(device_id());
    if (multiprocessors == 0)
    {
        throw std::runtime_error("Configured CUDA device reported zero multiprocessors");
    }

    m_intensity = 2 * multiprocessors;
    m_logger->info("{}Using CUDA device {} ({}) with {} multiprocessor(s)",
        m_log_leader,
        device_id(),
        cuda_devicename(device_id()),
        multiprocessors);
    m_logger->debug("{} intensity set to {}", cuda_devicename(device_id()), m_intensity);

    // Calcluate the throughput for the cuda hash mining
    m_throughput = 256 * m_threads_per_block * m_intensity;

    // Start persistent thread
    m_shutdown = false;
    m_run_thread = std::thread(&Worker_hash::run, this);
    m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "Persistent worker thread started"));
}

Worker_hash::~Worker_hash()
{
    m_stop = true;  // Interrupt mining loops before competing for m_mtx during shutdown

    // Signal shutdown and wake up the worker thread
    {
        std::lock_guard<std::mutex> lock(m_mtx);
        m_shutdown = true;
        m_running = false;
    }
    m_cv.notify_all();

    // Wait for thread to finish
    if (m_run_thread.joinable())
    {
        m_run_thread.join();
    }

    // Free the GPU device memory associated with hashing
    if (bind_device_context("worker shutdown"))
    {
        cuda_sk1024_free(device_id());

        // Free the GPU device memory and reset them
        cuda_free(device_id());
    }
}

void Worker_hash::set_block(::LLP::CBlock block, std::uint32_t nbits, Worker::Block_found_handler result)
{
    // Update work data atomically and signal worker thread
    {
        std::lock_guard<std::mutex> lock(m_mtx);
        m_found_nonce_callback = result;
        m_block = Block_data{block};
        if (nbits != 0)
        {
            // take nbits provided by pool
            m_pool_nbits = nbits;
        }

        // Stone — bug #7 fix: per-worker nonce sharding.  Mirrors CPU
        // worker_hash.cpp:88 and GPU worker_prime.cpp:144.  Without this,
        // multiple GPU hash workers on the same template would race over the
        // same nonce space.
        m_starting_nonce = static_cast<std::uint64_t>(m_config.m_internal_id) << 48;
        m_block.nNonce = m_starting_nonce;

        // Set the block for this device
        if (!bind_device_context("set_block"))
        {
            m_running = false;
            return;
        }
        cuda_sk1024_setBlock(&m_block.nVersion, m_block.nHeight);

        /* Get the target difficulty. */
        auto const nbits_cuda = m_pool_nbits != 0 ? m_pool_nbits : m_block.nBits;

        double mainnet_difficulty = TAO::Ledger::GetDifficulty(m_block.nBits, m_block.nChannel);
        double pool_difficulty = TAO::Ledger::GetDifficulty(m_pool_nbits, m_block.nChannel);
        if (m_pool_nbits != 0)
            m_logger->debug("Leading zeros required mainnet:{}  pool:{}", log2(mainnet_difficulty)+34, log2(pool_difficulty)+34);
        else
            m_logger->debug("Leading zeros required:{}", log2(mainnet_difficulty) + 34);

        /* Get the target difficulty. */
        LLC::CBigNum target;
        target.SetCompact(nbits_cuda);
        m_target = target.getuint1024();

        // Set the target hash on this device for the difficulty.
        cuda_sk1024_set_Target((uint64_t*)m_target.begin());

        // Signal new work is available.
        // m_stop = true interrupts the current while(!m_stop) loop iteration immediately,
        // matching the pattern used in the prime workers (PR #343).
        // Reset m_hashes to prevent INF MH/s on the first stats interval after recovery.
        m_hashes.store(0, std::memory_order_relaxed);
        // Stone — bug #8 fix: reset the consecutive-mismatch counter on every
        // new work, so a healthy session re-arms the fault threshold.  The
        // total counter (m_keccak_mismatch_total) is intentionally NOT reset
        // — it's a lifetime metric for the operator.
        m_keccak_mismatch_count.store(0, std::memory_order_relaxed);
        m_stop = true;
        m_new_work = true;
        m_running = true;
    }

    // Wake up the worker thread
    m_cv.notify_one();
    m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "New work set"));
}

void Worker_hash::set_block(std::shared_ptr<WorkPackage> work_package, Worker::Block_found_handler result)
{
    // Update work data atomically and signal worker thread
    {
        std::lock_guard<std::mutex> lock(m_mtx);
        m_found_nonce_callback = result;

        // Use precomputed data from WorkPackage
        const auto& block = work_package->get_block();
        m_block = Block_data{block};

        std::uint32_t nbits = work_package->get_nbits();
        if (nbits != 0)
        {
            // take nbits provided by pool
            m_pool_nbits = nbits;
        }

        // Stone — bug #7 fix (see set_block(CBlock,...) above for rationale).
        m_starting_nonce = static_cast<std::uint64_t>(m_config.m_internal_id) << 48;
        m_block.nNonce = m_starting_nonce;

        // Set the block for this device
        if (!bind_device_context("set_block"))
        {
            m_running = false;
            return;
        }
        cuda_sk1024_setBlock(&m_block.nVersion, m_block.nHeight);

        /* Get the target difficulty. */
        auto const nbits_cuda = m_pool_nbits != 0 ? m_pool_nbits : m_block.nBits;

        double mainnet_difficulty = TAO::Ledger::GetDifficulty(m_block.nBits, m_block.nChannel);
        double pool_difficulty = TAO::Ledger::GetDifficulty(m_pool_nbits, m_block.nChannel);
        if (m_pool_nbits != 0)
            m_logger->debug("Leading zeros required mainnet:{}  pool:{}", log2(mainnet_difficulty)+34, log2(pool_difficulty)+34);
        else
            m_logger->debug("Leading zeros required:{}", log2(mainnet_difficulty) + 34);

        /* Get the target difficulty. */
        LLC::CBigNum target;
        target.SetCompact(nbits_cuda);
        m_target = target.getuint1024();

        // Set the target hash on this device for the difficulty.
        cuda_sk1024_set_Target((uint64_t*)m_target.begin());

        // Signal new work is available.
        m_hashes.store(0, std::memory_order_relaxed);
        // Stone — bug #8 fix (see set_block(CBlock,...) above for rationale).
        m_keccak_mismatch_count.store(0, std::memory_order_relaxed);
        m_stop = true;
        m_new_work = true;
        m_running = true;
    }

    // Wake up the worker thread
    m_cv.notify_one();
    m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "New work set"));
}

void Worker_hash::run()
{
    m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "Persistent worker thread ready, waiting for work..."));

    if (!bind_device_context("worker thread start"))
    {
        m_running = false;
        return;
    }

    // Persistent thread loop - runs until shutdown
    while (true) {
        // Wait for new work or shutdown signal
        {
            std::unique_lock<std::mutex> lock(m_mtx);
            m_cv.wait(lock, [this] { return m_new_work || m_shutdown; });

            // Check for shutdown
            if (m_shutdown) {
                m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "Worker thread shutting down"));
                break;
            }

            // Clear new work flag
            m_new_work = false;
            m_stop = false;
        }

        // Start mining with the new work
        m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "Starting GPU mining"));

        // Stone — bug #9 fix: snapshot the mutable shared state once per
        // work-batch under m_mtx, then drive the inner loop entirely off
        // local copies.  Previously cuda_sk1024_hash was invoked with
        // &m_block.nVersion, m_target, m_block.nNonce, m_block.nHeight while
        // set_block() could be writing those fields under m_mtx — the same
        // race pattern we already fixed on the prime side.  Mirrors
        // worker_prime.cpp:195-208 (Block_data local_block; uint1k local_base_hash;
        // uint64_t local_nonce; under scoped_lock).
        //
        // PR #681 follow-up — Variant B of §3 of the plan.  We deliberately do
        // NOT keep `local_nonce` as a separate variable from local_block.nNonce.
        // cuda_sk1024_hash's contract (sk1024.cu:758-829, sk1024.h:21-38) is
        // that `TheNonce` must alias `((uint64_t*)TheData)[26]`: the kernel
        // advances the nonce by mutating TheData[26] on the non-winner path
        // and computes hashes_done as `doneNonce - first_nonce + 1` where
        // `first_nonce = TheNonce` snapshotted at entry.  If TheNonce and
        // TheData[26] drift apart between iterations, hashes_done is reported
        // against a stale baseline and grows quadratically (`K(K+1)/2 · T`
        // instead of `K · T`) — the "INF MH/s" failure mode that PR #343 was
        // originally written to prevent.  Pre-PR #681 the kernel was called
        // with m_block.nNonce by-reference, which preserved that aliasing.
        // PR #681's snapshot block introduced a separate `local_nonce` that
        // only updated on a winner, breaking the invariant.  We restore the
        // invariant by passing local_block.nNonce by-reference (see the
        // cuda_sk1024_hash call below) so there is exactly one piece of state
        // for "the current nonce" and it lives in exactly one place.  The
        // kernel-layout coupling is locked in by the static_assert at the top
        // of this function (offsetof nNonce == 26 * sizeof(uint64_t)).
        // device_id() does a std::get on a std::variant — hoist out of the
        // hot loop (§4(2) of the plan); the configured device cannot change
        // mid-session.
        Block_data local_block;
        uint1024_t local_target;
        std::uint32_t local_throughput;
        std::uint32_t local_threads_per_block;
        std::uint32_t local_device_id;
        {
            std::scoped_lock<std::mutex> lck(m_mtx);
            local_block = m_block;
            local_target = m_target;
            local_throughput = m_throughput;
            local_threads_per_block = m_threads_per_block;
            local_device_id = device_id();
        }

        while (!m_stop)
        {
            // Check for new work at the top of the loop
            {
                std::unique_lock<std::mutex> lck(m_mtx);
                if (m_new_work)
                {
                    break;
                }
            }

            if (step_once(local_block, local_target,
                          local_throughput, local_threads_per_block,
                          local_device_id))
            {
                break;
            }
        }

        m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "Mining stopped, waiting for new work..."));
    }  // End of persistent thread loop
}


// step_once — inner-loop body, extracted for deterministic testing.
// Calls cuda_sk1024_hash once with the provided snapshot state, accumulates
// hashes into m_hashes, handles keccak-mismatch accounting, and dispatches
// a found-block callback on a credited winner.
//
// Returns true when the caller should break the mining loop:
//   - keccak-mismatch fault threshold reached (m_running set to false), OR
//   - a real winner was found and the callback dispatched.
// Returns false to continue mining.
//
// Precondition: local_block.nNonce == ((uint64_t*)&local_block.nVersion)[26]
// (the static_assert at the top of this file locks that invariant).
bool Worker_hash::step_once(Block_data& local_block,
                             const uint1024_t& local_target,
                             std::uint32_t local_throughput,
                             std::uint32_t local_threads_per_block,
                             std::uint32_t local_device_id)
{
    std::uint64_t hashes = 0;
    // Stone — bug #8 fix: surface keccak (CPU revalidation) mismatches
    // to the worker so we can route them through spdlog and treat
    // repeated mismatches as a hardware fault.  Counter is delta per
    // call; cuda_sk1024_hash adds 1 on each mismatch.
    std::uint32_t keccak_mismatch_delta = 0;

    // Do hashing on a CUDA device.  All inputs are local snapshots so
    // a concurrent set_block() cannot tear them mid-kernel.  We pass
    // local_block.nNonce by-reference (see snapshot-block comment
    // above for the kernel-contract rationale) so the kernel's
    // "advance TheData[26], leave TheNonce alone on miss" semantics
    // remain coherent across iterations.
    bool found = cuda_sk1024_hash(
        local_device_id,
        reinterpret_cast<uint32_t*>(&local_block.nVersion),
        local_target,
        local_block.nNonce,
        &hashes,
        local_throughput,
        local_threads_per_block,
        local_block.nHeight,
        &keccak_mismatch_delta);

    m_hashes.fetch_add(hashes, std::memory_order_relaxed);

    // Stone — bug #8 fix: handle keccak mismatches.  These represent
    // a GPU that returned a "winner" the host could not revalidate —
    // a hardware-fault signal (bit-flip / marginal clocks / thermal).
    if (keccak_mismatch_delta > 0)
    {
        m_keccak_mismatch_total.fetch_add(keccak_mismatch_delta,
                                          std::memory_order_relaxed);
        const auto consec = m_keccak_mismatch_count.fetch_add(
            keccak_mismatch_delta, std::memory_order_relaxed)
            + keccak_mismatch_delta;
        m_logger->warn(spdlog::fmt_lib::runtime(
            m_log_leader +
            "GPU keccak (CPU-revalidation) mismatch: "
            "consecutive={} total={} threshold={} — possible hardware fault"),
            consec,
            m_keccak_mismatch_total.load(std::memory_order_relaxed),
            kKeccakMismatchFaultThreshold);
        if (consec >= kKeccakMismatchFaultThreshold)
        {
            m_logger->error(spdlog::fmt_lib::runtime(
                m_log_leader +
                "Keccak mismatch fault threshold reached "
                "({} consecutive); marking worker offline. "
                "Operator action required."),
                consec);
            m_running = false;
            m_stop = true;
            return true;  // caller should break
        }
    }

    // If a nonce with the right difficulty was found submit block.
    if (found && !m_stop.load())
    {
        // Stone — bug #8 fix: a healthy submission resets the
        // consecutive-mismatch counter.
        m_keccak_mismatch_count.store(0, std::memory_order_relaxed);

        ++m_met_difficulty_count;
        // Copy the winning nonce back into the canonical m_block, and snapshot
        // both the callback and winning block while holding the lock.  The
        // asio callback can run after new work arrives; it must submit the
        // block that actually won, not later mutable worker state.
        Worker::Block_found_handler found_nonce_callback;
        Block_data found_block;
        // Lock-protected because publish_statistics / set_block may also touch
        // m_block.  PR #681 follow-up:
        // local_block.nNonce already holds the winning nonce because
        // cuda_sk1024_hash sets `TheNonce = foundNonce` (sk1024.cu:796)
        // and TheNonce now aliases local_block.nNonce.
        {
            std::scoped_lock<std::mutex> lck(m_mtx);
            // Submit the block that was hashed.  Copying m_block here races
            // set_block(): the callback would post a new template with the
            // old nonce.  Only publish the winning nonce into the live
            // template when that template has not already been replaced.
            if (!m_new_work)
                m_block.nNonce = local_block.nNonce;
            found_block = local_block;
            found_nonce_callback = m_found_nonce_callback;
        }
        // Calculate the number of leading zero-bits (use the local
        // snapshot — local_block already has the correct nVersion etc).
        uint1024_t hash_proof = LLC::SK1024(BEGIN(local_block.nVersion), END(local_block.nNonce));
        std::uint32_t leading_zeros = 1024 - hash_proof.BitCount();
        int current_best = m_best_leading_zeros.load(std::memory_order_relaxed);
        const int new_best = static_cast<int>(leading_zeros);
        while (new_best > current_best &&
               !m_best_leading_zeros.compare_exchange_weak(
                   current_best, new_best, std::memory_order_relaxed))
        {
        }

        if (found_nonce_callback)
        {
            m_logger->info(spdlog::fmt_lib::runtime(m_log_leader + "💎 Block found! Posting to main io_context..."));
            ::asio::post(*m_io_context,
                [callback = std::move(found_nonce_callback),
                 internal_id = m_config.m_internal_id,
                 block = std::move(found_block)]() mutable
            {
                callback(internal_id, std::make_unique<Block_data>(block));
            });
        }
        else
        {
            m_logger->debug(spdlog::fmt_lib::runtime(m_log_leader + "Miner callback function not set."));
        }

        m_stop = true;
        return true;  // caller should break
    }

    return false;  // continue mining
}


void Worker_hash::update_statistics(stats::Collector& stats_collector)
{
    auto& typed = stats::as_typed<stats::Hash>(stats_collector);
    auto hash_stats = typed.get_worker_stats(m_config.m_internal_id);
    // PR #681 follow-up §4(4): exchange-with-zero so the publish + reset is
    // atomic from the writer's POV — no chance of double-counting hashes
    // posted between the read and the reset.
    hash_stats.m_hash_count += m_hashes.exchange(0, std::memory_order_relaxed);
    hash_stats.m_best_leading_zeros = m_best_leading_zeros.load(std::memory_order_relaxed);
    hash_stats.m_met_difficulty_count = m_met_difficulty_count.load(std::memory_order_relaxed);

    typed.update_worker_stats(m_config.m_internal_id, hash_stats);
}

}
}
