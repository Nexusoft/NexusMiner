#include "segment_allocator.hpp"

#include <spdlog/spdlog.h>
#include <string>
#include <utility>

namespace nexusminer {
namespace cpu {

Per_worker_segment_allocator::Per_worker_segment_allocator(std::uint64_t segment_size)
    : m_segment_size{segment_size}
{
}

void Per_worker_segment_allocator::reset(std::uint64_t starting_nonce)
{
    m_cursor = starting_nonce;
}

std::uint64_t Per_worker_segment_allocator::next_segment_start()
{
    const std::uint64_t out = m_cursor;
    m_cursor += m_segment_size;
    return out;
}

Shared_segment_allocator::Shared_segment_allocator(std::uint64_t segment_size)
    : m_segment_size{segment_size}
{
}

void Shared_segment_allocator::reset(std::uint64_t starting_nonce)
{
    // Single-writer (the engine consumer thread) on a new template; release
    // so that pool threads which subsequently observe the published
    // EngineSession (acquire) also see the reset cursor value.
    m_cursor.store(starting_nonce, std::memory_order_release);
}

std::uint64_t Shared_segment_allocator::next_segment_start()
{
    // Wait-free hot path: relaxed is sufficient because the cursor value is
    // not used to synchronise other state — it is the data being protected.
    return m_cursor.fetch_add(m_segment_size, std::memory_order_relaxed);
}

std::uint64_t Shared_segment_allocator::current() const
{
    return m_cursor.load(std::memory_order_acquire);
}

std::uint64_t Shared_segment_allocator::next_segment_chunk(std::uint64_t chunk_segments)
{
    // Wait-free hot path: relaxed is sufficient because the cursor value is
    // not used to synchronise other state — it is the data being protected.
    // Stride is chunk_segments * m_segment_size so the returned base value
    // owns exactly that many contiguous segments.
    return m_cursor.fetch_add(chunk_segments * m_segment_size,
                              std::memory_order_relaxed);
}

std::unique_ptr<Segment_allocator>
make_segment_allocator_for_engine_mode(const std::string& engine_mode,
                                       std::uint64_t segment_size)
{
    if (engine_mode == "engine")
    {
        // Stone 7: the cooperative PrimeMiningEngine path is now active and
        // owns its own Shared_segment_allocator.  Worker_prime under engine
        // mode never calls next_segment_start() on the per-worker allocator
        // returned here — it is a vestigial member kept for ABI/constructor
        // compatibility and will be removed in Stone 8.  No warning is
        // emitted; the engine startup banner ([PrimeMiningEngine] ...) is
        // the user-visible signal that engine mode is live.
    }
    else if (!engine_mode.empty() && engine_mode != "workers")
    {
        if (auto logger = spdlog::get("logger"))
        {
            logger->warn("[cpu] engine_mode = \"{}\" is not recognised; "
                         "expected \"workers\" or \"engine\". Falling back to "
                         "\"workers\".",
                         engine_mode);
        }
    }
    return std::make_unique<Per_worker_segment_allocator>(segment_size);
}

} // namespace cpu
} // namespace nexusminer
