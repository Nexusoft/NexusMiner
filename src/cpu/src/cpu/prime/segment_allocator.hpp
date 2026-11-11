#ifndef NEXUSMINER_CPU_PRIME_SEGMENT_ALLOCATOR_HPP
#define NEXUSMINER_CPU_PRIME_SEGMENT_ALLOCATOR_HPP

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

namespace nexusminer {
namespace cpu {

// Stone 3: pluggable segment cursor for the prime mining loop.
//
// The current Worker_prime maintains a private `low += segment_size` cursor
// seeded from `internal_id << 48`.  Routing it through this interface (a) makes
// the cursor explicit and unit-testable, and (b) gives the eventual
// PrimeMiningEngine (Option 3) a single place to swap in a cooperative shared
// cursor without rewriting Worker_prime.
//
// Each `next_segment_start()` returns the absolute starting nonce offset (from
// the worker's base hash) for the next segment to sieve; segment length is
// fixed and supplied at construction time.  `reset()` is called whenever a new
// template arrives so the cursor restarts at the worker's seed.
class Segment_allocator
{
public:
    virtual ~Segment_allocator() = default;

    // Reset the cursor for a new template/work cycle.
    virtual void reset(std::uint64_t starting_nonce) = 0;

    // Return the absolute starting nonce offset for the next segment, then
    // advance the cursor by one segment length.
    virtual std::uint64_t next_segment_start() = 0;

    // Current cursor value (== nonce offset of the *next* segment to be handed
    // out by next_segment_start()).  Useful for callers that need to seed
    // worker-local bookkeeping after reset().
    virtual std::uint64_t current() const = 0;
};

// Default allocator: each worker owns its own cursor, exactly preserving
// today's behaviour.  Construction takes the segment length so callers don't
// have to thread it through every call.
class Per_worker_segment_allocator : public Segment_allocator
{
public:
    explicit Per_worker_segment_allocator(std::uint64_t segment_size);

    void reset(std::uint64_t starting_nonce) override;
    std::uint64_t next_segment_start() override;
    std::uint64_t current() const override { return m_cursor; }

private:
    std::uint64_t m_segment_size;
    std::uint64_t m_cursor{0};
};

// Stone 5: cooperative cursor for the upcoming PrimeMiningEngine pool.
//
// One instance is owned by the engine and shared across N pool sieve threads.
// reset() is called by the engine consumer thread on a new template (single
// writer); next_segment_start() is called by every pool thread on the hot path
// (many concurrent callers).  Implemented as a single std::atomic<uint64_t>
// fetch_add — wait-free, no mutex on the hot path.
//
// The engine never calls reset() while a pool thread is mid-segment: pool
// threads observe the new EngineSession at the top of their loop and that
// observation is sequenced after the engine's session publish, which is in
// turn published *after* reset() has run on the consumer thread.  Pool threads
// therefore never see a partially-reset cursor for the new session.
class Shared_segment_allocator : public Segment_allocator
{
public:
    explicit Shared_segment_allocator(std::uint64_t segment_size);

    void reset(std::uint64_t starting_nonce) override;
    std::uint64_t next_segment_start() override;
    std::uint64_t current() const override;

    // Stone 6.5: atomically reserve a CHUNK of `chunk_segments` contiguous
    // segments and return the base offset of the chunk.  The returned value
    // is the start of `chunk_segments * segment_size` bytes of cursor space
    // owned exclusively by the calling thread — no other thread will draw any
    // segment in [base, base + chunk_segments * segment_size).  This is the
    // contract that lets the chunk-internal Sieve::sieve_segment() calls
    // advance the wheel safely (contiguous-segment assumption holds within
    // the chunk).
    //
    // Implemented as a single atomic fetch_add with the chunk-sized stride —
    // wait-free, no mutex.  `chunk_segments` must be > 0; passing 0 is a
    // programmer error and yields undefined-but-safe behaviour (returns the
    // current cursor without advancing it).
    std::uint64_t next_segment_chunk(std::uint64_t chunk_segments);

private:
    std::uint64_t m_segment_size;
    std::atomic<std::uint64_t> m_cursor{0};
};

// Factory: returns the allocator implied by Worker_config_cpu::m_engine_mode.
// Today this is always a Per_worker_segment_allocator; the "engine" mode is
// reserved/forward-compatible and currently falls back to the per-worker
// allocator (logged at construction).
std::unique_ptr<Segment_allocator>
make_segment_allocator_for_engine_mode(const std::string& engine_mode,
                                       std::uint64_t segment_size);

} // namespace cpu
} // namespace nexusminer

#endif
