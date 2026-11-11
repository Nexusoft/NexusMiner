#ifndef NEXUSMINER_WORKER_HPP
#define NEXUSMINER_WORKER_HPP

#include <memory>
#include <functional>
#include <optional>
#include <vector>

#if defined(__has_include)
#if __has_include(<boost/multiprecision/cpp_int.hpp>)
#define NEXUSMINER_HAS_BOOST_MULTIPRECISION 1
#else
#define NEXUSMINER_HAS_BOOST_MULTIPRECISION 0
#endif
#else
#define NEXUSMINER_HAS_BOOST_MULTIPRECISION 0
#endif

#if NEXUSMINER_HAS_BOOST_MULTIPRECISION
#include <boost/multiprecision/cpp_int.hpp>
#endif
#include "LLC/types/uint1024.h"
#include "block.hpp"
#include "worker/block_header_utils.hpp"

namespace nexusminer {
namespace stats { class Collector; }
class WorkerTemplateFeed;  // Stone 4: shared template publication; defined in worker/template_feed.hpp

class Block_data
{
public:

	Block_data(const ::LLP::CBlock& block)
		:merkle_root{block.hashMerkleRoot}
		, previous_hash{block.hashPrevBlock}
		, nHeight {block.nHeight}
		, nVersion {block.nVersion}
		, nChannel {block.nChannel}
		, nBits {block.nBits}
		, nNonce{ block.nNonce }{}

	Block_data() {}

	std::vector<unsigned char> GetHeaderBytes(bool excludeNonce = false)
	{
		return GetBlockHeaderBytes(ToBlock(), excludeNonce);
	}

#if NEXUSMINER_HAS_BOOST_MULTIPRECISION
	boost::multiprecision::uint1024_t GetPrimeBaseHash() const
	{
		const auto proof_hash = GetPrimeProofHash(ToBlock());
		return boost::multiprecision::uint1024_t("0x" + proof_hash.GetHex());
	}
#endif
	//The order of the block header data below matters for the cuda miner.  Be careful.
	uint32_t nVersion = 4;
    uint1024_t previous_hash;
	uint512_t merkle_root;
	uint32_t nChannel = 2;
	uint32_t nHeight = 2023276;
	uint32_t nBits = 0x7b032ed8;
	uint64_t nNonce = 21155560019;

	// Prime channel offsets (Cunningham chain offsets from ValidatePrimeCandidate).
	// Empty for Hash channel. CPU and GPU prime workers both populate these from
	// the shared prime_validation.cpp pathway before firing the callback so that
	// worker_manager can forward the exact solved snapshot to
	// prepare_block_submission_from_solved(...).
	std::vector<uint8_t> vOffsets;

private:
	::LLP::CBlock ToBlock() const
	{
		::LLP::CBlock block;
		block.nVersion = nVersion;
		block.hashPrevBlock = previous_hash;
		block.hashMerkleRoot = merkle_root;
		block.nChannel = nChannel;
		block.nHeight = nHeight;
		block.nBits = nBits;
		block.nNonce = nNonce;
		return block;
	}

};

/**
 * WorkPackage: Immutable shared work data for template distribution
 *
 * Created once per template by Worker_manager and shared across all workers
 * via std::shared_ptr. Eliminates repeated Block_data construction and
 * serialization per worker.
 *
 * Contains precomputed immutable data:
 * - Original CBlock from protocol layer
 * - nBits (difficulty target)
 * - Precomputed header bytes (208 or 216 bytes depending on nonce inclusion)
 * - Optional precomputed base hash for prime workers (Skein + Keccak of header)
 *
 * Workers maintain per-worker mutable state (starting nonce, Block_data copy).
 */
class WorkPackage
{
public:
#if NEXUSMINER_HAS_BOOST_MULTIPRECISION
    using uint1k = boost::multiprecision::uint1024_t;
#endif

    WorkPackage(const ::LLP::CBlock& block, std::uint32_t nbits)
        : m_block{block}
        , m_nbits{nbits}
    {
        // Precompute header bytes once for all workers
        // Note: Block_data constructor copies from CBlock
        Block_data temp_block_data{block};
        m_header_bytes = temp_block_data.GetHeaderBytes();
    }

    // Immutable accessors
    const ::LLP::CBlock& get_block() const { return m_block; }
    std::uint32_t get_nbits() const { return m_nbits; }
    const std::vector<unsigned char>& get_header_bytes() const { return m_header_bytes; }

    // Optional precomputed base hash for prime workers
    // Set via set_prime_base_hash() after construction for prime channel
#if NEXUSMINER_HAS_BOOST_MULTIPRECISION
    const std::optional<uint1k>& get_prime_base_hash() const { return m_prime_base_hash; }

    // Setter for prime base hash (called only for prime channel blocks)
    void set_prime_base_hash(const uint1k& base_hash) { m_prime_base_hash = base_hash; }
#endif

private:
    ::LLP::CBlock m_block;                      // Original block from protocol
    std::uint32_t m_nbits;                      // Difficulty target
    std::vector<unsigned char> m_header_bytes;  // Precomputed header (shared)
#if NEXUSMINER_HAS_BOOST_MULTIPRECISION
    std::optional<uint1k> m_prime_base_hash;    // Precomputed base hash for prime (Skein+Keccak)
#endif
};

class Worker {
public:

	virtual ~Worker() = default;

    // A call to the BlockFoundHandler informs the user about a new found block.
    using Block_found_handler = std::function<void(std::uint32_t id, std::unique_ptr<Block_data>&& block)>;

    // Sets a new block (nexus data type) for the miner worker. The miner worker must reset the current work.
    // When  the worker finds a new block, the BlockFoundHandler has to be called with the found BlockData
    // DEPRECATED: Use set_block(shared_ptr<WorkPackage>, Block_found_handler) for better performance
    virtual void set_block(::LLP::CBlock block, std::uint32_t nbits, Block_found_handler result) = 0;

    // Optimized version: accepts shared WorkPackage to eliminate repeated block data construction
    // Default implementation delegates to legacy set_block() for backward compatibility
    virtual void set_block(std::shared_ptr<WorkPackage> work_package, Block_found_handler result)
    {
        // Default fallback: extract block and nbits from WorkPackage
        set_block(work_package->get_block(), work_package->get_nbits(), result);
    }

    // Stone 4: Attach a shared WorkerTemplateFeed for lock-free template
    // publication.  Default is a no-op so subclasses can opt in incrementally:
    // until a worker overrides this and starts consuming epochs from the feed
    // in its mining loop, Worker_manager keeps feeding it via the legacy
    // set_block(WorkPackage,...) shim (now invoked outside m_worker_mutex).
    // The feed pointer must outlive the worker; Worker_manager owns one feed
    // per worker batch and tears workers down before releasing the feed.
    //
    // CONTRACT FOR OVERRIDES (Stones 5–7 worker migrations, READ THIS):
    //
    //   This method is invoked by Worker_manager::create_workers_locked()
    //   AFTER every worker constructor has returned.  Several existing worker
    //   subclasses (Worker_hash, future migrated Worker_prime, …) start their
    //   mining thread inside the constructor, so by the time attach_template_feed
    //   runs the consumer thread is ALREADY RUNNING.  An override that simply
    //   stores the feed in a non-atomic member therefore races with the worker
    //   loop reading it.
    //
    //   Overrides MUST therefore:
    //     1. Store the feed in a thread-safe slot
    //        (e.g. std::atomic<std::shared_ptr<WorkerTemplateFeed>>) and use
    //        acquire/release ordering so the consumer can publish the pointer
    //        to its own loop without a mutex.
    //     2. Treat a not-yet-attached feed as "no work yet" — the worker loop
    //        must idle (or fall back to the legacy m_new_work CV) until the
    //        atomic load returns a non-null pointer, NOT crash on null deref.
    //     3. Be idempotent: Worker_manager calls attach_template_feed() exactly
    //        once per batch today, but the contract permits re-attachment on
    //        future restarts.
    //
    //   The cleaner long-term shape (deferred to a follow-up) is constructor
    //   injection of the feed into migrated workers so the thread can never
    //   start before the feed is bound.  Until then, the atomic-slot rule
    //   above is the supported pattern.
    virtual void attach_template_feed(std::shared_ptr<WorkerTemplateFeed> /*feed*/) {}

    // Stone 4: Opt-in marker so Worker_manager can skip the per-worker
    // set_block() shim once a subclass has been migrated to consume work
    // exclusively from the WorkerTemplateFeed.  Default false preserves
    // existing fanout behaviour for unmigrated workers.
    virtual bool uses_template_feed() const { return false; }

    // Returns true if the worker's mining thread is actively running (i.e. set_block() started it).
    // Implementations backed by an m_stop atomic should override this to return !m_stop.
    // The default returns true for async I/O workers (e.g. FPGA) that have no explicit stop flag.
    // NOTE: All thread-based worker subclasses MUST override this to return !m_stop so that
    // the workers_fed counter only counts threads that were actually started by set_block().
    virtual bool is_running() const { return true; }

    virtual void update_statistics(stats::Collector& stats_collector) = 0;
};

}


#endif
