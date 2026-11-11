#ifndef CHAIN_SIEVE_HPP
#define CHAIN_SIEVE_HPP

#include <vector>
#include <atomic>
#include <algorithm>
#include <numeric>
#include <spdlog/spdlog.h>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/multiprecision/gmp.hpp>
#include "sieve_utils.hpp"
#include "mining/mining_constants.hpp"
#include "mining/prime_thresholds.hpp"

namespace nexusminer {
	namespace cpu
	{
		enum class Fermat_test_status {
			untested,
			fail,
			pass
		};

		static constexpr int maxGap = 12;  //the largest allowable prime gap.

		//a candidate for a dense prime cluster.  A chain consists of a base integer plus a list of offsets. 
		class Chain
		{
		public:
			// Stone 6.9 — once the chain has accumulated this many proven
			// Fermat primes, is_there_still_hope() returns true unconditionally
			// so the histogram's upper buckets keep growing (matches GPU's
			// cuda_chain.cu behaviour: "if we've already found 4, keep
			// counting regardless").  Public so tests can lock down the value
			// without duplicating the magic number.
			static constexpr int kHopeKeepaliveThreshold = 4;

			enum class Chain_state {
				open, //immature chain empty or in process of being built
				closed, //chain is complete and available for fermat testing
				in_process, //fermat testing is in process. 
				complete  //fermat testing is complete. 
			};

			class Chain_offset {
			public:
				int m_offset = 0;  //offset from the base offset
				Fermat_test_status m_fermat_test_status = Fermat_test_status::untested;
				Chain_offset(int offset) :m_offset{ offset }, m_fermat_test_status{ Fermat_test_status::untested }{};
			};

			Chain();
			Chain(uint64_t base_offset);
			void open(uint64_t base_offset);
			void close();
			int length() { return m_offsets.size(); }
			void get_best_fermat_chain(uint64_t& base_offset, int& offset, int& length);
			bool is_there_still_hope();  //is it possible this chain can result in a valid fermat chain
			bool get_next_fermat_candidate(uint64_t& base_offset, int& offset);
			bool update_fermat_status(bool is_prime);
			void push_back(int offset);
			const std::string str();

			int m_min_chain_length = 8;
			// Aligned with GPU (cuda_chain.cuh / gpu chain.hpp).  The CPU value used
			// to be 4, which caused length-4 candidates to be pushed into
			// m_long_chain_starts and silently rejected later by the difficulty
			// gate in the dispatch loop.  Engine mode now overwrites this per
			// session via Sieve::set_target_length(); 5 is just a safer default
			// for any caller that bypasses set_target_length().
			int m_min_chain_report_length = 5;
			Chain_state m_chain_state = Chain_state::open;
			uint64_t m_base_offset = 0;
			std::vector<Chain_offset> m_offsets; //offsets including 0
			int m_gap_in_process = 0;
			int m_next_fermat_test_offset_index = 0;
			int m_prime_count = 0;
			int m_untested_count = 0;

		private:

		};

		class Sieve
		{
		public:
			Sieve();

			// Stone 1: kept for backward compatibility — first call resolves the
			// process-shared sieving prime table and sizes the per-worker mutable
			// wheel state (m_prime_state) accordingly.  Subsequent calls are
			// cheap (no re-generation, no extra allocation).
			void generate_sieving_primes();

			// Stone 2: prefer prepare(sieve_start).  set_sieve_start() still
			// rounds the start to a multiple of 30 and stores it as a fallback
			// for legacy callers that go on to invoke the no-arg overloads.
			void set_sieve_start(boost::multiprecision::uint1024_t);
			boost::multiprecision::uint1024_t get_sieve_start();

			// Stone 2: parameterised entry points.  These never read m_sieve_start;
			// the caller threads the base hash explicitly.  The no-arg overloads
			// below remain for legacy callers and forward to the cached value.
			//
			// prepare() bundles set_sieve_start + clear_chains +
			// calculate_starting_multiples and returns the rounded start that the
			// sieve actually uses (callers compare against the requested start to
			// adjust their nonce bookkeeping).
			boost::multiprecision::uint1024_t prepare(boost::multiprecision::uint1024_t sieve_start);
			// Overload that ALSO sets the per-session target chain length in one
			// call so the set-then-use ordering hazard cannot occur.  A
			// target_length <= 0 means "use the legacy default" (mining::MIN_CHAIN_LENGTH).
			boost::multiprecision::uint1024_t prepare(boost::multiprecision::uint1024_t sieve_start,
			                                          int target_length);
			// Set the per-session target Cunningham chain length.  Drives:
			//   * Sieve::close_chain  — minimum slot count to keep a chain candidate
			//   * Chain::is_there_still_hope  — early-abort threshold during Fermat testing
			//   * Sieve::test_chains / clean_chains — gate for pushing into m_long_chain_starts
			// Values < 2 are clamped to 2 (a single-prime "chain" is meaningless to dispatch).
			// Default after construction is mining::MIN_CHAIN_LENGTH (8) — preserves
			// pre-Stone behaviour for legacy callers (Worker_prime).
			void set_target_length(int target_length);
			int  get_target_length() const { return m_target_chain_length; }

			// Sieve-stage filter thresholds.  Both delegate to the shared
			// single-source-of-truth helpers in mining/prime_thresholds.hpp
			// so the CPU and GPU paths cannot drift apart.
			//
			// Two SEPARATELY-NAMED accessors (not one shared `slot_filter_min`)
			// because the two filters have fundamentally different semantics:
			//
			//   * popcount_window_floor() — necessary lower bound on the
			//     density of a 120-integer sieve window.  A window with fewer
			//     than T survivors cannot host a length-T chain; this is the
			//     provably-correct early-exit threshold.  Anything stricter
			//     would discard windows that could have produced a winner.
			//
			//   * close_chain_min() — minimum sieve-survivor slot count for an
			//     ASSEMBLED chain candidate to be kept for Fermat testing.
			//     Same correctness rationale — a chain of exactly T survivors
			//     where every slot passes Fermat is a winner, so we cannot
			//     reject it.
			//
			// Conflating these two thresholds (as the pre-fix `slot_filter_min`
			// did) hid PR #672's regression: bumping the close_chain heuristic
			// also tightened the popcount filter, silently dropping ~90% of
			// the candidate windows that could have produced length-T chains.
			// See mining/prime_thresholds.hpp for the full post-mortem.
			int popcount_window_floor() const noexcept
			{
				return mining::popcount_window_floor(m_target_chain_length);
			}
			int close_chain_min() const noexcept
			{
				return mining::close_chain_min(m_target_chain_length);
			}
			void calculate_starting_multiples(const boost::multiprecision::uint1024_t& sieve_start);
			void calculate_starting_multiples();
			void test_chains(const boost::multiprecision::uint1024_t& sieve_start);
			void test_chains();

			void sieve_segment();
			void sieve_batch(uint64_t low);
			void sieve_batch_cpu(uint64_t low);
			// Stone 7: returns the compile-time segment size constant.
			// Made static + constexpr so callers (e.g. Worker_manager
			// during PrimeMiningEngine wiring) can read it without
			// instantiating a Sieve — Sieve's constructor allocates
			// non-trivial buffers.  Existing instance-style callers
			// (sieve->get_segment_size()) continue to compile.
			static constexpr std::uint32_t get_segment_size() { return m_segment_size; }
			std::uint32_t get_segment_batch_size();
			void reset_sieve();
			void reset_sieve_batch(uint64_t low);
			void clear_chains();
			void reset_stats();
			void find_chains(uint64_t low, bool batch_sieve_mode);
			uint64_t count_fermat_primes(uint64_t sieve_size, uint64_t low);
			bool primality_test(boost::multiprecision::uint1024_t p);
			void primality_batch_test();
			void primality_batch_test_cpu();
			void clean_chains();
			uint64_t get_current_chain_list_length();
			int get_fermat_test_batch_size() { return m_fermat_test_batch_size; }
			std::vector<std::uint64_t> m_long_chain_starts;
			uint64_t m_sieve_batch_start_offset;

			//stats
			std::vector<std::uint32_t> m_chain_histogram;

			// Stone 6.9 — "attempted" companion to m_chain_histogram.  Bumped
			// once per chain candidate that survived close_chain()'s slot
			// filter, indexed by the candidate's sieve-survivor slot count
			// (m_offsets.size()) at the moment we started Fermat-testing it.
			//
			// Why this exists: m_chain_histogram only records the BEST Fermat
			// run actually achieved for each chain.  When is_there_still_hope()
			// aborts a chain early, that chain still appears in
			// m_chain_histogram (with a small best-run length), so operators
			// cannot tell from m_chain_histogram alone whether a 7:0 cell is
			// "we never got a length-7 candidate" vs "we got 200 of them and
			// the early-abort predicate killed every one before length 7".
			// The attempted histogram disambiguates: bucket k now means
			// "this many chain candidates had >= k sieve-survivor slots
			// available to Fermat-test".
			std::vector<std::uint32_t> m_chain_histogram_attempted;

			// Stone 6.8: read-side snapshot of m_chain_histogram for cross-thread
			// readers (the PrimeMiningEngine pool-thread fan-in path).  Returns a
			// by-value copy so callers never observe a torn vector mid-resize.
			//
			// Calling contract (relied on by PrimeMiningEngine::snapshot_stats):
			//   * The Sieve's m_chain_histogram is sized exactly ONCE — in
			//     reset_stats(), invoked from the Sieve constructor — and never
			//     resized again for the lifetime of the Sieve.  The engine pool
			//     thread does NOT call reset_stats() during sieving (find_chains
			//     only does in-place bucket increments), so the vector's
			//     size/capacity / data pointer are stable against an external
			//     reader once the Sieve has published its first
			//     Pool_histogram_snapshot via publish_histogram_snapshot().
			//   * Bucket increments are non-atomic uint32_t fetch-add writes,
			//     which on x86_64 / ARMv8 cannot tear a single 32-bit word; the
			//     worst observable case is a slightly stale bucket count, which
			//     is acceptable for a diagnostic histogram.
			// Callers outside the engine fan-in path that DO resize the
			// histogram (legacy Worker_prime same-thread snapshot) must call
			// from the Sieve-owning thread; cross-thread callers must NOT call
			// reset_stats()/clear_chains() concurrently.
			std::vector<std::uint32_t> snapshot_chain_histogram() const
			{
				return m_chain_histogram;
			}
			// Same calling-contract as snapshot_chain_histogram (see above):
			// the vector is sized once in reset_stats() and only ever in-place
			// fetch-add'd thereafter, so an external reader can copy it
			// without observing a torn resize.
			std::vector<std::uint32_t> snapshot_chain_histogram_attempted() const
			{
				return m_chain_histogram_attempted;
			}
			uint64_t m_fermat_test_count = 0;
			uint64_t m_fermat_prime_count = 0;
			uint64_t m_chain_count = 0;
			int m_chain_candidate_max_length = 0;
			uint64_t m_chain_candidate_total_length = 0;
			double m_best_chain = 0;

			// ── Diagnostic counters: written ONLY by mining thread, read ONLY by stats thread ──
			// std::atomic so the stats collector thread can safely read without a mutex lock.
			// The mining thread does only fetch_add (relaxed); the stats thread does
			// .load(std::memory_order_relaxed) — no synchronisation needed beyond atomicity.
			std::atomic<uint64_t> m_diag_sieve_calls{0};   // total sieve_segment() calls since last reset
			std::atomic<uint64_t> m_diag_inner_hits{0};    // total inner-loop sieve-write hits since last reset
			std::atomic<uint64_t> m_diag_starting_multiples_us{0};  // µs spent in calculate_starting_multiples (no sort since Stone 1)
			std::atomic<uint32_t> m_diag_prime_count{0};   // count of sieving primes in m_primes_aos

			// Stone 6.9 — find→test→dispatch funnel diagnostics.  Bumped by the
			// owning sieve thread; read by the engine stats fan-in.
			//   * m_diag_chain_candidates_found — chains that survived
			//     close_chain()'s slot filter (i.e. m_chain.push_back fired).
			//   * m_diag_chains_started_fermat — chains we actually entered the
			//     Fermat-test loop for in test_chains() (those for which we
			//     called get_next_fermat_candidate at least once).
			//   * m_diag_chains_pushed_long  — chains pushed onto
			//     m_long_chain_starts (i.e. best Fermat run >= target_length).
			std::atomic<uint64_t> m_diag_chain_candidates_found{0};
			std::atomic<uint64_t> m_diag_chains_started_fermat{0};
			std::atomic<uint64_t> m_diag_chains_pushed_long{0};

			// Stone 6.9.2 — popcount-stage funnel observability.  Bumped once
			// per 4-byte sieve window that survived popcount_window_floor()
			// (i.e. that the wheel walk in find_chains() actually entered).
			// Paired with m_diag_chain_candidates_found to give operators
			// stage-by-stage visibility:
			//
			//   popcount_windows_passed  → close_chain stage
			//   close_chain candidates   → Fermat stage
			//   chains_pushed_long       → dispatch stage
			//
			// A future regression that tightens either filter beyond its
			// provably-correct lower bound now shows up as a step-down at
			// the appropriate stage rather than collapsing into the single
			// "chains_found_by_sieve" counter (which is what let PR #672's
			// regression go unnoticed for so long).
			std::atomic<uint64_t> m_diag_popcount_windows_passed{0};

		private:
			// Stone 6.9 — per-session target Cunningham chain length.  Default
			// preserves legacy Worker_prime behaviour (mining::MIN_CHAIN_LENGTH).
			// Engine mode rewrites this on every session bind.  Read by the
			// owning thread only (close_chain / test_chains / clean_chains
			// / open_chain).
			int m_target_chain_length{mining::MIN_CHAIN_LENGTH};

			class Fermat_test_candidate {
			public:
				uint64_t base_offset = 0;
				int offset = 0;
				Fermat_test_status fermat_test_status = Fermat_test_status::untested;
				uint64_t get_offset() { return base_offset + offset; }
			};

			std::shared_ptr<spdlog::logger> m_logger;

			static constexpr uint8_t sieve30 = 0xFF;  //compressed sieve for primorial 2*3*5 = 30.  Each bit represents a possible prime location in the wheel {1,7,11,13,17,19,23,29} 
			static constexpr int sieve30_offsets[]{ 1,7,11,13,17,19,23,29 };  // each bit in the sieve30 represets an offset from the base mod 30
			static constexpr int sieve30_gaps[]{ 6,4,2,4,2,4,6,2 };
			static constexpr int sieve30_index[]{ -1,0,-1,-1,-1,-1,-1, 1, -1, -1, -1, 2, -1, 3, -1, -1, -1, 4, -1, 5, -1, -1, -1, 6, -1, -1, -1, -1, -1, 7 };  //reverse lookup table (offset mod 30 to index)
			static constexpr int L1_CACHE_SIZE = mining::L1_CACHE_BYTES;
			static constexpr int L2_CACHE_SIZE = mining::L2_CACHE_BYTES;
			//upper limit of the sieving range
			//static constexpr uint64_t sieve_range = 3e9;//3e9;
			//upper limit of the sieving primes. 
			static constexpr uint32_t sieving_prime_limit = mining::CPU_SIEVING_PRIME_LIMIT;
			static constexpr uint32_t sieve_size = L2_CACHE_SIZE * 16;
			//each segment byte covers a range of 30 sieving primes 
			static constexpr uint32_t m_segment_size = sieve_size * 30;
			//number of segments needed to cover the sieving range
			//static constexpr int segments = sieve_range / m_segment_size + (sieve_range % m_segment_size != 0);
			//we start sieving at 7
			static constexpr int sieving_start_prime = mining::SIEVING_START_PRIME;
			static constexpr int m_min_chain_length = mining::MIN_CHAIN_LENGTH;

			/// Bitmasks used to unset bits
			static constexpr uint8_t unset_bit_mask[30] =
			{
				(uint8_t)~(1 << 0), (uint8_t)~(1 << 0),
				(uint8_t)~(1 << 1), (uint8_t)~(1 << 1), (uint8_t)~(1 << 1), (uint8_t)~(1 << 1), (uint8_t)~(1 << 1), (uint8_t)~(1 << 1),
				(uint8_t)~(1 << 2), (uint8_t)~(1 << 2), (uint8_t)~(1 << 2), (uint8_t)~(1 << 2),
				(uint8_t)~(1 << 3), (uint8_t)~(1 << 3),
				(uint8_t)~(1 << 4), (uint8_t)~(1 << 4), (uint8_t)~(1 << 4), (uint8_t)~(1 << 4),
				(uint8_t)~(1 << 5), (uint8_t)~(1 << 5),
				(uint8_t)~(1 << 6), (uint8_t)~(1 << 6), (uint8_t)~(1 << 6), (uint8_t)~(1 << 6),
				(uint8_t)~(1 << 7), (uint8_t)~(1 << 7), (uint8_t)~(1 << 7), (uint8_t)~(1 << 7), (uint8_t)~(1 << 7), (uint8_t)~(1 << 7)
			};

			//how many bits are set in a byte
			static constexpr int popcnt[256] =
			{
			  0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
			  1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
			  1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
			  2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
			  1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
			  2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
			  2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
			  3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
			  1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
			  2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
			  2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
			  3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
			  2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
			  3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
			  3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
			  4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8
			};

			/// Stone 1: per-worker mutable wheel state for one sieving prime.
			/// The prime *value* lives in the process-shared
			/// Sieving_prime_table — only the per-worker (multiple, wheel_index)
			/// pair is duplicated across workers.  Order matches
			/// Sieving_prime_table::primes() (large-prime-first).
			struct SievePrimeState {
				uint32_t multiple;     ///< current multiple (updated each segment)
				int32_t  wheel_index;  ///< index into sieve30_offsets[] / unset_bit_mask[]
			};

			/// Per-wheel-position step sizes precomputed once per sieving prime.
			/// byte_delta   = (k * gap) / 30  — whole sieve-bytes to advance
			/// offset_delta = (k * gap) % 30  — sub-byte remainder
			struct WheelStep {
				uint32_t byte_delta;
				uint8_t  offset_delta;
			};

			//the sieve.  each bit that is set represents a possible prime.
			std::vector<uint8_t> m_sieve;
			// Stone 1: per-worker mutable wheel state, indexed parallel to
			// Sieving_prime_table::instance().primes().  The prime values
			// themselves are not duplicated here.
			std::vector<SievePrimeState> m_prime_state;
			std::vector<Chain> m_chain;
			std::vector<uint8_t> m_sieve_results;  //accumulated results of sieving
			boost::multiprecision::uint1024_t m_sieve_start;  //starting integer for the sieve.  This must be a multiple of 30.
			bool m_chain_in_process = false;
			Chain m_current_chain;
			static constexpr int m_fermat_test_batch_size = 100;
			static constexpr int m_segment_batch_size = 1; //number of segments to batch process
			static constexpr int m_sieve_batch_buffer_size = sieve_size * m_segment_batch_size;
			void close_chain();
			void open_chain(uint64_t base_offset);
		};
	}
}

#endif