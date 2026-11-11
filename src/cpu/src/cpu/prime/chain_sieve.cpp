#include "chain_sieve.hpp"
#include "sieving_prime_table.hpp"
#include <primesieve.hpp>
#include <vector>
#include <queue>
#include <chrono>
#include <bitset>
#include <sstream>
#include <boost/integer/mod_inverse.hpp>

namespace nexusminer {
    namespace cpu
    {
        using namespace boost::multiprecision;
        Chain::Chain()
        {
        }
        Chain::Chain(uint64_t base_offset)
        {
            open(base_offset);
        }
        void Chain::open(uint64_t base_offset)
        {
            m_base_offset = base_offset;
            m_chain_state = Chain_state::open;
            push_back(0);  //the first offset is always zero
            m_gap_in_process = 0;
            m_prime_count = 0;
        }
        void Chain::close()
        {
            m_chain_state = Chain_state::closed;
        }

        //analyze the chain fermat test results.  
        //return the starting offset and length of the longest fermat chain that meets the mininmum gap requirement
        void Chain::get_best_fermat_chain(uint64_t& base_offset, int& offset, int& best_length)
        {
            base_offset = m_base_offset;
            offset = 0;
            int chain_length = 0;
            best_length = 0;
            if (length() == 0)
                return;

            int gap = 0;
            int starting_offset = 0;
            auto previous_offset = m_offsets[0].m_offset;
            for (int i = 0; i < m_offsets.size(); i++)
            {
                if (chain_length > 0)
                    gap += m_offsets[i].m_offset - previous_offset;
                if (gap > maxGap)
                {
                    //end of the fermat chain
                    if (chain_length > best_length)
                    {
                        best_length = chain_length;
                        offset = starting_offset;
                        chain_length = 0;
                        gap = 0;
                    }
                }
                if (m_offsets[i].m_fermat_test_status == Fermat_test_status::pass)
                {
                    chain_length++;
                    gap = 0;
                    if (chain_length == 1)
                    {
                        starting_offset = m_offsets[i].m_offset;
                    }
                }
                previous_offset = m_offsets[i].m_offset;

            }
            if (chain_length > best_length)
            {
                best_length = chain_length;
                offset = starting_offset;
            }
            return;
        }

        //return true if there is more testing we can do. returns false if we should give up.
        bool Chain::is_there_still_hope()
        {
            //nothing left to test
            if (m_untested_count == 0)
            {
                return false;
            }

            // Stone 6.9 — Mirror the GPU implementation in cuda_chain.cu.
            //
            // (a) Stats keepalive: once we've already proven N (== 4) Fermat
            //     primes in this chain, keep testing regardless of the cheap
            //     totals/contiguous predicates.  This is what populates the
            //     bucket-6 / bucket-7 cells of the chain histogram (without
            //     it those cells are deterministically empty whenever the
            //     target length is > 4).  GPU has shown this has negligible
            //     throughput cost.
            if (m_prime_count >= kHopeKeepaliveThreshold)
            {
                return true;
            }

            // (b) Cheap upper-bound prune on totals.  If even assuming every
            //     untested slot passes Fermat we can't reach m_min_chain_length
            //     primes total, give up.  Necessary but not sufficient.
            const int upper_bound = m_prime_count + m_untested_count;
            if (upper_bound < m_min_chain_length)
            {
                return false;
            }

            // (c) Fast path — no failures yet, so every slot is either pass
            //     or untested.  The chain is still contiguous (the sieve
            //     guarantees the slots themselves are within maxGap; failures
            //     are the only way contiguity is broken), so the totals
            //     bound IS exact and the expensive Chain-copy walk in (d)
            //     would just confirm it.  Skipping the allocation here is
            //     measurable: prior to the keepalive most chains take this
            //     path before the first failure.
            const int failure_count = static_cast<int>(m_offsets.size())
                                    - m_prime_count - m_untested_count;
            if (failure_count == 0)
            {
                return true;  // upper_bound >= m_min_chain_length already checked
            }

            // (d) Tighter prune that respects the predicate/scorer agreement
            //     we want: the scorer (get_best_fermat_chain) only credits
            //     contiguous, gap-bounded runs, but the totals predicate above
            //     ignores both the gap and the already-broken-up structure of
            //     the chain.  So a chain like
            //         pass · fail · pass · untested · untested · untested
            //     can pass (b) (1+3 >= 4) yet be unable to ever produce a
            //     contiguous run of length m_min_chain_length.  Resolve that
            //     by simulating "every untested slot passes" through the
            //     existing scorer and asking whether the resulting longest
            //     contiguous run reaches the target.  This is the
            //     dead-coded approach left in place by previous developers;
            //     it now ships.
            Chain temp_chain(*this);
            for (auto& slot : temp_chain.m_offsets)
            {
                if (slot.m_fermat_test_status == Fermat_test_status::untested)
                {
                    slot.m_fermat_test_status = Fermat_test_status::pass;
                }
            }
            int max_possible_length = 0;
            int dummy_offset = 0;
            uint64_t dummy_base_offset = 0;
            temp_chain.get_best_fermat_chain(dummy_base_offset, dummy_offset,
                                             max_possible_length);
            return (max_possible_length >= m_min_chain_length);
        }

        //get the next untested fermat candidate.  if there are none return false.
        bool Chain::get_next_fermat_candidate(uint64_t& base_offset, int& offset)
        {
            //This returns the next untested prime candidate.
            //There are other more complex ways to do this to minimize primality testing
            //like search for the first candidate that busts the chain if it fails
            for (auto i = 0; i < m_offsets.size(); i++)
            {
                if (m_offsets[i].m_fermat_test_status == Fermat_test_status::untested)
                {
                    base_offset = m_base_offset;
                    offset = m_offsets[i].m_offset;
                    //save the offset under test index for later
                    m_next_fermat_test_offset_index = i;
                    return true;
                }
            }
            return false;
        }

        //set the fermat test status of an offset.  if the offset is not found return false.
        bool Chain::update_fermat_status(bool is_prime)
        {
            m_untested_count--;
            if (is_prime)
            {
                m_offsets[m_next_fermat_test_offset_index].m_fermat_test_status = Fermat_test_status::pass;
                m_prime_count++;
            }
            else
            {
                m_offsets[m_next_fermat_test_offset_index].m_fermat_test_status = Fermat_test_status::fail;
            }

            return true;

        }

        void Chain::push_back(int offset)
        {
            Chain_offset chain_offset{ offset };
            m_offsets.push_back(chain_offset);
            m_untested_count++;
            //m_offset_map[offset] = m_offsets.size();
        }
		
		//create a string with information about the chain
        const std::string Chain::str()
        {
            std::stringstream ss;
            uint64_t base_offset;
            int offset, best_length;
            get_best_fermat_chain(base_offset, offset, best_length);
            ss << "len " << best_length << "/" << length() << " " << m_prime_count << "p/" << m_untested_count
                << "u best_start:" << offset << " test_next:" << m_next_fermat_test_offset_index << " ";
            ss << m_base_offset << " + ";
            for (const auto& x : m_offsets)
            {
                ss << x.m_offset;
                std::string test_status = "?";
                if (x.m_fermat_test_status == Fermat_test_status::pass)
                    test_status = "*";
                else if (x.m_fermat_test_status == Fermat_test_status::fail)
                    test_status = "x";
                ss << test_status << " ";
            }
            return ss.str();
        }

        Sieve::Sieve()
            : m_logger{ spdlog::get("logger") }
        {
            m_sieve.resize(sieve_size);
            reset_stats();
			reset_sieve_batch(0);
        }

        void Sieve::generate_sieving_primes()
        {
            // Stone 1: the prime list lives in a process-wide singleton.  We
            // only need to (re)size our parallel mutable wheel state to match.
            const auto& shared = Sieving_prime_table::instance().primes();
            if (m_prime_state.size() != shared.size())
            {
                m_prime_state.assign(shared.size(), SievePrimeState{0, 0});
            }
            else
            {
                std::fill(m_prime_state.begin(), m_prime_state.end(),
                          SievePrimeState{0, 0});
            }
        }

        void Sieve::set_sieve_start(boost::multiprecision::uint1024_t sieve_start)
        {
            //set the sieve start to a multiple of 30
            if (sieve_start % 30 > 0)
            {
                sieve_start += 30 - (sieve_start % 30);
            }
            m_sieve_start = sieve_start;
        }

        boost::multiprecision::uint1024_t Sieve::get_sieve_start()
        {
            return m_sieve_start;
        }

        boost::multiprecision::uint1024_t Sieve::prepare(
            boost::multiprecision::uint1024_t sieve_start)
        {
            // Stone 2: single entry point that bundles align-to-30 +
            // clear_chains + calculate_starting_multiples and returns the
            // rounded start so callers can adjust their nonce bookkeeping.
            set_sieve_start(sieve_start);
            clear_chains();
            calculate_starting_multiples(m_sieve_start);
            return m_sieve_start;
        }

        boost::multiprecision::uint1024_t Sieve::prepare(
            boost::multiprecision::uint1024_t sieve_start, int target_length)
        {
            // Stone 6.9 — set target length BEFORE clear_chains so any future
            // chain we open during the next find_chains() call inherits the
            // correct per-session m_min_chain_length.  No set-then-use
            // ordering hazard.
            set_target_length(target_length);
            return prepare(sieve_start);
        }

        void Sieve::set_target_length(int target_length)
        {
            // A single-prime "chain" is meaningless to dispatch; clamp to 2.
            // 0 / negative is treated as "use legacy default".
            if (target_length <= 0)
            {
                m_target_chain_length = mining::MIN_CHAIN_LENGTH;
            }
            else
            {
                m_target_chain_length = std::max(2, target_length);
            }
        }

        void Sieve::calculate_starting_multiples()
        {
            // Stone 2: legacy no-arg overload — forwards to the cached value
            // populated by set_sieve_start().
            calculate_starting_multiples(m_sieve_start);
        }

        void Sieve::calculate_starting_multiples(
            const boost::multiprecision::uint1024_t& sieve_start)
        {
            //generate starting multiples of the sieving primes
            // Stone 6.5: demoted from info to debug.  Under engine_mode the
            // PrimeMiningEngine pool path calls calculate_starting_multiples()
            // once per chunk per pool thread, which on an 18-core box can
            // still fire many times per second.  Under legacy workers mode it
            // fires once per template (correct cadence at info level).  Debug
            // keeps the diagnostic available without flooding production logs.
            m_logger->debug("Calculating starting multiples.");
            const auto& shared_primes = Sieving_prime_table::instance().primes();
            // Defensive: if generate_sieving_primes() was not called yet, do it
            // now so the parallel arrays line up.
            if (m_prime_state.size() != shared_primes.size())
            {
                m_prime_state.assign(shared_primes.size(), SievePrimeState{0, 0});
            }

            // With Stone 1 the prime list is pre-sorted (large-prime-first) by
            // Sieving_prime_table, so this loop only computes per-prime
            // starting multiples + wheel indices — no sort happens here.
            const auto starting_multiples_start = std::chrono::steady_clock::now();
            for (std::size_t i = 0; i < shared_primes.size(); ++i)
            {
                const uint32_t prime = shared_primes[i];
                uint32_t m = get_offset_to_next_multiple(sieve_start, prime);
                m_prime_state[i].multiple = m;
                //where is the starting multiple relative to the wheel
                int wheel_index = (boost::integer::mod_inverse(static_cast<int>(prime), 30) * m) % 30;
                m_prime_state[i].wheel_index = sieve30_index[wheel_index];
            }
            const auto starting_multiples_end = std::chrono::steady_clock::now();

            uint64_t starting_multiples_us =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    starting_multiples_end - starting_multiples_start).count();
            m_diag_starting_multiples_us.store(starting_multiples_us, std::memory_order_relaxed);
            m_diag_prime_count.store(static_cast<uint32_t>(m_prime_state.size()),
                                      std::memory_order_relaxed);
        }

        void Sieve::sieve_segment()
        {
            // segment_bytes == sieve_size; loop bound in byte domain avoids per-hit j/30.
            const uint32_t segment_bytes = m_segment_size / 30;
            uint64_t seg_hits = 0;

            // Stone 1: prime values come from the process-shared table; mutable
            // wheel state is per-worker in m_prime_state, parallel-indexed.
            const auto& shared_primes = Sieving_prime_table::instance().primes();
            const std::size_t prime_count = m_prime_state.size();
            for (std::size_t pi = 0; pi < prime_count; ++pi)
            {
                SievePrimeState& sp = m_prime_state[pi];
                const uint32_t k = shared_primes[pi];
                int wheel_index  = sp.wheel_index;

                // ── Precompute wheel step table for this prime ────────────────────
                // 8 divisions total here instead of one division per inner-loop hit
                // (potentially millions per prime per segment call).
                WheelStep steps[8];
                for (int w = 0; w < 8; ++w)
                {
                    const uint32_t adv = k * static_cast<uint32_t>(sieve30_gaps[w]);
                    steps[w].byte_delta   = adv / 30;
                    steps[w].offset_delta = static_cast<uint8_t>(adv % 30);
                }

                // ── Decompose starting multiple into byte + offset domain ──────────
                uint32_t j = sp.multiple;
                uint32_t sieve_byte   = j / 30;  // 1 division per prime — unavoidable
                uint32_t sieve_offset = j % 30;  // 1 division per prime — unavoidable

                // ── Inner loop: zero integer divisions ────────────────────────────
                while (sieve_byte < segment_bytes)
                {
                    m_sieve[sieve_byte] &= unset_bit_mask[sieve_offset];
                    ++seg_hits;

                    const WheelStep& step = steps[wheel_index];
                    sieve_byte   += step.byte_delta;
                    sieve_offset += step.offset_delta;
                    if (sieve_offset >= 30)
                    {
                        sieve_offset -= 30;
                        ++sieve_byte;
                    }
                    wheel_index = (wheel_index + 1) & 7;
                }

                // ── Save-back: convert byte domain back to multiple ───────────────
                sp.multiple    = sieve_byte * 30 + sieve_offset - m_segment_size;
                sp.wheel_index = wheel_index;
            }

            // Update diagnostics: one atomic op per sieve_segment() call (relaxed — stats thread reads)
            m_diag_sieve_calls.fetch_add(1, std::memory_order_relaxed);
            m_diag_inner_hits.fetch_add(seg_hits, std::memory_order_relaxed);
        }
		
		//batch sieve on the cpu for debug
        void Sieve::sieve_batch_cpu(uint64_t low)
        {
            reset_sieve_batch(low);
            for (auto i = 0; i < m_segment_batch_size; i++)
            {
                reset_sieve();
                sieve_segment();
                //save the results of the sieve
                m_sieve_results.insert(m_sieve_results.end(), m_sieve.begin(), m_sieve.end());
            }
        }

        void Sieve::reset_sieve()
        {
            //fill the sieve with default values (all ones)
            std::fill(m_sieve.begin(), m_sieve.end(), sieve30);
            //m_fermat_candidates = {};
            m_long_chain_starts = {};
        }
		
		void Sieve::reset_sieve_batch(uint64_t low)
        {
            m_sieve_results = {};
            m_sieve_batch_start_offset = low;
        }

        void Sieve::clear_chains()
        {
            m_chain = {};
        }

        void Sieve::reset_stats()
        {
            // Sized to 12 buckets (was 10) so chain lengths 10-11 are
            // observable in operator stats — these are reachable post-#675
            // (variable-length vOffsets up to chain length 19) but the prior
            // size-10 vector silently saturated everything >= 9 into bucket 9.
            // Kept at 12 (not the full 19) to bound per-Sieve memory.
            m_chain_histogram = std::vector<std::uint32_t>(12, 0);
            // Stone 6.9 — same fixed size as m_chain_histogram so engine fan-in
            // can iterate buckets in lockstep without an additional bounds
            // dance.  Sized exactly once here (calling contract for
            // snapshot_chain_histogram_attempted, see chain_sieve.hpp).
            m_chain_histogram_attempted = std::vector<std::uint32_t>(12, 0);
            m_fermat_test_count = 0;
            m_fermat_prime_count = 0;
            m_chain_count = 0;
            m_chain_candidate_max_length = 0;
            m_chain_candidate_total_length = 0;
            m_diag_sieve_calls.store(0, std::memory_order_relaxed);
            m_diag_inner_hits.store(0, std::memory_order_relaxed);
            m_diag_starting_multiples_us.store(0, std::memory_order_relaxed);
            m_diag_prime_count.store(0, std::memory_order_relaxed);
            m_diag_chain_candidates_found.store(0, std::memory_order_relaxed);
            m_diag_chains_started_fermat.store(0, std::memory_order_relaxed);
            m_diag_chains_pushed_long.store(0, std::memory_order_relaxed);
            m_diag_popcount_windows_passed.store(0, std::memory_order_relaxed);
        }

        //search the sieve for chains that meet the minimum length requirement.  Chains can cross segment boundaries.
        void Sieve::find_chains(uint64_t low, bool batch_sieve_mode)
        {
            std::vector<uint8_t>& sieve = batch_sieve_mode?m_sieve_results:m_sieve;
            uint64_t sieve_size = sieve.size();

            //get popcount of the first three bytes  
            int hits_next_four_bytes = 0;
            std::queue<int> pop_count;
            pop_count.push(0);
            for (int i = 0; i < 3; i++)
            {
                int pop_count_this_byte = popcnt[sieve[i]];
                pop_count.push(pop_count_this_byte);
                hits_next_four_bytes += pop_count_this_byte;
            }
            for (uint64_t n = 0; n < sieve_size; n++)
            {
                //remove the oldest popcount from the running sum.
                hits_next_four_bytes -= pop_count.front();
                pop_count.pop();
                //get popcount of the current byte
                if (n + 3 < sieve_size)
                {
                    pop_count.push(popcnt[sieve[n + 3]]);
                    hits_next_four_bytes += pop_count.back(); 
                }
                if (!m_chain_in_process && hits_next_four_bytes < popcount_window_floor())
                {
                    //not enough prime candidates in the next 120 numbers to make a long enough chain

                }
                else
                {
                    // Stone 6.9.2 — count windows that survived the popcount
                    // floor (or that we entered because a chain was already
                    // in progress) so the engine funnel can disambiguate
                    // popcount-stage drops from close_chain-stage drops.
                    // Sized as a relaxed atomic increment per byte position;
                    // amortised cost is negligible vs. the wheel walk that
                    // follows.
                    if (!m_chain_in_process)
                    {
                        m_diag_popcount_windows_passed.fetch_add(
                            1, std::memory_order_relaxed);
                    }
                    if (sieve[n] == 0)
                    {
                        //no primes in this group of 30.  end the current chain if it is open.
                        if (m_chain_in_process)
                            close_chain();
                    }
                    else
                    {
                    int index_of_highest_set_bit = 0;
                    int sieve_offset = 0;
                    int previous_sieve_offset = 0;
                    for (uint8_t b = sieve[n]; b > 0; b &= b - 1)
                    {
                        int index_of_lowest_set_bit = boost::multiprecision::lsb(b);//c++20 alternative to lsb(b) is std::countr_zero(b);
                        sieve_offset = sieve30_offsets[index_of_lowest_set_bit];
                        uint64_t prime_candidate_offset = low + n * 30 + sieve_offset;
                        if (m_chain_in_process)
                        {
                            if (m_current_chain.m_gap_in_process + sieve_offset - previous_sieve_offset > maxGap)
                            {
                                //max gap exceeded.  close open chain and start a new one.
                                close_chain();
                                open_chain(prime_candidate_offset);
                            }
                            else
                            {
                                //continue chain
                                m_current_chain.push_back(prime_candidate_offset - m_current_chain.m_base_offset);
                                m_current_chain.m_gap_in_process = 0;
                            }
                        }
                        else
                        {
                            //start a new chain
                            open_chain(prime_candidate_offset);
                        }
                        index_of_highest_set_bit = index_of_lowest_set_bit;
                        previous_sieve_offset = sieve_offset;
                    }
                    if (m_chain_in_process)
                    {
                        //accumulate the gap at the end of the sieve word
                        m_current_chain.m_gap_in_process = 30 - sieve_offset;
                        //only keep the chain going if the final gap is smaller than the max
                        if (m_current_chain.m_gap_in_process > maxGap)
                        {
                            close_chain();
                        }
                    }
                    }
                }
               
            }
        }

        void Sieve::close_chain()
        {
            // Stone 6.9.2 / PR #678 follow-up — gate on close_chain_min(),
            // which is now target_length + 1 (empirical slack validated
            // against production data: image 15 vs image 8).  See
            // mining/prime_thresholds.hpp for the post-mortem on PR #672
            // / Stone 6.9.1, which had bumped this gate to target+slack(T)
            // AND shared that same threshold with the popcount window —
            // silently dropping ~90% of the candidates that could have
            // produced a length-T Fermat run.  The two gates are now
            // decoupled (popcount_window_floor() vs close_chain_min()):
            // popcount stays at the strict lower bound T (correctness
            // floor — never tighten), close_chain_min carries the
            // empirical `+1` quality slack to keep validate_attempts off
            // a flood of degenerate near-zero-quality chains.
            if (m_current_chain.length() >= close_chain_min())
            {
                //we found a chain candidate.  save it.
                m_chain.push_back(m_current_chain);
                m_chain_count++;
                m_chain_candidate_max_length = std::max(m_current_chain.length(), m_chain_candidate_max_length);
                m_chain_candidate_total_length += m_current_chain.length();
                m_diag_chain_candidates_found.fetch_add(1, std::memory_order_relaxed);

                // Stone 6.9 — bump the "attempted" histogram CUMULATIVELY for
                // every bucket from 0 up to the candidate's sieve-survivor
                // slot count.  Bucket k therefore stores the count of chains
                // that had >= k sieve-survivor slots available to Fermat-test
                // (matches the doc-comment on Engine_stats_snapshot::
                // chain_histogram_attempted and prime-mining-flow.md).  This
                // is what makes
                //     histogram[k] / attempted[k]
                // an interpretable per-bucket survival probability: the
                // denominator must include EVERY chain wide enough to
                // possibly produce a length-k Fermat run, not just chains
                // whose width was exactly k.  Cost is bounded by the chain
                // length cap (~10 increments); negligible vs the Fermat
                // tests that follow.
                {
                    // Defensive bounds: length() returns int(m_offsets.size()) and
                    // m_chain_histogram_attempted is sized at >=1 in reset_stats,
                    // so neither degenerate value can occur in production — but
                    // the explicit max(0, …) keeps the negative→huge-unsigned
                    // pitfall from sneaking back in if length() is ever changed.
                    const int hist_max = static_cast<int>(m_chain_histogram_attempted.size()) - 1;
                    const int top_bucket = std::max(0, std::min(m_current_chain.length(), hist_max));
                    for (int b = 0; b <= top_bucket; ++b)
                    {
                        m_chain_histogram_attempted[b]++;
                    }
                }
            }
            m_current_chain.close();
            m_chain_in_process = false;
        }

        void Sieve::open_chain(uint64_t base_offset)
        {
            //reset chain in process to the default
            m_current_chain = { base_offset };
            // Stone 6.9 — propagate the per-session target into the chain so
            // is_there_still_hope() and any later chain-level decision
            // honour the same threshold close_chain() used to keep this
            // candidate.  Default-constructed Chain has m_min_chain_length=8.
            m_current_chain.m_min_chain_length = m_target_chain_length;
            // PR #681 follow-up — note re §4(1) of the plan: deliberately do
            // NOT propagate m_target_chain_length into m_min_chain_report_length
            // here.  The CPU side intentionally keeps the report-length floor
            // at the Chain default of 5 so length-5/6 chains reach
            // ValidatePrimeCandidate and are counted in the funnel
            // (rejected_below_diff) for diagnostic visibility — see the
            // comment at test_chains/clean_chains' push_gate computation
            // (~line 688-700) and the stored "prime sieve" memory.  The GPU
            // side (cuda_chain.cu:46-55) writes both fields equal because it
            // doesn't have the same diagnostic surface; that divergence is
            // intentional and not a footgun.
            m_chain_in_process = true;
        }

        //get the next prime to test from each chain
        void Sieve::test_chains()
        {
            // Stone 2: legacy no-arg overload — forwards to the cached value.
            test_chains(m_sieve_start);
        }

        void Sieve::test_chains(const boost::multiprecision::uint1024_t& sieve_start)
        {
            for (auto i = 0; i < m_chain.size(); i++)
            {
                bool there_is_still_hope = true;
                int prime_count_this_chain = 0;
                bool started_fermat = false;
                uint64_t base_offset;
                int offset;
                while (there_is_still_hope)
                {
                    if (m_chain[i].get_next_fermat_candidate(base_offset, offset))
                    {
                        if (!started_fermat)
                        {
                            started_fermat = true;
                            m_diag_chains_started_fermat.fetch_add(1, std::memory_order_relaxed);
                        }
                        boost::multiprecision::uint1024_t candidate = sieve_start + base_offset + offset;
                        bool is_prime = primality_test(candidate);
                        m_chain[i].update_fermat_status(is_prime);
                        if (is_prime)
                        {
                            prime_count_this_chain++;
                        }
                        there_is_still_hope = m_chain[i].is_there_still_hope();
                    }
                }
                m_chain[i].m_chain_state = Chain::Chain_state::complete;
                if (prime_count_this_chain > 0)
                {
                    int length;
                    m_chain[i].get_best_fermat_chain(base_offset, offset, length);

                    //collect stats — see close_chain bucket-index comment.
                    const int hist_max = static_cast<int>(m_chain_histogram.size()) - 1;
                    const int bucket_index = std::max(0, std::min(length, hist_max));
                    m_chain_histogram[bucket_index]++;

                    // Stone 6.9 — push gate is min(per-session target, the
                    // chain's own m_min_chain_report_length).  This preserves
                    // legacy Worker_prime behaviour (callers that never call
                    // set_target_length() keep the historical report-length
                    // floor of 5 and continue to emit "Found a fermat chain"
                    // info-lines for length-5..7 runs) while engine mode's
                    // downstream dispatch is still difficulty-gated, so the
                    // extra short-chain pushes are observed by
                    // m_diag_validate_rejected_below_diff rather than silently
                    // dropped.  The funnel counters localize the loss either
                    // way; truncating here would just hide diagnostic info.
                    const int push_gate = std::min(m_target_chain_length,
                                                   m_chain[i].m_min_chain_report_length);
                    if (length >= push_gate)
                    {
                        //we found a long chain.  save it.
                        m_logger->info("Found a fermat chain of length {} (target={}, gate={}).",
                                       length, m_target_chain_length, push_gate);
                        m_long_chain_starts.push_back(base_offset + offset);
                        m_diag_chains_pushed_long.fetch_add(1, std::memory_order_relaxed);
                    }
                }

            }

        }


        //batch process the list of prime candidates to be fermat tested.  
        void Sieve::primality_batch_test()
        {

            for (auto& chain : m_chain)
            {
                uint64_t base_offset;
                int offset;
                bool success = chain.get_next_fermat_candidate(base_offset, offset);
                boost::multiprecision::uint1024_t candidate = m_sieve_start + base_offset + offset;
                bool is_prime = primality_test(candidate);
                /*uint1024_t T("0x0000005ff320ec9f9599b9cb0156c793f61060c8a8c49185df9d25603e37259c2f0213d6d96745bbbbe7ea1e4e9da371aeeb5d20c204c22a038b10957b53c67d9eb3a00acfaeb6ccd4c231a8088d5a5745e19f70387a7d91463d9b318a1f0503819a32f5fa32cf3579c7d6a3546cbdceaa364cfa2e989defeb4f5fe29de687cc");
                uint64_t nNonce = 4933493377870005061;
                if (candidate >= T + nNonce && candidate <= T + nNonce + 100)
                {
                    std::cout << "base offset: " << base_offset << " offset: " << offset << " is prime: " << is_prime << std::endl;
                }*/
                chain.update_fermat_status(is_prime);
            }

        }

        uint64_t Sieve::get_current_chain_list_length()
        {
            return m_chain.size();
        }

        //search for winners.  delete finished or hopeless chains.
        //run this after batch primality testing.
        void Sieve::clean_chains()
        {
            size_t chain_count_before = m_chain.size();
            for (auto& chain : m_chain)
            {
                uint64_t base_offset;
                int offset, length;

                //this approach keeps chains until all offsets in the chain have been tested.
                //This runs more primality tests but finds alot of short chains. 
                //Use is_there_still_hope() instead to reduce fermat testing.  Fewer short chains will be found which feels worse but is acutally faster for finding long chains.
                if (chain.m_untested_count <= 0)  
                {
                    //chain is tested.  mark as complete.
                    chain.m_chain_state = Chain::Chain_state::complete;
                    chain.get_best_fermat_chain(base_offset, offset, length);
                    if (length > 0)
                    {
                        //collect stats — see close_chain bucket-index comment.
                        const int hist_max = static_cast<int>(m_chain_histogram.size()) - 1;
                        const int bucket_index = std::max(0, std::min(length, hist_max));
                        m_chain_histogram[bucket_index]++;
                    }
                    // Stone 6.9 — same min(target, report_length) gate as
                    // test_chains() (see the longer comment there).  Restores
                    // legacy Worker_prime length-5..7 "Found a fermat chain"
                    // diagnostic noise while engine mode's downstream
                    // dispatch is still difficulty-gated.
                    const int push_gate = std::min(m_target_chain_length,
                                                   chain.m_min_chain_report_length);
                    if (length >= push_gate)
                    {
                        //we found a long chain.  save it.
                        m_logger->info("Found a fermat chain of length {} (target={}, gate={}).",
                                       length, m_target_chain_length, push_gate);
                        m_long_chain_starts.push_back(base_offset + offset);
                        m_diag_chains_pushed_long.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
            //remove completed chains
            m_chain.erase(std::remove_if(m_chain.begin(), m_chain.end(),
                [](Chain& c) {return c.m_chain_state == Chain::Chain_state::complete; }), m_chain.end());
            size_t chain_count_after = m_chain.size();

        }


        uint64_t Sieve::count_fermat_primes(uint64_t sieve_size, uint64_t low)
        {
            uint64_t count = 0;
            for (uint64_t n = 0; n < sieve_size; n++)
            {
                for (uint8_t b = m_sieve[n]; b > 0; b &= b - 1)
                {
                    int index_of_lowest_set_bit = boost::multiprecision::lsb(b);//std::countr_zero(b);
                    uint64_t prime_candidate_offset = low + n * 30 + sieve30_offsets[index_of_lowest_set_bit];
                    uint1024_t p = m_sieve_start + prime_candidate_offset;
                    count += primality_test(p) ? 1 : 0;
                }
            }
            return count;
        }

        bool Sieve::primality_test(boost::multiprecision::uint1024_t p)
        {
            //gmp powm is about 7 times faster than boost backend
            mpz_int base = 2;
            mpz_int result;
            mpz_int p1 = static_cast<mpz_int>(p);
            result = boost::multiprecision::powm(base, p1 - 1, p1);
            m_fermat_test_count++;
            bool isPrime = (result == 1);
            if (isPrime)
            {
                ++m_fermat_prime_count;
            }
            return (isPrime);
        }
    }
}