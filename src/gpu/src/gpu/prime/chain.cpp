#include "chain.hpp"
#include <algorithm>
#include <cstddef>
#include <sstream>

namespace nexusminer {
    namespace gpu
    {
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
        // Stone — mirrors the CUDA reference impl in cuda_chain.cu::is_there_still_hope:
        //   (1) early-out if no untested offsets remain;
        //   (2) keepalive once we've already proven >= kHopeKeepaliveThreshold
        //       Fermat primes, so the upper histogram buckets keep counting
        //       (matches the CPU/GPU device-side gates' policy);
        //   (3) totals prune (prime + untested >= m_min_chain_length);
        //   (4) walk through get_best_fermat_chain assuming all currently
        //       untested offsets pass to ensure the maximum-possible
        //       CONTIGUOUS run could still reach m_min_chain_length.
        // The prior naive totals-only test (kept (1)+(3) only) caused the
        // length-T bucket to silently read 0 even at length-T target — see
        // the CPU-side chain_sieve.cpp comment for the full post-mortem.
        bool Chain::is_there_still_hope()
        {
            //nothing left to test
            if (m_untested_count == 0)
            {
                return false;
            }

            // Keepalive: once we've already found >= kHopeKeepaliveThreshold
            // primes, keep this chain alive regardless of contiguous-run
            // analysis so the stats see the longer chains.  Same policy as
            // CUDA cuda_chain.cu and CPU chain_sieve.cpp.
            if (m_prime_count >= kHopeKeepaliveThreshold)
                return true;

            // Necessary condition (totals prune).
            if ((m_prime_count + m_untested_count) < m_min_chain_length)
                return false;

            // Sufficient-condition refinement: temporarily flip every
            // untested offset to "pass" and ask get_best_fermat_chain for
            // the longest run that satisfies the maxGap rule.  If even an
            // optimistic walk cannot reach m_min_chain_length, the chain is
            // definitely busted.  Use a stack-resident array sized to the
            // chain's hard cap so this never allocates (matches the device-side
            // CudaChain layout — see cuda_chain.cuh).
            constexpr std::size_t kMaxOffsets = 32;
            Fermat_test_status saved[kMaxOffsets];
            const std::size_t n = std::min(m_offsets.size(), kMaxOffsets);
            for (std::size_t i = 0; i < n; ++i)
            {
                saved[i] = m_offsets[i].m_fermat_test_status;
                if (m_offsets[i].m_fermat_test_status == Fermat_test_status::untested)
                    m_offsets[i].m_fermat_test_status = Fermat_test_status::pass;
            }
            uint64_t fake_base_offset = 0;
            int fake_offset = 0;
            int fake_length = 0;
            get_best_fermat_chain(fake_base_offset, fake_offset, fake_length);
            // Restore real statuses.
            for (std::size_t i = 0; i < n; ++i)
                m_offsets[i].m_fermat_test_status = saved[i];

            return fake_length >= m_min_chain_length;
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

        //add a new offset to the chain
        void Chain::push_back(int offset)
        {
            Chain_offset chain_offset{ offset };
            m_offsets.push_back(chain_offset);
            m_untested_count++;
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
    }

}