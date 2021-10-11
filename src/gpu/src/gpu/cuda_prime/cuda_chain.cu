#include "cuda_chain.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

namespace nexusminer {
    namespace gpu
    {
       
        //add a new offset to the chain
        __device__  void cuda_chain_push_back(CudaChain& chain, uint16_t offset)
        {
            if (chain.m_offset_count < chain.m_max_chain_length)
            {
                chain.m_offsets[chain.m_offset_count] = offset;
                chain.m_fermat_test_status[chain.m_offset_count] = Fermat_test_status::untested;
                chain.m_untested_count++;
                chain.m_offset_count++;
            }
        }
       
        __device__  void cuda_chain_open(CudaChain& chain, uint64_t base_offset)
        {
            chain.m_base_offset = base_offset;
            chain.m_chain_state = CudaChain::Chain_state::open;
            cuda_chain_push_back(chain, 0);  //the first offset is always zero
            chain.m_gap_in_process = 0;
            chain.m_prime_count = 0;
        }
        __device__  void cuda_chain_close(CudaChain& chain)
        {
            chain.m_chain_state = CudaChain::Chain_state::closed;
        }

        //analyze the chain fermat test results.  
        //return the starting offset and length of the longest fermat chain that meets the mininmum gap requirement
        __device__  void get_best_fermat_chain(const CudaChain& chain, uint64_t& base_offset, int& offset, int& best_length)
        {
            base_offset = chain.m_base_offset;
            offset = 0;
            int chain_length = 0;
            best_length = 0;
            if (chain.m_offset_count == 0)
                return;

            int gap = 0;
            int starting_offset = 0;
            auto previous_offset = chain.m_offsets[0];
            for (int i = 0; i < chain.m_offset_count; i++)
            {
                if (chain_length > 0)
                    gap += chain.m_offsets[i] - previous_offset;
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
                if (chain.m_fermat_test_status[i] == Fermat_test_status::pass)
                {
                    chain_length++;
                    gap = 0;
                    if (chain_length == 1)
                    {
                        starting_offset = chain.m_offsets[i];
                    }
                }
                previous_offset = chain.m_offsets[i];

            }
            if (chain_length > best_length)
            {
                best_length = chain_length;
                offset = starting_offset;
            }
            return;
        }

        //return true if there is more testing we can do. returns false if we should give up.
        __device__  bool is_there_still_hope(CudaChain& chain)
        {
            //nothing left to test
            if (chain.m_untested_count == 0)
            {
                return false;
            }

            return (chain.m_prime_count + chain.m_untested_count) >= chain.m_min_chain_length;
        }

        //get the next untested fermat candidate.  if there are none return false.
        __device__  bool get_next_fermat_candidate(CudaChain& chain, uint64_t& base_offset, int& offset)
        {
            //This returns the next untested prime candidate.
            //There are other more complex ways to do this to minimize primality testing
            //like search for the first candidate that busts the chain if it fails
            for (auto i = 0; i < chain.m_offset_count; i++)
            {
                if (chain.m_fermat_test_status[i] == Fermat_test_status::untested)
                {
                    base_offset = chain.m_base_offset;
                    offset = chain.m_offsets[i];
                    //save the offset under test index for later
                    chain.m_next_fermat_test_offset_index = i;
                    return true;
                }
            }
            return false;
        }


        //set the fermat test status of an offset.  if the offset is not found return false.
        __device__  bool update_fermat_status(CudaChain& chain, bool is_prime)
        {
            chain.m_untested_count--;
            if (is_prime)
            {
                chain.m_fermat_test_status[chain.m_next_fermat_test_offset_index] = Fermat_test_status::pass;
                chain.m_prime_count++;
            }
            else
            {
                chain.m_fermat_test_status[chain.m_next_fermat_test_offset_index] = Fermat_test_status::fail;
            }

            return true;

        }

        
    }
}