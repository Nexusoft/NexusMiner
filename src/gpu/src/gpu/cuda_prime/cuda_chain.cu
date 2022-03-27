#include "gpu_helper.hpp"
#include "cuda_chain.cuh"

//#include "device_launch_parameters.h"
#include <stdio.h>

namespace nexusminer {
    namespace gpu
    {
       
        //add a new offset to the chain
        __device__ void cuda_chain_push_back(CudaChain& chain, uint16_t offset)
        {
            if (chain.m_offset_count < chain.m_max_chain_length)
            {
                chain.m_offsets[chain.m_offset_count] = offset;
                chain.m_fermat_test_status[chain.m_offset_count] = Fermat_test_status::untested;
                chain.m_untested_count++;
                chain.m_offset_count++;
            }
        }
       
        __device__ void cuda_chain_open(CudaChain& chain, uint64_t base_offset)
        {
            chain.m_base_offset = base_offset;
            chain.m_offsets[0] = 0; //the first offset is always zero
            chain.m_fermat_test_status[0] = Fermat_test_status::untested;
            chain.m_untested_count = 1;
            chain.m_offset_count = 1;
            chain.m_prime_count = 0;
        }
       

        //analyze the chain fermat test results.  
        //return the starting offset and length of the longest fermat chain that meets the mininmum gap requirement
        __device__ void get_best_fermat_chain(const CudaChain& chain, uint64_t& base_offset, int& offset, int& best_length)
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
        __device__ bool is_there_still_hope(CudaChain& chain)
        {
            //there is nothing left to test
            if (chain.m_untested_count == 0)
            {
                return false;
            }

            //If we've already found 4, keep counting regardless. 
            //this makes the stats look better and doesn't impact performance much since it is relatively rare
            if (chain.m_prime_count >= 4)
                return true;

            if ((chain.m_prime_count + chain.m_untested_count) < chain.m_min_chain_length)
                return false;

            int extra_links = chain.m_offset_count - chain.m_min_chain_length;
            //no failures yet
            if (extra_links == 0)
                return true;

            //there are extra links.  check for a broken chain
            for (auto i = 0; i < chain.m_offset_count; i++)
            {
                if (chain.m_fermat_test_status[i] == Fermat_test_status::fail)
                {
                    
                    bool interior_link = ((i >= extra_links) && i < (chain.m_offset_count - extra_links));
                    if (interior_link)
                    {
                        int gap = chain.m_offsets[i + 1] - chain.m_offsets[i - 1];
                        if (gap > maxGap && chain.m_offset_count < chain.m_min_chain_length * 2)
                            return false;
                    }
                }
            }

            return true;
        }

        //get the next untested fermat candidate.  prioritize candidates that bust the chain if they fail.  if there are no candidates return false.
        __device__ bool get_next_fermat_candidate(CudaChain& chain, uint64_t& base_offset, int& offset)
        {
            // are there any extra links in the current chain?  i.e. if we are looking for 8-chains, and this chain has 9 candidates, there is one extra link.
            int extra_links = chain.m_prime_count + chain.m_untested_count - chain.m_min_chain_length;
            //This is to handle the special case where there are untested offsets but none are weak links.  This case should be rare.  
            int an_untested_index = -1;
            for (auto i = 0; i < chain.m_offset_count; i++)
            {
                //weak links can't be the first or last links when there are extra links.
                bool interior_link = ((i >= extra_links) && i < (chain.m_offset_count - extra_links));
                bool weak_link = true;
                if (extra_links > 0)
                {
                    if (interior_link)
                    {
                        int gap = chain.m_offsets[i + 1] - chain.m_offsets[i - 1];
                        weak_link = gap > maxGap;
                    }
                    else
                    {
                        weak_link = false;
                    }
                }
                if (chain.m_fermat_test_status[i] == Fermat_test_status::untested)
                {
                    an_untested_index = i;
                    if (weak_link)
                    {
                        base_offset = chain.m_base_offset;
                        offset = chain.m_offsets[i];
                        //save the offset under test index for later
                        chain.m_next_fermat_test_offset_index = i;
                        return true;
                    }
                }
            }
            //speical case - return the last known untested offset if none are weak links but there is an untested link
            if (an_untested_index > -1)
            {
                base_offset = chain.m_base_offset;
                offset = chain.m_offsets[an_untested_index];
                chain.m_next_fermat_test_offset_index = an_untested_index;
                return true;
            }
            return false;
        }


        //set the fermat test status of an offset.  if the offset is not found return false.
        __device__ bool update_fermat_status(CudaChain& chain, bool is_prime)
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