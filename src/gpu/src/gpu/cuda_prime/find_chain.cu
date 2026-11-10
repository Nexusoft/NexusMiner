#include "gpu_helper.hpp"
#include "sieve_impl.cuh"
#include "sieve.hpp"
#include "find_chain.cuh"
#include "sieve_lookup_tables.cuh"
#include "cuda_chain.cuh"
#include "mining/prime_thresholds.hpp"

#include <stdio.h>
#include <math.h>
#include <inttypes.h>


namespace nexusminer {
    namespace gpu {

        __device__ void cuda_chain_push_back(CudaChain& chain, uint16_t offset);
        __device__ void cuda_chain_open(CudaChain& chain, uint64_t base_offset, int target_length);
        __device__  bool is_there_still_hope(CudaChain& chain);
        __device__  void get_best_fermat_chain(const CudaChain& chain, uint64_t& base_offset, int& offset, int& best_length);

        // atomicInc(counter, capacity) returns `capacity` and stores 0 once the
        // buffer is full.  Callers then write one past the allocation and the
        // next claim overwrites slot 0.  Claim only an in-range slot.  A rejected
        // claim is not a stored chain, so the counter is rolled back and the
        // caller must not count it.
        __device__ __forceinline__ bool claim_chain_slot(uint32_t* counter, uint32_t capacity, uint32_t& out_index)
        {
            const uint32_t idx = atomicAdd(counter, 1u);
            if (idx >= capacity)
            {
                atomicSub(counter, 1u);
                return false;
            }
            out_index = idx;
            return true;
        }


        //get the nth bit from the sieve.
        __device__ __forceinline__ bool get_bit(uint64_t bit_position, Cuda_sieve::sieve_word_t* sieve)
        {
            const uint32_t sieve_bits_per_word = Cuda_sieve::m_sieve_word_byte_count * 8;
            
            uint64_t word = bit_position / sieve_bits_per_word;
            unsigned bit_position_in_word = bit_position % sieve_bits_per_word;
            return ((sieve[word] >> bit_position_in_word) & 1) == 1;

        }

        //search the sieve for chains that meet the minimum length requirement.  
        __global__ void find_chain_kernel(Cuda_sieve::sieve_word_t* sieve, CudaChain* chains, uint32_t* chain_index, uint64_t sieve_start_offset,
            unsigned long long* chain_stat_count, Cuda_sieve::Cuda_sieve_properties sieve_properties)
        {

            //const uint64_t sieve_size = Cuda_sieve::m_sieve_total_size;
            const uint32_t sieve_bits_per_word = Cuda_sieve::m_sieve_word_byte_count * 8;
            const uint64_t sieve_total_bits = sieve_properties.m_sieve_total_size * sieve_bits_per_word;
            uint64_t num_blocks = gridDim.x;
            uint64_t num_threads = blockDim.x;
            uint64_t block_id = blockIdx.x;
            uint64_t index = block_id * num_threads + threadIdx.x;
            uint64_t stride = num_blocks * num_threads;
            unsigned int sieve_offset;
            unsigned int gap;
            uint64_t chain_start, prime_candidate_offset;

            //shared copies of lookup tables
            __shared__ unsigned int sieve30_offsets_shared[8];
            __shared__ unsigned int sieve30_gaps_shared[8];
            //local stats
            __shared__ uint32_t chain_count_shared;
            
            if (threadIdx.x < 8)
            {
                int i = threadIdx.x;
                sieve30_offsets_shared[i] = sieve30_offsets[i];
                sieve30_gaps_shared[i] = sieve30_gaps[i];
            }
            
            if (threadIdx.x == 0)
                chain_count_shared = 0;
            __syncthreads();
           
            //search each sieve location for a possible chain
            for (uint64_t i = index; i < sieve_total_bits; i += stride)
            {
              
                //gross checks to ensure its possible to form a chain
                uint64_t word = i / sieve_bits_per_word;
                if (sieve[word] == 0)
                    continue;
                //check if the next 4 bytes (4*30 = range of 120 integers) has enough prime candidates to form a chain 
                //this is only valid up to min chain length 9.  above 9 requires 5 bytes.
                // Stone — per-session SSOT for the popcount window floor.
                // Was hard-coded to Cuda_sieve::m_min_chain_length=9; now reads
                // the live per-session T from sieve_properties.m_min_chain_length
                // (set by Cuda_sieve::set_target_length() before find_chains()).
                // Apply the early-exit only when T is within the kernel's
                // 4-byte popcount-window correctness ceiling — for T > 9 the
                // window is too narrow to be a tight necessary condition, so
                // skipping the check is the only correctness-preserving move.
                const int target_T = sieve_properties.m_min_chain_length;
                const bool popcount_filter_safe =
                    target_T <= nexusminer::mining::popcount_window_supported_max();
                if (popcount_filter_safe && word < sieve_properties.m_sieve_total_size - 1)
                {
                    unsigned int next_4_bytes = 0;
                    unsigned int byte_index = (i/8) % 4;
                    next_4_bytes = (sieve[word] >> (byte_index * 8)) & 0xFF;
                    next_4_bytes |= (((sieve[word + (byte_index >= 3 ? 1 : 0)] >> ((byte_index + 1) % 4) * 8) & 0xFF) << 8);
                    next_4_bytes |= (((sieve[word + (byte_index >= 2 ? 1 : 0)] >> ((byte_index + 2) % 4) * 8) & 0xFF) << 16);
                    next_4_bytes |= (((sieve[word + (byte_index >= 1 ? 1 : 0)] >> ((byte_index + 3) % 4) * 8) & 0xFF) << 24);

                    int popc = __popc(next_4_bytes);
                    if (popc < nexusminer::mining::popcount_window_floor(target_T))
                        continue;
                }

                //chain must start with a prime
                if (!get_bit(i, sieve))
                {
                    continue;
                }
                //search left for another prime less than max gap away
                uint64_t j = i - 1;
                gap = sieve30_gaps_shared[j % 8];
                while (j < i && gap <= maxGap)
                {
                    if (get_bit(j, sieve))
                    {
                        //there is a valid element to the left.  this is not the first element in a chain. abort.
                        break;
                    }
                    j--;
                    gap += sieve30_gaps_shared[j % 8];
                }
                if (gap <= maxGap)
                    continue;
                //this is the start of a possible chain.  search right
                //where are we in the wheel
                sieve_offset = sieve30_offsets_shared[i % 8u];
                chain_start = sieve_start_offset + i / 8 * 30 + sieve_offset;
                CudaChain current_chain;
                cuda_chain_open(current_chain, chain_start, target_T);
                j = i;
                gap = sieve30_gaps_shared[j % 8u];
                j++;
                while (j < sieve_total_bits && gap <= maxGap)
                {
                    if (get_bit(j, sieve))
                    {
                        //another possible candidate.  add it to the chain
                        gap = 0;
                        sieve_offset = sieve30_offsets_shared[j % 8u];
                        prime_candidate_offset = sieve_start_offset + j / 8 * 30 + sieve_offset;
                        uint16_t offset = prime_candidate_offset - chain_start;
                        //printf("%" PRIu64 " %u\n", chain_start, prime_candidate_offset);
                        cuda_chain_push_back(current_chain, offset);
                    }
                    gap += sieve30_gaps_shared[j % 8u];
                    j++;
                        
                }
                //we reached the end of the chain.  check if it meets the length requirement
                // Stone — per-session SSOT for the close_chain quality gate
                // (see top-of-kernel comment).
                if (current_chain.m_offset_count >= nexusminer::mining::close_chain_min(target_T))
                {
                    uint32_t chain_idx = 0;
                    if (claim_chain_slot(chain_index, Cuda_sieve::m_max_chains, chain_idx))
                    {
                        chains[chain_idx] = current_chain;
                        // Dropped chains are not stored, so they are not counted.
                        atomicInc(&chain_count_shared, 0xFFFFFFFF);
                    }
                }
            }
            //update global chain stats
            __syncthreads();
            if (threadIdx.x == 0)
                atomicAdd(chain_stat_count, chain_count_shared);
        }

        __device__ void close_chain(const CudaChain& chain, uint32_t* chain_index, CudaChain* chains, uint32_t* chain_count_shared,
            int target_length)
        {
            //We reached the end of the chain.  
            // Stone — per-session SSOT close_chain gate (mirrors find_chain_kernel).
            // target_length is the live per-session T plumbed through from
            // sieve_properties.m_min_chain_length by the caller.
            if (chain.m_offset_count >= nexusminer::mining::close_chain_min(target_length))
            {
                uint32_t chain_idx = 0;
                if (claim_chain_slot(chain_index, Cuda_sieve::m_max_chains, chain_idx))
                {
                    chains[chain_idx] = chain;
                    atomicInc(chain_count_shared, 0xFFFFFFFF);
                }
            }
        }

        //alternative chain finder
        //each kernel block is a sieve segment.  Each thread searches a range of 2310*4 within a segment.   
        __global__ void find_chain_kernel2(Cuda_sieve::sieve_word_t* sieve, CudaChain* chains, uint32_t* chain_index, uint64_t sieve_start_offset,
            unsigned long long* chain_stat_count, Cuda_sieve::Cuda_sieve_properties sieve_properties)
        {
            const unsigned int search_range = Cuda_sieve::m_sieve_chain_search_boundary * Cuda_sieve::m_sieve_word_byte_count;
            const unsigned int search_words = search_range / Cuda_sieve::m_sieve_word_range;
            unsigned int total_search_regions = sieve_properties.m_sieve_range / search_range;
            unsigned int num_blocks = gridDim.x;
            unsigned int block_id = blockIdx.x / Cuda_sieve::m_kernel_segments_per_block;
            unsigned int segment_id = blockIdx.x % Cuda_sieve::m_kernel_segments_per_block;
            unsigned int index = threadIdx.x;
            unsigned int search_regions_per_kernel_block = (total_search_regions + num_blocks - 1) / num_blocks;
            unsigned int stride = blockDim.x;
            unsigned int gap = 0;
            uint32_t chain_start;
            uint64_t segment_offset = sieve_start_offset + block_id * sieve_properties.m_block_range + segment_id * sieve_properties.m_segment_range;
            uint32_t sieve_segment_index = block_id * sieve_properties.m_kernel_sieve_size_words_per_block + segment_id * sieve_properties.m_kernel_sieve_size_words;
            uint32_t sieve_index;
            //shared copies of lookup tables
            __shared__ unsigned int sieve30_offsets_shared[8];
            //local stats
            __shared__ uint32_t chain_count_shared;
            //local shared copy of the sieve
            //__shared__ Cuda_sieve::sieve_word_t sieve_shared[Cuda_sieve::m_kernel_sieve_size_words];

            if (threadIdx.x < 8)
            {
                int i = threadIdx.x;
                sieve30_offsets_shared[i] = sieve30_offsets[i];
            }

            if (threadIdx.x == 0)
            {
                chain_count_shared = 0;
            }
                
            __syncthreads();

            // Stone — per-session SSOT for both popcount window floor and
            // close_chain quality gate (find_chain_kernel2 path).  Reads the
            // live per-session T from sieve_properties.m_min_chain_length.
            // Apply the per-word/byte popcount early-exits only when T is
            // within the kernel's correctness ceiling (the popcount window
            // here is a single 32-bit word == 30*4 = 120 integers, same as
            // find_chain_kernel's 4-byte window).  Above the ceiling the
            // necessary-condition reasoning no longer holds; skip the
            // early-exit so no winner is dropped.
            const int target_T = sieve_properties.m_min_chain_length;
            const bool popcount_filter_safe =
                target_T <= nexusminer::mining::popcount_window_supported_max();
            const int popcount_floor = nexusminer::mining::popcount_window_floor(target_T);

            sieve_index = sieve_segment_index + index * search_words;
            for (unsigned int region = index; region < search_regions_per_kernel_block; region += stride)
            {
                bool chain_in_process = false;
                CudaChain current_chain;
                uint64_t region_offset = segment_offset + region * search_range;
                chain_start = 0;
                //sieve_index = region * search_words + sieve_segment_index;
                //sieve_index += sieve_index_increment;
                uint32_t last_offset = 0;
                //iterate through each word in the search region
                bool previous_word_last_bit_set = false;
                for (unsigned int word = 0; word < search_words; word++, sieve_index++)
                {
                    uint32_t sieve_word = sieve[sieve_index];
                    uint32_t next_word = word >= search_words ? 0 : sieve[sieve_index + 1];
                    bool next_word_first_bit_set = (next_word & 1) == 1;
                    if (chain_in_process)
                    {
                        uint32_t word_start = sieve_word & 0x7;
                        bool first_bit_set = (sieve_word & 1) == 1;
                        //if the first 3 bits are zeros any in process chain is broken
                        //if the last bit of the previous word and the first bit of the currnet word are both 0 the chain is broken
                        if (word_start == 0 || (!previous_word_last_bit_set && !first_bit_set))
                        {
                            //We reached the end of the chain.  
                            close_chain(current_chain, chain_index, chains, &chain_count_shared, target_T);
                            chain_in_process = false;
                        }
                    }
                    bool last_bit_set = sieve_word >= 0x80000000;
                    previous_word_last_bit_set = last_bit_set;
                    //gross check to ensure there are enough set bits in the word to make it worthwhile to process
                    if (!chain_in_process)
                    {
                        //if bits 7 and 8 are both zero, no chain originating in the first byte can make it through.  discard any set bits in the first byte
                        uint32_t first_byte_transition = sieve_word & 0x000000180;
                        if (first_byte_transition == 0)
                            sieve_word &= 0xFFFFFF00;
                        if (popcount_filter_safe)
                        {
                            uint32_t second_byte_transition = sieve_word & 0x000018000;
                            if (second_byte_transition == 0 && __popc(sieve_word & 0x0000FFFF) < popcount_floor)
                                sieve_word &= 0xFFFF0000;
                            uint32_t third_byte_transition = sieve_word & 0x001800000;
                            if (third_byte_transition == 0 && __popc(sieve_word & 0x00FFFFFF) < popcount_floor)
                                sieve_word &= 0xFF000000;

                            //if the last 3 bits are all zero, or the last bit is zero and first bit of the next word is zero, any chain must end at the current word
                            uint32_t word_end = sieve_word & 0xE0000000;
                            bool chain_must_end = (word_end == 0 || (!last_bit_set && !next_word_first_bit_set));
                            int popc = __popc(sieve_word);
                            if (chain_must_end && popc < popcount_floor)
                                continue;
                        }
                        
                    }

                    //iterate through each set bit in the sieve word
                    for (unsigned int b = sieve_word; b > 0; b &= b - 1)
                    {
                        //determine the position of the set bit in the sieve word.
                        uint8_t lowest_set_bit = __ffs(b) - 1;  //__ffs is a cuda primitive that finds the index of the lowest set bit in a word (ones based).
                        uint8_t byte_index = lowest_set_bit / 8;
                        unsigned int sieve30_offset = sieve30_offsets_shared[lowest_set_bit % 8];
                        uint32_t local_offset = word * Cuda_sieve::m_sieve_word_range +
                            byte_index * Cuda_sieve::m_sieve_byte_range + sieve30_offset;
                        gap = local_offset - last_offset;
                        /*if (region_offset + local_offset == 2055301)
                            printf("sieve word %u %x region offset %llu local offset % u\n", sieve_index + word, sieve[sieve_index + word], region_offset, local_offset);*/
                        if (chain_in_process)
                        {
                            if (gap > maxGap)
                            {
                                //We reached the end of the chain.  
                                close_chain(current_chain, chain_index, chains, &chain_count_shared, target_T);
                                //start a new chain
                                cuda_chain_open(current_chain, region_offset + local_offset, target_T);
                                chain_start = local_offset;
                                last_offset = local_offset;
                            }
                            else
                            {
                                //grow the chain
                                uint16_t offset_from_chain_start = local_offset - chain_start;
                                cuda_chain_push_back(current_chain, offset_from_chain_start);
                                last_offset = local_offset;
                            }
                        }
                        else
                        {
                            //start a new chain
                            cuda_chain_open(current_chain, region_offset + local_offset, target_T);
                            last_offset = local_offset;
                            chain_start = local_offset;
                            chain_in_process = true;
                        }
                    }
                }
                //we reached the end of the search region.  do a final check on the chain in process
                if (chain_in_process)
                {
                    close_chain(current_chain, chain_index, chains, &chain_count_shared, target_T);
                }
            }
            
            //update global chain stats
            __syncthreads();
            if (threadIdx.x == 0)
            {
                atomicAdd(chain_stat_count, chain_count_shared);
            }
            
        }

        //go through the list of chains.  copy winners to the long chain list.  copy survivors to a temporary chain
        __global__ void filter_busted_chains(CudaChain* chains, uint32_t* chain_index, CudaChain* surviving_chains,
            uint32_t* surviving_chain_index, CudaChain* long_chains, uint32_t* long_chain_index, uint32_t* histogram)
        {
            uint32_t num_threads = blockDim.x;
            uint32_t block_id = blockIdx.x;
            uint32_t index = block_id * num_threads + threadIdx.x;

            if (index >= *chain_index)
                return;
            // Survivor and long-chain counters are zeroed by the host before
            // launch.  Resetting them here only reaches one thread of one
            // block, so other blocks race the store.
            if (!is_there_still_hope(chains[index]))
            {
                //this chain is busted.  check how long it is
                //collect stats
                //only count chains 3 or longer to minimize memory accesses
                if (chains[index].m_prime_count >= 3)
                {
                    int chain_length, local_offset;
                    uint64_t base_offset;
                    get_best_fermat_chain(chains[index], base_offset, local_offset, chain_length);
                    uint32_t histogram_chain_length = min(chain_length, Cuda_sieve::chain_histogram_max);
                    if (chain_length >= 3)
                        atomicInc(&histogram[histogram_chain_length], 0xFFFFFFFF);

                    //check for winners
                    if (chain_length >= chains[index].m_min_chain_report_length)
                    {
                        uint32_t last_long_chain_index = 0;
                        if (claim_chain_slot(long_chain_index, Cuda_sieve::m_max_long_chains, last_long_chain_index))
                            long_chains[last_long_chain_index] = chains[index];
                    }
                }
            }
            else
            {
                uint32_t last_surviving_chain_index = 0;
                if (claim_chain_slot(surviving_chain_index, Cuda_sieve::m_max_chains, last_surviving_chain_index))
                    surviving_chains[last_surviving_chain_index] = chains[index];
            }
        }
        
    }
}
