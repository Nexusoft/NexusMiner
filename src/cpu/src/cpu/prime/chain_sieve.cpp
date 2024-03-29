#include "chain_sieve.hpp"
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

            return ((m_prime_count + m_untested_count) >= m_min_chain_length);
            

                // a more complex method that screens out more chains   
                //create a fake temporary chain where all untested candidates pass
                /*Chain temp_chain(*this);
                for (auto& offset : temp_chain.m_offsets)
                {
                    if (offset.m_fermat_test_status == Fermat_test_status::untested)
                    {
                        offset.m_fermat_test_status = Fermat_test_status::pass;
                    }
                }
                int max_possible_length, offset;
                uint64_t base_offset;
                temp_chain.get_best_fermat_chain(base_offset, offset, max_possible_length);
                return (max_possible_length >= m_min_chain_length);*/


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
            //generate sieving primes
            m_logger->info("Generating sieving primes up to {}...", sieving_prime_limit);
            auto start = std::chrono::steady_clock::now();
            primesieve::generate_primes(sieving_start_prime, sieving_prime_limit, &m_sieving_primes);
            auto end = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::stringstream ss;
            ss << "Done. " << m_sieving_primes.size() << " primes generated in " << std::fixed << std::setprecision(3) << elapsed.count() / 1000.0 << " seconds.";
            m_logger->info(ss.str());
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

        void Sieve::calculate_starting_multiples()
        {
            //generate starting multiples of the sieving primes
            m_multiples = {};
            m_wheel_indices = {};
            m_logger->info("Calculating starting multiples.");
            auto start = std::chrono::steady_clock::now();
            for (auto s : m_sieving_primes)
            {
                uint32_t m = get_offset_to_next_multiple(m_sieve_start, s);
                m_multiples.push_back(m);
                //where is the starting multiple relative to the wheel
                int wheel_index = (boost::integer::mod_inverse((int)s, 30) * m) % 30;
                m_wheel_indices.push_back(sieve30_index[wheel_index]);
            }
            auto end = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            //std::stringstream ss;
            //ss << "Done. (" << std::fixed << std::setprecision(3) << elapsed.count() / 1000.0 << " seconds)";
            //m_logger->info(ss.str());
        }

        void Sieve::sieve_segment()
        {
            for (std::size_t i = 0; i < m_sieving_primes.size(); i++)
            {
                uint32_t j = m_multiples[i];
                uint32_t k = m_sieving_primes[i];
                //where are we in the wheel
                int wheel_index = m_wheel_indices[i];
                int next_wheel_gap = sieve30_gaps[wheel_index];
                while (j < m_segment_size)
                {
                    m_sieve[j / 30] &= unset_bit_mask[j % 30];
                    //increment the next multiple of the current prime (rotate the wheel).
                    j += k * next_wheel_gap;
                    wheel_index = (wheel_index + 1) % 8;
                    next_wheel_gap = sieve30_gaps[wheel_index];
                }
                //save the starting multiple and wheel index for the next segment
                m_multiples[i] = j - m_segment_size;
                m_wheel_indices[i] = wheel_index;
            }
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

        std::uint32_t Sieve::get_segment_size()
        {
            return m_segment_size;
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
            m_chain_histogram = std::vector<std::uint32_t>(10, 0);
            m_fermat_test_count = 0;
            m_fermat_prime_count = 0;
            m_chain_count = 0;
            m_chain_candidate_max_length = 0;
            m_chain_candidate_total_length = 0;
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
                if (!m_chain_in_process && hits_next_four_bytes < m_min_chain_length)
                {
                    //not enough prime candidates in the next 120 numbers to make a long enough chain

                }
                else if (sieve[n] == 0)
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

        void Sieve::close_chain()
        {
            if (m_current_chain.length() >= m_current_chain.m_min_chain_length)
            {
                //we found a chain candidate.  save it.
                m_chain.push_back(m_current_chain);
                m_chain_count++;
                m_chain_candidate_max_length = std::max(m_current_chain.length(), m_chain_candidate_max_length);
                m_chain_candidate_total_length += m_current_chain.length();
            }
            m_current_chain.close();
            m_chain_in_process = false;
        }

        void Sieve::open_chain(uint64_t base_offset)
        {
            //reset chain in process to the default
            m_current_chain = { base_offset };
            m_chain_in_process = true;
        }

        //get the next prime to test from each chain 
        void Sieve::test_chains()
        {
            for (auto i = 0; i < m_chain.size(); i++)
            {
                bool there_is_still_hope = true;
                int count = 0;
                uint64_t base_offset;
                int offset;
                while (there_is_still_hope)
                {
                    if (m_chain[i].get_next_fermat_candidate(base_offset, offset))
                    {
                        boost::multiprecision::uint1024_t candidate = m_sieve_start + base_offset + offset;
                        bool is_prime = primality_test(candidate);
                        m_chain[i].update_fermat_status(is_prime);
                        if (is_prime)
                        {
                            count++;
                        }
                        there_is_still_hope = m_chain[i].is_there_still_hope();
                    }
                }
                m_chain[i].m_chain_state = Chain::Chain_state::complete;
                if (count > 0)
                {
                    int length;
                    m_chain[i].get_best_fermat_chain(base_offset, offset, length);
                    
                    //collect stats
                    int count = std::min(static_cast<size_t>(length), m_chain_histogram.size());
                    m_chain_histogram[count]++;
                    
                    if (length >= m_chain[i].m_min_chain_report_length)
                    {
                        //we found a long chain.  save it.
                        m_logger->info("Found a fermat chain of length {}.", length);
                        m_long_chain_starts.push_back(base_offset + offset);

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
                        //collect stats
                        int count = std::min(static_cast<size_t>(length), m_chain_histogram.size());
                        m_chain_histogram[count]++;
                    }
                    if (length >= chain.m_min_chain_report_length)
                    {
                        //we found a long chain.  save it.
                        m_logger->info("Found a fermat chain of length {}.", length);
                        m_long_chain_starts.push_back(base_offset + offset);
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