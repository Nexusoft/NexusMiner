#include "chain_sieve.hpp"
#include <primesieve.hpp>
#include <vector>
#include <chrono>
#include <bitset>
#include <sstream>
#include <boost/integer/mod_inverse.hpp>
#include "../cuda_prime/fermat_test.cuh"

namespace nexusminer {
    namespace gpu
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

        //iterate through the chain and return the offset and length of the longest fermat chain that meets the mininmum gap requirement
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
                //quit if there are not enough candidates remaining to make a full fermat chain
               /* if (static_cast<int>(m_offsets.size()) - i < (m_min_chain_length - chain_length))
                    return;*/
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
            return;
        }


        //is it still possible for this chain to produce a fermat chain greater than the minimum length?  
        //use this to determine if we should give up on this chain or keep testing
        bool Chain::is_there_still_hope()
        {


            if ((m_prime_count + m_untested_count) < m_min_chain_length)
                return false;
            else
                return true;

            //if (m_untested_count == length())
            //{
            //    //only should happen on a brand new chain. 
            //    return true;
            //}


            ////create a fake temporary chain where all untested candidates pass
            //Chain temp_chain(*this);
            //for (auto& offset : temp_chain.m_offsets)
            //{
            //    if (offset.m_fermat_test_status == Fermat_test_status::untested)
            //    {
            //        offset.m_fermat_test_status = Fermat_test_status::pass;
            //    }
            //}
            //int max_possible_length, offset;
            //uint64_t base_offset;
            //temp_chain.get_best_fermat_chain(base_offset, offset, max_possible_length);
            //return (max_possible_length >= m_min_chain_length);

        }

        //get the next untested fermat candidate.  if there are none return false.
        bool Chain::get_next_fermat_candidate(uint64_t& base_offset, int& offset)
        {
            //This finds the first untested prime candidate.
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

        //set the fermat test status of an offset.  if the offset is not found return false
        //check if the chain meets the minimum requirement or should be discarded.
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

            //chain offset vector must be sorted for this to work correctly
            //std::pair<std::vector<Chain_offset>::iterator, std::vector<Chain_offset>::iterator>  candidate;
            //Chain_offset co{ offset };
            //candidate = std::equal_range(m_offsets.begin(), m_offsets.end(), offset);

            //if (candidate.first == candidate.second)
            //{
            //    //offset was not found
            //    return false;
            //}
            //else
            //{
            //    if (is_prime)
            //    {
            //        //std::cout << "found a prime at offset " << offset << " index " << candidate.first - m_offsets.begin() << " offset=" << candidate.first->m_offset << std::endl;
            //    }
            //    //offset was found.  set the primality test status
            //    candidate.first->m_fermat_test_status = is_prime?Fermat_test_status::pass:Fermat_test_status::fail;   
            //    return true;
            //}

        }

        void Chain::push_back(int offset)
        {
            Chain_offset chain_offset{ offset };
            m_offsets.push_back(chain_offset);
            m_untested_count++;
            //m_offset_map[offset] = m_offsets.size();
        }




        Sieve::Sieve()
            : m_logger{ spdlog::get("logger") }
        {
            m_sieve.resize(sieve_size);
            reset_stats();
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
                //std::cout << "multiples[" << i << "]: " << j - segment_size << std::endl;

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
            //clear the list of chains
            //m_chain = {};
            m_long_chain_starts = {};

        }

        /*std::vector<std::uint64_t> Sieve::get_valid_chain_starting_offsets()
        {
            return m_long_chain_starts;
        }*/

        //void Sieve::set_chain_length_threshold(int min_chain_length)
        //{
        //    m_min_chain_length_threshold = min_chain_length;
        //}

        void Sieve::reset_stats()
        {
            m_chain_histogram = std::vector<std::uint32_t>(10, 0);
            m_fermat_test_count = 0;
            m_fermat_prime_count = 0;
            m_chain_count = 0;
            m_chain_candidate_max_length = 0;
            m_chain_candidate_total_length = 0;


        }

        //search the sieve for chains.  Chains can cross segment boundaries.
        void Sieve::find_chains(uint64_t sieve_size, uint64_t low)
        {
            for (uint64_t n = 0; n < sieve_size; n++)
            {
                if (m_sieve[n] == 0)
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
                    for (uint8_t b = m_sieve[n]; b > 0; b &= b - 1)
                    {
                        int index_of_lowest_set_bit = boost::multiprecision::lsb(b);//std::countr_zero(b);
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


        //batch process the list of prime candidates to be fermat tested.  
        void Sieve::primality_batch_test()
        {
            bool cpu_verify = false;
            uint64_t prime_count = 0;
            int prime_test_actual_batch_size = m_chain.size();

            std::vector<uint64_t> offsets;
            for (auto i = 0; i < prime_test_actual_batch_size; i++)
            {
                uint64_t base_offset;
                int offset;
                bool success = m_chain[i].get_next_fermat_candidate(base_offset, offset);
                if (!success)
                    m_logger->debug("error getting next fermat candidate.");
                offsets.push_back(base_offset + static_cast<uint64_t>(offset));
            }

            mpz_int base_as_mpz_int = static_cast<mpz_int>(m_sieve_start);
            mpz_t base_as_mpz_t;
            mpz_init(base_as_mpz_t);
            mpz_set(base_as_mpz_t, base_as_mpz_int.backend().data());
            std::vector<uint8_t> primality_test_results;
            primality_test_results.resize(prime_test_actual_batch_size);

            run_primality_test(base_as_mpz_t, offsets.data(), prime_test_actual_batch_size, primality_test_results.data());

            for (auto i = 0; i < prime_test_actual_batch_size; i++)
            {
                if (primality_test_results[i] == 1)
                    ++prime_count;
                m_chain[i].update_fermat_status(primality_test_results[i]);
                if (cpu_verify)
                {
                    bool is_prime_cpu = primality_test(m_sieve_start + offsets[i]);
                    if (is_prime_cpu != (primality_test_results[i] == 1))
                    {
                        m_logger->debug("GPU/CPU primality test mismatch at offset {} {}", i, offsets[i]);
                    }
                }
            }
            m_fermat_prime_count += prime_count;
            m_fermat_test_count += prime_test_actual_batch_size;
            mpz_clear(base_as_mpz_t);
            //std::cout << "GPU batch fermat test results: " << prime_count << "/" << prime_test_actual_batch_size << " (" << 100.0 * prime_count / prime_test_actual_batch_size << "%)" << std::endl;

        }

        void Sieve::primality_batch_test_cpu()
        {
            uint64_t prime_count = 0;
            for (auto& chain : m_chain)
            {
                uint64_t base_offset;
                int offset;
                bool success = chain.get_next_fermat_candidate(base_offset, offset);
                if (!success)
                {
                    m_logger->debug("error getting next fermat candidate.");
                }
                boost::multiprecision::uint1024_t candidate = m_sieve_start + base_offset + offset;
                bool is_prime = primality_test(candidate);
                /*uint1024_t T("0x0000005ff320ec9f9599b9cb0156c793f61060c8a8c49185df9d25603e37259c2f0213d6d96745bbbbe7ea1e4e9da371aeeb5d20c204c22a038b10957b53c67d9eb3a00acfaeb6ccd4c231a8088d5a5745e19f70387a7d91463d9b318a1f0503819a32f5fa32cf3579c7d6a3546cbdceaa364cfa2e989defeb4f5fe29de687cc");
                uint64_t nNonce = 4933493377870005061;
                if (candidate >= T + nNonce && candidate <= T + nNonce + 100)
                {
                    std::cout << "base offset: " << base_offset << " offset: " << offset << " is prime: " << is_prime << std::endl;
                }*/
                chain.update_fermat_status(is_prime);
                if (is_prime)
                {
                    prime_count++;
                }
            }
            std::cout << "CPU batch fermat test results: " << prime_count << "/" << m_chain.size() << " (" << 100.0 * prime_count / m_chain.size() << "%)" << std::endl;

        }

        uint64_t Sieve::get_current_chain_list_length()
        {
            return m_chain.size();
        }

        //search for winners.  delete finished or hopeless chains.
        void Sieve::clean_chains()
        {
            size_t chain_count_before = m_chain.size();
            for (auto& chain : m_chain)
            {
                uint64_t base_offset;
                int offset, length;
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
                    chain.m_chain_state = Chain::Chain_state::complete;
                                    
                }
                else if (!chain.is_there_still_hope())
                {
                    chain.m_chain_state = Chain::Chain_state::complete;
                }
            }
            //remove completed chains
            m_chain.erase(std::remove_if(m_chain.begin(), m_chain.end(),
                [](Chain& c) {return c.m_chain_state == Chain::Chain_state::complete; }), m_chain.end());
            size_t chain_count_after = m_chain.size();
            //std::cout << "removed " << chain_count_before - chain_count_after << " chains" << std::endl;

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

        bool operator < (const Chain::Chain_offset& c1, const Chain::Chain_offset& c2)
        {
            return c1.m_offset < c2.m_offset;
        }

        bool operator == (const Chain::Chain_offset& c1, const Chain::Chain_offset& c2)
        {
            return c1.m_offset == c2.m_offset;
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