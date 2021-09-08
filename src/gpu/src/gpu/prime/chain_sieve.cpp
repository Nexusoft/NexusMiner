#include "chain_sieve.hpp"
#include <primesieve.hpp>
#include <vector>
#include <queue>
#include <chrono>
#include <bitset>
#include <sstream>
#include <boost/integer/mod_inverse.hpp>


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

            return (m_prime_count + m_untested_count) >= m_min_chain_length;

            // a more complex method that screens out more chains   
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

        Sieve::Sieve()
            : m_logger{ spdlog::get("logger") }
        {
            m_sieve.resize(sieve_size);
            m_sieve_results.resize(static_cast<uint64_t>(sieve_size) * static_cast<uint64_t>(m_segment_batch_size));
            reset_stats();
            reset_sieve_batch(0);
        }

        void Sieve::generate_sieving_primes()
        {
            //generate sieving primes
            m_logger->info("Generating sieving primes up to {}...", m_sieving_prime_limit);
            auto start = std::chrono::steady_clock::now();
            primesieve::generate_primes(sieving_start_prime, m_sieving_prime_limit, &m_sieving_primes);
            auto end = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::stringstream ss;
            ss << "Done. " << m_sieving_primes.size() << " primes generated in " << std::fixed << std::setprecision(3) << elapsed.count() / 1000.0 << " seconds.";
            m_logger->info(ss.str());
        }

        void Sieve::set_sieve_start(boost::multiprecision::uint1024_t sieve_start)
        {
            //set the sieve start to the next highest multiple of 30
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
            //generate starting multiples of the sieving primes.  
            //This gets us aligned so that we may start sieving at an arbitrary starting point instead of at 0.
            //do this once at the start of a new block to initialize the sieve.
            m_multiples = {};
            m_wheel_indices = {};
            m_logger->info("Calculating starting multiples.");
            auto start = std::chrono::steady_clock::now();
            for (auto s : m_sieving_primes)
            {
                uint32_t m = get_offset_to_next_multiple(m_sieve_start, s);
                m_multiples.push_back(m);
                //where is the starting multiple relative to the wheel
                //int64_t wheel_index = (boost::integer::mod_inverse((int64_t)s, (int64_t)30) * m) % 30;
                //int wheel_lookup = sieve30_index[wheel_index];
                int64_t prime_mod_inverse = boost::integer::mod_inverse((int64_t)s, (int64_t)30);

                m_wheel_indices.push_back(prime_mod_inverse);
            }
            auto end = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            //std::stringstream ss;
            //ss << "Done. (" << std::fixed << std::setprecision(3) << elapsed.count() / 1000.0 << " seconds)";
            //m_logger->info(ss.str());
        }

        void Sieve::gpu_sieve_init()
        {
            load_sieve(m_sieving_primes.data(), m_sieving_primes.size());
        }

        void Sieve::gpu_sieve_free()
        {
            free_sieve();
        }

        //run the sieve on one segment
        void Sieve::sieve_segment()
        {
            for (std::size_t i = 0; i < m_sieving_primes.size(); i++)
            {
                uint64_t j = m_multiples[i];
                uint64_t k = m_sieving_primes[i];
                //where are we in the wheel
                int wheel_index = sieve30_index[(m_wheel_indices[i]*j) % 30];
                int next_wheel_gap = sieve30_gaps[wheel_index];
                while (j < m_segment_size)
                {
                    //cross off a multiple of the sieving prime
                    //m_sieve[j / 30] &= unset_bit_mask[j % 30];
                    uint64_t sieve_index = (j / 30) * 8 + sieve30_index[j % 30];
                    
                    m_sieve[sieve_index] = 0;
                    
                    //increment the next multiple of the current prime (rotate the wheel).
                    j += k * next_wheel_gap;  
                    wheel_index = (wheel_index + 1) % 8;
                    next_wheel_gap = sieve30_gaps[wheel_index];
                }
                //save the starting multiple and wheel index for the next segment
                //experiment to calculate future multiples
                uint32_t t = 0;
                if (m_segment_size >= m_multiples[i])
                    t = get_offset_to_next_multiple(m_segment_size - m_multiples[i], k);
                else
                    t = m_multiples[i] - m_segment_size;
                if (t != j - m_segment_size)
                {
                    std::cout << t << " " << j - m_segment_size << std::endl;
                }
                
                //end experiment
                m_multiples[i] = j - m_segment_size;
                //new experiment
                //int64_t tmp = sieve30_index[(boost::integer::mod_inverse((int64_t)k, (int64_t)30) * m_multiples[i]) % 30];
                //if (tmp != wheel_index)
                //{
                //    std::cout << tmp << " " << wheel_index << std::endl;
               // }
                //m_wheel_indices[i] = wheel_index;


            }
        }
       

        //run the sieve on the gpu
        void Sieve::sieve_batch(uint64_t low)
        {
            
            reset_sieve_batch(low);
            uint32_t sieve_results_size = m_sieve_batch_buffer_size;
            m_sieve_results.resize(m_sieve_batch_buffer_size);
            run_sieve(m_sieving_primes.data(), m_sieving_primes.size(), m_multiples.data(),
                m_wheel_indices.data(), m_sieve_results.data(), sieve_results_size, sieve_run_count*sieve_results_size/8*30);
            if (sieve_results_size != m_sieve_batch_buffer_size)
            {
                std::cout << "unexpected sieve results buffer size got "
                    << sieve_results_size << " expected " << m_sieve_batch_buffer_size << std::endl;
            }
            sieve_run_count++;
           
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

        std::uint32_t Sieve::get_segment_batch_size()
        {
            return m_segment_batch_size;
        }

        void Sieve::reset_sieve()
        {
            //fill the sieve with default values (all ones)
            std::fill(m_sieve.begin(), m_sieve.end(), 1);
            //clear the list of long chains found. 
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
            int previous_sieve_offset = 0;
            int sieve_offset = 0;
            uint64_t prime_candidate_offset = 0;

            //look ahead to do a basic density check to ensure there are enough candidates to form a chain
            //get count of the first  bytes
            int look_ahead_by = (((m_min_chain_length - 1) * maxGap + 30 - 1) / 30) * 8;
            int hits_in_look_ahead_region = 0;
            std::queue<int> look_ahead;
            look_ahead.push(0);
            for (int i = 0; i < look_ahead_by - 1; i++)
            {
                look_ahead.push(sieve[i]);
                hits_in_look_ahead_region += sieve[i];
            }
            
            for (uint64_t n = 0; n < sieve_size; n++)
            {
                //remove the oldest value from the look ahead region running sum.
                hits_in_look_ahead_region -= look_ahead.front();
                look_ahead.pop();
                //add the next look ahead byte
                if (n + look_ahead_by-1 < sieve_size)
                {
                    look_ahead.push(sieve[n + look_ahead_by - 1]);
                    hits_in_look_ahead_region += look_ahead.back();
                }
                else
                {
                    //we are at the end of the sieve. since we don't know what comes next, assume it is a prime candidate
                    look_ahead.push(1);
                    hits_in_look_ahead_region += look_ahead.back();
                }
                if (!m_chain_in_process && hits_in_look_ahead_region < m_min_chain_length)
                {
                    //not enough prime candidates in the next region of numbers to make a long enough chain

                }
                else
                {
                    sieve_offset = sieve30_offsets[n % 8];
                    prime_candidate_offset = low + n / 8 * 30 + sieve_offset;
                    if (sieve[n] == 1)
                    {
                        
                        if (m_chain_in_process)
                        {
                            uint64_t previous_prime_candidate_offset = 
                                m_current_chain.m_base_offset + m_current_chain.m_offsets[m_current_chain.length() - 1].m_offset;

                            if (prime_candidate_offset - previous_prime_candidate_offset > maxGap)
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
                    }
                    else
                    {
                        if (m_chain_in_process)
                        {
                            //accumulate the gap
                            int sieve_offset = sieve30_offsets[n % 8];
                            uint64_t prime_candidate_offset = low + n / 8 * 30 + sieve_offset;
                            uint64_t previous_prime_candidate_offset =
                                m_current_chain.m_base_offset + m_current_chain.m_offsets[m_current_chain.length() - 1].m_offset;
                            m_current_chain.m_gap_in_process = prime_candidate_offset - previous_prime_candidate_offset;
                            //only keep the chain going if the current gap is smaller than the max
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
                //test to verify the gpu primality test results vs the cpu.  this slows things down significantly.
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
            double fermat_positive_rate = 100.0 * prime_count / prime_test_actual_batch_size;
            //std::cout << "GPU batch fermat test results: " << prime_count << "/" << prime_test_actual_batch_size << " (" << fermat_positive_rate << "%)" << std::endl;
        }

        //use the cpu to prime test in batch mode.  This is slower than the gpu and is intended for debug.
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

        //calculates the probability than a random unsigned number that passed through the sieve is actually prime
        //compare with primality test stats to verify the sieve is working. 
        double Sieve::probability_is_prime_after_sieve()
        {
            int bits = 1024;
            double p_not_divisible_by_nth_prime = sieve_pass_through_rate_expected();
           
            //the probability that a large random unsieved number is prime
            double p = 1.0 / (log(2) * bits);

            //the probability that a large random number is prime after sieving with the set of small primes
            double p_after_sieve = p / p_not_divisible_by_nth_prime;

            return p_after_sieve;
        }

        //what percent of numbers do we expect to pass through the sieve
        double Sieve::sieve_pass_through_rate_expected()
        {
            //the probability that a large random number is not divisible by 2 is 1/2.
            //the probability that a large random number is not divisiby by 3 is 2/3.
            //the probability that a large random number is not divisible by 2 and 3 is 1/2*2/3 = 1/3 = 0.33.
            //the probability that a large random number is not divisible by the nth prime is 1/2*2/3*4/5*6/7*...*(pn-1)/pn

            double p_not_divisible_by_nth_prime = 1.0;
            primesieve::iterator it;
            uint64_t prime = it.next_prime();

            for (; prime < m_sieving_prime_limit; prime = it.next_prime())
                p_not_divisible_by_nth_prime *= (prime - 1.0) / prime;

            return p_not_divisible_by_nth_prime;
        }

        //count prime candidates in the sieve
        uint64_t Sieve::count_prime_candidates()
        {
            uint64_t candidate_count = 0;
            for (uint64_t n = 0; n < m_sieve_results.size(); n++)
            {
                //candidate_count += popcnt[m_sieve_results[n]];
                candidate_count += m_sieve_results[n];
            }
            return candidate_count;
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
                //if (chain.m_untested_count <= 0)  
                if (!chain.is_there_still_hope())
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

        //for debug
        uint64_t Sieve::count_fermat_primes_cpu(int sample_size)
        {
            uint64_t count = 0;
            int low = 0;
            int tests = 0;
            for (uint64_t n = 0; n < m_sieve_results.size(); n++)
            {
                //for (uint8_t b = m_sieve_results[n]; b > 0; b &= b - 1)
                //{
                    //int index_of_lowest_set_bit = boost::multiprecision::lsb(b);//std::countr_zero(b);
                    uint64_t prime_candidate_offset = low + n/8 * 30 + sieve30_offsets[n%8];
                    uint1024_t p = m_sieve_start + prime_candidate_offset;
                    if (primality_test(p))
                        count++;
                    tests++;
                //}
                if (tests >= sample_size)
                    break;
            }
            return count;
        }

        uint64_t Sieve::count_fermat_primes(int sample_size)
        {
            uint64_t prime_count = 0;
            int low = 0;
            int tests = 0;
            std::vector<uint64_t> offsets;
            for (uint64_t n = 0; n < m_sieve_results.size(); n++)
            {
                //for (uint8_t b = m_sieve_results[n]; b > 0; b &= b - 1)
                //{
                if (m_sieve_results[n] == 1)
                {
                    uint64_t prime_candidate_offset = low + n / 8 * 30 + sieve30_offsets[n % 8];
                    offsets.push_back(prime_candidate_offset);
                    tests++;
                }
                //}
                if (tests >= sample_size)
                    break;
            }
            int prime_test_actual_batch_size = offsets.size();
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
            }
            
            mpz_clear(base_as_mpz_t);

            return prime_count;
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