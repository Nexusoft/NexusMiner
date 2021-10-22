#include "sieve.hpp"
#include "chain.hpp"
#include <primesieve.hpp>
#include <vector>
#include <queue>
#include <chrono>
#include <bitset>
#include <sstream>
#include <boost/integer/mod_inverse.hpp>
#include "small_sieve_tools.hpp"
#include "fastmod.h"


namespace nexusminer {
    namespace gpu
    {
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
            //small primes are hardcoded in sieve.cu
            m_logger->info("Generating sieving primes...");
            auto start = std::chrono::steady_clock::now();
            //generate medium sieving primes
            primesieve::generate_n_primes(Cuda_sieve::m_medium_prime_count, m_sieving_start_prime, &m_sieving_primes);
            //the largest medium sieving prime
            if (m_sieving_primes.size() > 0)
                m_sieving_prime_limit = m_sieving_primes.back();
            else
                m_sieving_prime_limit = 0;
            //reorder the primes
            /*std::vector<Cuda_sieve::sieve_word_t> temp_primes;
            uint32_t index = 0;
            const uint32_t primes_per_thread = (m_sieving_primes.size() + Cuda_sieve::m_threads_per_block - 1) / Cuda_sieve::m_threads_per_block;

            for (auto i = 0; i < m_sieving_primes.size(); i++)
            {
                index = (i % primes_per_thread) * Cuda_sieve::m_threads_per_block + i / primes_per_thread;
                temp_primes.push_back(m_sieving_primes[index]);
            }
            m_sieving_primes = temp_primes;*/
            primesieve::generate_n_primes(Cuda_sieve::m_large_prime_count, m_sieving_prime_limit + 1ull, &m_large_sieving_primes);
            if (m_large_sieving_primes.size() > 0)
                m_large_prime_limit = m_large_sieving_primes.back();
            else
                m_large_prime_limit = m_sieving_prime_limit;
            auto end = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::stringstream ss;
            ss << "Done. " << m_sieving_primes.size() + m_large_sieving_primes.size() << " primes generated in " << std::fixed << std::setprecision(3) << elapsed.count() / 1000.0 << " seconds.";
            m_logger->info(ss.str());

        }

        void Sieve::set_sieve_start(boost::multiprecision::uint1024_t sieve_start)
        {
            //set the sieve start to the next highest multiple of the sieve alignment value
            if (sieve_start % Cuda_sieve::m_sieve_alignment > 0)
            {
                sieve_start += Cuda_sieve::m_sieve_alignment - (sieve_start % Cuda_sieve::m_sieve_alignment);
            }
            sieve_start += Cuda_sieve::m_sieve_alignment_offset;
            m_sieve_start = sieve_start;
        }

        boost::multiprecision::uint1024_t Sieve::get_sieve_start()
        {
            return m_sieve_start;
        }

        void Sieve::calculate_starting_multiples()
        {

            //calculate the starting offsets of the small primes relative to the sieve start
            m_small_prime_offsets = {};
            for (int i = 0; i < Cuda_sieve::m_small_prime_count; i++)
            {
                int s = Cuda_sieve::m_small_primes[i];
                uint32_t offset = static_cast<uint32_t>(((m_sieve_start / m_sieve_range_per_byte) % s)) * m_sieve_range_per_byte;
                m_small_prime_offsets.push_back(offset); 
            }

            //generate starting multiples of the sieving primes.  
            //This gets us aligned so that we may start sieving at an arbitrary starting point instead of at 0.
            //do this once at the start of a new block to initialize the sieve.
            m_multiples = {};
            m_logger->info("Calculating starting multiples.");
            auto start = std::chrono::steady_clock::now();
            for (auto s : m_sieving_primes)
            {
                uint32_t m = get_offset_to_next_multiple(m_sieve_start, s);
                m_multiples.push_back(m);
            }
            auto end = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            //std::stringstream ss;
            //ss << "Done. (" << std::fixed << std::setprecision(3) << elapsed.count() / 1000.0 << " seconds)";
            //m_logger->info(ss.str());
            m_large_multiples = {};
            for (const auto& p : m_large_sieving_primes)
            {
                uint32_t m = get_offset_to_next_multiple(m_sieve_start, p);
                m_large_multiples.push_back(m);
            }
        }

        void Sieve::gpu_sieve_load(uint16_t device=0)
        {
            m_cuda_sieve.load_sieve(m_sieving_primes.data(), m_sieving_primes.size(), m_large_sieving_primes.data(), m_sieve_batch_buffer_size, device);
            m_cuda_sieve_allocated = true;
            reset_stats();
        }

        void Sieve::gpu_sieve_init()
        {
            m_cuda_sieve.init_sieve(m_multiples.data(), m_small_prime_offsets.data(), m_large_multiples.data());
            m_sieve_run_count = 0;
        }

        void Sieve::gpu_fermat_test_set_base_int(boost::multiprecision::uint1024_t base_big_int)
        {
            boost::multiprecision::mpz_int base_as_mpz_int = static_cast<boost::multiprecision::mpz_int>(base_big_int);
            mpz_t base_as_mpz_t;
            mpz_init(base_as_mpz_t);
            mpz_set(base_as_mpz_t, base_as_mpz_int.backend().data());
            m_cuda_prime_test.set_base_int(base_as_mpz_t);
            mpz_clear(base_as_mpz_t);

        }

        uint64_t Sieve::gpu_get_prime_candidate_count()
        {
            uint64_t prime_candidate_count = 0;
            m_cuda_sieve.get_prime_candidate_count(prime_candidate_count);
            return prime_candidate_count;
        }

        void Sieve::gpu_get_sieve()
        {
            m_cuda_sieve.get_sieve(m_sieve_results.data());
        }

        void Sieve::gpu_fermat_test_init(uint16_t device)
        {
            
            m_cuda_prime_test.fermat_init(m_fermat_test_batch_size_max, device);
            gpu_fermat_test_set_base_int(m_sieve_start);
            //pass a device pointer to the list of chains to the fermat test
            //do this after initializing the gpu sieve so the pointer is valid
            CudaChain* chains;
            uint32_t* chain_count;
            m_cuda_sieve.get_chain_pointer(chains, chain_count);
            m_cuda_prime_test.set_chain_ptr(chains, chain_count);

        }

        void Sieve::gpu_sieve_free()
        {
            m_cuda_sieve.free_sieve();
            m_cuda_sieve_allocated = false;
            
        }

        void Sieve::gpu_fermat_free()
        {
            m_cuda_prime_test.fermat_free();
        }

        void Sieve::gpu_sieve_small_primes(uint64_t sieve_start_offset)
        {
            m_cuda_sieve.run_small_prime_sieve(sieve_start_offset);
        }

        void Sieve::gpu_sieve_large_primes(uint64_t sieve_start_offset)
        {
            m_cuda_sieve.run_large_prime_sieve(sieve_start_offset);
        }


       
       

        //run the sieve on the gpu
        void Sieve::sieve_batch(uint64_t low)
        {
            
            reset_sieve_batch(low);
            m_cuda_sieve.run_sieve(m_sieve_run_count* m_sieve_batch_buffer_size * m_sieve_range_per_word);
            m_sieve_run_count++;
           
        }

        //experimental.  bit sieve with small primes prototype for cuda port
        //small primes hit the sieve every word.  iterate by sieve word and cross off small primes using precomputed masks
        void Sieve::sieve_small_primes()
        {
            //generate "soft" prime masks.  Todo compare the performance vs hardcoding the masks.  
            Small_sieve_tools small_sieve_tool;
            std::vector<uint32_t> p7 = small_sieve_tool.prime_mask(7);
            std::vector<uint32_t> p11 = small_sieve_tool.prime_mask(11);
            std::vector<uint32_t> p13 = small_sieve_tool.prime_mask(13);
            std::vector<uint32_t> p17 = small_sieve_tool.prime_mask(17);
            std::vector<uint32_t> p19 = small_sieve_tool.prime_mask(19);
            std::vector<uint32_t> p23 = small_sieve_tool.prime_mask(23);
            std::vector<uint32_t> p29 = small_sieve_tool.prime_mask(29);
            std::vector<uint32_t> p31 = small_sieve_tool.prime_mask(31);

            //small_sieve_tool.print_code(199);

            
            //const uint32_t p7[7] = { 0x02201081, 0x08044002, 0x20108108, 0x04400220, 0x10810804, 0x40022010, 0x81080440 };

            uint64_t start_offset = 0;
            uint32_t sieve_words = m_sieve.size();
            uint32_t increment = m_sieve_range_per_word;
            //Equivalent offsets to the start of the sieve mode the prime.  Use to avoid bignum math in the sieve.
            //Compute these once per block.  
            uint32_t offset7 = static_cast<uint32_t>(((m_sieve_start / 30) % 7)) * 30;
            uint32_t offset11 = static_cast<uint32_t>(((m_sieve_start / 30) % 11)) * 30;
            uint32_t offset13 = static_cast<uint32_t>(((m_sieve_start / 30) % 13)) * 30;
            uint32_t offset17 = static_cast<uint32_t>(((m_sieve_start / 30) % 17)) * 30;
            uint32_t offset19 = static_cast<uint32_t>(((m_sieve_start / 30) % 19)) * 30;
            uint32_t offset23 = static_cast<uint32_t>(((m_sieve_start / 30) % 23)) * 30;
            uint32_t offset29 = static_cast<uint32_t>(((m_sieve_start / 30) % 29)) * 30;
            uint32_t offset31 = static_cast<uint32_t>(((m_sieve_start / 30) % 31)) * 30;


           //std::cout << "offset 7 = " << offset7 << std::endl;
           //std::cout << "offset 11 = " << offset11 << std::endl;
           // std::cout << "offset 31 = " << offset31 << std::endl;


            for (uint32_t i = 0; i < sieve_words; i++)
            {
                //if the modulus operations cause a performance hit 
                //we could reorder the mask words and use an increment + compare instead.
                uint32_t inc = i * increment;
                uint16_t index7 = (offset7 + inc) % 7;
                uint16_t index11 = (offset11 + inc) % 11;
                uint16_t index13 = (offset13 + inc) % 13;
                uint16_t index17 = (offset17 + inc) % 17;
                uint16_t index19 = (offset19 + inc) % 19;
                uint16_t index23 = (offset23 + inc) % 23;
                uint16_t index29 = (offset29 + inc) % 29;
                uint16_t index31 = (offset31 + inc) % 31;


                /*uint32_t mod7 = static_cast<uint32_t>(((m_sieve_start + i * increment) % 7));
                std::cout << "sieve start + " << i* increment << " mod 7 = " << mod7 << std::endl;
                std::cout << "offset7 + " << i * increment << " mod 7 = " << index7 << std::endl;

               uint32_t mod11 = static_cast<uint32_t>(((m_sieve_start + i * increment) % 11));
                std::cout << "sieve start + " << i * increment << " mod 11 = " << mod11 << std::endl;
                std::cout << "offset11 + " << i * increment << " mod 11 = " << index11 << std::endl;*/

                /*uint32_t mod31 = static_cast<uint32_t>(((m_sieve_start + i * increment) % 31));
                std::cout << "sieve start + " << i * increment << " mod 31 = " << mod31 << std::endl;
                std::cout << "offset31 + " << i * increment << " mod 31 = " << index31 << std::endl;*/
                
                //m_sieve[i] &= p7[index7];  
                m_sieve[i] &= p11[index11];
                //m_sieve[i] &= p13[index13];
                //m_sieve[i] &= p17[index17];
                //m_sieve[i] &= p19[index19];
                //m_sieve[i] &= p23[index23];
                //m_sieve[i] &= p29[index29];
                //m_sieve[i] &= p31[index31];
                //std::cout << "here" << std::endl;
            }
           // std::cout << "here" << std::endl;
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
            std::fill(m_sieve.begin(), m_sieve.end(), 0xFFFFFFFF);
            //clear the list of long chains found. 
            m_long_chain_starts = {};
        }

        void Sieve::reset_sieve_batch(uint64_t low)
        {
            
            m_sieve_batch_start_offset = low;
            m_sieve_results = {};
            m_sieve_results.resize(m_sieve_batch_buffer_size);
            m_long_chain_starts = {};
        }

        void Sieve::reset_batch_run_count()
        {
            m_sieve_run_count = 0;
        }

        void Sieve::clear_chains()
        {
            m_chain = {};
        }

        void Sieve::reset_stats()
        {
            m_chain_histogram = std::vector<std::uint32_t>(Cuda_sieve::chain_histogram_max+1, 0);
            m_fermat_test_count = 0;
            m_fermat_prime_count = 0;
            m_chain_count = 0;
            m_chain_candidate_max_length = 0;
            m_chain_candidate_total_length = 0;
            m_trial_division_chains_busted = 0;
            if(m_cuda_sieve_allocated)
                m_cuda_sieve.reset_stats();
        }

        //find chains on the gpu
        void Sieve::find_chains()
        {
            m_cuda_sieve.find_chains();
        }

        void Sieve::get_chains()
        {
            uint32_t chain_count = 0;
            m_cuda_chains = {};
            m_cuda_chains.resize(m_cuda_sieve.m_max_chains);
            m_cuda_sieve.get_chains(m_cuda_chains.data(), chain_count);
            //convert gpu chains to cpu chains
            for (auto i = 0; i < chain_count; i++)
            {
                Chain chain(m_cuda_chains[i].m_base_offset);
                for (auto j = 0; j < m_cuda_chains[i].m_offset_count; j++)
                {
                    if (j>0)
                        chain.push_back(m_cuda_chains[i].m_offsets[j]);
                    chain.m_offsets[j].m_fermat_test_status = m_cuda_chains[i].m_fermat_test_status[j];
                }
                chain.m_untested_count = m_cuda_chains[i].m_untested_count;
                chain.m_prime_count = m_cuda_chains[i].m_prime_count;
                m_chain_candidate_max_length = std::max(chain.length(), m_chain_candidate_max_length);
                m_chain_candidate_total_length += chain.length();
                m_chain.push_back(chain);
            }
            m_chain_count += chain_count;
        
        }

        //sort chains in order of base offset for debug
        void Sieve::sort_chains()
        {
            std::sort(m_chain.begin(), m_chain.end(),
                [](const Chain& a, const Chain& b)
                {
                    return a.m_base_offset < b.m_base_offset;
                });

        }

        void Sieve::get_long_chains()
        {
            uint32_t chain_count = 0;
            m_cuda_chains = {};
            m_cuda_chains.resize(m_cuda_sieve.m_max_long_chains);
            m_cuda_sieve.get_long_chains(m_cuda_chains.data(), chain_count);
            //convert gpu chains to cpu chains
            for (auto i = 0; i < chain_count; i++)
            {
                Chain chain(m_cuda_chains[i].m_base_offset);
                for (auto j = 0; j < m_cuda_chains[i].m_offset_count; j++)
                {
                    if (j > 0)
                        chain.push_back(m_cuda_chains[i].m_offsets[j]);
                    chain.m_offsets[j].m_fermat_test_status = m_cuda_chains[i].m_fermat_test_status[j];
                    
                }
                chain.m_untested_count = m_cuda_chains[i].m_untested_count;
                chain.m_prime_count = m_cuda_chains[i].m_prime_count;
                uint64_t base_offset;
                int offset, length;
                chain.get_best_fermat_chain(base_offset, offset, length);
                m_logger->info("Found a fermat chain of length {}.", length);
                //std::cout << chain.str() << std::endl;
                m_long_chain_starts.push_back(base_offset + offset);
            }
        }

        void Sieve::gpu_clean_chains()
        {
            m_cuda_sieve.clean_chains();
        }

        void Sieve::gpu_run_fermat_chain_test()
        {
            m_cuda_prime_test.fermat_chain_run();
        }

        void Sieve::gpu_get_fermat_stats(uint64_t& tests, uint64_t& passes)
        {
            m_cuda_prime_test.get_stats(tests, passes);
        }

        void Sieve::gpu_reset_fermat_stats()
        {
            m_cuda_prime_test.reset_stats();
        }

        uint32_t Sieve::get_chain_count()
        {
            uint32_t chain_count;
            m_cuda_sieve.get_chain_count(chain_count);
            return chain_count;
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
        void Sieve::primality_batch_test(uint16_t device=0)
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

           /* boost::multiprecision::mpz_int base_as_mpz_int = static_cast<boost::multiprecision::mpz_int>(m_sieve_start);
            mpz_t base_as_mpz_t;
            mpz_init(base_as_mpz_t);
            mpz_set(base_as_mpz_t, base_as_mpz_int.backend().data());*/
            std::vector<uint8_t> primality_test_results;
            primality_test_results.resize(prime_test_actual_batch_size);
            m_cuda_prime_test.set_offsets(offsets.data(), prime_test_actual_batch_size);
            m_cuda_prime_test.fermat_run( );
            m_cuda_prime_test.get_results(primality_test_results.data());
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
            //mpz_clear(base_as_mpz_t);
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

        uint64_t Sieve::get_cuda_chain_list_length()
        {
            return m_cuda_chains.size();
        }

        //calculates the probability than a random unsigned number that passed through the sieve is actually prime
        //compare with primality test stats to verify the sieve is working. 
        double Sieve::probability_is_prime_after_sieve(double bits)
        {
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

            for (; prime <= m_large_prime_limit; prime = it.next_prime())
                p_not_divisible_by_nth_prime *= (prime - 1.0) / prime;

            return p_not_divisible_by_nth_prime;
        }

        //calculate the probability of finding a chain of length n at bitwidth bits
        double Sieve::expected_chain_density(int n, int bits)
        {
            double density = 0;
            if (n >= 1 && n <= 11)
                density = hardy_littlewood_constants[n] / std::pow((log(2.0) * bits), n);
            
            return density;
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

        //get a list of prime candidates from the sieve
        std::vector<uint64_t> Sieve::get_prime_candidate_offsets()
        {
            std::vector<uint64_t> offsets;
            for (uint64_t n = 0; n < m_sieve_results.size(); n++)
            {
                for (auto b = m_sieve_results[n]; b > 0; b &= b - 1)
                {
                    int index_of_lowest_set_bit = boost::multiprecision::lsb(b);//std::countr_zero(b);
                    uint64_t prime_candidate_offset = n * m_sieve_range_per_word +
                        (index_of_lowest_set_bit / 8) * static_cast<uint64_t>(m_sieve_range_per_byte) +
                        sieve30_offsets[index_of_lowest_set_bit % 8];
                    offsets.push_back(prime_candidate_offset);

                }
            }
            return offsets;
        }

        std::vector<uint32_t> Sieve::get_sieving_primes()
        {
            return m_sieving_primes;
        }

        
        std::vector<Sieve::sieve_word_t> Sieve::get_sieve()
        {
            return m_sieve;
        }

        //experimental - filter chain using trial division with large primes prior to fermat test. 
        bool Sieve::chain_trial_division(Chain& chain)
        {
            uint64_t base_offset = chain.m_base_offset;
            int chain_offset;
            for (auto j = 0; j < m_large_sieving_primes.size(); j++)
            {
                uint32_t remainder = (base_offset + m_large_multiples[j]) % m_large_sieving_primes[j];
                uint64_t q = (base_offset + m_large_multiples[j]) * m_large_prime_mod_constants[j];
                //uint32_t remainder2 = (base_offset + m_large_multiples[j]) - q * m_large_sieving_primes[j];
                //remainder2 = remainder2 == m_large_sieving_primes[j] ? 0 : remainder2;
                //if (remainder != remainder2)
                //    std::cout << "mod mismatch expected " << remainder << " got " << q << " R " << remainder2 << std::endl;
                remainder = remainder > 0 ? m_large_sieving_primes[j] - remainder : 0;
                for (auto i = 0; i < chain.length(); i++)
                {
                    chain.get_next_fermat_candidate(base_offset, chain_offset);
                    if (chain_offset == remainder)
                    {
                        chain.update_fermat_status(false);
                        if (!chain.is_there_still_hope())
                            return false;
                        break;
                    }
                }
                
            }
            return true;
        }

        void Sieve::gpu_get_stats()
        {
            m_cuda_sieve.get_stats(m_chain_histogram.data(), m_chain_count);
        }

        void Sieve::gpu_sieve_synchronize()
        {
            m_cuda_sieve.synchronize();
        }

        void Sieve::gpu_fermat_synchronize()
        {
            m_cuda_prime_test.synchronize();
        }

        void Sieve::do_chain_trial_division_check()
        {
            for (auto& chain : m_chain)
            {
                if (!chain_trial_division(chain))
                {
                    m_trial_division_chains_busted++;
                    //m_logger->info("chain busted by trial division.");
                }
            }
            clean_chains();
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
                    //std::cout << chain.str() << std::endl;
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
                        //std::cout << chain.str() << std::endl;
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
            int tests = 0;

            std::vector<uint64_t> offsets = get_prime_candidate_offsets();
            for (auto i = 0; i < offsets.size(); i++)
            {
                boost::multiprecision::uint1024_t p = m_sieve_start + offsets[i];
                if (primality_test(p))
                    count++;
                tests++;
                if (tests >= sample_size)
                    break;
            }

            return count;
        }

        uint64_t Sieve::count_fermat_primes(int sample_size, uint16_t device)
        {
            uint64_t prime_count = 0;
            int tests = 0;
            bool cpu_verify = false;
            std::vector<uint64_t> offsets;
            for (uint64_t n = 0; n < m_sieve_results.size(); n++)
            {
                for (auto b = m_sieve_results[n]; b > 0; b &= b - 1)
                {
                    int index_of_lowest_set_bit = boost::multiprecision::lsb(b);//std::countr_zero(b);
                    uint64_t prime_candidate_offset = n * m_sieve_range_per_word +
                        (index_of_lowest_set_bit / 8) * static_cast<uint64_t>(m_sieve_range_per_byte) +
                        sieve30_offsets[index_of_lowest_set_bit % 8];
                    offsets.push_back(prime_candidate_offset);
                    tests++;
                
                }
                if (tests >= sample_size)
                    break;
            }
            int prime_test_actual_batch_size = offsets.size();
            
            std::vector<uint8_t> primality_test_results;
            primality_test_results.resize(prime_test_actual_batch_size);
                        
            m_cuda_prime_test.set_offsets(offsets.data(), prime_test_actual_batch_size);
            m_cuda_prime_test.fermat_run();
            m_cuda_prime_test.get_results(primality_test_results.data());
            for (auto i = 0; i < prime_test_actual_batch_size; i++)
            {
                if (cpu_verify)
                {
                    boost::multiprecision::uint1024_t p = m_sieve_start + offsets[i];
                    uint8_t cpu_is_prime = primality_test(p) ? 1 : 0;
                    if (cpu_is_prime != primality_test_results[i])
                    {
                        m_logger->error("cpu/gpu primality test mismatch at offset {}. GPU {}/CPU {}.", offsets[i],
                            primality_test_results[i], cpu_is_prime);
                    }
                }
                if (primality_test_results[i] == 1)
                {
                    ++prime_count;
                }
            }
            

            return prime_count;
        }

        bool Sieve::primality_test(boost::multiprecision::uint1024_t p)
        {
            //gmp powm is about 7 times faster than boost backend
            boost::multiprecision::mpz_int base = 2;
            boost::multiprecision::mpz_int result;
            boost::multiprecision::mpz_int p1 = static_cast<boost::multiprecision::mpz_int>(p);
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