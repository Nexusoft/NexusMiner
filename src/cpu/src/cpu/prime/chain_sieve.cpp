#include "chain_sieve.hpp"
#include <primesieve.hpp>
#include <vector>
#include <chrono>
#include <bitset>
#include <sstream>
//#include <bit>
#include <boost/integer/mod_inverse.hpp>

namespace nexusminer {
namespace cpu
{
using namespace boost::multiprecision;
Chain::Chain()
{


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
    m_chain = {};
    m_long_chain_starts = {};

}

std::vector<std::uint64_t> Sieve::get_valid_chain_starting_offsets()
{
    return m_long_chain_starts;
}

void Sieve::set_chain_length_threshold(int min_chain_length)
{
    m_min_chain_length_threshold = min_chain_length;
}

void Sieve::reset_stats()
{
    m_chain_histogram = std::vector<std::uint32_t>(10, 0);
    m_fermat_test_count = 0;
    m_fermat_prime_count = 0;
    m_chain_count = 0;

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
                    if (m_gap_in_process + sieve_offset - previous_sieve_offset > maxGap)
                    {
                        //max gap exceeded.  close open chain and start a new one.
                        close_chain();
                        open_chain(prime_candidate_offset);
                    }
                    else
                    {
                        //continue chain
                        m_current_chain.offsets.push_back(prime_candidate_offset - m_current_chain.base_offset);
                        m_gap_in_process = 0;
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
                m_gap_in_process = 30 - sieve_offset;
                //only keep the chain going if the final gap is smaller than the max
                if (m_gap_in_process > maxGap)
                {
                    close_chain();
                }
            }
        }

    }
}

void Sieve::close_chain()
{
    if (m_current_chain.length() >= minChainLength)
    {
        //we found a chain candidate.  save it.
        m_chain.push_back(m_current_chain);
    }
    //reset current chain to default (empty)
    m_current_chain = {};
    m_gap_in_process = 0;
    m_chain_in_process = false;
}

void Sieve::open_chain(uint64_t base_offset)
{
    //reset current chain to default (empty)
    m_current_chain = {};
    m_gap_in_process = 0;
    m_current_chain.base_offset = base_offset;
    m_chain_in_process = true;
    m_current_chain.zero_length = false;

}

//primality test the vector of chains. give up after first max gap is found.
void Sieve::test_chains()
{

    for (auto i = 0; i < m_chain.size(); i++)
    {
        int count = 0;
        auto chain = m_chain[i];
        int gap = 0;
        auto prev_offset = 0;
        uint1024_t bb = m_sieve_start + chain.base_offset;
        uint1024_t b = bb;
        //test the base offset
        if (primality_test(b))
        {
            count = 1;
            for (auto j = 0; j < chain.offsets.size(); j++)
            {
                gap += chain.offsets[j] - prev_offset;
                if (gap > maxGap)
                {
                    break;
                }
                b = bb + chain.offsets[j];
                prev_offset = chain.offsets[j];
                if (primality_test(b))
                {
                    gap = 0;
                    count++;
                }
            }
        }
        if (count > 0)
        {
            //collect stats
            count = std::min(static_cast<size_t>(count), m_chain_histogram.size());
            m_chain_histogram[count]++;
            if (count > m_min_chain_length_threshold)
            {
                //we found a long chain.  save it.
                m_logger->info("Found a fermat chain of length {}.", count);
                m_long_chain_starts.push_back(chain.base_offset);
                m_chain_count++;

            }
        }
        
    }

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