//interface to the cuda sieve implementation
#include "sieve.hpp"
#include <stdint.h>
#include "sieve_impl.cuh"

namespace nexusminer {
    namespace gpu {

        //primes for reference 7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101
        //                     1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22, 23

        const int Cuda_sieve::m_small_primes[] = { 7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,
                103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211 };
        Cuda_sieve::Cuda_sieve() : m_impl(std::make_unique<Cuda_sieve_impl>()) {}
        Cuda_sieve::~Cuda_sieve() = default;
        void Cuda_sieve::run_sieve(uint64_t sieve_start_offset)
        {
            m_impl->run_sieve(sieve_start_offset);
        }
       
        void Cuda_sieve::load_sieve(uint32_t primes[], uint32_t prime_count,
            uint32_t prime_mod_inverses[], uint32_t sieve_size, uint16_t device)
        {
            m_impl->load_sieve(primes, prime_count, prime_mod_inverses, sieve_size, device);
        }

        void Cuda_sieve::init_sieve(uint32_t starting_multiples[], uint32_t small_prime_offsets[])
        {
            m_impl->init_sieve(starting_multiples, small_prime_offsets);
        }

        void Cuda_sieve::reset_stats()
        {
            m_impl->reset_stats();
        }

        void Cuda_sieve::free_sieve()
        {
            m_impl->free_sieve();
        }

        void Cuda_sieve::run_small_prime_sieve(uint64_t sieve_start_offset)
        {
            m_impl->run_small_prime_sieve(sieve_start_offset);
        }

        void Cuda_sieve::find_chains()
        {
            m_impl->find_chains();
        }

        void Cuda_sieve::clean_chains()
        {
            m_impl->clean_chains();
        }

        void Cuda_sieve::get_chains(CudaChain chains[], uint32_t& chain_count)
        {
            m_impl->get_chains(chains, chain_count);
        }

        void Cuda_sieve::get_long_chains(CudaChain chains[], uint32_t& chain_count)
        {
            m_impl->get_long_chains(chains, chain_count);
        }

        void Cuda_sieve::get_chain_count(uint32_t& chain_count)
        {
            m_impl->get_chain_count(chain_count);
        }

        void Cuda_sieve::get_chain_pointer(CudaChain*& chains_ptr, uint32_t*& chain_count_ptr)
        {
            m_impl->get_chain_pointer(chains_ptr, chain_count_ptr);
        }

        void Cuda_sieve::get_sieve(sieve_word_t sieve[])
        {
            m_impl->get_sieve(sieve);
        }

        void Cuda_sieve::get_prime_candidate_count(uint64_t& prime_candidate_count)
        {
            m_impl->get_prime_candidate_count(prime_candidate_count);
        }

        void Cuda_sieve::get_stats(uint32_t chain_histogram[], uint64_t& chain_count)
        {
            m_impl->get_stats(chain_histogram, chain_count);
        }

        void Cuda_sieve::run_large_prime_sieve(uint64_t sieve_start_offset)
        {
            m_impl->run_large_prime_sieve(sieve_start_offset);
        }

       

    }
}
