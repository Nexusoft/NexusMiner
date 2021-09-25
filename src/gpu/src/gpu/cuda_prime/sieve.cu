//interface to the cuda sieve implementation
#include "sieve.hpp"
#include <stdint.h>
#include "sieve_impl.cuh"

namespace nexusminer {
    namespace gpu {

        //primes for reference 7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97
        //                     1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22   
        const int Cuda_sieve::m_small_primes[] = { 7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97 };
        Cuda_sieve::Cuda_sieve() : m_impl(std::make_unique<Cuda_sieve_impl>()) {}
        Cuda_sieve::~Cuda_sieve() = default;
        void Cuda_sieve::run_sieve(uint64_t sieve_start_offset)
        {
            m_impl->run_sieve(sieve_start_offset);
        }
       
        void Cuda_sieve::load_sieve(uint32_t primes[], uint32_t prime_count, uint32_t starting_multiples[],
            uint8_t prime_mod_inverses[], uint32_t small_prime_offsets[], uint32_t sieve_size, uint16_t device)
        {
            m_impl->load_sieve(primes, prime_count, starting_multiples, prime_mod_inverses, small_prime_offsets, sieve_size, device);
        }

        void Cuda_sieve::free_sieve()
        {
            m_impl->free_sieve();
        }

        void Cuda_sieve::run_small_prime_sieve(uint64_t sieve_start_offset)
        {
            m_impl->run_small_prime_sieve(sieve_start_offset);
        }

        void Cuda_sieve::find_chains(CudaChain chains[], uint32_t& chain_count)
        {
            m_impl->find_chains(chains, chain_count);
        }

        void Cuda_sieve::get_sieve(sieve_word_t sieve[])
        {
            m_impl->get_sieve(sieve);
        }

        void Cuda_sieve::get_prime_candidate_count(uint64_t& prime_candidate_count)
        {
            m_impl->get_prime_candidate_count(prime_candidate_count);
        }

       

    }
}
