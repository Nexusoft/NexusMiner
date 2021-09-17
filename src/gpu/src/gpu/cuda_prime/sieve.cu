//interface to the cuda sieve implementation
#include "sieve.hpp"
#include <stdint.h>
#include "sieve_impl.cuh"

namespace nexusminer {
    namespace gpu {

        Cuda_sieve::Cuda_sieve() : m_impl(std::make_unique<Cuda_sieve_impl>()) {}
        Cuda_sieve::~Cuda_sieve() = default;
        void Cuda_sieve::run_sieve(uint64_t sieve_start_offset, uint8_t sieve[])
        {
            m_impl->run_sieve(sieve_start_offset, sieve);
        }
       
        void Cuda_sieve::load_sieve(uint32_t primes[], uint32_t prime_count, uint32_t starting_multiples[],
            uint32_t prime_mod_inverses[], uint32_t sieve_size, uint16_t device)
        {
            m_impl->load_sieve(primes, prime_count, starting_multiples, prime_mod_inverses, sieve_size, device);
        }

        void Cuda_sieve::free_sieve()
        {
            m_impl->free_sieve();
        }

        void Cuda_sieve::find_chains(CudaChain chains[], uint32_t& chain_count)
        {
            m_impl->find_chains(chains, chain_count);
        }

       

    }
}
