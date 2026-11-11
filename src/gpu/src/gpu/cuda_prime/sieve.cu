//interface to the cuda sieve implementation
#include "sieve.hpp"
#include <stdint.h>
#include <algorithm>
#include "sieve_impl.cuh"
#include "mining/prime_thresholds.hpp"

namespace nexusminer {
    namespace gpu {


        Cuda_sieve::Cuda_sieve() : m_impl(std::make_unique<Cuda_sieve_impl>())
        {
            // Initialise the per-session target T inside m_sieve_properties so
            // that any kernel launch that reads m_sieve_properties.m_min_chain_length
            // before set_target_length() is ever called still sees a sensible
            // value (= mining::MIN_CHAIN_LENGTH, mirroring m_min_chain_length's
            // default).  Other m_sieve_properties fields are populated later by
            // Cuda_sieve_impl::init_sieve_size(), which preserves this field.
            m_sieve_properties.m_min_chain_length = m_min_chain_length;
        }
        Cuda_sieve::~Cuda_sieve() = default;

        void Cuda_sieve::set_target_length(int target_length)
        {
            const int clamped =
                std::max(target_length, nexusminer::mining::kMinTargetChainLength);
            m_min_chain_length = clamped;
            m_sieve_properties.m_min_chain_length = clamped;
            // Forward to the impl so its mirrored properties (used as the
            // kernel argument in find_chains()/sieve dispatchers) stay in sync.
            m_impl->set_target_length(clamped);
        }

        void Cuda_sieve::run_sieve(uint64_t sieve_start_offset)
        {
            m_impl->run_sieve(sieve_start_offset);
        }

        void Cuda_sieve::run_medium_small_prime_sieve(uint64_t sieve_start_offset)
        {
            m_impl->run_medium_small_prime_sieve(sieve_start_offset);
        }
       
        void Cuda_sieve::load_sieve(uint32_t primes[], uint32_t prime_count, uint32_t large_primes[], uint32_t medium_small_primes[],
            uint32_t small_prime_masks[], uint32_t small_prime_mask_count, uint8_t small_primes[], uint16_t device)
        {
            m_impl->init_sieve_size(device, m_sieve_properties);
            m_impl->load_sieve(primes, prime_count, large_primes, medium_small_primes, small_prime_masks, small_prime_mask_count,
                small_primes, device);
        }

        void Cuda_sieve::init_sieve(uint32_t starting_multiples[], uint16_t small_prime_offsets[], uint32_t large_prime_starting_multiples[],
            uint32_t medium_small_prime_starting_multiples[])
        {
            m_impl->init_sieve(starting_multiples, small_prime_offsets, large_prime_starting_multiples, medium_small_prime_starting_multiples);
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

        void Cuda_sieve::synchronize()
        {
            m_impl->synchronize();
        }

        void Cuda_sieve::run_large_prime_sieve(uint64_t sieve_start_offset)
        {
            m_impl->run_large_prime_sieve(sieve_start_offset);
        }


       

    }
}
