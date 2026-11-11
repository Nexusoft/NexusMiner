#ifndef NEXUSMINER_GPU_CUDA_SIEVE_IMPL_CUH
#define NEXUSMINER_GPU_CUDA_SIEVE_IMPL_CUH

#include "sieve.hpp"
#include "cuda_chain.cuh"
#include <gpu/cuda_memory.hpp>
#include <stdint.h>
namespace nexusminer {
	namespace gpu {

		class Cuda_sieve_impl
		{
		public:
			uint32_t m_sieving_prime_count;
			uint64_t m_sieve_start_offset;
			
			void load_sieve(uint32_t primes[], uint32_t prime_count, uint32_t large_primes[], uint32_t medium_small_primes[],
				uint32_t small_prime_masks[], uint32_t small_prime_mask_count, uint8_t small_primes[], uint16_t device);
			void init_sieve(uint32_t starting_multiples[], uint16_t small_prime_offsets[], uint32_t large_prime_starting_multiples[],
				uint32_t medium_small_prime_multiples[]);
			void reset_stats();
			void free_sieve();
			void run_small_prime_sieve(uint64_t sieve_start_offset);
			void run_large_prime_sieve(uint64_t sieve_start_offset);
			void run_sieve(uint64_t sieve_start_offset);
			void run_medium_small_prime_sieve(uint64_t sieve_start_offset);
			void get_sieve(Cuda_sieve::sieve_word_t sieve[]);
			void get_prime_candidate_count(uint64_t& prime_candidate_count);
			void find_chains();
			void get_chains(CudaChain chains[], uint32_t& chain_count);
			void get_chain_count(uint32_t& chain_count);
			void get_chain_pointer(CudaChain*& chains_ptr, uint32_t*& chain_count_ptr);
			void clean_chains();
			void get_long_chains(CudaChain chains[], uint32_t& chain_count);
			void get_stats(uint32_t chain_histogram[], uint64_t& chain_count);
			void synchronize();
			void init_sieve_size(int device, Cuda_sieve::Cuda_sieve_properties& sieve_properties);

			// Update the per-session target T mirrored into m_sieve_properties.
			// Called by Cuda_sieve::set_target_length(); kept here so the
			// kernel-arg copy of properties (which Cuda_sieve_impl owns) tracks
			// the same value the public Cuda_sieve sees.
			void set_target_length(int target_length);


		private:
			int m_device = 0;
			Cuda_sieve::Cuda_sieve_properties m_sieve_properties;
			//device memory pointers - using RAII smart pointers
			cuda_unique_ptr<uint32_t> d_sieving_primes;  //medium sieving primes
			cuda_unique_ptr<uint32_t> d_starting_multiples;
			cuda_unique_ptr<Prime_plus_multiple_32> d_medium_primes;
			cuda_unique_ptr<uint32_t> d_large_primes;
			cuda_unique_ptr<uint32_t> d_large_prime_starting_multiples;
			cuda_unique_ptr<uint16_t> d_small_prime_offsets;
			cuda_unique_ptr<uint32_t> d_medium_small_primes;
			cuda_unique_ptr<uint32_t> d_medium_small_prime_starting_multiples;
			cuda_unique_ptr<uint32_t> d_small_prime_masks;
			cuda_unique_ptr<uint8_t> d_small_primes;
			//temporary storage for sorting large primes used in large prime bucket sieve
			cuda_unique_ptr<uint32_t> d_large_prime_buckets;
			cuda_unique_ptr<uint32_t> d_bucket_indices;
			//the full sieve
			cuda_unique_ptr<Cuda_sieve::sieve_word_t> d_sieve;
			//temporary storage for the next multiple of the sieving primes
			cuda_unique_ptr<uint32_t> d_multiples;
			//array of chains
			cuda_unique_ptr<CudaChain> d_chains;
			cuda_unique_ptr<uint32_t> d_last_chain_index;
			//array of winners
			cuda_unique_ptr<CudaChain> d_long_chains;
			cuda_unique_ptr<uint32_t> d_last_long_chain_index;
			//count of 1s in the sieve
			cuda_unique_ptr<unsigned long long> d_prime_candidate_count;
			//temporary holding spot for unbusted chains during sorting
			cuda_unique_ptr<uint32_t> d_good_chain_index;
			cuda_unique_ptr<CudaChain> d_good_chains;
			//histogram of confirmed fermat chains
			cuda_unique_ptr<uint32_t> d_chain_histogram;
			//chain stats
			cuda_unique_ptr<unsigned long long> d_chain_stat_count;
			

		};
	}
}

#endif