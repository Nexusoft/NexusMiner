#ifndef NEXUSMINER_GPU_CUDA_CHAIN_CUH
#define NEXUSMINER_GPU_CUDA_CHAIN_CUH

#include <stdint.h>
#include "gpu/prime_common.hpp"

//A class representing a chain of nexus prime candidates.
//We avoid the standard library (no vectors or std::strings) so we can use this with cuda.
namespace nexusminer {
	namespace gpu
	{

		//a candidate for a dense prime cluster.  A chain consists of a base integer plus a list of offsets. 
		struct CudaChain
		{

		public:
			
			static constexpr int m_max_chain_length = 32;  //the longest chain we can represent. 
			// Per-session target Cunningham chain length.  Was previously
			// `static constexpr int = 8`; now an instance field initialised
			// per-chain by cuda_chain_open() from the live per-session T
			// (= sieve_properties.m_min_chain_length).  Defaults preserve
			// the prior compile-time values for any caller that constructs
			// a CudaChain without calling cuda_chain_open(), and for the
			// short window between the chain being default-constructed on
			// the kernel stack and cuda_chain_open() initialising it.
			//
			// is_there_still_hope() and get_next_fermat_candidate() in
			// cuda_chain.cu read m_min_chain_length per-chain.
			// filter_busted_chains in find_chain.cu reads m_min_chain_report_length
			// per-chain — that is what makes the per-session T propagate to
			// the long-chain dispatch gate without changing kernel signatures.
			int m_min_chain_length = 8;
			int m_min_chain_report_length = 5;
			uint64_t m_base_offset = 0;
			uint16_t m_offsets[m_max_chain_length]; //offsets including 0
			Fermat_test_status m_fermat_test_status[m_max_chain_length];
			uint8_t m_next_fermat_test_offset_index = 0;
			int8_t m_prime_count = 0;
			uint8_t m_untested_count = 0;
			uint8_t m_offset_count = 0;

		};


	}
}
#endif