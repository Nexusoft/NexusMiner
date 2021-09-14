#ifndef CUDA_CHAIN_CUH
#define CUDA_CHAIN_CUH
#include <stdint.h>

//A class representing a chain of nexus prime candidates.
//We avoid the standard library (no vectors or std::strings) so we can use this with cuda.
namespace nexusminer {
	namespace gpu
	{
		

		enum class Fermat_test_status {
			untested,
			fail,
			pass
		};

		static constexpr int maxGap = 12;  //the largest allowable prime gap.

		//a candidate for a dense prime cluster.  A chain consists of a base integer plus a list of offsets. 
		struct CudaChain
		{

		public:
			enum class Chain_state {
				open, //immature chain empty or in process of being built
				closed, //chain is complete and available for fermat testing
				in_process, //fermat testing is in process. 
				complete  //fermat testing is complete. 
			};

			static constexpr int m_max_chain_length = 22;  //the longest chain we can represent. 
			int m_min_chain_length = 8;
			int m_min_chain_report_length = 5;
			Chain_state m_chain_state = Chain_state::open;
			uint64_t m_base_offset = 0;
			uint16_t m_offsets[m_max_chain_length]; //offsets including 0
			Fermat_test_status m_fermat_test_status[m_max_chain_length];
			int m_gap_in_process = 0;
			int m_next_fermat_test_offset_index = 0;
			int m_prime_count = 0;
			int m_untested_count = 0;
			int m_offset_count = 0;

		};


	}
}
#endif