#ifndef NEXUSMINER_GPU_CHAIN_HPP
#define NEXUSMINER_GPU_CHAIN_HPP

#include <string>
#include <vector>
//#include "sieve_utils.hpp"
//#include "../cuda_prime/fermat_test.cuh"
#include "gpu/prime_common.hpp"

namespace nexusminer {
	namespace gpu
	{
		//a candidate for a dense prime cluster.  A chain consists of a base integer plus a list of offsets. 
		class Chain
		{
		public:
			enum class Chain_state {
				open, //immature chain empty or in process of being built
				closed, //chain is complete and available for fermat testing
				in_process, //fermat testing is in process. 
				complete  //fermat testing is complete. 
			};

			class Chain_offset {
			public:
				int m_offset = 0;  //offset from the base offset
				Fermat_test_status m_fermat_test_status = Fermat_test_status::untested;
				Chain_offset(int offset) :m_offset{ offset }, m_fermat_test_status{ Fermat_test_status::untested }{};
			};

			Chain();
			Chain(uint64_t base_offset);
			void open(uint64_t base_offset);
			void close();
			int length() { return m_offsets.size(); }
			void get_best_fermat_chain(uint64_t& base_offset, int& offset, int& length);
			bool is_there_still_hope();  //is it possible this chain can result in a valid fermat chain
			bool get_next_fermat_candidate(uint64_t& base_offset, int& offset);
			bool update_fermat_status(bool is_prime);
			void push_back(int offset);
			const std::string str();

			int m_min_chain_length = 8;
			int m_min_chain_report_length = 5;
			Chain_state m_chain_state = Chain_state::open;
			uint64_t m_base_offset = 0;
			std::vector<Chain_offset> m_offsets; //offsets including 0
			int m_gap_in_process = 0;
			int m_next_fermat_test_offset_index = 0;
			int m_prime_count = 0;
			int m_untested_count = 0;

		};
	}

}

#endif