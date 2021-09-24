#ifndef NEXUSMINER_SMALL_SIEVE_TOOLS_HPP
#define NEXUSMINER_SMALL_SIEVE_TOOLS_HPP

//tool to generate precomputed masks used by the small prime sieve

#include <vector>
//#include "sieve_utils.hpp"

namespace nexusminer {
	namespace gpu
	{
		class Small_sieve_tools
		{
		public:
			//bit sieve with a mod 30 wheel uses 8 bits to represent a span of 30 integers excluding multiples of 2,3, and 5
			static constexpr uint32_t sieve_span_per_byte = 30;
			//use use four bytes per sieve word.  
			using sieve_word_t = uint32_t;
			static constexpr int sieve_word_bytes = sizeof(sieve_word_t);
			static constexpr uint32_t sieve_span_per_word = sieve_span_per_byte * sieve_word_bytes;
			static constexpr int sieve30_offsets[]{ 1,7,11,13,17,19,23,29 };  // each bit in the sieve30 represets an offset from the base mod 30
			static constexpr int sieve30_gaps[]{ 6,4,2,4,2,4,6,2 };
			static constexpr int sieve30_index[]{ -1,0,-1,-1,-1,-1,-1, 1, -1, -1, -1, 2, -1, 3, -1, -1, -1, 4, -1, 5, -1, -1, -1, 6, -1, -1, -1, -1, -1, 7 };  //reverse lookup table (offset mod 30 to index)
			//list of primes starting at 7
			std::vector<uint16_t> primes{ 7,11,13,17,19,23,29,31,37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97 };

			std::vector<sieve_word_t> prime_mask(uint16_t prime);
			std::vector<uint16_t> word_index_ring(uint16_t prime);
			void print_code(uint16_t max_prime);
			void print_mask(uint16_t prime);


		};
	}
}

#endif