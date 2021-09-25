#include <bitset>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
#include "small_sieve_tools.hpp"
//#include "sieve_utils.hpp"


namespace nexusminer {
    namespace gpu
    {

        std::vector<Small_sieve_tools::sieve_word_t> Small_sieve_tools::prime_mask(uint16_t prime)
        {
            std::vector<sieve_word_t> mask;
            uint32_t low = 0;
            uint32_t position = prime;
            for (auto i = 0; i < prime; i++)
            {
                sieve_word_t w = 0;
                std::bitset<8 * sieve_word_bytes> bits = 0;
                while (position < low + sieve_span_per_word)
                {
                    //find the next multiple of the prime that is not divisible by 2,3, or 5
                    if (position % 2 != 0 && position % 3 != 0 && position % 5 != 0)
                    {
                        int index = 8*(position/sieve_span_per_byte) + sieve30_index[position % sieve_span_per_byte];
                        index = index % bits.size();
                        bits.set(index);
                    }
                    position += prime;
                }
                w = ~bits.to_ulong();
                mask.push_back(w);
                low += sieve_span_per_word;
            }
            
            return mask;
        }

        std::vector<uint16_t> Small_sieve_tools::word_index_ring(uint16_t prime)
        {
            return std::vector<uint16_t>();
        }

        //for debug.  print out the bit mask in human readable format
        void Small_sieve_tools::print_mask(uint16_t prime)
        {
            std::vector<Small_sieve_tools::sieve_word_t> mask = prime_mask(prime);
            for (auto i = 0; i < mask.size() - 1; i++)
            {
                auto m = mask[i];
                std::cout << std::bitset<8 * sizeof(m)>(m) << ", ";
            }
            auto m = mask.back();
            std::cout << std::bitset<8 * sizeof(m)>(m) << std::endl;

        }

        //generate text that can be pasted into c++ code to create hardcoded bitmask arrays
        void Small_sieve_tools::print_code(uint16_t max_prime)
        {
            std::stringstream ss;
            
            for (auto p : primes)
            {
                if (p > max_prime)
                    break;
                ss << "__constant__ const uint32_t p";
                ss << p << "[" << p << "] = ";
                int digits = log10(p) + 1;
                int spaces = 10 - 2 * digits;
                ss << std::string(spaces, ' ') << "{";
                std::vector<Small_sieve_tools::sieve_word_t> mask = prime_mask(p);
                int numbers_in_row = 0;
                int indent_spaces = ss.str().length();
                for (auto i=0; i<mask.size()-1; i++)
                {
                    auto m = mask[i];
                    ss << "0x" << std::setfill('0') << std::setw(8) << std::hex << m << ", ";
                    numbers_in_row++;
                    if (numbers_in_row >= 4)
                    {
                        ss << std::endl << std::string(indent_spaces, ' ');
                        numbers_in_row = 0;
                    }
                }
                //the last number in the set
                ss << "0x" << std::setfill('0') << std::setw(8) << std::hex << mask.back() << "};" << std::endl << std::endl;
                std::cout << ss.str();
                ss = {};
            }
        }
    }
}