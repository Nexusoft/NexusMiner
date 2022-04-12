#ifndef NEXUSMINER_CPU_PRIME_HPP
#define NEXUSMINER_CPU_PRIME_HPP
//taken from nxs primesolominer

#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <spdlog/spdlog.h>
#include "LLC/types/bignum.h"
#include "ump.hpp"

namespace nexusminer{
namespace cpu
{
using uint1k = ump::uint1024_t;
class Prime
{
public:
	Prime();

	void InitializePrimes();
	unsigned int SetBits(double nDiff);
	double GetPrimeDifficulty(LLC::CBigNum prime, int checks, std::vector<unsigned int>& vPrimes);
	double GetSieveDifficulty(LLC::CBigNum next, unsigned int clusterSize);
	unsigned int GetPrimeBits(LLC::CBigNum prime, int checks, std::vector<unsigned int>& vPrimes);
	unsigned int GetFractionalDifficulty(LLC::CBigNum composite);
	std::vector<unsigned int> Eratosthenes(int nSieveSize);
	bool DivisorCheck(LLC::CBigNum test);
	bool PrimeCheck(LLC::CBigNum test, int checks);
	LLC::CBigNum FermatTest(LLC::CBigNum n, LLC::CBigNum a);

private:

	std::shared_ptr<spdlog::logger> m_logger;
	unsigned int* primes;
	unsigned int* inverses;

	unsigned int nBitArray_Size;

	unsigned int prime_limit;
	unsigned int nPrimeLimit;
	unsigned int nPrimorialEndPrime;

	uint64_t octuplet_origins[256];
	/** Divisor bit_array_sieve for Prime Searching. **/
	std::vector<unsigned int> DIVISOR_SIEVE;
};



}
}

#endif