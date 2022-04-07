#ifndef NEXUSMINER_GPU_PRIME_HPP
#define NEXUSMINER_GPU_PRIME_HPP
//taken from nxs primesolominer

#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <gmp.h>
#include <spdlog/spdlog.h>
#include <boost/multiprecision/cpp_int.hpp>
#include "LLC/types/bignum.h"
#include "ump.hpp"


namespace nexusminer{
namespace gpu
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
	unsigned long PrimeSieve(LLC::CBigNum BaseHash, unsigned int nDifficulty, unsigned int nHeight);
	bool PrimeCheck(LLC::CBigNum test, int checks);
	LLC::CBigNum FermatTest(LLC::CBigNum n, LLC::CBigNum a);
	//bool Miller_Rabin(LLC::CBigNum n, std::uint32_t checks);

private:

	std::shared_ptr<spdlog::logger> m_logger;
	unsigned int* primes;
	unsigned int* inverses;

	unsigned int nBitArray_Size;
	boost::multiprecision::cpp_int zPrimorial;

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