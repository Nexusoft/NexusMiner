#ifndef NEXUSMINER_GPU_PRIME_TESTS_HPP
#define NEXUSMINER_GPU_PRIME_TESTS_HPP
#include <spdlog/spdlog.h>
#include <boost/multiprecision/cpp_int.hpp>
#include "../cuda_prime/fermat_test.hpp"


namespace nexusminer {
namespace gpu
{

	class PrimeTests
	{
	public:
		PrimeTests(int device);
		void fermat_performance_test();
		void sieve_performance_test();
		bool primality_test_cpu(boost::multiprecision::uint1024_t p);
		void reset_stats();
	private:
		
		std::shared_ptr<spdlog::logger> m_logger;
		int m_device = 0;
		uint64_t m_fermat_test_count = 0;
		uint64_t m_fermat_prime_count = 0;
	};

}

}


#endif