#include "prime_tests.hpp"
#include <gmp.h>
#include <boost/random.hpp>
#include <boost/multiprecision/gmp.hpp>
#include "sieve.hpp"
#include "../cuda_prime/sieve.hpp"


namespace nexusminer
{
namespace gpu
{
	

	PrimeTests::PrimeTests(int device)
		: m_logger{ spdlog::get("logger") }
		, m_device{device}
	{}

	void PrimeTests::fermat_performance_test()
		//test the throughput of fermat primality test
	{
		using namespace boost::multiprecision;
		using namespace boost::random;

		m_logger->info("Starting fermat primality test performance test.");
		bool cpu_verify = false;
		typedef independent_bits_engine<mt19937, 1024, boost::multiprecision::uint1024_t> generator1024_type;
		generator1024_type gen1024;
		gen1024.seed(time(0));
		// Generate a random 1024-bit unsigned value:
		boost::multiprecision::uint1024_t pp = gen1024();
		//make it odd
		pp += 1 ? (pp % 2) == 0 : 0;

		static constexpr uint32_t primality_test_batch_size = 1e5;
		//uint64_t offsets[primality_test_batch_size];
		std::vector<uint64_t> offsets;
		//generate an array of offsets for batch prime testing
		uint64_t offset_start = 0xFFFFFFFFFFFFFE;
		for (uint64_t i = offset_start; i < offset_start + primality_test_batch_size; i++)
		{
			offsets.push_back(i * 2);
		}
		boost::multiprecision::mpz_int base_as_mpz_int = static_cast<mpz_int>(pp);
		mpz_t base_as_mpz_t;
		mpz_init(base_as_mpz_t);
		mpz_set(base_as_mpz_t, base_as_mpz_int.backend().data());
		std::vector<uint8_t> primality_test_results;
		primality_test_results.resize(primality_test_batch_size);
		//bool primality_test_results[primality_test_batch_size];
		m_cuda_fermat_test.fermat_init(primality_test_batch_size, m_device);
		m_cuda_fermat_test.set_base_int(base_as_mpz_t);
		m_cuda_fermat_test.set_offsets(offsets.data(), primality_test_batch_size);
		auto start = std::chrono::steady_clock::now();
		m_cuda_fermat_test.fermat_run();
		auto end = std::chrono::steady_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		m_cuda_fermat_test.get_results(primality_test_results.data());
		uint64_t cuda_primality_test_count, cuda_primality_pass_count;
		m_cuda_fermat_test.get_stats(cuda_primality_test_count, cuda_primality_pass_count);
		m_cuda_fermat_test.fermat_free();
		mpz_clear(base_as_mpz_t);

		

		int primes_found = 0;
		for (auto i = 0; i < primality_test_batch_size; i++)
		{
			if (primality_test_results[i] == 1)
				primes_found++;
			if (cpu_verify)
			{
				bool is_prime_cpu = primality_test_cpu(pp + offsets[i]);
				if (is_prime_cpu != (primality_test_results[i] == 1))
				{
					m_logger->debug("GPU/CPU primality test mismatch at offset {} {}", i, offsets[i]);
				}
			}
		}

		if (cuda_primality_test_count != primality_test_batch_size)
		{
			m_logger->debug("Primality stats mismatch. GPU reports {}/{} passed/attempted vs {}/{}",
				cuda_primality_pass_count, cuda_primality_test_count, primes_found, primality_test_batch_size);
		}

		double expected_primes = primality_test_batch_size * 2 / (1024 * 0.693147);
		std::stringstream ss;
		ss << "Found " << primes_found << " primes out of " << primality_test_batch_size << " tested. Expected about " << expected_primes << ". ";
		m_logger->info(ss.str());
		ss = {};
		ss << std::fixed << std::setprecision(2) << 1000.0 * primality_test_batch_size / elapsed.count() << " primality tests/second. (" << 1000.0 * elapsed.count() / primality_test_batch_size << "us)";
		m_logger->info(ss.str());
	}

	//test sieving for speed and accuracy
	void PrimeTests::sieve_performance_test()
	{
		//known starting point
		boost::multiprecision::uint1024_t T200("0x53bf18ac03f0adfb36fc4864b42013375ebdc0bb311f06636771e605ad731ca1383c7d9056522ed9bda4f608ef71498bc9c7dade6c56bf1534494e0ef371e79f09433e4c9e64624695a42d7920bd5022f449156d2f93f3be3a429159794ac9e49f69c706793ef249a284f9173a82379e62dffac42c0f53f155f65a784f31f42c");
		uint64_t nonce200 = 127171;
		double diff200 = 3.2608808;
		m_logger->info("Starting sieve performance test.");
		Sieve test_sieve;
		test_sieve.set_sieve_start(T200);
		//test_sieve.m_sieving_prime_limit = 1000;
		//test_sieve.m_segment_batch_size = 100;
		test_sieve.generate_sieving_primes();
		test_sieve.calculate_starting_multiples();
		test_sieve.reset_sieve();
		test_sieve.reset_sieve_batch(0);
		test_sieve.gpu_sieve_init(m_device);
		auto start = std::chrono::steady_clock::now();
		//test_sieve.sieve_batch_cpu(0);
		test_sieve.sieve_batch(0);
		auto end = std::chrono::steady_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		double sieve_elapsed_s = elapsed.count() / 1000.0;
		start = std::chrono::steady_clock::now();
		test_sieve.find_chains();
		end = std::chrono::steady_clock::now();
		elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		double find_chains_elapsed_s = elapsed.count() / 1000.0;
		//test_sieve.clear_chains();
		//test_sieve.find_chains_cpu(0, true);

		uint64_t prime_candidate_count = test_sieve.count_prime_candidates();
		uint64_t sieve_range = test_sieve.m_sieve_results.size() / 8 * 30;
		double candidate_ratio = static_cast<double>(prime_candidate_count) / sieve_range;
		double candidate_ratio_expected = test_sieve.sieve_pass_through_rate_expected();

		m_logger->info("Sieved {:.1E} integers using primes up to {:.1E} in {:.3f} seconds ({:.1f} MISPS).",
			(double)sieve_range, (double)test_sieve.m_sieving_prime_limit, sieve_elapsed_s, sieve_range / sieve_elapsed_s / 1e6);
		m_logger->info("Got {:.3f}% sieve pass through rate.  Expected about {:.3f}%.",
			candidate_ratio * 100, candidate_ratio_expected * 100);
		double fermat_positive_rate_expected = test_sieve.probability_is_prime_after_sieve();
		int fermat_sample_size = std::min<uint64_t>(100000, prime_candidate_count);
		uint64_t fermat_count = test_sieve.count_fermat_primes(fermat_sample_size, m_device);
		m_logger->info("Got {:.3f}% fermat positive rate. Expected about {:.3f}%",
			100.0 * fermat_count / fermat_sample_size, fermat_positive_rate_expected * 100.0);
		m_logger->info("Found {} chains in {:.4f} seconds ({:.2f} chains/MIS @ {:.1f} MISPS).",
			test_sieve.get_current_chain_list_length(),
			find_chains_elapsed_s, 1.0e6 * test_sieve.get_current_chain_list_length() / sieve_range,
			sieve_range / find_chains_elapsed_s / 1e6);

		test_sieve.gpu_sieve_free();

	}

	bool PrimeTests::primality_test_cpu(boost::multiprecision::uint1024_t p)
	{

		boost::multiprecision::mpz_int base = 2;
		boost::multiprecision::mpz_int result;
		boost::multiprecision::mpz_int p1 = static_cast<boost::multiprecision::mpz_int>(p);
		result = boost::multiprecision::powm(base, p1 - 1, p1);
		m_fermat_test_count++;
		bool isPrime = (result == 1);
		if (isPrime)
		{
			++m_fermat_prime_count;
		}
		return (isPrime);
	}

	void PrimeTests::reset_stats()
	{
		m_fermat_prime_count = 0;
		m_fermat_test_count = 0;
	}

}
}