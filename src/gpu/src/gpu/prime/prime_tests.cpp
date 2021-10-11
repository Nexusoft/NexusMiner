#include "prime_tests.hpp"
#include <gmp.h>
#include <bitset>
#include <boost/random.hpp>
#include <boost/multiprecision/gmp.hpp>
#include "sieve.hpp"
#include "../cuda_prime/sieve.hpp"
#include <cmath>


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
		//typedef independent_bits_engine<mt19937, 1024, boost::multiprecision::uint1024_t> generator1024_type;
		//generator1024_type gen1024;
		//gen1024.seed(time(0));
		// Generate a random 1024-bit unsigned value:
		//boost::multiprecision::uint1024_t pp = gen1024();
		boost::multiprecision::uint1024_t T200("0x53bf18ac03f0adfb36fc4864b42013375ebdc0bb311f06636771e605ad731ca1383c7d9056522ed9bda4f608ef71498bc9c7dade6c56bf1534494e0ef371e79f09433e4c9e64624695a42d7920bd5022f449156d2f93f3be3a429159794ac9e49f69c706793ef249a284f9173a82379e62dffac42c0f53f155f65a784f31f42c");
		boost::multiprecision::uint1024_t pp = T200;
		//make it odd
		pp += 1 ? (pp % 2) == 0 : 0;

		static constexpr uint32_t primality_test_batch_size = 1e5;
		uint64_t offset_start = 0xFFFFFFFFFFFFFE;
		int expected_prime_count = 269;
		//uint64_t offsets[primality_test_batch_size];
		std::vector<uint64_t> offsets;
		//generate an array of offsets for batch prime testing
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
		Cuda_fermat_test cuda_fermat_test;
		cuda_fermat_test.fermat_init(primality_test_batch_size, m_device);
		cuda_fermat_test.set_base_int(base_as_mpz_t);
		cuda_fermat_test.set_offsets(offsets.data(), primality_test_batch_size);
		auto start = std::chrono::steady_clock::now();
		cuda_fermat_test.fermat_run();
		auto end = std::chrono::steady_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		cuda_fermat_test.get_results(primality_test_results.data());
		uint64_t cuda_primality_test_count, cuda_primality_pass_count;
		cuda_fermat_test.get_stats(cuda_primality_test_count, cuda_primality_pass_count);
		cuda_fermat_test.fermat_free();
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

		if (cuda_primality_test_count != primality_test_batch_size || cuda_primality_pass_count != primes_found)
		{
			m_logger->debug("Primality stats mismatch. GPU reports {}/{} passed/attempted vs {}/{}",
				cuda_primality_pass_count, cuda_primality_test_count, primes_found, primality_test_batch_size);
		}

		double expected_primes = primality_test_batch_size * 2 / (1024 * 0.693147);
		std::stringstream ss;
		if (primes_found != expected_prime_count)
		{
			m_logger->error("Prime count mismatch.  Got {}. Expected {}.", primes_found, expected_prime_count);
		}
		ss << "Found " << primes_found << " primes out of " << primality_test_batch_size << " tested. Expected " << expected_prime_count << ". ";
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
		boost::multiprecision::uint1024_t low_start = 30*7*11 - 30;
		boost::multiprecision::uint1024_t med_start = 2;
		med_start = boost::multiprecision::pow(med_start, 128);
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
		test_sieve.sieve_small_primes();
		auto start = std::chrono::steady_clock::now();
		test_sieve.gpu_sieve_small_primes(0);
		auto end = std::chrono::steady_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		double small_prime_sieve_elapsed_s = elapsed.count() / 1000.0;
		//test_sieve.sieve_batch_cpu(0);
		//std::vector<uint8_t> cpu_sieve = test_sieve.get_sieve();
		//test_sieve.reset_sieve_batch(0);
		start = std::chrono::steady_clock::now();
		test_sieve.sieve_batch(0);
		end = std::chrono::steady_clock::now();
		elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		double sieve_elapsed_s = elapsed.count() / 1000.0;
		start = std::chrono::steady_clock::now();
		test_sieve.gpu_sieve_large_primes(0);
		end = std::chrono::steady_clock::now();
		elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		double large_prime_sieve_elapsed_s = elapsed.count() / 1000.0;
		uint64_t prime_candidate_count = test_sieve.gpu_get_prime_candidate_count();
		test_sieve.gpu_get_sieve();
		//uint64_t candidate_count = test_sieve.count_prime_candidates();
		start = std::chrono::steady_clock::now();
		test_sieve.find_chains();
		end = std::chrono::steady_clock::now();
		elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		double find_chains_elapsed_s = elapsed.count() / 1000.0;
		uint32_t gpu_chain_count = test_sieve.get_chain_count();
		test_sieve.get_chains();
		uint64_t chain_count_before = test_sieve.get_current_chain_list_length();
		//test_sieve.clear_chains();
		//test_sieve.find_chains_cpu(0, true);
		//test_sieve.reset_batch_run_count();
		//test_sieve.sieve_batch(0);
		//test_sieve.do_chain_trial_division_check();
		uint64_t chain_count_after = test_sieve.get_current_chain_list_length();
		uint32_t busted_chain_count = chain_count_before - chain_count_after;

		

		test_sieve.gpu_fermat_test_init(m_device);

		//std::vector<uint8_t> gpu_sieve = test_sieve.get_sieve();
		/*if (cpu_sieve.size() != gpu_sieve.size())
		{
			m_logger->info("Unexpected sieve size.");
		}
		else
		{
			for (auto i=0; i<cpu_sieve.size(); i++)
			{
				if (cpu_sieve[i] != gpu_sieve[i])
				{
					m_logger->info("Sieve mismatch at index {}. CPU[{}]={}, GPU[{}]={}.", i,i,cpu_sieve[i],i,gpu_sieve[i]);
				}
			}
		}*/


		//uint64_t prime_candidate_count = test_sieve.count_prime_candidates();
		uint64_t sieve_range = test_sieve.m_sieve_results.size() * Cuda_sieve::m_sieve_word_range;
		double candidate_ratio = static_cast<double>(prime_candidate_count) / sieve_range;
		double candidate_ratio_expected = test_sieve.sieve_pass_through_rate_expected();
		m_logger->info("Small prime sieved {:.1E} integers using primes up to {} in {:.3f} seconds ({:.1f} MISPS).",
			(double)sieve_range, Cuda_sieve::m_small_primes[Cuda_sieve::m_small_prime_count-1], small_prime_sieve_elapsed_s, sieve_range / small_prime_sieve_elapsed_s / 1e6);
		auto sieving_primes = test_sieve.get_sieving_primes();
		uint32_t medium_prime_index = std::min(static_cast<size_t>(Cuda_sieve::m_large_prime_cutoff_index), sieving_primes.size());
		double medium_prime_limit = sieving_primes[medium_prime_index-1];
		
		m_logger->info("Medium prime sieved {:.1E} integers using {} primes up to {:.1E} in {:.3f} seconds ({:.1f} MISPS).",
			(double)sieve_range, medium_prime_index, medium_prime_limit, sieve_elapsed_s, sieve_range / sieve_elapsed_s / 1e6);

		m_logger->info("Large prime sieved {:.1E} integers using {} primes up to {:.1E} in {:.3f} seconds ({:.1f} MISPS).",
			(double)sieve_range, sieving_primes.size() - medium_prime_index, (double)test_sieve.m_sieving_prime_limit, large_prime_sieve_elapsed_s, sieve_range / large_prime_sieve_elapsed_s / 1e6);
		
		m_logger->info("Combined sieve ({:.1f} MISPS).", sieve_range / (sieve_elapsed_s + small_prime_sieve_elapsed_s + large_prime_sieve_elapsed_s) / 1e6);
		m_logger->info("Got {:.3f}% sieve pass through rate.  Expected about {:.3f}%.",
			candidate_ratio * 100, candidate_ratio_expected * 100);
		double bits = static_cast<double>(boost::multiprecision::log2(static_cast<boost::multiprecision::mpf_float>(test_sieve.get_sieve_start())));
		double fermat_positive_rate_expected = test_sieve.probability_is_prime_after_sieve(bits);
		int fermat_sample_size = std::min<uint64_t>(100000, prime_candidate_count);
		uint64_t fermat_count = test_sieve.count_fermat_primes(fermat_sample_size, m_device);
		m_logger->info("Got {:.3f}% fermat positive rate. Expected about {:.3f}%",
			100.0 * fermat_count / fermat_sample_size, fermat_positive_rate_expected * 100.0);
		//how many chains we expect to find is equal to the density at the bit width we stop sieving,
		//which is equivalent to the square of the max sieving prime or twice the max sieving prime bit width
		//this is not precise for low bit widths.   9 chains contain 2 or more 8 chains etc.
		//int effective_bit_width = std::log2(test_sieve.m_sieving_prime_limit) * 2;
		//double expected_chain_density = test_sieve.expected_chain_density(test_sieve.m_min_chain_length, effective_bit_width);
		//double expected_chain_count = expected_chain_density * sieve_range;
		m_logger->info("Found {} chains in {:.4f} seconds ({:.2f} chains/MIS @ {:.1f} MISPS).",
			gpu_chain_count,
			find_chains_elapsed_s, 1.0e6 * gpu_chain_count / sieve_range,
			sieve_range / find_chains_elapsed_s / 1e6);
		m_logger->info("Busted {} chains ({:.3f}%) with trial division", busted_chain_count, (double)busted_chain_count / chain_count_before);

		double eight_chain_probability = std::pow(fermat_positive_rate_expected, 8);
		double chains_per_eight_chain = 1.0 / eight_chain_probability;
		double range_per_eight_chain = sieve_range * chains_per_eight_chain / gpu_chain_count;
		double expected_chain_density = test_sieve.expected_chain_density(test_sieve.m_min_chain_length, bits);
		//range_per_eight_chain = 1 / expected_chain_density;
		//m_logger->info("Approximate range to find one 8-chain: {:.1E} ", range_per_eight_chain);
		//process chains
		test_sieve.gpu_reset_fermat_stats();
		test_sieve.gpu_run_fermat_chain_test();
		uint64_t test_attempts, passes;
		test_sieve.gpu_get_fermat_stats(test_attempts, passes);
		m_logger->info("Fermat tests after one round of fermat testing {}/{} ({:.3f}%)", passes, test_attempts, 100.0*passes/test_attempts);
		test_sieve.clear_chains();
		test_sieve.get_chains();
		test_sieve.clean_chains();
		uint32_t gpu_chain_count_after = test_sieve.get_current_chain_list_length();
		m_logger->info("Chain count after one round of fermat testing and cpu clean chains {}", gpu_chain_count_after);
		test_sieve.gpu_clean_chains();
		gpu_chain_count = test_sieve.get_chain_count();
		m_logger->info("Chain count after gpu clean chains {}", gpu_chain_count);
		test_sieve.clear_chains();
		test_sieve.get_chains();
		test_sieve.clean_chains();
		chain_count_after = test_sieve.get_current_chain_list_length();
		m_logger->info("Chain count after cpu clean chains {}", chain_count_after);
		test_sieve.get_long_chains();
		while (test_sieve.get_chain_count() > 0)
		{
			test_sieve.gpu_run_fermat_chain_test();
			test_sieve.gpu_clean_chains();
			test_sieve.get_long_chains();
		}
		//m_logger->info("Found {} 8 chains.  Expected {}", test_sieve.m_chain_histogram[8], sieve_range/range_per_eight_chain);
		test_sieve.gpu_get_stats();
		test_sieve.gpu_sieve_free();
		test_sieve.gpu_fermat_free();
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