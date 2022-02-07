#include "prime_tests.hpp"
#include <gmp.h>
#include <bitset>
#include <boost/random.hpp>
#include <boost/multiprecision/gmp.hpp>
#include <boost/integer/mod_inverse.hpp>
#include "sieve.hpp"
#include "../cuda_prime/sieve.hpp"
#include <cmath>
#include "../cuda_prime/big_int/big_int.hpp"
#include <random>


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
		test_sieve.generate_small_prime_tables();
		test_sieve.calculate_starting_multiples();
		
		//test_sieve.reset_sieve();
		//test_sieve.reset_sieve_batch(0);
		test_sieve.gpu_sieve_load(m_device);
		test_sieve.gpu_sieve_init();
		Cuda_sieve::Cuda_sieve_properties sieve_properties = test_sieve.get_sieve_properties();
		uint64_t sieve_range = sieve_properties.m_sieve_range;
		//test_sieve.sieve_small_primes();
		
		auto start = std::chrono::steady_clock::now();
		test_sieve.gpu_sieve_small_primes(0);
		test_sieve.gpu_sieve_synchronize();
		auto end = std::chrono::steady_clock::now();
		auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		double small_prime_sieve_elapsed_s = elapsed.count() / 1000.0;
		m_logger->info("Small prime sieved {:.1E} integers using {} primes up to {} in {:.3f} seconds ({:.1f} MISPS).",
			(double)sieve_range, Cuda_sieve::m_small_prime_count, (double)test_sieve.m_small_prime_limit, small_prime_sieve_elapsed_s,
			sieve_range / small_prime_sieve_elapsed_s / 1e6);
		start = std::chrono::steady_clock::now();
		test_sieve.gpu_sieve_medium_small_primes(0);
		test_sieve.gpu_sieve_synchronize();
		end = std::chrono::steady_clock::now();
		elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		double medium_small_sieve_elapsed_s = elapsed.count() / 1000.0;
		m_logger->info("Medium-small prime sieved {:.1E} integers using {} primes up to {:.1E} in {:.3f} seconds ({:.1f} MISPS).",
			(double)sieve_range, Cuda_sieve::m_medium_small_prime_count, (double)test_sieve.m_medium_small_prime_limit, medium_small_sieve_elapsed_s,
			sieve_range / medium_small_sieve_elapsed_s / 1e6);
		start = std::chrono::steady_clock::now();
		test_sieve.sieve_batch(0);
		test_sieve.gpu_sieve_synchronize();
		end = std::chrono::steady_clock::now();
		elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		double sieve_elapsed_s = elapsed.count() / 1000.0;
		m_logger->info("Medium prime sieved {:.1E} integers using {} primes up to {:.1E} in {:.3f} seconds ({:.1f} MISPS).",
			(double)sieve_range, Cuda_sieve::m_medium_prime_count, (double)test_sieve.m_sieving_prime_limit, sieve_elapsed_s,
			sieve_range / sieve_elapsed_s / 1e6);
		start = std::chrono::steady_clock::now();
		test_sieve.gpu_sieve_large_primes(0);
		test_sieve.gpu_sieve_synchronize();
		end = std::chrono::steady_clock::now();
		elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		double large_prime_sieve_elapsed_s = elapsed.count() / 1000.0;
		m_logger->info("Large prime sieved {:.1E} integers using {} primes up to {:.1E} in {:.3f} seconds ({:.1f} MISPS).",
			(double)sieve_range, Cuda_sieve::m_large_prime_count, (double)test_sieve.m_large_prime_limit, large_prime_sieve_elapsed_s,
			sieve_range / large_prime_sieve_elapsed_s / 1e6);
		uint64_t prime_candidate_count = test_sieve.gpu_get_prime_candidate_count();
		double candidate_ratio = static_cast<double>(prime_candidate_count) / sieve_range;
		double candidate_ratio_expected = test_sieve.sieve_pass_through_rate_expected();
		double combined_sieve_time = sieve_elapsed_s + medium_small_sieve_elapsed_s + small_prime_sieve_elapsed_s + large_prime_sieve_elapsed_s;
		m_logger->info("Combined sieve time {:.3f} seconds ({:.1f} MISPS).", combined_sieve_time, sieve_range / (combined_sieve_time) / 1e6);
		m_logger->info("Got {:.3f}% sieve pass through rate.  Expected about {:.3f}%.",
			candidate_ratio * 100, candidate_ratio_expected * 100);
		//test_sieve.gpu_get_sieve();
		//uint64_t candidate_count = test_sieve.count_prime_candidates();
		start = std::chrono::steady_clock::now();
		test_sieve.find_chains();
		test_sieve.gpu_sieve_synchronize();
		end = std::chrono::steady_clock::now();
		elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		double find_chains_elapsed_s = elapsed.count() / 1000.0;
		uint32_t gpu_chain_count = test_sieve.get_chain_count();
		//test_sieve.get_chains();
		//test_sieve.sort_chains();
		//uint64_t chain_count_before = test_sieve.get_current_chain_list_length();
		//test_sieve.clear_chains();
		//test_sieve.find_chains_cpu(0, true);
		//test_sieve.reset_batch_run_count();
		//test_sieve.sieve_batch(0);
		//test_sieve.do_chain_trial_division_check();
		//uint64_t chain_count_after = test_sieve.get_current_chain_list_length();
		//uint32_t busted_chain_count = chain_count_before - chain_count_after;
		test_sieve.gpu_fermat_test_init(m_device);
		//uint64_t prime_candidate_count = test_sieve.count_prime_candidates();
		
		double bits = static_cast<double>(boost::multiprecision::log2(static_cast<boost::multiprecision::mpf_float>(test_sieve.get_sieve_start())));
		double fermat_positive_rate_expected = test_sieve.probability_is_prime_after_sieve(bits);
		//int fermat_sample_size = std::min<uint64_t>(100000, prime_candidate_count);
		//uint64_t fermat_count = test_sieve.count_fermat_primes(fermat_sample_size, m_device);
		
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
		//m_logger->info("Busted {} chains ({:.3f}%) with trial division", busted_chain_count, (double)busted_chain_count / chain_count_before);

		double eight_chain_probability = std::pow(fermat_positive_rate_expected, 8);
		double chains_per_eight_chain = 1.0 / eight_chain_probability;
		double range_per_eight_chain = sieve_range * chains_per_eight_chain / gpu_chain_count;
		double expected_chain_density = test_sieve.expected_chain_density(test_sieve.m_min_chain_length, bits);
		//range_per_eight_chain = 1 / expected_chain_density;
		//m_logger->info("Approximate range to find one 8-chain: {:.1E} ", range_per_eight_chain);
		//process chains
		test_sieve.gpu_reset_fermat_stats();
		test_sieve.gpu_run_fermat_chain_test();
		test_sieve.gpu_fermat_synchronize();
		uint64_t test_attempts, passes;
		test_sieve.gpu_get_fermat_stats(test_attempts, passes);
		m_logger->info("Fermat primes: {}/{} ({:.3f}%). Expected about {:.3f}%.",
			passes, test_attempts, 100.0*passes/test_attempts, fermat_positive_rate_expected * 100.0);
		//test_sieve.clear_chains();
		//test_sieve.get_chains();
		//test_sieve.clean_chains();
		//uint32_t gpu_chain_count_after = test_sieve.get_current_chain_list_length();
		//m_logger->info("Chain count after one round of fermat testing and cpu clean chains {}", gpu_chain_count_after);
		test_sieve.gpu_clean_chains();
		gpu_chain_count = test_sieve.get_chain_count();
		m_logger->info("Chain count after gpu clean chains {}", gpu_chain_count);
		//test_sieve.clear_chains();
		//test_sieve.get_chains();
		//test_sieve.clean_chains();
		//chain_count_after = test_sieve.get_current_chain_list_length();
		//m_logger->info("Chain count after cpu clean chains {}", chain_count_after);
		test_sieve.get_long_chains();
		/*while (test_sieve.get_chain_count() > 0)
		{
			test_sieve.gpu_run_fermat_chain_test();
			test_sieve.gpu_clean_chains();
			test_sieve.get_long_chains();
		}*/
		//m_logger->info("Found {} 8 chains.  Expected {}", test_sieve.m_chain_histogram[8], sieve_range/range_per_eight_chain);
		test_sieve.gpu_get_stats();
		test_sieve.gpu_sieve_free();
		test_sieve.gpu_fermat_free();
		m_logger->info("Sieve test complete.");
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

	void PrimeTests::math_test()
	{
		using namespace boost::multiprecision;
		using namespace boost::random;
		const uint64_t batch_size = 100000;
		m_logger->info("Starting big_int math performance test with batch size {}.", batch_size);
		bool cpu_verify = false;
		typedef independent_bits_engine<mt19937, 1024, boost::multiprecision::uint1024_t> generator1024_type;
		generator1024_type gen1024;
		gen1024.seed(time(0));
		
		mpz_t* a = new mpz_t[batch_size];
		mpz_t* b = new mpz_t[batch_size];
		mpz_t* c = new mpz_t[batch_size];
		mpz_t* results = new mpz_t[batch_size];

		boost::multiprecision::uint1024_t a1, b1;

		for (auto i = 0; i < batch_size; i++)
		{
			//generate a few contrived test cases
			switch (i){
			case 1:
				a1 = 0x15d79a8e0;
				b1 = 0x1e05d54f9;
				break;
			case 0:
				a1.assign("0xeed0ea40daaec1031b6c8172b9c3714846a8784736de503369e58e9c25499cb5a034c76ec59511778affe2150ae1e07623d5418a6c2132303a22fe599add9e12ad6b5434b5fd21a84befce066758dc418832b01fce21d6be1b4519e3f3d5b9bff3effd9ba963847ffb95c8c88ea3b854bfa576e6d63badc99cbc114adb27122");
				b1.assign("0xfe0584cba13b76ba4d468a779f158f2f6d2a4ba21e19383cff0ab2918db49f4a52ac1550fcdc3a600efedf7c47883a8f61425118dc7679b50291fafe0115315406c855e074ed3c78bbaf1d3ad74ea046a682900a01e10b810b795381c0c971222d88b8b27ad13de26129110c97be2aaf5807a90c7151ee383c01808f94faf5df");
				break;
			case 2:
				a1.assign("0xe2792767ec01880f6178d32f5aad3a9b4c2316acf5eb694913b86b71f4497078b1dc808296c8b1e0eac87f7a4c104097d42b93000a1bd8c340ea59fcc9f6a402df7a1eeb65b814228df2ffe887935baf0bcbcadf2cd7791fd8766bdc261e27dd8a1f3dafc24fbe5e673b1cb7eb771759c6e0c5605835c27236af25c6e1ba3231");
				b1.assign("0x46fde95a7bd8da283b895d6bb9a66cfd4e22d7686e6e1863856ff5c29aee6ebcabb073547d7bf");
				break;
			case 3:
				a1 = 1;
				b1 = 0;
				break;
			case 4:
				a1 = 0;
				b1 = -1;
				break;
			case 5:
				a1 = -1;
				b1 = 0;
				break;
			case 6:
				a1 = 0;
				b1 = 0;
				break;
			case 7:
				a1 = 1;
				b1 = 1;
				break;
			case 8:
				a1 = -1;
				b1 = -1;
				break;
			case 9:
				a1 = 721948327;
				b1 = 84461;
				break;
			case 10:
				a1 = -1;
				b1 = 2;
				break;
			case 11:
				a1 = -1;
				b1 = 1;
				break;
			case 12:
				a1 = -1;
				b1 = 4;
				break;
			case 13:
				a1 = -1;
				b1 = 5;
				break;
			case 14:
				a1 = boost::multiprecision::uint1024_t(1) << (31*32);
				b1 = 1;
				break;
			case 15:
				a1 = -1;
				b1 = boost::multiprecision::uint1024_t(1) << 32;
				break;
			case 16:
				a1 = boost::multiprecision::uint1024_t(1) << (31*32);
				b1 = boost::multiprecision::uint1024_t(1) << (30*32);
				break;
			case 17:
				a1 = boost::multiprecision::uint1024_t (-1) >> 37;
				b1 = boost::multiprecision::uint1024_t (-1) >> 666;
				break;
			case 18:
				a1 = boost::multiprecision::uint1024_t(-1) >> 666;
				b1 = boost::multiprecision::uint1024_t(-1) >> 37;
				break;
			default:
				//the rest are random numbers
				a1 = gen1024();
				b1 = gen1024();
				//random shift
				std::random_device dev;
				std::mt19937 rng(dev());
				std::uniform_int_distribution<std::mt19937::result_type> dist(0, 15);
				int shift = dist(rng);
				b1 = b1 >> shift;
				//a1 = a1 >> (1023 - 32);
				//b1 = b1 >> (1023 - 32);

			}
			
			//make b1 odd
			b1 += (b1 % 2 == 0) ? 1 : 0;
			mpz_init2(a[i],1024);
			mpz_set(a[i], static_cast<mpz_int>(a1).backend().data());
			mpz_init2(b[i], 1024);
			mpz_set(b[i], static_cast<mpz_int>(b1).backend().data());
			//initialize output containers
			mpz_init2(c[i], 1024);
			mpz_init2(results[i], 1024);
		}
		
		m_logger->info("Test data generation complete.");
		
		Big_int big_int;
		big_int.test_init(batch_size, 0);
		m_logger->info("Loading test vectors to GPU RAM.");
		big_int.set_input_a(a, batch_size);
		big_int.set_input_b(b, batch_size);
		m_logger->info("Running aritmetic/logic operation under test.");
		auto start = std::chrono::steady_clock::now();
		big_int.logic_test();
		//big_int.subtract();
		auto end = std::chrono::steady_clock::now();
		auto add_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		m_logger->info("Copying results to CPU RAM.");
		big_int.get_test_results(results);
		big_int.test_free();
		uint64_t passes = 0;
		uint64_t attempts = 0;
		if (cpu_verify)
		{
			m_logger->info("Verifying Results.");

			for (auto i = 0; i < batch_size; i++)
			{
				boost::multiprecision::uint1024_t a_1024(static_cast<mpz_int>(a[i]));
				boost::multiprecision::uint1024_t b_1024(static_cast<mpz_int>(b[i]));
				boost::multiprecision::uint1024_t c_1024 = 0, d_1024 = 0;
				boost::multiprecision::uint1024_t results_1024(static_cast<mpz_int>(results[i]));
				

				//this must match the math/logic function under test used in the gpu
				if (boost::multiprecision::msb(b_1024) >= 992)
				{

					//c_1024 = a_1024 * b_1024;
					mpz_int bm = static_cast<mpz_int>(b[i]);
					mpz_int two = 2;
					mpz_int c = boost::multiprecision::powm(two,bm - 1, bm);
					//c = c << (2*1024);
					//c = c % static_cast<mpz_int>(b[i]);
					c_1024 = static_cast<boost::multiprecision::uint1024_t>(c);

					
					//mpz_int d = 1;
					//d = d << 1024;
					//mpz_int m = static_cast<mpz_int>(b[i]);
					//int leading_zeros = 1023 - boost::multiprecision::msb(m);
					//mpz_int m_primed = m << leading_zeros;
					//mpz_int r = d; 
					//int counta = 0, countb = 0;
					//while (r > m)
					//{
					//	counta++;
					//	//std::cout << "outer" << std::endl;
					//	//std::cout << "r " << std::setfill('0') << std::hex << r << std::endl;
					//	//std::cout << "m " << std::setfill('0') << std::hex << m << std::endl;
					//	//std::cout << "m_primed " << std::setfill('0') << std::hex << m_primed << std::endl;
					//	while (r > m_primed)
					//	{
					//		r -= m_primed;
					//		//std::cout << "inner" << std::endl;
					//		//std::cout << "r " << std::setfill('0') << std::hex << r << std::endl;
					//		//std::cout << "m " << std::setfill('0') << std::hex << m << std::endl;
					//		//std::cout << "m_primed " << std::setfill('0') << std::hex << m_primed << std::endl;
					//		countb++;
					//	}
					//	leading_zeros = boost::multiprecision::msb(m_primed) - boost::multiprecision::msb(r);
					//	m_primed = m_primed >> 1;// std::max(leading_zeros, 1);
					//}
					//std::cout << "a " << counta << " b " << countb << std::endl;
					//d_1024 = static_cast<boost::multiprecision::uint1024_t>(r);
					//results_1024 = d_1024;
					//if (i == 0)
					//	std::cout << "c[0]:" << c_1024 << std::endl;

					if (c_1024 != results_1024)
					{

						m_logger->debug("GPU/CPU math test mismatch at offset {}", i);
						std::stringstream result_ss, a_ss, b_ss, c_ss;
						result_ss << std::setfill('0') << std::hex << results_1024;
						c_ss << std::setfill('0') << std::hex << c_1024;
						a_ss << std::setfill('0') << std::hex << a_1024;
						b_ss << std::setfill('0') << std::hex << b_1024;
						m_logger->debug("Input a {}", a_ss.str());
						m_logger->debug("Input b {}", b_ss.str());
						m_logger->debug("Got {}", result_ss.str());
						m_logger->debug("Expected {}", c_ss.str());
						

					}
					else
					{
						passes++;
					}
					attempts++;
				}
				
			}
		}

		for (auto i = 0; i < batch_size; i++)
		{
			mpz_clear(a[i]);
			mpz_clear(b[i]);
			mpz_clear(c[i]);
			mpz_clear(results[i]);
		}

		delete[] a;
		delete[] b;
		delete[] c;
		delete[] results;

		std::stringstream ss;
		if (cpu_verify)
		{
			ss << "Test result: " << passes << "/" << attempts << " results match.";
			m_logger->info(ss.str());
		}
		else
		{
			attempts = batch_size;
		}
		
		ss = {};
		ss << "Run time: " << add_elapsed.count() << " ms. " << std::fixed << std::setprecision(1) << attempts / (add_elapsed.count()) << " thousand operations/second. (" << (attempts > 0 ? 1.0e3 * add_elapsed.count() / attempts : -999) << "us)";
		m_logger->info(ss.str());
		
		
	}

}
}