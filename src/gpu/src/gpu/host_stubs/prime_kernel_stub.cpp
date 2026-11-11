// Host-stubs build — stub implementations for the GPU-prime CUDA boundary.
// Compiled only when WITH_GPU_HOST_STUBS=ON so worker_prime.cpp can be built
// and smoke-tested without nvcc or CUDA runtime libraries.

#include "../cuda_prime/fermat_prime/fermat_prime.hpp"
#include "../cuda_prime/sieve.hpp"
#include "../prime/prime.hpp"
#include "../prime/sieve.hpp"
#include "cpu/prime_validation.hpp"
#include "mining/prime_thresholds.hpp"
#include "stats/prime_stats_snapshot.hpp"

#include <algorithm>
#include <cstdint>

namespace nexusminer::gpu {

class Cuda_sieve_impl {};
class Fermat_prime_impl {};

Cuda_sieve::Cuda_sieve()
{
    m_sieve_properties.m_shared_mem_size_kbytes = 0;
    m_sieve_properties.m_shared_mem_size_bytes = 0;
    m_sieve_properties.m_kernel_sieve_size_bytes = 0;
    m_sieve_properties.m_kernel_sieve_size_words = 1;
    m_sieve_properties.m_segment_range = m_sieve_word_range;
    m_sieve_properties.m_kernel_sieve_size_words_per_block = 1;
    m_sieve_properties.m_block_range = m_sieve_word_range;
    m_sieve_properties.m_sieve_total_size = 1;
    m_sieve_properties.m_sieve_range = m_sieve_word_range;
    m_sieve_properties.m_bucket_ram_budget = 0;
    m_sieve_properties.m_large_prime_bucket_size = 0;
    m_sieve_properties.m_min_chain_length = m_min_chain_length;
}

Cuda_sieve::~Cuda_sieve() = default;

void Cuda_sieve::load_sieve(uint32_t[], uint32_t, uint32_t[], uint32_t[],
    uint32_t[], uint32_t, uint8_t[], uint16_t) {}
void Cuda_sieve::init_sieve(uint32_t[], uint16_t[], uint32_t[], uint32_t[]) {}
void Cuda_sieve::reset_stats() {}
void Cuda_sieve::free_sieve() {}
void Cuda_sieve::run_small_prime_sieve(uint64_t) {}
void Cuda_sieve::run_large_prime_sieve(uint64_t) {}
void Cuda_sieve::run_sieve(uint64_t) {}
void Cuda_sieve::run_medium_small_prime_sieve(uint64_t) {}
void Cuda_sieve::find_chains() {}
void Cuda_sieve::clean_chains() {}
void Cuda_sieve::get_chains(CudaChain[], uint32_t& chain_count) { chain_count = 0; }
void Cuda_sieve::get_long_chains(CudaChain[], uint32_t& chain_count) { chain_count = 0; }
void Cuda_sieve::get_chain_count(uint32_t& chain_count) { chain_count = 0; }
void Cuda_sieve::get_chain_pointer(CudaChain*& chains_ptr, uint32_t*& chain_count_ptr)
{
    chains_ptr = nullptr;
    chain_count_ptr = nullptr;
}
void Cuda_sieve::get_sieve(sieve_word_t[]) {}
void Cuda_sieve::get_prime_candidate_count(uint64_t& prime_candidate_count)
{
    prime_candidate_count = 0;
}
void Cuda_sieve::get_stats(uint32_t chain_histogram[], uint64_t& chain_count)
{
    chain_count = 0;
    if (chain_histogram != nullptr)
    {
        std::fill(chain_histogram,
                  chain_histogram + stats::kPrimeHistogramBuckets,
                  0u);
    }
}
void Cuda_sieve::synchronize() {}
void Cuda_sieve::set_target_length(int target_length)
{
    m_min_chain_length = nexusminer::mining::clamp_target_length(target_length);
    m_sieve_properties.m_min_chain_length = m_min_chain_length;
}

Fermat_prime::Fermat_prime() = default;
Fermat_prime::~Fermat_prime() = default;
void Fermat_prime::fermat_run() {}
void Fermat_prime::fermat_chain_run() {}
void Fermat_prime::fermat_init(uint32_t, int) {}
void Fermat_prime::fermat_free() {}
void Fermat_prime::set_base_int(mpz_t) {}
void Fermat_prime::set_chain_ptr(CudaChain*, uint32_t*) {}
void Fermat_prime::set_offsets(uint64_t[], uint64_t) {}
void Fermat_prime::get_results(uint8_t[]) {}
void Fermat_prime::get_stats(uint64_t& fermat_tests, uint64_t& fermat_passes,
    uint64_t& trial_division_tests, uint64_t& trial_division_composites)
{
    fermat_tests = 0;
    fermat_passes = 0;
    trial_division_tests = 0;
    trial_division_composites = 0;
}
void Fermat_prime::reset_stats() {}
void Fermat_prime::synchronize() {}
void Fermat_prime::trial_division_chain_run() {}
void Fermat_prime::trial_division_init(uint32_t, trial_divisors_uint32_t[], int) {}
void Fermat_prime::trial_division_free() {}
void Fermat_prime::test_init(uint64_t, int) {}
void Fermat_prime::test_free() {}
void Fermat_prime::set_input_a(mpz_t*, uint64_t) {}
void Fermat_prime::set_input_b(mpz_t*, uint64_t) {}
void Fermat_prime::get_test_results(mpz_t*) {}
void Fermat_prime::logic_test() {}

Prime::Prime() = default;

Sieve::Sieve()
    : m_logger{spdlog::get("logger")}
{
    reset_stats();
    m_sieve_batch_start_offset = 0;
    m_sieve_range = Cuda_sieve::m_sieve_word_range;
}

void Sieve::generate_sieving_primes() {}
void Sieve::generate_small_prime_tables() {}
void Sieve::generate_trial_divisors() {}
void Sieve::set_sieve_start(boost::multiprecision::uint1024_t sieve_start)
{
    m_sieve_start = sieve_start;
}
boost::multiprecision::uint1024_t Sieve::get_sieve_start()
{
    return m_sieve_start;
}
void Sieve::calculate_starting_multiples() {}
void Sieve::gpu_sieve_load(uint16_t) { m_cuda_sieve_allocated = true; }
void Sieve::gpu_sieve_init() {}
void Sieve::gpu_fermat_test_init(uint16_t) {}
void Sieve::gpu_sieve_free() { m_cuda_sieve_allocated = false; }
void Sieve::gpu_fermat_free() {}
void Sieve::gpu_fermat_test_set_base_int(boost::multiprecision::uint1024_t) {}
uint64_t Sieve::gpu_get_prime_candidate_count() { return 0; }
void Sieve::gpu_get_sieve() {}
void Sieve::gpu_sieve_small_primes(uint64_t) {}
void Sieve::gpu_sieve_large_primes(uint64_t) {}
void Sieve::gpu_sieve_medium_small_primes(uint64_t) {}
void Sieve::sieve_small_primes() {}
void Sieve::sieve_batch(uint64_t) {}
void Sieve::sieve_batch_cpu(uint64_t) {}
std::uint32_t Sieve::get_segment_batch_size() { return m_segment_batch_size; }
void Sieve::reset_sieve() {}
void Sieve::reset_sieve_batch(uint64_t low) { m_sieve_batch_start_offset = low; }
void Sieve::reset_batch_run_count() { m_sieve_run_count = 0; }
void Sieve::clear_chains() { m_long_chain_starts.clear(); }
void Sieve::reset_stats()
{
    m_chain_histogram.assign(stats::kPrimeHistogramBuckets, 0);
    m_fermat_test_count = 0;
    m_fermat_prime_count = 0;
    m_chain_count = 0;
    m_chain_candidate_max_length = 0;
    m_chain_candidate_total_length = 0;
    m_trial_division_chains_busted = 0;
    m_best_chain = 0.0;
}
void Sieve::find_chains() {}
void Sieve::get_chains() {}
void Sieve::sort_chains() {}
void Sieve::get_long_chains() {}
void Sieve::gpu_clean_chains() {}
void Sieve::gpu_run_fermat_chain_test() {}
void Sieve::gpu_run_trial_division_chain_test() {}
void Sieve::gpu_get_fermat_stats(uint64_t& tests, uint64_t& passes,
    uint64_t& trial_division_tests, uint64_t& trial_division_composites)
{
    tests = 0;
    passes = 0;
    trial_division_tests = 0;
    trial_division_composites = 0;
}
void Sieve::gpu_reset_fermat_stats() {}
uint32_t Sieve::get_chain_count() { return 0; }
uint64_t Sieve::count_fermat_primes(int, uint16_t) { return 0; }
uint64_t Sieve::count_fermat_primes_cpu(int) { return 0; }
bool Sieve::primality_test(boost::multiprecision::uint1024_t) { return false; }
void Sieve::test_chains() {}
void Sieve::primality_batch_test(uint16_t) {}
void Sieve::primality_batch_test_cpu() {}
void Sieve::clean_chains() {}
uint64_t Sieve::get_current_chain_list_length() { return 0; }
uint64_t Sieve::get_cuda_chain_list_length() { return 0; }
double Sieve::probability_is_prime_after_sieve(double) { return 0.0; }
double Sieve::sieve_pass_through_rate_expected() { return 0.0; }
double Sieve::expected_chain_density(int, int) { return 0.0; }
uint64_t Sieve::count_prime_candidates() { return 0; }
std::vector<Sieve::sieve_word_t> Sieve::get_sieve() { return {}; }
std::vector<uint64_t> Sieve::get_prime_candidate_offsets() { return {}; }
std::vector<uint32_t> Sieve::get_sieving_primes() { return {}; }
void Sieve::gpu_get_stats() {}
void Sieve::gpu_sieve_synchronize() {}
void Sieve::gpu_fermat_synchronize() {}
Cuda_sieve::Cuda_sieve_properties Sieve::get_sieve_properties()
{
    return m_cuda_sieve.m_sieve_properties;
}
void Sieve::set_target_length(int target_length)
{
    m_min_chain_length = nexusminer::mining::clamp_target_length(target_length);
    m_cuda_sieve.set_target_length(m_min_chain_length);
}

} // namespace nexusminer::gpu

namespace nexusminer::prime {

bool ValidatePrimeCandidate(const uint1024_t&, double, std::vector<uint8_t>& vOffsets,
    double& nDifficulty)
{
    vOffsets.clear();
    nDifficulty = 0.0;
    return false;
}

} // namespace nexusminer::prime
