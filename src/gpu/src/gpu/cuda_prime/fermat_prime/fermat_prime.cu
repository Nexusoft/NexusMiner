//interface to the cuda sieve implementation
#include "fermat_prime.hpp"
#include <stdint.h>
#include "fermat_prime_impl.cuh"

namespace nexusminer {
    namespace gpu {


        Fermat_prime::Fermat_prime() : m_impl(std::make_unique<Fermat_prime_impl>()) {}
        Fermat_prime::~Fermat_prime() = default;

        void Fermat_prime::fermat_run()
        {
            m_impl->fermat_run();
        }

        void Fermat_prime::fermat_chain_run()
        {
            m_impl->fermat_chain_run();
        }

        void Fermat_prime::fermat_init(uint32_t batch_size, int device)
        {
            m_impl->fermat_init(batch_size, device);
        }

        void Fermat_prime::fermat_free()
        {
            m_impl->fermat_free();
        }

        void Fermat_prime::set_base_int(mpz_t base_big_int)
        {
            m_impl->set_base_int(base_big_int);
        }

        void Fermat_prime::set_chain_ptr(CudaChain* chains, uint32_t* chain_count)
        {
            m_impl->set_chain_ptr(chains, chain_count);
        }

        void Fermat_prime::set_offsets(uint64_t offsets[], uint64_t offset_count)
        {
            m_impl->set_offsets(offsets, offset_count);
        }

        void Fermat_prime::get_results(uint8_t results[])
        {
            m_impl->get_results(results);
        }

        void Fermat_prime::get_stats(uint64_t& fermat_tests, uint64_t& fermat_passes,
            uint64_t& trial_division_tests, uint64_t& trial_division_composites)
        {
            m_impl->get_stats(fermat_tests, fermat_passes, trial_division_tests, trial_division_composites);
        }

        void Fermat_prime::reset_stats()
        {
            m_impl->reset_stats();
        }

        void Fermat_prime::synchronize()
        {
            m_impl->synchronize();
        }

        void Fermat_prime::trial_division_chain_run()
        {
            m_impl->trial_division_chain_run();
        }

        void Fermat_prime::trial_division_init(uint32_t trial_divisor_count, trial_divisors_uint32_t trial_divisors[], int device)
        {
            m_impl->trial_division_init(trial_divisor_count, trial_divisors, device);
        }

        void Fermat_prime::trial_division_free()
        {
            m_impl->trial_division_free();
        }

        void Fermat_prime::test_init(uint64_t batch_size, int device)
        {
            m_impl->test_init(batch_size, device);
        }

        void Fermat_prime::test_free()
        {
            m_impl->test_free();
        }

        void Fermat_prime::set_input_a(mpz_t* a, uint64_t count)
        {
            m_impl->set_input_a(a, count);
        }

        void Fermat_prime::set_input_b(mpz_t* b, uint64_t count)
        {
            m_impl->set_input_b(b, count);
        }

        void Fermat_prime::get_test_results(mpz_t* test_results)
        {
            m_impl->get_test_results(test_results);
        }


        void Fermat_prime::logic_test()
        {
            m_impl->logic_test();
        }


        
    }
}
