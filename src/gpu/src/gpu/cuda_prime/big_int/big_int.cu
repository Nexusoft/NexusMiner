//interface to the cuda sieve implementation
#include "big_int.hpp"
#include <stdint.h>
#include "big_int_impl.cuh"

namespace nexusminer {
    namespace gpu {


        Big_int::Big_int() : m_impl(std::make_unique<Big_int_impl>()) {}
        Big_int::~Big_int() = default;

        void Big_int::fermat_run()
        {
            m_impl->fermat_run();
        }

        void Big_int::fermat_chain_run()
        {
            m_impl->fermat_chain_run();
        }

        void Big_int::fermat_init(uint32_t batch_size, int device)
        {
            m_impl->fermat_init(batch_size, device);
        }

        void Big_int::fermat_free()
        {
            m_impl->fermat_free();
        }

        void Big_int::set_base_int(mpz_t base_big_int)
        {
            m_impl->set_base_int(base_big_int);
        }

        void Big_int::set_chain_ptr(CudaChain* chains, uint32_t* chain_count)
        {
            m_impl->set_chain_ptr(chains, chain_count);
        }

        void Big_int::set_offsets(uint64_t offsets[], uint64_t offset_count)
        {
            m_impl->set_offsets(offsets, offset_count);
        }

        void Big_int::get_results(uint8_t results[])
        {
            m_impl->get_results(results);
        }

        void Big_int::get_stats(uint64_t& fermat_tests, uint64_t& fermat_passes)
        {
            m_impl->get_stats(fermat_tests, fermat_passes);
        }

        void Big_int::reset_stats()
        {
            m_impl->reset_stats();
        }

        void Big_int::synchronize()
        {
            m_impl->synchronize();
        }

        void Big_int::test_init(uint64_t batch_size, int device)
        {
            m_impl->test_init(batch_size, device);
        }

        void Big_int::test_free()
        {
            m_impl->test_free();
        }

        void Big_int::set_input_a(mpz_t* a, uint64_t count)
        {
            m_impl->set_input_a(a, count);
        }

        void Big_int::set_input_b(mpz_t* b, uint64_t count)
        {
            m_impl->set_input_b(b, count);
        }

        void Big_int::get_test_results(mpz_t* test_results)
        {
            m_impl->get_test_results(test_results);
        }

        void Big_int::add()
        {
            m_impl->add();
        }

        void Big_int::subtract()
        {
            m_impl->subtract();
        }

        void Big_int::logic_test()
        {
            m_impl->logic_test();
        }


        
    }
}
