//interface to the fermat test implementation
#include "fermat_test.hpp"
#include <stdint.h>
#include <gmp.h>
#include "fermat_test_impl.cuh"

namespace nexusminer {
    namespace gpu {

        Cuda_fermat_test::Cuda_fermat_test() : m_impl(std::make_unique<Cuda_fermat_test_impl>()) {}
        Cuda_fermat_test::~Cuda_fermat_test() = default;
        void Cuda_fermat_test::fermat_run()
        {
            m_impl->fermat_run();
        }

        void Cuda_fermat_test::fermat_chain_run()
        {
            m_impl->fermat_chain_run();
        }

        void Cuda_fermat_test::fermat_init(uint32_t batch_size, int device)
        {
            m_impl->fermat_init(batch_size, device);
        }

        void Cuda_fermat_test::fermat_free()
        {
            m_impl->fermat_free();
        }

        void Cuda_fermat_test::set_base_int(mpz_t base_big_int)
        {
            m_impl->set_base_int(base_big_int);
        }

        void Cuda_fermat_test::set_chain_ptr(CudaChain* chains, uint32_t* chain_count)
        {
            m_impl->set_chain_ptr(chains, chain_count);
        }

        void Cuda_fermat_test::set_offsets(uint64_t offsets[], uint64_t offset_count)
        {
            m_impl->set_offsets(offsets, offset_count);
        }

        void Cuda_fermat_test::get_results(uint8_t results[])
        {
            m_impl->get_results(results);
        }

        void Cuda_fermat_test::get_stats(uint64_t& fermat_tests, uint64_t& fermat_passes)
        {
            m_impl->get_stats(fermat_tests, fermat_passes);
        }

        void Cuda_fermat_test::reset_stats()
        {
            m_impl->reset_stats();
        }
         
    }
}
