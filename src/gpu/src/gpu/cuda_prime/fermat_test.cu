//interface to the fermat test implementation
#include "fermat_test.hpp"
#include <stdint.h>
#include <gmp.h>
#include "fermat_test_impl.cuh"

namespace nexusminer {
    namespace gpu {

        Cuda_fermat_test::Cuda_fermat_test() : m_impl(std::make_unique<Cuda_fermat_test_impl>()) {}
        Cuda_fermat_test::~Cuda_fermat_test() = default;
        void Cuda_fermat_test::fermat_run(mpz_t base_big_int, uint64_t offsets[], uint32_t offset_count, uint8_t results[], int device)
        {
            m_impl->fermat_run(base_big_int, offsets, offset_count, results, device);
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
         
    }
}
