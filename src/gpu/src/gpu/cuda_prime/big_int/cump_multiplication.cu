

namespace nexusminer {
    namespace gpu {

        //low half of mutliplication
        //HAC 14.12 modified for fixed width
        template<int BITS>
        __host__ __device__ Cump<BITS> Cump<BITS>::multiply(const Cump& y) const
        {
            Cump<BITS> result;
            
            for (auto i = 0; i < LIMBS; i++) //y index
            {
                uint32_t c = 0;
                for (auto j = 0; j < LIMBS - i; j++) //x index
                {
                    uint64_t uv = result.m_limbs[i+j] + static_cast<uint64_t>(m_limbs[j]) * y.m_limbs[i] + c;
                    result.m_limbs[i + j] = uv;  //store the lower bits
                    c = uv >> 32;  //the upper bits is the carry
                }
            }
            return result;
        }

        template<int BITS>
        __host__ __device__ Cump<BITS> operator * (const Cump<BITS>& lhs, const Cump<BITS>& rhs)
        {
            return lhs.multiply(rhs);
        }

        template<int BITS>
        __host__ __device__ void Cump<BITS>::operator*=(const Cump<BITS>& y)
        {
            *this = multiply(y);
        }


        //multiplication of a big uint by an unsigned integer.  
        template<int BITS>
        __host__ __device__ Cump<BITS> Cump<BITS>::multiply(uint32_t x) const
        {
            Cump<BITS> result;
            uint32_t c = 0;
            for (auto i = 0; i < LIMBS; i++)
            {
                uint64_t uv = static_cast<uint64_t>(m_limbs[i]) * x + c;
                result.m_limbs[i] = uv;  //store the lower bits
                c = uv >> 32;  //the upper bits is the carry
                
            }
            return result;
        }

        template<int BITS>
        __host__ __device__ Cump<BITS> operator * (const Cump<BITS>& lhs, uint32_t rhs)
        {
            return lhs.multiply(rhs);
        }

        template<int BITS>
        __host__ __device__ void Cump<BITS>::operator*=(uint32_t x)
        {
            uint32_t c = 0;
            for (auto i = 0; i < LIMBS; i++)
            {
                uint64_t uv = static_cast<uint64_t>(m_limbs[i]) * x + c;
                m_limbs[i] = uv;  //store the lower bits
                c = uv >> 32;  //the upper bits is the carry

            }

        }
    }
}
