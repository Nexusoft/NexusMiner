
#include "cump.cuh"

namespace nexusminer {
    namespace gpu {

        //montgomery multiplication.  See HAC ch 14 algorithm 14.36
        //returns xyR^-1
        template<int BITS> __device__ Cump<BITS> montgomery_multiply(const Cump<BITS>& x, const Cump<BITS>& y, const Cump<BITS>& m, uint32_t m_primed)
        {
            Cump<BITS> A, u;
            for (auto i = 0; i <= A.HIGH_WORD; i++)
            {
                u.m_limbs[i] = (A.m_limbs[0] + x.m_limbs[i] * y.m_limbs[0]) * m_primed;
                //this step requires two extra words to handle "double" overflow that can happen when the top bit of m is set
                A += y * x.m_limbs[i] + m * u.m_limbs[i];
                //divide by 32 (right shift one whole word)
                for (int j = 0; j < A.LIMBS - 1; j++)
                {
                    A.m_limbs[j] = A.m_limbs[j + 1];
                }
                A.m_limbs[A.LIMBS - 1] = 0;
                //A >>= 32;
            }
            if (A >= m)
            {
                A -= m;
            }
             return A;
        }

        //speical squaring case of montgomery multiplication
        //returns xxR^-1
        template<int BITS> __device__ Cump<BITS> montgomery_square(const Cump<BITS>& x, const Cump<BITS>& m, uint32_t m_primed)
        {
            Cump<BITS> A;
            //unroll the first iteration to save a few operations
            uint32_t u = x.m_limbs[0] * x.m_limbs[0] * m_primed;
            A = x * x.m_limbs[0] + m * u;
            A >>= 32;
            for (auto i = 1; i <= A.HIGH_WORD; i++)
            {
                u = (A.m_limbs[0] + x.m_limbs[i] * x.m_limbs[0]) * m_primed;
                //this step requires two extra words to handle "double" overflow that can happen when the top bit of m is set
                A += x * x.m_limbs[i];
                A += m * u;
                A >>= 32;
                
            }
            
            if (A >= m)
            {
                A -= m;
            }
            return A;
        }

        //full width square followed by montgomery reduction
        //HAC 14.16 
        //returns xxR^-1
        //this is 10x slower than montgomery_square
        template<int BITS> __device__ Cump<BITS> montgomery_square_2(const Cump<BITS>& x, const Cump<BITS>& m, uint32_t m_primed)
        {
            //square
            const int t = x.HIGH_WORD + 1;
            uint64_t w[2*t];
            for (int i = 0; i < 2 * t; i++)
            {
                w[i] = 0;
            }
            for (int i = 0; i < t; i++)
            {
                
                uint64_t uv = w[2 * i] + static_cast<uint64_t>(x.m_limbs[i]) * x.m_limbs[i];
                w[2 * i] = uv & 0xFFFFFFFF;
                uint64_t c = uv >> 32;
                for (int j = i + 1; j < t; j++)
                {
                    uv = static_cast<uint64_t>(x.m_limbs[j]) * x.m_limbs[i];
                    bool carry = uv & (0x1ull << 63);
                    uv = w[i + j] + 2 * uv + c;
                    w[i + j] = uv & 0xFFFFFFFF;
                    c = (uv >> 32) | (carry ? (1ull << 32) : 0);
                }
                w[i + t] = c;
            }

            //reduce
            Cump<2*BITS> A, m2, um;
            for (auto i = 0; i <= A.HIGH_WORD; i++)
            {
                A.m_limbs[i] = w[i];
            }

            for (auto i = 0; i <= m.HIGH_WORD; i++)
            {
                m2.m_limbs[i] = m.m_limbs[i];
            }

            for (auto i = 0; i <= m.HIGH_WORD; i++)
            {
                uint32_t u = A.m_limbs[i] * m_primed;
                um = (m2 * u) << (32 * i);
                A += um;
            }
            A >>= (32 * (m.HIGH_WORD + 1));
            
            Cump<BITS> C;
            for (auto i = 0; i <= C.HIGH_WORD; i++)
            {
                C.m_limbs[i] = A.m_limbs[i];
            }
            if (C >= m)
            {
                C -= m;
            }

            return C;
            
        }

        //reduce x to xR^-1 mod m
        //this is the same as montgomery multiply replacing y with 1
        template<int BITS> __device__ Cump<BITS> montgomery_reduce(const Cump<BITS>& x, const Cump<BITS>& m, uint32_t m_primed)
        {
            Cump<BITS> A;
            for (auto i = 0; i <= A.HIGH_WORD; i++)
            {
                uint32_t u = (A.m_limbs[0] + x.m_limbs[i]) * m_primed;
                A += m * u;
                A += x.m_limbs[i];
                A >>= 32;
            }
            if (A >= m)
            {
                A -= m;
            }
            return A;
        }

        //returns 2^(m-1) mod m
        //m_primed and rmodm are precalculated values.  See hac 14.94.  
        //R^2 mod m is not needed because with base 2 is trivial to calculate 2*R mod m given R mod m
        template<int BITS> __device__ Cump<BITS> powm_2(const Cump<BITS>& m, const Cump<BITS>& rmodm, uint32_t m_primed)
        {
            //initialize the product to R mod m, the equivalent of 1 in the montgomery domain
            Cump<BITS> A = rmodm;
            Cump<BITS> exp = m - 1;

            //unroll the first few iterations manually - we are squaring small numbers
            //The first iteration requires no squaring
            int word = A.HIGH_WORD;
            int bit = (A.BITS_PER_WORD - 1) % A.BITS_PER_WORD;
            uint32_t mask = 1 << bit;
            bool bit_is_set = (exp.m_limbs[word] & mask) != 0;
            if (bit_is_set)
            {
                //multiply by the base (2) if the exponent bit is set
                A = double_and_reduce(A, m);
            }

            for (auto i = BITS - 2; i >= 0; i--)
            {
                //square
                A = montgomery_square(A, m, m_primed);
                word = i / A.BITS_PER_WORD;
                bit = i % A.BITS_PER_WORD;
                mask = 1 << bit;
                bit_is_set = (exp.m_limbs[word] & mask) != 0;
                if (bit_is_set)
                {
                    //multiply by the base (2) if the exponent bit is set
                    A = double_and_reduce(A, m);
                }
            }
            //convert back from montgomery domain
            A = montgomery_reduce(A, m, m_primed);
            return A;

        }
        
        // return 2 * x mod m given x and m using shift and subtract.
        //For this to be efficient, m must be similar magnitude (within a few bits) of x. 
        template<int BITS> __device__ Cump<BITS> double_and_reduce(const Cump<BITS>& x, const Cump<BITS>& m)
        {
            Cump<BITS> A = x << 1;
            while (A >= m)
            {
                A -= m;
            }
            return A;
        }




    }
}
