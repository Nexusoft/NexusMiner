
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
                A >>= 32;
            }
            if (A >= m)
            {
                A -= m;
            }
             return A;
        }

        //reduce x to xR^-1 mod m
        //this is the same as montgomery multiply replacing y with 1
        template<int BITS> __device__ Cump<BITS> montgomery_reduce(const Cump<BITS>& x, const Cump<BITS>& m, uint32_t m_primed)
        {
            Cump<BITS> A, u;
            for (auto i = 0; i <= A.HIGH_WORD; i++)
            {
                u.m_limbs[i] = (A.m_limbs[0] + x.m_limbs[i]) * m_primed;
                A += m * u.m_limbs[i];
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
            Cump<BITS> A = rmodm;
            Cump<BITS> exp = m;
            exp -= 1;
            for (auto i = BITS - 1; i >= 0; i--)
            {
                A = montgomery_multiply(A, A, m, m_primed);
                int word = i / A.BITS_PER_WORD;
                int bit = i % A.BITS_PER_WORD;
                uint32_t mask = 1 << bit;
                bool bit_is_set = (exp.m_limbs[word] & mask) != 0;
                if (bit_is_set)
                {
                    A = double_and_reduce(A, m);
                }
            }
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
