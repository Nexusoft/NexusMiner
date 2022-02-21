
#include "cump.cuh"

namespace nexusminer {
    namespace gpu {

        //montgomery multiplication.  See HAC ch 14 algorithm 14.36
        //returns xyR^-1
        template<int BITS> __device__ Cump<BITS> montgomery_multiply(const Cump<BITS>& x, const Cump<BITS>& y, const Cump<BITS>& m, uint32_t m_primed)
        {
            Cump<BITS> A;
            uint32_t u;
            for (auto i = 0; i <= A.HIGH_WORD; i++)
            {
                u = (A.m_limbs[0] + x.m_limbs[i] * y.m_limbs[0]) * m_primed;
                //this step requires two extra words to handle "double" overflow that can happen when the top bit of m is set
                A += y * x.m_limbs[i] + m * u;
                //divide by 32 (right shift one whole word)
                A >>= 32;
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
           
            Cump<BITS> A, mp;
            for (auto i = 0; i <= A.HIGH_WORD; i++)
            {
                uint32_t u = (A.m_limbs[0] + x.m_limbs[i] * x.m_limbs[0])* m_primed;
                A = A + x * x.m_limbs[i] + m * u;
                A >>= 32;
            }
            if (A >= m)
            {
                A -= m;
            }
            return A;
        }

        //montgomery square - optimized to take advantage of symmetry of squaring operation
        //Fermat testing spends most of its time inside this function
        //returns xxR^-1
        template<int BITS> __device__ Cump<BITS> montgomery_square_2(const Cump<BITS>& x, const Cump<BITS>& m, uint32_t m_primed)
        {
            
            //low half of the square
            Cump<BITS> AA, BB, y, result;
            const int y_size = x.HIGH_WORD + 2;
            uint32_t yy[y_size + 1];
            
#pragma unroll
            for (int i = 0; i <= x.HIGH_WORD/2; i++)
            {
                uint64_t w = x.m_limbs[i];
                //the square term
                uint64_t sq = w * w;
                //multiply the remaining upper portion of x by the current word (y = x * w)
                uint32_t carry = 0;
                
                //mulitply y = x * xi excluding the square term
#pragma unroll
                for (int j = 1; j <= y_size - i - 1; j++)
                {
                    uint64_t uv = x.m_limbs[i + j] * w + carry;
                    yy[j] = uv; //low word
                    carry = uv >> 32;  //the upper bits is the carry
                }
               
                yy[y_size - i] = 0;
                yy[0] = 0;
                //multiply y by 2 (funnel shift left)
#pragma unroll
                for (int j = y_size - i; j >= 1; j--)
                {
                    yy[j] = __funnelshift_l(yy[j - 1], yy[j], 1);
                }
                //add the square term to the doubled terms
                yy[0] = sq;
                uint32_t sq_upper = sq >> 32;
                yy[1] += sq_upper;
                //the carry can propogate max 1 word since all the double terms are even
                yy[2] += ((yy[1] < sq_upper) ? 1 : 0);

                //Accumulate
                AA.m_limbs[0] = add_cc(AA.m_limbs[0], yy[0]);
#pragma unroll
                for (int j = 1; j <= y_size - i; j++)
                {
                    AA.m_limbs[j] = addc_cc(AA.m_limbs[j], yy[j]);
                                    
                }
                
                //The lowest two terms are now complete and can be moved to the reduction step.
                BB.m_limbs[2 * i] = AA.m_limbs[0];
                BB.m_limbs[2 * i + 1] = AA.m_limbs[1];
              
                //we are done with the lowest word
                AA >>= 64;
            }
            
            result = montgomery_reduce(BB, m, m_primed);

            //repeat for the top half of the square
#pragma unroll
            for (int i = 0; i <= x.HIGH_WORD / 2; i++)
            {
                uint64_t w = x.m_limbs[i + (x.HIGH_WORD+1)/2];
                //the square term
                uint64_t sq = w * w;
                //multiply the remaining upper portion of x by the current word (y = x * w)
                uint32_t carry = 0;

                //mulitply y = x * xi excluding the square term
#pragma unroll
                for (int j = 1; j <= y_size - i - (x.HIGH_WORD + 1) / 2 - 1; j++)
                {
                    uint64_t uv = x.m_limbs[i + j + (x.HIGH_WORD+1) / 2] * w + carry;
                    yy[j] = uv; //low word
                    carry = uv >> 32;  //the upper bits is the carry
                }

                yy[y_size - i - (x.HIGH_WORD + 1) / 2] = 0;

                yy[0] = 0;
                //multiply y by 2 (funnel shift left)
#pragma unroll
                for (int j = y_size - i - (x.HIGH_WORD + 1) / 2; j >= 1; j--)
                {
                    yy[j] = __funnelshift_l(yy[j - 1], yy[j], 1);
                }
                //add the square term to the doubled terms
                yy[0] = sq;
                uint32_t sq_upper = sq >> 32;
                yy[1] += sq_upper;
                //the carry can propogate max 1 word since all the double terms are even
                yy[2] += ((yy[1] < sq_upper) ? 1 : 0);

                //Accumulate
                AA.m_limbs[0] = add_cc(AA.m_limbs[0], yy[0]);
#pragma unroll
                for (int j = 1; j <= y_size - i - (x.HIGH_WORD + 1) / 2; j++)
                {
                    AA.m_limbs[j] = addc_cc(AA.m_limbs[j], yy[j]);
                }

                //The lowest two terms are now complete and can be moved to the reduction step.
                BB.m_limbs[2 * i] = AA.m_limbs[0];
                BB.m_limbs[2 * i + 1] = AA.m_limbs[1];
                
                //we are done with the lowest word
                AA >>= 64;
            }

            result += BB;

            if (result >= m)
            {
                result -= m;
            }

            return result;
            
        }

        //reduce x to xR^-1 mod m
        //HAC 14.32
        template<int BITS> __device__ Cump<BITS> montgomery_reduce(const Cump<BITS>& x, const Cump<BITS>& m, uint32_t m_primed)
        {
            Cump<BITS> A = x;
#pragma unroll
            for (auto i = 0; i <= m.HIGH_WORD; i++)
            {
                uint32_t u = A.m_limbs[0] * m_primed;
                A += m * u;
                A >>= 32;
            }
            
            if (A >= m)
            {
                A -= m;
            }
            return A;
        }

        //montgomery reduce x mod m one word at a time. The result is stored in A.
        //Use this function inside a loop over the words of x, starting with the least significant word, iterating over the number of words in m.
        //Initialize A with 0 before the first call.
        // if x has more words than m (x may have up to 2x more words than m), add the top half of x to the result.
        //At the end 0 < A <= 2m.  To get A mod m, check for A > m and subract m if needed. 
        template<int BITS> __device__ void montgomery_reduce(Cump<BITS>& A, uint32_t x, const Cump<BITS>& m, uint32_t m_primed)
        {
            uint32_t u = (A.m_limbs[0] + x) * m_primed;
            A += m * u + x;
            A >>= 32;
        }

        //returns true if 2^(m-1) mod m == 1, false otherwise
        //m_primed and rmodm are precalculated values.  See hac 14.94.  
        //R^2 mod m is not needed because with base 2 is trivial to calculate 2*R mod m given R mod m
        template<int BITS> __device__ bool powm_2(const Cump<BITS>& base_m, uint64_t offset)
        {
            const Cump<BITS>& m = base_m + offset;
            
            //precalculation of some constants based on the modulus
            const uint32_t m_primed = -mod_inverse_32(m.m_limbs[0]);

            //initialize the product to R mod m, the equivalent of 1 in the montgomery domain
            Cump<BITS> A = m.R_mod_m();
            
            //Cump<BITS> exp = m - 1;  //m is odd, so m-1 only changes the lowest bit of m from 1 to 0. 

            //perform the first few iterations without montgomery multiplication - we are multiplying by small powers of 2
            int word = A.HIGH_WORD;
            const int top_bits_window = 4;
            int shift = (A.BITS_PER_WORD - top_bits_window) % A.BITS_PER_WORD;
            int mask = ((1 << top_bits_window) - 1) << shift;
            int top_bits = (m.m_limbs[word] & mask) >> shift;
            A = double_and_reduce(A, m, top_bits);
            //Go through each bit of the exponment.  We assume all words except the lower three match the base big int
            for (auto i = BITS - top_bits_window - 1; i >= 1; i--)
            {
                //square
                A = montgomery_square_2(A, m, m_primed);
                word = i / A.BITS_PER_WORD;
                int bit = i % A.BITS_PER_WORD;
                mask = 1 << bit;
                bool bit_is_set = (m.m_limbs[word] & mask) != 0;
                if (bit_is_set)
                {
                    //multiply by the base (2) if the exponent bit is set
                    A = double_and_reduce(A, m, 1);

                }
                
            }
            //the final iteration happens here. the exponent m-1 lowest bit is always 0 so we never need to double and reduce after squaring
            //The final squaring can be avoided if we check the if the current value is +1 or -1 in the montgomery domain 
            bool pass = false;
            if (A == m.R_mod_m() || A == (m - m.R_mod_m()))
            {
                pass = true;
            }    
            //A = montgomery_square_2(A, m, m_primed);
            //convert back from montgomery domain
            //A = montgomery_reduce(A, m, m_primed);
            return pass;

        }
        
        // return 2 * x mod m given x and m using shift and subtract.
        //For this to be efficient, m must be similar magnitude (within a few bits) of x. 
        template<int BITS> __device__ Cump<BITS> double_and_reduce(const Cump<BITS>& x, const Cump<BITS>& m, int shift)
        {
            Cump<BITS> A = x << shift;
            while (A >= m) 
            {
                A -= m;
            }
            return A;
        }




    }
}
