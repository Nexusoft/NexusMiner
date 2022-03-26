
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
        template<int BITS> __device__ void montgomery_square(Cump<BITS>& x, const Cump<BITS>& m, uint32_t m_primed)
        {
           
            Cump<BITS> A;
            #pragma unroll
            for (auto i = 0; i <= Cump<BITS>::HIGH_WORD; i++)
            {
                uint32_t u = (A.m_limbs[0] + x.m_limbs[i] * x.m_limbs[0])* m_primed;
                A = A + x * x.m_limbs[i] + m * u;
                A >>= 32;
            }
            if (A >= m)
            {
                A -= m;
            }
            x = A;
        }

        //montgomery square - optimized to take advantage of symmetry of squaring operation
        //Fermat testing spends most of its time inside this function
        //returns xxR^-1
        template<int BITS> __device__ void montgomery_square_2(Cump<BITS>& x, const Cump<BITS>& m, uint32_t m_primed)
        {
            
            //low half of the square
            Cump<BITS> AA, BB, y, result;
            constexpr int y_size = Cump<BITS>::HIGH_WORD + 2;
            uint32_t yy[y_size + 1];
            
#pragma unroll
            for (int i = 0; i <= Cump<BITS>::HIGH_WORD/2; i++)
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
                //AA.m_limbs[0] = AA.m_limbs[0] + yy[0];
                //unsigned char cc = AA.m_limbs[0] < yy[0];
#pragma unroll
                for (int j = 1; j <= y_size - i - 1; j++)
                {
                    AA.m_limbs[j] = addc_cc(AA.m_limbs[j], yy[j]);
                    //cc = add_carry(cc, AA.m_limbs[j], yy[j], &AA.m_limbs[j]);
                }
                
                //The lowest two terms are now complete and can be moved to the reduction step.
                BB.m_limbs[2 * i] = AA.m_limbs[0];
                BB.m_limbs[2 * i + 1] = AA.m_limbs[1];
              
                //we are done with the lowest word
                AA >>= 64;
            }
            result = BB;
            montgomery_reduce(result, m, m_primed);

            //repeat for the top half of the square
#pragma unroll
            for (int i = 0; i <= Cump<BITS>::HIGH_WORD / 2; i++)
            {
                uint64_t w = x.m_limbs[i + (Cump<BITS>::HIGH_WORD+1)/2];
                //the square term
                uint64_t sq = w * w;
                //multiply the remaining upper portion of x by the current word (y = x * w)
                uint32_t carry = 0;

                //mulitply y = x * xi excluding the square term
#pragma unroll
                for (int j = 1; j <= y_size - i - (Cump<BITS>::HIGH_WORD + 1) / 2 - 1; j++)
                {
                    uint64_t uv = x.m_limbs[i + j + (Cump<BITS>::HIGH_WORD+1) / 2] * w + carry;
                    yy[j] = uv; //low word
                    carry = uv >> 32;  //the upper bits is the carry
                }

                yy[y_size - i - (Cump<BITS>::HIGH_WORD + 1) / 2] = 0;

                yy[0] = 0;
                //multiply y by 2 (funnel shift left)
#pragma unroll
                for (int j = y_size - i - (Cump<BITS>::HIGH_WORD + 1) / 2; j >= 1; j--)
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
#if defined(GPU_CUDA_ENABLED)
                AA.m_limbs[0] = add_cc(AA.m_limbs[0], yy[0]); //inline asm gives the wrong answer here with amd.  why?
#else
                uint64_t tmp = (uint64_t)AA.m_limbs[0] + yy[0];
                uint8_t cc = tmp > 0xFFFFFFFF?1:0;
                AA.m_limbs[0] = tmp;
#endif
#pragma unroll
                for (int j = 1; j <= y_size - i - (Cump<BITS>::HIGH_WORD + 1) / 2; j++)
                {
#if defined(GPU_CUDA_ENABLED)
                    AA.m_limbs[j] = addc_cc(AA.m_limbs[j], yy[j]);
#else               
                    tmp = (uint64_t)AA.m_limbs[j] + yy[j] + cc;
                    cc = tmp > 0xFFFFFFFF?1:0;
                    AA.m_limbs[j] = tmp;
#endif
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

            x = result;
            
        }

        template<int BITS> __device__ void montgomery_square_3(Cump<BITS>& x, const Cump<BITS>& m, uint32_t m_primed)
        {
            static constexpr int accumulator_size = (Cump<BITS>::HIGH_WORD + 1) * 2 + 1;
            uint32_t Accumulator[accumulator_size];
            for (int i = 0; i<accumulator_size; i++)
            {
                Accumulator[i] = 0;
            }

            uint32_t sq, sq_upper, yy;
            //square
            for (int i=0; i <= Cump<BITS>::HIGH_WORD; i++)
            {
                //the current term
                uint32_t w = x.m_limbs[i];
                //the square term
                sq = mul_carry(w, w, 0, &sq_upper);
                uint32_t mul_carry_in = sq_upper / 2;
                unsigned char carry = sq_upper % 2;
                unsigned char accumulator_carry = add_carry(0, Accumulator[2*i], sq, &Accumulator[2*i]);
                for (int j=1; j < Cump<BITS>::LIMBS - i; j++)
                {
                    yy = mul_carry(x.m_limbs[i + j], w, mul_carry_in, &mul_carry_in);
                    unsigned char upper_bit = yy >> (Cump<BITS>::BITS_PER_WORD - 1);
                    yy = yy*2 + carry;
                    carry = upper_bit;
                    //accumulate
                    accumulator_carry = add_carry(accumulator_carry, Accumulator[2 * i + j], yy, &Accumulator[2 * i + j]);
                }
            }

            //reduce
            for (int i = 0; i <= Cump<BITS>::HIGH_WORD; i++)
            {
                uint32_t u = Accumulator[i] * m_primed;
                uint32_t carry = 0;
                unsigned char carry2 = 0;
                uint32_t mu[Cump<BITS>::LIMBS];
                for (int j = 0; j < Cump<BITS>::LIMBS; j++)
                {
                    //uint32_t muj = mul_carry(m.m_limbs[j], u, carry, &carry);
                    mu[j] = mul_carry(m.m_limbs[j], u, carry, &carry);
                    //carry2 = add_carry(carry2, Accumulator[i + j], muj, &Accumulator[i + j]);
                }
                for (int j = 0; j < Cump<BITS>::LIMBS; j++)
                {
                    carry2 = add_carry(carry2, Accumulator[i + j], mu[j], &Accumulator[i + j]);
                }
                Accumulator[i + Cump<BITS>::LIMBS] += carry2;
                
            }

            for (int i = 0; i <= Cump<BITS>::HIGH_WORD; i++)
            {
                x.m_limbs[i] = Accumulator[i + Cump<BITS>::HIGH_WORD + 1];
            }
            if (x >= m)
            {
                x -= m;
            }
        }

        //reduce x to xR^-1 mod m
        //HAC 14.32
        template<int BITS> __device__ void montgomery_reduce(Cump<BITS>& A, const Cump<BITS>& m, uint32_t m_primed)
        {
#pragma unroll
            for (int i = 0; i <= Cump<BITS>::HIGH_WORD; i++)
            {
                uint32_t u = A.m_limbs[0] * m_primed;
                A += m * u;
                A >>= 32;
            }
            
            if (A >= m)
            {
                A -= m;
            }
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
            int word = Cump<BITS>::HIGH_WORD;
            const int top_bits_window = 4;
            int shift = (Cump<BITS>::BITS_PER_WORD - top_bits_window) % Cump<BITS>::BITS_PER_WORD;
            int mask = ((1 << top_bits_window) - 1) << shift;
            int top_bits = (m.m_limbs[word] & mask) >> shift;
            A = double_and_reduce(A, m, top_bits);
            //Go through each bit of the exponent.  We assume all words except the lower three match the base big int
            for (auto i = BITS - top_bits_window - 1; i >= 1; i--)
            {
                //square
                montgomery_square_2(A, m, m_primed);
                word = i / Cump<BITS>::BITS_PER_WORD;
                int bit = i % Cump<BITS>::BITS_PER_WORD;
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
