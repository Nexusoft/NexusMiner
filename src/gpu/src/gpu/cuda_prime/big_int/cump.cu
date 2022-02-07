//this file is included in cump.cuh
#include "cump_multiplication.cu"
#include "cump_division.cu"
#include "cump_montgomery.cu"


namespace nexusminer {
    namespace gpu {

        template<int BITS>
        __host__ __device__ Cump<BITS>::Cump() : m_limbs{}
        {
        }

        template<int BITS>
        __host__ __device__ Cump<BITS>::Cump(uint32_t init32) : m_limbs{}
        {
            m_limbs[0] = init32;
        }

        template<int BITS>
        Cump<BITS>::Cump(int init) : m_limbs{}
        {
            m_limbs[0] = init;
            if (init < 0)
            {
                for (auto i = 1; i < LIMBS; i++)
                {
                    m_limbs[i] = ~(0u);
                }
            }
        }

        template<int BITS>
        __host__ __device__ Cump<BITS> Cump<BITS>::add(const Cump<BITS>& b) const
        {
            Cump result;
            bool propagate = false;
            bool generate = false;
            uint32_t carry = 0;
            uint32_t x = 0;
            for (auto i = 0; i < LIMBS; i++)
            {
                x = m_limbs[i] + b.m_limbs[i];
                result.m_limbs[i] = x + carry;
                propagate = x == 0xFFFFFFFF;
                generate = x < m_limbs[i];
                carry = generate || (propagate && carry) ? 1 : 0;
            }
            return result;
        }

        template<int BITS>
        __host__ __device__ Cump<BITS> Cump<BITS>::add(int b) const
        {
            if (b < 0)
            {
                return sub(static_cast<uint32_t>(-b));
            }
            return add(static_cast<uint32_t>(b));
           
        }

        template<int BITS>
        __host__ __device__ Cump<BITS> Cump<BITS>::add(uint32_t b) const
        {
            Cump result;
            result.m_limbs[0] = m_limbs[0] + b;
            uint32_t carry = result.m_limbs[0] < m_limbs[0] ? 1 : 0;
            for (int i = 1; i < LIMBS; i++)
            {
                result.m_limbs[i] = m_limbs[i] + carry;
                carry = result.m_limbs[i] < m_limbs[i] ? 1 : 0;
            }
            return result;
        }

        template<int BITS>
        __host__ __device__ Cump<BITS> Cump<BITS>::sub(const Cump<BITS>& b) const
        {
            //subtraction as 2s complement addition.  for some reason this uses more registers than addition.
            Cump<BITS> result;
            bool propagate = false;
            bool generate = false;
            uint32_t carry = 1;  //initialize carry to 1 for two's complement addition
            uint32_t x = 0;
            for (auto i = 0; i < LIMBS; i++)
            {
                x = m_limbs[i] + ~b.m_limbs[i]; //add the ones complement plus carry = 1 in the next step for two's complement 
                result.m_limbs[i] = x + carry;
                propagate = x == 0xFFFFFFFF;
                generate = x < m_limbs[i];
                carry = generate || (propagate && carry) ? 1 : 0;
            }
            return result;
        }

        template<int BITS>
        __host__ __device__ Cump<BITS> Cump<BITS>::sub(int b) const
        {
            if (b < 0)
            {
                return add(static_cast<uint32_t>(-b));
            }
            return sub(static_cast<uint32_t>(b));
        }

        template<int BITS>
        __host__ __device__ Cump<BITS> Cump<BITS>::sub(uint32_t b) const
        {
            Cump result;
            result.m_limbs[0] = m_limbs[0] - b;
            uint32_t carry = result.m_limbs[0] > m_limbs[0] ? 1 : 0;
            for (int i = 1; i < LIMBS; i++)
            {
                result.m_limbs[i] = m_limbs[i] - carry;
                carry = result.m_limbs[i] > m_limbs[i] ? 1 : 0;
            }
            return result;
        }

        //same as add, but results are stored in the current object
        template<int BITS>
        __host__ __device__ void Cump<BITS>::increment(const Cump<BITS>& b)
        {
            bool propagate = false;
            bool generate = false;
            uint32_t carry = 0;
            uint32_t x = 0;
            for (auto i = 0; i < LIMBS; i++)
            {
                x = m_limbs[i] + b.m_limbs[i];
                propagate = x == 0xFFFFFFFF;
                generate = x < m_limbs[i];
                m_limbs[i] = x + carry;
                carry = generate || (propagate && carry) ? 1 : 0;
            }
        }

        template<int BITS>
        __host__ __device__ void Cump<BITS>::increment(int b)
        {
            if (b < 0)
                decrement(static_cast<uint32_t>(-b));
            else
                increment(static_cast<uint32_t>(b));

        }

        template<int BITS>
        __host__ __device__ void Cump<BITS>::increment(uint32_t b)
        {
            uint32_t temp = m_limbs[0];
            m_limbs[0] += b;
            uint32_t carry = m_limbs[0] < temp ? 1 : 0;
            for (int i = 1; i < LIMBS; i++)
            {
                temp = m_limbs[i];
                m_limbs[i] += carry;
                carry = m_limbs[i] < temp ? 1 : 0;
            }
        }

        template<int BITS>
        __host__ __device__ void Cump<BITS>::operator+=(const Cump<BITS>& b)
        {
            increment(b);
        }

        template<int BITS>
        __host__ __device__ void Cump<BITS>::operator+=(int b)
        {
            increment(b);
        }

        template<int BITS>
        __host__ __device__ void Cump<BITS>::operator+=(uint32_t b)
        {
            increment(b);
        }

        //same as subtract, but results are stored in the current object
        template<int BITS>
        __host__ __device__ void Cump<BITS>::decrement(const Cump<BITS>& b)
        {
            bool propagate = false;
            bool generate = false;
            uint32_t carry = 1;  //initialize carry to 1 for two's complement addition
            uint32_t x = 0;
            for (auto i = 0; i < LIMBS; i++)
            {
                x = m_limbs[i] + ~b.m_limbs[i]; //add the ones complement plus carry = 1 in the next step for two's complement 
                propagate = x == 0xFFFFFFFF;
                generate = x < m_limbs[i];
                m_limbs[i] = x + carry;
                carry = generate || (propagate && carry) ? 1 : 0;
            }
        }

        template<int BITS>
        __host__ __device__ void Cump<BITS>::decrement(int b)
        {
            if (b < 0)
                increment(static_cast<uint32_t>(-b));
            else
                decrement(static_cast<uint32_t>(b));
        }

        template<int BITS>
        __host__ __device__ void Cump<BITS>::decrement(uint32_t b)
        {
            uint32_t temp = m_limbs[0];
            m_limbs[0] -= b;
            uint32_t carry = m_limbs[0] > temp ? 1 : 0;
            for (int i = 1; i < LIMBS; i++)
            {
                temp = m_limbs[i];
                m_limbs[i] -= carry;
                carry = m_limbs[i] > temp ? 1 : 0;
            }
        }

        template<int BITS>
        __host__ __device__ void Cump<BITS>::operator-=(const Cump<BITS>& b)
        {
            decrement(b);
        }

        template<int BITS>
        __host__ __device__ void Cump<BITS>::operator-=(int b)
        {
            decrement(b);
        }

        template<int BITS>
        __host__ __device__ void Cump<BITS>::operator-=(uint32_t b)
        {
            decrement(b);
        }

        template<int BITS>
        __host__ __device__ Cump<BITS> Cump<BITS>::operator<<(int shift) const
        {
            
            Cump<BITS> result;
            if (shift == 0)
                return *this;
            if (shift < 0)
            {
                return (*this >> -shift);
            }
            int whole_word_shift = shift / BITS_PER_WORD;
            if (whole_word_shift >= LIMBS)
            {
                return 0;
            }
            int bit_shift = shift % BITS_PER_WORD;
            uint32_t save_bits = 0;
            for (int i = whole_word_shift; i < LIMBS; i++)
            {
                uint32_t mask1 = bit_shift > 0 ? save_bits >> (BITS_PER_WORD - bit_shift) : 0;
                uint32_t mask2 = bit_shift > 0 ? ~0 << (BITS_PER_WORD - bit_shift) : ~0;
                result.m_limbs[i] = ((m_limbs[i - whole_word_shift] << bit_shift) | mask1);
                save_bits = m_limbs[i - whole_word_shift] & mask2;
            }
            return result;
        }

        //left shift in place
        template<int BITS>
        __host__ __device__ void Cump<BITS>::operator<<=(int shift)
        {
            if (shift == 0)
                return;
            if (shift < 0)
            {
                return;
            }
            int whole_word_shift = shift / BITS_PER_WORD;
            
            const int bit_shift = shift % BITS_PER_WORD;
            const uint32_t upper_bits_mask = bit_shift > 0 ? ~0 << (BITS_PER_WORD - bit_shift) : ~0;

            for (int i = LIMBS-1; i >= whole_word_shift; i--)
            {
                int source_word_index = i - whole_word_shift;
                uint32_t upper_bits = (source_word_index - 1 >= 0) ? m_limbs[source_word_index - 1] & upper_bits_mask : 0;
                uint32_t lower_bits = bit_shift > 0 ? upper_bits >> (BITS_PER_WORD - bit_shift) : 0;
                m_limbs[i] = ((m_limbs[source_word_index] << bit_shift) | lower_bits);
                
            }
            for (int i = 0; i < whole_word_shift && i < LIMBS; i++)
            {
                m_limbs[i] = 0;
            }
            return;
        }

        //right shift in place
        template<int BITS>
        __host__ __device__ void Cump<BITS>::operator>>=(int shift)
        {
            if (shift == 0)
                return;
            if (shift < 0)
            {
                return;
            }
            int whole_word_shift = shift / BITS_PER_WORD;

            const int bit_shift = shift % BITS_PER_WORD;
            const uint32_t lower_bits_mask = bit_shift > 0 ? ~0 >> (BITS_PER_WORD - bit_shift) : ~0;

            for (int i = 0; i < LIMBS - whole_word_shift && i < LIMBS; i++)
            {
                int source_word_index = i + whole_word_shift;
                uint32_t lower_bits = (source_word_index + 1 < LIMBS) ? m_limbs[source_word_index + 1] & lower_bits_mask : 0;
                uint32_t upper_bits = bit_shift > 0 ? lower_bits << (BITS_PER_WORD - bit_shift) : 0;
                m_limbs[i] = ((m_limbs[source_word_index] >> bit_shift) | upper_bits);

            }
            for (int i = LIMBS - whole_word_shift; i < LIMBS && i >= 0; i++)
            {
                m_limbs[i] = 0;
            }
            return;
        }

        template<int BITS>
        __host__ __device__ Cump<BITS> Cump<BITS>::operator>>(int shift) const
        {
            Cump<BITS> result;
            if (shift == 0)
                return *this;
            if (shift < 0)
            {
                return (*this << -shift);
            }
            int whole_word_shift = shift / BITS_PER_WORD;
            if (whole_word_shift >= LIMBS)
            {
                return 0;
            }
            int bit_shift = shift % BITS_PER_WORD;
            uint32_t save_bits = 0;
            for (int i = LIMBS - 1 - whole_word_shift; i >= 0; i--)
            {
                uint32_t mask1 = bit_shift > 0 ? save_bits << (BITS_PER_WORD - bit_shift) : 0;
                uint32_t mask2 = bit_shift > 0 ? ~0 >> (BITS_PER_WORD - bit_shift) : ~0;
                result.m_limbs[i] = ((m_limbs[i + whole_word_shift] >> bit_shift) | mask1);
                save_bits = m_limbs[i + whole_word_shift] & mask2;
            }
            return result;
        }

        template<int BITS>
        __host__ __device__ Cump<BITS> Cump<BITS>::operator~() const
        {
            Cump<BITS> result;
            for (auto i = 0; i < LIMBS; i++)
            {
                result.m_limbs[i] = ~m_limbs[i];
            }
            return result;
        }

        //return the modular inverse if it exists using the binary extended gcd algorithm 
        //Reference HAC chapter 14 algorithm 14.61
        //The modulus m must be odd
        template<int BITS>
        __host__ __device__ Cump<BITS> Cump<BITS>::modinv(const Cump<BITS>& m) const
        {
            //the modulus m must be odd and greater than 1 and the number to invert must be non zero plus other restrictions
            if (m.m_limbs[0] % 2 == 0 || m < 1 || *this == 0 )
                return 0;
            const bool debug = false;
            //const Cump<BITS> y = *this;
            const Cump<BITS> x = m;  //x == m is odd and > 0
            Cump<BITS> u = m;  //u, x >= 0
            Cump<BITS> v = *this;  //v, y >= 0
            //steps 1 and 2 don't happen if m is odd
            //step 3
            Cump<BITS> B = 0, D = 1;  //B and D can be negative, we must deal with the signs    
            int i = 0;
            if (debug)
            {
                char us[400], vs[400], Bs[400], Ds[400];
                u.to_cstr(us);
                v.to_cstr(vs);
                B.to_cstr(Bs);
                D.to_cstr(Ds);
                printf("iteration %i\nu=%s\nv=%s\nB=%s\nD=%s\n", i, us, vs, Bs, Ds);
            }
            //max iterations is 2 * (2 * 1024 + 2) = 4100
            while (u != 0 && !u.m_limbs[LIMBS - 1])  //if u goes negative, stop.  something is wrong
            {
                //step 4
                while (!(u.m_limbs[0] & 1))  //while u is even
                {
                    u = u >> 1;
                    if (B.m_limbs[0] & 1)  //if B is odd
                    {
                        B = B - x;  //x is always odd so B should be even after this
                    }
                    
                    B = B >> 1;  //divide by 2
                    //copy the top bit to preserve the sign
                    B.m_limbs[LIMBS - 1] = B.m_limbs[LIMBS - 1] | ((B.m_limbs[LIMBS - 1] & (1u << 30)) << 1);
                    if (debug)
                    {
                        char us[400], vs[400], Bs[400], Ds[400];
                        u.to_cstr(us);
                        v.to_cstr(vs);
                        B.to_cstr(Bs);
                        D.to_cstr(Ds);
                        printf("4. u was even.  iteration %i\nu=%s\nv=%s\nB=%s\nD=%s\n", i, us, vs, Bs, Ds);
                    }
                    ++i;
                }
                //step 5
                while (!(v.m_limbs[0] & 1))  //while v is even
                {
                    v = v >> 1;
                    if (D.m_limbs[0] & 1)
                    {
                        D = D - x;  //x is always odd.  D should be even after this.
                    }
                    D = D >> 1;  //divide by 2
                    //copy the top bit to preserve the sign
                    D.m_limbs[LIMBS - 1] = D.m_limbs[LIMBS - 1] | ((D.m_limbs[LIMBS - 1] & (1u << 30)) << 1);
                    if (debug)
                    {
                        char us[400], vs[400], Bs[400], Ds[400];
                        u.to_cstr(us);
                        v.to_cstr(vs);
                        B.to_cstr(Bs);
                        D.to_cstr(Ds);
                        printf("5. v was even.  iteration %i\nu=%s\nv=%s\nB=%s\nD=%s\n", i, us, vs, Bs, Ds);
                    }
                    ++i;
                }
                //step 6
                if (u >= v)
                {
                    u -= v;
                    B -= D;
                }
                else
                {
                    v -= u;
                    D -= B;
                }
                if (debug)
                {
                    
                    char us[400], vs[400], Bs[400], Ds[400];
                    u.to_cstr(us);
                    v.to_cstr(vs);
                    B.to_cstr(Bs);
                    D.to_cstr(Ds);
                    printf("6. u=%s\nv=%s\nB=%s\nD=%s\n", us, vs, Bs, Ds);
                }
                
            }
            //if the result is negative, add moduli
            while (D.m_limbs[LIMBS - 1] & (1u << 31))
                D += m;

            //if the result is larger than the modulus, subtract moduli
            while (D > m)
                D -= m;

            if (debug)
            {
                char Ds[400];
                char vs[400];
                D.to_cstr(Ds);
                v.to_cstr(vs);
                printf("result=%s\nv=%s\n", Ds, vs);
            }

            //the inverse modulus does not exist
            if (v != 1)
                return 0;

            return D;

        }


        //returns 1 if the integer is greater than the value to compare, 0 if equal, -1 if less than
        template<int BITS>
        __host__ __device__ int Cump<BITS>::compare(const Cump<BITS>& b) const
        {
          
            for (auto i = LIMBS - 1; i >= 0; i--)
            {
                if (m_limbs[i] > b.m_limbs[i])
                    return 1;
                if (m_limbs[i] < b.m_limbs[i])
                    return -1;
            }
            return 0;

        }

        //create a string representing the big int in hexadecimal with word separators
        //the caller must allocate memory for the string prior to calling - use 384 bytes.   
        //use for debugging device code
        template<int BITS>
        __host__ __device__ void Cump<BITS>::to_cstr(char* s) const
        {
            int string_index = 0;
            
            for (auto i = LIMBS - 1; i >= 0; i--)
            {
                char limb_str[9];
                itoa(m_limbs[i], limb_str);
                
                //concatenate
                for (auto j = 0; j < 8; j++)
                {
                    s[string_index] = limb_str[j];
                    string_index++;
                }
                s[string_index] = ' ';
                string_index++;
            }
            s[string_index] = '\0';
        }

        template<int BITS>
        __host__ __device__ Cump<BITS> operator + (const Cump<BITS>& lhs, const Cump<BITS>& rhs)
        {
            return lhs.add(rhs);
        }

        template<int BITS>
        __host__ __device__ Cump<BITS> operator + (const Cump<BITS>& lhs, int rhs)
        {
            return lhs.add(rhs);
        }

        template<int BITS>
        __host__ __device__ Cump<BITS> operator - (const Cump<BITS>& lhs, const Cump<BITS>& rhs)
        {
            return lhs.sub(rhs);
        }

        template<int BITS>
        __host__ __device__ Cump<BITS> operator - (const Cump<BITS>& lhs, int rhs)
        {
            return lhs.sub(rhs);
        }

        template<int BITS>
        __device__ Cump<BITS> operator / (const Cump<BITS>& lhs, const Cump<BITS>& rhs)
        {
            Cump<BITS> q, r;
            lhs.divide(rhs, q, r);
            return q;
        }

        template<int BITS>
        __device__ Cump<BITS> operator % (const Cump<BITS>& lhs, const Cump<BITS>& rhs)
        {
            Cump<BITS> q, r;
            lhs.divide(rhs, q, r);
            return r;
        }

        template<int BITS>
        __host__ __device__ bool operator > (const Cump<BITS>& lhs, const Cump<BITS>& rhs)
        {
            return lhs.compare(rhs) == 1;
        }

        template<int BITS> __host__ __device__ bool operator>(const Cump<BITS>& lhs, int rhs)
        {
            return lhs.compare(rhs) == 1;
        }

        template<int BITS>
        __host__ __device__ bool operator < (const Cump<BITS>& lhs, const Cump<BITS>& rhs)
        {
            return lhs.compare(rhs) == -1;
        }

        template<int BITS> __host__ __device__ bool operator<(const Cump<BITS>& lhs, int rhs)
        {
            return lhs.compare(rhs) == -1;
        }

        template<int BITS>
        __host__ __device__ bool operator==(const Cump<BITS>& lhs, const Cump<BITS>& rhs)
        {
            return lhs.compare(rhs) == 0;
        }

        template<int BITS> __host__ __device__ bool operator==(const Cump<BITS>& lhs, int rhs)
        {
            return lhs.compare(rhs) == 0;
        }

        template<int BITS>
        __host__ __device__ bool operator>=(const Cump<BITS>& lhs, const Cump<BITS>& rhs)
        {
            return lhs.compare(rhs) >= 0;
        }

        template<int BITS> __host__ __device__ bool operator>=(const Cump<BITS>& lhs, int rhs)
        {
            return lhs.compare(rhs) >= 0;
        }

        template<int BITS>
        __host__ __device__ bool operator<=(const Cump<BITS>& lhs, const Cump<BITS>& rhs)
        {
            return lhs.compare(rhs) <= 0;
        }

        template<int BITS> __host__ __device__ bool operator<=(const Cump<BITS>& lhs, int rhs)
        {
            return lhs.compare(rhs) <= 0;
        }

        template<int BITS>
        __host__ __device__ bool operator!=(const Cump<BITS>& lhs, const Cump<BITS>& rhs)
        {
            return lhs.compare(rhs) != 0;
        }

        template<int BITS> __host__ __device__ bool operator!=(const Cump<BITS>& lhs, int rhs)
        {
            return lhs.compare(rhs) != 0;
        }

        

        //convert the cuda big unsigned int to a gmp multiprecision integer
        template<int BITS>
        __host__ void Cump<BITS>::to_mpz(mpz_t r)
        {
            mpz_import(r, LIMBS, -1, sizeof(uint32_t), 0, 0, m_limbs);
        }

        //convert a gmp multiprecision integer to a cuda unsigned int 
        template<int BITS>
        __host__ void Cump<BITS>::from_mpz(mpz_t s) {
            size_t words = 0;

            if (mpz_sizeinbase(s, 2) > LIMBS * 32) {
                fprintf(stderr, "from_mpz error. Data does not fit.\n");
            }
            else
            {
                mpz_export(m_limbs, &words, -1, sizeof(uint32_t), 0, 0, s);
            }

            while (words < LIMBS)
                m_limbs[words++] = 0;

        }


        

        //forward declaration of known template instantiations allows us to compile this code separate from the header
        //and avoid link errors
        //template class Cump<1024>;
        //template class Cump<2048>;

    }
}
