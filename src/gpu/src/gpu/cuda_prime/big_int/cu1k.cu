#include "cu1k.cuh"

namespace nexusminer {
    namespace gpu {

        __host__ __device__ Cu1k::Cu1k() : m_limbs{}
        {
        }

        __host__ __device__ Cu1k::Cu1k(uint32_t init32) : m_limbs{}
        {
            m_limbs[0] = init32;
        }

        __host__ __device__ Cu1k Cu1k::add(const Cu1k& b) const
        {
            Cu1k result;
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

        __host__ __device__ Cu1k Cu1k::sub(const Cu1k& b) const
        {
            //subtraction as 2s complement addition.  for some reason this uses more registers than addition.
            Cu1k result;
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

        //same as add, but results are stored in the current object
        __host__ __device__ void Cu1k::increment(const Cu1k& b)
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

        __host__ __device__ void Cu1k::operator+=(const Cu1k& b)
        {
            increment(b);
        }

        //same as subtract, but results are stored in the current object
        __host__ __device__ void Cu1k::decrement(const Cu1k& b)
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

        __host__ __device__ void Cu1k::operator-=(const Cu1k& b)
        {
            decrement(b);
        }

        //normalized multiprecision integer division.  Return both quotient and remainder.  Reference HAC chapter 14 algorithm 14.20.
        __device__ void Cu1k::divide(const Cu1k& divisor, Cu1k& quotient, Cu1k& remainder) const
        {
            const bool divide_debug = false;  //enable printfs
            if (divisor == 0)
            {
                if (divide_debug) printf("gpu division by zero detected.\n");
                quotient = 0;
                remainder = 0;
                return;
            }
            if (divisor == 1)
            {
                quotient = *this;
                remainder = 0;
                return;
            }
            if (divisor == 2)
            {
                quotient = *this >> 1;
                remainder = m_limbs[0] & 0x01;
                return;
            }
            int cmp = compare(divisor);
            if (cmp == 0)
            {
                //the dividend and divisor are equal
                quotient = 1;
                remainder = 0;
                return;
            }
            if (cmp == -1)
            {
                //the divisor is larger than the dividend
                quotient = 0;
                remainder = *this;
                return;
            }
            quotient = 0;
            Cu1k x = *this;
            Cu1k y = divisor;
            
            //t is the highest non-0 word of the divisor
            int t;
            for (t = divisor.LIMBS - 1; t >= 0; t--)
            {
                if (divisor.m_limbs[t] > 0)
                    break;
            }
            //normalize by shifting both dividend and divisor left until the MSB of the first divisor word is 1.
            //this can overflow the dividend so we have to use the extra word.
            int normalize_shift = __clz(divisor.m_limbs[t]);  //clz is an nvidia primitive that counts the leading zeros of a word
            
            x = x << normalize_shift;
            y = y << normalize_shift;
            //n is the highest non-0 word of the dividend
            int n;
            for (n = x.LIMBS - 1; n >= 0; n--)
            {
                if (x.m_limbs[n] > 0)
                    break;
            }

            if (divide_debug) printf("n %i t %i normalize_shift %i \n", n, t, normalize_shift);
            //after normalization, this loop should execute max once
            Cu1k temp = y << (32 * (n - t));
            while (x >= temp)
            {
                quotient.m_limbs[n - t] += 1;  
                x -= temp;
            }
            char s[400];
            if (divide_debug) {
                printf("step 2\n");
                x.to_cstr(s);
                printf("x = %s\n", s);
                y.to_cstr(s);
                printf("y = %s\n", s);
                quotient.to_cstr(s);
                printf("q = %s\n", s);
            }

            //step 3
            for (auto i = n; i > t; i--)  //t can be 0.  i is >= 1;
            {
                
                int j = i - t - 1;  //the index of the current quotient word.  j >= 0;
                uint32_t xi = x.m_limbs[i];
                if (divide_debug) printf("3.1 i %i j %i\n", i, j);
                //3.1
                if (xi == y.m_limbs[t])
                {
                    quotient.m_limbs[j] = 0xFFFFFFFF;
                    if (divide_debug) printf("3.1a\n");
                }
                else
                {
                    //perform double precision division using the upper words
                    quotient.m_limbs[j] = ((static_cast<uint64_t>(xi) << 32) + x.m_limbs[i - 1]) / y.m_limbs[t];
                    if (divide_debug) printf("3.1b q = %08x\n", quotient.m_limbs[j]);

                }
                //3.2
                //determine if the estimate for qy is greater than x.  this gives a triple precision result so we use two 64 bit words for qy
                uint64_t y_upper = (static_cast<uint64_t>(y.m_limbs[t]) << 32) | static_cast<uint64_t>(t > 0 ? y.m_limbs[t - 1] : 0);
                uint64_t qy_low = y_upper * quotient.m_limbs[j];
                //this is triple precision so we only need the lower 32 bits from the upper 64 bit multiplication result
                uint32_t qy_upper = __umul64hi(y_upper, quotient.m_limbs[j]);  //todo: deal with non-portable cuda intrinsic for high word multiply.  
                if (divide_debug) printf("y_t = %08x y_upper = %016llx qy_low = %016llx qy_upper = %08x xi = %08x\n",
                    y.m_limbs[t], y_upper, qy_low, qy_upper, xi);
                //while the estimate for qy is greater than x
                while ((qy_upper > xi) || ((qy_upper == xi) && ((qy_low >> 32) > x.m_limbs[i-1])) ||
                    ((qy_upper == xi) && ((qy_low >> 32) == x.m_limbs[i - 1]) && (static_cast<uint32_t>(qy_low) > i>=2?x.m_limbs[i - 2]:0)))
                {
                    quotient.m_limbs[j]--;
                    //update the estimate
                    qy_low = y_upper * quotient.m_limbs[j];
                    qy_upper = __umul64hi(y_upper, quotient.m_limbs[j]);
                    if (divide_debug) printf("Inisde 3.2 correction loop.  qy_upper = %08x q_j = %08x\n", qy_upper, quotient.m_limbs[j]);
                }
                //3.3 subtract q*y from x, where q is the current single precision quotient word we are working on and y is the full precision y. 
                
                uint32_t multiplication_carry = 0;  //carry for the multiply
                uint32_t addition_carry = 1; //borrow for the subtraction
                bool propagate = false;  //addition carry propagate
                bool generate = false; //addition carry generate
                //multiply and subtract in one loop to minimize need for intermediate storage
                for (auto k = 0; k <= t; k++)
                {
                    uint32_t yk = y.m_limbs[k];
                    uint64_t q = quotient.m_limbs[j];  //cast to 64 bits 
                    uint64_t qy64 = q * yk + multiplication_carry;  //perform multiplication of one word
                    uint32_t qy = static_cast<uint32_t>(qy64);  //keep the lower part of the multiplication
                    multiplication_carry = qy64 >> 32;  //carry the upper part of the multiplication result
                    //subtract qy from x by adding the two's complement
                    int x_index = j + k;
                    uint32_t xx = x.m_limbs[x_index];
                    uint32_t x_pre_carry = xx + ~qy;
                    propagate = x_pre_carry == 0xFFFFFFFF;
                    generate = x_pre_carry < xx;
                    x.m_limbs[x_index] = x_pre_carry + addition_carry;
                    addition_carry = generate || (propagate && addition_carry) ? 1 : 0;
                    if (divide_debug) printf("3.3 k %i j %i q_j %08x y_k %08x qy %08x mult_carry %08x xx %08x x_pre %08x add_carry %i\n",
                        k, j, quotient.m_limbs[j], yk, qy, multiplication_carry, xx, x_pre_carry, addition_carry);
                }

                //handle carries to the final word
                int x_index = j + t + 1;
                uint32_t original_word = x.m_limbs[x_index];
                uint32_t x_pre_carry = x.m_limbs[x_index] + ~multiplication_carry;
                x.m_limbs[x_index] = x_pre_carry + addition_carry;
                //check for overflow
                bool overflow = x.m_limbs[x_index] > original_word;
                if (divide_debug){
                    x.to_cstr(s);
                    printf("After 3.3 %s\n", s);
                    quotient.to_cstr(s);
                    printf("q = %s\n", s);
                }
                //3.4 check if the previous subtraction of qy overflowed.  if so add back one y
                if (overflow)
                {
                    if (divide_debug) printf("3.4 correction\n");
                    addition_carry = 0;
                    for (auto k = 0; k <= t; k++)
                    {
                        int x_index = j + k;
                        uint32_t xx = x.m_limbs[x_index];
                        uint32_t yk = y.m_limbs[k];
                        uint32_t x_pre_carry = xx + yk;
                        propagate = x_pre_carry == 0xFFFFFFFF;
                        generate = x_pre_carry < xx;
                        x.m_limbs[x_index] = x_pre_carry + addition_carry;
                        addition_carry = generate || (propagate && addition_carry) ? 1 : 0;
                    }
                    //handle carries to the final word
                    int x_index = j + t + 1;
                    x.m_limbs[x_index] += addition_carry;
                    
                    //decrement the quotient word
                    quotient.m_limbs[j]--;

                    if (divide_debug) {
                        x.to_cstr(s);
                        printf("After 3.4 correction %s\n", s);
                        quotient.to_cstr(s);
                        printf("q = %s\n", s);
                    }
                }
            }
            remainder = x >> normalize_shift;  //denormalize the remainder
            if (divide_debug) {
                remainder.to_cstr(s);
                printf("remainder = %s\n", s);
            }
            return;
        }

        __host__ __device__ Cu1k Cu1k::operator<<(int shift) const
        {
            Cu1k result;
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

        __host__ __device__ Cu1k Cu1k::operator>>(int shift) const
        {
            Cu1k result;
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

        __host__ __device__ Cu1k Cu1k::operator~() const
        {
            Cu1k result;
            for (auto i = 0; i < LIMBS; i++)
            {
                result.m_limbs[i] = ~m_limbs[i];
            }
            return result;
        }

        //return the modular inverse if it exists using the binary extended gcd algorithm 
        //Reference HAC chapter 14 algorithm 14.61
        //The modulus m must be odd
        __host__ __device__ Cu1k Cu1k::modinv(const Cu1k& m) const
        {
            //the modulus m must be odd and greater than 1 and the number to invert must be non zero plus other restrictions
            if (m.m_limbs[0] % 2 == 0 || m < 1 || *this == 0 )
                return 0;
            const bool debug = false;
            //const Cu1k y = *this;
            const Cu1k x = m;  //x == m is odd and > 0
            Cu1k u = m;  //u, x >= 0
            Cu1k v = *this;  //v, y >= 0
            //steps 1 and 2 don't happen if m is odd
            //step 3
            Cu1k B = 0, D = 1;  //B and D can be negative, we must deal with the signs    
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
        __host__ __device__ int Cu1k::compare(const Cu1k& b) const
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
        __host__ __device__ void Cu1k::to_cstr(char* s)
        {
            int string_index = 0;
            /*if (m_sign < 0)
            {
                s[0] = '-';
                string_index = 1;
            }*/
            
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

        __host__ __device__ Cu1k operator + (const Cu1k& lhs, const Cu1k& rhs)
        {
            return lhs.add(rhs);
        }

        __host__ __device__ Cu1k operator - (const Cu1k& lhs, const Cu1k& rhs)
        {
            return lhs.sub(rhs);
        }

        __device__ Cu1k operator / (const Cu1k& lhs, const Cu1k& rhs)
        {
            Cu1k q, r;
            lhs.divide(rhs, q, r);
            return q;
        }

        __device__ Cu1k operator % (const Cu1k& lhs, const Cu1k& rhs)
        {
            Cu1k q, r;
            lhs.divide(rhs, q, r);
            return r;
        }

        __host__ __device__ bool operator > (const Cu1k& lhs, const Cu1k& rhs)
        {
            return lhs.compare(rhs) == 1;
        }
        __host__ __device__ bool operator < (const Cu1k& lhs, const Cu1k& rhs)
        {
            return lhs.compare(rhs) == -1;
        }

        __host__ __device__ bool operator==(const Cu1k& lhs, const Cu1k& rhs)
        {
            return lhs.compare(rhs) == 0;
        }

        __host__ __device__ bool operator>=(const Cu1k& lhs, const Cu1k& rhs)
        {
            return lhs.compare(rhs) >= 0;
        }

        __host__ __device__ bool operator<=(const Cu1k& lhs, const Cu1k& rhs)
        {
            return lhs.compare(rhs) <= 0;
        }

        __host__ __device__ bool operator!=(const Cu1k& lhs, const Cu1k& rhs)
        {
            return lhs.compare(rhs) != 0;
        }

        //convert the cuda 1024 bit unsigned int to a gmp multiprecision integer
        __host__ void Cu1k::to_mpz(mpz_t r)
        {
            mpz_import(r, LIMBS, -1, sizeof(uint32_t), 0, 0, m_limbs);
        }

        //convert a gmp multiprecision integer to a cuda 1024 bit unsigned int 
        __host__ void Cu1k::from_mpz(mpz_t s) {
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


        //for debug/printing
        /* A utility function to reverse a string  */
        __host__ __device__ void reverse(char str[], int length)
        {
            int start = 0;
            int end = length - 1;
            while (start < end)
            {
                //swap
                char tmp = *(str + start);
                *(str + start) = *(str + end);
                *(str + end) = tmp;
                start++;
                end--;
            }
        }

        // unsigned itoa() always use base 16
        __host__ __device__ char* itoa(unsigned int num, char* str)
        {
            int i = 0;
            int base = 16;

            // Process individual digits
            while (i < 2*sizeof(num))
            {
                int rem = num % base;
                str[i++] = (rem > 9) ? (rem - 10) + 'a' : rem + '0';
                num = num / base;
            }

            str[i] = '\0'; // Append string terminator

            // Reverse the string
            reverse(str, i);

            return str;
        }

    }
}
