

namespace nexusminer {
    namespace gpu {

        //normalized multiprecision integer division.  Return both quotient and remainder.  Reference HAC chapter 14 algorithm 14.20.
        template<int BITS>
        __device__ void Cump<BITS>::divide(const Cump<BITS>& divisor, Cump<BITS>& quotient, Cump<BITS>& remainder) const
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
            Cump<BITS> y = divisor;
            Cump<BITS> x = *this;
           
           

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
            y <<= normalize_shift;
            x <<= normalize_shift;
            
            //n is the highest non-0 word of the dividend
            int n;
            for (n = x.LIMBS - 1; n >= 0; n--)
            {
                if (x.m_limbs[n] > 0)
                    break;
            }

            if (divide_debug) printf("n %i t %i normalize_shift %i \n", n, t, normalize_shift);
            //after normalization, this loop should execute max once
            Cump<BITS>* temp = new Cump<BITS>;
            *temp = y << (32 * (n - t));
            while (x >= *temp)
            {
                quotient.m_limbs[n - t] += 1;
                x -= *temp;
            }
            delete temp;
            char s[LIMBS * 10];
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
                while ((qy_upper > xi) || ((qy_upper == xi) && ((qy_low >> 32) > x.m_limbs[i - 1])) ||
                    ((qy_upper == xi) && ((qy_low >> 32) == x.m_limbs[i - 1]) && (static_cast<uint32_t>(qy_low) > i >= 2 ? x.m_limbs[i - 2] : 0)))
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
                if (divide_debug) {
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

        //Same as multiprecision divide but return just the remainder, which uses fewer resources. Reference HAC chapter 14 algorithm 14.20.
        template<int BITS>
        __device__ void Cump<BITS>::remainder(const Cump<BITS>& divisor, Cump<BITS>& remainder) const
        {
            const bool divide_debug = false;  //enable printfs
            if (divisor == 0)
            {
                if (divide_debug) printf("gpu division by zero detected.\n");
                remainder = 0;
                return;
            }
            if (divisor == 1)
            {
                remainder = 0;
                return;
            }

            int cmp = compare(divisor);
            if (cmp == 0)
            {
                //the dividend and divisor are equal
                remainder = 0;
                return;
            }
            if (cmp == -1)
            {
                //the divisor is larger than the dividend
                remainder = *this;
                return;
            }

            uint32_t quotient_word = 0;
            Cump<BITS> y = divisor;
            Cump<BITS> x = *this;

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
            y <<= normalize_shift;
            x <<= normalize_shift;

            //n is the highest non-0 word of the dividend
            int n;
            for (n = x.LIMBS - 1; n >= 0; n--)
            {
                if (x.m_limbs[n] > 0)
                    break;
            }

            if (divide_debug) printf("n %i t %i normalize_shift %i \n", n, t, normalize_shift);
            //after normalization, this loop should execute max once
            Cump<BITS>* temp = new Cump<BITS>;
            *temp = y << (32 * (n - t));
            while (x >= *temp)
            {
                x -= *temp;
            }
            delete temp;

            char s[LIMBS * 10];
            if (divide_debug) {
                printf("step 2\n");
                x.to_cstr(s);
                printf("x = %s\n", s);
                y.to_cstr(s);
                printf("y = %s\n", s);
                
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
                    quotient_word = 0xFFFFFFFF;
                    if (divide_debug) printf("3.1a\n");
                }
                else
                {
                    //perform double precision division using the upper words
                    quotient_word = ((static_cast<uint64_t>(xi) << 32) + x.m_limbs[i - 1]) / y.m_limbs[t];
                    if (divide_debug) printf("3.1b q = %08x\n", quotient_word);

                }
                //3.2
                //determine if the estimate for qy is greater than x.  this gives a triple precision result so we use two 64 bit words for qy
                uint64_t y_upper = (static_cast<uint64_t>(y.m_limbs[t]) << 32) | static_cast<uint64_t>(t > 0 ? y.m_limbs[t - 1] : 0);
                uint64_t qy_low = y_upper * quotient_word;
                //this is triple precision so we only need the lower 32 bits from the upper 64 bit multiplication result
                uint32_t qy_upper = __umul64hi(y_upper, quotient_word);  //todo: deal with non-portable cuda intrinsic for high word multiply.  
                if (divide_debug) printf("y_t = %08x y_upper = %016llx qy_low = %016llx qy_upper = %08x xi = %08x\n",
                    y.m_limbs[t], y_upper, qy_low, qy_upper, xi);
                //while the estimate for qy is greater than x
                while ((qy_upper > xi) || ((qy_upper == xi) && ((qy_low >> 32) > x.m_limbs[i - 1])) ||
                    ((qy_upper == xi) && ((qy_low >> 32) == x.m_limbs[i - 1]) && (static_cast<uint32_t>(qy_low) > i >= 2 ? x.m_limbs[i - 2] : 0)))
                {
                    quotient_word--;
                    //update the estimate
                    qy_low = y_upper * quotient_word;
                    qy_upper = __umul64hi(y_upper, quotient_word);
                    if (divide_debug) printf("Inisde 3.2 correction loop.  qy_upper = %08x q_j = %08x\n", qy_upper, quotient_word);
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
                    uint64_t q = quotient_word;  //cast to 64 bits 
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
                        k, j, quotient_word, yk, qy, multiplication_carry, xx, x_pre_carry, addition_carry);
                }

                //handle carries to the final word
                int x_index = j + t + 1;
                uint32_t original_word = x.m_limbs[x_index];
                uint32_t x_pre_carry = x.m_limbs[x_index] + ~multiplication_carry;
                x.m_limbs[x_index] = x_pre_carry + addition_carry;
                //check for overflow
                bool overflow = x.m_limbs[x_index] > original_word;
                if (divide_debug) {
                    x.to_cstr(s);
                    printf("After 3.3 %s\n", s);

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
                    quotient_word--;

                    if (divide_debug) {
                        x.to_cstr(s);
                        printf("After 3.4 correction %s\n", s);
                        
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

        //Calculate R mod m where m is the modulus (this), R is 2^BITS 
        //The upper word (but not the extra word) of m must be non-zero.  the object is m. 
        //m must be less than, but relatively close to 2^BITS for this to work efficiently.
        //We find the remainder by repeatedly shifting and subtracting.
        template<int BITS>
        __device__ Cump<BITS> Cump<BITS>::R_mod_m() const
        {
            Cump<BITS> r;
            //set r to 2^bits
            const int extra_word = HIGH_WORD + 1;
            r.m_limbs[extra_word] = 1;
            if (m_limbs[extra_word] != 0 || m_limbs[HIGH_WORD] == 0)
                //error condition
                return 0;
            
            //leading zeros of the modulus
            int leading_zeros = __clz(m_limbs[HIGH_WORD]);
            //shift the modulus so the msb is set.  multiplying the divisor by an integer does not change the result mod m
            Cump<BITS> m_primed = *this << leading_zeros;
            
            while (r >= *this) //continue until the remainder is between 0 and m
            {
                while (r >= m_primed)
                {
                    r -= m_primed;
                }
                m_primed >>= 1;
            }

            return r;
        }

    }
}
