#include "fermat_utils.cuh"
namespace nexusminer {
    namespace gpu {

        //find the modular mutiplicative inverse of d mod 2^32
        //uses newtown's method.  See Hackers' Delight 10-16
        //d must be odd
        __host__ __device__ uint32_t mod_inverse_32(uint32_t d)
        {
            uint32_t xn, t;
            //initialize the estimate so that it is correct to 4 bits
            xn = d * d + d - 1;
            //for 32 bits the solution should converge after 3 iterations max (7 multiplies total)
            for (auto i = 0; i < 3; i++)
            {
                t = d * xn;
                xn = xn * (2 - t);
            }
            return xn;

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
            while (i < 2 * sizeof(num))
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
