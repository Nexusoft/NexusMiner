

namespace nexusminer {
    namespace gpu {

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
        __host__ __device__ Cump<BITS> Cump<BITS>::add(uint64_t b) const
        {
            Cump result;
            uint64_t x = (static_cast<uint64_t>(m_limbs[1]) << 32) + m_limbs[0];
            x += b;
            result.m_limbs[0] = x;
            result.m_limbs[1] = x >> 32;
            bool carry = x < b;
            for (int i = 2; i < LIMBS; i++)
            {
                result.m_limbs[i] = m_limbs[i] + (carry ? 1 : 0);
                carry = result.m_limbs[i] < m_limbs[i];
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

        template<int BITS>
        __host__ __device__ void Cump<BITS>::operator+=(uint64_t b)
        {
            *this = add(b);
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
        __host__ __device__ Cump<BITS> operator+(const Cump<BITS>& lhs, uint32_t rhs)
        {
            return lhs.add(rhs);
        }

        template<int BITS>
        __host__ __device__ Cump<BITS> operator+(const Cump<BITS>& lhs, uint64_t rhs)
        {
            return lhs.add(rhs);
        }


        //add two uint1024s directly in ptx (cuda assembly code)
        /* template<int BITS>
        __device__ Cump<BITS> Cump<BITS>::add_ptx(const Cump<BITS>& x) const
        {
            Cump<BITS> result;
            
            asm("{\n\t"
                "add.cc.u32       %0, %34, %68;    \n\t" //a[0] + x[0] with carry out
                "addc.cc.u32      %1, %35, %69;    \n\t" //a[1] + x[1] with carry in and carry out
                "addc.cc.u32      %2, %36, %70;    \n\t" 
                "addc.cc.u32      %3, %37, %71;    \n\t"
                "addc.cc.u32      %4, %38, %72;    \n\t"
                "addc.cc.u32      %5, %39, %73;    \n\t"
                "addc.cc.u32      %6, %40, %74;    \n\t"
                "addc.cc.u32      %7, %41, %75;    \n\t"
                "addc.cc.u32      %8, %42, %76;    \n\t"
                "addc.cc.u32      %9, %43, %77;    \n\t"
                "addc.cc.u32      %10, %44, %78;    \n\t"
                "addc.cc.u32      %11, %45, %79;    \n\t"
                "addc.cc.u32      %12, %46, %80;    \n\t"
                "addc.cc.u32      %13, %47, %81;    \n\t"
                "addc.cc.u32      %14, %48, %82;    \n\t"
                "addc.cc.u32      %15, %49, %83;    \n\t"
                "addc.cc.u32      %16, %50, %84;    \n\t"
                "addc.cc.u32      %17, %51, %85;    \n\t"
                "addc.cc.u32      %18, %52, %86;    \n\t"
                "addc.cc.u32      %19, %53, %87;    \n\t"
                "addc.cc.u32      %20, %54, %88;    \n\t"
                "addc.cc.u32      %21, %55, %89;    \n\t"
                "addc.cc.u32      %22, %56, %90;    \n\t"
                "addc.cc.u32      %23, %57, %91;    \n\t"
                "addc.cc.u32      %24, %58, %92;    \n\t"
                "addc.cc.u32      %25, %59, %93;    \n\t"
                "addc.cc.u32      %26, %60, %94;    \n\t"
                "addc.cc.u32      %27, %61, %95;    \n\t"
                "addc.cc.u32      %28, %62, %96;    \n\t"
                "addc.cc.u32      %29, %63, %97;    \n\t"
                "addc.cc.u32      %30, %64, %98;    \n\t"
                "addc.cc.u32      %31, %65, %99;    \n\t"
                "addc.cc.u32      %32, %66, %100;    \n\t"
                "addc.u32         %33, %67, %101;    \n\t"
                
                "}"
                : "=r"(result.m_limbs[0]), "=r"(result.m_limbs[1]), "=r"(result.m_limbs[2]), "=r"(result.m_limbs[3]),
                "=r"(result.m_limbs[4]), "=r"(result.m_limbs[5]), "=r"(result.m_limbs[6]), "=r"(result.m_limbs[7]),
                "=r"(result.m_limbs[8]), "=r"(result.m_limbs[9]), "=r"(result.m_limbs[10]), "=r"(result.m_limbs[11]),
                "=r"(result.m_limbs[12]), "=r"(result.m_limbs[13]), "=r"(result.m_limbs[14]), "=r"(result.m_limbs[15]),
                "=r"(result.m_limbs[16]), "=r"(result.m_limbs[17]), "=r"(result.m_limbs[18]), "=r"(result.m_limbs[19]),
                "=r"(result.m_limbs[20]), "=r"(result.m_limbs[21]), "=r"(result.m_limbs[22]), "=r"(result.m_limbs[23]),
                "=r"(result.m_limbs[24]), "=r"(result.m_limbs[25]), "=r"(result.m_limbs[26]), "=r"(result.m_limbs[27]),
                "=r"(result.m_limbs[28]), "=r"(result.m_limbs[29]), "=r"(result.m_limbs[30]), "=r"(result.m_limbs[31]),
                "=r"(result.m_limbs[32]), "=r"(result.m_limbs[33])
                : "r"(m_limbs[0]), "r"(m_limbs[1]), "r"(m_limbs[2]), "r"(m_limbs[3]),
                "r"(m_limbs[4]), "r"(m_limbs[5]), "r"(m_limbs[6]), "r"(m_limbs[7]),
                "r"(m_limbs[8]), "r"(m_limbs[9]), "r"(m_limbs[10]), "r"(m_limbs[11]),
                "r"(m_limbs[12]), "r"(m_limbs[13]), "r"(m_limbs[14]), "r"(m_limbs[15]),
                "r"(m_limbs[16]), "r"(m_limbs[17]), "r"(m_limbs[18]), "r"(m_limbs[19]),
                "r"(m_limbs[20]), "r"(m_limbs[21]), "r"(m_limbs[22]), "r"(m_limbs[23]),
                "r"(m_limbs[24]), "r"(m_limbs[25]), "r"(m_limbs[26]), "r"(m_limbs[27]),
                "r"(m_limbs[28]), "r"(m_limbs[29]), "r"(m_limbs[30]), "r"(m_limbs[31]),
                "r"(m_limbs[32]), "r"(m_limbs[33]),
                "r"(x.m_limbs[0]), "r"(x.m_limbs[1]), "r"(x.m_limbs[2]), "r"(x.m_limbs[3]),
                "r"(x.m_limbs[4]), "r"(x.m_limbs[5]), "r"(x.m_limbs[6]), "r"(x.m_limbs[7]),
                "r"(x.m_limbs[8]), "r"(x.m_limbs[9]), "r"(x.m_limbs[10]), "r"(x.m_limbs[11]),
                "r"(x.m_limbs[12]), "r"(x.m_limbs[13]), "r"(x.m_limbs[14]), "r"(x.m_limbs[15]),
                "r"(x.m_limbs[16]), "r"(x.m_limbs[17]), "r"(x.m_limbs[18]), "r"(x.m_limbs[19]),
                "r"(x.m_limbs[20]), "r"(x.m_limbs[21]), "r"(x.m_limbs[22]), "r"(x.m_limbs[23]),
                "r"(x.m_limbs[24]), "r"(x.m_limbs[25]), "r"(x.m_limbs[26]), "r"(x.m_limbs[27]),
                "r"(x.m_limbs[28]), "r"(x.m_limbs[29]), "r"(x.m_limbs[30]), "r"(x.m_limbs[31]),
                "r"(x.m_limbs[32]), "r"(x.m_limbs[33])
            );
            
            return result;

        }

        
        template<int BITS>
        __device__ void Cump<BITS>::increment_ptx(const Cump<BITS>& x)
        {
            *this = add_ptx(x);

        } */
    }
}
