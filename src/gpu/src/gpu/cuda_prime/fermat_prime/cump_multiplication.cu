

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


        //multiply a 1024 bit uint by one uint32_t directly in ptx (cuda assembly code)
        //this works but compiles to similar code as the c++.  The performance is slightly worse compared to the c++.
        template<int BITS>
        __device__ Cump<BITS> Cump<BITS>::multiply_ptx(uint32_t x) const
        {
            Cump<BITS> result;
            
            asm("{\n\t"
                "mul.lo.u32      %0, %34, %68;    \n\t" //a[0] * x low word
                "mul.hi.u32      %1, %34, %68;    \n\t" //a[0] * x high word -> overflow goes to next word
                "mad.lo.cc.u32   %1, %35, %68, %1;\n\t" //a[1] * x + previous with carry out
                "madc.hi.u32     %2, %35, %68,  0;\n\t" //a[1] * x + carry in 
                "mad.lo.cc.u32   %2, %36, %68, %2;\n\t"  
                "madc.hi.u32     %3, %36, %68,  0;\n\t"  
                "mad.lo.cc.u32   %3, %37, %68, %3;\n\t"
                "madc.hi.u32     %4, %37, %68,  0;\n\t"
                "mad.lo.cc.u32   %4, %38, %68, %4;\n\t"
                "madc.hi.u32     %5, %38, %68,  0;\n\t"
                "mad.lo.cc.u32   %5, %39, %68, %5;\n\t"
                "madc.hi.u32     %6, %39, %68,  0;\n\t"
                "mad.lo.cc.u32   %6, %40, %68, %6;\n\t"
                "madc.hi.u32     %7, %40, %68,  0;\n\t"
                "mad.lo.cc.u32   %7, %41, %68, %7;\n\t"
                "madc.hi.u32     %8, %41, %68,  0;\n\t"
                "mad.lo.cc.u32   %8, %42, %68, %8;\n\t"
                "madc.hi.u32     %9, %42, %68,  0;\n\t"
                "mad.lo.cc.u32   %9, %43, %68, %9;\n\t"
                "madc.hi.u32     %10, %43, %68,  0;\n\t"
                "mad.lo.cc.u32   %10, %44, %68, %10;\n\t"
                "madc.hi.u32     %11, %44, %68,  0;\n\t"
                "mad.lo.cc.u32   %11, %45, %68, %11;\n\t"
                "madc.hi.u32     %12, %45, %68,  0;\n\t"
                "mad.lo.cc.u32   %12, %46, %68, %12;\n\t"
                "madc.hi.u32     %13, %46, %68,  0;\n\t"
                "mad.lo.cc.u32   %13, %47, %68, %13;\n\t"
                "madc.hi.u32     %14, %47, %68,  0;\n\t"
                "mad.lo.cc.u32   %14, %48, %68, %14;\n\t"
                "madc.hi.u32     %15, %48, %68,  0;\n\t"
                "mad.lo.cc.u32   %15, %49, %68, %15;\n\t"
                "madc.hi.u32     %16, %49, %68,  0;\n\t"
                "mad.lo.cc.u32   %16, %50, %68, %16;\n\t"
                "madc.hi.u32     %17, %50, %68,  0;\n\t"
                "mad.lo.cc.u32   %17, %51, %68, %17;\n\t"
                "madc.hi.u32     %18, %51, %68,  0;\n\t"
                "mad.lo.cc.u32   %18, %52, %68, %18;\n\t"
                "madc.hi.u32     %19, %52, %68,  0;\n\t"
                "mad.lo.cc.u32   %19, %53, %68, %19;\n\t"
                "madc.hi.u32     %20, %53, %68,  0;\n\t"
                "mad.lo.cc.u32   %20, %54, %68, %20;\n\t"
                "madc.hi.u32     %21, %54, %68,  0;\n\t"
                "mad.lo.cc.u32   %21, %55, %68, %21;\n\t"
                "madc.hi.u32     %22, %55, %68,  0;\n\t"
                "mad.lo.cc.u32   %22, %56, %68, %22;\n\t"
                "madc.hi.u32     %23, %56, %68,  0;\n\t"
                "mad.lo.cc.u32   %23, %57, %68, %23;\n\t"
                "madc.hi.u32     %24, %57, %68,  0;\n\t"
                "mad.lo.cc.u32   %24, %58, %68, %24;\n\t"
                "madc.hi.u32     %25, %58, %68,  0;\n\t"
                "mad.lo.cc.u32   %25, %59, %68, %25;\n\t"
                "madc.hi.u32     %26, %59, %68,  0;\n\t"
                "mad.lo.cc.u32   %26, %60, %68, %26;\n\t"
                "madc.hi.u32     %27, %60, %68,  0;\n\t"
                "mad.lo.cc.u32   %27, %61, %68, %27;\n\t"
                "madc.hi.u32     %28, %61, %68,  0;\n\t"
                "mad.lo.cc.u32   %28, %62, %68, %28;\n\t"
                "madc.hi.u32     %29, %62, %68,  0;\n\t"
                "mad.lo.cc.u32   %29, %63, %68, %29;\n\t"
                "madc.hi.u32     %30, %63, %68,  0;\n\t"
                "mad.lo.cc.u32   %30, %64, %68, %30;\n\t"
                "madc.hi.u32     %31, %64, %68,  0;\n\t"
                "mad.lo.cc.u32   %31, %65, %68, %31;\n\t"
                "madc.hi.u32     %32, %65, %68,  0;\n\t"
                "mad.lo.cc.u32   %32, %66, %68, %32;\n\t"
                "madc.hi.u32     %33, %66, %68,  0;\n\t"
                "mad.lo.u32      %33, %67, %68, %33;\n\t"

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
                "r"(x));
            
            return result;

        }

        
    }
}
