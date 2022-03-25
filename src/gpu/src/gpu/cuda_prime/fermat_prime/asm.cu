
namespace nexusminer {
    namespace gpu {
        __device__ __forceinline__ uint32_t add_cc(uint32_t a, uint32_t b) {
            uint32_t r;
            #if defined(GPU_CUDA_ENABLED)
            asm volatile ("add.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
            #else 
            #if defined(__gfx1030__)
            asm volatile ("V_ADD_CO_U32 %0, vcc_lo, %1, %2;" : "=v"(r) : "v"(a), "v"(b));
            #else
            asm volatile ("V_ADD_CO_U32 %0, vcc, %1, %2;" : "=v"(r) : "v"(a), "v"(b));
            #endif
            #endif
            return r;
        }

        __device__ __forceinline__ uint32_t addc_cc(uint32_t a, uint32_t b) {
            uint32_t r;
            #if defined(GPU_CUDA_ENABLED)
            asm volatile ("addc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
            #else 
            #if defined(__gfx1030__)
            asm volatile ("V_ADD_CO_CI_U32 %0, %1, %2;" : "=v"(r) : "v"(a), "v"(b));
            #else
            asm volatile ("v_addc_co_u32 %0, vcc, %1, %2, vcc;" : "=v"(r) : "v"(a), "v"(b));
            #endif
            #endif
            return r;
        }

        /* __device__ __forceinline__ uint32_t addc(uint32_t a, uint32_t b) {
            uint32_t r;

            r = addc_cc(a, b);
            return r;
        }

        __device__ __forceinline__ uint32_t sub_cc(uint32_t a, uint32_t b) {
            uint32_t r;

            asm volatile ("sub.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
            return r;
        }

        __device__ __forceinline__ uint32_t subc_cc(uint32_t a, uint32_t b) {
            uint32_t r;

            asm volatile ("subc.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
            return r;
        }

        __device__ __forceinline__ uint32_t subc(uint32_t a, uint32_t b) {
            uint32_t r;

            asm volatile ("subc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
            return r;
        }

        __device__ __forceinline__ uint32_t madlo(uint32_t a, uint32_t b, uint32_t c) {
            uint32_t r;

            asm volatile ("mad.lo.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
            return r;
        }

        __device__ __forceinline__ uint32_t madlo_cc(uint32_t a, uint32_t b, uint32_t c) {
            uint32_t r;

            asm volatile ("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
            return r;
        }

        __device__ __forceinline__ uint32_t madloc_cc(uint32_t a, uint32_t b, uint32_t c) {
            uint32_t r;

            asm volatile ("madc.lo.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
            return r;
        }

        __device__ __forceinline__ uint32_t madloc(uint32_t a, uint32_t b, uint32_t c) {
            uint32_t r;

            asm volatile ("madc.lo.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
            return r;
        }

        __device__ __forceinline__ uint32_t madhi(uint32_t a, uint32_t b, uint32_t c) {
            uint32_t r;

            asm volatile ("mad.hi.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
            return r;
        }

        __device__ __forceinline__ uint32_t madhi_cc(uint32_t a, uint32_t b, uint32_t c) {
            uint32_t r;

            asm volatile ("mad.hi.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
            return r;
        }

        __device__ __forceinline__ uint32_t madhic_cc(uint32_t a, uint32_t b, uint32_t c) {
            uint32_t r;

            asm volatile ("madc.hi.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
            return r;
        }

        __device__ __forceinline__ uint32_t madhic(uint32_t a, uint32_t b, uint32_t c) {
            uint32_t r;

            asm volatile ("madc.hi.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
            return r;
        } */

        /* __device__ __forceinline__ uint64_t mad_wide(uint32_t a, uint32_t b, uint64_t c) {
            uint64_t r;

            asm volatile ("mad.wide.u32 %0, %1, %2, %3;" : "=l"(r) : "r"(a), "r"(b), "l"(c));
            return r;
        } */
    }
}


